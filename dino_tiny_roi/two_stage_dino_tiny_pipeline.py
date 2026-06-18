from __future__ import annotations

import argparse
import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def configure_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"two_stage_dino_tiny_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("two_stage_dino_tiny")
    logger.info("Log file: %s", log_path)
    return logger


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def collect_images(input_dirs: Iterable[Path]) -> list[tuple[Path, str]]:
    images: list[tuple[Path, str]] = []
    for folder in input_dirs:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append((path, folder.name))
    return images


def safe_name(path: Path) -> str:
    return (
        path.stem.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )


def clip_box(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    return [x1, y1, x2, y2]


def detect_objects(
    image_input: Path | Image.Image,
    prompt: str,
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> list[dict]:
    if isinstance(image_input, Path):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    width, height = image.size
    image_area = width * height
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(height, width)],
    )[0]

    labels = results.get("text_labels", [""] * len(results["scores"]))
    detections = []
    for bbox, score, label in zip(results["boxes"], results["scores"], labels):
        box = clip_box(bbox.cpu().numpy().astype(int).tolist(), width, height)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        area_ratio = ((x2 - x1) * (y2 - y1)) / image_area
        detections.append(
            {
                "bbox": box,
                "score": float(score),
                "label": str(label),
                "area_ratio": float(area_ratio),
            }
        )
    return detections


def filter_detections(
    detections: list[dict],
    min_area_ratio: float,
    max_area_ratio: float,
    require_horizontal: bool = False,
) -> list[dict]:
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        width = x2 - x1
        height = y2 - y1
        if not min_area_ratio <= det["area_ratio"] <= max_area_ratio:
            continue
        if require_horizontal and width <= height * 0.9:
            continue
        filtered.append(det)
    return sorted(filtered, key=lambda item: item["score"], reverse=True)


def preprocess_lcd_crop(crop_bgr: np.ndarray) -> np.ndarray:
    height, width = crop_bgr.shape[:2]
    resized = cv2.resize(crop_bgr, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def draw_box(image: np.ndarray, box: list[int], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def run_pipeline(config: dict) -> None:
    base_dir = Path(config.get("base_dir", ".")).resolve()
    input_dirs = [base_dir / item for item in config.get("input_dirs", ["Data/N to E", "Data/P to E", "Data/P to N"])]
    output_dir = (base_dir / config.get("output_dir", "output")).resolve()

    dirs = {
        "stage1_annotated": output_dir / "stage1_full_annotated",
        "stage1_crop": output_dir / "stage1_meter_crops",
        "stage2_annotated": output_dir / "stage2_meter_annotated",
        "lcd_raw": output_dir / "stage2_lcd_crops_raw",
        "lcd_processed": output_dir / "stage2_lcd_crops_processed",
        "final_annotated": output_dir / "final_full_annotated",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(output_dir)
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    images = collect_images(input_dirs)
    logger.info("Found %d images", len(images))

    model_id = config.get("model_id", DEFAULT_MODEL_ID)
    logger.info("Loading Grounding DINO model: %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    meter_prompt = config.get("meter_prompt", "Kazam EV charger wall box . handheld digital multimeter .")
    lcd_prompt = config.get(
        "lcd_prompt",
        "small rectangular LCD display screen on multimeter, grey green digital display, only seven segment digits, numeric reading screen, not rotary dial, not labels, not buttons, not ports.",
    )

    stage1_box_threshold = float(config.get("stage1_box_threshold", 0.20))
    stage1_text_threshold = float(config.get("stage1_text_threshold", 0.20))
    stage1_min_area = float(config.get("stage1_min_area_ratio", 0.01))
    stage1_max_area = float(config.get("stage1_max_area_ratio", 0.40))

    stage2_box_threshold = float(config.get("stage2_box_threshold", 0.15))
    stage2_text_threshold = float(config.get("stage2_text_threshold", 0.15))
    stage2_min_area = float(config.get("stage2_min_area_ratio", 0.04))
    stage2_max_area = float(config.get("stage2_max_area_ratio", 0.50))

    rows = []
    for image_path, folder_name in tqdm(images, desc="Running two-stage DINO Tiny"):
        image_name = f"{folder_name}_{safe_name(image_path)}"
        row = {
            "image_path": str(image_path),
            "connection_type_folder": folder_name,
            "meter_detected": False,
            "meter_confidence": "",
            "meter_bbox_global": "",
            "meter_crop_path": "",
            "lcd_detected": False,
            "lcd_confidence": "",
            "lcd_bbox_local": "",
            "lcd_bbox_global": "",
            "lcd_crop_raw_path": "",
            "lcd_crop_processed_path": "",
            "final_annotated_path": "",
            "final_status": "",
            "failure_reason": "",
        }

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            row["final_status"] = "error"
            row["failure_reason"] = "could not read image"
            rows.append(row)
            continue

        try:
            raw_stage1 = detect_objects(
                image_path,
                meter_prompt,
                processor,
                model,
                device,
                stage1_box_threshold,
                stage1_text_threshold,
            )
            meter_candidates = [
                det
                for det in raw_stage1
                if "multimeter" in det["label"].lower()
                or "handheld" in det["label"].lower()
                or det["label"] == ""
            ]
            stage1 = filter_detections(meter_candidates, stage1_min_area, stage1_max_area)
        except Exception as exc:
            row["final_status"] = "error"
            row["failure_reason"] = f"stage1_error: {exc}"
            rows.append(row)
            continue

        if not stage1:
            row["final_status"] = "meter_not_detected"
            row["failure_reason"] = "no meter found in stage 1"
            rows.append(row)
            continue

        best_meter = stage1[0]
        mx1, my1, mx2, my2 = best_meter["bbox"]
        row["meter_detected"] = True
        row["meter_confidence"] = f"{best_meter['score']:.4f}"
        row["meter_bbox_global"] = str(best_meter["bbox"])

        stage1_annotated = image_bgr.copy()
        draw_box(stage1_annotated, best_meter["bbox"], (0, 255, 0), f"meter {best_meter['score']:.2f}")
        cv2.imwrite(str(dirs["stage1_annotated"] / f"{image_name}_s1.jpg"), stage1_annotated)

        meter_crop = image_bgr[my1:my2, mx1:mx2]
        if meter_crop.size == 0:
            row["final_status"] = "error"
            row["failure_reason"] = "empty meter crop"
            rows.append(row)
            continue

        meter_crop_path = dirs["stage1_crop"] / f"{image_name}_meter.jpg"
        cv2.imwrite(str(meter_crop_path), meter_crop)
        row["meter_crop_path"] = str(meter_crop_path)

        try:
            meter_rgb = Image.fromarray(cv2.cvtColor(meter_crop, cv2.COLOR_BGR2RGB))
            raw_stage2 = detect_objects(
                meter_rgb,
                lcd_prompt,
                processor,
                model,
                device,
                stage2_box_threshold,
                stage2_text_threshold,
            )
            stage2 = filter_detections(raw_stage2, stage2_min_area, stage2_max_area, require_horizontal=True)
            if not stage2 and raw_stage2:
                stage2 = filter_detections(raw_stage2, stage2_min_area, stage2_max_area, require_horizontal=False)
        except Exception as exc:
            row["final_status"] = "error"
            row["failure_reason"] = f"stage2_error: {exc}"
            rows.append(row)
            continue

        if not stage2:
            row["final_status"] = "lcd_not_detected"
            row["failure_reason"] = "no lcd found in stage 2"
            rows.append(row)
            continue

        best_lcd = stage2[0]
        lx1, ly1, lx2, ly2 = best_lcd["bbox"]
        lcd_global = [lx1 + mx1, ly1 + my1, lx2 + mx1, ly2 + my1]

        row["lcd_detected"] = True
        row["lcd_confidence"] = f"{best_lcd['score']:.4f}"
        row["lcd_bbox_local"] = str(best_lcd["bbox"])
        row["lcd_bbox_global"] = str(lcd_global)

        stage2_annotated = meter_crop.copy()
        draw_box(stage2_annotated, best_lcd["bbox"], (0, 165, 255), f"lcd {best_lcd['score']:.2f}")
        cv2.imwrite(str(dirs["stage2_annotated"] / f"{image_name}_lcd_local.jpg"), stage2_annotated)

        final_annotated = image_bgr.copy()
        draw_box(final_annotated, best_meter["bbox"], (0, 255, 0), f"meter {best_meter['score']:.2f}")
        draw_box(final_annotated, lcd_global, (0, 0, 255), f"lcd {best_lcd['score']:.2f}")
        final_path = dirs["final_annotated"] / f"{image_name}_final.jpg"
        cv2.imwrite(str(final_path), final_annotated)
        row["final_annotated_path"] = str(final_path)

        lcd_crop = meter_crop[ly1:ly2, lx1:lx2]
        if lcd_crop.size == 0:
            row["final_status"] = "error"
            row["failure_reason"] = "empty lcd crop"
            rows.append(row)
            continue

        lcd_raw_path = dirs["lcd_raw"] / f"{image_name}_lcd_raw.jpg"
        cv2.imwrite(str(lcd_raw_path), lcd_crop)
        row["lcd_crop_raw_path"] = str(lcd_raw_path)

        lcd_processed = preprocess_lcd_crop(lcd_crop)
        lcd_processed_path = dirs["lcd_processed"] / f"{image_name}_lcd_processed.jpg"
        cv2.imwrite(str(lcd_processed_path), lcd_processed)
        row["lcd_crop_processed_path"] = str(lcd_processed_path)

        row["final_status"] = "ready_for_ocr"
        rows.append(row)

    csv_path = output_dir / "results.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "image_path",
        "final_status",
        "failure_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Results saved to %s", csv_path)
    logger.info("Complete. Images: %d", len(rows))


def main() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    parser = argparse.ArgumentParser(description="Two-stage Grounding DINO Tiny ROI pipeline.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config if args.config.exists() else None)
    run_pipeline(config)


if __name__ == "__main__":
    main()
