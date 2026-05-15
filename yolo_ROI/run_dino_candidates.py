from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data"
WORKSPACE_ROOT = Path(__file__).resolve().parent
OUT_ROOT = WORKSPACE_ROOT / "data"

MODEL_ID = "IDEA-Research/grounding-dino-base"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

PROMPTS = {
    "kazam_box": "Kazam EV charger wall box, white charger box, QR code charger unit.",
    "multimeter": "handheld digital multimeter, clamp multimeter, electrical meter held by technician.",
    "lcd_screen": "small rectangular LCD display screen on multimeter, grey green numeric display, seven segment digit screen.",
}

ROI_STYLES = {
    "kazam_box": {"class_id": 1, "color_bgr": (255, 80, 40), "label": "1 kazam_box"},
    "multimeter": {"class_id": 2, "color_bgr": (40, 210, 80), "label": "2 multimeter"},
    "lcd_screen": {"class_id": 3, "color_bgr": (40, 40, 255), "label": "3 lcd_screen"},
}


@dataclass(frozen=True)
class Paths:
    raw_images: Path = OUT_ROOT / "raw_images"
    dino_candidates: Path = OUT_ROOT / "dino_candidates"
    crops_root: Path = OUT_ROOT / "crops"
    yolo_dataset: Path = OUT_ROOT / "yolo_dataset"
    manifest_csv: Path = OUT_ROOT / "review_manifest.csv"
    manifest_json: Path = OUT_ROOT / "review_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3-ROI DINO candidate labels for YOLO review.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--box-threshold", type=float, default=0.18)
    parser.add_argument("--text-threshold", type=float, default=0.18)
    parser.add_argument("--lcd-box-threshold", type=float, default=0.12)
    parser.add_argument("--lcd-text-threshold", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images for smoke testing.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda, cpu.")
    parser.add_argument("--overwrite", action="store_true", help="Refresh unreviewed DINO candidates that already exist.")
    parser.add_argument("--force-rerun", action="store_true", help="Rerun DINO even for approved or rejected review rows.")
    return parser.parse_args()


def ensure_dirs(paths: Paths) -> None:
    for path in [
        paths.raw_images,
        paths.dino_candidates,
        paths.crops_root / "kazam_box",
        paths.crops_root / "multimeter",
        paths.crops_root / "lcd_screen",
        paths.yolo_dataset,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def reset_generated_dirs(paths: Paths) -> None:
    for path in [paths.raw_images, paths.dino_candidates, paths.crops_root]:
        if path.exists():
            shutil.rmtree(path)


def discover_images(data_root: Path) -> list[Path]:
    return sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def safe_name(path: Path, data_root: Path) -> str:
    rel = path.relative_to(data_root)
    parts = [part.replace(" ", "_").replace("(", "").replace(")", "") for part in rel.parts]
    return "__".join(parts)


def image_id_for(path: Path, data_root: Path) -> str:
    rel = path.relative_to(data_root).as_posix()
    digest = hashlib.md5(rel.encode("utf-8")).hexdigest()[:8]
    return f"{safe_name(path, data_root).replace('.', '_')}_{digest}"


def load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(row.get("image_id")): row for row in rows if row.get("image_id")}


def parse_box(value: Any) -> list[int] | None:
    if not value:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, list) or len(value) != 4:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def has_three_boxes(row: dict[str, Any]) -> bool:
    return all(parse_box(row.get(f"{roi}_bbox_xyxy")) for roi in ROI_STYLES)


def is_review_decision(row: dict[str, Any]) -> bool:
    return row.get("review_status") in {"approved", "rejected"}


def has_candidate_artifacts(row: dict[str, Any]) -> bool:
    preview = row.get("candidate_preview_path")
    copied = row.get("copied_image_path")
    if not copied or not Path(copied).exists():
        return False
    if row.get("review_status") == "rejected":
        return True
    return bool(preview and Path(preview).exists() and has_three_boxes(row))


def should_regenerate(row: dict[str, Any] | None, args: argparse.Namespace) -> bool:
    if row is None:
        return True
    if args.force_rerun:
        return True
    if is_review_decision(row):
        return False
    if args.overwrite:
        return True
    return not has_candidate_artifacts(row)


def copy_raw_image(src: Path, data_root: Path, paths: Paths) -> Path:
    rel = src.relative_to(data_root)
    dest = paths.raw_images / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def load_model(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def run_dino(
    image_input: Path | Image.Image,
    prompt: str,
    processor: Any,
    model: Any,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> list[dict[str, Any]]:
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

    detections: list[dict[str, Any]] = []
    labels = results.get("text_labels", [""] * len(results["scores"]))
    for box_tensor, score, label in zip(results["boxes"], results["scores"], labels):
        box = [int(round(v)) for v in box_tensor.cpu().numpy().tolist()]
        x1, y1, x2, y2 = clamp_box(box, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        area_ratio = ((x2 - x1) * (y2 - y1)) / image_area
        detections.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "confidence": float(score),
                "dino_label": str(label),
                "area_ratio": area_ratio,
            }
        )
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


def clamp_box(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return [x1, y1, x2, y2]


def choose_candidate(detections: list[dict[str, Any]], min_area: float, max_area: float, horizontal: bool = False):
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        width = x2 - x1
        height = y2 - y1
        if not (min_area <= det["area_ratio"] <= max_area):
            continue
        if horizontal and width < height * 0.9:
            continue
        return det
    return None


def crop_and_save(image_bgr, box: list[int] | None, out_path: Path) -> str:
    if not box:
        return ""
    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return str(out_path)


def draw_box(image_bgr, roi: str, box: list[int] | None, confidence: float | None) -> None:
    if not box:
        return
    style = ROI_STYLES[roi]
    color = style["color_bgr"]
    x1, y1, x2, y2 = box
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 3)
    text = style["label"]
    if confidence is not None:
        text += f" {confidence:.2f}"
    y_text = max(20, y1 - 8)
    cv2.putText(image_bgr, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def preferred_manifest_fields() -> list[str]:
    fields = [
        "image_id",
        "relative_image_path",
        "original_image_path",
        "copied_image_path",
        "candidate_preview_path",
        "image_width",
        "image_height",
        "review_status",
        "lcd_screen_value",
        "created_at",
        "updated_at",
    ]
    for roi in ["kazam_box", "multimeter", "lcd_screen"]:
        fields.extend(
            [
                f"{roi}_bbox_xyxy",
                f"{roi}_confidence",
                f"{roi}_dino_label",
                f"{roi}_crop_path",
            ]
        )
    return fields


def write_manifest(rows: list[dict[str, Any]], paths: Paths) -> None:
    paths.manifest_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    preferred = preferred_manifest_fields()
    extras = sorted({key for row in rows for key in row.keys() if key not in preferred})
    fields = preferred + extras
    with paths.manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def row_for_image(
    image_path: Path,
    data_root: Path,
    paths: Paths,
    processor: Any,
    model: Any,
    device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image_bgr.shape[:2]
    image_id = image_id_for(image_path, data_root)
    copied = copy_raw_image(image_path, data_root, paths)
    rel_image = image_path.relative_to(data_root).as_posix()
    now = datetime.now().isoformat(timespec="seconds")

    row: dict[str, Any] = {
        "image_id": image_id,
        "relative_image_path": rel_image,
        "original_image_path": str(image_path),
        "copied_image_path": str(copied),
        "image_width": width,
        "image_height": height,
        "review_status": "needs_recheck",
        "lcd_screen_value": "",
        "created_at": now,
        "updated_at": "",
    }

    kazam_dets = run_dino(
        image_path,
        PROMPTS["kazam_box"],
        processor,
        model,
        device,
        args.box_threshold,
        args.text_threshold,
    )
    meter_dets = run_dino(
        image_path,
        PROMPTS["multimeter"],
        processor,
        model,
        device,
        args.box_threshold,
        args.text_threshold,
    )

    selected = {
        "kazam_box": choose_candidate(kazam_dets, min_area=0.03, max_area=0.55),
        "multimeter": choose_candidate(meter_dets, min_area=0.01, max_area=0.45),
        "lcd_screen": None,
    }

    meter_box = selected["multimeter"]["bbox_xyxy"] if selected["multimeter"] else None
    if meter_box:
        mx1, my1, mx2, my2 = meter_box
        meter_crop_bgr = image_bgr[my1:my2, mx1:mx2]
        if meter_crop_bgr.size:
            meter_pil = Image.fromarray(cv2.cvtColor(meter_crop_bgr, cv2.COLOR_BGR2RGB))
            lcd_dets = run_dino(
                meter_pil,
                PROMPTS["lcd_screen"],
                processor,
                model,
                device,
                args.lcd_box_threshold,
                args.lcd_text_threshold,
            )
            lcd_local = choose_candidate(lcd_dets, min_area=0.002, max_area=0.60, horizontal=True)
            if lcd_local is None:
                lcd_local = choose_candidate(lcd_dets, min_area=0.002, max_area=0.60, horizontal=False)
            if lcd_local:
                lx1, ly1, lx2, ly2 = lcd_local["bbox_xyxy"]
                lcd_local = dict(lcd_local)
                lcd_local["bbox_xyxy"] = [lx1 + mx1, ly1 + my1, lx2 + mx1, ly2 + my1]
                selected["lcd_screen"] = lcd_local

    preview = image_bgr.copy()
    for roi, det in selected.items():
        box = det["bbox_xyxy"] if det else None
        confidence = det["confidence"] if det else None
        draw_box(preview, roi, box, confidence)

        row[f"{roi}_bbox_xyxy"] = json.dumps(box) if box else ""
        row[f"{roi}_confidence"] = f"{confidence:.4f}" if confidence is not None else ""
        row[f"{roi}_dino_label"] = det["dino_label"] if det else ""
        crop_path = paths.crops_root / roi / f"{image_id}_{roi}{image_path.suffix.lower()}"
        row[f"{roi}_crop_path"] = crop_and_save(image_bgr, box, crop_path) if box else ""

    preview_path = paths.dino_candidates / f"{image_id}_candidate.jpg"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), preview)
    row["candidate_preview_path"] = str(preview_path)
    return row


def main() -> None:
    args = parse_args()
    effective_paths = Paths(
        raw_images=args.output_root / "raw_images",
        dino_candidates=args.output_root / "dino_candidates",
        crops_root=args.output_root / "crops",
        yolo_dataset=args.output_root / "yolo_dataset",
        manifest_csv=args.output_root / "review_manifest.csv",
        manifest_json=args.output_root / "review_manifest.json",
    )
    ensure_dirs(effective_paths)

    images = discover_images(args.data_root)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found under {args.data_root}")

    existing_by_id = load_existing_rows(effective_paths.manifest_json)
    rows: list[dict[str, Any] | None] = []
    pending: list[tuple[int, Path]] = []
    for image_path in images:
        image_id = image_id_for(image_path, args.data_root)
        existing = existing_by_id.get(image_id)
        if should_regenerate(existing, args):
            pending.append((len(rows), image_path))
            rows.append(None)
        else:
            rows.append(existing)

    if pending:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {args.model_id} on {device}")
        processor, model = load_model(args.model_id, device)
        print(f"Processing {len(pending)} / {len(images)} images from {args.data_root}")
    else:
        processor = model = None
        device = ""
        print(f"No DINO work needed for {len(images)} images. Existing manifest is up to date.")

    for row_index, image_path in tqdm(pending, desc="DINO ROI candidates"):
        try:
            rows[row_index] = row_for_image(image_path, args.data_root, effective_paths, processor, model, device, args)
        except Exception as exc:
            now = datetime.now().isoformat(timespec="seconds")
            rows[row_index] = {
                "image_id": image_id_for(image_path, args.data_root),
                "relative_image_path": image_path.relative_to(args.data_root).as_posix(),
                "original_image_path": str(image_path),
                "review_status": "needs_recheck",
                "created_at": now,
                "updated_at": "",
                "error": repr(exc),
            }

    final_rows = [row for row in rows if row is not None]
    write_manifest(final_rows, effective_paths)
    print(f"Wrote {effective_paths.manifest_csv}")
    print(f"Wrote {effective_paths.manifest_json}")


if __name__ == "__main__":
    main()
