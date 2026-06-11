"""
Step 1 — DINO Auto-Labeling Pipeline
=====================================
Runs Grounding DINO on technician installation photos to generate
candidate bounding boxes for three ROI classes:
  0: kazam_box
  1: multimeter
  2: lcd_screen

Two-stage detection:
  Stage 1: Full image → kazam_box + multimeter
  Stage 2: Multimeter crop → lcd_screen (mapped back to full-image coords)

Usage:
  # Test on first 10 installation folders (30 images)
  python run_dino.py --limit-folders 10

  # Full run on all 1000 folders
  python run_dino.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
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


# ──── Constants ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "wcr_installation_request-1000-11062026"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "dino_auto_label"

MODEL_ID = "IDEA-Research/grounding-dino-base"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})

PROMPTS: dict[str, str] = {
    "kazam_box": (
        "Kazam EV charger wall box, white charger box, "
        "QR code charger unit."
    ),
    "multimeter": (
        "handheld digital multimeter, clamp multimeter, "
        "electrical meter held by technician."
    ),
    "lcd_screen": (
        "small rectangular LCD display screen on multimeter, "
        "grey green numeric display, seven segment digit screen."
    ),
    "kazam_box_black_cover": (
        "dark translucent plastic flap, black plastic cover on the bottom of the white charger box."
    ),
    "qr_code": (
        "square QR code sticker on the white charger box, QR code."
    ),
}

CLASS_STYLES: dict[str, dict[str, Any]] = {
    "kazam_box":  {"class_id": 0, "color_bgr": (255, 80, 40),  "label": "0 kazam_box"},
    "multimeter": {"class_id": 1, "color_bgr": (0, 255, 255),  "label": "1 multimeter"},
    "lcd_screen": {"class_id": 2, "color_bgr": (40, 40, 255),  "label": "2 lcd_screen"},
    "kazam_box_black_cover": {"class_id": 3, "color_bgr": (0, 0, 0), "label": "3 black_cover"},
    "qr_code": {"class_id": 4, "color_bgr": (255, 0, 255), "label": "4 qr_code"},
}

MEASUREMENT_PATTERNS: dict[str, str] = {
    "(N-E)": "neutral_earth",
    "(P-E)": "phase_earth",
    "(P-N)": "phase_neutral",
}


# ──── Logging ──────────────────────────────────────────────────────────────

def setup_logging(output_root: Path) -> logging.Logger:
    """Configure dual file + console logging."""
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"dino_run_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger("dino_auto_label")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ──── Data Structures ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class OutputPaths:
    """Resolved output directory tree."""
    root: Path

    @property
    def dino_candidates(self) -> Path:
        return self.root / "dino_candidates"

    @property
    def manifest_json(self) -> Path:
        return self.root / "review_manifest.json"

    @property
    def manifest_csv(self) -> Path:
        return self.root / "review_manifest.csv"

    def ensure_dirs(self) -> None:
        for path in [
            self.dino_candidates,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# ──── Image Discovery ─────────────────────────────────────────────────────

def discover_installation_folders(data_root: Path) -> list[Path]:
    """Return sorted list of installation folders under data_root."""
    return sorted(p for p in data_root.iterdir() if p.is_dir())


def discover_all_images(
    data_root: Path, folder_limit: int | None = None,
) -> list[Path]:
    """Discover images across installation folders, optionally capped."""
    folders = discover_installation_folders(data_root)
    if folder_limit:
        folders = folders[:folder_limit]
    images: list[Path] = []
    for folder in folders:
        images.extend(
            sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        )
    return images


# ──── Naming & ID Helpers ──────────────────────────────────────────────────

def extract_measurement_type(filename: str) -> str:
    """Parse measurement type from filename, e.g. '(N-E)' → 'neutral_earth'."""
    for pattern, label in MEASUREMENT_PATTERNS.items():
        if pattern in filename:
            return label
    return "unknown"


def extract_installation_id(image_path: Path, data_root: Path) -> str:
    """Extract the installation folder name as a unique installation ID."""
    rel = image_path.relative_to(data_root)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def _safe_name(path: Path, data_root: Path) -> str:
    rel = path.relative_to(data_root)
    parts = [
        part.replace(" ", "_").replace("(", "").replace(")", "")
        for part in rel.parts
    ]
    return "__".join(parts)


def image_id_for(path: Path, data_root: Path) -> str:
    """Generate a unique, deterministic identifier for an image."""
    rel = path.relative_to(data_root).as_posix()
    digest = hashlib.md5(rel.encode()).hexdigest()[:8]
    return f"{_safe_name(path, data_root).replace('.', '_')}_{digest}"


# ──── Manifest I/O ─────────────────────────────────────────────────────────

_MANIFEST_FIELD_ORDER: list[str] = [
    "image_id", "installation_id", "measurement_type",
    "relative_image_path", "original_image_path", "copied_image_path",
    "candidate_preview_path", "image_width", "image_height",
    "review_status", "lcd_screen_value", "created_at", "updated_at",
]
for _roi in CLASS_STYLES:
    _MANIFEST_FIELD_ORDER += [
        f"{_roi}_bbox_xyxy", f"{_roi}_confidence",
        f"{_roi}_dino_label", f"{_roi}_crop_path",
    ]


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load existing manifest rows keyed by image_id."""
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(r["image_id"]): r for r in rows if r.get("image_id")}


def save_manifest(rows: list[dict[str, Any]], paths: OutputPaths) -> None:
    """Persist manifest as JSON + CSV (for easy inspection)."""
    paths.manifest_json.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    extras = sorted({k for r in rows for k in r if k not in _MANIFEST_FIELD_ORDER})
    fields = _MANIFEST_FIELD_ORDER + extras
    with paths.manifest_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({f: r.get(f, "") for f in fields})


# ──── DINO Model ───────────────────────────────────────────────────────────

def load_model(model_id: str, device: str):
    """Load Grounding DINO model + processor."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def _clamp(box: list[int], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [max(0, min(w - 1, x1)), max(0, min(h - 1, y1)),
            max(0, min(w, x2)),     max(0, min(h, y2))]


def _iou(box1: list[int], box2: list[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / float(area1 + area2 - inter)


def _intersection_over_area(box1: list[int], box2: list[int]) -> float:
    # How much of box1 is inside box2?
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    return inter / float(area1)


def run_dino(
    image_input: Path | Image.Image,
    prompt: str,
    processor: Any,
    model: Any,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> list[dict[str, Any]]:
    """Run Grounding DINO inference; return detections sorted by confidence."""
    img = (
        Image.open(image_input).convert("RGB")
        if isinstance(image_input, Path)
        else image_input.convert("RGB")
    )
    w, h = img.size
    area = w * h
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"],
        box_threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]

    detections: list[dict[str, Any]] = []
    labels = results.get("labels", results.get("text_labels", [""] * len(results["scores"])))
    for box_t, score, lbl in zip(results["boxes"], results["scores"], labels):
        box = [int(round(v)) for v in box_t.cpu().numpy().tolist()]
        x1, y1, x2, y2 = _clamp(box, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "bbox_xyxy": [x1, y1, x2, y2],
            "confidence": float(score),
            "dino_label": str(lbl),
            "area_ratio": ((x2 - x1) * (y2 - y1)) / area,
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def choose_candidate(
    dets: list[dict[str, Any]],
    min_area: float,
    max_area: float,
    horizontal: bool = False,
    avoid_box: list[int] | None = None,
    max_overlap: float = 0.3,
) -> dict[str, Any] | None:
    """Pick the highest-confidence detection within area constraints, optionally avoiding another box."""
    for d in dets:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        if not (min_area <= d["area_ratio"] <= max_area):
            continue
        if horizontal and (x2 - x1) < (y2 - y1) * 0.9:
            continue
        if avoid_box is not None:
            overlap = _intersection_over_area([x1, y1, x2, y2], avoid_box)
            if overlap > max_overlap:
                continue
        return d
    return None


# ──── Image Helpers ────────────────────────────────────────────────────────

def _draw_box(bgr, roi: str, box: list[int] | None, conf: float | None) -> None:
    if not box:
        return
    style = CLASS_STYLES[roi]
    c = style["color_bgr"]
    x1, y1, x2, y2 = box
    cv2.rectangle(bgr, (x1, y1), (x2, y2), c, 3)
    text = style["label"] + (f" {conf:.2f}" if conf is not None else "")
    cv2.putText(bgr, text, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)


# ──── Core Processing ─────────────────────────────────────────────────────

def process_image(
    image_path: Path,
    data_root: Path,
    paths: OutputPaths,
    processor: Any,
    model: Any,
    device: str,
    box_th: float,
    txt_th: float,
    lcd_box_th: float,
    lcd_txt_th: float,
) -> dict[str, Any]:
    """Full two-stage DINO pipeline for a single image → manifest row."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = bgr.shape[:2]
    img_id = image_id_for(image_path, data_root)
    now = datetime.now().isoformat(timespec="seconds")

    row: dict[str, Any] = {
        "image_id": img_id,
        "installation_id": extract_installation_id(image_path, data_root),
        "measurement_type": extract_measurement_type(image_path.name),
        "relative_image_path": image_path.relative_to(data_root).as_posix(),
        "original_image_path": str(image_path),
        "copied_image_path": "",
        "image_width": w,
        "image_height": h,
        "review_status": "pending",
        "lcd_screen_value": "",
        "created_at": now,
        "updated_at": "",
    }

    # ── Stage 1: full-image detections ──
    kazam_dets = run_dino(image_path, PROMPTS["kazam_box"], processor, model,
                          device, box_th, txt_th)
    meter_dets = run_dino(image_path, PROMPTS["multimeter"], processor, model,
                          device, box_th, txt_th)
    cover_dets = run_dino(image_path, PROMPTS["kazam_box_black_cover"], processor, model,
                          device, box_th, txt_th)
    qr_dets = run_dino(image_path, PROMPTS["qr_code"], processor, model,
                       device, box_th, txt_th)

    kazam_cand = choose_candidate(kazam_dets, 0.03, 0.55)
    avoid_b = kazam_cand["bbox_xyxy"] if kazam_cand else None

    # The multimeter is almost never inside the kazam box, so reject candidates that overlap the box heavily
    meter_cand = choose_candidate(meter_dets, 0.01, 0.45, avoid_box=avoid_b, max_overlap=0.4)
    cover_cand = choose_candidate(cover_dets, 0.01, 0.40)
    qr_cand = choose_candidate(qr_dets, 0.001, 0.15)

    selected: dict[str, dict[str, Any] | None] = {
        "kazam_box": kazam_cand,
        "multimeter": meter_cand,
        "kazam_box_black_cover": cover_cand,
        "qr_code": qr_cand,
        "lcd_screen": None,
    }

    # ── Stage 2: lcd_screen inside multimeter crop ──
    meter_det = selected["multimeter"]
    if meter_det:
        mx1, my1, mx2, my2 = meter_det["bbox_xyxy"]
        crop_bgr = bgr[my1:my2, mx1:mx2]
        if crop_bgr.size:
            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            lcd_dets = run_dino(crop_pil, PROMPTS["lcd_screen"], processor,
                                model, device, lcd_box_th, lcd_txt_th)
            lcd = (choose_candidate(lcd_dets, 0.002, 0.60, horizontal=True)
                   or choose_candidate(lcd_dets, 0.002, 0.60))
            if lcd:
                lx1, ly1, lx2, ly2 = lcd["bbox_xyxy"]
                lcd = dict(lcd)
                lcd["bbox_xyxy"] = [lx1 + mx1, ly1 + my1, lx2 + mx1, ly2 + my1]
                selected["lcd_screen"] = lcd

    # ── Draw preview & save crops ──
    preview = bgr.copy()
    for roi, det in selected.items():
        box = det["bbox_xyxy"] if det else None
        conf = det["confidence"] if det else None
        _draw_box(preview, roi, box, conf)

        row[f"{roi}_bbox_xyxy"] = json.dumps(box) if box else ""
        row[f"{roi}_bbox_xyxy_dino_raw"] = json.dumps(box) if box else ""
        row[f"{roi}_confidence"] = f"{conf:.4f}" if conf is not None else ""
        row[f"{roi}_dino_label"] = det["dino_label"] if det else ""
        row[f"{roi}_crop_path"] = ""

    preview_path = paths.dino_candidates / f"{img_id}_candidate.jpg"
    cv2.imwrite(str(preview_path), preview)
    row["candidate_preview_path"] = str(preview_path)
    return row


# ──── CLI ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Grounding DINO auto-labeling on installation photos.",
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="Root dir with installation folders.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                    help="Output directory for DINO results.")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--box-threshold", type=float, default=0.18)
    p.add_argument("--text-threshold", type=float, default=0.18)
    p.add_argument("--lcd-box-threshold", type=float, default=0.12)
    p.add_argument("--lcd-text-threshold", type=float, default=0.12)
    p.add_argument("--limit-folders", type=int, default=None,
                    help="Process only the first N installation folders.")
    p.add_argument("--device", default=None, help="Force device (cuda / cpu).")
    p.add_argument("--overwrite", action="store_true",
                    help="Re-run DINO on images that already have candidates.")
    p.add_argument("--save-interval", type=int, default=10,
                    help="Save manifest every N processed images.")
    return p.parse_args()


# ──── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    paths = OutputPaths(root=args.output_root)
    paths.ensure_dirs()

    logger = setup_logging(args.output_root)
    logger.info("=" * 60)
    logger.info("DINO Auto-Label Pipeline")
    logger.info("=" * 60)
    logger.info("Data root   : %s", args.data_root)
    logger.info("Output root : %s", args.output_root)
    logger.info("Model       : %s", args.model_id)
    if args.limit_folders:
        logger.info("Folder limit: %d", args.limit_folders)

    # ── Discover images ──
    images = discover_all_images(args.data_root, args.limit_folders)
    if not images:
        raise SystemExit(f"No images found under {args.data_root}")

    n_folders = len(discover_installation_folders(args.data_root))
    if args.limit_folders:
        n_folders = min(n_folders, args.limit_folders)
    logger.info("Found %d images across %d folders", len(images), n_folders)

    # ── Determine work ──
    existing = load_manifest(paths.manifest_json)
    pending: list[tuple[int, Path]] = []
    all_rows: list[dict[str, Any] | None] = []

    for img_path in images:
        img_id = image_id_for(img_path, args.data_root)
        prev = existing.get(img_id)
        if prev and not args.overwrite:
            if prev.get("review_status") in {"approved", "rejected"}:
                all_rows.append(prev)
                continue
            preview = prev.get("candidate_preview_path", "")
            if preview and Path(preview).exists():
                all_rows.append(prev)
                continue
        pending.append((len(all_rows), img_path))
        all_rows.append(None)

    logger.info("Pending: %d  |  Skipped (existing): %d",
                len(pending), len(images) - len(pending))

    if not pending:
        logger.info("Nothing to do — manifest is up to date.")
        return

    # ── Load model ──
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on %s …", args.model_id, device)
    processor, model = load_model(args.model_id, device)
    logger.info("Model loaded successfully")

    # ── Process ──
    ok, fail = 0, 0
    for row_idx, img_path in tqdm(pending, desc="DINO auto-labeling"):
        try:
            all_rows[row_idx] = process_image(
                img_path, args.data_root, paths,
                processor, model, device,
                args.box_threshold, args.text_threshold,
                args.lcd_box_threshold, args.lcd_text_threshold,
            )
            ok += 1
        except Exception as exc:
            logger.error("FAIL %s — %s", img_path.name, exc)
            all_rows[row_idx] = {
                "image_id": image_id_for(img_path, args.data_root),
                "installation_id": extract_installation_id(img_path, args.data_root),
                "measurement_type": extract_measurement_type(img_path.name),
                "relative_image_path": img_path.relative_to(args.data_root).as_posix(),
                "original_image_path": str(img_path),
                "review_status": "error",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "error": repr(exc),
            }
            fail += 1

        # Incremental save for crash recovery
        if ok % args.save_interval == 0:
            save_manifest([r for r in all_rows if r], paths)

    # ── Final save ──
    final = [r for r in all_rows if r is not None]
    save_manifest(final, paths)

    logger.info("─" * 40)
    logger.info("Done  |  OK: %d  |  Errors: %d  |  Total rows: %d", ok, fail, len(final))
    logger.info("JSON  : %s", paths.manifest_json)
    logger.info("CSV   : %s", paths.manifest_csv)


if __name__ == "__main__":
    main()
