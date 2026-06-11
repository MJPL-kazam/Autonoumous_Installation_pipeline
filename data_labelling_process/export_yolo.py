"""
Step 3 — Export Approved Records to YOLO Format
================================================
Reads the review manifest, filters approved records with all 3 boxes
and LCD values, and exports a clean YOLO-format dataset ready for training.

Usage:
  python export_yolo.py
  python export_yolo.py --overwrite          # rebuild from scratch
  python export_yolo.py --val-ratio 0.15     # custom val split
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


# ──── Constants ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "dino_auto_label" / "review_manifest.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "yolo_dataset"

CLASSES = ["kazam_box", "multimeter", "lcd_screen", "kazam_box_black_cover", "qr_code"]
REQUIRED_CLASSES = ["kazam_box", "multimeter", "lcd_screen"]


# ──── Manifest I/O ─────────────────────────────────────────────────────────

def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist manifest with YOLO export metadata added."""
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    preferred = [
        "image_id", "installation_id", "measurement_type",
        "relative_image_path", "original_image_path",
        "copied_image_path", "candidate_preview_path",
        "image_width", "image_height",
        "review_status", "lcd_screen_value",
        "yolo_split", "yolo_image_path", "yolo_label_path", "yolo_exported_at",
        "created_at", "updated_at",
    ]
    fields = preferred + sorted({k for r in rows for k in r if k not in preferred})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fields})


# ──── Box Parsing ──────────────────────────────────────────────────────────

def parse_box(value: Any) -> list[float] | None:
    if not value:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or len(value) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in value]
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def to_yolo_line(class_id: int, box: list[float], img_w: float, img_h: float) -> str:
    """Convert xyxy box to YOLO normalized center format."""
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ──── Filtering ────────────────────────────────────────────────────────────

def filter_approved(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records that are approved, have all 3 boxes, LCD value, and valid image."""
    approved = []
    for row in records:
        if row.get("review_status") != "approved":
            continue
        if not row.get("lcd_screen_value"):
            continue
        src_path = row.get("copied_image_path") or row.get("original_image_path")
        if not src_path:
            continue
        src = Path(src_path)
        if not src.exists():
            continue
        if not row.get("image_width") or not row.get("image_height"):
            continue
        if all(parse_box(row.get(f"{c}_bbox_xyxy")) for c in REQUIRED_CLASSES):
            approved.append(row)
    return approved


# ──── Export ───────────────────────────────────────────────────────────────

def write_data_yaml(root: Path) -> None:
    yaml = "\n".join([
        "path: .",
        "train: images/train",
        "val: images/val",
        "names:",
        *(f"  {i}: {c}" for i, c in enumerate(CLASSES)),
        "",
    ])
    (root / "data.yaml").write_text(yaml, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export approved labels to YOLO format.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Reset output if requested
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    for split in ("train", "val"):
        (args.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_records = load_records(args.manifest)
    records = filter_approved(all_records)

    if not records:
        print("No approved records to export.")
        print("Run review_app.py first to approve records with all boxes and LCD values.")
        return

    # Shuffle and split
    export_rows = list(records)
    random.Random(args.seed).shuffle(export_rows)
    val_count = max(1, int(round(len(export_rows) * args.val_ratio))) if len(export_rows) > 1 else 0
    val_ids = {r["image_id"] for r in export_rows[:val_count]}

    # Export
    exported_map: dict[str, dict] = {}
    gt_rows: list[dict] = []

    for row in export_rows:
        split = "val" if row["image_id"] in val_ids else "train"
        src_path = row.get("copied_image_path") or row.get("original_image_path")
        src = Path(src_path)
        img_dest = args.output_root / "images" / split / f"{row['image_id']}{src.suffix.lower()}"
        lbl_dest = args.output_root / "labels" / split / f"{row['image_id']}.txt"

        shutil.copy2(src, img_dest)

        img_w = float(row["image_width"])
        img_h = float(row["image_height"])
        lines = []
        for cid, cname in enumerate(CLASSES):
            box = parse_box(row.get(f"{cname}_bbox_xyxy"))
            if box:
                lines.append(to_yolo_line(cid, box, img_w, img_h))
        lbl_dest.write_text("\n".join(lines) + "\n", encoding="utf-8")

        gt_rows.append({
            "image_id": row["image_id"],
            "installation_id": row.get("installation_id", ""),
            "measurement_type": row.get("measurement_type", ""),
            "split": split,
            "image_path": img_dest.relative_to(args.output_root).as_posix(),
            "lcd_screen_value": row.get("lcd_screen_value", ""),
        })
        exported_map[row["image_id"]] = {
            "yolo_split": split,
            "yolo_image_path": str(img_dest),
            "yolo_label_path": str(lbl_dest),
        }

    # Write data.yaml and ground truth CSV
    write_data_yaml(args.output_root)
    with (args.output_root / "ground_truth_lcd_values.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "image_id", "installation_id", "measurement_type",
            "split", "image_path", "lcd_screen_value",
        ])
        w.writeheader()
        w.writerows(gt_rows)

    # Update manifest with export metadata
    exported_at = datetime.now().isoformat(timespec="seconds")
    for row in all_records:
        exp = exported_map.get(row.get("image_id"))
        if exp:
            row.update(exp)
            row["yolo_exported_at"] = exported_at
        else:
            for k in ("yolo_split", "yolo_image_path", "yolo_label_path", "yolo_exported_at"):
                row.pop(k, None)
    write_records(args.manifest, all_records)

    # Summary
    train_n = sum(1 for r in gt_rows if r["split"] == "train")
    val_n = sum(1 for r in gt_rows if r["split"] == "val")
    print(f"Exported {len(records)} approved records")
    print(f"  Train: {train_n}  |  Val: {val_n}")
    print(f"  Dataset: {args.output_root}")
    print(f"  data.yaml: {args.output_root / 'data.yaml'}")
    print(f"  LCD values: {args.output_root / 'ground_truth_lcd_values.csv'}")


if __name__ == "__main__":
    main()
