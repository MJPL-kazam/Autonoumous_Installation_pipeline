from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = WORKSPACE_ROOT / "data"
MANIFEST_JSON = DATA_ROOT / "review_manifest.json"
YOLO_ROOT = DATA_ROOT / "yolo_dataset"

CLASSES = ["kazam_box", "multimeter", "lcd_screen"]
ROI_PREVIEW_DIR_NAME = "dino_candidates"
CROPS_DIR_NAME = "crops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export approved review records into YOLO format.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_JSON)
    parser.add_argument("--output-root", type=Path, default=YOLO_ROOT)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    preferred = [
        "image_id",
        "relative_image_path",
        "original_image_path",
        "copied_image_path",
        "candidate_preview_path",
        "image_width",
        "image_height",
        "review_status",
        "lcd_screen_value",
        "yolo_split",
        "yolo_image_path",
        "yolo_label_path",
        "yolo_exported_at",
        "created_at",
        "updated_at",
    ]
    fields = preferred + sorted({key for row in rows for key in row.keys() if key not in preferred})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_box(value: Any) -> list[float] | None:
    if not value:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if len(value) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in value]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def yolo_line(class_id: int, box: list[float], image_width: float, image_height: float) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / image_width
    cy = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def approved_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    approved = []
    for row in records:
        if row.get("review_status") != "approved":
            continue
        if not row.get("lcd_screen_value"):
            continue
        if not row.get("copied_image_path"):
            continue
        src = Path(row["copied_image_path"])
        if not src.exists():
            continue
        if ROI_PREVIEW_DIR_NAME in src.parts or CROPS_DIR_NAME in src.parts:
            continue
        if not row.get("image_width") or not row.get("image_height"):
            continue
        if all(parse_box(row.get(f"{name}_bbox_xyxy")) for name in CLASSES):
            approved.append(row)
    return approved


def reset_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def ensure_output(path: Path) -> None:
    for split in ["train", "val"]:
        (path / "images" / split).mkdir(parents=True, exist_ok=True)
        (path / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(path: Path) -> None:
    data_yaml = "\n".join(
        [
            "path: .",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: kazam_box",
            "  1: multimeter",
            "  2: lcd_screen",
            "",
        ]
    )
    (path / "data.yaml").write_text(data_yaml, encoding="utf-8")


def export() -> None:
    args = parse_args()
    if args.overwrite:
        reset_output(args.output_root)
    ensure_output(args.output_root)

    all_records = load_records(args.manifest)
    records = approved_records(all_records)
    export_rows = list(records)
    random.Random(args.seed).shuffle(export_rows)
    val_count = int(round(len(export_rows) * args.val_ratio)) if len(export_rows) > 1 else 0
    val_ids = {row["image_id"] for row in export_rows[:val_count]}
    exported_by_id = {}

    gt_rows = []
    for row in export_rows:
        split = "val" if row["image_id"] in val_ids else "train"
        src = Path(row["copied_image_path"])
        image_dest = args.output_root / "images" / split / f"{row['image_id']}{src.suffix.lower()}"
        label_dest = args.output_root / "labels" / split / f"{row['image_id']}.txt"
        shutil.copy2(src, image_dest)

        image_width = float(row["image_width"])
        image_height = float(row["image_height"])
        lines = []
        for class_id, class_name in enumerate(CLASSES):
            box = parse_box(row.get(f"{class_name}_bbox_xyxy"))
            if box:
                lines.append(yolo_line(class_id, box, image_width, image_height))
        label_dest.write_text("\n".join(lines) + "\n", encoding="utf-8")

        gt_rows.append(
            {
                "image_id": row["image_id"],
                "split": split,
                "image_path": image_dest.relative_to(args.output_root).as_posix(),
                "lcd_screen_value": row.get("lcd_screen_value", ""),
            }
        )
        exported_by_id[row["image_id"]] = {
            "yolo_split": split,
            "yolo_image_path": str(image_dest),
            "yolo_label_path": str(label_dest),
        }

    write_data_yaml(args.output_root)
    with (args.output_root / "ground_truth_lcd_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split", "image_path", "lcd_screen_value"])
        writer.writeheader()
        writer.writerows(gt_rows)

    from datetime import datetime

    exported_at = datetime.now().isoformat(timespec="seconds")
    for row in all_records:
        exported = exported_by_id.get(row.get("image_id"))
        if exported:
            row.update(exported)
            row["yolo_exported_at"] = exported_at
        else:
            row.pop("yolo_split", None)
            row.pop("yolo_image_path", None)
            row.pop("yolo_label_path", None)
            row.pop("yolo_exported_at", None)
    write_records(args.manifest, all_records)

    print(f"Approved records exported: {len(records)}")
    print(f"YOLO dataset: {args.output_root}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")


if __name__ == "__main__":
    export()
