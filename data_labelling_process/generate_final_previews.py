import argparse
import json
from pathlib import Path
import cv2
from typing import Optional

# Define classes and BGR colors
CLASS_STYLES = {
    "kazam_box": {"label": "0 Kazam box", "color": (40, 80, 255)},          # Orange (BGR)
    "multimeter": {"label": "1 Multimeter", "color": (0, 255, 255)},        # Yellow
    "lcd_screen": {"label": "2 LCD screen", "color": (40, 40, 255)},        # Red
    "kazam_box_black_cover": {"label": "3 Black cover", "color": (0, 0, 0)},# Black
    "qr_code": {"label": "4 QR code", "color": (255, 0, 255)},              # Magenta
}

def parse_box(value) -> Optional[list[int]]:
    if not value:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or len(value) != 4:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None

def draw_box(bgr, roi: str, box: Optional[list[int]]) -> None:
    if not box:
        return
    x1, y1, x2, y2 = box
    style = CLASS_STYLES.get(roi, {"color": (255, 255, 255), "label": roi})
    color = style["color"]
    label = style["label"]

    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness=4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thick = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
    text_y = max(y1, text_h + 10)
    
    cv2.rectangle(bgr, (x1, text_y - text_h - 5), (x1 + text_w, text_y + baseline - 5), color, -1)
    # White text (except for yellow/white boxes where we might use black, but white is fine for most)
    text_color = (0, 0, 0) if roi in ("multimeter",) else (255, 255, 255)
    cv2.putText(bgr, label, (x1, text_y - 5), font, font_scale, text_color, font_thick)

def main():
    parser = argparse.ArgumentParser(description="Generate visual previews with manually approved boxes.")
    parser.add_argument("--manifest", type=Path, default=Path("dino_auto_label/review_manifest.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("../mannual_label"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing previews")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    approved_rows = [r for r in rows if r.get("review_status") == "approved"]
    print(f"Found {len(approved_rows)} approved records.")

    processed = 0
    skipped = 0

    for row in approved_rows:
        img_id = row.get("image_id", "")
        if not img_id:
            continue

        out_path = out_dir / f"{img_id}_approved.jpg"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        src_path = row.get("original_image_path", "")
        if not src_path or not Path(src_path).exists():
            print(f"Original image not found for {img_id}: {src_path}")
            continue

        # Load image
        bgr = cv2.imread(str(src_path))
        if bgr is None:
            print(f"Failed to read image: {src_path}")
            continue

        # Draw boxes
        for class_name in CLASS_STYLES.keys():
            box = parse_box(row.get(f"{class_name}_bbox_xyxy"))
            if box:
                draw_box(bgr, class_name, box)

        # Draw LCD value text if present
        lcd_val = row.get("lcd_screen_value", "")
        if lcd_val:
            cv2.putText(bgr, f"LCD: {lcd_val}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Save result
        cv2.imwrite(str(out_path), bgr)
        processed += 1
        print(f"Saved: {out_path.name}")

    print("-" * 40)
    print(f"Done. Generated {processed} new images. Skipped {skipped} existing images.")
    print(f"Output directory: {out_dir}")

if __name__ == "__main__":
    main()
