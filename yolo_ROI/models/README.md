# YOLO ROI Models

This folder stores selected trained YOLO ROI checkpoints.

## Current Checkpoint

```text
yolo26n_roi_e50_img640_best.pt
```

Source run:

```text
yolo_ROI/yolyo_train/yolo26n_roi_e50_img640/weights/best.pt
```

Training setup:

```text
model: yolo26n.pt
epochs: 50
imgsz: 640
classes:
  0 kazam_box
  1 multimeter
  2 lcd_screen
```

Use this checkpoint for ROI inference unless a later validation run proves a newer model is better.
