# YOLO ROI Dataset

This folder contains the manually verified YOLO dataset and helper tools for detecting three ROIs in technician installation photos:

- `kazam_box`
- `multimeter`
- `lcd_screen`

The dataset is intended for ROI detection only. LCD readings are stored as metadata in `data/yolo_dataset/ground_truth_lcd_values.csv`; they are not YOLO classes.

## Dataset

Final exported dataset:

```text
yolo_ROI/data/yolo_dataset/
  data.yaml
  ground_truth_lcd_values.csv
  images/train/    67 images
  images/val/      17 images
  labels/train/    67 YOLO label files
  labels/val/      17 YOLO label files
```

YOLO class mapping:

```text
0 kazam_box
1 multimeter
2 lcd_screen
```

Each approved image has one box for each class.

## Environment Used

The dataset preparation was run with Python `3.9.13` in a local virtual environment.

Installed packages used by the preparation scripts:

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
transformers==4.57.6
opencv-python==4.13.0.92
Pillow==11.3.0
tqdm==4.67.3
```

For YOLO training, install Ultralytics separately:

```powershell
python -m pip install ultralytics
```

Do not commit `.venv`, package folders, model weights, or training runs.

## How The Dataset Was Created

1. Raw technician photos were read from the project `Data/` folder.
2. `run_dino_candidates.py` used Grounding DINO Base (`IDEA-Research/grounding-dino-base`) to create candidate boxes.
3. DINO was run in two stages:
   - Stage 1 on the full image to suggest `kazam_box` and `multimeter`.
   - Stage 2 on the selected multimeter crop to suggest `lcd_screen`.
4. The DINO boxes were treated only as suggestions.
5. `review_app.py` was used to manually review every photo:
   - correct or redraw all three ROIs,
   - approve or reject the photo,
   - enter the LCD screen value.
6. `export_yolo_dataset.py` exported only approved images into YOLO format.

The final YOLO training images are clean raw images. They do not contain drawn boxes, confidence text, or preview overlays.

## Commands

Run commands from the repository root.

Generate or refresh DINO candidate suggestions:

```powershell
python yolo_ROI/run_dino_candidates.py
```

Open the manual review UI:

```powershell
python yolo_ROI/review_app.py
```

Export approved records to YOLO format:

```powershell
python yolo_ROI/export_yolo_dataset.py --overwrite
```

Train a baseline YOLO detector:

```powershell
python -m ultralytics train model=yolo26n.pt data=yolo_ROI/data/yolo_dataset/data.yaml epochs=100 imgsz=640 batch=4 device=0
```

Validate the trained model:

```powershell
python -m ultralytics val model=runs/detect/train/weights/best.pt data=yolo_ROI/data/yolo_dataset/data.yaml
```

Preview predictions:

```powershell
python -m ultralytics predict model=runs/detect/train/weights/best.pt source=yolo_ROI/data/yolo_dataset/images/val save=True conf=0.25
```

## Notes

- DINO confidence values are only review metadata and are not used as YOLO training targets.
- Intermediate DINO previews, copied raw review images, and crop folders are intentionally excluded from Git.
- If LCD detection is weak after first training, use this dataset as the baseline and consider a second-stage model for `multimeter crop -> lcd_screen`.
