# Two-Stage Grounding DINO Tiny ROI Pipeline

This folder contains the earlier two-stage ROI detection pipeline used before YOLO training. It uses Grounding DINO Tiny to first locate the multimeter in the full technician photo, then runs Grounding DINO again on the multimeter crop to find the LCD screen.

The goal of this code is ROI discovery and inspection. It does not create final YOLO labels by itself.

## Pipeline

Stage 1 runs on the full image:

```text
full technician image -> Grounding DINO Tiny -> multimeter bbox
```

The selected multimeter bbox is cropped and saved.

Stage 2 runs on the meter crop:

```text
meter crop -> Grounding DINO Tiny -> LCD bbox inside meter crop
```

The local LCD bbox is then converted back into full-image coordinates:

```text
lcd_global = lcd_local + meter_top_left
```

The script saves:

```text
output/
  stage1_full_annotated/       full image with meter box
  stage1_meter_crops/          cropped meter images
  stage2_meter_annotated/      meter crop with local LCD box
  stage2_lcd_crops_raw/        raw LCD crop
  stage2_lcd_crops_processed/  thresholded LCD crop for OCR experiments
  final_full_annotated/        full image with meter and LCD boxes
  results.csv                  bbox coordinates, confidences, and statuses
  logs/                        run logs
```

## Model

The pipeline uses:

```text
IDEA-Research/grounding-dino-tiny
```

This was chosen as the lightweight DINO option for quick experiments. It is useful for bootstrapping ROI candidates, but it should not be treated as final ground truth.

## Prompts

Stage 1 prompt:

```text
Kazam EV charger wall box . handheld digital multimeter .
```

The Stage 1 code keeps meter-like detections and filters out large boxes using area thresholds so charger boxes do not dominate the crop selection.

Stage 2 prompt:

```text
small rectangular LCD display screen on multimeter, grey green digital display, only seven segment digits, numeric reading screen, not rotary dial, not labels, not buttons, not ports.
```

Stage 2 prefers horizontal rectangular boxes because LCD screens are usually wider than tall.

## Packages

Create and activate a Python environment, then install:

```powershell
pip install -r dino_tiny_roi/requirements.txt
```

Main packages:

```text
torch
torchvision
transformers
opencv-python
Pillow
numpy
tqdm
PyYAML
jupyter
easyocr
```

`easyocr` is included because the original notebook had an OCR experiment after LCD crop extraction. The standalone script focuses on the two-stage DINO ROI part and writes crops/results for later OCR or review.

## How To Run

Put raw images under these folders by default:

```text
Data/N to E/
Data/P to E/
Data/P to N/
```

Then run from the repository root:

```powershell
python dino_tiny_roi/two_stage_dino_tiny_pipeline.py
```

To change paths, prompts, model, or thresholds, edit:

```text
dino_tiny_roi/config.yaml
```

You can also run with an alternate config:

```powershell
python dino_tiny_roi/two_stage_dino_tiny_pipeline.py --config dino_tiny_roi/config.yaml
```

## Notebook

The original exploration notebook is included as:

```text
dino_tiny_roi/screen_detection_pipeline.ipynb
```

It contains the full experimental flow:

1. Load raw images from `Data/`.
2. Load Grounding DINO Tiny.
3. Detect multimeter in the full image.
4. Crop the multimeter.
5. Detect LCD inside the meter crop.
6. Save raw and processed LCD crops.
7. Run OCR experiments on the LCD crop.
8. Save `results.csv`.

## Important Notes

- DINO confidence scores are only candidate metadata.
- The annotated preview images are not training data.
- This branch is for the old two-stage Tiny DINO approach.
- The later YOLO dataset branch uses manually corrected boxes and trains YOLO on clean raw images.
