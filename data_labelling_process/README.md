# DINO Auto-Labeling and Manual Review Pipeline

This repository contains a semi-autonomous data labeling pipeline for Kazam installation images. It uses Grounding DINO to automatically generate zero-shot bounding boxes for required components, followed by a local web interface for a human-in-the-loop to verify and correct those boxes.

## Requirements & Setup
Install the necessary dependencies using:
```bash
pip install -r requirements.txt
```

## How It Works

### Step 1: AI Auto-Labeling (`run_dino.py`)
This script processes a folder of raw images, loads the Grounding DINO model on your GPU, and automatically predicts bounding boxes for 5 classes:
1. `kazam_box`: The Kazam EV wall charger.
2. `multimeter`: A clamp multimeter or digital meter.
3. `lcd_screen`: The small LCD display on the multimeter.
4. `kazam_box_black_cover`: The dark translucent flap on the box.
5. `qr_code`: The QR sticker on the box.

**To run:**
```bash
python run_dino.py --limit-folders 100
```
This generates a `dino_auto_label/review_manifest.json` file which holds all the predictions. The script will automatically skip files that have already been processed if you stop and restart it.

### Step 2: Manual Verification (`review_app.py`)
Because AI isn't perfect, we use a custom local web app to review and adjust the AI's bounding boxes.

**To run:**
```bash
python review_app.py
```
This will launch a web server at `http://127.0.0.1:7862`.

**Workflow in the UI:**
- **Adjusting Boxes:** Click a class on the left (or use keyboard numbers 1-5). Click and drag on the image to draw a new bounding box. The new box instantly replaces the old one for that class.
- **Entering LCD Value:** Type the physical reading on the multimeter screen into the text box.
- **Rejecting:** If the image is unusable or bad, click the **Reject** button. The image path will be saved to `bad_images.txt` and the system will move to the next image.
- **Saving:** Click **Save + Next** to approve the image. 

> **Important Rule:** The system strictly enforces that an image can ONLY be approved if it has the 3 required boxes (`kazam_box`, `multimeter`, `lcd_screen`) AND an entered LCD value. The black cover and QR code are optional.

### Step 3: Dataset Export (`export_yolo.py`)
*(Note: This script is a first step for making the dataset for training purposes and is **still not tested**.)*

Once you have manually approved your images, run this script to convert the `review_manifest.json` database into a clean YOLO-formatted dataset (`.txt` files with normalized xywh coordinates). It will only export images that have been marked as `approved`.

### Bonus: Visual Previews (`generate_final_previews.py`)
This script will draw the final, manually-approved boxes onto the images and save them. This is useful if you want to quickly scroll through the images in Windows Explorer to verify that your manual labels are perfect before starting training.
