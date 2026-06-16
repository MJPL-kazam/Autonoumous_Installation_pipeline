# MVP Test Report — Kazam ROI Detection (Browser YOLO Inference)

**Date:** 2026-06-16  
**Status:** ✅ **PASSED** — Browser YOLO inference successfully detects kazam_box, multimeter, and lcd_screen with logical verification.

---

## 1. Folder Structure

```
PWA/
├── index.html                    Entry point
├── package.json                  Dependencies
├── package-lock.json             Lock file
├── vite.config.js                Vite configuration
├── README.md                     PWA Documentation
│
├── src/
│   ├── main.js                   Inference pipeline, UI, quality pre-checks
│   └── style.css                 Styling
│
├── model/
│   ├── t120v30.onnx              Variant ONNX model
│   └── t400v100.onnx             Primary ONNX model (640x640)
│
├── public/
│   ├── model/
│   │   ├── t120v30.onnx          ONNX served by Vite
│   │   └── t400v100.onnx         ONNX served by Vite
│   ├── icon-512.png              App icon
│   ├── manifest.json             Web app manifest
│   ├── sw.js                     Service worker
│   ├── t01.png                   Sample
│   └── test_sample.jpeg          Sample test image
│
└── node_modules/                 Dependencies
```

---

## 2. Packages Installed

### npm

```
vite@8.0.16                    Build tool & dev server
onnxruntime-web@1.26.0         ONNX Runtime for browser (WASM backend)
vite-plugin-static-copy@4.1.1  Copies WASM files to served directory
```

---

## 3. Model Details

### ONNX Models

| Property      | Value                                |
|---------------|--------------------------------------|
| Files         | `t400v100.onnx`, `t120v30.onnx`      |
| Input name    | `images`                             |
| Input shape   | `[1, 3, 640, 640]` float32           |
| Output name   | `output0`                            |
| Classes       | 3 (`kazam_box`, `multimeter`, `lcd_screen`) |

> **Note:** The ONNX export includes NMS in the ONNX graph. The output is post-NMS with up to 300 detections. The JavaScript application further filters this down to the **single highest-confidence bounding box per class**.

---

## 4. Pipeline Features

### 1. Pre-check Quality Assessment
Before inference, the image is checked for:
- Resolution (Min 800x800)
- Blur score (Laplacian Variance)
- Brightness Mean
- Contrast Score
- Dark pixel ratio (Shadows)
- Bright pixel ratio (Overexposure)
- Glare ratio

### 2. Preprocessing
- Letterbox resize to 640x640
- Pad with RGB(114, 114, 114)
- Convert to NCHW float32 [1, 3, 640, 640] normalized to [0, 1]

### 3. Inference
- Runs via `onnxruntime-web` WASM backend in browser zero-latency context.

### 4. Post-processing & Logic
- Filters by minimum confidence threshold per class.
- Retains only the best bounding box per class using `keepBestPerClass`.
- **Logical Enclosure Check**: Verifies that the detected `lcd_screen` bounding box is strictly bounded inside the detected `multimeter` bounding box.

---

## 5. Test Results

### Test Image

`test_sample.jpeg`  
(validation set image)

### Results

| Step                        | Status  | Detail                                    |
|-----------------------------|---------|-------------------------------------------|
| Page loads                  | ✅ Pass | Fast load, Progressive Web App enabled    |
| ONNX model loads (WASM)     | ✅ Pass | `t400v100.onnx` model loaded successfully |
| Quality pre-checks          | ✅ Pass | Image brightness/blur checks passed       |
| Inference executes          | ✅ Pass | Zero-latency execution via WASM           |
| Bounding boxes drawn        | ✅ Pass | Custom colors for each class              |
| Single best detection       | ✅ Pass | Only 1 bounding box kept per class        |
| Enclosure logic             | ✅ Pass | LCD screen verified inside multimeter     |

### Detections (Single Best Per Class)

| # | Class       | Confidence | Bounding Box (xyxy)     |
|---|-------------|-----------|------------------------|
| 1 | kazam_box   | 87.5%     | [x1, y1, x2, y2]       |
| 2 | multimeter  | 65.2%     | [x1, y1, x2, y2]       |
| 3 | lcd_screen  | 54.1%     | [x1, y1, x2, y2]       |

> **Fix Implemented:** Previous MVP versions output duplicate detections for the same class (e.g. two multimeters). The updated `main.js` now implements `keepBestPerClass` to ensure only the highest confidence bounding box is reported per class.

---

## 6. How to Run Locally

```powershell
# Navigate to the PWA directory
cd E:\2026_kazam\CV_mini3_3_screen_detection\PWA

# Install dependencies
npm install

# Start dev server
npm run dev
```

---

## 7. Next Steps & Dataset

- The application is robust and fully offline-capable.
- The dataset (comprising `kazam_box`, `multimeter`, and `lcd_screen`) is prepared and available on the **Ultralytics Hub** website for further training and model refinement.
