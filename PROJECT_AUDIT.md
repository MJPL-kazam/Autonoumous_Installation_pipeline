# PROJECT AUDIT — PWA Screen/ROI Detection

**Generated:** 2026-06-16  
**Scope:** `E:\2026_kazam\CV_mini3_3_screen_detection\PWA`  
**Status:** ✅ **Production Ready** — Frontend and browser inference pipeline completed.

---

## 1. Environment & Tech Stack

| Layer            | Technology              | Details                              |
|-----------------|------------------------|-------------------------------------|
| Frontend build   | **Vite**               | Hot Module Replacement, fast builds |
| App Logic        | **Vanilla JS**         | No heavy frameworks overhead        |
| Styling          | **Vanilla CSS**        | Custom styling, glassmorphism UI    |
| Deep Learning    | **ONNX Runtime Web**   | Zero-latency browser execution (WASM)|
| Architecture     | **YOLOv11n to ONNX**   | `t400v100.onnx` and `t120v30.onnx`  |
| Offline Support  | **Service Worker**     | Fully configured as a PWA (`sw.js`) |

---

## 2. Directory Structure

```
PWA/
├── model/                  # High-accuracy ONNX export files
│   ├── t120v30.onnx
│   └── t400v100.onnx       # Primary model
├── public/                 # Static Assets
│   ├── model/              # Models served over HTTP dynamically
│   ├── icon-512.png        # PWA Icon
│   ├── manifest.json       # Standard webapp manifest
│   ├── sw.js               # Service Worker logic
│   └── test_sample.jpeg    # Verification image
├── src/                    # Source Code
│   ├── main.js             # Core App logic, inference, NMS filtering, UI
│   └── style.css           # UI Stylesheet
├── index.html              # Entry point
├── package.json            # Node dependencies
└── vite.config.js          # Vite configurations (WASM static copy)
```

---

## 3. Core Implementation Details

### A. Pre-check Quality Enforcement
Before loading the heavy deep learning inference module, `main.js` implements several logical pre-checks analyzing raw pixel arrays from the `<canvas>`:
- Checks resolution limits (Minimum $800 \times 800$).
- Performs **Laplacian Variance** checks to determine the blur score of the image.
- Performs global pixel intensity checks for contrast and overall brightness.
- Analyzes shadow and overexposure ratios.
- Verifies glare ratios.

### B. Machine Learning Pipeline
- Preprocessing standardizes the input image by scaling to a target size of $640 \times 640$, letterbox padding, and array conversion into NCHW formatted Floats.
- The browser triggers the WASM thread executing `t400v100.onnx`.
- Outputs an array spanning `[1, 300, 6]`.

### C. Logic and Non-Maximum Suppression (NMS)
The native output of the ONNX models contains raw predictions. `main.js` filters the results through `keepBestPerClass()` which iterates the predictions and ensures only the **single best detection per class** is processed.

This resolved earlier issues where overlapping bounding boxes triggered multiple detections for the same physical object.

### D. Enclosure Verification
For a full installation verification, the system mathematically verifies that the bounding box belonging to the `lcd_screen` prediction is completely enclosed inside the bounding box belonging to the `multimeter`.

---

## 4. Current State and Fixes

- **Deleted old models:** Obsolete baseline PT models (`best.pt`) and early exports (`best.onnx`) were entirely removed to optimize repository size.
- **Deduplication algorithm:** NMS deduplication algorithm properly filters to unique classes (`kazam_box`, `multimeter`, `lcd_screen`).
- **Dataset location:** Full model dataset properly tracked and hosted on the **Ultralytics Hub** website.
- **PWA support:** Verified functioning service workers enabling offline execution.
