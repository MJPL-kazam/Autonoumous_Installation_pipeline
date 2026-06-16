# PROJECT AUDIT — PWA Screen/ROI Detection

**Generated:** 2026-06-10  
**Scope:** `E:\2026_kazam\CV_mini3_3_screen_detection\PWA`  
**Status:** Discovery & Planning — No code changes made

---

## 1. Folder Structure

```
CV_mini3_3_screen_detection/
├── .gitignore                              (60 B)
├── .venv/                                  ← Python 3.9.13 virtual environment
├── benchmark.py                            (18 KB)
├── evaluation.csv
├── generate_variant_comparisons.py         (3 KB)
├── ground_truth.csv                        (3.7 KB)
├── PIPELINE_REPORT.md                      (10 KB)
├── screen_detection_pipeline.ipynb         (34 KB)
├── yolo26n.pt                              (5.29 MB) ← pretrained base model
│
├── Data/                                   ← raw technician photos
├── Data_preprocessing_quality_check/
│   ├── config.yaml
│   ├── quality_gate.py                     (24 KB)
│   ├── mixed_images_for_testing/
│   └── outputs/
│
├── logs/
├── mannual_labelling/
│   ├── manual_bbox_label_app.py            (21 KB)
│   └── manual_bbox_annotations/
│
├── output/
│
├── PWA/                                    ← ★ TARGET DIRECTORY
│   └── model/
│       └── best.pt                         (19.79 MB) ← trained YOLO weights
│
├── runs/
│   └── detect/yolo_ROI/                    ← detection run outputs
│
├── sam_collab/
│   └── label_app.py                        (7 KB)
│
├── src/
│   ├── benchmark/
│   │   ├── metrics.py                      (2 KB)
│   │   └── visualization.py                (5 KB)
│   ├── ocr/
│   │   ├── engines.py                      (6 KB)
│   │   └── seven_segment_reader.py         (10 KB)
│   └── preprocessing/
│       └── variants.py                     (5 KB)
│
├── yolo_ROI/                               ← YOLO training pipeline
│   ├── README.md
│   ├── train_config.yaml
│   ├── train_yolo.py
│   ├── export_yolo_dataset.py
│   ├── review_app.py
│   ├── run_dino_candidates.py
│   ├── data/yolo_dataset/                  ← 84 images (67 train / 17 val)
│   └── yolyo_train/
│       ├── yolo26n_roi_baseline/weights/
│       │   ├── best.pt                     (19.79 MB)
│       │   └── last.pt                     (19.79 MB)
│       └── yolo26n_roi_e50_img640/weights/
│           ├── best.pt                     (5.12 MB) ← best stripped model
│           └── last.pt                     (5.12 MB)
│
└── _github_publish/
    └── Autonoumous_Installation_pipeline/
        └── yolo_ROI/models/
            └── yolo26n_roi_e50_img640_best.pt  (5.12 MB)
```

### Key Observations

- The `PWA/` directory is **nearly empty** — it contains only a `model/` folder with a single `.pt` file.
- **No frontend code exists.** No HTML, CSS, JS, React, or any web framework files are present.
- **No build system exists.** No `package.json`, `vite.config.*`, `webpack.config.*`, or `tsconfig.*` anywhere in the PWA directory.
- The project is a fully **Python-based CV pipeline** with no web component yet.

---

## 2. Environment Summary

| Tool       | Version      | Status       |
|------------|-------------|-------------|
| Python     | 3.9.13      | ✅ Available |
| Node.js    | v25.6.1     | ✅ Available |
| npm        | 11.9.0      | ✅ Available |
| pnpm       | 10.28.2     | ✅ Available |
| yarn       | —           | ❌ Not installed |
| conda      | —           | ❌ Not installed |
| poetry     | —           | ❌ Not installed |

---

## 3. Virtual Environment

| Item                 | Status                                    |
|----------------------|-------------------------------------------|
| Location             | `.venv/` (standard venv)                  |
| Python version       | 3.9.13                                    |
| pip version          | 22.0.4 (outdated — latest is 26.0.1)     |
| ultralytics          | ✅ 8.4.51                                 |
| onnx                 | ❌ Not installed                           |
| onnxruntime          | ❌ Not installed                           |
| torch                | 2.5.1+cu121                               |
| torchvision          | 0.20.1+cu121                              |

### Key Installed Packages

| Package              | Version        | Relevance                    |
|----------------------|---------------|------------------------------|
| ultralytics          | 8.4.51        | YOLO training & export       |
| torch                | 2.5.1+cu121   | PyTorch with CUDA 12.1       |
| torchvision          | 0.20.1+cu121  | Vision transforms            |
| transformers         | 4.57.6        | Grounding DINO               |
| opencv-python        | 4.13.0.92     | Image processing             |
| easyocr              | 1.7.2         | OCR engine                   |
| pytesseract          | 0.3.13        | OCR engine                   |
| paddlepaddle         | 3.3.1         | PaddleOCR backend            |
| paddlex              | 3.5.2         | PaddleX toolkit              |
| supervision          | 0.28.0        | Detection visualization      |
| scikit-image         | 0.24.0        | Image processing             |
| pillow               | 11.3.0        | Image I/O                    |

---

## 4. YOLO Model Summary

### All Discovered Model Files

| File | Size | Location | Notes |
|------|------|----------|-------|
| `best.pt` | **19.79 MB** | `PWA/model/` | Baseline run, epoch 9, **includes optimizer state** |
| `best.pt` | 19.79 MB | `yolo_ROI/yolyo_train/yolo26n_roi_baseline/weights/` | Same baseline checkpoint |
| `last.pt` | 19.79 MB | `yolo_ROI/yolyo_train/yolo26n_roi_baseline/weights/` | Baseline last epoch |
| `best.pt` | **5.12 MB** | `yolo_ROI/yolyo_train/yolo26n_roi_e50_img640/weights/` | **Best candidate — 50 epoch, stripped** |
| `last.pt` | 5.12 MB | `yolo_ROI/yolyo_train/yolo26n_roi_e50_img640/weights/` | 50 epoch last checkpoint |
| `yolo26n_roi_e50_img640_best.pt` | 5.12 MB | `_github_publish/.../models/` | Published copy of e50 best |
| `yolo26n.pt` | 5.29 MB | Root | Pretrained base model (no fine-tune) |

### No `.onnx` or `.tflite` files found anywhere in the workspace.

### Model Details

| Property             | Value                                    |
|----------------------|------------------------------------------|
| Architecture         | YOLOv11n (YOLO26n / Ultralytics)         |
| Task                 | **Detection** (`detect`)                 |
| Parameters           | **2,504,970** (~2.5M)                    |
| Classes              | 3                                        |
| Class names          | `kazam_box`, `multimeter`, `lcd_screen`  |
| Training image size  | 640×640                                  |
| Training epochs      | 50 (best run)                            |
| Best fitness         | 0.579 (baseline), expected higher for e50 |
| Model loadable?      | ✅ Yes — verified with ultralytics 8.4.51 |

### Critical Finding: Wrong Model in PWA

> [!WARNING]
> The `PWA/model/best.pt` (19.79 MB) is the **baseline** model (only 10 epochs, fitness 0.578) and includes optimizer state bloat. The **better model** is `yolo26n_roi_e50_img640/weights/best.pt` (5.12 MB, 50 epochs, stripped). For PWA deployment, use the 5.12 MB version.

---

## 5. Frontend Summary

| Item                     | Status                |
|--------------------------|-----------------------|
| Frontend framework       | ❌ **None**           |
| HTML files               | ❌ None               |
| JavaScript / TypeScript  | ❌ None               |
| CSS files                | ❌ None               |
| package.json             | ❌ None               |
| Build system             | ❌ None               |
| Dev server               | ❌ None               |

**The PWA directory has no frontend code.** It is a blank slate containing only the model weights file. The entire existing project is Python-based (Jupyter notebooks, Python scripts, CLI tools).

---

## 6. PWA Readiness

| Item                  | Status              |
|-----------------------|---------------------|
| `manifest.json`       | ❌ Does not exist    |
| Service Worker        | ❌ Does not exist    |
| PWA plugin/library    | ❌ None installed    |
| HTTPS config          | ❌ Not configured    |
| Responsive meta tags  | ❌ None              |
| Icons / splash        | ❌ None              |

**PWA readiness: 0%.** Everything must be built from scratch.

---

## 7. Browser Inference Feasibility

### Option A: PT → ONNX → ONNX Runtime Web

| Criterion              | Assessment                                                |
|------------------------|----------------------------------------------------------|
| **Complexity**         | **Low–Medium.** Ultralytics has built-in `model.export(format='onnx')`. ONNX Runtime Web is mature with good API. |
| **Export effort**      | One command: `yolo export model=best.pt format=onnx opset=17 simplify=True` |
| **Dependencies to install** | `onnx`, `onnxruntime`, `onnxslim` (for export). Browser side: `onnxruntime-web` npm package. |
| **Browser compat**     | ✅ All modern browsers (Chrome, Firefox, Safari, Edge). WebAssembly backend universal. WebGL backend for GPU. |
| **Android compat**     | ✅ Chrome Android fully supported. WebGL acceleration available on most devices. |
| **WASM backend**       | ✅ CPU fallback, works everywhere                          |
| **WebGL/WebGPU**       | ✅ GPU acceleration, 3–10× faster than WASM               |
| **Expected perf**      | ~50–150ms per frame on mid-range Android (WebGL). ~200–500ms on WASM. Model is small (2.5M params). |
| **Model size (ONNX)**  | ~5–6 MB (similar to stripped .pt, possibly slightly larger due to float32) |
| **Quantization**       | Supports INT8 quantization to ~1.5 MB if needed            |
| **Post-processing**    | Must implement NMS in JavaScript (or include in ONNX graph via Ultralytics export flag) |

### Option B: PT → TensorFlow.js

| Criterion              | Assessment                                                |
|------------------------|----------------------------------------------------------|
| **Complexity**         | **High.** Requires PT → ONNX → TF SavedModel → TFJS. Multi-step, fragile conversion chain. |
| **Export effort**      | Ultralytics does not directly export to TFJS. Must go through intermediate formats. |
| **Browser compat**     | ✅ Good. TensorFlow.js is mature.                          |
| **Android compat**     | ✅ Chrome Android supported.                               |
| **Expected perf**      | Comparable to ONNX Runtime Web, sometimes slightly slower. |
| **Model size**         | Often larger due to graph format overhead.                 |
| **Post-processing**    | Must implement NMS in JavaScript.                          |
| **Risk**               | ⚠️ YOLO architectures frequently have conversion issues with TFJS. Debugging is painful. |

### Option C: PT → TFLite + WASM

| Criterion              | Assessment                                                |
|------------------------|----------------------------------------------------------|
| **Complexity**         | **High.** Requires PT → ONNX → TF → TFLite, then a WASM runtime wrapper. |
| **Export effort**      | Ultralytics supports `format='tflite'` but adds TensorFlow dependency (~2GB install). |
| **Browser compat**     | ⚠️ Limited. TFLite WASM support is experimental and less maintained. |
| **Android compat**     | ⚠️ Better suited for native Android (Java/Kotlin), not browser. |
| **Expected perf**      | WASM only — no GPU acceleration in browser. Slowest option. |
| **Risk**               | ⚠️ TFLite Web is not production-ready for complex YOLO models. |

### ✅ Recommendation: **Option A — ONNX Runtime Web**

> [!IMPORTANT]
> **Option A is the clear winner.** It has the simplest export path (single command), best browser/Android compatibility, GPU acceleration via WebGL/WebGPU, the most active maintenance, and the smallest risk surface. The 2.5M parameter YOLO model is well within ONNX Runtime Web's performance envelope for real-time mobile inference.

---

## 8. Model Export Assessment

### Export Command (do not run yet)

```bash
# From the project root, using the best trained model (5.12 MB):
.\.venv\Scripts\python.exe -c "
from ultralytics import YOLO
model = YOLO('yolo_ROI/yolyo_train/yolo26n_roi_e50_img640/weights/best.pt')
model.export(format='onnx', opset=17, simplify=True, imgsz=640)
"
```

### Missing Dependencies for Export

| Package       | Required For              | Status          |
|--------------|--------------------------|-----------------|
| `onnx`       | ONNX graph construction   | ❌ Not installed |
| `onnxruntime`| ONNX model verification   | ❌ Not installed |
| `onnxslim`   | Graph optimization        | ❌ Not installed |

Install command (when ready):
```bash
.\.venv\Scripts\python.exe -m pip install onnx onnxruntime onnxslim
```

### Estimated Export Sizes

| Format              | Estimated Size | Notes                          |
|--------------------|---------------|--------------------------------|
| ONNX (float32)     | ~5–6 MB       | Direct export                  |
| ONNX (float16)     | ~2.5–3 MB     | Half precision, minimal accuracy loss |
| ONNX (int8 quant)  | ~1.3–1.5 MB   | Requires calibration dataset   |

---

## 9. Recommended Architecture

```
┌──────────────────────────────────────────────────────┐
│                    PWA Application                     │
│                                                        │
│  ┌──────────┐  ┌───────────────┐  ┌────────────────┐ │
│  │ Camera   │→ │ ONNX Runtime  │→ │ Post-process   │ │
│  │ Stream   │  │ Web (WebGL)   │  │ (NMS + Draw)   │ │
│  │ (getUserM│  │               │  │                │ │
│  │  edia)   │  │ best.onnx     │  │ Bounding boxes │ │
│  └──────────┘  └───────────────┘  └────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Service Worker (cache model + app shell)         │ │
│  └──────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────┐ │
│  │ manifest.json (installable PWA)                  │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### Tech Stack Recommendation

| Layer            | Technology              | Reason                              |
|-----------------|------------------------|-------------------------------------|
| Build tool       | **Vite**               | Fast, modern, excellent PWA plugin  |
| Framework        | **Vanilla JS** or **React** (Lite) | Minimal overhead for camera + canvas |
| Inference        | **ONNX Runtime Web**   | Best perf, WebGL GPU, WASM fallback |
| Camera           | `getUserMedia` API     | Standard browser camera access      |
| PWA              | `vite-plugin-pwa`      | Auto-generates manifest + SW        |
| Rendering        | HTML5 Canvas           | Draw bounding boxes on video frame  |

---

## 10. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Wrong model in PWA/model/** | 🟡 Medium | Replace with `yolo26n_roi_e50_img640` best.pt (5.12 MB) before export |
| **Python 3.9 may lack ONNX export features** | 🟡 Medium | Ultralytics 8.4.51 supports 3.9; verify export works before upgrading |
| **No ONNX/onnxruntime installed** | 🟢 Low | Simple pip install, no conflicts expected |
| **Camera permissions on mobile** | 🟡 Medium | Requires HTTPS in production; `localhost` works for dev |
| **YOLO post-processing in JS** | 🟡 Medium | NMS implementation needed; libraries like `yolov8-tfjs` offer reference code |
| **Model accuracy (fitness 0.578)** | 🟡 Medium | Small dataset (84 images). May need more data for production use |
| **WebGL not available on all devices** | 🟢 Low | WASM fallback ensures universal compat; just slower |
| **Service Worker caching 5 MB model** | 🟢 Low | Well within SW cache limits; use cache-first strategy |
| **iOS Safari getUserMedia quirks** | 🟡 Medium | Test on iOS Safari; may need `playsinline` attribute on video |

---

## 11. Next Implementation Steps

### Phase 2: Setup (requires approval)

1. **Replace model** — Copy `yolo26n_roi_e50_img640/weights/best.pt` → `PWA/model/best.pt`
2. **Install ONNX dependencies** — `pip install onnx onnxruntime onnxslim`
3. **Export to ONNX** — `model.export(format='onnx', opset=17, simplify=True, imgsz=640)`
4. **Verify ONNX model** — Load and run a test inference with onnxruntime

### Phase 3: Frontend Build

5. **Initialize Vite project** — `npx -y create-vite@latest ./ --template vanilla`
6. **Install PWA plugin** — `npm install vite-plugin-pwa -D`
7. **Install ONNX Runtime Web** — `npm install onnxruntime-web`
8. **Create app shell** — Camera view + canvas overlay + detection UI
9. **Implement inference pipeline** — Load ONNX → preprocess frame → run → NMS → draw boxes
10. **Add PWA manifest + service worker** — Icons, offline cache, installability

### Phase 4: Testing & Polish

11. **Test on Android Chrome** — Verify camera + inference performance
12. **Test on iOS Safari** — Verify compatibility
13. **Optimize** — Consider float16 or int8 quantization if perf is insufficient
14. **Deploy** — HTTPS hosting (GitHub Pages, Netlify, or Vercel)

---

> [!NOTE]
> **This report is complete. No code was modified, no packages were installed, and no models were exported.** Awaiting approval before proceeding to Phase 2.
