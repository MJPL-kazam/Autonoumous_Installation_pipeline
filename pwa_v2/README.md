# Kazam Site Check PWA v2

Kazam Site Check is an offline-capable Progressive Web App for installation photo verification. It runs two YOLO ONNX models directly in the browser with `onnxruntime-web` and WebAssembly, so technicians can validate photos without sending images to a backend inference service.

## What This App Checks

The app guides a technician through required installation photos and applies model-based and rule-based validation:

- Full setup photo with Kazam box, multimeter, and LCD screen.
- Multimeter reading photo with the LCD screen visible.
- Additional installation evidence photos.
- Earth pit photos, including open pit and covered pit states.
- Basic image quality checks before inference, including resolution, blur, brightness, contrast, shadow, overexposure, and glare.
- Spatial validation that the detected LCD screen is inside the detected multimeter for relevant sections.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | Vanilla JavaScript, HTML, CSS |
| Build tool | Vite |
| Runtime ML | ONNX Runtime Web |
| Inference backend | WebAssembly |
| Model family | Ultralytics YOLO ONNX exports |
| Offline support | Web App Manifest and Service Worker |
| Package manager | npm |

## ONNX Models

The app uses two separate ONNX models because the installation checks require different detection behavior.

| Model | File | Architecture | Classes | Output |
| --- | --- | --- | --- | --- |
| Multimeter setup model | `public/model/multimeter_t400v100.onnx` | YOLO26n-obb | `lcd_screen`, `multimeter`, `kazam_box` | `[1, 300, 7]`: `[cx, cy, w, h, confidence, class_id, angle]` |
| Earth pit model | `public/model/pothole_t417v100.onnx` | YOLO26n detection | `earthpit_open`, `earthpit_cover` | `[1, 300, 6]`: `[x1, y1, x2, y2, confidence, class_id]` |

Both models are exported with end-to-end NMS. The browser code still performs final confidence filtering, class mapping, coordinate scaling, and best-per-class selection for the UI.

## Important Files

Only the files required to run, build, and manually test the PWA are included in this folder.

```text
pwa_v2/
|-- index.html
|-- package.json
|-- package-lock.json
|-- vite.config.js
|-- README.md
|-- MOBILE_TESTING.md
|-- .gitignore
|-- src/
|   |-- main.js
|   `-- style.css
`-- public/
    |-- manifest.json
    |-- sw.js
    |-- icon-512.png
    |-- model/
    |   |-- multimeter_t400v100.onnx
    |   `-- pothole_t417v100.onnx
    `-- test-images/
        |-- MAH-cparxutx9xf__Meter_Photo_(N-E).jpg
        |-- MAH-cparydw2rne__Open_Earthpit.jpg
        `-- MAH-cparyhksyoe__Earthpit.jpg
```

## What Is Intentionally Excluded

The clean folder does not include:

- `node_modules/`
- `dist/`
- old `testing/` scripts
- preprocessing comparison scripts
- generated JSON reports
- raw tensor dumps
- full `test_image/` diagnostics
- duplicate or obsolete ONNX model folders

The three files in `public/test-images/` are the only sample images kept for manual verification.

## Architecture

1. `index.html` loads the app shell and `src/main.js`.
2. `src/main.js` initializes the PWA UI, service worker, and two ONNX Runtime sessions.
3. Images are loaded from camera or file input and passed through canvas-based preprocessing.
4. Preprocessing letterboxes each image to `640x640`, pads with RGB `114,114,114`, normalizes to `[0, 1]`, and creates an NCHW tensor.
5. The correct model is selected by section:
   - Earth pit sections use `pothole_t417v100.onnx`.
   - All other AI-enabled installation sections use `multimeter_t400v100.onnx`.
6. `postprocess()` applies model-specific output parsing:
   - OBB model output is parsed as center-width-height plus angle.
   - Earth pit model output is parsed as direct `xyxy` boxes.
7. Detections are filtered, scaled back to the original image size, and drawn in the UI.
8. `keepBestPerClass()` keeps the strongest detection for each class before final section validation.
9. `public/sw.js` caches the app shell, ONNX models, WASM runtime files, and sample images for offline use.

## Local Development

Install dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

Open the app:

```text
http://localhost:5173/
```

Build production assets:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

## Manual Test Images

Use these files for quick validation:

| File | Expected primary detection |
| --- | --- |
| `public/test-images/MAH-cparxutx9xf__Meter_Photo_(N-E).jpg` | Kazam box, multimeter, LCD screen |
| `public/test-images/MAH-cparydw2rne__Open_Earthpit.jpg` | `earthpit_open` |
| `public/test-images/MAH-cparyhksyoe__Earthpit.jpg` | `earthpit_cover` |

Developer mode also includes buttons to load the full setup sample and open earth pit sample from `public/test-images/`.

## Deployment Notes

- The app uses Vite `base: './'` so built assets can be served from a subfolder.
- Runtime model, WASM, service worker, and sample-image URLs are derived from `import.meta.env.BASE_URL`.
- The service worker is scope-aware and avoids serving HTML fallbacks for JavaScript, CSS, model, or WASM asset requests.
- Host production builds over HTTPS to enable service worker registration and PWA installation.
