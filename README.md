# Kazam ROI Quality Inspection PWA

A Progressive Web Application (PWA) designed for real-time, on-device region-of-interest (ROI) detection and quality checks on technician installation photos. 

This application uses ONNX Runtime Web (`onnxruntime-web`) to run deep learning inference directly inside the browser using WebAssembly (WASM), providing high-speed, serverless, offline-first image analysis.

---

## 📸 Supported Classes

The model detects three primary classes essential for verifying installation quality:
1. **`kazam_box`**: The Kazam IoT EV charger / device enclosure.
2. **`multimeter`**: The digital multimeter used for electrical validation.
3. **`lcd_screen`**: The active digital screen on the multimeter showing electrical readouts.

> [!NOTE]
> The complete dataset for training and refining these classes is prepared and ready on the **Ultralytics Hub** website.

---

## ⚡ Key Features

* **On-Device Inference**: Runs the lightweight `t400v100.onnx` model (640x640 input size) in the browser's WebAssembly thread.
* **Offline Capability**: Configured as a Progressive Web Application with service worker caching (`sw.js`) for reliable operation in areas with poor internet connection.
* **Image Quality Pre-check**: Prior to running the model, the app measures image resolution, blur, brightness, contrast, shadow ratio, overexposure, and glare.
* **Logical Verdict Enforcement**:
  * Minimum resolution check (requires at least $800 \times 800$ pixels).
  * Minimum and optimal confidence threshold checks for each object class.
  * **Enclosure Check**: Spatially verifies that the detected `lcd_screen` bounding box is entirely within the detected `multimeter` bounding box.

---

## 🛠️ Development & Production

To run the application locally or build it for production, make sure you have [Node.js](https://nodejs.org) installed.

### 1. Install Dependencies
```bash
npm install
```

### 2. Run in Development Mode
Starts the Vite dev server with fast hot module replacement:
```bash
npm run dev
```

### 3. Build for Production
Compiles and optimizes the assets into the `dist` folder:
```bash
npm run build
```

### 4. Preview the Build
Launches a local server to preview the production build:
```bash
npm run preview
```

---

## 📂 Project Structure

```
PWA/
├── model/                  # Model checkpoints
│   ├── t120v30.onnx        # Variant checkpoint
│   └── t400v100.onnx       # Primary ONNX model
├── public/                 # Static assets served at root
│   ├── model/              # Public directory model files
│   ├── icon-512.png        # App icon
│   ├── manifest.json       # Web app manifest for PWA installation
│   ├── sw.js               # Service Worker for offline caching
│   └── test_sample.jpeg    # Sample photo for quick demo testing
├── src/
│   ├── main.js             # Preprocessing, ONNX inference, postprocessing, and logical verdict rules
│   └── style.css           # Vanilla CSS styles (Glassmorphism & Responsive layout)
├── index.html              # HTML markup
├── package.json            # Node dependencies and scripts
└── vite.config.js          # Vite configuration with static WASM copy plugin
```
