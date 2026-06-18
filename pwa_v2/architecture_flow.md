# Kazam Site Check PWA v2 Architecture Flow

This document explains the actual runtime flow implemented in `src/main.js`. It is written from the current code behavior, including the exact thresholds, image transformations, model routing, and validation rules used by the app.

## 1. High-Level App Purpose

The app is a browser-based installation photo verification PWA. A technician uploads or captures required site photos. The app checks photo quality locally, runs ONNX object detection locally through WebAssembly, applies section-specific rules, and marks each required photo section as complete, warning, error, partial, or empty.

No backend inference server is used. Both ONNX models run inside the browser through `onnxruntime-web`.

## 2. Main Technologies

| Area | Implementation |
| --- | --- |
| UI | HTML, CSS, vanilla JavaScript |
| Build system | Vite |
| Runtime inference | `onnxruntime-web` |
| ONNX backend | WebAssembly |
| Image handling | Browser `FileReader`, `Image`, and `CanvasRenderingContext2D` |
| PWA support | Web app manifest and service worker |
| Model input tensor | Float32 NCHW tensor `[1, 3, 640, 640]` |

## 3. App Boot Flow

On `DOMContentLoaded`, the app calls `initApp()`.

The initialization sequence is:

1. Render the progress bar segments.
2. Render all photo cards.
3. Initialize the saved light/dark theme.
4. Configure the developer menu.
5. Log device info when available:
   - `navigator.deviceMemory`
   - `navigator.hardwareConcurrency`
   - PWA display mode status
6. Configure ONNX Runtime Web:
   - `ort.env.wasm.wasmPaths = assetUrl('wasm/')`
   - `ort.env.wasm.numThreads = 1`
7. Load both ONNX sessions:
   - `multimeterSession` from `public/model/multimeter_t400v100.onnx`
   - `potholeSession` from `public/model/pothole_t417v100.onnx`
8. Mark `state.modelLoaded = true`.
9. Hide the loading overlay and show the form.

The loading progress bar is visual feedback only. It increments randomly up to 90 percent while the models load, then is set to 100 percent after both sessions are created.

## 4. Photo Sections

The app defines six required user-facing sections.

| Section ID | Title | AI enabled | Required detections | Extra logic |
| --- | --- | --- | --- | --- |
| `full_setup` | Full Setup Photo | Yes | `kazam_box`, `multimeter`, `lcd_screen` | LCD must be enclosed inside multimeter |
| `multimeter_reading` | Multimeter Reading | Yes | `multimeter`, `lcd_screen` | LCD must be enclosed inside multimeter |
| `kazam_box` | Kazam Box | Yes | `kazam_box` | No enclosure check |
| `earth_pit` | Earth Pit | Yes | `earthpit_open`, `earthpit_cover` | Two-slot open/covered workflow |
| `vehicle_charging` | Vehicle Charging | No | None | Quality check only |
| `electric_meter` | Electric Meter | No | None | Quality check only |

The progress counter counts a section as complete only when status is `done` or `warning`. It enables the submit button only when all six sections are complete or warning.

## 5. User Upload Flow

Each upload card offers two hidden file inputs:

| Input | Browser behavior |
| --- | --- |
| Camera | `<input type="file" accept="image/*" capture="environment">` |
| Gallery | `<input type="file" accept="image/*">` |

For normal sections, the selected file goes to `handleFileUpload(event, sectionId)`.

For earth pit, each sub-slot has its own handler:

- `handleEarthPitUpload(event, 'open_pit')`
- `handleEarthPitUpload(event, 'covered_pit')`

The file reading flow is:

1. Read selected image file with `FileReader.readAsDataURL(file)`.
2. Create a browser `Image`.
3. Assign the Data URL to `img.src`.
4. Wait for `img.onload`.
5. Run quality checks.
6. If quality passes, route to AI inference if the section is AI-enabled.

## 6. Photo Quality Check

Quality is calculated in `calculateQualityMetrics(img)`.

The image is first downsampled to a fixed quality-analysis canvas:

```text
quality sample size = 320 x 320
```

This downsampled image is used only for quality scoring. It is not the model input.

### 6.1 Brightness

For each pixel, luma is calculated using the standard RGB weighted formula:

```text
luma = 0.299 * R + 0.587 * G + 0.114 * B
```

Brightness is the average luma across all pixels:

```text
brightness = sum(luma) / totalPixels
```

The current pass range is:

```text
brightness > 40
brightness < 220
```

If brightness is not in this range, the upload fails with a dark/light quality error message.

### 6.2 Glare Ratio

Glare is measured as the ratio of very bright pixels.

A pixel is counted as bright/glare if:

```text
luma > 240
```

The ratio is:

```text
glareRatio = brightPixels / totalPixels
```

The current pass threshold is:

```text
glareRatio < 0.05
```

So fewer than 5 percent of the 320 x 320 sampled pixels may be above luma 240.

### 6.3 Blur Score

Blur is measured by `laplacianVariance(imgData)`.

The code first converts the 320 x 320 sample to grayscale using the same luma formula:

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

Then it applies a 4-neighbor Laplacian kernel to internal pixels:

```text
laplacian = 4 * center - left - right - top - bottom
```

The blur metric is the variance of the Laplacian values:

```text
blur = variance(laplacian)
```

The current pass threshold is:

```text
blur > 100
```

Higher values mean sharper local edges. Values at or below 100 are treated as too blurry.

### 6.4 Current Quality Gates

The upload passes quality only if all three checks pass:

```text
blur > 100
brightness > 40
brightness < 220
glareRatio < 0.05
```

If any check fails, inference is not run for that image.

### 6.5 What Is Not Currently Enforced

The current `pwa_v2/src/main.js` does not separately compute or gate these metrics:

- contrast score
- shadow ratio
- overexposure ratio separate from glare
- minimum image resolution
- maximum image file size
- JPEG compression score

The UI and documentation should not claim those are active gates unless code is added for them. The active quality gates in the current code are brightness, glare ratio, and Laplacian blur.

## 7. Quality Failure Behavior

For normal sections:

1. The section status becomes `error`.
2. The original image preview is kept.
3. The measured metrics are stored in state.
4. The user sees one of these messages:
   - too blurry
   - too dark or too bright
   - too much glare

For earth pit:

1. The failed slot becomes `error`.
2. `updateEarthPitSectionStatus()` updates the overall earth pit status.
3. Inference is skipped for that slot.

## 8. AI Model Routing

After quality passes, AI-enabled sections call:

```text
runDetection(img, sectionId)
```

The model session is selected by section:

```text
sectionId === 'earth_pit' -> potholeSession
otherwise -> multimeterSession
```

So:

| Section | Model |
| --- | --- |
| `full_setup` | `multimeter_t400v100.onnx` |
| `multimeter_reading` | `multimeter_t400v100.onnx` |
| `kazam_box` | `multimeter_t400v100.onnx` |
| `earth_pit` | `pothole_t417v100.onnx` |

Non-AI sections do not run ONNX inference. If their quality check passes, they are marked `done`.

## 9. Image Preprocessing For ONNX

Preprocessing is implemented in `preprocess(img)`.

The target model input size is fixed:

```text
targetSize = 640
```

### 9.1 Stage A: Original Image

The original uploaded camera/gallery image is loaded into an `Image` object. Its original `naturalWidth` and `naturalHeight` are used for scaling and final coordinate restoration.

No resize has happened yet.

### 9.2 Stage B: Quality Canvas

The original image is drawn to a separate quality-analysis canvas:

```text
320 x 320
```

This stage is used for brightness, glare, and blur only.

### 9.3 Stage C: Proportional Resize

The model preprocessing computes a scale that fits the original image into a 640 x 640 square without cropping:

```text
scale = min(640 / originalWidth, 640 / originalHeight)
newW = round(originalWidth * scale)
newH = round(originalHeight * scale)
```

The app creates a debug resize snapshot of:

```text
newW x newH
```

This image has no padding yet.

### 9.4 Stage D: Letterbox Padding

The model canvas is fixed:

```text
640 x 640
```

Padding is calculated as:

```text
padX = (640 - newW) / 2
padY = (640 - newH) / 2
```

The canvas is first filled with:

```text
rgb(114, 114, 114)
```

Then the resized image is drawn at:

```text
x = padX
y = padY
width = newW
height = newH
```

This is standard letterbox-style preprocessing: the full image is preserved, aspect ratio is maintained, and empty areas are gray padded.

### 9.5 Stage E: Float32 Tensor

The code reads the 640 x 640 canvas with:

```text
ctx.getImageData(0, 0, 640, 640)
```

It converts RGBA canvas pixels to normalized planar RGB:

```text
R plane: data[i] / 255.0
G plane: data[i + 1] / 255.0
B plane: data[i + 2] / 255.0
```

The final tensor layout is:

```text
NCHW float32
[1, 3, 640, 640]
```

The tensor is created as:

```text
new ort.Tensor('float32', float32Data, [1, 3, 640, 640])
```

## 10. Inference Execution

The feed key is:

```text
images
```

The app runs:

```text
activeSession.run({ images: tensor })
```

Then it reads the first output by:

```text
results[activeSession.outputNames[0]]
```

The output is passed to:

```text
postprocess(output, ratios, padParams, originalWidth, originalHeight, sectionId)
```

The runtime is measured with:

```text
performance.now()
```

## 11. Postprocessing Overview

Postprocessing does these jobs:

1. Read output dimensions.
2. Determine whether output is transposed.
3. Parse boxes according to model and output channel count.
4. Apply confidence threshold.
5. Convert boxes to `xyxy` if needed.
6. Remove letterbox padding.
7. Scale coordinates back to original image size.
8. Clamp coordinates within original image bounds.
9. Map class ID to class name.
10. Return detection objects:

```text
{
  box: [x1, y1, x2, y2],
  conf,
  classId,
  className
}
```

## 12. Confidence Thresholds

The active confidence thresholds are:

| Section/model type | Threshold |
| --- | --- |
| Earth pit model | `0.25` |
| Multimeter/Kazam/LCD model | `0.40` |

In code:

```text
CONF_THRESHOLD = sectionId === 'earth_pit' ? 0.25 : 0.40
```

## 13. Model Output Formats

### 13.1 Multimeter/Kazam/LCD Model

Runtime file:

```text
public/model/multimeter_t400v100.onnx
```

Used for:

- `full_setup`
- `multimeter_reading`
- `kazam_box`

The current parser treats 7-channel non-transposed output as OBB end-to-end output:

```text
[x_center, y_center, width, height, confidence, class_id, angle]
```

The app currently converts this to an axis-aligned box for UI drawing:

```text
x1 = x_center - width / 2
y1 = y_center - height / 2
x2 = x_center + width / 2
y2 = y_center + height / 2
```

The angle is not used for drawing rotated boxes in the current UI.

Class mapping in postprocess for this model:

```text
0 -> lcd_screen
1 -> multimeter
2 -> kazam_box
```

### 13.2 Earth Pit Model

Runtime file:

```text
public/model/pothole_t417v100.onnx
```

Used for:

- `earth_pit`

The current model output is parsed as regular detection output:

```text
[x1, y1, x2, y2, confidence, class_id]
```

Expected output dimensions:

```text
[1, 300, 6]
```

Class mapping:

```text
0 -> earthpit_open
1 -> earthpit_cover
```

The code also keeps a fallback parser for a possible 7-channel earth pit output:

```text
[batch_idx, x1, y1, x2, y2, class_id, confidence]
```

## 14. Coordinate Restoration

Model outputs are in the 640 x 640 letterboxed image coordinate system.

The app restores coordinates to original image space using:

```text
x1 = (x1 - padX) / scale
y1 = (y1 - padY) / scale
x2 = (x2 - padX) / scale
y2 = (y2 - padY) / scale
```

Then it clamps every coordinate:

```text
x = max(0, min(x, originalWidth))
y = max(0, min(y, originalHeight))
```

This prevents boxes from rendering outside the original image.

## 15. Best Detection Per Class

After raw detections return, the app calls:

```text
keepBestPerClass(detections)
```

This keeps only the highest-confidence detection for each class name.

Example:

If the model returns three `multimeter` boxes, only the one with the highest `conf` is kept.

This final list is used for rule evaluation and UI rendering.

## 16. Section-Level Validation Logic

Validation is implemented in `evaluateSectionLogic(config, detections)` for normal AI sections.

### 16.1 Required Class Check

Each AI-enabled normal section has `requiredClasses`.

The app creates:

```text
detectedClasses = detections.map(d => d.className)
missing = requiredClasses not present in detectedClasses
```

If any required class is missing:

```text
status = error
```

The verdict tells the user which expected objects were not found.

### 16.2 LCD Inside Multimeter Check

For sections with `checkEnclosure: true`, the app checks whether the LCD box is fully inside the multimeter box.

The exact condition is:

```text
lcd.x1 >= multimeter.x1
lcd.y1 >= multimeter.y1
lcd.x2 <= multimeter.x2
lcd.y2 <= multimeter.y2
```

If this fails:

```text
status = warning
```

The warning still counts as completed in the progress bar, but the user sees that the LCD may not be on the multimeter.

### 16.3 Success

If all required classes are found and enclosure checks pass:

```text
status = done
```

## 17. Earth Pit Workflow

The earth pit section is different from normal sections. It has two internal slots:

- `open_pit`
- `covered_pit`

Each slot stores:

```text
status
imgDataUrl
metrics
detections
runtime
```

### 17.1 Earth Pit Quality

Each earth pit upload uses the same quality thresholds:

```text
blur > 100
brightness > 40
brightness < 220
glareRatio < 0.05
```

If the upload fails quality, only the target earth pit slot becomes `error`.

### 17.2 Earth Pit Inference

Earth pit uploads always run:

```text
runDetection(img, 'earth_pit')
```

That routes to:

```text
potholeSession
```

After inference:

1. `keepBestPerClass()` keeps one `earthpit_open` and/or one `earthpit_cover`.
2. The app reads `detectedClasses`.
3. Slot assignment is based on detected class, not only on which upload button the user clicked.

### 17.3 Single-Image Shortcut

If one image contains both classes:

```text
earthpit_open
earthpit_cover
```

Then the same uploaded image populates both slots:

- `open_pit.status = done`
- `covered_pit.status = done`

Each slot keeps only its matching class detection.

### 17.4 Smart Slot Assignment

If the user uploads an image into the wrong slot, the app reassigns it based on detected class.

Examples:

- User uploads to `covered_pit`, but model detects `earthpit_open`: the image is stored in `open_pit`.
- User uploads to `open_pit`, but model detects `earthpit_cover`: the image is stored in `covered_pit`.

If the opposite slot contained the same image or an incompatible class, that opposite slot is reset to `empty`.

### 17.5 Earth Pit Status Rules

`updateEarthPitSectionStatus()` sets the overall earth pit section status:

| Slot state | Overall status | Verdict |
| --- | --- | --- |
| Either slot processing | `processing` | Processing uploads |
| Open done and covered done | `done` | Both open and covered earth pits verified |
| Open done and covered empty | `partial` | Covered earth pit still required |
| Covered done and open empty | `partial` | Open earth pit still required |
| Either slot error | `error` | Failed slot verdict |
| Otherwise | `empty` | No verdict |

The earth pit section counts toward final submission only when overall status is `done` or `warning`.

## 18. Non-AI Section Behavior

The two non-AI sections are:

- `vehicle_charging`
- `electric_meter`

For these sections:

1. The image is still read and loaded.
2. The same quality checks are applied.
3. If quality passes, the section is immediately marked:

```text
status = done
verdict = Photo quality check passed.
```

No ONNX model is run for these sections.

## 19. UI Rendering Of Boxes

The app draws detection boxes using absolutely positioned DOM elements, not canvas overlays.

For each detection:

```text
left = x1 / imageNaturalWidth * 100%
top = y1 / imageNaturalHeight * 100%
width = (x2 - x1) / imageNaturalWidth * 100%
height = (y2 - y1) / imageNaturalHeight * 100%
```

Colors:

| Class | Color |
| --- | --- |
| `kazam_box` | Blue `#2f80ff` |
| `multimeter` | Green `#00e68a` |
| `lcd_screen` | Pink `#ff4dff` |
| `earthpit_open` | Orange `#ff9900` |
| `earthpit_cover` | Purple `#9933ff` |

In developer mode, labels include confidence percentage. Outside developer mode, labels show only the formatted class name.

## 20. Developer Mode

Developer mode can show:

- system logs
- exact quality values
- top 20 model predictions before final best-per-class filtering
- preprocessing debug gallery

The preprocessing debug gallery shows:

| Stage | Meaning |
| --- | --- |
| Stage A | Original uploaded image |
| Stage B | 320 x 320 quality-check canvas |
| Stage C | Proportional resize before padding |
| Stage D | 640 x 640 letterboxed image with RGB 114 padding |
| Stage E | Image reconstructed from the final Float32 tensor |

The app also logs the first five raw earth pit model rows in developer mode through `postprocess()` logging.

## 21. Service Worker And Offline Cache

The service worker caches the app shell, models, WASM runtime files, icon, manifest, and sample images.

It is scope-aware:

```text
BASE_PATH = self.registration.scope pathname
```

It avoids serving HTML fallbacks for JavaScript and CSS asset requests. This prevents browsers from receiving `text/html` where a module script is expected.

Cached important assets include:

- `index.html`
- `manifest.json`
- `icon-512.png`
- both ONNX models
- selected ONNX Runtime WASM files
- the three manual test images

## 22. Current Benchmarks And Gates Summary

| Area | Value |
| --- | --- |
| Quality canvas | `320 x 320` |
| Model input | `640 x 640` |
| Padding color | `rgb(114, 114, 114)` |
| Tensor dtype | `float32` |
| Tensor layout | `[1, 3, 640, 640]` NCHW |
| Pixel normalization | `/ 255.0` |
| Brightness pass | `brightness > 40 && brightness < 220` |
| Glare pixel definition | `luma > 240` |
| Glare pass | `glareRatio < 0.05` |
| Blur method | Laplacian variance |
| Blur pass | `blur > 100` |
| Earth pit confidence threshold | `0.25` |
| Multimeter/Kazam/LCD confidence threshold | `0.40` |
| ONNX Runtime backend | WASM |
| ONNX Runtime threads | `1` |
| Submit enabled when | all 6 sections are `done` or `warning` |

## 23. Important Implementation Notes

- All inference happens client-side.
- Image files are not uploaded to a backend in this app.
- The app currently uses axis-aligned boxes for rendering, even for the OBB model output.
- The OBB angle output is parsed but not used for rotated drawing.
- Earth pit output is regular detection `xyxy`, not OBB.
- Quality failure stops inference immediately.
- Warnings count as complete in the progress bar.
- Developer force-accept sets sections to warning and is intended only for testing.
- The current code does not enforce contrast, shadow ratio, separate overexposure ratio, or minimum resolution.
