import * as ort from 'onnxruntime-web';
import './style.css';

const MODEL_URL = '/model/t400v100.onnx?v=20260616-model400';
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.25;
const CLASS_NAMES = ['kazam_box', 'multimeter', 'lcd_screen'];
const CLASS_COLORS = {
  kazam_box: '#2f80ff',
  multimeter: '#00e68a',
  lcd_screen: '#ff4d6a',
};

const QUALITY_LIMITS = {
  minWidth: 800,
  minHeight: 800,
  blurWarn: 120,
  blurReject: 60,
  brightnessMinWarn: 65,
  brightnessMaxWarn: 205,
  brightnessMinReject: 45,
  brightnessMaxReject: 225,
  contrastWarn: 40,
  contrastReject: 25,
  glareWarn: 0.06,
  glareReject: 0.12,
  darkWarn: 0.30,
  darkReject: 0.45,
  brightWarn: 0.22,
  brightReject: 0.35,
};

const ROI_LIMITS = {
  minConfidence: {
    kazam_box: 0.25,
    multimeter: 0.30,
    lcd_screen: 0.20,
  },
  goodConfidence: {
    kazam_box: 0.40,
    multimeter: 0.50,
    lcd_screen: 0.35,
  },
};

let session = null;
let currentImage = null;
let currentQuality = null;

const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const uploadSection = document.getElementById('uploadSection');
const canvasSection = document.getElementById('canvasSection');
const canvas = document.getElementById('canvas');
const canvasWrapper = document.querySelector('.canvas-wrapper');
const ctx = canvas.getContext('2d');
const detectBtn = document.getElementById('detectBtn');
const detectBtnLabel = detectBtn.querySelector('.btn-label');
const resetBtn = document.getElementById('resetBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsList = document.getElementById('resultsList');
const statusDot = document.querySelector('.status-dot');
const statusText = document.getElementById('statusText');
const testBtn = document.getElementById('testBtn');
const clearCacheBtn = document.getElementById('clearCacheBtn');
const themeToggle = document.getElementById('themeToggle');
const themeMeta = document.querySelector('meta[name="theme-color"]');

initTheme();

function setStatus(text, state = 'ready') {
  statusText.textContent = text;
  statusDot.className = `status-dot ${state}`;
}

async function loadModel() {
  setStatus('Loading t400v100 ONNX model...', 'working');
  showSystemPanel('loading', 'Loading detection model', [
    ['active', 'Preparing browser inference'],
    ['pending', 'Loading ROI model'],
    ['pending', 'Ready for upload'],
  ]);
  try {
    ort.env.wasm.numThreads = 1;
    const modelBytes = await fetchModelBytes(MODEL_URL);
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
    });
    console.log('Model loaded', {
      model: MODEL_URL,
      inputs: session.inputNames,
      outputs: session.outputNames,
    });
    setStatus('Model loaded. Upload a technician photo.', 'ready');
    detectBtn.disabled = !currentImage;
    showSystemPanel('ready', 'Ready for technician photo', [
      ['done', 'Browser inference ready'],
      ['done', 'ROI model loaded'],
      ['active', 'Upload or capture a photo'],
    ]);
  } catch (err) {
    console.error('Model load error:', err);
    setStatus(`Model failed to load: ${err.message}`, 'error');
    showSystemPanel('error', 'Model failed to load', [
      ['error', 'Check internet/local tunnel and refresh the app'],
      ['pending', 'Use Refresh Installed App if this was installed earlier'],
    ]);
  }
}

async function fetchModelBytes(url) {
  const response = await fetch(url, {
    cache: 'no-store',
    headers: {
      'ngrok-skip-browser-warning': 'true',
    },
  });
  if (!response.ok) {
    throw new Error(`Model download failed: HTTP ${response.status}`);
  }
  const contentType = response.headers.get('content-type') || '';
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength < 1024 * 1024) {
    throw new Error(`Model download looks invalid (${buffer.byteLength} bytes, ${contentType || 'unknown type'})`);
  }
  return new Uint8Array(buffer);
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropzone.classList.add('drag-over');
});
dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('drag-over');
});
dropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropzone.classList.remove('drag-over');
  const file = event.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});
fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) handleFile(file);
});
if (testBtn) {
  testBtn.addEventListener('click', async () => {
    setStatus('Loading test image...', 'working');
    try {
      const response = await fetch('/test_sample.jpeg');
      const blob = await response.blob();
      handleFile(new File([blob], 'test_sample.jpeg', { type: 'image/jpeg' }));
    } catch (err) {
      setStatus(`Failed to load test image: ${err.message}`, 'error');
    }
  });
}
if (clearCacheBtn) {
  clearCacheBtn.addEventListener('click', clearInstalledAppCache);
}
if (themeToggle) {
  themeToggle.addEventListener('click', toggleTheme);
}
resetBtn.addEventListener('click', () => {
  currentImage = null;
  currentQuality = null;
  fileInput.value = '';
  uploadSection.style.display = '';
  canvasSection.classList.remove('visible');
  resultsSection.classList.remove('visible');
  setStatus('Upload a new image', 'ready');
});
detectBtn.addEventListener('click', runDetection);

function handleFile(file) {
  setStatus('Reading image...', 'working');
  showSystemPanel('loading', 'Reading technician photo', [
    ['active', 'Opening image'],
    ['pending', 'Measuring quality'],
    ['pending', 'Waiting for ROI check'],
  ]);
  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      currentImage = img;
      currentQuality = analyzeImageQuality(img);
      showImage(img);
      showPrecheck(currentQuality);
      setStatus('Image loaded. Run ROI quality check.', 'ready');
    };
    img.onerror = () => setStatus('Could not load image', 'error');
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

function showImage(img) {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  uploadSection.style.display = 'none';
  canvasSection.classList.add('visible');
  resultsSection.classList.add('visible');
  detectBtn.disabled = !session;
}

function showPrecheck(quality) {
  resultsList.innerHTML = `
    <div class="result-summary neutral">
      Image loaded: ${quality.width} x ${quality.height}. Run detection to validate ROIs.
    </div>
    ${renderQualityGrid(quality)}
  `;
}

async function runDetection() {
  if (!session || !currentImage) return;
  setDetectButtonLoading(true, 'Starting...');
  canvasWrapper?.classList.add('processing');
  setStatus('Running ROI inference...', 'working');
  showSystemPanel('loading', 'Checking image quality', [
    ['done', 'Image quality measured'],
    ['active', 'Preparing 640 x 640 model input'],
    ['pending', 'Running ROI detection'],
    ['pending', 'Generating verdict'],
  ]);
  await nextPaint();

  const startedAt = performance.now();
  try {
    setDetectButtonLoading(true, 'Preparing image...');
    const input = preprocess(currentImage);
    showSystemPanel('loading', 'Running ROI model', [
      ['done', 'Image quality measured'],
      ['done', 'Model input prepared'],
      ['active', 'Detecting Kazam box, multimeter, and LCD'],
      ['pending', 'Generating verdict'],
    ]);
    setDetectButtonLoading(true, 'Running model...');
    await nextPaint();

    const feeds = { [session.inputNames[0]]: input.tensor };
    const outputMap = await session.run(feeds);
    showSystemPanel('loading', 'Generating result', [
      ['done', 'Image quality measured'],
      ['done', 'Model input prepared'],
      ['done', 'ROI detection complete'],
      ['active', 'Checking confidence and geometry'],
    ]);
    setDetectButtonLoading(true, 'Checking result...');
    await nextPaint();

    const outputTensor = outputMap[session.outputNames[0]];
    const detections = postprocess(outputTensor, input);
    const elapsedMs = Math.round(performance.now() - startedAt);
    const verdict = evaluatePhoto(currentQuality, detections);
    ctx.drawImage(currentImage, 0, 0);
    drawDetections(detections);
    showResults(detections, currentQuality, verdict, elapsedMs);
    setStatus(
      `${verdict.status.toUpperCase()} - ${detections.length} ROI detection(s), ${elapsedMs} ms`,
      verdict.status === 'approved' ? 'ready' : verdict.status === 'warning' ? 'working' : 'error'
    );
  } catch (err) {
    console.error('Inference error:', err);
    setStatus(`Inference failed: ${err.message}`, 'error');
    showSystemPanel('error', 'Inference failed', [
      ['error', err.message],
      ['pending', 'Try Refresh Installed App or upload another image'],
    ]);
  } finally {
    setDetectButtonLoading(false, 'Run ROI Check');
    canvasWrapper?.classList.remove('processing');
  }
}

function setDetectButtonLoading(isLoading, label) {
  detectBtn.disabled = isLoading;
  detectBtn.classList.toggle('loading', isLoading);
  if (detectBtnLabel) detectBtnLabel.textContent = label;
}

function nextPaint() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      setTimeout(resolve, 0);
    });
  });
}

async function clearInstalledAppCache() {
  setStatus('Refreshing installed app cache...', 'working');
  try {
    if ('serviceWorker' in navigator) {
      const registrations = await navigator.serviceWorker.getRegistrations();
      for (const registration of registrations) {
        registration.active?.postMessage({ type: 'CLEAR_APP_CACHE' });
        await registration.update();
      }
    }
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      await Promise.all(cacheNames.map((cacheName) => caches.delete(cacheName)));
    }
    window.location.reload();
  } catch (err) {
    console.error('Cache refresh failed:', err);
    setStatus(`Cache refresh failed: ${err.message}`, 'error');
  }
}

function preprocess(img) {
  const origW = img.naturalWidth;
  const origH = img.naturalHeight;
  const scale = Math.min(INPUT_SIZE / origW, INPUT_SIZE / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);
  const padX = Math.round((INPUT_SIZE - newW) / 2);
  const padY = Math.round((INPUT_SIZE - newH) / 2);
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = INPUT_SIZE;
  tmpCanvas.height = INPUT_SIZE;
  const tmpCtx = tmpCanvas.getContext('2d', { willReadFrequently: true });
  tmpCtx.fillStyle = 'rgb(114, 114, 114)';
  tmpCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  tmpCtx.drawImage(img, padX, padY, newW, newH);
  const pixels = tmpCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const area = INPUT_SIZE * INPUT_SIZE;
  const tensorData = new Float32Array(3 * area);
  for (let i = 0; i < area; i += 1) {
    tensorData[i] = pixels[i * 4] / 255;
    tensorData[area + i] = pixels[i * 4 + 1] / 255;
    tensorData[2 * area + i] = pixels[i * 4 + 2] / 255;
  }
  return {
    tensor: new ort.Tensor('float32', tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scale,
    padX,
    padY,
    origW,
    origH,
  };
}

function postprocess(outputTensor, input) {
  const { data, dims } = outputTensor;
  let numDetections;
  let cols;
  if (dims.length === 3) {
    numDetections = dims[1];
    cols = dims[2];
  } else if (dims.length === 2) {
    numDetections = dims[0];
    cols = dims[1];
  } else {
    throw new Error(`Unsupported ONNX output shape: ${dims.join('x')}`);
  }
  if (cols < 6) {
    throw new Error(`Expected NMS output columns [x1,y1,x2,y2,conf,class], got ${cols}`);
  }
  const results = [];
  for (let i = 0; i < numDetections; i += 1) {
    const offset = i * cols;
    const confidence = Number(data[offset + 4]);
    if (!Number.isFinite(confidence) || confidence < CONF_THRESHOLD) continue;
    const classId = Math.round(Number(data[offset + 5]));
    const className = CLASS_NAMES[classId] || `class_${classId}`;
    if (!CLASS_NAMES.includes(className)) continue;
    const x1 = clamp((Number(data[offset]) - input.padX) / input.scale, 0, input.origW);
    const y1 = clamp((Number(data[offset + 1]) - input.padY) / input.scale, 0, input.origH);
    const x2 = clamp((Number(data[offset + 2]) - input.padX) / input.scale, 0, input.origW);
    const y2 = clamp((Number(data[offset + 3]) - input.padY) / input.scale, 0, input.origH);
    if (x2 <= x1 || y2 <= y1) continue;
    results.push({ classId, className, confidence, x1, y1, x2, y2 });
  }
  return keepBestPerClass(results);
}

function keepBestPerClass(detections) {
  const best = new Map();
  for (const det of detections) {
    const current = best.get(det.className);
    if (!current || det.confidence > current.confidence) best.set(det.className, det);
  }
  return [...best.values()].sort((a, b) => a.classId - b.classId);
}

function analyzeImageQuality(img) {
  const sampleSize = 320;
  const scale = Math.min(sampleSize / img.naturalWidth, sampleSize / img.naturalHeight, 1);
  const width = Math.max(1, Math.round(img.naturalWidth * scale));
  const height = Math.max(1, Math.round(img.naturalHeight * scale));
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = width;
  tmpCanvas.height = height;
  const tmpCtx = tmpCanvas.getContext('2d', { willReadFrequently: true });
  tmpCtx.drawImage(img, 0, 0, width, height);
  const pixels = tmpCtx.getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height);
  let sum = 0;
  let dark = 0;
  let bright = 0;
  let glare = 0;
  for (let i = 0; i < gray.length; i += 1) {
    const r = pixels[i * 4];
    const g = pixels[i * 4 + 1];
    const b = pixels[i * 4 + 2];
    const value = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[i] = value;
    sum += value;
    if (value <= 40) dark += 1;
    if (value >= 220) bright += 1;
    if (r >= 245 && g >= 245 && b >= 245 && Math.max(r, g, b) - Math.min(r, g, b) <= 25) glare += 1;
  }
  const mean = sum / gray.length;
  let variance = 0;
  for (let i = 0; i < gray.length; i += 1) {
    const delta = gray[i] - mean;
    variance += delta * delta;
  }
  return {
    width: img.naturalWidth,
    height: img.naturalHeight,
    blurScore: laplacianVariance(gray, width, height),
    brightnessMean: mean,
    contrastScore: Math.sqrt(variance / gray.length),
    darkPixelRatio: dark / gray.length,
    brightPixelRatio: bright / gray.length,
    glareRatio: glare / gray.length,
  };
}

function laplacianVariance(gray, width, height) {
  if (width < 3 || height < 3) return 0;
  const values = [];
  let sum = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      const lap = gray[index - width] + gray[index - 1] - 4 * gray[index] + gray[index + 1] + gray[index + width];
      values.push(lap);
      sum += lap;
    }
  }
  const mean = sum / values.length;
  let variance = 0;
  for (const value of values) {
    const delta = value - mean;
    variance += delta * delta;
  }
  return variance / values.length;
}

function evaluatePhoto(quality, detections) {
  const reject = [];
  const warn = [];
  const byClass = Object.fromEntries(detections.map((det) => [det.className, det]));
  if (quality.width < QUALITY_LIMITS.minWidth || quality.height < QUALITY_LIMITS.minHeight) reject.push('Image resolution is too low. Retake with a clearer photo.');
  thresholdCheck(quality.blurScore, QUALITY_LIMITS.blurReject, QUALITY_LIMITS.blurWarn, 'low', reject, warn, 'Image is blurry. Hold the phone steady and retake.');
  thresholdCheck(quality.brightnessMean, QUALITY_LIMITS.brightnessMinReject, QUALITY_LIMITS.brightnessMinWarn, 'low', reject, warn, 'Image is too dark. Retake with better lighting.');
  thresholdCheck(quality.brightnessMean, QUALITY_LIMITS.brightnessMaxReject, QUALITY_LIMITS.brightnessMaxWarn, 'high', reject, warn, 'Image is too bright. Avoid harsh direct light.');
  thresholdCheck(quality.contrastScore, QUALITY_LIMITS.contrastReject, QUALITY_LIMITS.contrastWarn, 'low', reject, warn, 'Image contrast is low. Retake in clearer lighting.');
  thresholdCheck(quality.darkPixelRatio, QUALITY_LIMITS.darkReject, QUALITY_LIMITS.darkWarn, 'high', reject, warn, 'Image has too much shadow.');
  thresholdCheck(quality.brightPixelRatio, QUALITY_LIMITS.brightReject, QUALITY_LIMITS.brightWarn, 'high', reject, warn, 'Image is overexposed.');
  thresholdCheck(quality.glareRatio, QUALITY_LIMITS.glareReject, QUALITY_LIMITS.glareWarn, 'high', reject, warn, 'Image has glare. Change the camera angle.');
  for (const className of CLASS_NAMES) {
    const det = byClass[className];
    if (!det) {
      reject.push(`${labelFor(className)} is not detected clearly.`);
    } else if (det.confidence < ROI_LIMITS.minConfidence[className]) {
      reject.push(`${labelFor(className)} confidence is too low.`);
    } else if (det.confidence < ROI_LIMITS.goodConfidence[className]) {
      warn.push(`${labelFor(className)} confidence is borderline.`);
    }
  }
  if (byClass.lcd_screen && byClass.multimeter && !isInside(byClass.lcd_screen, byClass.multimeter)) reject.push('LCD screen is not inside the detected multimeter box.');
  const status = reject.length ? 'rejected' : warn.length ? 'warning' : 'approved';
  const messages = status === 'approved' ? ['Approved. Kazam box, multimeter, and LCD screen are clearly visible.'] : [...new Set([...(reject.length ? reject : warn)])].slice(0, 3);
  return { status, reject, warn, messages };
}

function thresholdCheck(value, rejectLimit, warnLimit, direction, reject, warn, message) {
  if (direction === 'low') {
    if (value < rejectLimit) reject.push(message);
    else if (value < warnLimit) warn.push(message);
  } else if (value > rejectLimit) {
    reject.push(message);
  } else if (value > warnLimit) {
    warn.push(message);
  }
}

function drawDetections(detections) {
  for (const det of detections) {
    const color = CLASS_COLORS[det.className] || '#ffffff';
    const { x1, y1, x2, y2, confidence, className } = det;
    const width = x2 - x1;
    const height = y2 - y1;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, Math.min(5, canvas.width / 260));
    ctx.strokeRect(x1, y1, width, height);
    const label = `${labelFor(className)} ${(confidence * 100).toFixed(1)}%`;
    const fontSize = Math.max(13, Math.min(18, canvas.width / 44));
    ctx.font = `bold ${fontSize}px Inter, sans-serif`;
    const labelW = ctx.measureText(label).width + 14;
    const labelH = fontSize + 10;
    const labelY = y1 > labelH + 4 ? y1 - labelH - 4 : y1 + 4;
    ctx.fillStyle = color;
    roundRect(ctx, x1, labelY, labelW, labelH, 4);
    ctx.fill();
    ctx.fillStyle = '#0b0e14';
    ctx.fillText(label, x1 + 7, labelY + fontSize + 2);
  }
}

function showResults(detections, quality, verdict, elapsed) {
  resultsSection.classList.add('visible');
  const roiCards = detections.length ? detections.map(renderDetectionCard).join('') : '<div class="no-results">No ROI detections above threshold.</div>';
  resultsList.innerHTML = `
    <div class="verdict-card ${verdict.status}">
      <div>
        <span class="verdict-label">${verdict.status}</span>
        <p>${verdict.messages.join(' ')}</p>
      </div>
      <span class="runtime">${elapsed} ms</span>
    </div>
    <div class="result-summary">Detected ${detections.length} ROI class${detections.length === 1 ? '' : 'es'}</div>
    ${roiCards}
    ${renderQualityGrid(quality)}
  `;
}

function showSystemPanel(state, title, steps) {
  resultsSection.classList.add('visible');
  resultsList.innerHTML = `
    <div class="process-card ${state}">
      <div class="process-head">
        <span class="process-spinner" aria-hidden="true"></span>
        <div>
          <strong>${title}</strong>
          <p>${state === 'error' ? 'Action needed before continuing.' : 'Keep this screen open while processing.'}</p>
        </div>
      </div>
      <div class="process-steps">
        ${steps.map(([stepState, text]) => `
          <div class="process-step ${stepState}">
            <span></span>
            <p>${text}</p>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

function renderDetectionCard(det) {
  const color = CLASS_COLORS[det.className] || '#ffffff';
  return `
    <div class="result-card" style="border-left-color:${color}">
      <span class="result-class">${labelFor(det.className)}</span>
      <span class="result-conf">${(det.confidence * 100).toFixed(1)}%</span>
      <span class="result-box">[${det.x1.toFixed(0)}, ${det.y1.toFixed(0)}, ${det.x2.toFixed(0)}, ${det.y2.toFixed(0)}]</span>
    </div>
  `;
}

function renderQualityGrid(quality) {
  const rows = [
    ['Resolution', `${quality.width} x ${quality.height}`],
    ['Blur score', quality.blurScore.toFixed(1)],
    ['Brightness', quality.brightnessMean.toFixed(1)],
    ['Contrast', quality.contrastScore.toFixed(1)],
    ['Dark pixels', `${(quality.darkPixelRatio * 100).toFixed(1)}%`],
    ['Bright pixels', `${(quality.brightPixelRatio * 100).toFixed(1)}%`],
    ['Glare', `${(quality.glareRatio * 100).toFixed(1)}%`],
  ];
  return `
    <div class="quality-grid">
      ${rows.map(([name, value]) => `
        <div class="quality-item">
          <span>${name}</span>
          <strong>${value}</strong>
        </div>
      `).join('')}
    </div>
  `;
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + radius);
  context.lineTo(x + width, y + height - radius);
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  context.lineTo(x + radius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}

function isInside(inner, outer) {
  return inner.x1 >= outer.x1 && inner.y1 >= outer.y1 && inner.x2 <= outer.x2 && inner.y2 <= outer.y2;
}

function labelFor(className) {
  return className.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function initTheme() {
  const savedTheme = localStorage.getItem('kazam-roi-theme');
  const prefersLight = window.matchMedia?.('(prefers-color-scheme: light)').matches;
  applyTheme(savedTheme || (prefersLight ? 'light' : 'dark'));
}

function toggleTheme() {
  const nextTheme = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  localStorage.setItem('kazam-roi-theme', nextTheme);
  applyTheme(nextTheme);
}

function applyTheme(theme) {
  const normalized = theme === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = normalized;
  if (themeMeta) {
    themeMeta.setAttribute('content', normalized === 'light' ? '#f6f8fb' : '#0b0e14');
  }
  if (themeToggle) {
    const isLight = normalized === 'light';
    themeToggle.setAttribute('aria-label', isLight ? 'Switch to dark theme' : 'Switch to light theme');
    themeToggle.querySelector('.theme-toggle-icon').textContent = isLight ? 'Moon' : 'Sun';
    themeToggle.querySelector('.theme-toggle-text').textContent = isLight ? 'Dark' : 'Light';
  }
}

loadModel();

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((registration) => {
        console.log('Service worker registered:', registration.scope);
        registration.update();
      })
      .catch((err) => console.error('Service worker registration failed:', err));
  });

  navigator.serviceWorker.addEventListener('message', (event) => {
    if (event.data?.type === 'APP_CACHE_CLEARED') {
      window.location.reload();
    }
  });
}
