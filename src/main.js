import * as ort from 'onnxruntime-web';

// ─── Configuration & State ─────────────────────────────────────────

const MODEL_URL = '/model/t400v100.onnx';
let session = null;

const CLASSES = ['kazam_box', 'multimeter', 'lcd_screen'];
const COLORS = {
  'kazam_box': '#2f80ff', // Blue
  'multimeter': '#00e68a', // Green
  'lcd_screen': '#ff4dff'  // Pink
};

// Section configurations
const PHOTO_SECTIONS = [
  {
    id: 'full_setup',
    emoji: '📸',
    title: 'Full Setup Photo',
    instruction: 'Must show Kazam box, multimeter, and LCD screen together',
    aiEnabled: true,
    requiredClasses: ['kazam_box', 'multimeter', 'lcd_screen'],
    checkEnclosure: true,
  },
  {
    id: 'multimeter_reading',
    emoji: '🔌',
    title: 'Multimeter Reading',
    instruction: 'Close-up of the multimeter showing the LCD digits clearly',
    aiEnabled: true,
    requiredClasses: ['multimeter', 'lcd_screen'],
    checkEnclosure: true,
  },
  {
    id: 'kazam_box',
    emoji: '📦',
    title: 'Kazam Box',
    instruction: 'Full view of the installed Kazam EV charger box',
    aiEnabled: true,
    requiredClasses: ['kazam_box'],
    checkEnclosure: false,
  },
  {
    id: 'earth_pit',
    emoji: '🌍',
    title: 'Earth Pit',
    instruction: 'Clear photo of the earthing pit installation',
    aiEnabled: false,
    requiredClasses: [],
  },
  {
    id: 'vehicle_charging',
    emoji: '🚗',
    title: 'Vehicle Charging',
    instruction: 'Photo showing the EV connected to the charger',
    aiEnabled: false,
    requiredClasses: [],
  },
  {
    id: 'electric_meter',
    emoji: '⚡',
    title: 'Electric Meter',
    instruction: 'Clear photo of the utility electric meter reading',
    aiEnabled: false,
    requiredClasses: [],
  }
];

// App State
const state = {
  modelLoaded: false,
  expandedSection: null, // ID of currently expanded card
  devMode: false,
  sections: {}, // e.g. { 'full_setup': { status: 'empty', imgDataUrl: '', metrics: {}, detections: [], verdict: null, processText: '' } }
  logs: []
};

// Initialize state
PHOTO_SECTIONS.forEach(s => {
  state.sections[s.id] = { status: 'empty' };
});

function logToDev(msg) {
  const ts = new Date().toISOString().split('T')[1].replace('Z', '');
  state.logs.push(`[${ts}] ${msg}`);
  if (dom.devStatsContent) {
    dom.devStatsContent.innerText = state.logs.join('\n');
    dom.devStatsContent.scrollTop = dom.devStatsContent.scrollHeight;
  }
}

// ─── DOM Elements ──────────────────────────────────────────────────

const dom = {
  loadingOverlay: document.getElementById('loadingOverlay'),
  modelStep: document.getElementById('modelStep'),
  readyStep: document.getElementById('readyStep'),
  modelDownloadProgress: document.getElementById('modelDownloadProgress'),
  modelDownloadText: document.getElementById('modelDownloadText'),
  mainForm: document.getElementById('mainForm'),
  cardsContainer: document.getElementById('cardsContainer'),
  progressSegments: document.getElementById('progressSegments'),
  progressText: document.getElementById('progressText'),
  submitBtn: document.getElementById('submitBtn'),
  hiddenCanvas: document.getElementById('hiddenCanvas'),
  statusDot: document.getElementById('statusDot'),
  statusText: document.getElementById('statusText'),
  themeToggle: document.getElementById('themeToggle'),
  themeIcon: document.getElementById('themeIcon'),
  themeText: document.getElementById('themeText'),
  versionFooter: document.getElementById('versionFooter'),
  devMenu: document.getElementById('devMenu'),
  testBtn: document.getElementById('testBtn'),
  clearCacheBtn: document.getElementById('clearCacheBtn'),
  closeDevMenuBtn: document.getElementById('closeDevMenuBtn'),
  devModeToggleBtnHeader: document.getElementById('devModeToggleBtn'),
  devStatsCard: document.getElementById('devStatsCard'),
  devStatsContent: document.getElementById('devStatsContent'),
  clearLogsBtn: document.getElementById('clearLogsBtn')
};

// ─── Initialization & Loading ──────────────────────────────────────

async function initApp() {
  renderProgressSegments();
  renderCards();
  initTheme();
  setupDevMenu();
  
  if (dom.clearLogsBtn) {
    dom.clearLogsBtn.addEventListener('click', () => {
      state.logs = [];
      if (dom.devStatsContent) dom.devStatsContent.innerText = '';
      logToDev('Logs cleared.');
    });
  }

  const mem = navigator.deviceMemory ? `${navigator.deviceMemory} GB` : 'Unknown';
  const cores = navigator.hardwareConcurrency || 'Unknown';
  logToDev(`System Init: RAM=${mem}, CPU Cores=${cores}, PWA=${window.matchMedia('(display-mode: standalone)').matches ? 'Yes' : 'No'}`);

  // Fake download progress for visual feedback
  let p = 0;
  const progressInterval = setInterval(() => {
    p += Math.random() * 15;
    if (p > 90) p = 90;
    dom.modelDownloadProgress.style.width = `${p}%`;
  }, 300);

  try {
    ort.env.wasm.numThreads = 1;
    
    // Create session
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    
    clearInterval(progressInterval);
    dom.modelDownloadProgress.style.width = `100%`;
    dom.modelDownloadText.innerText = 'Model Loaded Successfully';
    
    state.modelLoaded = true;
    
    // Update Loading Steps
    dom.modelStep.classList.remove('active');
    dom.modelStep.classList.add('done');
    dom.readyStep.classList.add('done');
    
    // Update Footer Status
    dom.statusDot.className = 'status-dot ready';
    dom.statusText.innerText = 'AI Model Ready';

    // Hide overlay, show form
    setTimeout(() => {
      dom.loadingOverlay.style.opacity = '0';
      setTimeout(() => {
        dom.loadingOverlay.style.display = 'none';
        dom.mainForm.style.display = 'block';
      }, 400);
    }, 800);

  } catch (error) {
    clearInterval(progressInterval);
    console.error('Failed to load ONNX model', error);
    dom.modelStep.classList.remove('active');
    dom.modelStep.classList.add('error');
    dom.modelStep.innerHTML = `<span class="step-dot"></span> Failed to load AI model`;
    dom.statusDot.className = 'status-dot error';
    dom.statusText.innerText = 'Model Error';
    alert('Critical Error: Failed to load AI model. Please ensure you have a connection and try again.');
  }
}

// ─── UI Rendering ──────────────────────────────────────────────────

function renderProgressSegments() {
  dom.progressSegments.innerHTML = '';
  let completedCount = 0;
  
  PHOTO_SECTIONS.forEach(sec => {
    const s = state.sections[sec.id];
    const seg = document.createElement('div');
    seg.className = 'progress-segment';
    seg.title = sec.title; // Add tooltip
    
    if (s.status === 'done') {
      seg.classList.add('done');
      completedCount++;
    } else if (s.status === 'warning') {
      seg.classList.add('warning');
      completedCount++; // Counts as done but with warning
    } else if (s.status === 'error') {
      seg.classList.add('error');
    }
    
    // Add click handler to jump to this card
    seg.onclick = () => {
      state.expandedSection = sec.id;
      renderCards();
      setTimeout(() => {
        const el = document.getElementById(`card-${sec.id}`);
        if (el) {
          // scroll to the card, accounting for the sticky progress bar height (~60px)
          const y = el.getBoundingClientRect().top + window.scrollY - 70;
          window.scrollTo({ top: y, behavior: 'smooth' });
        }
      }, 50);
    };
    
    dom.progressSegments.appendChild(seg);
  });
  
  dom.progressText.innerText = `${completedCount} of ${PHOTO_SECTIONS.length}`;
  dom.submitBtn.disabled = completedCount !== PHOTO_SECTIONS.length;
}

function renderCards() {
  try {
    dom.cardsContainer.innerHTML = '';
    
    PHOTO_SECTIONS.forEach(sec => {
    const s = state.sections[sec.id];
    const isExpanded = state.expandedSection === sec.id;
    
    let statusIcon = '';
    if (s.status === 'done') statusIcon = '✅';
    else if (s.status === 'warning') statusIcon = '⚠️';
    else if (s.status === 'error') statusIcon = '❌';

    // Card Container
    const card = document.createElement('div');
    card.className = `photo-card ${isExpanded ? 'expanded' : ''} ${s.status}`;
    card.id = `card-${sec.id}`;
    
    // Header
    const header = document.createElement('div');
    header.className = 'card-header';
    header.onclick = () => toggleCard(sec.id);
    
    let thumbnailHtml = '';
    if (s.imgDataUrl) {
      thumbnailHtml = `<img src="${s.imgDataUrl}" class="card-thumbnail" alt="thumbnail" />`;
    }

    header.innerHTML = `
      <div class="card-emoji">${sec.emoji}</div>
      <div class="card-title-group">
        <h3 class="card-title">${sec.title}</h3>
        <div class="card-subtitle">${sec.instruction}</div>
      </div>
      ${thumbnailHtml}
      <div class="card-status-icon">${statusIcon}</div>
    `;
    card.appendChild(header);

    // Body (Upload / Result area)
    const body = document.createElement('div');
    body.className = 'card-body';
    
    if (s.status === 'processing') {
      body.innerHTML = `
        <div class="upload-area" style="cursor: default; background: transparent;">
          <div class="upload-area-icon">⏳</div>
          <div class="upload-area-text">${s.processText || 'Processing...'}</div>
          <div class="upload-area-hint">Please wait</div>
        </div>
      `;
    } else if (s.status === 'empty') {
      body.innerHTML = `
        <input type="file" id="file-cam-${sec.id}" accept="image/*" capture="environment" hidden />
        <input type="file" id="file-gal-${sec.id}" accept="image/*" hidden />
        <div class="upload-actions-grid">
          <div class="upload-area" onclick="document.getElementById('file-cam-${sec.id}').click()">
            <div class="upload-area-icon">📷</div>
            <div class="upload-area-text">Camera</div>
            <div class="upload-area-hint">Take photo</div>
          </div>
          <div class="upload-area" onclick="document.getElementById('file-gal-${sec.id}').click()">
            <div class="upload-area-icon">🖼️</div>
            <div class="upload-area-text">Gallery</div>
            <div class="upload-area-hint">Browse files</div>
          </div>
        </div>
      `;
      // Attach listener immediately
      setTimeout(() => {
        const camIn = document.getElementById(`file-cam-${sec.id}`);
        const galIn = document.getElementById(`file-gal-${sec.id}`);
        if(camIn) camIn.addEventListener('change', (e) => handleFileUpload(e, sec.id));
        if(galIn) galIn.addEventListener('change', (e) => handleFileUpload(e, sec.id));
      }, 0);
    } else {
      // Show Results
      let qualityPillsHtml = '';
      if (s.metrics) {
        const blurPass = s.metrics.blur > 100;
        const brightPass = s.metrics.brightness > 40 && s.metrics.brightness < 220;
        const glarePass = s.metrics.glareRatio < 0.05;
        
        const blurText = state.devMode ? `Blur: ${s.metrics.blur.toFixed(1)}` : `Blur: ${blurPass ? 'Good ✓' : 'Failed ❌'}`;
        const brightText = state.devMode ? `Light: ${s.metrics.brightness.toFixed(1)}` : `Light: ${brightPass ? 'OK ✓' : 'Failed ❌'}`;
        const glareText = state.devMode ? `Glare: ${(s.metrics.glareRatio*100).toFixed(1)}%` : `Glare: ${glarePass ? 'None ✓' : 'High ❌'}`;

        qualityPillsHtml = `
          <div class="quality-pills">
            <span class="quality-pill ${blurPass ? 'pass' : 'fail'}">${blurText}</span>
            <span class="quality-pill ${brightPass ? 'pass' : 'fail'}">${brightText}</span>
            <span class="quality-pill ${glarePass ? 'pass' : 'fail'}">${glareText}</span>
          </div>
        `;
      }

      let detectionHtml = '';
      if (s.detections && s.detections.length > 0) {
        detectionHtml = `<div class="detection-list">`;
        s.detections.forEach(det => {
          detectionHtml += `
            <div class="detection-item ${det.className}">
              <span class="detection-name">${formatClassName(det.className)}</span>
              <span class="detection-conf">${(det.conf * 100).toFixed(1)}%</span>
            </div>
          `;
        });
        detectionHtml += `</div>`;
      } else if (sec.aiEnabled && s.status !== 'error') {
        detectionHtml = `<div class="detection-list"><div class="detection-item"><span class="detection-name">No objects detected</span></div></div>`;
      }

      let verdictClass = 'rejected';
      if (s.status === 'done') verdictClass = 'approved';
      if (s.status === 'warning') verdictClass = 'warning';

      let verdictTitle = s.status === 'done' ? '✅ APPROVED' : (s.status === 'warning' ? '⚠️ WARNING' : '❌ REJECTED');

      body.innerHTML = `
        <div class="preview-container" id="preview-${sec.id}">
          <img src="${s.imgDataUrl}" class="preview-image" id="img-${sec.id}" />
          <!-- Bounding boxes will be injected here -->
        </div>
        ${qualityPillsHtml}
        
        <div class="verdict-card ${verdictClass}">
          <div class="verdict-header">
            <span class="verdict-title">${verdictTitle}</span>
            ${s.runtime ? `<span class="verdict-time">${s.runtime}ms</span>` : ''}
          </div>
          <div class="verdict-text">${s.verdict || 'Processed'}</div>
        </div>
        
        ${detectionHtml}
        
        <div class="card-actions" style="flex-wrap: wrap;">
          ${s.status === 'done' || s.status === 'warning' ? `<button class="btn btn-primary btn-full" onclick="collapseCard()">✓ Accept</button>` : ''}
          ${state.devMode && s.status === 'error' ? `<button class="btn btn-ghost btn-full" onclick="forceAccept('${sec.id}')">⚠️ Force Accept (Dev)</button>` : ''}
          <div style="display:flex; gap:12px; width:100%; margin-top:8px;">
            <button class="btn btn-ghost" style="flex:1;" onclick="retakePhoto('${sec.id}', 'camera')">📷 Retake</button>
            <button class="btn btn-ghost" style="flex:1;" onclick="retakePhoto('${sec.id}', 'gallery')">🖼️ Upload</button>
          </div>
        </div>
      `;
      
      // Draw bounding boxes after image loads
      if (s.detections && s.detections.length > 0) {
        setTimeout(() => {
          drawBoundingBoxesDOM(sec.id, s.detections);
        }, 50);
      }
    }

    card.appendChild(body);
    dom.cardsContainer.appendChild(card);
  });
  } catch (err) {
    console.error('renderCards error:', err);
    dom.cardsContainer.innerHTML = `<div style="color:red; padding: 20px;">Error rendering UI: ${err.message}<br/>${err.stack}</div>`;
  }
}

function toggleCard(id) {
  if (state.expandedSection === id) {
    state.expandedSection = null;
  } else {
    state.expandedSection = id;
  }
  renderCards();
  if (state.expandedSection) {
    setTimeout(() => {
      document.getElementById(`card-${id}`).scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
  }
}

window.collapseCard = function() {
  state.expandedSection = null;
  renderCards();
};

window.forceAccept = function(id) {
  if (state.sections[id]) {
    state.sections[id].status = 'warning'; // Override to warning so it counts as done
    state.sections[id].verdict = 'Forced acceptance via Developer Mode.';
    renderCards();
    renderProgressSegments();
  }
};

window.retakePhoto = function(id, source) {
  state.sections[id] = { status: 'empty' };
  renderCards();
  renderProgressSegments();
  setTimeout(() => {
    const fileInput = document.getElementById(`file-${source === 'camera' ? 'cam' : 'gal'}-${id}`);
    if (fileInput) fileInput.click();
  }, 100);
};

function formatClassName(name) {
  return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

// ─── Image Processing & Logic ──────────────────────────────────────

async function handleFileUpload(event, sectionId) {
  const file = event.target.files[0];
  if (!file) return;

  logToDev(`File selected for section '${sectionId}': ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);

  // Set processing state
  state.sections[sectionId] = { status: 'processing', processText: 'Reading image...' };
  state.expandedSection = sectionId;
  renderCards();

  // Helper to await UI repaint
  const waitRepaint = () => new Promise(r => setTimeout(r, 50));

  const reader = new FileReader();
  reader.onload = async (e) => {
    const imgDataUrl = e.target.result;
    
    // Load image into an Image object
    const img = new Image();
    img.src = imgDataUrl;
    await new Promise(r => img.onload = r);

    try {
        const s = state.sections[sectionId];
        s.processText = 'Analyzing image quality...';
        renderCards();
        
        logToDev(`[${sectionId}] Image loaded: ${img.width}x${img.height}. Starting analysis...`);
        await waitRepaint();

        // 1. Quality Checks
        const metrics = calculateQualityMetrics(img);
        const blurPass = metrics.blur > 100;
        const brightPass = metrics.brightness > 40 && metrics.brightness < 220;
        const glarePass = metrics.glareRatio < 0.05;

        if (!blurPass || !brightPass || !glarePass) {
          state.sections[sectionId] = {
            status: 'error',
            imgDataUrl,
            metrics,
            verdict: 'Photo failed quality checks (too blurry, dark, or glaring). Please retake.'
          };
          renderCards();
          renderProgressSegments();
          return;
        }

        const config = PHOTO_SECTIONS.find(s => s.id === sectionId);

        // 2. AI Inference
        if (config.aiEnabled && state.modelLoaded) {
          // Update UI step
          state.sections[sectionId].processText = 'Running AI Object Detection...';
          renderCards();
          await waitRepaint();

          const startTime = performance.now();
          const detections = await runDetection(img, sectionId);
          const runtime = Math.round(performance.now() - startTime);

          // Filter best per class
          const bestDetections = keepBestPerClass(detections);
          
          // 3. Logic Evaluation
          const evaluation = evaluateSectionLogic(config, bestDetections);

          state.sections[sectionId] = {
            status: evaluation.status,
            imgDataUrl,
            metrics,
            detections: bestDetections,
            verdict: evaluation.verdict,
            runtime
          };
        } else {
          // Non-AI section, passes automatically if quality is good
          state.sections[sectionId] = {
            status: 'done',
            imgDataUrl,
            metrics,
            verdict: 'Photo quality is acceptable.'
          };
        }

        logToDev(`[${sectionId}] Processing complete. Status: ${state.sections[sectionId].status.toUpperCase()}`);
        renderCards();
        renderProgressSegments();
    } catch (err) {
        console.error(err);
        logToDev(`[${sectionId}] ERROR: ${err.message}`);
    }
  };
  reader.readAsDataURL(file);
}

function evaluateSectionLogic(config, detections) {
  const detectedClasses = detections.map(d => d.className);
  
  // Check required classes
  const missing = config.requiredClasses.filter(c => !detectedClasses.includes(c));
  if (missing.length > 0) {
    return {
      status: 'error',
      verdict: `Missing required objects: ${missing.map(formatClassName).join(', ')}`
    };
  }

  // Check enclosure if required
  if (config.checkEnclosure && detectedClasses.includes('lcd_screen') && detectedClasses.includes('multimeter')) {
    const lcd = detections.find(d => d.className === 'lcd_screen');
    const mm = detections.find(d => d.className === 'multimeter');
    
    const isEnclosed = lcd.box[0] >= mm.box[0] &&
                       lcd.box[1] >= mm.box[1] &&
                       lcd.box[2] <= mm.box[2] &&
                       lcd.box[3] <= mm.box[3];
                       
    if (!isEnclosed) {
      return {
        status: 'warning',
        verdict: 'Required objects found, but LCD screen does not appear to be inside the multimeter. Proceed with caution.'
      };
    }
  }

  return {
    status: 'done',
    verdict: 'All required objects detected perfectly.'
  };
}

// Draw bounding boxes on top of the image using absolutely positioned DIVs
function drawBoundingBoxesDOM(sectionId, detections) {
  const container = document.getElementById(`preview-${sectionId}`);
  const img = document.getElementById(`img-${sectionId}`);
  if (!container || !img) return;

  // Get displayed image dimensions
  const rect = img.getBoundingClientRect();
  const scaleX = rect.width / img.naturalWidth;
  const scaleY = rect.height / img.naturalHeight;

  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.box;
    const color = COLORS[det.className] || '#fff';
    
    const boxDiv = document.createElement('div');
    boxDiv.className = 'bbox';
    boxDiv.style.left = `${x1 * scaleX}px`;
    boxDiv.style.top = `${y1 * scaleY}px`;
    boxDiv.style.width = `${(x2 - x1) * scaleX}px`;
    boxDiv.style.height = `${(y2 - y1) * scaleY}px`;
    boxDiv.style.borderColor = color;

    const labelDiv = document.createElement('div');
    labelDiv.className = 'bbox-label';
    labelDiv.style.backgroundColor = color;
    labelDiv.innerText = `${formatClassName(det.className)} ${(det.conf * 100).toFixed(1)}%`;
    
    boxDiv.appendChild(labelDiv);
    container.appendChild(boxDiv);
  });
}

// ─── AI Pipeline Functions ─────────────────────────────────────────

async function runDetection(img, sectionId) {
  const { tensor, ratios, padParams } = preprocess(img);
  
  logToDev(`[${sectionId}] Preprocessing complete. Running ONNX inference...`);

  const start = performance.now();
  const feeds = { images: tensor };
  const results = await session.run(feeds);
  const end = performance.now();
  const runtime = Math.round(end - start);

  const output = results[session.outputNames[0]];
  const detections = postprocess(output, ratios, padParams, img.naturalWidth, img.naturalHeight);
  
  logToDev(`[${sectionId}] Inference finished in ${runtime}ms. Found ${detections.length} raw boxes.`);
  return detections;
}

function preprocess(img) {
  const targetSize = 640;
  const ctx = dom.hiddenCanvas.getContext('2d', { willReadFrequently: true });
  dom.hiddenCanvas.width = targetSize;
  dom.hiddenCanvas.height = targetSize;

  const scale = Math.min(targetSize / img.naturalWidth, targetSize / img.naturalHeight);
  const newW = Math.round(img.naturalWidth * scale);
  const newH = Math.round(img.naturalHeight * scale);
  const padX = (targetSize - newW) / 2;
  const padY = (targetSize - newH) / 2;

  ctx.fillStyle = 'rgb(114, 114, 114)';
  ctx.fillRect(0, 0, targetSize, targetSize);
  ctx.drawImage(img, padX, padY, newW, newH);

  const imgData = ctx.getImageData(0, 0, targetSize, targetSize);
  const data = imgData.data;
  
  const float32Data = new Float32Array(3 * targetSize * targetSize);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    float32Data[j] = data[i] / 255.0; // R
    float32Data[j + targetSize * targetSize] = data[i + 1] / 255.0; // G
    float32Data[j + 2 * targetSize * targetSize] = data[i + 2] / 255.0; // B
  }

  const tensor = new ort.Tensor('float32', float32Data, [1, 3, targetSize, targetSize]);
  return { tensor, ratios: [scale, scale], padParams: [padX, padY] };
}

function postprocess(output, ratios, padParams, origW, origH) {
  const data = output.data;
  logToDev(`YOLO Output Dims: [${output.dims.join(', ')}]`);
  
  const results = [];
  const CONF_THRESHOLD = 0.25;

  let numBoxes, numChannels, isTransposed;
  if (output.dims.length === 3) {
    if (output.dims[1] < output.dims[2]) {
      // Shape [1, 7, 8400] (Standard YOLOv8)
      numChannels = output.dims[1];
      numBoxes = output.dims[2];
      isTransposed = true;
    } else {
      // Shape [1, 300, 7] (YOLOv5/v8 with NMS)
      numBoxes = output.dims[1];
      numChannels = output.dims[2];
      isTransposed = false;
    }
  } else if (output.dims.length === 2) {
    // Shape [300, 7]
    numBoxes = output.dims[0];
    numChannels = output.dims[1];
    isTransposed = false;
  } else {
    logToDev(`Warning: Unknown output shape [${output.dims.join(', ')}]`);
    return [];
  }

  for (let i = 0; i < numBoxes; i++) {
    let conf = 0;
    let classId = -1;
    let x, y, w, h;
    let x1, y1, x2, y2;

    if (isTransposed) {
      // Data is flattened as [channel0_all_boxes, channel1_all_boxes, ...]
      // For YOLOv8: [cx, cy, w, h, cls1, cls2, cls3]
      
      // Find max class score
      for (let c = 4; c < numChannels; c++) {
        let score = data[c * numBoxes + i];
        if (score > conf) {
          conf = score;
          classId = c - 4; // Classes start at index 4
        }
      }
      
      if (conf < CONF_THRESHOLD) continue;

      x = data[0 * numBoxes + i];
      y = data[1 * numBoxes + i];
      w = data[2 * numBoxes + i];
      h = data[3 * numBoxes + i];
      
      x1 = x - w / 2;
      y1 = y - h / 2;
      x2 = x + w / 2;
      y2 = y + h / 2;

    } else {
      // Data is [box0_c0, box0_c1, ..., box1_c0, ...]
      const offset = i * numChannels;
      
      if (numChannels === 7) {
        // Ultralytics YOLOv8 OBB export format: [x_center, y_center, width, height, conf, class_id, angle]
        x = data[offset + 0];
        y = data[offset + 1];
        w = data[offset + 2];
        h = data[offset + 3];
        conf = data[offset + 4];
        classId = Math.round(data[offset + 5]);
        
        if (conf < CONF_THRESHOLD) continue;
        
        x1 = x - w / 2;
        y1 = y - h / 2;
        x2 = x + w / 2;
        y2 = y + h / 2;
      } else {
        // Assume YOLOv5 default [cx, cy, w, h, obj_conf, cls1, cls2]
        const objConf = data[offset + 4];
        if (objConf < CONF_THRESHOLD) continue;
        
        for (let c = 5; c < numChannels; c++) {
          let score = data[offset + c] * objConf;
          if (score > conf) {
            conf = score;
            classId = c - 5;
          }
        }
        if (conf < CONF_THRESHOLD) continue;
        
        x = data[offset + 0];
        y = data[offset + 1];
        w = data[offset + 2];
        h = data[offset + 3];
        
        x1 = x - w / 2;
        y1 = y - h / 2;
        x2 = x + w / 2;
        y2 = y + h / 2;
      }
    }

    // Remove padding and scale back
    x1 = (x1 - padParams[0]) / ratios[0];
    y1 = (y1 - padParams[1]) / ratios[1];
    x2 = (x2 - padParams[0]) / ratios[0];
    y2 = (y2 - padParams[1]) / ratios[1];

    // Clamp
    x1 = Math.max(0, Math.min(x1, origW));
    y1 = Math.max(0, Math.min(y1, origH));
    x2 = Math.max(0, Math.min(x2, origW));
    y2 = Math.max(0, Math.min(y2, origH));

    const classNames = ['lcd_screen', 'multimeter', 'kazam_box'];
    let className = classNames[classId] || `unknown_${classId}`;

    results.push({
      box: [x1, y1, x2, y2],
      conf,
      classId,
      className
    });
  }
  
  return results;
}

function keepBestPerClass(detections) {
  const best = {};
  for (const d of detections) {
    if (!best[d.className] || d.conf > best[d.className].conf) {
      best[d.className] = d;
    }
  }
  return Object.values(best);
}

// ─── Quality Metrics ───────────────────────────────────────────────

function calculateQualityMetrics(img) {
  const sampleSize = 320;
  const ctx = dom.hiddenCanvas.getContext('2d', { willReadFrequently: true });
  dom.hiddenCanvas.width = sampleSize;
  dom.hiddenCanvas.height = sampleSize;
  ctx.drawImage(img, 0, 0, sampleSize, sampleSize);
  
  const imgData = ctx.getImageData(0, 0, sampleSize, sampleSize);
  const data = imgData.data;
  
  let brightnessSum = 0;
  let brightPixels = 0;
  const totalPixels = sampleSize * sampleSize;

  for (let i = 0; i < data.length; i += 4) {
    const luma = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
    brightnessSum += luma;
    if (luma > 240) brightPixels++;
  }

  const brightness = brightnessSum / totalPixels;
  const glareRatio = brightPixels / totalPixels;
  const blur = laplacianVariance(imgData);

  return { brightness, glareRatio, blur };
}

function laplacianVariance(imgData) {
  const { width, height, data } = imgData;
  const gray = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    gray[j] = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
  }

  let mean = 0;
  const laplacian = new Int16Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const val = 4 * gray[idx] - gray[idx - 1] - gray[idx + 1] - gray[idx - width] - gray[idx + width];
      laplacian[idx] = val;
      mean += val;
    }
  }
  mean /= (width * height);

  let variance = 0;
  for (let i = 0; i < laplacian.length; i++) {
    variance += Math.pow(laplacian[i] - mean, 2);
  }
  return variance / laplacian.length;
}

// ─── Theming & Utilities ───────────────────────────────────────────

function initTheme() {
  const saved = localStorage.getItem('kazam_theme');
  if (saved === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    dom.themeIcon.innerText = '🌙';
    dom.themeText.innerText = 'Dark';
    document.getElementById('theme-color-meta').setAttribute('content', '#f6f8fb');
  }

  dom.themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    if (current === 'light') {
      document.documentElement.removeAttribute('data-theme');
      localStorage.setItem('kazam_theme', 'dark');
      dom.themeIcon.innerText = '☀️';
      dom.themeText.innerText = 'Light';
      document.getElementById('theme-color-meta').setAttribute('content', '#090e19');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
      localStorage.setItem('kazam_theme', 'light');
      dom.themeIcon.innerText = '🌙';
      dom.themeText.innerText = 'Dark';
      document.getElementById('theme-color-meta').setAttribute('content', '#f6f8fb');
    }
  });
}

// Developer Menu logic
function setupDevMenu() {
  let timer;
  dom.versionFooter.addEventListener('touchstart', () => { timer = setTimeout(showDevMenu, 1500); });
  dom.versionFooter.addEventListener('touchend', () => { clearTimeout(timer); });
  dom.versionFooter.addEventListener('mousedown', () => { timer = setTimeout(showDevMenu, 1500); });
  dom.versionFooter.addEventListener('mouseup', () => { clearTimeout(timer); });
  dom.versionFooter.addEventListener('mouseleave', () => { clearTimeout(timer); });

  function showDevMenu() {
    dom.devMenu.style.display = 'flex';
    
    // Render device stats
    const stats = document.getElementById('deviceStats');
    if (stats) {
      const mem = navigator.deviceMemory ? `${navigator.deviceMemory} GB` : 'Unknown';
      const cores = navigator.hardwareConcurrency || 'Unknown';
      const isPWA = window.matchMedia('(display-mode: standalone)').matches ? 'Yes' : 'No';
      const isOnline = navigator.onLine ? '<span style="color:var(--success)">Online</span>' : '<span style="color:var(--danger)">Offline</span>';
      
      stats.innerHTML = `
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;"><span>Device RAM:</span> <strong>${mem}</strong></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;"><span>CPU Cores:</span> <strong>${cores}</strong></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;"><span>Installed PWA:</span> <strong>${isPWA}</strong></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;"><span>Network:</span> <strong>${isOnline}</strong></div>
        <div style="margin-top:8px; border-top:1px solid rgba(255,255,255,0.1); padding-top:8px; font-size:10px; word-break:break-all; opacity:0.7;">
          ${navigator.userAgent}
        </div>
      `;
    }
    
    const toggleBtn = document.getElementById('toggleDevModeBtn');
    if (toggleBtn) {
      toggleBtn.innerText = state.devMode ? 'Disable Dev Mode UI (Hide Stats)' : 'Enable Dev Mode UI (Show Exact Numbers)';
      toggleBtn.style.color = state.devMode ? 'var(--text)' : 'var(--warning)';
    }
  }

  const toggleBtn = document.getElementById('toggleDevModeBtn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      state.devMode = !state.devMode;
      toggleBtn.innerText = state.devMode ? 'Disable Dev Mode UI (Hide Stats)' : 'Enable Dev Mode UI (Show Exact Numbers)';
      toggleBtn.style.color = state.devMode ? 'var(--text)' : 'var(--warning)';
      renderCards(); // Re-render to show dev metrics if cards are expanded
    });
  }

  dom.closeDevMenuBtn.addEventListener('click', () => {
    dom.devMenu.style.display = 'none';
  });

  dom.clearCacheBtn.addEventListener('click', async () => {
    if (confirm('Clear PWA Cache and unregister service worker? This will remove offline capability until you reload online.')) {
      const registrations = await navigator.serviceWorker.getRegistrations();
      for (const reg of registrations) await reg.unregister();
      const keys = await caches.keys();
      for (const key of keys) await caches.delete(key);
      alert('Cache cleared. Reloading...');
      window.location.reload();
    }
  });

  dom.testBtn.addEventListener('click', async () => {
    dom.devMenu.style.display = 'none';
    const response = await fetch('/test_sample.jpeg');
    const blob = await response.blob();
    const file = new File([blob], "test_sample.jpeg", { type: "image/jpeg" });
    const dt = new DataTransfer();
    dt.items.add(file);
    const mockEvent = { target: { files: dt.files } };
    handleFileUpload(mockEvent, 'full_setup');
  });

  // Easy Dev Mode Toggle via PWA Badge
  if (dom.devModeToggleBtnHeader) {
    dom.devModeToggleBtnHeader.addEventListener('click', () => {
      state.devMode = !state.devMode;
      if (state.devMode) {
        dom.devModeToggleBtnHeader.innerText = 'DEV';
        dom.devModeToggleBtnHeader.style.background = 'var(--warning-dim)';
        dom.devModeToggleBtnHeader.style.color = 'var(--warning)';
        dom.devModeToggleBtnHeader.style.borderColor = 'rgba(255, 197, 61, 0.4)';
        if (dom.devStatsCard) dom.devStatsCard.style.display = 'block';
        logToDev('Dev Mode Enabled.');
      } else {
        dom.devModeToggleBtnHeader.innerText = 'PWA';
        dom.devModeToggleBtnHeader.style.background = 'var(--accent-dim)';
        dom.devModeToggleBtnHeader.style.color = 'var(--accent)';
        dom.devModeToggleBtnHeader.style.borderColor = 'rgba(0, 230, 138, 0.2)';
        if (dom.devStatsCard) dom.devStatsCard.style.display = 'none';
        logToDev('Dev Mode Disabled.');
      }
      // Keep the Dev Menu toggle button in sync if the menu is open
      const toggleBtn = document.getElementById('toggleDevModeBtn');
      if (toggleBtn) {
        toggleBtn.innerText = state.devMode ? 'Disable Dev Mode UI (Hide Stats)' : 'Enable Dev Mode UI (Show Exact Numbers)';
        toggleBtn.style.color = state.devMode ? 'var(--text)' : 'var(--warning)';
      }
      renderCards();
    });
  }
}

// Start
window.addEventListener('DOMContentLoaded', initApp);
