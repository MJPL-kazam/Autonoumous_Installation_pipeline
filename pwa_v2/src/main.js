import * as ort from 'onnxruntime-web';

// ─── Configuration & State ─────────────────────────────────────────

const APP_BASE_URL = import.meta.env.BASE_URL || '/';
const assetUrl = (path) => `${APP_BASE_URL}${path.replace(/^\/+/, '')}`;

const MULTIMETER_MODEL_URL = assetUrl('model/multimeter_t400v100.onnx');
const POTHOLE_MODEL_URL = assetUrl('model/pothole_t417v100.onnx');
const SAMPLE_FULL_SETUP_URL = assetUrl('test-images/MAH-cparxutx9xf__Meter_Photo_(N-E).jpg');
const SAMPLE_EARTHPIT_OPEN_URL = assetUrl('test-images/MAH-cparydw2rne__Open_Earthpit.jpg');

let multimeterSession = null;
let potholeSession = null;

const COLORS = {
  'kazam_box': '#2f80ff', // Blue
  'multimeter': '#00e68a', // Green
  'lcd_screen': '#ff4dff', // Pink
  'earthpit_open': '#ff9900', // Orange
  'earthpit_cover': '#9933ff' // Purple
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
    instruction: 'Clear photo(s) of the earthing pit installation (Open & Covered)',
    aiEnabled: true,
    requiredClasses: ['earthpit_open', 'earthpit_cover'],
    checkEarthPit: true,
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
  logs: [],
  lastQCanvasSnap: null
};

// Initialize state
PHOTO_SECTIONS.forEach(s => {
  if (s.id === 'earth_pit') {
    state.sections[s.id] = {
      status: 'empty',
      open_pit: { status: 'empty', imgDataUrl: '', metrics: null, detections: null, runtime: 0 },
      covered_pit: { status: 'empty', imgDataUrl: '', metrics: null, detections: null, runtime: 0 },
      verdict: '',
      runtime: 0
    };
  } else {
    state.sections[s.id] = { status: 'empty' };
  }
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
  clearLogsBtn: document.getElementById('clearLogsBtn'),
  debugGallery: document.getElementById('debugGallery'),
  debugGalleryContent: document.getElementById('debugGalleryContent')
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
    ort.env.wasm.wasmPaths = assetUrl('wasm/');
    ort.env.wasm.numThreads = 1;
    
    // Create sessions for both models
    multimeterSession = await ort.InferenceSession.create(MULTIMETER_MODEL_URL, { executionProviders: ['wasm'] });
    potholeSession = await ort.InferenceSession.create(POTHOLE_MODEL_URL, { executionProviders: ['wasm'] });
    
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
    
    if (sec.id === 'earth_pit') {
      const sOpen = s.open_pit;
      const sCover = s.covered_pit;

      const renderSlot = (slotId, title, slotState) => {
        if (slotState.status === 'processing') {
          return `
            <div class="earthpit-slot processing">
              <div class="slot-title">${title}</div>
              <div class="upload-area-icon">⏳</div>
              <div class="upload-area-text">Processing...</div>
            </div>
          `;
        } else if (slotState.status === 'empty') {
          return `
            <div class="earthpit-slot empty">
              <div class="slot-title">${title}</div>
              <input type="file" id="file-cam-${sec.id}-${slotId}" accept="image/*" capture="environment" hidden />
              <input type="file" id="file-gal-${sec.id}-${slotId}" accept="image/*" hidden />
              <div class="upload-actions-grid compact">
                <div class="upload-area compact" onclick="document.getElementById('file-cam-${sec.id}-${slotId}').click()">
                  <div class="upload-area-icon">📷</div>
                  <div class="upload-area-text">Camera</div>
                </div>
                <div class="upload-area compact" onclick="document.getElementById('file-gal-${sec.id}-${slotId}').click()">
                  <div class="upload-area-icon">🖼️</div>
                  <div class="upload-area-text">Gallery</div>
                </div>
              </div>
            </div>
          `;
        } else {
          // Done, error, or warning
          let qualityPillsHtml = '';
          if (slotState.metrics) {
            const blurPass = slotState.metrics.blur > 100;
            const brightPass = slotState.metrics.brightness > 40 && slotState.metrics.brightness < 220;
            const glarePass = slotState.metrics.glareRatio < 0.05;
            
            const blurText = state.devMode ? `B: ${slotState.metrics.blur.toFixed(0)}` : `Blur: ${blurPass ? 'Good' : 'Fail'}`;
            const brightText = state.devMode ? `L: ${slotState.metrics.brightness.toFixed(0)}` : `Light: ${brightPass ? 'OK' : 'Fail'}`;
            const glareText = state.devMode ? `G: ${(slotState.metrics.glareRatio*100).toFixed(0)}%` : `Glare: ${glarePass ? 'None' : 'High'}`;

            qualityPillsHtml = `
              <div class="quality-pills compact">
                <span class="quality-pill ${blurPass ? 'pass' : 'fail'}">${blurText}</span>
                <span class="quality-pill ${brightPass ? 'pass' : 'fail'}">${brightText}</span>
                <span class="quality-pill ${glarePass ? 'pass' : 'fail'}">${glareText}</span>
              </div>
            `;
          }

          let detectionHtml = '';
          if (slotState.detections && slotState.detections.length > 0) {
            detectionHtml = `<div class="detection-list compact">`;
            slotState.detections.forEach(det => {
              detectionHtml += `
                <div class="detection-item compact ${det.className}">
                  <span class="detection-name">${formatClassName(det.className)}</span>
                  <span class="detection-conf">${(det.conf * 100).toFixed(0)}%</span>
                </div>
              `;
            });
            detectionHtml += `</div>`;
          } else {
            detectionHtml = `<div class="detection-list compact"><div class="detection-item compact"><span class="detection-name">No classes detected</span></div></div>`;
          }

          return `
            <div class="earthpit-slot filled">
              <div class="slot-title">${title}</div>
              <div class="preview-container compact" id="preview-${sec.id}-${slotId}" onclick="toggleZoom(this)" style="cursor: zoom-in;">
                <div class="zoom-hint-badge compact">🔍 Zoom</div>
                <div class="preview-wrapper" id="wrapper-${sec.id}-${slotId}">
                  <img src="${slotState.imgDataUrl}" class="preview-image" id="img-${sec.id}-${slotId}" />
                </div>
              </div>
              ${qualityPillsHtml}
              ${detectionHtml}
              <div style="display:flex; gap:6px; margin-top:8px;">
                <button type="button" class="btn btn-ghost btn-compact" onclick="retakeEarthPitPhoto('${slotId}', 'camera')">📷 Retake</button>
                <button type="button" class="btn btn-ghost btn-compact" onclick="retakeEarthPitPhoto('${slotId}', 'gallery')">🖼️ Upload</button>
              </div>
            </div>
          `;
        }
      };

      let verdictClass = 'rejected';
      if (s.status === 'done') verdictClass = 'approved';
      if (s.status === 'warning') verdictClass = 'warning';
      let verdictTitle = s.status === 'done' ? '✅ APPROVED' : (s.status === 'warning' ? '⚠️ WARNING' : '❌ INCOMPLETE');

      body.innerHTML = `
        <div class="earthpit-slots-container">
          ${renderSlot('open_pit', 'Open Earth Pit', sOpen)}
          ${renderSlot('covered_pit', 'Covered Earth Pit', sCover)}
        </div>
        
        ${s.status !== 'empty' && s.status !== 'processing' ? `
          <div class="verdict-card ${verdictClass}" style="margin-top:16px;">
            <div class="verdict-header">
              <span class="verdict-title">${verdictTitle}</span>
              ${s.runtime ? `<span class="verdict-time">${s.runtime}ms</span>` : ''}
            </div>
            <div class="verdict-text">${s.verdict || 'Processed'}</div>
          </div>
        ` : ''}

        <div class="card-actions" style="flex-wrap: wrap; margin-top:16px;">
          ${s.status === 'done' || s.status === 'warning' ? `<button type="button" class="btn btn-primary btn-full" onclick="collapseCard()">✓ Accept</button>` : ''}
          ${state.devMode && s.status !== 'done' && s.status !== 'warning' ? `<button type="button" class="btn btn-ghost btn-full" onclick="forceAccept('${sec.id}')">⚠️ Force Accept (Dev)</button>` : ''}
        </div>
      `;

      // Attach file input listeners
      setTimeout(() => {
        ['open_pit', 'covered_pit'].forEach(slotId => {
          const camIn = document.getElementById(`file-cam-${sec.id}-${slotId}`);
          const galIn = document.getElementById(`file-gal-${sec.id}-${slotId}`);
          if (camIn) camIn.addEventListener('change', (e) => handleEarthPitUpload(e, slotId));
          if (galIn) galIn.addEventListener('change', (e) => handleEarthPitUpload(e, slotId));
        });
      }, 0);

      // Draw bounding boxes for filled slots after image loads
      if (sOpen.status !== 'empty' && sOpen.status !== 'processing' && sOpen.detections && sOpen.detections.length > 0) {
        setTimeout(() => {
          drawBoundingBoxesDOM(`${sec.id}-open_pit`, sOpen.detections, `img-${sec.id}-open_pit`, `wrapper-${sec.id}-open_pit`);
        }, 50);
      }
      if (sCover.status !== 'empty' && sCover.status !== 'processing' && sCover.detections && sCover.detections.length > 0) {
        setTimeout(() => {
          drawBoundingBoxesDOM(`${sec.id}-covered_pit`, sCover.detections, `img-${sec.id}-covered_pit`, `wrapper-${sec.id}-covered_pit`);
        }, 50);
      }

    } else {
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
          <div class="preview-container" id="preview-${sec.id}" onclick="toggleZoom(this)" style="cursor: zoom-in;">
            <div class="zoom-hint-badge">🔍 Tap to zoom</div>
            <div class="preview-wrapper" id="wrapper-${sec.id}">
              <img src="${s.imgDataUrl}" class="preview-image" id="img-${sec.id}" />
            </div>
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
    if (id === 'earth_pit') {
      const s = state.sections[id];
      s.status = 'warning';
      s.verdict = 'Forced acceptance via Developer Mode.';
      if (s.open_pit.status === 'empty' || s.open_pit.status === 'error') {
        s.open_pit.status = 'warning';
        s.open_pit.verdict = 'Forced';
      }
      if (s.covered_pit.status === 'empty' || s.covered_pit.status === 'error') {
        s.covered_pit.status = 'warning';
        s.covered_pit.verdict = 'Forced';
      }
    } else {
      state.sections[id].status = 'warning'; // Override to warning so it counts as done
      state.sections[id].verdict = 'Forced acceptance via Developer Mode.';
    }
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

window.retakeEarthPitPhoto = function(slotId, source) {
  const s = state.sections['earth_pit'];
  s[slotId] = { status: 'empty' };
  updateEarthPitSectionStatus();
  renderCards();
  renderProgressSegments();
  setTimeout(() => {
    const fileInput = document.getElementById(`file-${source === 'camera' ? 'cam' : 'gal'}-earth_pit-${slotId}`);
    if (fileInput) fileInput.click();
  }, 100);
};

function formatClassName(name) {
  return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

function triggerAutoAcceptProgression(sectionId) {
  const cardEl = document.getElementById(`card-${sectionId}`);
  if (cardEl) {
    // Add success flash effect
    cardEl.classList.add('approved-flash');
  }

  // Await 1.2s before auto-collapsing and moving to next
  setTimeout(() => {
    if (cardEl) {
      cardEl.classList.remove('approved-flash');
    }
    
    // Auto-collapse current
    state.expandedSection = null;

    // Find next incomplete section
    const nextIncomplete = PHOTO_SECTIONS.find(sec => {
      const s = state.sections[sec.id];
      return s.status === 'empty' || s.status === 'partial' || s.status === 'error';
    });

    if (nextIncomplete) {
      state.expandedSection = nextIncomplete.id;
      renderCards();
      renderProgressSegments();
      
      // Scroll to the next card
      setTimeout(() => {
        const nextEl = document.getElementById(`card-${nextIncomplete.id}`);
        if (nextEl) {
          const y = nextEl.getBoundingClientRect().top + window.scrollY - 70;
          window.scrollTo({ top: y, behavior: 'smooth' });
        }
      }, 50);
    } else {
      // All cards completed! Close all and scroll to bottom submit button
      renderCards();
      renderProgressSegments();
      setTimeout(() => {
        const submitBtn = dom.submitBtn;
        if (submitBtn) {
          submitBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 50);
    }
  }, 1200);
}

// ─── Image Processing & Logic ──────────────────────────────────────

async function handleFileUpload(event, sectionId) {
  const file = event.target.files[0];
  if (!file) return;

  logToDev(`File selected for section '${sectionId}': ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);

  // Set processing state
  state.sections[sectionId] = { status: 'processing', processText: 'Reading photo...' };
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
        s.processText = 'Checking photo quality...';
        renderCards();
        
        logToDev(`[${sectionId}] Image loaded: ${img.width}x${img.height}. Starting analysis...`);
        await waitRepaint();

        // 1. Quality Checks
        const metrics = calculateQualityMetrics(img);
        const blurPass = metrics.blur > 100;
        const brightPass = metrics.brightness > 40 && metrics.brightness < 220;
        const glarePass = metrics.glareRatio < 0.05;

        if (!blurPass || !brightPass || !glarePass) {
          let reason = 'Photo quality check failed.';
          if (!blurPass) reason = '❌ Photo is too blurry. Please hold the phone steady and retake.';
          else if (!brightPass) reason = '❌ Photo is too dark. Please turn on a light/flash and retake.';
          else if (!glarePass) reason = '❌ Photo has too much glare. Please change your angle and retake.';

          state.sections[sectionId] = {
            status: 'error',
            imgDataUrl,
            metrics,
            verdict: reason
          };
          renderCards();
          renderProgressSegments();
          return;
        }

        const config = PHOTO_SECTIONS.find(s => s.id === sectionId);

        // 2. AI Inference
        if (config.aiEnabled && state.modelLoaded) {
          s.processText = 'Running AI checker...';
          renderCards();
          await waitRepaint();

          const startTime = performance.now();
          const detections = await runDetection(img, sectionId);
          const runtime = Math.round(performance.now() - startTime);

          const bestDetections = keepBestPerClass(detections);
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
            verdict: 'Photo quality check passed.'
          };
        }

        logToDev(`[${sectionId}] Processing complete. Status: ${state.sections[sectionId].status.toUpperCase()}`);
        
        if (state.sections[sectionId].status === 'done') {
          triggerAutoAcceptProgression(sectionId);
        } else {
          renderCards();
          renderProgressSegments();
        }
    } catch (err) {
        console.error(err);
        logToDev(`[${sectionId}] ERROR: ${err.message}`);
        state.sections[sectionId] = {
          status: 'error',
          verdict: `Error: ${err.message}`
        };
        renderCards();
        renderProgressSegments();
    }
  };
  reader.readAsDataURL(file);
}

async function handleEarthPitUpload(event, targetSlotId) {
  const file = event.target.files[0];
  if (!file) return;

  logToDev(`File selected for earth_pit [${targetSlotId}]: ${file.name}`);

  const s = state.sections['earth_pit'];
  
  // Set processing state for the slot and section
  s.status = 'processing';
  s.open_pit.status = targetSlotId === 'open_pit' ? 'processing' : s.open_pit.status;
  s.covered_pit.status = targetSlotId === 'covered_pit' ? 'processing' : s.covered_pit.status;
  s.processText = 'Reading photo...';
  renderCards();

  const waitRepaint = () => new Promise(r => setTimeout(r, 50));

  const reader = new FileReader();
  reader.onload = async (e) => {
    const imgDataUrl = e.target.result;
    const img = new Image();
    img.src = imgDataUrl;
    await new Promise(r => img.onload = r);

    try {
      logToDev(`[earth_pit] Image loaded for ${targetSlotId}: ${img.width}x${img.height}. Starting analysis...`);
      s.processText = 'Checking photo quality...';
      renderCards();
      await waitRepaint();

      // 1. Quality Checks
      const metrics = calculateQualityMetrics(img);
      const blurPass = metrics.blur > 100;
      const brightPass = metrics.brightness > 40 && metrics.brightness < 220;
      const glarePass = metrics.glareRatio < 0.05;

      if (!blurPass || !brightPass || !glarePass) {
        let reason = 'Photo quality check failed.';
        if (!blurPass) reason = '❌ Photo is too blurry. Please hold the phone steady and retake.';
        else if (!brightPass) reason = '❌ Photo is too dark. Please turn on a light/flash and retake.';
        else if (!glarePass) reason = '❌ Photo has too much glare. Please change your angle and retake.';

        s[targetSlotId] = {
          status: 'error',
          imgDataUrl,
          metrics,
          detections: [],
          verdict: reason
        };
        updateEarthPitSectionStatus();
        renderCards();
        renderProgressSegments();
        return;
      }

      // 2. AI Inference
      s.processText = 'Running AI checker...';
      s[targetSlotId] = { status: 'processing', imgDataUrl, metrics };
      renderCards();
      await waitRepaint();

      const startTime = performance.now();
      const detections = await runDetection(img, 'earth_pit');
      const runtime = Math.round(performance.now() - startTime);

      const bestDetections = keepBestPerClass(detections);
      const detectedClasses = bestDetections.map(d => d.className);

      // Check for single-image containing both classes
      if (detectedClasses.includes('earthpit_open') && detectedClasses.includes('earthpit_cover')) {
        logToDev(`[earth_pit] Both classes detected in a single image. Populating both slots!`);
        
        // Single-image shortcut! Populate both slots with this image
        s.open_pit = {
          status: 'done',
          imgDataUrl,
          metrics,
          detections: bestDetections.filter(d => d.className === 'earthpit_open'),
          runtime
        };
        s.covered_pit = {
          status: 'done',
          imgDataUrl,
          metrics,
          detections: bestDetections.filter(d => d.className === 'earthpit_cover'),
          runtime
        };
      } else if (detectedClasses.includes('earthpit_open')) {
        // Smart slot assignment: if user uploaded to covered_pit, but we detected open_pit, put it in open_pit
        if (targetSlotId === 'covered_pit') {
          logToDev(`[earth_pit] Smart Assignment: detected 'earthpit_open' in 'covered_pit' upload. Reassigning.`);
        }
        s.open_pit = {
          status: 'done',
          imgDataUrl,
          metrics,
          detections: bestDetections,
          runtime
        };
        // Clean the other slot if it contained the same image to prevent issues
        if (s.covered_pit.imgDataUrl === imgDataUrl || (s.covered_pit.detections && s.covered_pit.detections.some(d => d.className === 'earthpit_open'))) {
          s.covered_pit = { status: 'empty' };
        }
      } else if (detectedClasses.includes('earthpit_cover')) {
        // Smart slot assignment: if user uploaded to open_pit, but we detected covered_pit, put it in covered_pit
        if (targetSlotId === 'open_pit') {
          logToDev(`[earth_pit] Smart Assignment: detected 'earthpit_cover' in 'open_pit' upload. Reassigning.`);
        }
        s.covered_pit = {
          status: 'done',
          imgDataUrl,
          metrics,
          detections: bestDetections,
          runtime
        };
        // Clean the other slot if it contained the same image to prevent issues
        if (s.open_pit.imgDataUrl === imgDataUrl || (s.open_pit.detections && s.open_pit.detections.some(d => d.className === 'earthpit_cover'))) {
          s.open_pit = { status: 'empty' };
        }
      } else {
        // Quality passed but no classes detected
        s[targetSlotId] = {
          status: 'error',
          imgDataUrl,
          metrics,
          detections: [],
          verdict: '❌ Neither open nor covered earth pit was detected. Please check your photo and retake.'
        };
      }

      updateEarthPitSectionStatus();
      
      if (s.status === 'done') {
        triggerAutoAcceptProgression('earth_pit');
      } else {
        renderCards();
        renderProgressSegments();
      }
    } catch (err) {
      console.error(err);
      logToDev(`[earth_pit] ERROR: ${err.message}`);
      s.status = 'error';
      s[targetSlotId].status = 'error';
      updateEarthPitSectionStatus();
      renderCards();
      renderProgressSegments();
    }
  };
  reader.readAsDataURL(file);
}

function updateEarthPitSectionStatus() {
  const s = state.sections['earth_pit'];
  
  if (s.open_pit.status === 'processing' || s.covered_pit.status === 'processing') {
    s.status = 'processing';
    s.verdict = 'Processing uploads...';
    return;
  }

  if (s.open_pit.status === 'done' && s.covered_pit.status === 'done') {
    s.status = 'done';
    s.verdict = 'Both open and covered earth pits verified successfully.';
    s.runtime = Math.max(s.open_pit.runtime || 0, s.covered_pit.runtime || 0);
    return;
  }

  if (s.open_pit.status === 'done' && s.covered_pit.status === 'empty') {
    s.status = 'partial';
    s.verdict = 'Open Earth Pit verified. Please capture/upload the Covered Earth Pit.';
    return;
  }
  if (s.covered_pit.status === 'done' && s.open_pit.status === 'empty') {
    s.status = 'partial';
    s.verdict = 'Covered Earth Pit verified. Please capture/upload the Open Earth Pit.';
    return;
  }

  if (s.open_pit.status === 'error' || s.covered_pit.status === 'error') {
    s.status = 'error';
    s.verdict = s.open_pit.status === 'error' ? s.open_pit.verdict : s.covered_pit.verdict;
    return;
  }

  s.status = 'empty';
  s.verdict = '';
}


function evaluateSectionLogic(config, detections) {
  const detectedClasses = detections.map(d => d.className);
  
  // Check required classes
  const missing = config.requiredClasses.filter(c => !detectedClasses.includes(c));
  if (missing.length > 0) {
    // Make names user-friendly
    const missingNames = missing.map(c => {
      if (c === 'kazam_box') return 'Kazam Box';
      if (c === 'multimeter') return 'Multimeter';
      if (c === 'lcd_screen') return 'LCD screen';
      return formatClassName(c);
    }).join(', ');
    
    return {
      status: 'error',
      verdict: `❌ Could not find: ${missingNames}. Please make sure they are visible and retake.`
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
        verdict: '⚠️ Screen found, but it does not seem to be on the multimeter. Please check.'
      };
    }
  }

  return {
    status: 'done',
    verdict: '✓ All items found perfectly!'
  };
}

// Draw bounding boxes on top of the image using absolutely positioned DIVs
function drawBoundingBoxesDOM(sectionId, detections, customImgId = null, customContainerId = null) {
  const containerId = customContainerId || `wrapper-${sectionId}`;
  const imgId = customImgId || `img-${sectionId}`;
  const container = document.getElementById(containerId);
  const img = document.getElementById(imgId);
  if (!container || !img) return;

  // Clear existing bounding boxes
  container.querySelectorAll('.bbox').forEach(el => el.remove());

  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.box;
    const color = COLORS[det.className] || '#fff';
    
    const boxDiv = document.createElement('div');
    boxDiv.className = 'bbox';
    boxDiv.style.left = `${(x1 / img.naturalWidth) * 100}%`;
    boxDiv.style.top = `${(y1 / img.naturalHeight) * 100}%`;
    boxDiv.style.width = `${((x2 - x1) / img.naturalWidth) * 100}%`;
    boxDiv.style.height = `${((y2 - y1) / img.naturalHeight) * 100}%`;
    boxDiv.style.borderColor = color;

    const labelDiv = document.createElement('div');
    labelDiv.className = 'bbox-label';
    labelDiv.style.backgroundColor = color;
    labelDiv.innerText = state.devMode
      ? `${formatClassName(det.className)} ${(det.conf * 100).toFixed(0)}%`
      : formatClassName(det.className);
    
    boxDiv.appendChild(labelDiv);
    container.appendChild(boxDiv);
  });
}

// ─── AI Pipeline Functions ─────────────────────────────────────────

async function runDetection(img, sectionId) {
  const { tensor, ratios, padParams, resizeCanvasSnap, padCanvasSnap, float32Data } = preprocess(img);
  
  if (state.devMode) {
    updateDebugGallery(img.src, state.lastQCanvasSnap, resizeCanvasSnap, padCanvasSnap, float32Data);
  }
  
  logToDev(`[${sectionId}] Preprocessing complete. Running ONNX inference...`);

  const start = performance.now();
  const feeds = { images: tensor };
  
  const activeSession = sectionId === 'earth_pit' ? potholeSession : multimeterSession;
  if (!activeSession) {
    throw new Error(`ONNX InferenceSession for ${sectionId} is not initialized.`);
  }

  const results = await activeSession.run(feeds);
  const end = performance.now();
  const runtime = Math.round(end - start);

  const output = results[activeSession.outputNames[0]];
  const detections = postprocess(output, ratios, padParams, img.naturalWidth, img.naturalHeight, sectionId);
  
  logToDev(`[${sectionId}] Inference finished in ${runtime}ms. Found ${detections.length} raw boxes.`);

  if (state.devMode) {
    const top20 = getTopDetections(output, ratios, padParams, img.naturalWidth, img.naturalHeight, sectionId, 20);
    
    // Log top 20 to the System logs area
    logToDev(`=== TOP 20 PREDICTIONS (BEFORE FILTERING) ===`);
    top20.forEach((det, idx) => {
      logToDev(`#${idx + 1}: ${det.className} (${(det.conf * 100).toFixed(1)}%) @ [${det.box.map(b => b.toFixed(0)).join(', ')}]`);
    });

  }

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

  // Stage C: Image after resize (No padding)
  const resizeCanvasSnap = document.createElement('canvas');
  resizeCanvasSnap.width = newW;
  resizeCanvasSnap.height = newH;
  resizeCanvasSnap.getContext('2d').drawImage(img, 0, 0, newW, newH);

  ctx.fillStyle = 'rgb(114, 114, 114)';
  ctx.fillRect(0, 0, targetSize, targetSize);
  ctx.drawImage(img, padX, padY, newW, newH);

  // Stage D: Image after padding / letterbox
  const padCanvasSnap = document.createElement('canvas');
  padCanvasSnap.width = targetSize;
  padCanvasSnap.height = targetSize;
  padCanvasSnap.getContext('2d').drawImage(dom.hiddenCanvas, 0, 0);

  const imgData = ctx.getImageData(0, 0, targetSize, targetSize);
  const data = imgData.data;
  
  const float32Data = new Float32Array(3 * targetSize * targetSize);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    float32Data[j] = data[i] / 255.0; // R
    float32Data[j + targetSize * targetSize] = data[i + 1] / 255.0; // G
    float32Data[j + 2 * targetSize * targetSize] = data[i + 2] / 255.0; // B
  }

  const tensor = new ort.Tensor('float32', float32Data, [1, 3, targetSize, targetSize]);
  return { tensor, ratios: [scale, scale], padParams: [padX, padY], resizeCanvasSnap, padCanvasSnap, float32Data };
}

function postprocess(output, ratios, padParams, origW, origH, sectionId) {
  const data = output.data;
  logToDev(`YOLO Output Dims: [${output.dims.join(', ')}]`);
  
  const results = [];
  const CONF_THRESHOLD = sectionId === 'earth_pit' ? 0.25 : 0.40;

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

  // ─── DEBUG: Dump raw values for first 5 boxes (earth_pit only) ───
  if (sectionId === 'earth_pit') {
    logToDev(`[DEBUG] Earth Pit output: ${numBoxes} boxes × ${numChannels} channels, transposed=${isTransposed}`);
    const dumpCount = Math.min(5, numBoxes);
    for (let d = 0; d < dumpCount; d++) {
      const vals = [];
      if (isTransposed) {
        for (let c = 0; c < numChannels; c++) vals.push(data[c * numBoxes + d].toFixed(4));
      } else {
        for (let c = 0; c < numChannels; c++) vals.push(data[d * numChannels + c].toFixed(4));
      }
      logToDev(`[DEBUG] Box ${d}: [${vals.join(', ')}]`);
    }
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
      
      if (sectionId === 'earth_pit' && numChannels === 6) {
        // Regular detection end2end (nms=True) format:
        // [x1, y1, x2, y2, confidence, class_id]
        x1 = data[offset + 0];
        y1 = data[offset + 1];
        x2 = data[offset + 2];
        y2 = data[offset + 3];
        conf = data[offset + 4];
        classId = Math.round(data[offset + 5]);
        
        if (conf < CONF_THRESHOLD) continue;

      } else if (sectionId === 'earth_pit' && numChannels === 7) {
        // Regular detection end2end (nms=True) may include batch index:
        // [batch_idx, x1, y1, x2, y2, class_id, confidence]
        x1 = data[offset + 1];
        y1 = data[offset + 2];
        x2 = data[offset + 3];
        y2 = data[offset + 4];
        classId = Math.round(data[offset + 5]);
        conf = data[offset + 6];
        
        if (conf < CONF_THRESHOLD) continue;

      } else if (numChannels === 7) {
        // OBB end2end (nms=True) format:
        // [x_center, y_center, width, height, conf, class_id, angle]
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

    const classNames = sectionId === 'earth_pit' ? ['earthpit_open', 'earthpit_cover'] : ['lcd_screen', 'multimeter', 'kazam_box'];
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

function getTopDetections(output, ratios, padParams, origW, origH, sectionId, topK = 20) {
  const data = output.data;
  const results = [];
  
  let numBoxes, numChannels, isTransposed;
  if (output.dims.length === 3) {
    if (output.dims[1] < output.dims[2]) {
      numChannels = output.dims[1];
      numBoxes = output.dims[2];
      isTransposed = true;
    } else {
      numBoxes = output.dims[1];
      numChannels = output.dims[2];
      isTransposed = false;
    }
  } else if (output.dims.length === 2) {
    numBoxes = output.dims[0];
    numChannels = output.dims[1];
    isTransposed = false;
  } else {
    return [];
  }

  for (let i = 0; i < numBoxes; i++) {
    let conf = 0;
    let classId = -1;
    let x, y, w, h;
    let x1, y1, x2, y2;

    if (isTransposed) {
      for (let c = 4; c < numChannels; c++) {
        let score = data[c * numBoxes + i];
        if (score > conf) {
          conf = score;
          classId = c - 4;
        }
      }
      x = data[0 * numBoxes + i];
      y = data[1 * numBoxes + i];
      w = data[2 * numBoxes + i];
      h = data[3 * numBoxes + i];
      
      x1 = x - w / 2;
      y1 = y - h / 2;
      x2 = x + w / 2;
      y2 = y + h / 2;
    } else {
      const offset = i * numChannels;
      if (sectionId === 'earth_pit' && numChannels === 6) {
        x1 = data[offset + 0];
        y1 = data[offset + 1];
        x2 = data[offset + 2];
        y2 = data[offset + 3];
        conf = data[offset + 4];
        classId = Math.round(data[offset + 5]);
      } else if (sectionId === 'earth_pit' && numChannels === 7) {
        x1 = data[offset + 1];
        y1 = data[offset + 2];
        x2 = data[offset + 3];
        y2 = data[offset + 4];
        classId = Math.round(data[offset + 5]);
        conf = data[offset + 6];
      } else if (numChannels === 7) {
        x = data[offset + 0];
        y = data[offset + 1];
        w = data[offset + 2];
        h = data[offset + 3];
        conf = data[offset + 4];
        classId = Math.round(data[offset + 5]);
        x1 = x - w / 2;
        y1 = y - h / 2;
        x2 = x + w / 2;
        y2 = y + h / 2;
      } else {
        const objConf = data[offset + 4];
        for (let c = 5; c < numChannels; c++) {
          let score = data[offset + c] * objConf;
          if (score > conf) {
            conf = score;
            classId = c - 5;
          }
        }
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

    // Scale back coordinates
    x1 = (x1 - padParams[0]) / ratios[0];
    y1 = (y1 - padParams[1]) / ratios[1];
    x2 = (x2 - padParams[0]) / ratios[0];
    y2 = (y2 - padParams[1]) / ratios[1];

    x1 = Math.max(0, Math.min(x1, origW));
    y1 = Math.max(0, Math.min(y1, origH));
    x2 = Math.max(0, Math.min(x2, origW));
    y2 = Math.max(0, Math.min(y2, origH));

    const classNames = sectionId === 'earth_pit' ? ['earthpit_open', 'earthpit_cover'] : ['lcd_screen', 'multimeter', 'kazam_box'];
    let className = classNames[classId] || `unknown_${classId}`;

    results.push({
      box: [x1, y1, x2, y2],
      conf,
      classId,
      className
    });
  }

  // Sort by confidence descending
  results.sort((a, b) => b.conf - a.conf);
  return results.slice(0, topK);
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
  
  // Save debug snapshot of Quality Check canvas
  const qCanvasSnap = document.createElement('canvas');
  qCanvasSnap.width = sampleSize;
  qCanvasSnap.height = sampleSize;
  qCanvasSnap.getContext('2d').drawImage(dom.hiddenCanvas, 0, 0);
  state.lastQCanvasSnap = qCanvasSnap;
  
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
    const response = await fetch(SAMPLE_FULL_SETUP_URL);
    const blob = await response.blob();
    const file = new File([blob], "MAH-cparxutx9xf__Meter_Photo_(N-E).jpg", { type: "image/jpeg" });
    const dt = new DataTransfer();
    dt.items.add(file);
    const mockEvent = { target: { files: dt.files } };
    handleFileUpload(mockEvent, 'full_setup');
  });

  const testEarthpitBtn = document.getElementById('testEarthpitBtn');
  if (testEarthpitBtn) {
    testEarthpitBtn.addEventListener('click', async () => {
      dom.devMenu.style.display = 'none';
      const response = await fetch(SAMPLE_EARTHPIT_OPEN_URL);
      const blob = await response.blob();
      const file = new File([blob], "MAH-cparydw2rne__Open_Earthpit.jpg", { type: "image/jpeg" });
      const dt = new DataTransfer();
      dt.items.add(file);
      const mockEvent = { target: { files: dt.files } };
      handleEarthPitUpload(mockEvent, 'open_pit');
    });
  }

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
        if (dom.debugGallery) dom.debugGallery.style.display = 'block';
        logToDev('Dev Mode Enabled.');
      } else {
        dom.devModeToggleBtnHeader.innerText = 'PWA';
        dom.devModeToggleBtnHeader.style.background = 'var(--accent-dim)';
        dom.devModeToggleBtnHeader.style.color = 'var(--accent)';
        dom.devModeToggleBtnHeader.style.borderColor = 'rgba(0, 230, 138, 0.2)';
        if (dom.devStatsCard) dom.devStatsCard.style.display = 'none';
        if (dom.debugGallery) dom.debugGallery.style.display = 'none';
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

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register(assetUrl('sw.js'))
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

window.toggleZoom = function(containerEl) {
  const isZoomed = containerEl.classList.toggle('zoomed');
  const badge = containerEl.querySelector('.zoom-hint-badge');
  if (badge) {
    const isCompact = badge.classList.contains('compact');
    if (isZoomed) {
      badge.innerHTML = isCompact ? '🔍 Shrink' : '🔍 Tap to shrink';
    } else {
      badge.innerHTML = isCompact ? '🔍 Zoom' : '🔍 Tap to zoom';
    }
  }
};

function updateDebugGallery(originalSrc, qCanvas, resizeCanvas, padCanvas, float32Data) {
  if (!dom.debugGalleryContent) return;
  
  dom.debugGalleryContent.innerHTML = '';
  
  // Stage A: Original
  const stageA = document.createElement('div');
  stageA.style.cssText = "background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);";
  stageA.innerHTML = `
    <div style="font-weight: 600; margin-bottom: 4px; color: var(--warning);">Stage A: Original Uploaded Image</div>
    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">Source: File Upload/Camera | Width: Unknown | Height: Unknown</div>
    <img src="${originalSrc}" style="max-width: 100%; border: 1px solid var(--border); border-radius: 4px; max-height: 200px; object-fit: contain; display: block;" />
  `;
  stageA.querySelector('img').onload = function() {
    stageA.querySelector('div:nth-child(2)').innerText = `Source: File/Camera Upload | Width: ${this.naturalWidth}px | Height: ${this.naturalHeight}px`;
  };
  dom.debugGalleryContent.appendChild(stageA);

  // Stage B: Quality Check
  if (qCanvas) {
    const stageB = document.createElement('div');
    stageB.style.cssText = "background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);";
    const canvasB = document.createElement('canvas');
    canvasB.width = qCanvas.width;
    canvasB.height = qCanvas.height;
    canvasB.getContext('2d').drawImage(qCanvas, 0, 0);
    canvasB.style.cssText = "border: 1px solid var(--border); border-radius: 4px; max-height: 200px; display: block; background: #000;";
    stageB.appendChild(document.createRange().createContextualFragment(`
      <div style="font-weight: 600; margin-bottom: 4px; color: var(--warning);">Stage B: Quality-Check Canvas Image</div>
      <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">Source: Downsampled Quality Test | Width: ${qCanvas.width}px | Height: ${qCanvas.height}px</div>
    `));
    stageB.appendChild(canvasB);
    dom.debugGalleryContent.appendChild(stageB);
  }

  // Stage C: Resize
  if (resizeCanvas) {
    const stageC = document.createElement('div');
    stageC.style.cssText = "background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);";
    const canvasC = document.createElement('canvas');
    canvasC.width = resizeCanvas.width;
    canvasC.height = resizeCanvas.height;
    canvasC.getContext('2d').drawImage(resizeCanvas, 0, 0);
    canvasC.style.cssText = "border: 1px solid var(--border); border-radius: 4px; max-height: 200px; display: block; background: #000;";
    stageC.appendChild(document.createRange().createContextualFragment(`
      <div style="font-weight: 600; margin-bottom: 4px; color: var(--warning);">Stage C: Image after Resize (No Padding)</div>
      <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">Source: Proportional Resize to Fit 640px | Width: ${resizeCanvas.width}px | Height: ${resizeCanvas.height}px</div>
    `));
    stageC.appendChild(canvasC);
    dom.debugGalleryContent.appendChild(stageC);
  }

  // Stage D: Pad
  if (padCanvas) {
    const stageD = document.createElement('div');
    stageD.style.cssText = "background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);";
    const canvasD = document.createElement('canvas');
    canvasD.width = padCanvas.width;
    canvasD.height = padCanvas.height;
    canvasD.getContext('2d').drawImage(padCanvas, 0, 0);
    canvasD.style.cssText = "border: 1px solid var(--border); border-radius: 4px; max-height: 200px; display: block; background: #000;";
    stageD.appendChild(document.createRange().createContextualFragment(`
      <div style="font-weight: 600; margin-bottom: 4px; color: var(--warning);">Stage D: Image after Padding / Letterbox</div>
      <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">Source: Letterboxed 640x640 Canvas with rgb(114, 114, 114) | Width: ${padCanvas.width}px | Height: ${padCanvas.height}px</div>
    `));
    stageD.appendChild(canvasD);
    dom.debugGalleryContent.appendChild(stageD);
  }

  // Stage E: Tensor Reconstruction
  if (float32Data) {
    const stageE = document.createElement('div');
    stageE.style.cssText = "background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; border: 1px solid var(--warning); box-shadow: 0 0 10px var(--warning-dim);";
    const canvasE = document.createElement('canvas');
    canvasE.width = 640;
    canvasE.height = 640;
    const ctxE = canvasE.getContext('2d');
    const imgDataE = ctxE.createImageData(640, 640);
    const dataE = imgDataE.data;
    const len = 640 * 640;
    for (let idx = 0; idx < len; idx++) {
      const r = Math.round(float32Data[idx] * 255.0);
      const g = Math.round(float32Data[idx + len] * 255.0);
      const b = Math.round(float32Data[idx + 2 * len] * 255.0);
      const pos = idx * 4;
      dataE[pos] = r;
      dataE[pos + 1] = g;
      dataE[pos + 2] = b;
      dataE[pos + 3] = 255;
    }
    ctxE.putImageData(imgDataE, 0, 0);
    canvasE.style.cssText = "border: 1px solid var(--warning); border-radius: 4px; max-height: 200px; display: block; background: #000;";
    stageE.appendChild(document.createRange().createContextualFragment(`
      <div style="font-weight: 600; margin-bottom: 4px; color: var(--warning);">Stage E: Final Image Reconstructed from Float32 Tensor</div>
      <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">Source: Planar Float32Array [1, 3, 640, 640] normalized /255 | Width: 640px | Height: 640px</div>
    `));
    stageE.appendChild(canvasE);
    dom.debugGalleryContent.appendChild(stageE);
  }
}
