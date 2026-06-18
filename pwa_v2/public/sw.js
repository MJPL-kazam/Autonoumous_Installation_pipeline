const CACHE_NAME = 'kazam-roi-cache-v12';
const SCOPE_URL = new URL(self.registration.scope);
const BASE_PATH = SCOPE_URL.pathname.replace(/\/$/, '');
const scopedPath = (path = '') => `${BASE_PATH}/${path}`.replace(/\/{2,}/g, '/');
const APP_SHELL = [
  '',
  'index.html',
  'manifest.json',
  'icon-512.png',
  'model/multimeter_t400v100.onnx',
  'model/pothole_t417v100.onnx',
  'test-images/MAH-cparxutx9xf__Meter_Photo_(N-E).jpg',
  'test-images/MAH-cparydw2rne__Open_Earthpit.jpg',
  'test-images/MAH-cparyhksyoe__Earthpit.jpg',
  'wasm/ort-wasm-simd-threaded.wasm',
  'wasm/ort-wasm-simd-threaded.jsep.wasm',
  'wasm/ort-wasm-simd-threaded.mjs',
  'wasm/ort-wasm-simd-threaded.jsep.mjs',
].map(scopedPath);

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => Promise.all(
        cacheNames
          .filter((cache) => cache !== CACHE_NAME)
          .map((cache) => caches.delete(cache))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('message', (event) => {
  if (event.data?.type === 'CLEAR_APP_CACHE') {
    event.waitUntil(
      caches.keys()
        .then((cacheNames) => Promise.all(cacheNames.map((cache) => caches.delete(cache))))
        .then(() => self.clients.matchAll())
        .then((clients) => clients.forEach((client) => client.postMessage({ type: 'APP_CACHE_CLEARED' })))
    );
  }
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);
  if (
    url.pathname.includes('@vite') ||
    url.pathname.includes('hot-update') ||
    url.pathname.includes('chrome-extension')
  ) {
    return;
  }

  if (event.request.mode === 'navigate' || url.pathname === '/' || url.pathname.endsWith('.html')) {
    event.respondWith(networkFirst(event.request));
    return;
  }

  if (url.pathname.endsWith('.js') || url.pathname.endsWith('.css')) {
    event.respondWith(networkFirstAsset(event.request));
    return;
  }

  event.respondWith(cacheFirstAsset(event.request));
});

async function networkFirst(request) {
  try {
    const response = await fetch(request, { cache: 'no-store' });
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    return cached || caches.match(scopedPath());
  }
}

async function networkFirstAsset(request) {
  try {
    const response = await fetch(request, { cache: 'no-store' });
    if (isValidAssetResponse(request, response)) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
      return response;
    }
  } catch {
    // Fall through to cache lookup.
  }

  const cached = await caches.match(request);
  if (cached) return cached;

  return new Response('', {
    status: 404,
    statusText: 'Asset not found',
    headers: { 'Content-Type': 'text/plain' },
  });
}

async function cacheFirstAsset(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (isValidAssetResponse(request, response)) {
    const cache = await caches.open(CACHE_NAME);
    cache.put(request, response.clone());
  }
  return response;
}

function isValidAssetResponse(request, response) {
  if (!response.ok) return false;

  const url = new URL(request.url);
  const contentType = response.headers.get('content-type') || '';

  if (url.pathname.endsWith('.js')) {
    return contentType.includes('javascript') || contentType.includes('ecmascript');
  }

  if (url.pathname.endsWith('.css')) {
    return contentType.includes('text/css');
  }

  return !contentType.includes('text/html');
}
