# Mobile Testing Guide

This PWA can run model inference on mobile browsers through ONNX Runtime Web and WebAssembly. Mobile PWA behavior depends heavily on secure context rules, so use HTTPS for realistic testing.

## Recommended: HTTPS Tunnel

Start the development server:

```bash
npm run dev -- --host
```

Expose the local server with a tunnel:

```bash
npx localtunnel --port 5173
```

Alternative:

```bash
ngrok http 5173
```

Open the generated HTTPS URL on the phone.

## Install As PWA

Android Chrome:

1. Open the HTTPS URL.
2. Open the browser menu.
3. Select Install app or Add to Home screen.

iOS Safari:

1. Open the HTTPS URL.
2. Tap Share.
3. Tap Add to Home Screen.

## Local Network Testing

For quick LAN testing:

```bash
npm run dev -- --host
```

Open the network URL shown by Vite, for example:

```text
http://<local-ip>:5173/
```

Service workers require HTTPS or localhost. If testing over plain HTTP on Android Chrome, you may need to add the local origin under:

```text
chrome://flags/#unsafely-treat-insecure-origin-as-secure
```

Use HTTPS tunnel testing when validating installability, offline cache behavior, and camera permissions.

## Cache Reset During Testing

If an older service worker is active, clear it from the browser console:

```js
navigator.serviceWorker.getRegistrations()
  .then((registrations) => Promise.all(registrations.map((registration) => registration.unregister())))
  .then(() => caches.keys())
  .then((keys) => Promise.all(keys.map((key) => caches.delete(key))))
  .then(() => location.reload());
```

The app also has a developer menu with a cache-clear action.

## Production Preview

Build the app:

```bash
npm run build
```

Preview locally:

```bash
npm run preview -- --host
```

Deploy the generated `dist/` folder to an HTTPS static host for final technician testing.
