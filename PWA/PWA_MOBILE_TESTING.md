# Testing Kazam ROI Detection PWA on Mobile Devices

The Progressive Web App (PWA) executes heavy deep learning inference (`t400v100.onnx`) directly on the device using WebAssembly (WASM). To test it on your mobile device, ensure you satisfy the following requirements:

1. **Network Connectivity**: Your phone must be on the same local network, or you must use a tunneling service.
2. **Secure Context (HTTPS)**: Mobile browsers require HTTPS (or `localhost`) to register Service Workers (`sw.js`). Without HTTPS, offline installation ("Add to Home Screen") will not function and the model will not cache properly.

---

## Method 1: Secure Tunnel (Recommended)

Using a tunnel (like `localtunnel` or `ngrok`) exposes your local Vite dev server via an HTTPS link. This is the easiest way to test service workers and installation on both Android and iOS.

1. **Start the local Vite server**:
   ```bash
   cd E:\2026_kazam\CV_mini3_3_screen_detection\PWA
   npm run dev
   ```

2. **Expose the port using localtunnel** (in a new terminal):
   ```bash
   npx localtunnel --port 5173
   ```
   *Alternative using ngrok:*
   ```bash
   ngrok http 5173
   ```

3. **Open the HTTPS URL** on your mobile device.
4. **Install as PWA**:
   - **iOS Safari**: Tap the **Share** button -> scroll down -> tap **Add to Home Screen**.
   - **Android Chrome**: Tap the three dots menu -> tap **Install app** or **Add to Home screen**.

---

## Method 2: Local Area Network (LAN)

If using a tunnel is not possible, you can access the server via your local IP address. Since it's HTTP, you must manually bypass secure context limitations.

1. **Start Vite with network exposed**:
   ```bash
   cd E:\2026_kazam\CV_mini3_3_screen_detection\PWA
   npm run dev -- --host
   ```
   Note the `Network: http://<your-ip>:5173/` address.

2. **Bypass Secure Context on Android Chrome**:
   - Go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.
   - Enable the flag.
   - Enter your local URL (e.g., `http://192.168.1.15:5173`).
   - Relaunch Chrome. The Service Worker will now successfully cache `t400v100.onnx`.

*(Note: Bypassing secure context on iOS Safari via HTTP is highly restricted. Use Method 1 for iOS).*

---

## Method 3: Production Build Deployment

To test a highly optimized build of the PWA suitable for real-world technician testing, deploy the static output.

1. **Build the production assets**:
   ```bash
   npm run build
   ```
   This packages the JS, CSS, and ONNX models into `PWA/dist/`.

2. **Host the static files**:
   - Upload the `dist/` folder to a secure static host (e.g., Netlify, Vercel, or GitHub Pages).
   - Once deployed, technicians can navigate to the secure HTTPS link, allow camera access, and install the application directly.
