import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import fs from 'node:fs';

const ortWasmSource = fs.existsSync('node_modules/onnxruntime-web/dist')
  ? 'node_modules/onnxruntime-web/dist/ort-wasm-*'
  : '../node_modules/onnxruntime-web/dist/ort-wasm-*';

export default defineConfig({
  base: './',
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: ortWasmSource,
          dest: 'wasm',
          rename: { stripBase: true },
        },
      ],
    }),
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    allowedHosts: true,
  },
});
