import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { offlineBuild } from './scripts/offline-build.mjs';

// The site is served from a project page, so assets resolve relatively.
export default defineConfig({
  base: './',
  plugins: [svelte(), offlineBuild()],
  // The Trained model stays ONNX, but its execution runtime is a model-specific
  // reduced build. Use the package's external-WASM JS entry so the generic
  // runtime is not bundled into dist/assets as a second download.
  resolve: {
    conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'],
  },
  build: {
    target: 'es2022',
    outDir: 'dist',
  },
  server: {
    port: 5173,
    // A phone on the same network can be given the server with
    // `VITE_LAN=1 npm run dev`; by default it answers only this machine.
    host: process.env.VITE_LAN ? '0.0.0.0' : '127.0.0.1',
    allowedHosts: ['terminal.local'],
  },
});
