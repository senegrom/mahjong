import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { offlineBuild } from './scripts/offline-build.mjs';

// The site is served from a project page, so assets resolve relatively.
export default defineConfig({
  base: './',
  plugins: [svelte(), offlineBuild()],
  // Select the external-WASM entry instead of bundling the generic ORT binary.
  resolve: {
    conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'],
  },
  build: {
    target: 'es2022',
    outDir: 'dist',
    // offlineBuild copies only the runtime inventory, preserving approved SVGs.
    copyPublicDir: false,
  },
  server: {
    port: 5173,
    host: process.env.VITE_LAN ? '0.0.0.0' : '127.0.0.1',
    allowedHosts: ['terminal.local'],
  },
});
