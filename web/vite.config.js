import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { offlineBuild } from './scripts/offline-build.mjs';

// The site is served from a project page, so assets resolve relatively.
export default defineConfig({
  base: './',
  plugins: [svelte(), offlineBuild()],
  build: {
    target: 'es2022',
    outDir: 'dist',
  },
  server: {
    port: 5173,
  },
});
