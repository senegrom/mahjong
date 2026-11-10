/** The packaged execution runtime. The trained networks are remote; each
 * one's size and digest are declared in model-manifest.js and checked on load. */
export const RUNTIME_FILES = Object.freeze([
  'ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs', 'memory-budget.mjs',
]);
/** The WebGPU runtime: onnxruntime-web's own files, unchanged, from the
 * package the page already depends on. The worker imports the first, which
 * loads the other two from beside it, and only on a device it plays on the
 * graphics card (webgpu.js); its 29.5 MB are not part of the offline AI
 * package (scripts/offline-build.mjs). */
export const WEBGPU_RUNTIME_FILES = Object.freeze([
  'ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm',
]);
