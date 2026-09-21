/** The packaged execution runtime. The single trained network is remote;
 * its size and digest are declared in model-manifest.js and checked on load. */
export const RUNTIME_FILES = Object.freeze([
  'ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs', 'memory-budget.mjs',
]);
