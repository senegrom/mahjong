/** The packaged execution runtime. The trained networks are remote; each
 * one's size and digest are declared in model-manifest.js and checked on load. */
export const RUNTIME_FILES = Object.freeze([
  'ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs', 'memory-budget.mjs',
]);
