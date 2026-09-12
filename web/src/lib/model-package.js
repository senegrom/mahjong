/** Files shared by runtime verification, offline packaging and inference. */
export const MODEL_FILES = Object.freeze({ full: 'model-full.onnx' });
export const RUNTIME_FILES = Object.freeze([
  'ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs', 'memory-budget.mjs',
]);
