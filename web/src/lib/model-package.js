/** Files shared by runtime verification, offline packaging and inference.
 *
 * The trained network is not among them. It is 116 MB, which is more than
 * GitHub stores in one file and more than Pages serves, so it comes from the
 * bucket named in `model-manifest.json` and is kept in Cache Storage; see
 * `network-store.js`. What ships here is the runtime that runs it. */
export const MODEL_FILES = Object.freeze({});
export const RUNTIME_FILES = Object.freeze([
  'ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs', 'memory-budget.mjs',
]);
