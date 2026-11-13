/**
 * The trained network on the graphics card, for a review's batches only.
 *
 * It runs nothing but ONNX Runtime Web's WebGPU provider, from onnxruntime-web's
 * own files in the page's hashed `ort-webgpu/<hash>/` folder, and it has no
 * WebAssembly of its own to fall back on: the page plays and reviews on the
 * WebAssembly worker (policy.worker.js) and sends batches here only once this
 * worker has answered as that one does, faster (gpu-batches.js). Whatever
 * fails here is said to the page, which ends this worker.
 *
 * It is asked, in turn:
 *  - `{ id, open: { url, expect, scope, webgpu } }`: open the network stored
 *    at `url` in the Cache Storage of the page's folder `scope`, checked
 *    against `expect` (its size and SHA-256), with the runtime in the folder
 *    `webgpu`. It never downloads the network: one not stored is refused.
 *  - `{ id, planes, mask, count }`: a batch of positions laid end to end, as
 *    the WebAssembly worker is asked one.
 * It answers `{ id, opened: true }`, `{ id, analyses, ms }` (each position's
 * best legal move and weights, and how long the run took), or `{ id, error,
 * remember }`, where `remember` says whether the failure is the device's and
 * worth remembering across visits, or this visit's only (a runtime the
 * browser could not fetch, a network not stored). While it opens a network
 * it says `{ id, fetched: true }` once, when the runtime is in hand: what
 * came before went at the connection's pace, and what follows is the
 * device's time (gpu-batches.js).
 */
import { policyWeights } from './policy-weights.js';
import { storedNetworkBytes } from './network-transfer.js';
import { webgpuAdapter } from './webgpu.js';

// Positions in a plane: the 34 tile kinds.
const POSITIONS = 34;
let runtime = null;
let session = null;

/** A failure as the page is told it. */
class Refusal extends Error {
  constructor(message, remember) { super(message); this.remember = remember; }
}
const said = error => String(error?.message ?? error);

async function open({ url, expect, scope, webgpu }, fetched) {
  const adapter = await webgpuAdapter(self.navigator?.gpu);
  if (!adapter) throw new Refusal('no graphics card adapter, or only one that runs on the processor', true);
  let bytes;
  try { bytes = await storedNetworkBytes({ url, expect, scope }); }
  catch (error) { throw new Refusal(`the network could not be read: ${said(error)}`, false); }
  if (!bytes) throw new Refusal('the network is not saved on this device', false);
  // The runtime's own files, fetched here first so that one the browser
  // cannot fetch, offline before it was ever saved, is told apart from a
  // runtime that fails on this device. Its WebAssembly is kept and handed to
  // the runtime, which would fetch it again otherwise: 30 MB a second time
  // wherever the browser could not save the first.
  let binary;
  try {
    for (const file of ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm']) {
      const response = await fetch(new URL(file, webgpu));
      if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
      const body = await response.arrayBuffer();
      if (file.endsWith('.wasm')) binary = body;
    }
    runtime = await import(/* @vite-ignore */ new URL('ort.min.mjs', webgpu).href);
  } catch (error) { throw new Refusal(`the WebGPU runtime could not be fetched: ${said(error)}`, false); }
  fetched();
  runtime.env.wasm.wasmPaths = webgpu;
  runtime.env.wasm.wasmBinary = binary;
  runtime.env.wasm.numThreads = 1;
  runtime.env.logLevel = 'error';
  runtime.env.webgpu.adapter = adapter;
  try {
    // Errors alone are logged: opening any graph, the runtime warns that the
    // nodes it computes on the processor by design, a Shape's and those fed
    // by one, were not given to WebGPU, twice, in the console.
    session = await runtime.InferenceSession.create(bytes, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
      logSeverityLevel: 3 });
  } catch (error) { throw new Refusal(`the network could not be opened on the card: ${said(error)}`, true); }
  // The runtime is running by now, and has no more use for its 30 MB.
  finally { runtime.env.wasm.wasmBinary = undefined; }
}

async function run({ planes, mask, count }) {
  if (!session) throw new Refusal('the network is not open on the card', true);
  let input, allowed, output;
  try {
    input = new runtime.Tensor('float32', planes, [count, planes.length / count / POSITIONS, POSITIONS]);
    allowed = session.inputNames.includes('legal')
      ? new runtime.Tensor('float32', Float32Array.from(mask, can => (can ? 1 : 0)), [count, mask.length / count])
      : null;
    const began = performance.now();
    output = await session.run(allowed ? { planes: input, legal: allowed } : { planes: input });
    const ms = performance.now() - began;
    const width = mask.length / count;
    return { ms, analyses: Array.from({ length: count }, (_, index) => policyWeights(
      output.policy.data.slice(index * width, (index + 1) * width), mask.slice(index * width, (index + 1) * width))) };
  } catch (error) {
    throw error instanceof Refusal ? error : new Refusal(`a batch failed on the card: ${said(error)}`, true);
  } finally {
    input?.dispose();
    allowed?.dispose();
    for (const tensor of Object.values(output ?? {})) tensor.dispose();
  }
}

// One request at a time, in the order sent.
let turn = Promise.resolve();
self.onmessage = ({ data, source }) => {
  // Only the page that started this worker posts to it, through its own port.
  if (source != null) return;
  turn = turn.then(async () => {
    try {
      if (data.open) {
        await open(data.open, () => self.postMessage({ id: data.id, fetched: true }));
        self.postMessage({ id: data.id, opened: true });
      } else self.postMessage({ id: data.id, ...await run(data) });
    } catch (error) {
      self.postMessage({ id: data.id, error: said(error), remember: error instanceof Refusal ? error.remember : true });
    }
  });
};
