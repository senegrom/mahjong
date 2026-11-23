/**
 * The trained opponent, kept off the main thread.
 *
 * The page holds the rules and the table; this worker holds only the policy.
 * It is sent a position, as the planes the engine produced and the mask of
 * what the rules allow, or a review's batch of them, and answers with the
 * entry of the action space it would choose. Nothing about the rules lives
 * here: an illegal answer is impossible because the mask decides what may be
 * picked.
 */
// The plain WebAssembly build. Play asks one position at a time, which
// gains nothing from a graphics card, and the WebGPU build's runtime is
// 29.9 MB to this one's four. A review's batches do run on WebGPU, on
// desktops, in a worker of their own (gpu-batch.worker.js).
import * as ort from 'onnxruntime-web/wasm';
import { policyWeights } from './policy-weights.js';
import { networkBytes } from './network-store.js';
import { networkAt } from './trained-network.js';
import { unpackNetwork } from './network-pack.js';
import { isMemoryError, MEMORY_LIMITS_MIB } from './memory-budget.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.logLevel = 'error';
let runtimeMemory = null;

// Positions in a plane: the 34 tile kinds. How many planes there are is
// whatever the engine sent, so neither the worker nor the model has to be
// told when the observation grows.
const POSITIONS = 34;

// A page plays one trained network, and its session is kept loaded
// between decisions: reloading it every turn cost a table most of a second
// at the ninetieth percentile on a desktop and far more on a phone. A
// runtime that cannot hold it fails the request and the page retries it in
// a fresh worker with a larger reservation. The session is held with the
// address it was loaded from, so a request naming another network, as the
// worker parity check's two models do, releases it before loading its own.
let held = null;
const queued = new Map();
let running = false;
let failed = false;
// Verified network bytes the page downloaded but could not store: used once,
// instead of downloading the same network again.
let handed = null;

/** scope is the page's own folder, whose Cache Storage holds the network. */
async function load({ url, runtimeBase, scope, memoryLimitMiB }, note) {
  if (!runtimeMemory) {
    // The bundler renames the runtime's own WebAssembly, which its loader
    // then cannot find. It is served from a folder the page names instead.
    ort.env.wasm.wasmPaths = runtimeBase;
    const controls = new URL('memory-budget.mjs', runtimeBase).href;
    ({ memoryBudget: runtimeMemory } = await import(/* @vite-ignore */ controls));
    runtimeMemory.configure(memoryLimitMiB);
  }
  runtimeMemory.beginRequest();
  if (held?.url === url) return held.session;
  if (held) {
    const { session } = held;
    held = null;
    await session.release();
  }
  // The bytes rather than the address: the network comes from a bucket on
  // another origin and is kept in Cache Storage, which the runtime knows
  // nothing about. What it is handed has already been hashed, against the
  // listed network at that address. Another window that is at the same
  // network goes first (network-transfer.js): the wait is said, which is
  // also the sign of life the page times this worker's silence by.
  let bytes = handed?.url === url ? handed.bytes : await networkBytes({ url, scope,
    onProgress: ({ bytes: received, total }) => note(`downloading the network ${Math.floor(100 * received / total)}%`),
    onWaiting: () => note('waiting for another window to download the network') });
  handed = null;
  // A packed network is widened here, off the page's thread, into the whole
  // file the runtime reads (network-pack.js); the packed bytes are let go
  // before the session is made, so that the two are held together only
  // while it is widened, 59 and 117 MB for network 3000.
  const listed = networkAt(url);
  const file = listed?.packing ? unpackNetwork(bytes, listed.bytes) : bytes;
  bytes = null;
  note('starting the network');
  const session = await ort.InferenceSession.create(file, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  held = { url, session };
  return session;
}

async function infer({ id, url, runtimeBase, scope, planes, mask, count, details, memoryLimitMiB = MEMORY_LIMITS_MIB[0] }) {
  let input, output, allowed;
  try {
    // Each note is also a sign of life: while the network loads, the page
    // times only the silence between notes. 'network ready' is said as each
    // request begins to run, loaded just now or long before, and the page
    // times the request's deadline from it, not from when it was sent.
    const note = progress => self.postMessage({ id, progress });
    note('loading the network');
    const model = await load({ url, runtimeBase, scope, memoryLimitMiB }, note);
    self.postMessage({ id, progress: 'network ready', ready: true });
    // A review sends several positions at once, their planes and masks laid
    // end to end, and the graph takes them as one batch. Natively eight cost
    // about half as much a position as one at a time; in this one-thread
    // runtime about the same, so a batch saves messages and costs nothing.
    const batch = count ?? 1;
    input = new ort.Tensor('float32', planes, [batch, planes.length / batch / POSITIONS, POSITIONS]);
    // The fused network's head reads the legality mask, so the graph takes it
    // too. An older single-input network is still served the planes alone.
    allowed = model.inputNames.includes('legal')
      ? new ort.Tensor('float32', Float32Array.from(mask, can => (can ? 1 : 0)), [batch, mask.length / batch])
      : null;
    output = await model.run(allowed ? { planes: input, legal: allowed } : { planes: input });
    if (count != null) {
      // Each position's best legal move and weights, which is all a review
      // reads; the value and the belief are left in the tensors.
      const width = mask.length / count;
      self.postMessage({ id, analyses: Array.from({ length: count }, (_, index) => policyWeights(
        output.policy.data.slice(index * width, (index + 1) * width), mask.slice(index * width, (index + 1) * width))) });
      return;
    }
    // Its action is the legal entry the policy rates highest: opponents play
    // their best move.
    const analysis = policyWeights(output.policy.data, mask);
    // The network answers all three questions in one pass, so the value and
    // the belief cost nothing beyond copying them out, which is done only for
    // an adviser that shows them. They are copied because the tensors are
    // disposed below.
    self.postMessage({ id, action: analysis.action, ...(details ? { analysis: { ...analysis,
      value: output.value ? output.value.data[0] : null,
      hands: output.hands ? Float32Array.from(output.hands.data) : null,
    } } : {}) });
  } finally {
    input?.dispose();
    allowed?.dispose();
    for (const tensor of Object.values(output ?? {})) tensor.dispose();
  }
}

async function drain() {
  if (running || failed) return;
  running = true;
  try {
    while (queued.size) {
      const [id, request] = queued.entries().next().value;
      queued.delete(id);
      try { await infer(request); }
      catch (error) {
        // An aborted ORT runtime cannot be initialized again in this worker.
        // The page discards it, even if this particular request was cancelled.
        failed = true;
        queued.clear();
        self.postMessage({ id, error: String(error),
          ...(isMemoryError(error) && runtimeMemory?.failure
            ? { memory: { ...runtimeMemory.failure, limitMiB: runtimeMemory.limitMiB } } : {}),
        });
        break;
      }
    }
  } finally { running = false; }
}

self.onmessage = ({ data, source }) => {
  // Only the page that started this worker posts to it, through the worker's
  // own port, which names no sender.
  if (source != null) return;
  if (data.cancel != null) { queued.delete(data.cancel); return; }
  // Kept apart from its request, which may yet be cancelled.
  const { network, ...request } = data;
  if (network) handed = { url: request.url, bytes: network };
  queued.set(request.id, request);
  return drain();
};
