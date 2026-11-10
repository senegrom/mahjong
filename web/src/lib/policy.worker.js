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
// The plain WebAssembly build, which every device runs. The WebGPU runtime,
// many times larger, is imported only where the page asks for the graphics
// card (webgpu.js), from its own folder.
import * as ort from 'onnxruntime-web/wasm';
import { policyWeights } from './policy-weights.js';
import { networkBytes } from './network-store.js';
import { isMemoryError, MEMORY_LIMITS_MIB } from './memory-budget.js';
import { openOnGpu } from './webgpu.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.logLevel = 'error';
let runtimeMemory = null;
// The WebGPU runtime, once imported, and whether the graphics card has
// failed or been too slow in this worker: then it is not tried again, and
// the network plays on WebAssembly until the page starts another worker.
let gpuRuntime = null;
let gpuRefused = false;

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

/** The network as `held` keeps it: its address, the session, the runtime
 * module that opened it, whose tensors it takes, and which runtime that is,
 * 'webgpu' or 'wasm'. scope is the page's own folder, whose Cache Storage
 * holds the network. `webgpu`, where the page asks for the graphics card
 * (webgpu.js), is the WebGPU runtime's folder; the first position asked,
 * `planes`, `mask` and `count`, is what the card is timed on. */
async function load({ url, runtimeBase, scope, memoryLimitMiB, webgpu, planes, mask, count }, note) {
  if (!runtimeMemory) {
    // The bundler renames the runtime's own WebAssembly, which its loader
    // then cannot find. It is served from a folder the page names instead.
    ort.env.wasm.wasmPaths = runtimeBase;
    const controls = new URL('memory-budget.mjs', runtimeBase).href;
    ({ memoryBudget: runtimeMemory } = await import(/* @vite-ignore */ controls));
    runtimeMemory.configure(memoryLimitMiB);
  }
  runtimeMemory.beginRequest();
  if (held?.url === url) return held;
  await release();
  // The bytes rather than the address: the network comes from a bucket on
  // another origin and is kept in Cache Storage, which the runtime knows
  // nothing about. What it is handed has already been hashed, against the
  // listed network at that address.
  const bytes = handed?.url === url ? handed.bytes : await networkBytes({ url, scope,
    onProgress: ({ bytes: received, total }) => note(`downloading the network ${Math.floor(100 * received / total)}%`) });
  handed = null;
  if (webgpu && !gpuRefused) {
    note('starting the network on the graphics card');
    const batch = count ?? 1, width = planes.length / batch, moves = mask.length / batch;
    const first = { planes: planes.slice(0, width), mask: mask.slice(0, moves) };
    const opened = await openOnGpu(bytes, {
      gpu: self.navigator?.gpu,
      // The stock runtime finds its WebAssembly and that file's loader
      // beside itself, in the folder the page names.
      importRuntime: async () => {
        gpuRuntime ??= await import(/* @vite-ignore */ new URL('ort.webgpu.min.mjs', webgpu).href);
        gpuRuntime.env.wasm.wasmPaths = webgpu;
        return gpuRuntime;
      },
      measure: (runtime, session) => forward({ ort: runtime, session }, first.planes, first.mask, null, false),
    });
    if (opened.session) {
      held = { url, session: opened.session, ort: opened.runtime, runtime: 'webgpu' };
      return held;
    }
    // Failed or slow, the card is not tried again by this worker.
    gpuRefused = true;
  }
  note('starting the network');
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  held = { url, session, ort, runtime: 'wasm' };
  return held;
}

/** Releases the network held, if any. */
async function release() {
  if (!held) return;
  const { session } = held;
  held = null;
  await session.release();
}

/** One forward pass of `model` over the positions in `planes` and `mask`,
 * laid end to end, `count` of them for a review's batch or null for one
 * decision, and the answer the page is sent for it. Every tensor is
 * disposed, the answer copied out of them first. */
async function forward(model, planes, mask, count, details) {
  const { ort: runtime, session } = model;
  let input, output, allowed;
  try {
    // A review sends several positions at once, their planes and masks laid
    // end to end, and the graph takes them as one batch. Natively eight cost
    // about half as much a position as one at a time; in this one-thread
    // runtime about the same, so a batch saves messages and costs nothing.
    // On a graphics card sixty-four cost about as much as one.
    const batch = count ?? 1;
    input = new runtime.Tensor('float32', planes, [batch, planes.length / batch / POSITIONS, POSITIONS]);
    // The fused network's head reads the legality mask, so the graph takes it
    // too. An older single-input network is still served the planes alone.
    allowed = session.inputNames.includes('legal')
      ? new runtime.Tensor('float32', Float32Array.from(mask, can => (can ? 1 : 0)), [batch, mask.length / batch])
      : null;
    output = await session.run(allowed ? { planes: input, legal: allowed } : { planes: input });
    if (count != null) {
      // Each position's best legal move and weights, which is all a review
      // reads; the value and the belief are left in the tensors.
      const width = mask.length / count;
      return { analyses: Array.from({ length: count }, (_, index) => policyWeights(
        output.policy.data.slice(index * width, (index + 1) * width), mask.slice(index * width, (index + 1) * width))) };
    }
    // Its action is the legal entry the policy rates highest: opponents play
    // their best move.
    const analysis = policyWeights(output.policy.data, mask);
    // The network answers all three questions in one pass, so the value and
    // the belief cost nothing beyond copying them out, which is done only for
    // an adviser that shows them. They are copied because the tensors are
    // disposed below.
    return { action: analysis.action, ...(details ? { analysis: { ...analysis,
      value: output.value ? output.value.data[0] : null,
      hands: output.hands ? Float32Array.from(output.hands.data) : null,
    } } : {}) };
  } finally {
    input?.dispose();
    allowed?.dispose();
    for (const tensor of Object.values(output ?? {})) tensor.dispose();
  }
}

async function infer({ id, url, runtimeBase, scope, planes, mask, count, details, webgpu, memoryLimitMiB = MEMORY_LIMITS_MIB[0] }) {
  // Each note is also a sign of life: while the network loads, the page
  // times only the silence between notes. 'network ready' is said as each
  // request begins to run, loaded just now or long before, and the page
  // times the request's deadline from it, not from when it was sent. It says
  // which runtime the network plays on.
  const note = progress => self.postMessage({ id, progress });
  const request = { url, runtimeBase, scope, memoryLimitMiB, webgpu, planes, mask, count };
  note('loading the network');
  let model = await load(request, note);
  self.postMessage({ id, progress: 'network ready', ready: true, runtime: model.runtime });
  let answer;
  try {
    answer = await forward(model, planes, mask, count, details);
  } catch (error) {
    if (model.runtime !== 'webgpu') throw error;
    // The graphics card failed, as a lost device does: the network goes to
    // WebAssembly, read again from where it is kept, and the position is
    // asked there, within this request's deadline. The page hears of it in
    // a message of its own.
    gpuRefused = true;
    try { await release(); } catch { held = null; }
    model = await load(request, note);
    self.postMessage({ runtime: model.runtime });
    answer = await forward(model, planes, mask, count, details);
  }
  self.postMessage({ id, ...answer });
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
