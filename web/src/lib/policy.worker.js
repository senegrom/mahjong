/**
 * The trained opponent, kept off the main thread.
 *
 * The page holds the rules and the table; this worker holds only the policy.
 * It is sent one position at a time, as the planes the engine produced and
 * the mask of what the rules allow, and answers with the entry of the action
 * space it would choose. Nothing about the rules lives here: an illegal
 * answer is impossible because the mask decides what may be picked.
 */
// The plain WebAssembly build: the WebGPU one carries a runtime many
// times larger, for a network this small to gain nothing from.
import * as ort from 'onnxruntime-web/wasm';
import { policyWeights } from './policy-weights.js';
import { isMemoryError, MEMORY_LIMITS_MIB } from './memory-budget.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.logLevel = 'error';
let runtimeMemory = null;

// Positions in a plane: the 34 tile kinds. How many planes there are is
// whatever the engine sent, so neither the worker nor the model has to be
// told when the observation grows.
const POSITIONS = 34;

// A session for each network the page carries, kept in order of last use,
// so a table that mixes the two does not reload one on every change of
// turn: measured, that cost a mixed table most of a second at the ninetieth
// percentile on a desktop and far more on a phone. Two is what exists; a
// runtime that cannot hold both fails the request and the page retries it
// in a fresh worker with a larger reservation.
const KEEP = 2;
const sessions = new Map();
const queued = new Map();
let running = false;
let failed = false;

async function load(url, runtimeBase, memoryLimitMiB) {
  if (!runtimeMemory) {
    // The bundler renames the runtime's own WebAssembly, which its loader
    // then cannot find. It is served from a known folder instead.
    ort.env.wasm.wasmPaths = runtimeBase;
    const controls = new URL('memory-budget.mjs', runtimeBase).href;
    ({ memoryBudget: runtimeMemory } = await import(/* @vite-ignore */ controls));
    runtimeMemory.configure(memoryLimitMiB);
  }
  runtimeMemory.beginRequest();
  const ready = sessions.get(url);
  if (ready) {
    sessions.delete(url);
    sessions.set(url, ready);
    return ready;
  }
  while (sessions.size >= KEEP) {
    const [oldest, session] = sessions.entries().next().value;
    sessions.delete(oldest);
    await session.release();
  }
  const session = await ort.InferenceSession.create(url, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  sessions.set(url, session);
  return session;
}

/**
 * Picks among the legal entries. Early in a hand the choice is sampled from
 * the policy's own odds so the opponents do not all play the same game;
 * later it takes the best it knows.
 */
function pick(logits, mask, temperature) {
  let best = -1;
  let bestValue = -Infinity;
  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index]) continue;
    if (logits[index] > bestValue) {
      bestValue = logits[index];
      best = index;
    }
  }
  if (temperature <= 0 || best < 0) return best;

  let total = 0;
  const weights = new Float64Array(mask.length);
  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index]) continue;
    const weight = Math.exp((logits[index] - bestValue) / temperature);
    weights[index] = weight;
    total += weight;
  }
  let target = Math.random() * total;
  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index]) continue;
    target -= weights[index];
    if (target <= 0) return index;
  }
  return best;
}

async function infer({ id, url, runtimeBase, planes, mask, temperature, details, memoryLimitMiB = MEMORY_LIMITS_MIB[0] }) {
  let input, output;
  try {
    self.postMessage({ id, progress: 'loading the network' });
    const model = await load(url, runtimeBase, memoryLimitMiB);
    self.postMessage({ id, progress: 'network ready' });
    input = new ort.Tensor('float32', planes, [1, planes.length / POSITIONS, POSITIONS]);
    output = await model.run({ planes: input });
    const logits = output.policy.data;
    const analysis = policyWeights(logits, mask);
    self.postMessage({ id, action: pick(logits, mask, temperature ?? 0), ...(details ? { analysis } : {}) });
  } finally {
    input?.dispose();
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

self.onmessage = ({ data }) => {
  if (data.cancel != null) { queued.delete(data.cancel); return; }
  queued.set(data.id, data);
  return drain();
};
