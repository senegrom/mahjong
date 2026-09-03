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

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.logLevel = 'error';
let configured = false;

const PLANES = 93;
const POSITIONS = 34;

let session = null;
let loading = null;

async function load(url, runtimeBase) {
  if (!configured && runtimeBase) {
    // The bundler renames the runtime's own WebAssembly, which its loader
    // then cannot find. It is served from a known folder instead.
    ort.env.wasm.wasmPaths = runtimeBase;
    configured = true;
  }
  if (!session && !loading) {
    loading = ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }).then((created) => {
      session = created;
      loading = null;
      return created;
    });
  }
  return session ?? loading;
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

self.onmessage = async (event) => {
  const { id, url, runtimeBase, planes, mask, temperature } = event.data;
  try {
    self.postMessage({ id, progress: 'loading the network' });
    const model = await load(url, runtimeBase);
    self.postMessage({ id, progress: 'network ready' });
    const input = new ort.Tensor('float32', planes, [1, PLANES, POSITIONS]);
    const output = await model.run({ planes: input });
    const logits = output.policy.data;
    self.postMessage({ id, action: pick(logits, mask, temperature ?? 0) });
  } catch (error) {
    self.postMessage({ id, error: String(error) });
  }
};
