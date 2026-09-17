// Real ONNX graphs, real ORT/reduced WASM, and the production worker source.
// Only browser transport and model-byte fetching are adapted for Node. No
// session.run, logits, legality input or tensor disposal is mocked.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import vm from 'node:vm';
import * as ort from 'onnxruntime-web/wasm';
import { policyWeights } from '../src/lib/policy-weights.js';
import { MEMORY_LIMITS_MIB, isMemoryError } from '../src/lib/memory-budget.js';

const folder = resolve(process.argv[2]);
const cases = JSON.parse(await readFile(resolve(folder, 'inputs.json'), 'utf8'));
assert.ok(cases.some(row => row.takes_legal && row.stage === 'riichi-discard'));
assert.ok(cases.some(row => !row.takes_legal));
const runtimeBase = new URL('../public/ort/', import.meta.url).href;
const source = (await readFile(new URL('../src/lib/policy.worker.js', import.meta.url), 'utf8'))
  .replace("import * as ort from 'onnxruntime-web/wasm';", '')
  .replace("import { policyWeights } from './policy-weights.js';", '')
  .replace("import { isMemoryError, MEMORY_LIMITS_MIB } from './memory-budget.js';", '')
  .replace("import { networkBytes } from './network-store.js';", '')
  .replace('import(/* @vite-ignore */ controls)', 'loadMemoryControls(controls)');
const tensors = [], sessions = [], messages = [];
let current;
function track(tensor) {
  const entry = { disposed: false };
  tensors.push(entry);
  const dispose = tensor.dispose.bind(tensor);
  tensor.dispose = () => {
    assert.equal(entry.disposed, false, 'each tensor is disposed once');
    entry.disposed = true;
    dispose();
  };
  return tensor;
}
function close(actual, expected, label) {
  assert.ok(Number.isFinite(actual), `${label}: nonfinite output`);
  assert.ok(Math.abs(actual - expected) <= 0.0001 * (1 + Math.abs(expected)),
    `${label}: ${actual} differs from ${expected}`);
}
const bridge = {
  env: ort.env,
  Tensor: function(type, data, dims) { return track(new ort.Tensor(type, data, dims)); },
  InferenceSession: { create: async (bytes, options) => {
    const session = await ort.InferenceSession.create(bytes, options);
    sessions.push(session);
    return {
      inputNames: session.inputNames,
      release: () => session.release(),
      run: async feed => {
        assert.deepEqual(Object.keys(feed), current.takes_legal ? ['planes', 'legal'] : ['planes']);
        if (current.takes_legal) {
          assert.equal(feed.legal.type, 'float32');
          assert.deepEqual(Array.from(feed.legal.dims), [1, 46]);
          assert.deepEqual(Array.from(feed.legal.data), current.mask.map(value => value ? 1 : 0));
        }
        const result = await session.run(feed);
        for (const tensor of Object.values(result)) track(tensor);
        for (let i = 0; i < 46; i++) {
          if (current.mask[i]) close(result.policy.data[i], current.logits[i], `policy ${i}`);
          else if (current.takes_legal) assert.equal(result.policy.data[i], -Infinity);
        }
        close(result.value.data[0], current.value, 'value');
        Array.from(result.hands.data).forEach((value, i) => close(value, current.hands[i], `hands ${i}`));
        return result;
      },
    };
  } },
};
const self = { postMessage: message => messages.push(message) };
const context = vm.createContext({
  ort: bridge, self, policyWeights, MEMORY_LIMITS_MIB, isMemoryError, URL,
  Float32Array, Float64Array,
  networkBytes: async ({ url }) => new Uint8Array(await readFile(url)),
  loadMemoryControls: url => import(url),
});
vm.runInContext(source, context);
try {
  for (let id = 0; id < cases.length; id++) {
    current = cases[id];
    await self.onmessage({ data: {
      id, url: resolve(folder, current.model), runtimeBase,
      planes: Float32Array.from(current.planes), mask: current.mask,
      temperature: 0, details: true,
    } });
    const failed = messages.find(message => message.id === id && message.error);
    assert.equal(failed, undefined, failed?.error);
    const answer = messages.find(message => message.id === id && message.action !== undefined);
    assert.ok(answer, 'worker must return an action');
    assert.equal(answer.action, current.action, `${current.model}/${current.stage}`);
    assert.equal(current.mask[answer.action], true);
    assert.ok(tensors.every(tensor => tensor.disposed), 'all inputs and outputs released after each request');
  }
} finally {
  await vm.runInContext('Promise.all([...sessions.values()].map(session => session.release()))', context);
}
console.log(`PASS: ${cases.length} production-worker inferences using real exported graphs and reduced WASM; both riichi stages, one/two-input compatibility, masked logits and tensor disposal`);
