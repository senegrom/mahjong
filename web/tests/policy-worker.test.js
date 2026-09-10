import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { setImmediate } from 'node:timers/promises';
import { policyWeights } from '../src/lib/policy-weights.js';
import { MemoryBudget, MEMORY_LIMITS_MIB, isMemoryError } from '../src/lib/memory-budget.js';

const source = (await readFile(new URL('../src/lib/policy.worker.js', import.meta.url), 'utf8'))
  .replace("import * as ort from 'onnxruntime-web/wasm';", '')
  .replace("import { policyWeights } from './policy-weights.js';", '')
  .replace("import { isMemoryError, MEMORY_LIMITS_MIB } from './memory-budget.js';", '')
  .replace('import(/* @vite-ignore */ controls)', 'loadMemoryControls(controls)');
const MODEL = 'model-full.onnx';
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};

function harness({ loadGate, runGate, invalidOutput = false, runError = false, memoryFailure = null } = {}) {
  const loads = [], releases = [], runs = [], tensors = [], messages = [];
  let live = 0, active = 0;
  const memoryBudget = new MemoryBudget();
  class Tensor {
    constructor(_type, data) { this.data = data; this.disposed = false; tensors.push(this); }
    dispose() { assert.equal(this.disposed, false); this.disposed = true; }
  }
  const ort = {
    env: { wasm: {} }, Tensor,
    InferenceSession: { create: async url => {
      loads.push(url);
      assert.ok(live++ < 1, 'only the one network the page carries may be resident');
      await loadGate?.promise;
      return {
        run: async () => {
          assert.equal(active++, 0, 'inferences must not overlap');
          runs.push(url);
          await runGate?.promise;
          active--;
          if (runError) { memoryBudget.failure = memoryFailure; throw new RangeError('Out of memory'); }
          return {
            policy: new Tensor('float32', invalidOutput ? [NaN, 2] : [1, 3]),
            auxiliary: new Tensor('float32', [0]),
          };
        },
        release: async () => { assert.equal(active, 0); releases.push(url); live--; },
      };
    } },
  };
  const self = { postMessage: message => messages.push(message) };
  vm.runInNewContext(source, { ort, policyWeights, self, URL, MEMORY_LIMITS_MIB, isMemoryError,
    loadMemoryControls: async url => {
      assert.equal(url, 'https://test.invalid/ort/memory-budget.mjs');
      return { memoryBudget };
    },
  });
  const send = (id, url = MODEL) => self.onmessage({ data: {
    id, url, runtimeBase: 'https://test.invalid/ort/', planes: new Float32Array(34), mask: [1, 1], details: true,
  } });
  return { send, cancel: id => self.onmessage({ data: { cancel: id } }), loads, releases, runs, tensors, messages };
}

test('queued decisions load the network once, run one at a time and answer in order', async () => {
  const loadGate = deferred(), runGate = deferred();
  const h = harness({ loadGate, runGate });
  const completed = h.send(1);
  h.send(2); h.send(3); h.send(4);
  await setImmediate();
  assert.deepEqual(h.loads, [MODEL]);
  loadGate.resolve();
  await setImmediate();
  assert.deepEqual(h.runs, [MODEL]);
  runGate.resolve();
  await completed;
  // The network stays loaded, so the later turns are not another load.
  assert.deepEqual(h.loads, [MODEL]);
  assert.deepEqual(h.releases, []);
  assert.deepEqual(h.runs, [MODEL, MODEL, MODEL, MODEL]);
  assert.deepEqual(h.messages.filter(message => message.analysis).map(({ id, action }) => [id, action]), [[1, 1], [2, 1], [3, 1], [4, 1]]);
  assert.ok(h.tensors.every(tensor => tensor.disposed), 'all input and output tensors must be disposed');
});

test('a cancelled queued decision is dropped before it reaches the network', async () => {
  const loadGate = deferred();
  const h = harness({ loadGate });
  const completed = h.send(1);
  h.send(2);
  h.cancel(2);
  h.send(3);
  loadGate.resolve();
  await completed;
  assert.deepEqual(h.loads, [MODEL]);
  assert.deepEqual(h.runs, [MODEL, MODEL]);
  assert.deepEqual(h.messages.filter(message => message.analysis).map(message => message.id), [1, 3]);
});

test('inference and invalid-output failures dispose tensors and stop work on the broken runtime', async () => {
  for (const options of [{ runError: true }, { invalidOutput: true }]) {
    const loadGate = deferred();
    const h = harness({ ...options, loadGate });
    const completed = h.send(1);
    h.send(2);
    loadGate.resolve();
    await completed;
    assert.deepEqual(h.loads, [MODEL]);
    assert.deepEqual(h.runs, [MODEL]);
    assert.equal(h.messages.filter(message => message.error).length, 1);
    assert.equal(h.messages.filter(message => message.analysis).length, 0);
    assert.ok(h.tensors.every(tensor => tensor.disposed));
  }
});

test('runtime failures identify application limits separately from browser allocation refusal', async () => {
  for (const kind of ['limit', 'browser']) {
    const failure = { kind, requestedBytes: 200 * 1048576 };
    const h = harness({ runError: true, memoryFailure: failure });
    await h.send(1);
    const error = h.messages.find(message => message.error);
    assert.deepEqual({ ...error.memory }, { ...failure, limitMiB: 192 });
    assert.ok(h.tensors.every(tensor => tensor.disposed));
  }
  const h = harness({ runError: true });
  await h.send(1);
  assert.equal(h.messages.find(message => message.error).memory, undefined,
    'an unclassified error must not trigger larger memory reservations');
});
