import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { setImmediate } from 'node:timers/promises';
import { policyWeights } from '../src/lib/policy-weights.js';

const source = (await readFile(new URL('../src/lib/policy.worker.js', import.meta.url), 'utf8'))
  .replace("import * as ort from 'onnxruntime-web/wasm';", '')
  .replace("import { policyWeights } from './policy-weights.js';", '');
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};

function harness({ loadGate, runGate, invalidOutput = false, runError = false } = {}) {
  const loads = [], releases = [], runs = [], tensors = [], messages = [];
  let live = 0, active = 0;
  class Tensor {
    constructor(_type, data) { this.data = data; this.disposed = false; tensors.push(this); }
    dispose() { assert.equal(this.disposed, false); this.disposed = true; }
  }
  const ort = {
    env: { wasm: {} }, Tensor,
    InferenceSession: { create: async url => {
      loads.push(url);
      assert.equal(live++, 0, 'the previous model must be released before allocating another');
      await loadGate?.promise;
      return {
        run: async () => {
          assert.equal(active++, 0, 'inferences must not overlap');
          runs.push(url);
          await runGate?.promise;
          active--;
          if (runError) throw new RangeError('Out of memory');
          return {
            policy: new Tensor('float32', invalidOutput ? [NaN, 2] : url === 'quick' ? [1, 3] : [4, 2]),
            auxiliary: new Tensor('float32', [0]),
          };
        },
        release: async () => { assert.equal(active, 0); releases.push(url); live--; },
      };
    } },
  };
  const self = { postMessage: message => messages.push(message) };
  vm.runInNewContext(source, { ort, policyWeights, self });
  const send = (id, url = 'quick') => self.onmessage({ data: {
    id, url, runtimeBase: '/ort/', planes: new Float32Array(34), mask: [1, 1], details: true,
  } });
  return { send, cancel: id => self.onmessage({ data: { cancel: id } }), loads, releases, runs, tensors, messages };
}

test('concurrent mixed agents share one session at a time and preserve their own model choices', async () => {
  const loadGate = deferred(), runGate = deferred();
  const h = harness({ loadGate, runGate });
  const completed = h.send(1);
  h.send(2); h.send(3, 'strong'); h.send(4);
  await setImmediate();
  assert.deepEqual(h.loads, ['quick']);
  loadGate.resolve();
  await setImmediate();
  assert.deepEqual(h.runs, ['quick']);
  runGate.resolve();
  await completed;
  assert.deepEqual(h.loads, ['quick', 'strong', 'quick']);
  assert.deepEqual(h.releases, ['quick', 'strong']);
  assert.deepEqual(h.runs, ['quick', 'quick', 'strong', 'quick']);
  assert.deepEqual(h.messages.filter(message => message.analysis).map(({ id, action }) => [id, action]), [[1, 1], [2, 1], [3, 0], [4, 1]]);
  assert.ok(h.tensors.every(tensor => tensor.disposed), 'all input and output tensors must be disposed');
});

test('cancelled queued analysis never loads an unwanted model', async () => {
  const loadGate = deferred();
  const h = harness({ loadGate });
  const completed = h.send(1);
  h.send(2, 'strong');
  h.cancel(2);
  h.send(3);
  loadGate.resolve();
  await completed;
  assert.deepEqual(h.loads, ['quick']);
  assert.deepEqual(h.runs, ['quick', 'quick']);
  assert.deepEqual(h.messages.filter(message => message.analysis).map(message => message.id), [1, 3]);
});

test('inference and invalid-output failures dispose tensors and stop work on the broken runtime', async () => {
  for (const options of [{ runError: true }, { invalidOutput: true }]) {
    const loadGate = deferred();
    const h = harness({ ...options, loadGate });
    const completed = h.send(1);
    h.send(2, 'strong');
    loadGate.resolve();
    await completed;
    assert.deepEqual(h.loads, ['quick']);
    assert.deepEqual(h.runs, ['quick']);
    assert.equal(h.messages.filter(message => message.error).length, 1);
    assert.equal(h.messages.filter(message => message.analysis).length, 0);
    assert.ok(h.tensors.every(tensor => tensor.disposed));
  }
});
