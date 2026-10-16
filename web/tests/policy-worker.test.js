import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { policyWeights } from '../src/lib/policy-weights.js';
import { MemoryBudget, MEMORY_LIMITS_MIB, isMemoryError } from '../src/lib/memory-budget.js';
import { loadModule } from './fixtures/load-module.js';

const MODEL = 'model-full.onnx';
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};

function harness({ takesMask = false, loadGate, runGate, invalidOutput = false, runError = false, memoryFailure = null, policy = null } = {}) {
  const loads = [], releases = [], runs = [], tensors = [], messages = [], scopes = [];
  let live = 0, active = 0;
  const memoryBudget = new MemoryBudget();
  class Tensor {
    constructor(_type, data, dims) { this.data = data; this.dims = dims; this.disposed = false; tensors.push(this); }
    dispose() { assert.equal(this.disposed, false); this.disposed = true; }
  }
  const ort = {
    env: { wasm: {} }, Tensor,
    InferenceSession: { create: async url => {
      loads.push(url);
      const inputNames = takesMask ? ['planes', 'legal'] : ['planes'];
      assert.ok(live++ < 1, 'only the one network the page carries may be resident');
      await loadGate?.promise;
      return {
        inputNames,
        run: async given => {
          assert.equal(active++, 0, 'inferences must not overlap');
          assert.deepEqual(Object.keys(given), inputNames, 'the graph is given what it asks for');
          runs.push(url);
          await runGate?.promise;
          active--;
          if (runError) { memoryBudget.failure = memoryFailure; throw new RangeError('Out of memory'); }
          return {
            policy: new Tensor('float32', policy ? policy(given) : invalidOutput ? [NaN, 2] : [1, 3]),
            value: new Tensor('float32', [0.25]),
            hands: new Tensor('float32', new Float32Array(102)),
          };
        },
        release: async () => { assert.equal(active, 0); releases.push(url); live--; },
      };
    } },
  };
  const self = { postMessage: message => messages.push(message) };
  loadModule('policy.worker.js', { ort, policyWeights, self, URL, MEMORY_LIMITS_MIB, isMemoryError,
    // The network arrives as bytes from its bucket; the address stands in for
    // them here, so what loaded is still recognisable.
    networkBytes: async ({ url, scope }) => { scopes.push(scope); return url; },
    loadMemoryControls: async url => {
      assert.equal(url, 'https://test.invalid/ort/0123abcd/memory-budget.mjs');
      return { memoryBudget };
    },
  }, [], { 'import(/* @vite-ignore */ controls)': 'loadMemoryControls(controls)' });
  const send = (id, url = MODEL, extra = {}) => self.onmessage({ data: {
    id, url, runtimeBase: 'https://test.invalid/ort/0123abcd/', scope: 'https://test.invalid/',
    planes: new Float32Array(34), mask: [1, 1], details: true, ...extra,
  } });
  return { send, cancel: id => self.onmessage({ data: { cancel: id } }), loads, releases, runs, tensors, messages, scopes };
}

test('queued decisions load the network once, run one at a time and answer in order', async () => {
  const loadGate = deferred(), runGate = deferred();
  const h = harness({ loadGate, runGate });
  const completed = h.send(1);
  h.send(2); h.send(3); h.send(4);
  await setImmediate();
  assert.deepEqual(h.loads, [MODEL]);
  assert.deepEqual(h.scopes, ['https://test.invalid/'], "the network is read from the page's cache, not the runtime folder's");
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

test('an opponent move carries its action alone; advice also carries the value and beliefs', async () => {
  const h = harness();
  await h.send(1, MODEL, { details: false });
  await h.send(2);
  const [move, advice] = h.messages.filter(message => message.action !== undefined);
  assert.deepEqual({ ...move }, { id: 1, action: 1 });
  assert.deepEqual(Object.keys(advice).sort(), ['action', 'analysis', 'id']);
  assert.equal(advice.action, 1);
  assert.equal(advice.analysis.action, 1);
  assert.equal(advice.analysis.value, 0.25);
  assert.equal(advice.analysis.hands.length, 102);
  assert.ok(h.tensors.every(tensor => tensor.disposed));
});

test('a request naming another network releases the one held before loading it', async () => {
  // The worker parity check runs two exported networks through one worker.
  const h = harness();
  await h.send(1);
  await h.send(2, 'fusion.onnx');
  await h.send(3, 'fusion.onnx');
  assert.deepEqual(h.loads, [MODEL, 'fusion.onnx']);
  assert.deepEqual(h.releases, [MODEL]);
  assert.deepEqual(h.runs, [MODEL, 'fusion.onnx', 'fusion.onnx']);
  assert.equal(h.messages.some(message => message.error), false);
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
    assert.deepEqual({ ...error.memory }, { ...failure, limitMiB: MEMORY_LIMITS_MIB[0] });
    assert.ok(h.tensors.every(tensor => tensor.disposed));
  }
  const h = harness({ runError: true });
  await h.send(1);
  assert.equal(h.messages.find(message => message.error).memory, undefined,
    'an unclassified error must not trigger larger memory reservations');
});

test('bytes the page could not store load the network without downloading it again', async () => {
  const h = harness();
  const handed = 'verified bytes storage refused';
  await h.send(1, MODEL, { network: handed });
  assert.deepEqual(h.loads, [handed], 'the runtime starts from the handed bytes');
  assert.deepEqual(h.scopes, [], 'nothing is fetched');
  assert.equal(h.messages.find(message => message.analysis)?.id, 1);
  assert.equal(h.messages.some(message => message.error), false);
});

test('a review batch is one forward pass, each position answered with its own best legal move', async () => {
  // Three positions of two moves each, laid end to end.
  const logits = [0, 2, 5, 9, 3, 1];
  const h = harness({ takesMask: true, policy: given => {
    // The module runs in its own context: its arrays are copied out to compare.
    assert.deepEqual([...given.planes.dims], [3, 2, 34]);
    assert.deepEqual([...given.legal.dims], [3, 2]);
    assert.deepEqual(Array.from(given.legal.data), [1, 1, 1, 0, 1, 1]);
    return Float32Array.from(logits);
  } });
  await h.send(1, MODEL, { planes: new Float32Array(3 * 2 * 34), mask: [1, 1, 1, 0, 1, 1], count: 3 });
  assert.deepEqual(h.runs, [MODEL], 'one pass for the batch');
  const answer = h.messages.find(message => message.analyses);
  assert.equal(answer.id, 1);
  assert.equal('action' in answer, false);
  const analyses = [...answer.analyses].map(({ action, weights }) => ({ action, weights: [...weights] }));
  assert.deepEqual(analyses.map(analysis => analysis.action), [1, 0, 0]);
  // The second position's better logit is a move its mask does not allow.
  assert.deepEqual(analyses[1].weights, [1, 0]);
  assert.deepEqual(analyses, [policyWeights(logits.slice(0, 2), [1, 1]), policyWeights(logits.slice(2, 4), [1, 0]),
    policyWeights(logits.slice(4), [1, 1])]);
  assert.ok(h.tensors.every(tensor => tensor.disposed), 'all input and output tensors must be disposed');
});
