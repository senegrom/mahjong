import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { setImmediate } from 'node:timers/promises';
import { MEMORY_LIMITS_MIB, nextMemoryLimit } from '../src/lib/memory-budget.js';
import { MODEL_FILES } from '../src/lib/model-package.js';

const source = (await readFile(new URL('../src/lib/policy.js', import.meta.url), 'utf8'))
  .replace("import { startOffline, prepareOfflineAi } from './offline.js';", '')
  .replace("import { MEMORY_LIMITS_MIB, nextMemoryLimit } from './memory-budget.js';", '')
  .replace("import { MODEL_FILES } from './model-package.js';", '')
  .replaceAll('import.meta.url', "'https://test.invalid/mahjong/policy.js'")
  .replaceAll('export ', '');

// Run the real request coordinator with controllable downloads and responses.
// One trained network ships, so every request names the same one.
test('concurrent decisions share one worker and preserve a pending turn', async (t) => {
  const downloads = [], workers = [];
  class Worker {
    constructor() { this.messages = []; this.terminated = false; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    answer(message, payload) { this.onmessage({ data: { id: message.id, ...payload } }); }
  }
  const context = vm.createContext({
    document: { baseURI: 'https://test.invalid/mahjong/' },
    URL, DOMException, Worker, setTimeout, clearTimeout, MEMORY_LIMITS_MIB, nextMemoryLimit, MODEL_FILES,
    startOffline: async () => null,
    prepareOfflineAi: (model) => new Promise(resolve => downloads.push({ model, resolve })),
  });
  vm.runInContext(source + '\nglobalThis.api = { chooseAction, analyzePolicy, useModel, chosenModel, resetPolicy };', context);
  const { api } = context;
  t.after(() => api.resetPolicy());
  const planes = () => new Float32Array(34);
  const mask = [1, 1];

  const first = api.chooseAction(planes(), mask);
  // Observe rejection immediately, including when exercising the old bug.
  const firstResult = first.then(value => ({ value }), error => ({ error }));
  assert.equal(downloads[0].model, 'full');
  downloads[0].resolve();
  await setImmediate();
  const worker = workers[0], initial = worker.messages[0];
  assert.match(initial.url, /\/model-full\.onnx$/);

  // The shipped network is the only choice, so reselecting it changes nothing.
  assert.equal(api.useModel('full'), 'full');
  assert.equal(worker.terminated, false, 'reselecting the network must not abort an in-flight turn');
  worker.answer(initial, { action: 1 });
  assert.deepEqual(await firstResult, { value: 1 });

  const next = api.chooseAction(planes(), mask);
  const analysis = api.analyzePolicy(planes(), mask, undefined, 'full');
  const results = Promise.all([next, analysis]);
  assert.deepEqual(downloads.slice(1).map(download => download.model), ['full', 'full']);
  for (const download of downloads.slice(1)) download.resolve();
  await setImmediate();
  assert.equal(workers.length, 1, 'a move and a review share one worker without interrupting each other');
  const [move, detailed] = worker.messages.slice(1);
  assert.match(move.url, /\/model-full\.onnx$/);
  assert.match(detailed.url, /\/model-full\.onnx$/);
  assert.equal(move.details, false);
  assert.equal(detailed.details, true);
  worker.answer(move, { action: 0 });
  worker.answer(detailed, { analysis: { action: 1, weights: [0.2, 0.8] } });
  assert.deepEqual(await results, [0, { action: 1, weights: [0.2, 0.8] }]);
  assert.equal(api.chosenModel(), 'full');
  // Networks earlier builds carried are not agents this one can run.
  for (const retired of ['quick', 'strong']) {
    await assert.rejects(api.analyzePolicy(planes(), mask, undefined, retired), /Unknown trained agent/);
  }
});

test('memory errors discard the worker even after cancellation, and retry starts a fresh runtime', async (t) => {
  const workers = [];
  class Worker {
    constructor() { this.messages = []; this.terminated = false; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    answer(id, payload) { this.onmessage({ data: { id, ...payload } }); }
  }
  const context = vm.createContext({
    document: { baseURI: 'https://test.invalid/mahjong/' },
    URL, DOMException, Worker, setTimeout, clearTimeout, MEMORY_LIMITS_MIB, nextMemoryLimit, MODEL_FILES,
    startOffline: async () => null, prepareOfflineAi: async () => null,
  });
  vm.runInContext(source + '\nglobalThis.api = { chooseAction, analyzePolicy, resetPolicy };', context);
  const { api } = context;
  t.after(() => api.resetPolicy());
  const ask = signal => api.analyzePolicy(new Float32Array(34), [1, 1], signal, 'full');
  const owner = new AbortController();
  const cancelled = ask(owner.signal);
  const aborted = assert.rejects(cancelled, { name: 'AbortError' });
  await setImmediate();
  const first = workers[0], originalId = first.messages[0].id;
  owner.abort();
  await aborted;
  assert.equal(first.messages.at(-1).cancel, originalId);

  const next = ask();
  const rejected = assert.rejects(next, /Out of memory/);
  await setImmediate();
  first.answer(originalId, { error: 'RangeError: Out of memory' });
  await rejected;
  assert.equal(first.terminated, true, 'an error from a cancelled request still poisons the runtime');

  const retry = ask();
  await setImmediate();
  const fresh = workers[1];
  assert.ok(fresh, 'physical/watch retries must create a fresh worker automatically');
  assert.match(fresh.messages[0].url, /\/model-full\.onnx$/);
  first.answer(originalId, { error: 'late old failure' });
  assert.equal(fresh.terminated, false);
  fresh.answer(fresh.messages[0].id, { analysis: { action: 1, weights: [0.2, 0.8] } });
  assert.deepEqual(await retry, { action: 1, weights: [0.2, 0.8] });
});

function memoryClient(t) {
  const workers = [];
  class Worker {
    constructor() { this.messages = []; this.terminated = false; workers.push(this); }
    postMessage(message, transfer = []) { this.messages.push(structuredClone(message, { transfer })); }
    terminate() { this.terminated = true; }
    answer(id, payload) { this.onmessage({ data: { id, ...payload } }); }
    fail(message, kind, requestedMiB) {
      this.answer(message.id, { error: 'Out of memory', memory: { kind,
        requestedBytes: requestedMiB * 1048576, limitMiB: message.memoryLimitMiB } });
    }
  }
  const context = vm.createContext({
    document: { baseURI: 'https://test.invalid/mahjong/' },
    URL, DOMException, Worker, setTimeout, clearTimeout, MEMORY_LIMITS_MIB, nextMemoryLimit, MODEL_FILES,
    startOffline: async () => null, prepareOfflineAi: async () => null,
  });
  vm.runInContext(source + '\nglobalThis.api = { analyzePolicy, resetPolicy };', context);
  t.after(() => context.api.resetPolicy());
  return { workers, api: context.api, ask: (model = 'full', signal, planes = Float32Array.from({ length: 34 }, (_, i) => i / 2)) =>
    context.api.analyzePolicy(planes, [1, 1], signal, model) };
}

test('memory-limit retries replay only unresolved choices with their original observations', async t => {
  const { ask, workers } = memoryClient(t);
  const planes = Float32Array.from({ length: 34 }, (_, i) => i / 2), expected = Array.from(planes);
  const first = ask('full', undefined, planes);
  planes.fill(99); // Changes in the caller must not change a retained decision.
  await setImmediate();
  const original = workers[0], pending = original.messages.at(-1);
  const queued = ask();
  await setImmediate();
  const behind = original.messages.at(-1);
  const completed = ask();
  await setImmediate();
  const done = original.messages.at(-1);
  const abort = new AbortController(), cancelled = ask('full', abort.signal);
  const cancellation = assert.rejects(cancelled, { name: 'AbortError' });
  await setImmediate();
  original.answer(done.id, { analysis: { action: 1 } });
  await completed;
  abort.abort(); await cancellation;
  original.fail(pending, 'limit', 200);
  assert.equal(original.terminated, true);
  const larger = workers[1];
  assert.deepEqual(larger.messages.map(m => m.id), [pending.id, behind.id]);
  assert.ok(larger.messages.every(m => m.memoryLimitMiB === 256));
  assert.deepEqual(Array.from(larger.messages[0].planes), expected, 'transferred buffers remain replayable');
  assert.match(larger.messages[1].url, /model-full\.onnx$/);
  original.answer(pending.id, { analysis: { action: 99 } }); // Old-worker results are ignored.
  larger.fail(larger.messages[0], 'limit', 300);
  assert.equal(larger.terminated, true);
  const final = workers[2];
  assert.ok(final.messages.every(m => m.memoryLimitMiB === 384));
  assert.deepEqual(Array.from(final.messages[0].planes), expected);
  final.answer(pending.id, { analysis: { action: 0 } });
  final.answer(behind.id, { analysis: { action: 1 } });
  assert.deepEqual(await Promise.all([first, queued]), [{ action: 0 }, { action: 1 }]);
  assert.equal(workers.length, 3);
});

test('browser refusal and the final memory ceiling stop retries and allow a fresh small runtime', async t => {
  for (const refusedByBrowser of [true, false]) {
    const { ask, workers } = memoryClient(t);
    const pending = ask(), rejected = assert.rejects(pending, /Out of memory/);
    await setImmediate();
    workers[0].fail(workers[0].messages[0], 'limit', 200);
    if (!refusedByBrowser) workers[1].fail(workers[1].messages[0], 'limit', 300);
    const last = workers.at(-1), count = workers.length;
    last.fail(last.messages[0], refusedByBrowser ? 'browser' : 'limit', refusedByBrowser ? 256 : 400);
    await rejected;
    assert.equal(last.terminated, true);
    assert.equal(workers.length, count, 'do not keep asking for unavailable memory');
    const retry = ask(); await setImmediate();
    const fresh = workers.at(-1);
    assert.equal(fresh.messages[0].memoryLimitMiB, 192);
    fresh.answer(fresh.messages[0].id, { analysis: { action: 1 } });
    await retry;
  }
});

test('a memory ceiling that was found to work survives an ordinary reset', async t => {
  const { ask, workers, api } = memoryClient(t);
  const first = ask(); await setImmediate();
  workers[0].fail(workers[0].messages[0], 'limit', 200);
  await setImmediate();
  const grown = workers[1];
  assert.equal(grown.messages[0].memoryLimitMiB, 256);
  grown.answer(grown.messages[0].id, { analysis: { action: 1 } });
  await first;
  // A retry after some other failure keeps the size that worked, instead of
  // failing at the small size and growing again.
  api.resetPolicy(new Error('a timeout'));
  const again = ask(); await setImmediate();
  assert.equal(workers.at(-1).messages[0].memoryLimitMiB, 256);
  workers.at(-1).answer(workers.at(-1).messages[0].id, { analysis: { action: 0 } });
  await again;
});
