import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { setImmediate } from 'node:timers/promises';

const source = (await readFile(new URL('../src/lib/policy.js', import.meta.url), 'utf8'))
  .replace("import { startOffline, prepareOfflineAi } from './offline.js';", '')
  .replaceAll('import.meta.url', "'https://test.invalid/mahjong/policy.js'")
  .replaceAll('export ', '');

// Run the real request coordinator with controllable downloads and responses.
// Switching models may happen during either stage of an opponent's turn.
test('model changes preserve pending turns and explicitly configured agent requests', async (t) => {
  const downloads = [], workers = [];
  class Worker {
    constructor() { this.messages = []; this.terminated = false; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    answer(message, payload) { this.onmessage({ data: { id: message.id, ...payload } }); }
  }
  const context = vm.createContext({
    document: { baseURI: 'https://test.invalid/mahjong/' },
    URL, DOMException, Worker, setTimeout, clearTimeout,
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
  api.useModel('strong');
  assert.equal(downloads[0].model, 'quick');
  downloads[0].resolve();
  await setImmediate();
  const worker = workers[0], initial = worker.messages[0];
  assert.match(initial.url, /\/model\.onnx$/);

  api.useModel('quick');
  api.useModel('strong');
  assert.equal(worker.terminated, false, 'changing the default must not abort an in-flight turn');
  worker.answer(initial, { action: 1 });
  assert.deepEqual(await firstResult, { value: 1 });

  const next = api.chooseAction(planes(), mask);
  const analysis = api.analyzePolicy(planes(), mask, undefined, 'quick');
  const results = Promise.all([next, analysis]);
  assert.deepEqual(downloads.slice(1).map(download => download.model), ['strong', 'quick']);
  for (const download of downloads.slice(1)) download.resolve();
  await setImmediate();
  assert.equal(workers.length, 1, 'mixed agents share one worker without interrupting each other');
  const [strong, quick] = worker.messages.slice(1);
  assert.match(strong.url, /\/model-strong\.onnx$/);
  assert.match(quick.url, /\/model\.onnx$/);
  assert.equal(quick.details, true);
  worker.answer(strong, { action: 0 });
  worker.answer(quick, { analysis: { action: 1, weights: [0.2, 0.8] } });
  assert.deepEqual(await results, [0, { action: 1, weights: [0.2, 0.8] }]);
  assert.equal(api.chosenModel(), 'strong');
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
    URL, DOMException, Worker, setTimeout, clearTimeout,
    startOffline: async () => null, prepareOfflineAi: async () => null,
  });
  vm.runInContext(source + '\nglobalThis.api = { chooseAction, analyzePolicy, resetPolicy };', context);
  const { api } = context;
  t.after(() => api.resetPolicy());
  const ask = signal => api.analyzePolicy(new Float32Array(34), [1, 1], signal, 'quick');
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
  assert.match(fresh.messages[0].url, /\/model\.onnx$/);
  first.answer(originalId, { error: 'late old failure' });
  assert.equal(fresh.terminated, false);
  fresh.answer(fresh.messages[0].id, { analysis: { action: 1, weights: [0.2, 0.8] } });
  assert.deepEqual(await retry, { action: 1, weights: [0.2, 0.8] });
});
