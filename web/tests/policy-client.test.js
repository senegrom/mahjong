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
  assert.equal(workers.length, 1, 'mixed agents share the worker and its per-model sessions');
  const [strong, quick] = worker.messages.slice(1);
  assert.match(strong.url, /\/model-strong\.onnx$/);
  assert.match(quick.url, /\/model\.onnx$/);
  assert.equal(quick.details, true);
  worker.answer(strong, { action: 0 });
  worker.answer(quick, { analysis: { action: 1, weights: [0.2, 0.8] } });
  assert.deepEqual(await results, [0, { action: 1, weights: [0.2, 0.8] }]);
  assert.equal(api.chosenModel(), 'strong');
});
