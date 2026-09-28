import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { setImmediate } from 'node:timers/promises';
import { MEMORY_LIMITS_MIB, nextMemoryLimit } from '../src/lib/memory-budget.js';
import { NETWORK_URL } from '../src/lib/network-store.js';

const source = (await readFile(new URL('../src/lib/policy.js', import.meta.url), 'utf8'))
  .replace("import { prepareOfflineAi } from './offline.js';", '')
  .replace("import { MEMORY_LIMITS_MIB, nextMemoryLimit } from './memory-budget.js';", '')
  .replace("import { NETWORK_URL, networkIsStored } from './network-store.js';", '')
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
    URL, DOMException, Worker, setTimeout, clearTimeout, MEMORY_LIMITS_MIB, nextMemoryLimit, NETWORK_URL, __RUNTIME_DIRECTORY__: 'ort/0123abcd/',
    networkIsStored: async () => false,
    startOffline: async () => null,
    prepareOfflineAi: (...args) => new Promise(resolve => downloads.push({ args, resolve })),
  });
  vm.runInContext(source + '\nglobalThis.api = { chooseAction, analyzePolicy, modelIsAvailable, resetPolicy };', context);
  const { api } = context;
  t.after(() => api.resetPolicy());
  const planes = () => new Float32Array(34);
  const mask = [1, 1];

  const first = api.chooseAction(planes(), mask);
  const firstResult = first.then(value => ({ value }), error => ({ error }));
  assert.deepEqual(downloads[0].args, [], 'one network: preparation names none');
  downloads[0].resolve();
  await setImmediate();
  const worker = workers[0], initial = worker.messages[0];
  assert.equal(initial.url, NETWORK_URL);
  // The runtime's content-named folder under the page, and the page's own
  // folder, whose Cache Storage holds the network.
  assert.equal(initial.runtimeBase, 'https://test.invalid/mahjong/ort/0123abcd/');
  assert.equal(initial.scope, 'https://test.invalid/mahjong/');
  assert.equal(worker.terminated, false);
  worker.answer(initial, { action: 1 });
  assert.deepEqual(await firstResult, { value: 1 });

  const next = api.chooseAction(planes(), mask);
  const analysis = api.analyzePolicy(planes(), mask);
  const results = Promise.all([next, analysis]);
  assert.equal(downloads.length, 1, 'warm decisions reuse preparation');
  await setImmediate();
  assert.equal(workers.length, 1, 'a move and a review share one worker without interrupting each other');
  const [move, detailed] = worker.messages.slice(1);
  assert.equal(move.url, NETWORK_URL);
  assert.equal(detailed.url, NETWORK_URL);
  assert.equal(move.details, false);
  assert.equal(detailed.details, true);
  worker.answer(move, { action: 0 });
  worker.answer(detailed, { analysis: { action: 1, weights: [0.2, 0.8] } });
  assert.deepEqual(await results, [0, { action: 1, weights: [0.2, 0.8] }]);
  for (const message of [initial, move, detailed]) assert.equal('temperature' in message, false, 'moves are never sampled');
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
    URL, DOMException, Worker, setTimeout, clearTimeout, MEMORY_LIMITS_MIB, nextMemoryLimit, NETWORK_URL, __RUNTIME_DIRECTORY__: 'ort/0123abcd/',
    networkIsStored: async () => false,
    startOffline: async () => null, prepareOfflineAi: async () => null,
  });
  vm.runInContext(source + '\nglobalThis.api = { chooseAction, analyzePolicy, resetPolicy };', context);
  const { api } = context;
  t.after(() => api.resetPolicy());
  const ask = signal => api.analyzePolicy(new Float32Array(34), [1, 1], signal);
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
  assert.equal(fresh.messages[0].url, NETWORK_URL);
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
    URL, DOMException, Worker, setTimeout, clearTimeout, MEMORY_LIMITS_MIB, nextMemoryLimit, NETWORK_URL, __RUNTIME_DIRECTORY__: 'ort/0123abcd/',
    networkIsStored: async () => false,
    startOffline: async () => null, prepareOfflineAi: async () => null,
  });
  vm.runInContext(source + '\nglobalThis.api = { analyzePolicy, resetPolicy };', context);
  t.after(() => context.api.resetPolicy());
  return { workers, api: context.api, ask: (signal, planes = Float32Array.from({ length: 34 }, (_, i) => i / 2)) =>
    context.api.analyzePolicy(planes, [1, 1], signal) };
}

test('memory-limit retries replay only unresolved choices with their original observations', async t => {
  const { ask, workers } = memoryClient(t);
  const planes = Float32Array.from({ length: 34 }, (_, i) => i / 2), expected = Array.from(planes);
  const first = ask(undefined, planes);
  planes.fill(99); // Changes in the caller must not change a retained decision.
  await setImmediate();
  const original = workers[0], pending = original.messages.at(-1);
  const queued = ask();
  await setImmediate();
  const behind = original.messages.at(-1);
  const completed = ask();
  await setImmediate();
  const done = original.messages.at(-1);
  const abort = new AbortController(), cancelled = ask(abort.signal);
  const cancellation = assert.rejects(cancelled, { name: 'AbortError' });
  await setImmediate();
  original.answer(done.id, { analysis: { action: 1 } });
  await completed;
  abort.abort(); await cancellation;
  original.fail(pending, 'limit', MEMORY_LIMITS_MIB[0] + 16);
  assert.equal(original.terminated, true);
  const larger = workers[1];
  assert.deepEqual(larger.messages.map(m => m.id), [pending.id, behind.id]);
  assert.ok(larger.messages.every(m => m.memoryLimitMiB === MEMORY_LIMITS_MIB[1]));
  assert.deepEqual(Array.from(larger.messages[0].planes), expected, 'transferred buffers remain replayable');
  assert.equal(larger.messages[1].url, NETWORK_URL);
  original.answer(pending.id, { analysis: { action: 99 } }); // Old-worker results are ignored.
  larger.fail(larger.messages[0], 'limit', MEMORY_LIMITS_MIB[1] + 16);
  assert.equal(larger.terminated, true);
  const final = workers[2];
  assert.ok(final.messages.every(m => m.memoryLimitMiB === MEMORY_LIMITS_MIB[2]));
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
    workers[0].fail(workers[0].messages[0], 'limit', MEMORY_LIMITS_MIB[0] + 16);
    if (!refusedByBrowser) workers[1].fail(workers[1].messages[0], 'limit', MEMORY_LIMITS_MIB[1] + 16);
    const last = workers.at(-1), count = workers.length;
    last.fail(last.messages[0], refusedByBrowser ? 'browser' : 'limit',
      refusedByBrowser ? MEMORY_LIMITS_MIB[1] : MEMORY_LIMITS_MIB[1] + 16);
    await rejected;
    assert.equal(last.terminated, true);
    assert.equal(workers.length, count, 'do not keep asking for unavailable memory');
    const retry = ask(); await setImmediate();
    const fresh = workers.at(-1);
    assert.equal(fresh.messages[0].memoryLimitMiB, MEMORY_LIMITS_MIB[0]);
    fresh.answer(fresh.messages[0].id, { analysis: { action: 1 } });
    await retry;
  }
});

test('a memory ceiling that was found to work survives an ordinary reset', async t => {
  const { ask, workers, api } = memoryClient(t);
  const first = ask(); await setImmediate();
  workers[0].fail(workers[0].messages[0], 'limit', MEMORY_LIMITS_MIB[0] + 16);
  await setImmediate();
  const grown = workers[1];
  assert.equal(grown.messages[0].memoryLimitMiB, MEMORY_LIMITS_MIB[1]);
  grown.answer(grown.messages[0].id, { analysis: { action: 1 } });
  await first;
  // A retry after some other failure keeps the size that worked, instead of
  // failing at the small size and growing again.
  api.resetPolicy(new Error('a timeout'));
  const again = ask(); await setImmediate();
  assert.equal(workers.at(-1).messages[0].memoryLimitMiB, MEMORY_LIMITS_MIB[1]);
  workers.at(-1).answer(workers.at(-1).messages[0].id, { analysis: { action: 0 } });
  await again;
});

/** The coordinator as the page runs it when offline saving could not hold the
 * network: the worker downloads it itself. The clock is the test's own. */
function loadingClient(t) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const workers = [], notes = [];
  class Worker {
    constructor() { this.messages = []; this.terminated = false; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    say(id, payload) { this.onmessage({ data: { id, ...payload } }); }
  }
  const context = vm.createContext({
    document: { baseURI: 'https://test.invalid/mahjong/' },
    URL, DOMException, Worker, MEMORY_LIMITS_MIB, nextMemoryLimit, NETWORK_URL, __RUNTIME_DIRECTORY__: 'ort/0123abcd/',
    // Looked up when called, so the mocked clock drives the coordinator.
    setTimeout: (...args) => setTimeout(...args), clearTimeout: timer => clearTimeout(timer),
    networkIsStored: async () => false,
    prepareOfflineAi: async () => null, // no service worker, a failed start or a storage failure
  });
  vm.runInContext(source + '\nglobalThis.api = { chooseAction, reportProgress, resetPolicy, LOADING_SILENCE_MS };', context);
  context.api.reportProgress(note => notes.push(note));
  t.after(() => context.api.resetPolicy());
  const ask = signal => context.api.chooseAction(new Float32Array(34), [1, 1], signal);
  return { api: context.api, workers, notes, ask, silence: context.api.LOADING_SILENCE_MS };
}

test('a network still arriving is not cut off by the decision deadline', async t => {
  const { ask, workers, notes, silence } = loadingClient(t);
  const decision = ask();
  await setImmediate();
  const [worker] = workers, { id } = worker.messages[0];
  // Ten minutes of a slow download, far past a decision's twenty seconds.
  for (let percent = 1; percent <= 10; percent++) {
    worker.say(id, { progress: `downloading the network ${percent}%` });
    t.mock.timers.tick(silence - 1);
  }
  assert.equal(worker.terminated, false, 'the download is not discarded');
  worker.say(id, { progress: 'network ready', ready: true });
  worker.say(id, { action: 1 });
  assert.equal(await decision, 1);
  assert.ok(notes.includes('downloading the network 10%'), 'the player sees the download');
});

test('a move is timed once the network is loaded, and a worker that misses it is replaced', async t => {
  const { ask, workers } = loadingClient(t);
  const decision = ask(), late = assert.rejects(decision, /did not answer in time/);
  await setImmediate();
  const [worker] = workers, { id } = worker.messages[0];
  worker.say(id, { progress: 'starting the network' });
  t.mock.timers.tick(60000);
  assert.equal(worker.terminated, false, 'loading has no decision deadline');
  worker.say(id, { progress: 'network ready', ready: true });
  t.mock.timers.tick(19999);
  assert.equal(worker.terminated, false);
  t.mock.timers.tick(1);
  await late;
  assert.equal(worker.terminated, true);
  const retry = ask();
  await setImmediate();
  assert.equal(workers.length, 2, 'the retry loads a fresh worker');
  workers[1].say(workers[1].messages[0].id, { progress: 'network ready', ready: true });
  workers[1].say(workers[1].messages[0].id, { action: 0 });
  assert.equal(await retry, 0);
});

test('a loading worker that falls silent is abandoned', async t => {
  const { ask, workers, silence } = loadingClient(t);
  const decision = ask(), stalled = assert.rejects(decision, /stopped loading/);
  await setImmediate();
  const [worker] = workers, { id } = worker.messages[0];
  t.mock.timers.tick(silence - 1);
  worker.say(id, { progress: 'downloading the network 1%' });
  t.mock.timers.tick(silence - 1);
  assert.equal(worker.terminated, false, 'each note restarts the silence');
  t.mock.timers.tick(1);
  await stalled;
  assert.equal(worker.terminated, true);
});

test('a cancelled decision\'s download keeps loading for the next decision', async t => {
  const { ask, workers, silence } = loadingClient(t);
  const owner = new AbortController();
  const cancelled = assert.rejects(ask(owner.signal), { name: 'AbortError' });
  await setImmediate();
  const [worker] = workers, { id: first } = worker.messages[0];
  owner.abort(); await cancelled;
  const next = ask();
  await setImmediate();
  assert.equal(workers.length, 1, 'the worker loading the network stays');
  const { id: second } = worker.messages.at(-1);
  for (let i = 0; i < 3; i++) {
    worker.say(first, { progress: 'downloading the network 50%' });
    t.mock.timers.tick(silence - 1);
  }
  assert.equal(worker.terminated, false);
  worker.say(first, { progress: 'network ready', ready: true });
  t.mock.timers.tick(19999);
  assert.equal(worker.terminated, false, 'the waiting decision is timed from the network being ready');
  worker.say(second, { action: 1 });
  assert.equal(await next, 1);
});

test('bytes offline storage refused are handed to the fresh worker once, not downloaded again', async t => {
  const workers = [], bytes = new Uint8Array(8).fill(7);
  class Worker {
    constructor() { this.messages = []; this.transfers = []; this.terminated = false; workers.push(this); }
    postMessage(message, transfer = []) { this.messages.push(message); this.transfers.push(transfer); }
    terminate() { this.terminated = true; }
    say(id, payload) { this.onmessage({ data: { id, ...payload } }); }
  }
  const context = vm.createContext({
    document: { baseURI: 'https://test.invalid/mahjong/' },
    URL, DOMException, Worker, setTimeout, clearTimeout, MEMORY_LIMITS_MIB, nextMemoryLimit, NETWORK_URL, __RUNTIME_DIRECTORY__: 'ort/0123abcd/',
    networkIsStored: async () => false,
    prepareOfflineAi: async () => ({ aiReady: false, unstoredNetwork: bytes }),
  });
  vm.runInContext(source + '\nglobalThis.api = { chooseAction, resetPolicy };', context);
  t.after(() => context.api.resetPolicy());
  const ask = () => context.api.chooseAction(new Float32Array(34), [1, 1]);
  const first = ask(), second = ask();
  await setImmediate();
  const [worker] = workers, [one, two] = worker.messages;
  assert.equal(one.network, bytes);
  assert.ok(worker.transfers[0].includes(bytes.buffer), 'moved, not copied');
  assert.equal('network' in two, false, 'handed over once');
  worker.say(one.id, { progress: 'network ready', ready: true });
  worker.say(one.id, { action: 0 }); worker.say(two.id, { action: 1 });
  assert.deepEqual(await Promise.all([first, second]), [0, 1]);
});
