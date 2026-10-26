import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { isMemoryError, MemoryBudget, MEMORY_LIMITS_MIB, nextMemoryLimit } from '../src/lib/memory-budget.js';
import { MANIFEST, NETWORKS, networkUrl } from '../src/lib/trained-network.js';
import { policyWeights } from '../src/lib/policy-weights.js';
import { loadModule } from './fixtures/load-module.js';

// The page plays the default unless a test says otherwise.
const NETWORK_URL = networkUrl(MANIFEST);
/** The real request coordinator, given a stand-in Worker and offline
 * preparation; the page and its other imports are the same for every test. */
const policy = globals => loadModule('policy.js', {
  document: { baseURI: 'https://test.invalid/mahjong/' }, URL, DOMException, setTimeout, clearTimeout,
  MEMORY_LIMITS_MIB, nextMemoryLimit, networkUrl, playedNetwork: () => MANIFEST, __RUNTIME_DIRECTORY__: 'ort/0123abcd/', navigator: { onLine: true },
  networkIsStored: async () => false, prepareOfflineAi: async () => null, refreshOffline: async () => null, ...globals,
}, ['chooseAction', 'analyzePolicy', 'analyzeBatch', 'modelIsAvailable', 'reportProgress', 'resetPolicy', 'LOADING_SILENCE_MS']);

test('a page that plays another listed network probes and asks for that one alone', async t => {
  const older = NETWORKS.find(network => network.version === '0.1.410');
  const probes = [], workers = [];
  class Worker {
    constructor() { this.messages = []; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() {}
  }
  const api = policy({ Worker, playedNetwork: () => older, AbortSignal,
    fetch: async (url, options) => { probes.push([url, options.method]); return new Response(null, { status: 200 }); } });
  t.after(() => api.resetPolicy());
  assert.equal(await api.modelIsAvailable(), true);
  assert.deepEqual(probes, [[networkUrl(older), 'HEAD']]);
  const move = api.chooseAction(new Float32Array(34), [1]);
  await setImmediate();
  assert.equal(workers[0].messages[0].url, networkUrl(older));
  assert.notEqual(workers[0].messages[0].url, NETWORK_URL);
  workers[0].onmessage({ data: { id: workers[0].messages[0].id, action: 0 } });
  assert.equal(await move, 0);
});

test('offline, Trained is offered only when its runtime is saved as well as the network', async () => {
  // The opponent's worker can store the network by itself, before a service
  // worker controls the page, and then nothing saves the runtime that runs it.
  for (const aiReady of [false, true]) {
    const api = policy({ navigator: { onLine: false }, networkIsStored: async () => true,
      refreshOffline: async () => ({ coreReady: true, aiReady }) });
    assert.equal(await api.modelIsAvailable(), aiReady);
  }
  // Online, the stored network is enough: the runtime comes from the server.
  const online = policy({ networkIsStored: async () => true,
    refreshOffline: async () => { throw new Error('online needs no offline status'); } });
  assert.equal(await online.modelIsAvailable(), true);
});

// Run the real request coordinator with controllable downloads and responses.
// A page plays one trained network, so every request names the same one.
test('concurrent decisions share one worker and preserve a pending turn', async (t) => {
  const downloads = [], workers = [];
  class Worker {
    constructor() { this.messages = []; this.terminated = false; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    answer(message, payload) { this.onmessage({ data: { id: message.id, ...payload } }); }
  }
  const api = policy({ Worker, prepareOfflineAi: (...args) => new Promise(resolve => downloads.push({ args, resolve })) });
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
  const api = policy({ Worker });
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
  const api = policy({ Worker });
  t.after(() => api.resetPolicy());
  return { workers, api, ask: (signal, planes = Float32Array.from({ length: 34 }, (_, i) => i / 2)) =>
    api.analyzePolicy(planes, [1, 1], signal) };
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
  // Offline preparation resolves null: no service worker, a failed start or a
  // storage failure. The timers are looked up when called, so the mocked
  // clock drives the coordinator.
  const api = policy({ Worker, setTimeout: (...args) => setTimeout(...args), clearTimeout: timer => clearTimeout(timer) });
  api.reportProgress(note => notes.push(note));
  t.after(() => api.resetPolicy());
  const ask = signal => api.chooseAction(new Float32Array(34), [1, 1], signal);
  return { api, workers, notes, ask, silence: api.LOADING_SILENCE_MS };
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
  const api = policy({ Worker, prepareOfflineAi: async () => ({ aiReady: false, unstoredNetwork: bytes }) });
  t.after(() => api.resetPolicy());
  const ask = () => api.chooseAction(new Float32Array(34), [1, 1]);
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

test('a loaded network reports nothing more, so the table goes on saying who is thinking', async t => {
  // The real coordinator and the real worker, with only the runtime stood
  // in for. Messages cross on a later turn, as they do between threads.
  class Tensor { constructor(_type, data) { this.data = data; } dispose() {} }
  const ort = { env: { wasm: {} }, Tensor, InferenceSession: { create: async () => ({
    inputNames: ['planes', 'legal'],
    run: async () => ({ policy: new Tensor('float32', Float32Array.from({ length: 46 }, (_, i) => (i === 45 ? 2 : 0))) }),
    release: async () => {},
  }) } };
  class Worker {
    constructor() {
      const page = this;
      this.self = { postMessage: data => { void setImmediate().then(() => page.onmessage?.({ data })); } };
      loadModule('policy.worker.js', { ort, policyWeights, self: this.self, URL, MEMORY_LIMITS_MIB, isMemoryError,
        networkBytes: async () => new Uint8Array(1),
        loadMemoryControls: async () => ({ memoryBudget: new MemoryBudget() }),
      }, [], { 'import(/* @vite-ignore */ controls)': 'loadMemoryControls(controls)' });
    }
    postMessage(data) { void setImmediate().then(() => this.self.onmessage({ data })); }
    terminate() {}
  }
  const api = policy({ Worker });
  const notes = [];
  api.reportProgress(note => notes.push(note));
  t.after(() => api.resetPolicy());
  const move = () => api.chooseAction(new Float32Array(34), Array.from({ length: 46 }, (_, i) => Number(i === 0 || i === 45)));
  assert.equal(await move(), 45);
  assert.deepEqual(notes.splice(0), ['loading the network', 'starting the network', 'network ready'],
    'the first move shows the network loading');
  assert.equal(await move(), 45);
  assert.equal(await move(), 45);
  // The table shows the latest note in place of "… is thinking" until the
  // opponents have moved, so a warm move must send none.
  assert.deepEqual(notes, [], 'a move with the network already loaded');
});

test('an answer proves the network is loaded, even from a worker that sends no notes', async t => {
  const { ask, workers, silence } = loadingClient(t);
  const first = ask();
  await setImmediate();
  const [worker] = workers;
  worker.say(worker.messages[0].id, { action: 1 });
  assert.equal(await first, 1);
  t.mock.timers.tick(silence);
  assert.equal(worker.terminated, false, 'no loading deadline outlives the answer');
  const late = ask(), missed = assert.rejects(late, /did not answer in time/);
  await setImmediate();
  t.mock.timers.tick(20000);
  await missed;
  assert.equal(worker.terminated, true, 'the next decision is timed at once');
});

/** The coordinator with a worker as slow as a slow phone, on the test's own
 * clock. The worker runs its requests as policy.worker.js does, one at a
 * time, in the order sent, each to its end, and says that the network is
 * ready as it begins each one; but each takes `runMs(request)`, by default
 * `positionMs` a position. Its messages cross on a later turn, as they do
 * between threads, and `tick(ms)` moves the clock and lets them through. A
 * worker that has begun `begins` requests dies without a word, and `fail()`
 * fails the request under way as a runtime out of memory does. */
function slowClient(t, { positionMs = 2400, runMs = request => (request.count ?? 1) * positionMs, begins = Infinity } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const workers = [];
  class Worker {
    constructor() { this.messages = []; this.queued = []; this.current = null; this.begun = 0; this.terminated = false; workers.push(this); }
    postMessage(message) {
      this.messages.push(message);
      if (message.cancel != null) this.queued = this.queued.filter(request => request.id !== message.cancel);
      else this.queued.push(message);
      this.next();
    }
    next() {
      if (this.current || this.terminated || this.begun >= begins || !this.queued.length) return;
      const request = this.current = this.queued.shift();
      this.begun++;
      this.say({ id: request.id, progress: 'network ready', ready: true });
      setTimeout(() => {
        if (this.current !== request) return;
        this.current = null;
        this.say(request.count ? { id: request.id, analyses: Array.from({ length: request.count }, () => ({ action: 0, weights: [1, 0] })) }
          : { id: request.id, action: 0 });
        this.next();
      }, runMs(request));
    }
    /** As policy.worker.js fails: the request under way answered with the
     * error, and nothing run again in this runtime. */
    fail() {
      const request = this.current;
      this.current = null;
      this.begun = Infinity;
      this.say({ id: request.id, error: 'RangeError: Out of memory',
        memory: { kind: 'limit', requestedBytes: (request.memoryLimitMiB + 16) * 1048576, limitMiB: request.memoryLimitMiB } });
    }
    say(data) { void setImmediate().then(() => { if (!this.terminated) this.onmessage?.({ data }); }); }
    terminate() { this.terminated = true; }
  }
  const api = policy({ Worker, setTimeout: (...args) => setTimeout(...args), clearTimeout: timer => clearTimeout(timer) });
  t.after(() => api.resetPolicy());
  const flush = async () => { for (let turn = 0; turn < 10; turn++) await setImmediate(); };
  const position = () => ({ planes: new Float32Array(34), mask: [1, 1] });
  return {
    workers, flush,
    tick: async ms => { t.mock.timers.tick(ms); await flush(); },
    // A Trained opponent's move, a question of the hand's notes, and a
    // review's batch of eight.
    move: () => api.chooseAction(new Float32Array(34), [1, 1]),
    note: () => api.analyzePolicy(new Float32Array(34), [1, 1]),
    batch: () => api.analyzeBatch(Array.from({ length: 8 }, position)),
  };
}

test('a question sent behind a review batch on a slow device waits its turn untimed, and is timed from when the worker begins it', async t => {
  // About 2.4 seconds a position, some fifteen times desktop Chromium: the
  // batch takes 19.2 seconds, and the hand's note sent just behind it is
  // answered 21.6 seconds after it was sent. Timed from then, it would run
  // out of its twenty seconds while it waited, ending the worker and the
  // review's batch with it.
  const { workers, tick, flush, move, note, batch } = slowClient(t);
  const warm = move();
  await flush();
  await tick(2400);
  assert.equal(await warm, 0, 'the network is loaded');
  const reviewed = batch(), noted = note();
  await flush();
  await tick(19200);
  assert.equal((await reviewed).length, 8);
  await tick(2400);
  assert.equal(await noted, 0, 'answered once its own position has run');
  assert.deepEqual([workers.length, workers[0].terminated], [1, false]);
});

test('the next deal\'s first move waits for a review batch the worker has begun, however long it is with the worker', async t => {
  // Next hand stops the review, but the worker runs the batch it has begun
  // to its end: 38.4 seconds at 4.8 seconds a position, within its forty.
  const { workers, tick, flush, move, batch } = slowClient(t, { positionMs: 4800 });
  const reviewed = batch();
  await flush();
  await tick(10000);
  const moved = move();
  await flush();
  await tick(28400);
  assert.equal((await reviewed).length, 8);
  await tick(4800);
  assert.equal(await moved, 0);
  assert.equal(workers[0].terminated, false);
});

test('a request still runs out of time when its own run passes its deadline, and every request waiting fails with it', async t => {
  // A batch whose eight positions take 48 seconds, past its forty, and the
  // hand's note sent behind it: the worker is ended at the batch's deadline,
  // counted from when it began, and the note fails with it, as before.
  const { workers, tick, flush, note, batch } = slowClient(t, { positionMs: 6000 });
  const reviewed = batch(), noted = note();
  const late = Promise.all([assert.rejects(reviewed, /did not answer in time/), assert.rejects(noted, /did not answer in time/)]);
  await flush();
  await tick(39999);
  assert.equal(workers[0].terminated, false, 'the batch has forty seconds from when it began');
  await tick(1);
  await late;
  assert.equal(workers[0].terminated, true);
  // Asked again, a fresh worker loads the network and answers.
  const retry = note();
  await flush();
  assert.equal(workers.length, 2, 'the retry loads a fresh worker');
  await tick(6000);
  assert.equal(await retry, 0);
});

test('a question that began behind a batch has twenty seconds from when it began, not from when it was sent', async t => {
  const { workers, tick, flush, note, batch } = slowClient(t, { runMs: request => (request.count ? 19200 : 25000) });
  const reviewed = batch(), noted = note(), late = assert.rejects(noted, /did not answer in time/);
  await flush();
  await tick(19200);
  assert.equal((await reviewed).length, 8);
  await tick(19999);
  assert.equal(workers[0].terminated, false, 'begun 19.2 seconds after it was sent');
  await tick(1);
  await late;
  assert.equal(workers[0].terminated, true, 'its own position ran past its deadline');
});

test('a request the worker never begins is ended by its backstop, its own deadline and those of the requests ahead of it', async t => {
  // The worker answers the batch and dies without a word, as a worker can
  // without an error event: the note behind it is never begun, so nothing
  // starts its own deadline. It waits the batch's forty seconds and its own
  // twenty from when it was sent, and no longer.
  const { workers, tick, flush, move, note, batch } = slowClient(t, { begins: 2 });
  const warm = move();
  await flush();
  await tick(2400);
  await warm;
  const reviewed = batch(), noted = note(), missed = assert.rejects(noted, /did not answer in time/);
  await flush();
  await tick(19200);
  assert.equal((await reviewed).length, 8);
  await tick(60000 - 19200 - 1);
  assert.equal(workers[0].terminated, false, 'the requests ahead of it might still have run');
  await tick(1);
  await missed;
  assert.equal(workers[0].terminated, true, 'ended, as a request out of time ends it');
  const retry = note();
  await flush();
  assert.equal(workers.length, 2, 'the retry loads a fresh worker');
  await tick(2400);
  assert.equal(await retry, 0);
});

test('requests a memory failure sends again are timed afresh in the larger worker, each from when it begins there', async t => {
  const { workers, tick, flush, note, batch } = slowClient(t);
  const reviewed = batch(), noted = note();
  await flush();
  await tick(10000);
  workers[0].fail();
  await flush();
  assert.equal(workers.length, 2, 'a larger worker');
  assert.deepEqual(workers[1].messages.map(message => [message.count ?? 1, message.memoryLimitMiB]), [[8, MEMORY_LIMITS_MIB[1]], [1, MEMORY_LIMITS_MIB[1]]],
    'both sent again, in order');
  await tick(19200);
  assert.equal((await reviewed).length, 8);
  await tick(2400);
  assert.equal(await noted, 0);
  assert.equal(workers[1].terminated, false);
});

test('a review batch is one request, its positions end to end, answered position by position', async t => {
  const workers = [];
  class Worker {
    constructor() { this.messages = []; this.transfers = []; workers.push(this); }
    postMessage(message, transfer = []) { this.messages.push(message); this.transfers.push(transfer); }
    terminate() {}
    answer(message, payload) { this.onmessage({ data: { id: message.id, ...payload } }); }
  }
  const api = policy({ Worker });
  t.after(() => api.resetPolicy());
  assert.equal((await api.analyzeBatch([])).length, 0, 'nothing to ask, nothing asked');
  assert.equal(workers.length, 0);
  const positions = [0, 1, 2].map(n => ({ planes: Float32Array.from({ length: 34 }, () => n), mask: [1, n === 1 ? 0 : 1] }));
  const answers = api.analyzeBatch(positions);
  positions[0].planes.fill(9); // The caller's arrays are not what is sent.
  await setImmediate();
  const [worker] = workers, [message] = worker.messages;
  assert.equal(message.count, 3);
  assert.equal(message.details, true);
  assert.equal(message.url, NETWORK_URL);
  assert.deepEqual(Array.from(message.planes), [0, 1, 2].flatMap(n => Array(34).fill(n)));
  assert.deepEqual([...message.mask], [1, 1, 1, 0, 1, 1]);
  assert.ok(worker.transfers[0].includes(message.planes.buffer), 'the planes are moved, not copied');
  const analyses = [{ action: 0, weights: [0.6, 0.4] }, { action: 0, weights: [1, 0] }, { action: 1, weights: [0.3, 0.7] }];
  worker.answer(message, { analyses });
  assert.deepEqual(await answers, analyses);
  // A single decision still goes alone, without a count.
  const single = api.analyzePolicy(new Float32Array(34), [1, 1]);
  await setImmediate();
  assert.equal('count' in worker.messages[1], false);
  worker.answer(worker.messages[1], { analysis: analyses[0] });
  assert.deepEqual(await single, analyses[0]);
});
