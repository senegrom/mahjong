import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { policyWeights } from '../src/lib/policy-weights.js';
import { MEMORY_LIMITS_MIB, nextMemoryLimit } from '../src/lib/memory-budget.js';
import { MANIFEST, networkUrl } from '../src/lib/trained-network.js';
import { BATCH, GPU_BATCH, batchFor, batchMostFor } from '../src/lib/game-review.js';
import { BRING_UP_MS, CARD_BATCH_MS, TIMED_RUNS, WEIGHT_BOUND, agreement, fastEnough, mobileDevice, webgpuAdapter,
  webgpuWanted } from '../src/lib/webgpu.js';
import { cardBatches } from '../src/lib/gpu-batches.js';
import { loadModule } from './fixtures/load-module.js';

const READY = { ...MANIFEST, webgpu: true };
const DESKTOP = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36';
const adapter = (info = {}) => ({ info: { vendor: 'nvidia', ...info } });
const gpu = (given = adapter()) => ({ requestAdapter: async options => {
  assert.deepEqual(options, { powerPreference: 'high-performance' });
  return given;
} });
const desktop = (extra = {}) => ({ gpu: gpu(), userAgent: DESKTOP, userAgentData: { mobile: false, platform: 'Windows' }, ...extra });

test('the page asks for the graphics card only for a ready network on a desktop with WebGPU, unless told not to', () => {
  assert.equal(webgpuWanted(READY, { navigator: desktop() }), true);
  assert.equal(webgpuWanted(READY, { search: '?mode=play', navigator: desktop() }), true);
  // An entry published before the field, or one whose graph holds a Softplus.
  assert.equal(webgpuWanted(MANIFEST, { navigator: desktop() }), false);
  assert.equal(webgpuWanted({ ...MANIFEST, webgpu: false }, { navigator: desktop() }), false);
  assert.equal(webgpuWanted({ ...MANIFEST, webgpu: 'true' }, { navigator: desktop() }), false);
  // Turned off for troubleshooting.
  assert.equal(webgpuWanted(READY, { search: '?gpu=off', navigator: desktop() }), false);
  assert.equal(webgpuWanted(READY, { search: '?opponents=neural&gpu=off', navigator: desktop() }), false);
  // A browser without WebGPU, and none at all.
  assert.equal(webgpuWanted(READY, { navigator: desktop({ gpu: undefined }) }), false);
  assert.equal(webgpuWanted(READY, {}), false);
  // A phone.
  assert.equal(webgpuWanted(READY, { navigator: desktop({ userAgentData: { mobile: true, platform: 'Android' } }) }), false);
});

test('phones and tablets are told apart from desktops, with and without client hints', () => {
  assert.equal(mobileDevice(desktop()), false);
  assert.equal(mobileDevice({ userAgentData: { mobile: true, platform: 'Android' } }), true);
  // Chromium on an Android tablet says it is not mobile; its platform says Android.
  assert.equal(mobileDevice({ userAgentData: { mobile: false, platform: 'Android' }, userAgent: DESKTOP }), true);
  assert.equal(mobileDevice({ userAgentData: { mobile: false, platform: 'macOS' } }), false);
  // Safari and Firefox give no client hints: the user agent decides.
  const iphone = 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1';
  const mac = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15';
  assert.equal(mobileDevice({ userAgent: iphone }), true);
  assert.equal(mobileDevice({ userAgent: 'Mozilla/5.0 (Android 15; Mobile; rv:144.0) Gecko/144.0 Firefox/144.0' }), true);
  assert.equal(mobileDevice({ userAgent: mac, maxTouchPoints: 0 }), false);
  // iPadOS asks for desktop pages as a Mac, but a Mac has no touch screen.
  assert.equal(mobileDevice({ userAgent: mac, maxTouchPoints: 5 }), true);
  assert.equal(mobileDevice({ userAgent: 'Mozilla/5.0 (X11; Linux x86_64; rv:144.0) Gecko/20100101 Firefox/144.0' }), false);
});

test('a fallback adapter is the processor again, and is refused like none', async () => {
  assert.ok(await webgpuAdapter(gpu()));
  assert.equal(await webgpuAdapter(gpu(null)), null);
  assert.equal(await webgpuAdapter(gpu(adapter({ isFallbackAdapter: true }))), null);
  assert.equal(await webgpuAdapter(gpu({ isFallbackAdapter: true, info: {} })), null);
  // Software renderers that do not call themselves a fallback.
  assert.equal(await webgpuAdapter(gpu(adapter({ vendor: 'google', architecture: 'swiftshader' }))), null);
  assert.equal(await webgpuAdapter(gpu(adapter({ vendor: 'mesa', description: 'llvmpipe (LLVM 19.1.7, 256 bits)' }))), null);
  assert.equal(await webgpuAdapter(gpu(adapter({ vendor: 'microsoft', description: 'Microsoft Basic Render Driver' }))), null);
  assert.equal(await webgpuAdapter(undefined), null);
  assert.equal(await webgpuAdapter({}), null);
});

test('the card agrees only where every weight is within the bound, and the best move is the same unless a near tie', () => {
  const plain = [{ action: 0, weights: [0.7, 0.3, 0] }, { action: 1, weights: [0.4995, 0.5005, 0] }];
  assert.equal(agreement(plain, plain), null);
  assert.equal(agreement(plain, plain.map(row => ({ ...row, weights: row.weights.map(w => w + WEIGHT_BOUND / 2) }))), null);
  assert.match(agreement(plain, [{ action: 0, weights: [0.7 - 2 * WEIGHT_BOUND, 0.3, 0] }, plain[1]]), /position 0, move 0/);
  // Its second position's two best are 0.001 apart, under twice the bound: either may come first.
  assert.equal(agreement(plain, [plain[0], { action: 0, weights: [0.5003, 0.4997, 0] }]), null);
  assert.match(agreement([{ action: 0, weights: [0.503, 0.497] }], [{ action: 1, weights: [0.5025, 0.4975] }]), /best move 1/);
  assert.match(agreement(plain, plain.slice(1)), /wrong number of positions/);
  assert.match(agreement(plain, [plain[0], { action: 1, weights: [NaN, 1, 0] }]), /position 1, move 0/);
  // Twice as fast at least.
  assert.equal(fastEnough(50, 100), true);
  assert.equal(fastEnough(51, 100), false);
});

/** A clock that moves only when a test moves it, for every timer the card's
 * module uses. */
function clock() {
  let now = 0, last = 0;
  const timers = new Map();
  return {
    now: () => now,
    setTimeout: (fn, ms) => { timers.set(++last, { at: now + ms, fn }); return last; },
    clearTimeout: id => { timers.delete(id); },
    pass(ms) { now += ms; },
    advance(ms) {
      const end = now + ms;
      for (;;) {
        const due = [...timers].filter(([, timer]) => timer.at <= end).sort(([, a], [, b]) => a.at - b.at)[0];
        if (!due) break;
        timers.delete(due[0]); now = due[1].at; due[1].fn();
      }
      now = end;
    },
  };
}
const settle = async (turns = 20) => { for (let turn = 0; turn < turns; turn++) await setImmediate(); };
/** A position, and WebAssembly's answer to it. */
const position = index => ({ planes: Float32Array.of(index, 1), mask: [1, 1] });
const plainAnswer = () => ({ action: 0, weights: [0.7, 0.3] });

/** The card's batches with a stand-in for its worker and for WebAssembly.
 * WebAssembly takes `wasmMs` a batch; the card answers its requests as
 * `answer(message)` says, after `cardMs` (null for no answer). */
function setup({ storage = memoryStorage(), search = '', nav = desktop(), network = READY, runtime = 'ort-webgpu/aaaa/',
  wasmMs = 1000, cardMs = 50, answer } = {}) {
  const time = clock(), workers = [], wasmCalls = [], warnings = [], runtimes = [];
  const reply = answer ?? (message => (message.open ? { opened: true }
    : { analyses: Array.from({ length: message.count }, plainAnswer), ms: cardMs }));
  class FakeWorker {
    constructor() { this.messages = []; this.terminated = false; workers.push(this); }
    postMessage(message) {
      this.messages.push(message);
      const answered = reply(message, this);
      if (answered == null) return;
      Promise.resolve().then(() => {
        if (!message.open) time.pass(cardMs);
        this.onmessage?.({ data: { id: message.id, ...answered } });
      });
    }
    terminate() { this.terminated = true; }
  }
  const card = cardBatches({ createWorker: () => new FakeWorker(), storage, search, navigator: nav, network: () => network,
    runtime, scope: 'https://test.invalid/mahjong/', now: time.now, setTimeout: time.setTimeout, clearTimeout: time.clearTimeout,
    warn: message => warnings.push(message), onRuntime: value => runtimes.push(value) });
  const wasm = async (positions, signal) => {
    wasmCalls.push(positions.length);
    time.pass(wasmMs);
    if (signal?.aborted) throw new DOMException('Match changed', 'AbortError');
    return positions.map(plainAnswer);
  };
  const ask = (count, signal) => card.analyze(Array.from({ length: count }, (_, index) => position(index)), signal, wasm);
  return { card, ask, time, workers, wasmCalls, warnings, runtimes, storage };
}
function memoryStorage() {
  const entries = new Map();
  return { entries, getItem: key => entries.get(key) ?? null, setItem: (key, value) => entries.set(key, String(value)),
    removeItem: key => entries.delete(key) };
}
/** Brings the card up: two batches of eight on WebAssembly, then the bring-up. */
async function bringUp(s) {
  await s.ask(8);
  await s.ask(8);
  await settle();
}

test('the card is brought up only after WebAssembly has answered a second batch of two or more, and never delays it', async () => {
  const s = setup({ answer: message => (message.open ? null : { analyses: Array.from({ length: message.count }, plainAnswer), ms: 50 }) });
  await s.ask(1);
  await s.ask(8);
  await s.ask(1);
  assert.equal(s.workers.length, 0, 'single positions and the first batch bring nothing up');
  const answered = await s.ask(8);
  assert.equal(answered.length, 8, 'the batch that starts the bring-up is answered on WebAssembly at once');
  assert.equal(s.workers.length, 1);
  assert.equal(s.card.state(), 'starting');
  assert.equal(s.workers[0].messages[0].open.url, networkUrl(READY));
  assert.deepEqual({ ...s.workers[0].messages[0].open.expect }, { bytes: READY.bytes, sha256: READY.sha256 });
  assert.equal(s.workers[0].messages[0].open.webgpu, 'https://test.invalid/mahjong/ort-webgpu/aaaa/');
  // While it starts, everything still goes to WebAssembly.
  await s.ask(8);
  assert.deepEqual(s.wasmCalls, [1, 8, 1, 8, 8]);
});

test('a card that agrees and is twice as fast serves every batch of two or more, and single positions stay on WebAssembly', async () => {
  const s = setup();
  await bringUp(s);
  assert.equal(s.card.state(), 'ready');
  assert.equal(s.card.runtime(), 'webgpu');
  assert.deepEqual(s.runtimes, ['webgpu']);
  // A warm-up run and TIMED_RUNS timed ones of the sample.
  assert.deepEqual(s.workers[0].messages.slice(1).map(message => message.count), Array(TIMED_RUNS + 1).fill(8));
  const before = s.wasmCalls.length;
  assert.equal((await s.ask(64)).length, 64);
  assert.equal((await s.ask(1)).length, 1);
  assert.deepEqual(s.wasmCalls.slice(before), [1], 'only the single position went to WebAssembly');
  assert.equal(s.workers[0].messages.at(-1).count, 64);
  assert.deepEqual(s.warnings, []);
});

test('a card that disagrees with WebAssembly, or is not twice as fast, is declined and remembered', async () => {
  for (const [name, options, reason] of [
    ['disagrees', { answer: message => (message.open ? { opened: true }
      : { analyses: Array.from({ length: message.count }, () => ({ action: 1, weights: [0.3, 0.7] })) }) }, /disagrees/],
    ['slow', { wasmMs: 1000, cardMs: 501 }, /took 501 ms for a batch WebAssembly ran in 1000 ms/],
  ]) {
    const s = setup(options);
    await bringUp(s);
    assert.equal(s.card.state(), 'declined', name);
    assert.equal(s.card.runtime(), 'wasm');
    assert.ok(s.workers[0].terminated, name);
    assert.equal(s.warnings.length, 1);
    assert.match(s.warnings[0], reason);
    assert.equal(s.storage.entries.size, 1, `${name}: remembered`);
    const before = s.wasmCalls.length;
    await s.ask(8);
    assert.equal(s.wasmCalls.length, before + 1, `${name}: batches stay on WebAssembly`);
  }
});

test('a bring-up that takes longer than its 60 seconds is declined', async () => {
  const s = setup({ answer: message => (message.open ? null : {}) });
  await bringUp(s);
  assert.equal(s.card.state(), 'starting');
  s.time.advance(BRING_UP_MS - 1);
  assert.equal(s.card.state(), 'starting');
  s.time.advance(1);
  await settle();
  assert.equal(s.card.state(), 'declined');
  assert.ok(s.workers[0].terminated);
  assert.match(s.warnings[0], /longer than 60 s/);
  assert.equal(s.storage.entries.size, 1);
});

test('a remembered decline holds on later visits until the runtime, the network or ?gpu=on says to try again', async () => {
  const storage = memoryStorage();
  const first = setup({ storage, wasmMs: 100, cardMs: 90 });
  await bringUp(first);
  assert.equal(first.card.state(), 'declined');
  const [[key, record]] = storage.entries;
  assert.equal(key, 'mahjong-webgpu-declined:/mahjong/');
  assert.equal(JSON.parse(record).runtime, 'ort-webgpu/aaaa/');
  assert.equal(JSON.parse(record).network, READY.sha256);
  // The next visit does not try, and says nothing.
  const again = setup({ storage });
  await bringUp(again);
  assert.equal(again.workers.length, 0);
  assert.equal(again.card.state(), 'declined');
  assert.deepEqual(again.warnings, []);
  // A new runtime, a new network, or ?gpu=on, tries once again.
  for (const options of [{ runtime: 'ort-webgpu/bbbb/' }, { network: { ...READY, sha256: 'f'.repeat(64) } }, { search: '?gpu=on' }]) {
    const fresh = setup({ storage, ...options });
    await bringUp(fresh);
    assert.equal(fresh.card.state(), 'ready', JSON.stringify(options));
  }
  assert.equal(storage.entries.size, 0, '?gpu=on forgot the decline');
  // ?gpu=off never tries.
  const off = setup({ search: '?gpu=off' });
  await bringUp(off);
  assert.equal(off.workers.length, 0);
  assert.equal(off.card.state(), 'declined');
});

test('a runtime the browser could not fetch, or a network not saved, declines the card for this visit only', async () => {
  for (const error of ['the WebGPU runtime could not be fetched: HTTP 503', 'the network is not saved on this device']) {
    const storage = memoryStorage();
    const s = setup({ storage, answer: message => (message.open ? { error, remember: false } : {}) });
    await bringUp(s);
    assert.equal(s.card.state(), 'declined');
    assert.equal(s.warnings[0], `Trained AI reviews on WebAssembly: ${error}`);
    assert.equal(storage.entries.size, 0);
    const next = setup({ storage });
    await bringUp(next);
    assert.equal(next.card.state(), 'ready', 'the next visit tries again');
  }
});

test('a batch that fails or runs out of time on the card is answered on WebAssembly, and the card is declined', async () => {
  for (const [name, failure] of [['fails', { error: 'a batch failed on the card: device lost', remember: true }], ['overruns', null]]) {
    let failing = false;
    const s = setup({ answer: message => (message.open ? { opened: true } : failing ? failure
      : { analyses: Array.from({ length: message.count }, plainAnswer) }) });
    await bringUp(s);
    assert.equal(s.card.state(), 'ready', name);
    failing = true;
    const before = s.wasmCalls.length;
    const answer = s.ask(16);
    await settle();
    if (!failure) s.time.advance(CARD_BATCH_MS);
    const answers = await answer;
    assert.equal(answers.length, 16, `${name}: the caller gets its answers`);
    assert.deepEqual(s.wasmCalls.slice(before), [16], `${name}: from WebAssembly`);
    assert.equal(s.card.state(), 'declined');
    assert.ok(s.workers[0].terminated);
    assert.deepEqual(s.runtimes, ['webgpu', 'wasm']);
    assert.equal(s.storage.entries.size, 1);
  }
});

test('a caller who stops waiting is not a failure of the card, and a reset drops waiting batches but keeps the card', async () => {
  let hold = false;
  const s = setup({ answer: message => (message.open ? { opened: true } : hold ? null
    : { analyses: Array.from({ length: message.count }, plainAnswer) }) });
  await bringUp(s);
  hold = true;
  const stop = new AbortController();
  const stopped = s.ask(16, stop.signal);
  await settle();
  stop.abort();
  await assert.rejects(stopped, { name: 'AbortError' });
  assert.equal(s.card.state(), 'ready');
  // A reset tells a waiting caller its own reason, and the card stays.
  const waiting = s.ask(16);
  await settle();
  const reason = new Error('Match changed');
  s.card.reset(reason);
  await assert.rejects(waiting, error => error === reason);
  assert.equal(s.card.state(), 'ready');
  assert.equal(s.workers[0].terminated, false);
});

test('storage that throws breaks nothing', async () => {
  const broken = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); }, removeItem() { throw new Error('denied'); } };
  const ready = setup({ storage: broken, search: '?gpu=on' });
  await bringUp(ready);
  assert.equal(ready.card.state(), 'ready');
  const slow = setup({ storage: broken, wasmMs: 100, cardMs: 90 });
  await bringUp(slow);
  assert.equal(slow.card.state(), 'declined');
  assert.equal((await slow.ask(8)).length, 8);
});

test('the page sends single decisions to its WebAssembly worker alone, and batches through the card', async t => {
  const workers = [], routed = [];
  class Worker {
    constructor() { this.messages = []; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() {}
  }
  const api = loadModule('policy.js', {
    document: { baseURI: 'https://test.invalid/mahjong/' }, URL, DOMException, setTimeout, clearTimeout, Worker,
    MEMORY_LIMITS_MIB, nextMemoryLimit, networkUrl, playedNetwork: () => READY, __RUNTIME_DIRECTORY__: 'ort/0123abcd/',
    navigator: { onLine: true }, networkIsStored: async () => false, prepareOfflineAi: async () => null, refreshOffline: async () => null,
    batchesOnCard: (positions, signal, wasm) => { routed.push(positions.length); return wasm(positions, signal); }, resetCard: () => {},
  }, ['chooseAction', 'analyzePolicy', 'analyzeBatch', 'resetPolicy']);
  t.after(() => api.resetPolicy());
  const answered = async (promise, reply) => {
    await setImmediate();
    const worker = workers.at(-1), message = worker.messages.at(-1);
    worker.onmessage({ data: { id: message.id, ...reply(message) } });
    return promise;
  };
  assert.equal(await answered(api.chooseAction(new Float32Array(34), [1, 1]), () => ({ action: 1 })), 1);
  await answered(api.analyzePolicy(new Float32Array(34), [1, 1]), () => ({ analysis: { action: 0 } }));
  assert.deepEqual(routed, [], 'no single decision is offered to the card');
  const positions = [0, 1].map(index => ({ planes: new Float32Array(34).fill(index), mask: [1, 1] }));
  const batch = await answered(api.analyzeBatch(positions),
    message => ({ analyses: Array.from({ length: message.count }, () => policyWeights([1, 0], [1, 1])) }));
  assert.equal(batch.length, 2);
  assert.deepEqual(routed, [2]);
});

/** The card's worker, with a stand-in runtime, adapter, stored network and fetch. */
function cardWorker({ adapterGiven = adapter(), stored = Uint8Array.of(1, 2, 3), fetchStatus = 200, createError = null, runError = null } = {}) {
  const messages = [], imports = [], created = [], reads = [];
  class Tensor {
    constructor(type, data, dims) { Object.assign(this, { type, data, dims }); }
    dispose() {}
  }
  const runtime = { env: { wasm: {}, webgpu: {} }, Tensor, InferenceSession: { create: async (bytes, options) => {
    created.push({ bytes, options, adapter: runtime.env.webgpu.adapter });
    if (createError) throw createError;
    return { inputNames: ['planes', 'legal'], run: async feeds => {
      if (runError) throw runError;
      return { policy: new Tensor('float32', Float32Array.from(feeds.legal.data, (_, index) => (index % 2 ? 0 : 1))),
        value: new Tensor('float32', [0]), hands: new Tensor('float32', [0]) };
    } };
  } } };
  const self = { postMessage: message => messages.push(message), navigator: { gpu: gpu(adapterGiven) } };
  loadModule('gpu-batch.worker.js', { self, URL, Error, policyWeights, performance, Promise, Float32Array, Array, Object, String,
    webgpuAdapter, storedNetworkBytes: async read => { reads.push(read); return stored; },
    fetch: async () => new Response('runtime', { status: fetchStatus }),
    loadRuntime: async url => { imports.push(url); return runtime; },
  }, [], { "import(/* @vite-ignore */ new URL('ort.min.mjs', webgpu).href)": "loadRuntime(new URL('ort.min.mjs', webgpu).href)" });
  const send = data => { self.onmessage({ data }); return settle(); };
  const open = { url: 'https://cdn.invalid/models/g3000/x', expect: { bytes: 3, sha256: 'a'.repeat(64) }, scope: 'https://test.invalid/mahjong/',
    webgpu: 'https://test.invalid/mahjong/ort-webgpu/aaaa/' };
  return { send, open, messages, imports, created, reads, runtime };
}

test('the card\'s worker opens the stored network on WebGPU alone and answers batches with their analyses and time', async () => {
  const w = cardWorker();
  await w.send({ id: 1, open: w.open });
  assert.equal(JSON.stringify(w.messages.find(message => message.id === 1)), '{"id":1,"opened":true}');
  assert.equal(w.reads[0].scope, 'https://test.invalid/mahjong/', 'the network is read from the page\'s cache');
  assert.equal(w.reads[0].expect.sha256, 'a'.repeat(64));
  assert.deepEqual(w.imports, ['https://test.invalid/mahjong/ort-webgpu/aaaa/ort.min.mjs']);
  assert.equal(w.runtime.env.wasm.wasmPaths, 'https://test.invalid/mahjong/ort-webgpu/aaaa/');
  assert.equal(JSON.stringify(w.created[0].options), '{"executionProviders":["webgpu"],"graphOptimizationLevel":"all"}');
  assert.ok(w.created[0].adapter, 'the runtime is given the adapter that was checked');
  await w.send({ id: 2, planes: new Float32Array(68), mask: [1, 1, 1, 1], count: 2 });
  const answer = w.messages.find(message => message.id === 2);
  assert.equal(answer.analyses.length, 2);
  assert.equal(answer.analyses[0].action, 0);
  assert.ok(answer.ms >= 0);
});

test('the card\'s worker says why it cannot serve, and whether that is the device\'s', async () => {
  for (const [name, options, reason, remember] of [
    ['no adapter', { adapterGiven: null }, /no graphics card adapter/, true],
    ['a software adapter', { adapterGiven: adapter({ architecture: 'swiftshader' }) }, /no graphics card adapter/, true],
    ['a network not saved', { stored: null }, /not saved on this device/, false],
    ['a runtime it cannot fetch', { fetchStatus: 503 }, /could not be fetched: .*HTTP 503/, false],
    ['a session that fails', { createError: new Error('no kernel') }, /could not be opened on the card: no kernel/, true],
  ]) {
    const w = cardWorker(options);
    await w.send({ id: 1, open: w.open });
    const said = w.messages.find(message => message.id === 1);
    assert.match(said.error, reason, name);
    assert.equal(said.remember, remember, name);
    if (name !== 'a session that fails') assert.equal(w.created.length, 0, `${name}: no session is made`);
  }
  const failing = cardWorker({ runError: new Error('device lost') });
  await failing.send({ id: 1, open: failing.open });
  await failing.send({ id: 2, planes: new Float32Array(68), mask: [1, 1, 1, 1], count: 2 });
  assert.equal(JSON.stringify(failing.messages.find(message => message.id === 2)),
    '{"id":2,"error":"a batch failed on the card: device lost","remember":true}');
});

test('a review asks up to GPU_BATCH positions a pass on the card, and BATCH otherwise', () => {
  assert.equal(GPU_BATCH, 64);
  assert.equal(batchMostFor('webgpu'), GPU_BATCH);
  assert.equal(batchMostFor('wasm'), BATCH);
  assert.equal(batchMostFor(null), BATCH);
  // Before anything is measured, and on a card that takes a millisecond a position.
  assert.equal(batchFor(null, { most: GPU_BATCH }), GPU_BATCH);
  assert.equal(batchFor(1, { most: GPU_BATCH }), GPU_BATCH);
  assert.equal(batchFor(1), BATCH);
  // The ten seconds a pass may take still bound it: a slow device asks fewer.
  assert.equal(batchFor(500, { most: GPU_BATCH }), 20);
  assert.equal(batchFor(20000, { most: GPU_BATCH }), 1);
});
