import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { policyWeights } from '../src/lib/policy-weights.js';
import { MemoryBudget, MEMORY_LIMITS_MIB, isMemoryError, nextMemoryLimit } from '../src/lib/memory-budget.js';
import { MANIFEST, networkUrl } from '../src/lib/trained-network.js';
import { BATCH, GPU_BATCH, batchFor, batchMostFor } from '../src/lib/game-review.js';
import { GPU_POSITION_MS, mobileDevice, openOnGpu, timedRuns, webgpuAdapter, webgpuWanted } from '../src/lib/webgpu.js';
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

test('the card is timed by the median of its runs after one warm-up run', async () => {
  const times = [1200, 70, 50, 60], ran = [];
  let clock = 0, run = 0;
  const ms = await timedRuns(async () => { ran.push(run); clock += times[run++]; }, { now: () => clock });
  assert.deepEqual(ran, [0, 1, 2, 3], 'one warm-up and three timed runs');
  assert.equal(ms, 60);
});

/** A stand-in for the WebGPU runtime, whose sessions take `ms` a run, fail to
 * open with `createError`, or fail to run from their `failFrom`th run on. */
function gpuRuntime({ ms = 40, createError = null, failFrom = Infinity } = {}) {
  const log = { created: [], opened: 0, released: 0, runs: 0, clock: 0 };
  class Tensor {
    constructor(type, data, dims) { Object.assign(this, { type, data, dims }); this.disposed = false; }
    dispose() { this.disposed = true; }
  }
  const runtime = { env: { wasm: {}, webgpu: {} }, Tensor, log, InferenceSession: { create: async (bytes, options) => {
    log.created.push({ bytes, options, adapter: runtime.env.webgpu.adapter });
    if (createError) throw createError;
    log.opened++;
    return {
      inputNames: ['planes', 'legal'],
      run: async feeds => {
        assert.ok(feeds.planes instanceof Tensor, 'the card is given its own runtime\'s tensors');
        if (++log.runs >= failFrom) throw new Error('GPU device lost');
        log.clock += ms;
        return { policy: new Tensor('float32', new Float32Array(feeds.legal.data.length).fill(1)), value: new Tensor('float32', [0.5]),
          hands: new Tensor('float32', new Float32Array(102)) };
      },
      release: async () => { log.released++; },
    };
  } } };
  return runtime;
}

const measured = runtime => ({ now: () => runtime.log.clock });
const measure = (runtime, session) => session.run({ planes: new runtime.Tensor('float32', new Float32Array(34), [1, 1, 34]),
  legal: new runtime.Tensor('float32', new Float32Array(46), [1, 46]) });

test('a fast card is chosen, and a slow one or one that fails is released for WebAssembly', async () => {
  const fast = gpuRuntime({ ms: 58 });
  const chosen = await openOnGpu('bytes', { gpu: gpu(), importRuntime: async () => fast, measure, ...measured(fast) });
  assert.equal(chosen.runtime, fast);
  assert.equal(chosen.ms, 58);
  assert.ok(chosen.session);
  assert.deepEqual(fast.log.created[0].options, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });
  assert.ok(fast.log.created[0].adapter, 'the runtime is given the adapter that was checked');
  assert.equal(fast.env.wasm.numThreads, 1);
  assert.equal(fast.log.released, 0);

  const slow = gpuRuntime({ ms: GPU_POSITION_MS + 1 });
  const refused = await openOnGpu('bytes', { gpu: gpu(), importRuntime: async () => slow, measure, ...measured(slow) });
  assert.equal(refused.session, undefined);
  assert.match(refused.reason, /101 ms a position/);
  assert.equal(slow.log.released, 1, 'a slow card lets its session go');

  const broken = gpuRuntime({ createError: new Error('no kernel') });
  assert.match((await openOnGpu('bytes', { gpu: gpu(), importRuntime: async () => broken, measure })).reason, /no kernel/);
  const lost = gpuRuntime({ failFrom: 2 });
  assert.match((await openOnGpu('bytes', { gpu: gpu(), importRuntime: async () => lost, measure })).reason, /device lost/);
  assert.equal(lost.log.released, 1);
  assert.match((await openOnGpu('bytes', { gpu: gpu(), importRuntime: async () => { throw new Error('offline'); }, measure })).reason, /offline/);
  // Without an adapter the runtime is never loaded.
  let imported = false;
  assert.match((await openOnGpu('bytes', { gpu: gpu(adapter({ isFallbackAdapter: true })), importRuntime: async () => { imported = true; }, measure })).reason,
    /no graphics card adapter/);
  assert.equal(imported, false);
});

/** The real worker with a stand-in WebAssembly runtime and WebGPU runtime. */
function worker({ gpuOptions = {}, adapterGiven = adapter(), wasmRunError = false } = {}) {
  const messages = [], wasm = { created: 0, released: 0, runs: 0 }, imports = [];
  const gpuSide = gpuRuntime(gpuOptions);
  let live = 0;
  class Tensor {
    constructor(type, data, dims) { Object.assign(this, { type, data, dims }); }
    dispose() {}
  }
  const ort = { env: { wasm: {} }, Tensor, InferenceSession: { create: async (_bytes, options) => {
    assert.equal(JSON.stringify(options.executionProviders), '["wasm"]');
    assert.ok(live++ < 1);
    wasm.created++;
    return {
      inputNames: ['planes', 'legal'],
      run: async feeds => {
        assert.ok(feeds.planes instanceof Tensor, 'WebAssembly is given its own runtime\'s tensors');
        wasm.runs++;
        if (wasmRunError) throw new Error('wasm failed');
        return { policy: new Tensor('float32', new Float32Array(feeds.legal.data.length).fill(1)), value: new Tensor('float32', [0.25]),
          hands: new Tensor('float32', new Float32Array(102)) };
      },
      release: async () => { wasm.released++; live--; },
    };
  } } };
  const self = { postMessage: message => messages.push(message), navigator: { gpu: gpu(adapterGiven) } };
  loadModule('policy.worker.js', { ort, policyWeights, self, URL, MEMORY_LIMITS_MIB, isMemoryError,
    networkBytes: async ({ url }) => `bytes of ${url}`,
    loadMemoryControls: async () => ({ memoryBudget: new MemoryBudget() }),
    loadWebgpuRuntime: async url => { imports.push(url); return gpuSide; },
    openOnGpu: (bytes, options) => openOnGpu(bytes, { ...options, now: () => gpuSide.log.clock }),
  }, [], { 'import(/* @vite-ignore */ controls)': 'loadMemoryControls(controls)',
    "import(/* @vite-ignore */ new URL('ort.webgpu.min.mjs', webgpu).href)": "loadWebgpuRuntime(new URL('ort.webgpu.min.mjs', webgpu).href)" });
  const send = (id, extra = {}) => self.onmessage({ data: { id, url: 'model.onnx', runtimeBase: 'https://test.invalid/ort/0123abcd/',
    scope: 'https://test.invalid/', planes: new Float32Array(34 * 2), mask: Array(46).fill(1), details: false,
    webgpu: 'https://test.invalid/ort-webgpu/4567cdef/', ...extra } });
  const ready = () => messages.filter(message => message.ready).map(message => message.runtime);
  return { send, messages, wasm, gpu: gpuSide, imports, ready };
}

test('the worker plays on the card where it is asked to and the card is fast', async () => {
  const w = worker();
  await w.send(1);
  await w.send(2, { count: 1 });
  assert.deepEqual(w.ready(), ['webgpu', 'webgpu']);
  assert.deepEqual(w.imports, ['https://test.invalid/ort-webgpu/4567cdef/ort.webgpu.min.mjs']);
  assert.equal(w.gpu.env.wasm.wasmPaths, 'https://test.invalid/ort-webgpu/4567cdef/', 'the stock runtime finds its files beside itself');
  assert.equal(w.wasm.created, 0, 'WebAssembly is never started');
  assert.deepEqual(w.gpu.log.created.map(entry => entry.bytes), ['bytes of model.onnx']);
  // A warm-up run and three timed ones of the first position, then the two requests.
  assert.equal(w.gpu.log.runs, 6);
  assert.equal(w.messages.find(message => message.id === 1 && 'action' in message).action, 0);
  assert.equal(w.messages.find(message => message.analyses)?.analyses.length, 1);
  assert.ok(w.messages.some(message => message.progress === 'starting the network on the graphics card'));
});

test('the worker plays on WebAssembly where the page does not ask, the card is slow, it fails to open, or there is no adapter', async () => {
  for (const [name, options, extra] of [['not asked', {}, { webgpu: undefined }],
    ['slow', { gpuOptions: { ms: GPU_POSITION_MS * 2 } }, {}],
    ['failing to open', { gpuOptions: { createError: new Error('unsupported') } }, {}],
    ['a fallback adapter', { adapterGiven: adapter({ isFallbackAdapter: true }) }, {}]]) {
    const w = worker(options);
    await w.send(1, extra);
    await w.send(2, extra);
    assert.deepEqual(w.ready(), ['wasm', 'wasm'], name);
    assert.equal(w.wasm.created, 1, name);
    assert.equal(w.messages.filter(message => 'action' in message).length, 2, name);
    assert.equal(w.messages.some(message => message.error), false, name);
    assert.equal(w.gpu.log.created.length, ['slow', 'failing to open'].includes(name) ? 1 : 0, `${name}: the card is tried once at most`);
    assert.equal(w.gpu.log.opened - w.gpu.log.released, 0, `${name}: no session is left on the card`);
  }
});

test('a card that fails while playing hands the network to WebAssembly within the same request', async () => {
  // Four runs time the card; the fifth, the first request's own, fails.
  const w = worker({ gpuOptions: { failFrom: 5 } });
  await w.send(1, { count: 2, mask: Array(92).fill(1) });
  assert.deepEqual(w.ready(), ['webgpu']);
  assert.equal(w.gpu.log.released, 1);
  assert.equal(w.wasm.created, 1);
  assert.equal(JSON.stringify(w.messages.filter(message => message.id == null)), '[{"runtime":"wasm"}]', 'the page hears of it apart from any answer');
  assert.equal(w.messages.find(message => message.analyses)?.analyses.length, 2, 'the request is answered');
  assert.equal(w.messages.some(message => message.error), false);
  await w.send(2);
  assert.deepEqual(w.ready(), ['webgpu', 'wasm']);
  assert.equal(w.gpu.log.created.length, 1, 'the card is not tried again');
});

test('a failure on WebAssembly after the card failed is the request\'s failure, as before', async () => {
  const w = worker({ gpuOptions: { failFrom: 5 }, wasmRunError: true });
  await w.send(1);
  await setImmediate();
  assert.match(w.messages.find(message => message.error)?.error ?? '', /wasm failed/);
});

/** The page's coordinator with a stand-in worker, on a desktop with WebGPU. */
function page({ network = READY, search = '' } = {}) {
  const workers = [];
  class Worker {
    constructor() { this.messages = []; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() {}
    say(data) { this.onmessage({ data }); }
  }
  const api = loadModule('policy.js', {
    document: { baseURI: 'https://test.invalid/mahjong/' }, URL, DOMException, setTimeout, clearTimeout, Worker,
    MEMORY_LIMITS_MIB, nextMemoryLimit, networkUrl, playedNetwork: () => network, __RUNTIME_DIRECTORY__: 'ort/0123abcd/',
    __WEBGPU_RUNTIME_DIRECTORY__: 'ort-webgpu/4567cdef/', webgpuWanted, navigator: { onLine: true, ...desktop() }, location: { search },
    networkIsStored: async () => false, prepareOfflineAi: async () => null, refreshOffline: async () => null,
  }, ['chooseAction', 'analyzeBatch', 'policyRuntime', 'reportRuntime', 'resetPolicy']);
  return { api, workers };
}

test('the page asks its worker to try the card for a ready network, and hears where it plays', async t => {
  const { api, workers } = page();
  t.after(() => api.resetPolicy());
  const heard = [];
  api.reportRuntime(value => heard.push(value));
  assert.equal(api.policyRuntime(), null);
  const move = api.chooseAction(new Float32Array(34), [1]);
  await setImmediate();
  const [sent] = workers[0].messages;
  assert.equal(sent.webgpu, 'https://test.invalid/mahjong/ort-webgpu/4567cdef/');
  workers[0].say({ id: sent.id, progress: 'network ready', ready: true, runtime: 'webgpu' });
  assert.equal(api.policyRuntime(), 'webgpu');
  workers[0].say({ id: sent.id, action: 0 });
  assert.equal(await move, 0);
  // A card that fails later says so in a message of its own, which answers nothing.
  const next = api.chooseAction(new Float32Array(34), [1]);
  await setImmediate();
  const second = workers[0].messages[1];
  workers[0].say({ id: second.id, progress: 'network ready', ready: true, runtime: 'webgpu' });
  workers[0].say({ runtime: 'wasm' });
  assert.equal(api.policyRuntime(), 'wasm');
  workers[0].say({ id: second.id, action: 0 });
  assert.equal(await next, 0);
  assert.deepEqual(heard, ['webgpu', 'wasm']);
  // Having found the card failing, the page does not ask for it again.
  const third = api.chooseAction(new Float32Array(34), [1]);
  await setImmediate();
  assert.equal(workers[0].messages[2].webgpu, undefined);
  workers[0].say({ id: workers[0].messages[2].id, action: 0 });
  await third;
  api.resetPolicy();
  assert.equal(api.policyRuntime(), null, 'a new worker says again where it plays');
});

test('the page does not ask for the card for a network without the flag, or with ?gpu=off', async t => {
  for (const options of [{ network: MANIFEST }, { search: '?gpu=off' }]) {
    const { api, workers } = page(options);
    t.after(() => api.resetPolicy());
    const move = api.chooseAction(new Float32Array(34), [1]);
    await setImmediate();
    assert.equal(workers[0].messages[0].webgpu, undefined, JSON.stringify(options.search ?? 'no flag'));
    workers[0].say({ id: workers[0].messages[0].id, progress: 'network ready', ready: true, runtime: 'wasm' });
    workers[0].say({ id: workers[0].messages[0].id, action: 0 });
    await move;
    assert.equal(api.policyRuntime(), 'wasm');
  }
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
