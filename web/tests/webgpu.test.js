import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { setImmediate } from 'node:timers/promises';
import { policyWeights } from '../src/lib/policy-weights.js';
import { MEMORY_LIMITS_MIB, nextMemoryLimit } from '../src/lib/memory-budget.js';
import { MANIFEST, networkUrl } from '../src/lib/trained-network.js';
import { BATCH, GPU_BATCH, batchFor, batchMostFor } from '../src/lib/game-review.js';
import { BRING_UP_MS, CARD_BATCH_MS, DECLINE_LASTS_MS, RUNTIME_MS, TIMED_POSITIONS, TIMED_RUNS, WASM_PASS, WEIGHT_BOUND, agreement,
  fastEnough, median, mobileDevice, timedOrder, webgpuAdapter, webgpuWanted } from '../src/lib/webgpu.js';
import { CARD_AT_WORK, cardBatches } from '../src/lib/gpu-batches.js';
import { resolveObjectURL } from 'node:buffer';
import { loadModule } from './fixtures/load-module.js';

const READY = { ...MANIFEST, webgpu: true };
// The default network as an entry published before the field held it.
const { webgpu: _flag, ...BEFORE } = MANIFEST;
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
  assert.equal(webgpuWanted(BEFORE, { navigator: desktop() }), false);
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

test('the card is timed on a hand\'s worth of positions, the sample\'s own again and again', () => {
  assert.equal(TIMED_POSITIONS, 16);
  // The end of a hand, two positions, eight times over.
  assert.deepEqual(timedOrder(2), [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
  // A full pass of eight, twice over.
  assert.deepEqual(timedOrder(8), [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
  // Five do not divide sixteen: the last round stops short.
  assert.deepEqual(timedOrder(5), [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]);
  // A sample of more is timed as it is.
  assert.deepEqual(timedOrder(20), Array.from({ length: 20 }, (_, index) => index));
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

/** The day the tests are run on, in milliseconds. */
const TODAY = Date.UTC(2026, 10, 12);
/** Another network than READY, as ready for the card. */
const OTHER = { ...READY, version: '0.1.9999', object: `models/g9999/${'e'.repeat(64)}`, sha256: 'e'.repeat(64) };
/** The card's batches with a stand-in for its worker and for WebAssembly.
 * WebAssembly takes `wasmMs` a batch, or what `wasmMs(call, count)` says for
 * its call of that number, the first being 1; the card answers its requests
 * as `answer(message)` says, after `cardMs` (null for no answer), and says
 * that its runtime is in hand as soon as it is asked to open a network,
 * unless `fetched` is false. `play` changes the network the page plays. */
function setup({ storage = memoryStorage(), search = '', nav = desktop(), network = READY, runtime = 'ort-webgpu/aaaa/',
  wasmMs = 1000, cardMs = 50, answer, today = TODAY, fetched = true } = {}) {
  const time = clock(), workers = [], wasmCalls = [], warnings = [], runtimes = [], works = [], playing = { network };
  const reply = answer ?? (message => (message.open ? { opened: true }
    : { analyses: Array.from({ length: message.count }, plainAnswer), ms: cardMs }));
  class FakeWorker {
    constructor() { this.messages = []; this.terminated = false; workers.push(this); }
    postMessage(message) {
      this.messages.push(message);
      if (message.open && fetched) Promise.resolve().then(() => this.onmessage?.({ data: { id: message.id, fetched: true } }));
      const answered = reply(message, this);
      if (answered == null) return;
      Promise.resolve().then(() => {
        if (!message.open) time.pass(cardMs);
        this.onmessage?.({ data: { id: message.id, ...answered } });
      });
    }
    terminate() { this.terminated = true; }
  }
  const card = cardBatches({ createWorker: () => new FakeWorker(), storage, search, navigator: nav, network: () => playing.network,
    runtime, scope: 'https://test.invalid/mahjong/', now: time.now, setTimeout: time.setTimeout, clearTimeout: time.clearTimeout,
    today: () => today, warn: message => warnings.push(message), onRuntime: value => runtimes.push(value),
    onWork: working => works.push(working) });
  const wasm = async (positions, signal) => {
    wasmCalls.push(positions.length);
    time.pass(typeof wasmMs === 'function' ? wasmMs(wasmCalls.length, positions.length) : wasmMs);
    if (signal?.aborted) throw new DOMException('Match changed', 'AbortError');
    // Each answer says which position it is of, for the tests of their order.
    return positions.map(asked => ({ ...plainAnswer(), of: asked.planes[0] }));
  };
  const ask = (count, signal) => card.analyze(Array.from({ length: count }, (_, index) => position(index)), signal, wasm);
  return { card, ask, time, workers, wasmCalls, warnings, runtimes, works, storage, play: next => { playing.network = next; } };
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
  // A warm-up run and TIMED_RUNS timed ones, of the sample's eight positions twice over.
  const runs = s.workers[0].messages.slice(1);
  assert.deepEqual(runs.map(message => message.count), Array(TIMED_RUNS + 1).fill(TIMED_POSITIONS));
  const twice = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];
  assert.deepEqual(Array.from(runs[0].planes), twice.flatMap(index => [index, 1]));
  assert.equal(runs[0].mask.length, 2 * TIMED_POSITIONS);
  const before = s.wasmCalls.length;
  assert.equal((await s.ask(64)).length, 64);
  assert.equal((await s.ask(1)).length, 1);
  assert.deepEqual(s.wasmCalls.slice(before), [1], 'only the single position went to WebAssembly');
  assert.equal(s.workers[0].messages.at(-1).count, 64);
  assert.deepEqual(s.warnings, []);
});

test('the end of a hand is sample enough: the card is timed on sixteen positions, not on its two', async () => {
  // Two positions in 250 ms on WebAssembly, and 150 ms a batch on the card: not twice as fast
  // on the two, eight times as fast on a hand.
  const s = setup({ wasmMs: 250, cardMs: 150 });
  await s.ask(8);
  await s.ask(2);
  await settle();
  assert.equal(s.card.state(), 'ready');
  const runs = s.workers[0].messages.slice(1);
  assert.deepEqual(runs.map(message => message.count), Array(TIMED_RUNS + 1).fill(TIMED_POSITIONS));
  assert.deepEqual(Array.from(runs[0].planes), Array.from({ length: TIMED_POSITIONS }, (_, index) => [index % 2, 1]).flat());
  // A card that answers the second of the two otherwise, wherever it stands in the batch, disagrees.
  const wrong = setup({ answer: message => (message.open ? { opened: true } : { analyses: Array.from({ length: message.count },
    (_, index) => (index === 15 ? { action: 1, weights: [0.3, 0.7] } : plainAnswer())) }) });
  await wrong.ask(8);
  await wrong.ask(2);
  await settle();
  assert.equal(wrong.card.state(), 'declined');
  assert.match(wrong.warnings[0], /disagrees with WebAssembly: position 15/);
});

test('WebAssembly\'s time a position is the least any batch has shown, so a sample that waited its turn lets no slow card pass', async () => {
  // Eight positions take WebAssembly 1000 ms, two take 250. The sample, two positions, waited a second
  // behind another request: eight times over, its 1250 ms would say sixteen positions take 10000.
  const waited = (call, count) => (call === 2 ? 1250 : count * 125);
  // The card takes 3000 ms for sixteen, half as long again as WebAssembly's 2000.
  const answers = held => message => (message.open ? (held.open ? null : { opened: true })
    : { analyses: Array.from({ length: message.count }, plainAnswer) });
  const held = { open: true };
  const s = setup({ wasmMs: waited, cardMs: 3000, answer: answers(held) });
  await s.ask(8);
  await s.ask(2);
  assert.equal(s.card.state(), 'starting');
  // While the card's network opens, WebAssembly answers the next pass at its own pace.
  await s.ask(8);
  held.open = false;
  const opening = s.workers[0].messages[0];
  s.workers[0].onmessage({ data: { id: opening.id, opened: true } });
  await settle();
  assert.equal(s.card.state(), 'declined');
  assert.match(s.warnings[0], /took 3000 ms for 16 positions, which WebAssembly runs in 2000 ms/);
  // With nothing else to go by, the sample's own time stands.
  const alone = setup({ wasmMs: waited, cardMs: 3000 });
  await alone.ask(8);
  await alone.ask(2);
  await settle();
  assert.equal(alone.card.state(), 'ready');
});

test('another network played starts the card over, and blames no card for it', async () => {
  // In the middle of a bring-up: its worker is ended, and its 60 seconds no longer run.
  let hold = true;
  const answer = message => (message.open ? { opened: true } : hold ? null
    : { analyses: Array.from({ length: message.count }, plainAnswer) });
  const starting = setup({ answer });
  await bringUp(starting);
  assert.equal(starting.card.state(), 'starting');
  assert.deepEqual(starting.works, [true], 'the warm-up run is on the card');
  starting.play(OTHER);
  assert.equal((await starting.ask(8)).length, 8);
  await settle();
  assert.equal(starting.card.state(), 'untried');
  assert.ok(starting.workers[0].terminated);
  assert.deepEqual(starting.works, [true, false]);
  starting.time.advance(BRING_UP_MS);
  await settle();
  assert.equal(starting.card.state(), 'untried', 'the old bring-up\'s time is not this card\'s');
  assert.deepEqual(starting.warnings, []);
  assert.equal(starting.storage.entries.size, 0);
  // The card is brought up for the new network on its second batch.
  hold = false;
  await starting.ask(8);
  await settle();
  assert.equal(starting.workers.length, 2);
  assert.equal(starting.workers[1].messages[0].open.url, networkUrl(OTHER));
  assert.equal(starting.workers[1].messages[0].open.expect.sha256, OTHER.sha256);
  assert.equal(starting.card.state(), 'ready');
  assert.deepEqual(starting.runtimes, ['webgpu']);

  // From a card that serves: the page hears that batches run on WebAssembly again, and a batch still
  // on the card, its caller waiting, is answered there, nothing declined and nothing remembered.
  hold = false;
  const serving = setup({ answer });
  await bringUp(serving);
  assert.equal(serving.card.state(), 'ready');
  hold = true;
  const waiting = serving.ask(16);
  await settle();
  serving.play(OTHER);
  const before = serving.wasmCalls.length;
  const next = serving.ask(8);
  assert.equal((await waiting).length, 16);
  assert.equal((await next).length, 8);
  // The sixteen in WebAssembly's passes of eight, and the eight asked after.
  assert.deepEqual(serving.wasmCalls.slice(before), [8, 8, 8]);
  assert.deepEqual(serving.runtimes, ['webgpu', 'wasm']);
  assert.ok(serving.workers[0].terminated);
  assert.deepEqual(serving.warnings, []);
  assert.equal(serving.storage.entries.size, 0);
  assert.notEqual(serving.card.state(), 'declined');
});

test('a decline is remembered under the network the card was tried for, and another network played is noticed while the runtime is still fetched', async () => {
  // The bring-up's 60 seconds run out after the page has gone on to another network, and before it asks again.
  const late = setup({ answer: message => (message.open ? null : {}) });
  await bringUp(late);
  late.play(OTHER);
  late.time.advance(BRING_UP_MS);
  await settle();
  assert.equal(late.card.state(), 'declined');
  const [[, record]] = late.storage.entries;
  assert.equal(JSON.parse(record).network, READY.sha256, 'not under the network that was never tried');
  // Another network played while the first one's worker still fetches its runtime: the card starts over for it.
  let hold = true;
  const s = setup({ fetched: false, answer: message => (message.open ? (hold ? null : { opened: true })
    : { analyses: Array.from({ length: message.count }, plainAnswer) }) });
  await bringUp(s);
  assert.equal(s.card.state(), 'starting');
  s.play(OTHER);
  await s.ask(8);
  assert.equal(s.card.state(), 'untried');
  assert.ok(s.workers[0].terminated);
  s.time.advance(RUNTIME_MS);
  await settle();
  assert.equal(s.card.state(), 'untried', 'the old fetch\'s ten minutes are not this card\'s');
  hold = false;
  await s.ask(8);
  await settle();
  assert.equal(s.workers[1].messages[0].open.expect.sha256, OTHER.sha256);
  assert.equal(s.card.state(), 'ready');
  assert.deepEqual(s.warnings, []);
  assert.equal(s.storage.entries.size, 0);
});

test('a worker that does not load, or whose answer cannot be read, declines the card for this visit only', async () => {
  for (const [name, fail, reason] of [
    ['its file is gone', worker => worker.onerror({ message: 'Failed to load script' }), /the card's worker failed: Failed to load script/],
    ['its answer is unreadable', worker => worker.onmessageerror({}), /sent an unreadable answer/],
  ]) {
    // In the bring-up, where a worker that did not load has said nothing,
    // and once the card serves.
    for (const serving of [false, true]) {
      const s = setup({ fetched: serving, answer: message => (message.open ? (serving ? { opened: true } : null)
        : { analyses: Array.from({ length: message.count }, plainAnswer) }) });
      await bringUp(s);
      assert.equal(s.card.state(), serving ? 'ready' : 'starting', name);
      fail(s.workers[0]);
      await settle();
      assert.equal(s.card.state(), 'declined', name);
      assert.match(s.warnings[0], reason);
      assert.equal(s.storage.entries.size, 0, `${name}: nothing is said of the device`);
      assert.equal((await s.ask(8)).length, 8);
    }
  }
});

test('a worker that had answered and then fails outside any request in the bring-up is the device\'s, and remembered', async () => {
  // It said that its runtime is in hand, so its file loaded; what fails then
  // without an answer, as the runtime does where it gives up, is not the
  // connection's. An answer that could not be read still says nothing.
  for (const [name, fail, remembers] of [
    ['the runtime gives up', worker => worker.onerror({ message: 'RuntimeError: Aborted(OOM)' }), true],
    ['its answer is unreadable', worker => worker.onmessageerror({}), false],
  ]) {
    const s = setup({ answer: message => (message.open ? null : { analyses: Array.from({ length: message.count }, plainAnswer) }) });
    await bringUp(s);
    assert.equal(s.card.state(), 'starting', name);
    fail(s.workers[0]);
    await settle();
    assert.equal(s.card.state(), 'declined', name);
    assert.equal(s.storage.entries.size, remembers ? 1 : 0, name);
    if (!remembers) continue;
    const record = JSON.parse([...s.storage.entries.values()][0]);
    assert.equal(record.reason, 'the card\'s worker failed: RuntimeError: Aborted(OOM)');
    assert.equal(record.network, READY.sha256);
    // The next visit does not try again.
    const next = setup({ storage: s.storage });
    await bringUp(next);
    assert.equal(next.workers.length, 0);
    assert.equal(next.card.state(), 'declined');
  }
});

test('a card that disagrees with WebAssembly, or is not twice as fast, is declined and remembered', async () => {
  // Eight positions in 1000 ms on WebAssembly are sixteen in 2000.
  const fast = setup({ wasmMs: 1000, cardMs: 1000 });
  await bringUp(fast);
  assert.equal(fast.card.state(), 'ready', 'exactly twice as fast is fast enough');
  for (const [name, options, reason] of [
    ['disagrees', { answer: message => (message.open ? { opened: true }
      : { analyses: Array.from({ length: message.count }, () => ({ action: 1, weights: [0.3, 0.7] })) }) }, /disagrees/],
    ['slow', { wasmMs: 1000, cardMs: 1001 }, /took 1001 ms for 16 positions, which WebAssembly runs in 2000 ms/],
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

test('fetching the runtime is the connection\'s time: its own ten minutes, the 60 seconds only from there, and out of it nothing is remembered', async () => {
  // The worker has not said that its runtime is in hand: the bring-up's 60 seconds do not run yet.
  const slow = setup({ fetched: false, answer: message => (message.open ? null : { analyses: Array.from({ length: message.count }, plainAnswer) }) });
  await bringUp(slow);
  slow.time.advance(BRING_UP_MS);
  await settle();
  assert.equal(slow.card.state(), 'starting');
  // Five minutes on it says so, and has its 60 seconds from that moment.
  slow.time.advance(5 * 60000 - BRING_UP_MS);
  const opening = slow.workers[0].messages[0];
  slow.workers[0].onmessage({ data: { id: opening.id, fetched: true } });
  slow.time.advance(BRING_UP_MS - 1);
  await settle();
  assert.equal(slow.card.state(), 'starting');
  slow.workers[0].onmessage({ data: { id: opening.id, opened: true } });
  await settle();
  assert.equal(slow.card.state(), 'ready');
  // Ready, neither time runs any more.
  slow.time.advance(RUNTIME_MS);
  await settle();
  assert.equal(slow.card.state(), 'ready');
  assert.deepEqual(slow.warnings, []);
  // One that never has it in hand is declined after ten minutes, for this visit only.
  const never = setup({ fetched: false, answer: message => (message.open ? null : {}) });
  await bringUp(never);
  never.time.advance(RUNTIME_MS - 1);
  await settle();
  assert.equal(never.card.state(), 'starting');
  never.time.advance(1);
  await settle();
  assert.equal(never.card.state(), 'declined');
  assert.ok(never.workers[0].terminated);
  assert.match(never.warnings[0], /fetching the WebGPU runtime took longer than 10 minutes/);
  assert.equal(never.storage.entries.size, 0);
  const next = setup({ storage: never.storage });
  await bringUp(next);
  assert.equal(next.card.state(), 'ready', 'the next visit tries again');
  // In hand, and then nothing: the 60 seconds are the device's, and remembered.
  const stuck = setup({ answer: message => (message.open ? null : {}) });
  await bringUp(stuck);
  stuck.time.advance(BRING_UP_MS);
  await settle();
  assert.equal(stuck.card.state(), 'declined');
  assert.equal(stuck.storage.entries.size, 1);
});

test('a remembered decline holds on later visits until the runtime, the network or ?gpu=on says to try again', async () => {
  const storage = memoryStorage();
  const first = setup({ storage, wasmMs: 100, cardMs: 101 });
  await bringUp(first);
  assert.equal(first.card.state(), 'declined');
  const [[key, record]] = storage.entries;
  assert.equal(key, 'mahjong-webgpu-declined:/mahjong/');
  assert.equal(JSON.parse(record).runtime, 'ort-webgpu/aaaa/');
  assert.equal(JSON.parse(record).network, READY.sha256);
  assert.equal(JSON.parse(record).at, TODAY);
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

test('a remembered decline lapses after a week, and one kept without its date has lapsed', async () => {
  const storage = memoryStorage();
  const declined = setup({ storage, wasmMs: 100, cardMs: 101 });
  await bringUp(declined);
  assert.equal(declined.card.state(), 'declined');
  const [[key, record]] = storage.entries;
  // To the last millisecond of the week it holds.
  const held = setup({ storage, today: TODAY + DECLINE_LASTS_MS - 1 });
  await bringUp(held);
  assert.equal(held.workers.length, 0);
  assert.equal(held.card.state(), 'declined');
  // A week on, the card is tried again; still slow, it is remembered from that day.
  const slow = setup({ storage, today: TODAY + DECLINE_LASTS_MS, wasmMs: 100, cardMs: 101 });
  await bringUp(slow);
  assert.equal(slow.workers.length, 1);
  assert.equal(slow.card.state(), 'declined');
  assert.equal(JSON.parse(storage.entries.get(key)).at, TODAY + DECLINE_LASTS_MS);
  // Without a date, with one that is no number, or with one yet to come (a clock set back), it does not hold.
  const { at: _at, ...undated } = JSON.parse(record);
  for (const kept of [undated, { ...undated, at: 'yesterday' }, { ...undated, at: TODAY + 60000 }]) {
    storage.entries.set(key, JSON.stringify(kept));
    const tried = setup({ storage });
    await bringUp(tried);
    assert.equal(tried.card.state(), 'ready', JSON.stringify(kept.at));
  }
});

test('the page hears the card begin a run and end it, through bring-up, batches, failures and resets', async () => {
  const opening = setup({ answer: message => (message.open ? null : {}) });
  await bringUp(opening);
  assert.equal(opening.card.state(), 'starting');
  assert.deepEqual(opening.works, [], 'opening the network is no run');
  let hold = false;
  const s = setup({ answer: message => (message.open ? { opened: true } : hold ? null
    : { analyses: Array.from({ length: message.count }, plainAnswer) }) });
  await bringUp(s);
  // The warm-up and each timed run.
  assert.deepEqual(s.works, Array.from({ length: 2 * (TIMED_RUNS + 1) }, (_, index) => index % 2 === 0));
  s.works.length = 0;
  // A batch on the card, and none for a single position or a batch on WebAssembly.
  await s.ask(16);
  await s.ask(1);
  assert.deepEqual(s.works, [true, false]);
  s.works.length = 0;
  // A caller who stops waiting, and a reset, leave the card at its run.
  hold = true;
  const stop = new AbortController();
  const stopped = s.ask(16, stop.signal);
  await settle();
  stop.abort();
  await assert.rejects(stopped, { name: 'AbortError' });
  s.card.reset(new Error('Match changed'));
  assert.deepEqual(s.works, [true], 'the card is still at the run nobody waits for');
  // That run out of time, the card is declined, its worker ended and its work with it,
  // and the batch that waited its turn behind it is answered on WebAssembly at once.
  const before = s.wasmCalls.length;
  const late = s.ask(16);
  await settle();
  assert.equal(s.card.state(), 'ready');
  s.time.advance(CARD_BATCH_MS);
  assert.equal((await late).length, 16);
  assert.deepEqual(s.wasmCalls.slice(before), [8, 8]);
  assert.equal(s.card.state(), 'declined');
  assert.deepEqual(s.works, [true, false]);
});

test('a batch nobody waits for any more still declines the card when it runs out of time', async () => {
  let hold = false;
  const s = setup({ answer: message => (message.open ? { opened: true } : hold ? null
    : { analyses: Array.from({ length: message.count }, plainAnswer) }) });
  await bringUp(s);
  s.works.length = 0;
  hold = true;
  const stop = new AbortController();
  const stopped = s.ask(16, stop.signal);
  await settle();
  stop.abort();
  await assert.rejects(stopped, { name: 'AbortError' });
  // To the last millisecond the card is given its time.
  s.time.advance(CARD_BATCH_MS - 1);
  await settle();
  assert.equal(s.card.state(), 'ready');
  assert.deepEqual(s.works, [true]);
  s.time.advance(1);
  await settle();
  assert.equal(s.card.state(), 'declined');
  assert.ok(s.workers[0].terminated);
  assert.deepEqual(s.works, [true, false], 'the page is not left told that the card works');
  assert.deepEqual(s.runtimes, ['webgpu', 'wasm']);
  assert.match(s.warnings[0], /a batch took longer than 10 s on the card/);
  assert.equal(s.storage.entries.size, 0, 'a card that had served is declined for this visit only');
  const before = s.wasmCalls.length;
  assert.equal((await s.ask(16)).length, 16);
  assert.deepEqual(s.wasmCalls.slice(before), [8, 8]);
});

test('the sheen and the reveal hold still under the attribute the page\'s root carries while the card works', () => {
  assert.equal(CARD_AT_WORK, 'data-card-at-work');
  const tile = readFileSync(new URL('../src/lib/Tile.svelte', import.meta.url), 'utf8');
  assert.match(tile, /:global\(:root\[data-card-at-work\]\) :is\(\.foil, \.haku-dragon-reveal\) \{\s*animation-play-state: paused;\s*\}/);
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
    assert.deepEqual(s.wasmCalls.slice(before), [8, 8], `${name}: from WebAssembly, in its own passes`);
    assert.equal(s.card.state(), 'declined');
    assert.ok(s.workers[0].terminated);
    assert.deepEqual(s.runtimes, ['webgpu', 'wasm']);
    // The device had passed: what fails once the card serves is this visit's only.
    assert.equal(s.storage.entries.size, 0, name);
    const next = setup({ storage: s.storage });
    await bringUp(next);
    assert.equal(next.card.state(), 'ready', `${name}: the next visit tries the card again`);
  }
});

test('a batch sized for the card goes back to WebAssembly in passes of WebAssembly\'s own size', async () => {
  assert.equal(WASM_PASS, BATCH);
  const stop = new AbortController();
  let failing = false, stopAt = Infinity;
  const s = setup({ wasmMs: call => { if (call === stopAt) stop.abort(); return 1000; },
    answer: message => (message.open ? { opened: true } : failing ? { error: 'a batch failed on the card: device lost', remember: true }
      : { analyses: Array.from({ length: message.count }, plainAnswer) }) });
  await bringUp(s);
  failing = true;
  // Twenty positions fail on the card: WebAssembly is asked eight, eight and four, and the answers are theirs in order.
  let before = s.wasmCalls.length;
  const answers = await s.ask(20);
  assert.deepEqual(s.wasmCalls.slice(before), [8, 8, 4]);
  assert.deepEqual(answers.map(answer => answer.of), Array.from({ length: 20 }, (_, index) => index));
  // The card declined, a caller that still sizes a pass for it is served the same way.
  before = s.wasmCalls.length;
  assert.equal((await s.ask(64)).length, 64);
  assert.deepEqual(s.wasmCalls.slice(before), Array(8).fill(8));
  // A pass of WebAssembly's own size goes as it is, and a single position too.
  before = s.wasmCalls.length;
  await s.ask(8);
  await s.ask(1);
  assert.deepEqual(s.wasmCalls.slice(before), [8, 1]);
  // A caller who stops in the second pass is asked no third.
  before = s.wasmCalls.length;
  stopAt = before + 2;
  await assert.rejects(s.ask(64, stop.signal), { name: 'AbortError' });
  assert.equal(s.wasmCalls.length, before + 2);
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
  const slow = setup({ storage: broken, wasmMs: 100, cardMs: 101 });
  await bringUp(slow);
  assert.equal(slow.card.state(), 'declined');
  assert.equal((await slow.ask(8)).length, 8);
});

test('the page\'s own card marks its root while it works, and says which runtime a review\'s batches run on', async () => {
  const time = clock(), marks = [], heard = [], workers = [];
  class Worker {
    constructor(url, options) { this.url = String(url); this.options = options; workers.push(this); }
    postMessage(message) {
      if (message.open) Promise.resolve().then(() => this.onmessage?.({ data: { id: message.id, fetched: true } }));
      Promise.resolve().then(() => {
        if (!message.open) time.pass(50);
        this.onmessage?.({ data: message.open ? { id: message.id, opened: true }
          : { id: message.id, analyses: Array.from({ length: message.count }, plainAnswer), ms: 50 } });
      });
    }
    terminate() {}
  }
  const api = loadModule('gpu-batches.js', {
    document: { baseURI: 'https://test.invalid/mahjong/index.html', documentElement: { toggleAttribute: (name, on) => marks.push([name, on]) } },
    localStorage: memoryStorage(), location: { search: '' }, navigator: desktop(), console: { warn: message => heard.push(message) },
    performance: { now: time.now }, setTimeout: time.setTimeout, clearTimeout: time.clearTimeout, Date: { now: () => TODAY },
    URL, URLSearchParams, DOMException, Worker, Promise, Float32Array, Array,
    networkUrl, playedNetwork: () => READY, __WEBGPU_RUNTIME_DIRECTORY__: 'ort-webgpu/aaaa/',
    BRING_UP_MS, CARD_BATCH_MS, DECLINE_LASTS_MS, RUNTIME_MS, TIMED_RUNS, WASM_PASS, agreement, fastEnough, median, timedOrder, webgpuWanted,
  }, ['batchesOnCard', 'policyRuntime', 'reportRuntime', 'CARD_AT_WORK']);
  const wasm = async positions => { time.pass(1000); return positions.map(plainAnswer); };
  const ask = count => api.batchesOnCard(Array.from({ length: count }, (_, index) => position(index)), undefined, wasm);
  api.reportRuntime(value => heard.push(value));
  assert.equal(api.policyRuntime(), 'wasm');
  await ask(8);
  await ask(8);
  await settle();
  assert.equal(workers.length, 1);
  assert.match(workers[0].url, /gpu-batch\.worker\.js$/);
  assert.equal(api.policyRuntime(), 'webgpu');
  assert.deepEqual(heard, ['webgpu']);
  assert.deepEqual(marks, Array.from({ length: 2 * (TIMED_RUNS + 1) }, (_, index) => [api.CARD_AT_WORK, index % 2 === 0]));
  marks.length = 0;
  assert.equal((await ask(64)).length, 64);
  assert.deepEqual(marks, [['data-card-at-work', true], ['data-card-at-work', false]]);
});

test('the page sends single decisions to its WebAssembly worker alone, and batches through the card', async t => {
  const workers = [], routed = [], resets = [];
  class Worker {
    constructor() { this.messages = []; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() {}
  }
  const api = loadModule('policy.js', {
    document: { baseURI: 'https://test.invalid/mahjong/' }, URL, DOMException, setTimeout, clearTimeout, Worker,
    MEMORY_LIMITS_MIB, nextMemoryLimit, networkUrl, playedNetwork: () => READY, __RUNTIME_DIRECTORY__: 'ort/0123abcd/',
    navigator: { onLine: true }, networkIsStored: async () => false, prepareNetwork: async () => null, refreshOffline: async () => null,
    networkReleased: () => {},
    batchesOnCard: (positions, signal, wasm) => { routed.push(positions.length); return wasm(positions, signal); },
    resetCard: reason => resets.push(reason),
  }, ['analyzePolicy', 'analyzeBatch', 'resetPolicy']);
  t.after(() => api.resetPolicy());
  const answered = async (promise, reply) => {
    await setImmediate();
    const worker = workers.at(-1), message = worker.messages.at(-1);
    worker.onmessage({ data: { id: message.id, ...reply(message) } });
    return promise;
  };
  assert.equal(await answered(api.analyzePolicy(new Float32Array(34), [1, 1]), () => ({ action: 1 })), 1);
  await answered(api.analyzePolicy(new Float32Array(34), [1, 1]), () => ({ analysis: { action: 0 } }));
  assert.deepEqual(routed, [], 'no single decision is offered to the card');
  const positions = [0, 1].map(index => ({ planes: new Float32Array(34).fill(index), mask: [1, 1] }));
  const batch = await answered(api.analyzeBatch(positions),
    message => ({ analyses: Array.from({ length: message.count }, () => policyWeights([1, 0], [1, 1])) }));
  assert.equal(batch.length, 2);
  assert.deepEqual(routed, [2]);
  // A reset drops the callers waiting on the card as it drops the worker's, with its reason.
  const left = new Error('The match was left');
  api.resetPolicy(left);
  assert.deepEqual(resets, [left]);
});

/** The card's worker, with a stand-in runtime, adapter, stored network and fetch. */
function cardWorker({ adapterGiven = adapter(), stored = Uint8Array.of(1, 2, 3), fetchStatus = 200, importError = null, createError = null,
  runError = null, inputs = ['planes', 'legal'] } = {}) {
  const messages = [], imports = [], created = [], reads = [], fetches = [], fed = [], addresses = [];
  // The addresses the worker makes of its own, each noted as it is made.
  class Addressed extends URL {
    static createObjectURL(blob) { const made = URL.createObjectURL(blob); addresses.push(made); return made; }
  }
  class Tensor {
    constructor(type, data, dims) { Object.assign(this, { type, data, dims }); }
    dispose() {}
  }
  const runtime = { env: { wasm: {}, webgpu: {} }, Tensor, InferenceSession: { create: async (bytes, options) => {
    // What the runtime would read its WebAssembly from, read here as it would.
    const paths = { ...runtime.env.wasm.wasmPaths };
    created.push({ bytes, options, adapter: runtime.env.webgpu.adapter, paths, handed: await resolveObjectURL(paths.wasm)?.text() });
    if (createError) throw createError;
    return { inputNames: inputs, run: async feeds => {
      fed.push(feeds);
      if (runError) throw runError;
      // Two moves a position, the first the heavier.
      const moves = 2 * feeds.planes.dims[0];
      return { policy: new Tensor('float32', Float32Array.from({ length: moves }, (_, index) => (index % 2 ? 0 : 1))),
        value: new Tensor('float32', [0]), hands: new Tensor('float32', [0]) };
    } };
  } } };
  const self = { postMessage: message => messages.push(message), navigator: { gpu: gpu(adapterGiven) } };
  loadModule('gpu-batch.worker.js', { self, URL: Addressed, Blob, Error, policyWeights, performance, Promise, Float32Array, Array, Object, String,
    webgpuAdapter, storedNetworkBytes: async read => { reads.push(read); return stored; },
    fetch: async url => { fetches.push(String(url)); return new Response('runtime', { status: fetchStatus }); },
    loadRuntime: async url => { imports.push(url); if (importError) throw importError; return runtime; },
  }, [], { "import(/* @vite-ignore */ new URL('ort.min.mjs', webgpu).href)": "loadRuntime(new URL('ort.min.mjs', webgpu).href)" });
  const send = data => { self.onmessage({ data }); return settle(); };
  const open = { url: 'https://cdn.invalid/models/g3000/x', expect: { bytes: 3, sha256: 'a'.repeat(64) }, scope: 'https://test.invalid/mahjong/',
    webgpu: 'https://test.invalid/mahjong/ort-webgpu/aaaa/' };
  return { send, open, messages, imports, created, reads, fetches, fed, runtime, addresses };
}

test('the card\'s worker opens the stored network on WebGPU alone and answers batches with their analyses and time', async () => {
  const w = cardWorker();
  await w.send({ id: 1, open: w.open });
  // It says that its runtime is in hand, then that the network is open.
  assert.deepEqual(w.messages.filter(message => message.id === 1).map(message => JSON.stringify(message)),
    ['{"id":1,"fetched":true}', '{"id":1,"opened":true}']);
  // The runtime's WebAssembly, fetched once to see that it can be, is the one the runtime is handed, by an address
  // of the worker's own that is given up once the session is open; its loader it takes from the site.
  const folder = 'https://test.invalid/mahjong/ort-webgpu/aaaa/';
  assert.deepEqual(w.fetches, [`${folder}ort-wasm-simd-threaded.jsep.mjs`, `${folder}ort-wasm-simd-threaded.jsep.wasm`]);
  assert.equal(w.created[0].paths.mjs, `${folder}ort-wasm-simd-threaded.jsep.mjs`);
  assert.match(w.created[0].paths.wasm, /^blob:/);
  assert.equal(w.created[0].handed, 'runtime');
  assert.equal(resolveObjectURL(w.created[0].paths.wasm), undefined);
  assert.equal(w.runtime.env.wasm.wasmBinary, undefined, 'the bytes themselves the runtime would keep for as long as it runs');
  assert.equal(w.reads[0].scope, 'https://test.invalid/mahjong/', 'the network is read from the page\'s cache');
  assert.equal(w.reads[0].expect.sha256, 'a'.repeat(64));
  assert.deepEqual(w.imports, ['https://test.invalid/mahjong/ort-webgpu/aaaa/ort.min.mjs']);
  // Errors alone are logged: the runtime's warning about nodes it keeps on the processor by design is none.
  assert.equal(JSON.stringify(w.created[0].options), '{"executionProviders":["webgpu"],"graphOptimizationLevel":"all","logSeverityLevel":3}');
  assert.ok(w.created[0].adapter, 'the runtime is given the adapter that was checked');
  // Two positions of three planes, each with two moves, the second position's first not legal.
  await w.send({ id: 2, planes: new Float32Array(2 * 3 * 34), mask: [1, 1, 0, 1], count: 2 });
  const answer = w.messages.find(message => message.id === 2);
  assert.equal(answer.analyses.length, 2);
  assert.equal(answer.analyses[0].action, 0);
  assert.equal(answer.analyses[1].action, 1);
  assert.ok(answer.ms >= 0);
  // As the WebAssembly worker lays them: positions, planes, the 34 tile kinds; and a mask of ones and noughts.
  assert.deepEqual(Object.keys(w.fed[0]), ['planes', 'legal']);
  assert.deepEqual(Array.from(w.fed[0].planes.dims), [2, 3, 34]);
  assert.deepEqual(Array.from(w.fed[0].legal.dims), [2, 2]);
  assert.deepEqual(Array.from(w.fed[0].legal.data), [1, 1, 0, 1]);
  // A network that takes no mask is given none, and its answers are read against the mask all the same.
  const bare = cardWorker({ inputs: ['planes'] });
  await bare.send({ id: 1, open: bare.open });
  await bare.send({ id: 2, planes: new Float32Array(2 * 3 * 34), mask: [1, 1, 0, 1], count: 2 });
  assert.deepEqual(Object.keys(bare.fed[0]), ['planes']);
  assert.deepEqual(bare.messages.find(message => message.id === 2).analyses.map(analysis => analysis.action), [0, 1]);
});

test('the card\'s worker says why it cannot serve, and whether that is the device\'s', async () => {
  for (const [name, options, reason, remember] of [
    ['no adapter', { adapterGiven: null }, /no graphics card adapter/, true],
    ['a software adapter', { adapterGiven: adapter({ architecture: 'swiftshader' }) }, /no graphics card adapter/, true],
    ['a network not saved', { stored: null }, /not saved on this device/, false],
    ['a runtime it cannot fetch', { fetchStatus: 503 }, /could not be fetched: .*HTTP 503/, false],
    ['a runtime whose own file does not load', { importError: new Error('Failed to fetch dynamically imported module') },
      /could not be fetched: Failed to fetch dynamically imported module/, false],
    ['a session that fails', { createError: new Error('no kernel') }, /could not be opened on the card: no kernel/, true],
  ]) {
    const w = cardWorker(options);
    await w.send({ id: 1, open: w.open });
    const said = w.messages.find(message => message.id === 1 && 'error' in message);
    assert.match(said.error, reason, name);
    assert.equal(said.remember, remember, name);
    if (name !== 'a session that fails') assert.equal(w.created.length, 0, `${name}: no session is made`);
    // Only with the runtime in hand does it say so: before that the time is not the device's.
    assert.equal(w.messages.some(message => message.fetched), name === 'a session that fails', name);
    // Wherever it stops, the address it made of the runtime's WebAssembly is given up.
    assert.equal(w.addresses.length, ['a runtime whose own file does not load', 'a session that fails'].includes(name) ? 1 : 0, name);
    for (const address of w.addresses) assert.equal(resolveObjectURL(address), undefined, `${name}: the address of the runtime's WebAssembly is given up`);
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
