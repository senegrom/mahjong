import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { setImmediate } from 'node:timers/promises';
import { watchModelAvailability } from '../src/lib/model-availability.js';

const read = path => readFileSync(new URL(path, import.meta.url), 'utf8');
const policySource = read('../src/lib/policy.js')
  .replace(/^import .*;$/gm, '')
  .replaceAll('import.meta.url', "'https://test.invalid/mahjong/policy.js'")
  .replaceAll('export ', '');
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
function client(t, prepareOfflineAi) {
  const workers = [];
  class Worker {
    constructor() { this.messages = []; this.terminated = false; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    answer(payload = { action: 0 }) {
      this.onmessage({ data: { id: this.messages.filter(m => m.id != null).at(-1).id, ...payload } });
    }
  }
  const context = vm.createContext({
    document: { baseURI: 'https://test.invalid/mahjong/' },
    URL, DOMException, Worker, setTimeout, clearTimeout, prepareOfflineAi,
    // The memory allocator and model transport have their own integration tests.
    MEMORY_LIMITS_MIB: [256, 512, 1024], nextMemoryLimit: () => null,
    NETWORK: { generation: 1 }, NETWORK_URL: 'https://test.invalid/network',
    networkIsStored: async () => false,
  });
  vm.runInContext(policySource + '\nglobalThis.api = { chooseAction, resetPolicy };', context);
  const api = context.api;
  t.after(() => api.resetPolicy());
  return { api, workers, ask: signal => api.chooseAction(new Float32Array(34), [1], 0, 1000, signal) };
}

test('warm decisions do not recheck, hash or fetch offline model bytes', async t => {
  let preparations = 0;
  const { ask, workers } = client(t, async () => {
    preparations++;
    assert.equal(preparations, 1, 'warm inference must not enter offline preparation');
    return { aiReady: true };
  });
  for (let i = 0; i < 4; i++) {
    const result = ask();
    await setImmediate();
    assert.equal(workers.length, 1);
    workers[0].answer();
    assert.equal(await result, 0);
  }
  assert.equal(preparations, 1);
});

test('a resident model still answers after storage fallback and disconnection', async t => {
  let online = true, preparations = 0;
  const { ask, workers } = client(t, async () => {
    preparations++;
    if (!online) throw new TypeError('Failed to fetch');
    return null; // The offline coordinator's documented storage-failure result.
  });
  const first = ask(); await setImmediate(); workers[0].answer(); await first;
  online = false;
  const next = ask(), answered = next.then(value => ({ value }), error => ({ error }));
  await setImmediate(); workers[0].answer();
  assert.deepEqual(await answered, { value: 0 });
  assert.equal(preparations, 1);
});

test('cold concurrent requests share preparation and cancel independently', async t => {
  const download = deferred(); let preparations = 0;
  const { ask, workers } = client(t, () => { preparations++; return download.promise; });
  const abort = new AbortController();
  const first = ask(abort.signal), rejected = assert.rejects(first, { name: 'AbortError' });
  const second = ask();
  assert.equal(preparations, 1);
  abort.abort(); await rejected;
  download.resolve(null); await setImmediate();
  assert.equal(workers.length, 1);
  assert.equal(workers[0].messages.length, 1);
  workers[0].answer(); await second;
});

test('failed preparation is retryable without poisoning the client', async t => {
  let preparations = 0;
  const { ask, workers } = client(t, async () => {
    if (++preparations === 1) throw new Error('interrupted download');
    return null;
  });
  await assert.rejects(ask(), /interrupted download/);
  const retry = ask(); await setImmediate(); workers[0].answer(); await retry;
  assert.equal(preparations, 2);
});

test('worker failure invalidates preparation and a late old reply cannot answer the retry', async t => {
  let preparations = 0;
  const { ask, api, workers } = client(t, async () => { preparations++; return null; });
  const first = ask(); await setImmediate(); workers[0].answer(); await first;
  const second = ask(), failed = assert.rejects(second, /broken worker/);
  await setImmediate(); workers[0].answer({ error: 'broken worker' }); await failed;
  assert.equal(workers[0].terminated, true);
  const retry = ask(); await setImmediate();
  assert.equal(preparations, 2); assert.equal(workers.length, 2);
  workers[0].answer({ error: 'late failure' });
  assert.equal(workers[1].terminated, false);
  workers[1].answer(); await retry;
  api.resetPolicy();
});

test('reset during preparation prevents stale work from entering a replacement runtime', async t => {
  const downloads = [];
  const { ask, api, workers } = client(t, () => {
    const download = deferred(); downloads.push(download); return download.promise;
  });
  const old = ask(), rejected = assert.rejects(old, { name: 'AbortError' });
  api.resetPolicy();
  const current = ask();
  downloads[0].resolve(null); await rejected;
  assert.equal(workers.length, 0);
  downloads[1].resolve(null); await setImmediate();
  assert.equal(workers.length, 1); workers[0].answer(); await current;
});

function availability(t, probe, refreshOffline = async () => {}) {
  const values = [], events = new EventTarget(), page = new EventTarget();
  page.visibilityState = 'visible';
  let publish, unsubscribed = false;
  const stop = watchModelAvailability({ probe, refreshOffline, events, page,
    watchOffline(callback) { publish = callback; callback({ aiReady: false }); return () => { unsubscribed = true; }; },
    onChange(value) { values.push(value); },
  });
  t.after(stop);
  return { values, events, page, stop, publish: info => publish(info), unsubscribed: () => unsubscribed };
}

test('Trained becomes available on reconnect after a failed startup probe', async t => {
  let online = false, checks = 0, refreshes = 0;
  const view = availability(t, async () => { checks++; return online; }, async () => { refreshes++; });
  await setImmediate(); assert.equal(view.values.at(-1), false);
  online = true; view.events.dispatchEvent(new Event('online'));
  await setImmediate();
  assert.equal(view.values.at(-1), true); assert.equal(checks, 2); assert.equal(refreshes, 1);
});

test('verified downloads enable Trained immediately and supersede a stale negative probe', async t => {
  const probe = deferred();
  const view = availability(t, () => probe.promise);
  view.publish({ aiReady: true });
  assert.deepEqual(view.values, [true]);
  probe.resolve(false); await setImmediate();
  assert.deepEqual(view.values, [true]);
});

test('foreground checks retry independently of a failed offline refresh', async t => {
  let checks = 0;
  const view = availability(t, async () => ++checks > 1, async () => { throw new Error('storage unavailable'); });
  await setImmediate();
  view.page.visibilityState = 'hidden'; view.page.dispatchEvent(new Event('visibilitychange'));
  await setImmediate(); assert.equal(checks, 1);
  view.page.visibilityState = 'visible'; view.page.dispatchEvent(new Event('visibilitychange'));
  await setImmediate(); assert.equal(view.values.at(-1), true);
});

test('newer availability results win and unmount removes listeners and ignores late results', async t => {
  const checks = [];
  const view = availability(t, () => { const p = deferred(); checks.push(p); return p.promise; });
  view.events.dispatchEvent(new Event('online'));
  checks[1].resolve(true); await setImmediate();
  checks[0].resolve(false); await setImmediate();
  assert.deepEqual(view.values, [true]);
  view.events.dispatchEvent(new Event('online'));
  view.stop(); checks[2].resolve(false); view.publish({ aiReady: true });
  view.events.dispatchEvent(new Event('online')); view.page.dispatchEvent(new Event('visibilitychange'));
  await setImmediate();
  assert.equal(checks.length, 3); assert.equal(view.unsubscribed(), true);
  assert.deepEqual(view.values, [true]);
});

const physicalSource = read('../src/lib/PhysicalPlay.svelte');
function editor() {
  // Execute the real component functions, with runes/lifecycle and the rules
  // engine replaced only at the boundary. This is not a browser-render test.
  const emptyPosition = () => ({ wall: 69, kyoku: 1, counters: 0, riichi_sticks: 0, round: 0,
    seat: 0, turn: 0, phase: 'act', first_turns: true, after_quad: false, indicators: [],
    players: Array.from({ length: 4 }, () => ({ score: 31000, hand: [], melds: [], discards: [], riichi: 'none', ippatsu: false, furiten: false })) });
  const context = vm.createContext({
    structuredClone, emptyPosition, $state: value => value, $effect() {}, $props: () => ({}),
    onMount() {}, onDestroy() {}, PhysicalAnalysis: class {}, supportsTrainedAgent: () => false,
  });
  const script = physicalSource.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/^\s*import .*;$/gm, '');
  vm.runInContext(script + `\nloaded = true;
    globalThis.api = { edit, editNumber, endNumberEdit, undo,
      state: () => snapshot(), depth: () => history.length,
      block: kind => { saveConflict = kind === 'conflict' ? 'newer table' : ''; unreadable = kind === 'unreadable'; closed = kind === 'closed'; },
      setAgent: value => { agent = value; }, agent: () => agent,
    };`, context);
  const api = context.api;
  return { ...api, state: () => JSON.parse(JSON.stringify(api.state())) };
}
const score = (p, value) => { p.players[0].score = value; };
const wall = (p, value) => { p.wall = value; };

test('a numeric edit alone enables Undo and restores the pre-edit value', () => {
  const e = editor(); e.editNumber(32000, 'score-0', score);
  assert.equal(e.depth(), 1); assert.equal(e.state().players[0].score, 32000);
  e.undo(); assert.equal(e.depth(), 0); assert.equal(e.state().players[0].score, 31000);
});

test('interleaved tile, number and checkbox edits undo one logical change at a time', () => {
  const e = editor(), initial = e.state();
  e.edit(p => { p.players[0].hand.push('1m'); });
  for (const value of [undefined, 3, 32, 32000]) e.editNumber(value, 'score-0', score);
  e.endNumberEdit(); e.editNumber(60, 'wall', wall); e.endNumberEdit();
  e.edit(p => { p.players[0].furiten = true; });
  assert.equal(e.depth(), 4);
  e.undo(); assert.equal(e.state().players[0].furiten, false); assert.equal(e.state().wall, 60);
  e.undo(); assert.equal(e.state().wall, 69); assert.equal(e.state().players[0].score, 32000);
  e.undo(); assert.equal(e.state().players[0].score, 31000); assert.deepEqual(e.state().players[0].hand, ['1m']);
  e.undo(); assert.deepEqual(e.state(), initial);
});

test('refocusing the same numeric field starts a new step and empty input remains a saved null', () => {
  const e = editor();
  e.editNumber(32000, 'score-0', score); e.endNumberEdit();
  e.editNumber(undefined, 'score-0', score);
  assert.equal(e.state().players[0].score, null); assert.equal(e.depth(), 2);
  e.undo(); assert.equal(e.state().players[0].score, 32000);
  e.editNumber(-100, 'score-0', score);
  assert.equal(e.depth(), 2); e.undo(); assert.equal(e.state().players[0].score, 32000);
});

test('failed/no-op edits do not damage history and blocked editors cannot edit or undo', () => {
  const e = editor(); e.editNumber(32000, 'score-0', score); e.endNumberEdit();
  const before = e.state();
  e.edit(p => { p.wall = 1; throw new Error('invalid move'); });
  e.edit(() => {});
  assert.deepEqual(e.state(), before); assert.equal(e.depth(), 1);
  for (const kind of ['conflict', 'unreadable', 'closed']) {
    e.block(kind); assert.equal(e.editNumber(0, 'wall', wall), false); e.undo();
    assert.deepEqual(e.state(), before); assert.equal(e.depth(), 1);
  }
});

test('history is bounded and adviser preferences are outside position Undo', () => {
  const e = editor();
  for (let i = 0; i < 40; i++) { e.editNumber(i, 'wall', wall); e.endNumberEdit(); }
  assert.equal(e.depth(), 30);
  e.setAgent('full'); e.undo(); assert.equal(e.agent(), 'full'); assert.equal(e.depth(), 29);
});

test('the UI wires live availability and every persisted field through the tested paths', () => {
  const app = read('../src/App.svelte');
  assert.match(app, /watchModelAvailability\(\{/);
  assert.match(app, /probe: modelIsAvailable, watchOffline, refreshOffline/);
  assert.match(app, /unwatchModel\(\)/);
  assert.doesNotMatch(physicalSource, /bind:(?:value|checked)=\{(?:position|player|meld|discard)\./);
  assert.equal((physicalSource.match(/type="number"/g) ?? []).length, 6);
  assert.equal((physicalSource.match(/onblur=\{endNumberEdit\}/g) ?? []).length, 6);
  assert.equal((physicalSource.match(/value => editNumber\(/g) ?? []).length, 6);
  assert.match(physicalSource, /onclick=\{undo\}/);
});
