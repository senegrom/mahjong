import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { MessageChannel } from 'node:worker_threads';
import { render } from 'svelte/server';
import { componentLoader } from './fixtures/load-component.js';
import { loadModule } from './fixtures/load-module.js';

// The offline panel itself, to read what a state says to the player.
const Status = await componentLoader()('app/OfflineStatus.svelte');
const panel = offline => render(Status, { props: { offline, downloadAi: () => {} } }).body;

const NETWORK = { url: 'https://model.invalid/g1', version: '0.1.1', sha256: '1'.repeat(64), bytes: 116 };
const NEXT = { url: 'https://model.invalid/g2', version: '0.1.2', sha256: '2'.repeat(64), bytes: 117 };

/** A service worker speaking the real message contract: it saves the game and
 * the runtime, reports whether the network is stored, and never fetches it.
 * It answers about `network`, whatever version a page names; which network a
 * version names is the real worker's business (offline.test.js). */
function fakeWorker({ calls, asked, network, stored, info, faults, label = '' }) {
  return {
    scriptURL: 'https://test.invalid/mahjong/sw.js',
    postMessage({ type, version }, [port]) {
      calls.push(label + type);
      asked.push([label + type, version]);
      queueMicrotask(() => {
        const reply = () => ({ ...info, aiReady: info.runtime && stored.has(network.url), network });
        if (type === 'MAHJONG_PREPARE_CORE' && faults.core) port.postMessage(faults.core === true ? { error: 'Connection lost' } : faults.core);
        else if (type === 'MAHJONG_PREPARE_AI' && faults.ai) port.postMessage({ error: 'AI interrupted', storage: faults.storage });
        else {
          if (type === 'MAHJONG_PREPARE_CORE') info.coreReady = true;
          if (type === 'MAHJONG_PREPARE_AI') { info.runtime = true; info.aiRequested = true; }
          port.postMessage({ value: reply() });
        }
        port.close();
      });
    },
  };
}

// Exercise the real coordinator against the service-worker message contract.
// No browser globals or state are shared between tests. `network` is the one
// the page plays, and `pick` the version its player picked, if any.
function coordinator({ coreReady = false, aiReady = false, aiRequested = false, failCore = false, failAi = false,
  registered = true, held = true, storageFailure = false, online = true, waiting = null, installs = false,
  updating = null, controlled = true, impatient = false, network = NETWORK, pick } = {}) {
  const calls = [], asked = [], saves = [], stored = new Set(held || aiReady ? [network.url] : []);
  const info = { coreReady, runtime: aiReady, aiRequested, hasModel: true };
  // transfer, when set, is the network's transfer itself: it may fail or be
  // held open while the test looks at what runs beside it.
  const faults = { core: failCore, ai: failAi, storage: storageFailure, registration: false, network: null, transfer: null };
  let latest, registrations = 0;
  const active = fakeWorker({ calls, asked, network, stored, info, faults });
  // With installs, the first registration's worker is still installing: it
  // activates (or fails) only when the test finishes the install. One that
  // fails leaves no registration behind, as in a browser, and registering
  // again starts another install. With updating, the page loads under its
  // active worker while an update the browser found by itself is installing.
  const changes = new Set();
  const installer = () => ({ state: 'installing', addEventListener: (_, listener) => changes.add(listener),
    removeEventListener: (_, listener) => changes.delete(listener) });
  const nextWorker = info => fakeWorker({ calls, asked, network: NEXT, stored, faults: {},
    info: { coreReady: true, runtime: false, hasModel: true, ...info }, label: 'next:' });
  const registration = { scope: 'https://test.invalid/mahjong/', active: installs ? null : active,
    installing: installs || updating ? installer() : null,
    addEventListener() {}, async update() {}, waiting: waiting && nextWorker(waiting) };
  function finishInstall(state = 'activated') {
    registration.installing.state = state;
    registration.installing = null;
    if (state === 'activated') registration.active = active; else registered = false;
    for (const listener of [...changes]) listener();
  }
  /** The update has installed, and waits for this version's windows to close. */
  function finishUpdate() {
    registration.installing.state = 'installed';
    registration.installing = null;
    registration.waiting = nextWorker(updating);
    for (const listener of [...changes]) listener();
  }
  // Without control, the page is a first visit's before its worker claims it,
  // or one a hard reload loaded past the worker.
  const serviceWorker = Object.assign(new EventTarget(), {
    controller: controlled ? active : null,
    async getRegistration() { return registered ? registration : undefined; },
    async register() {
      registrations++;
      if (faults.registration) throw new Error('Connection lost');
      registered = true;
      if (installs && !registration.active) registration.installing ??= installer();
      return registration;
    },
  });
  /** The worker's clients.claim(): it takes control of the page. */
  function claim() {
    serviceWorker.controller = active;
    serviceWorker.dispatchEvent(new Event('controllerchange'));
  }
  const navigator = { onLine: online, storage: {}, serviceWorker };
  // Impatient, the page's two-minute wait for an install and its 15-second
  // wait for control end at once, as on a link too slow for either. The wait
  // before a failed install is tried again is never left to the clock: the
  // test ends it (retries), and reads how long it would have been.
  const retries = [];
  let waits = 0;
  const wait = (fn, ms) => {
    if ([120000, 15000, 180000].includes(ms)) return setTimeout(fn, impatient && ms !== 180000 ? 20 : ms);
    retries.push({ id: `retry ${++waits}`, fn, ms });
    return retries.at(-1).id;
  };
  const clear = id => {
    const at = retries.findIndex(retry => retry.id === id);
    if (at >= 0) retries.splice(at, 1); else clearTimeout(id);
  };
  /** The installing worker tells its pages why its install failed, before it goes. */
  function installFailed(error, storage, scriptURL = active.scriptURL) {
    serviceWorker.dispatchEvent(Object.assign(new Event('message'),
      { data: { type: 'MAHJONG_INSTALL_FAILED', error, storage }, source: { scriptURL } }));
  }
  // offline.js's imports, network-store.js and the page's choice of network
  // in trained-network.js, are stood in for here.
  const api = loadModule('offline.js', { document: { baseURI: 'https://test.invalid/mahjong/' }, navigator,
    URL, MessageChannel, setTimeout: wait, clearTimeout: clear, caches: {}, isSecureContext: true, WeakSet,
    networkUrl: entry => entry.url, playedNetwork: () => network, pickedVersion: () => pick,
    // The page's own record of a saved network: a header check, never a read.
    networkIsStored: async (url = network.url) => stored.has(url),
    // The page's download: verified, then stored durably, or it throws.
    networkBytes: async ({ url = network.url, expect, requireStored, onProgress } = {}) => {
      saves.push({ url, expect: expect && { ...expect }, requireStored });
      if (faults.network) throw faults.network;
      await faults.transfer?.(url);
      onProgress?.({ bytes: 1, total: 2 });
      stored.add(url);
      return new Uint8Array(1);
    } }, ['startOffline', 'refreshOffline', 'prepareOfflineAi', 'prepareNetwork', 'watchOffline'], { 'import.meta.env.DEV': 'false' });
  api.watchOffline(value => latest = value);
  return { api, calls, asked, info, faults, stored, saves, navigator, retries, state: () => latest,
    registrations: () => registrations, finishInstall, finishUpdate, installFailed, claim };
}
const settle = async () => { for (let i = 0; i < 20; i++) await setImmediate(); };
/** Settles until `met()` holds, for work that takes several messages. */
async function until(met, what) {
  for (let turn = 0; turn < 50 && !met(); turn++) await settle();
  assert.ok(met(), what);
}

test('startup automatically fills a missing game/graphics cache without requesting AI', async () => {
  const c = coordinator();
  await Promise.all([c.api.startOffline(), c.api.startOffline()]);
  assert.equal(c.calls.filter(type => type === 'MAHJONG_PREPARE_CORE').length, 1);
  assert.equal(c.calls.includes('MAHJONG_PREPARE_AI'), false);
  assert.equal(c.state().coreReady, true);
  assert.equal(c.state().aiReady, false);
});

test('a complete game cache needs neither a core download nor an AI click', async () => {
  const c = coordinator({ coreReady: true });
  await c.api.startOffline();
  await c.api.refreshOffline();
  assert.ok(c.calls.every(type => type === 'MAHJONG_STATUS'));
  assert.equal(c.state().coreReady, true);
});

test('foreground and reconnect repair missing base assets automatically and never opt into AI', async () => {
  const c = coordinator({ coreReady: true });
  await c.api.startOffline();
  c.info.coreReady = false;
  c.faults.core = true;
  await c.api.refreshOffline();
  assert.equal(c.state().coreReady, false);
  assert.match(c.state().coreWarning, /retry automatically/);
  c.faults.core = false;
  await Promise.all([c.api.refreshOffline(), c.api.refreshOffline()]);
  assert.equal(c.state().coreReady, true);
  assert.equal(c.state().coreWarning, '');
  assert.equal(c.calls.filter(type => type === 'MAHJONG_PREPARE_CORE').length, 2);
  assert.equal(c.calls.includes('MAHJONG_PREPARE_AI'), false);
});

test('an interrupted first registration retries on reconnect without an AI download', async () => {
  const c = coordinator({ registered: false });
  c.faults.registration = true;
  assert.equal(await c.api.startOffline(), null);
  assert.equal(c.state().supported, false);
  c.faults.registration = false;
  await c.api.refreshOffline();
  assert.equal(c.registrations(), 2);
  assert.equal(c.state().coreReady, true);
  assert.equal(c.calls.includes('MAHJONG_PREPARE_AI'), false);
});

test('a first visit says the game is installing until its worker activates or fails', async () => {
  // The install reports no progress of its own; the panel can only say that
  // it is running.
  for (const ending of ['activated', 'redundant']) {
    const c = coordinator({ registered: false, coreReady: true, installs: true });
    const started = c.api.startOffline();
    await settle();
    assert.equal(c.state().installing, true, ending);
    assert.equal(c.state().phase, 'checking');
    c.finishInstall(ending);
    const result = await started;
    assert.equal(c.state().installing, false, ending);
    assert.equal(c.state().phase, ending === 'activated' ? 'ready' : 'checking');
    assert.equal(Boolean(result), ending === 'activated');
    // A failed install is not storage this browser lacks: it says what failed.
    if (ending === 'redundant') assert.match(panel(c.state()), /Game download incomplete/);
  }
  // A returning visit's worker is already active: nothing is installing.
  const returning = coordinator({ coreReady: true });
  const seen = [];
  returning.api.watchOffline(value => seen.push(value.installing));
  await returning.api.startOffline();
  assert.ok(seen.length > 1 && seen.every(value => value === false));
});

test('a first install that outlasts the page\'s wait is shown as running, and as ready once its worker takes control', async () => {
  // Below about 0.5 Mbit/s the first install, about 7 MB, outlasts the page's
  // two minutes.
  const c = coordinator({ registered: false, coreReady: true, installs: true, controlled: false, impatient: true });
  const seen = [];
  c.api.watchOffline(value => seen.push(value));
  assert.equal(await c.api.startOffline(), null, 'the page stopped waiting');
  assert.match(panel(c.state()), /Still saving for offline play\. Stay connected until this status says ready\./);
  // It finishes later and takes control. Nothing else looks again: no
  // reconnect, no return to the app, no Trained request.
  c.finishInstall('activated');
  c.claim();
  await settle();
  assert.equal(c.state().supported, true);
  assert.equal(c.state().phase, 'ready');
  assert.equal(c.state().coreReady, true);
  assert.equal(c.state().coreWarning, '');
  assert.match(panel(c.state()), /Offline: game ready/);
  for (const state of seen) assert.doesNotMatch(panel(state), /unavailable here/);
});

test('a first install that fails after the page stopped waiting is said, and tried again by itself', async () => {
  const c = coordinator({ registered: false, coreReady: true, installs: true, controlled: false, impatient: true });
  assert.equal(await c.api.startOffline(), null, 'the page stopped waiting');
  assert.match(panel(c.state()), /Still saving for offline play\. Stay connected until this status says ready\./);
  // A file stalled, or the browser's few minutes ran out. Nothing else looks
  // again: no reconnect, no return to the app, no Trained request.
  c.finishInstall('redundant');
  await settle();
  assert.match(panel(c.state()), /Game download incomplete\. It will be tried again; stay connected\./);
  assert.doesNotMatch(panel(c.state()), /Still saving/);
  assert.deepEqual(c.retries.map(retry => retry.ms), [5000]);
  c.retries.shift().fn();
  await until(() => c.registrations() === 2 && c.state().installing, 'the page registers again');
  assert.doesNotMatch(panel(c.state()), /Game download incomplete/, 'another attempt is under way');
  // This attempt had only the missing files to fetch, and finishes.
  c.finishInstall('activated');
  c.claim();
  await until(() => c.state().coreReady && c.state().phase === 'ready', 'the game is saved');
  assert.match(panel(c.state()), /Offline: game ready/);
  assert.deepEqual(c.retries, []);
});

test('each failed install waits twice as long as the last before the next, up to five minutes, and offline none is tried', async () => {
  const c = coordinator({ registered: false, coreReady: true, installs: true });
  const waited = [];
  void c.api.startOffline();
  for (let attempt = 1; attempt <= 8; attempt++) {
    await until(() => c.state().installing, `attempt ${attempt} is installing`);
    c.finishInstall('redundant');
    await until(() => c.retries.length === 1 && !c.state().installing, `attempt ${attempt} failed`);
    assert.match(panel(c.state()), /Game download incomplete/);
    const retry = c.retries.shift();
    waited.push(retry.ms);
    if (attempt < 8) retry.fn();
    else {
      // Offline, the wait ends without an attempt: a reconnect starts one.
      c.navigator.onLine = false;
      retry.fn();
      await settle();
      assert.equal(c.registrations(), 8);
    }
  }
  assert.deepEqual(waited, [5000, 10000, 20000, 40000, 80000, 160000, 300000, 300000]);
});

test('a first install that storage refused says so, and is neither tried again nor told to reconnect', async () => {
  const why = 'The game is not saved offline: Quota exceeded.';
  // The worker's word and its end reach the page in either order.
  for (const order of ['said first', 'gone first']) {
    const c = coordinator({ registered: false, coreReady: true, installs: true });
    const started = c.api.startOffline();
    await until(() => c.state().installing, 'the install runs');
    if (order === 'said first') c.installFailed(why, true);
    c.finishInstall('redundant');
    assert.equal(await started, null);
    if (order === 'gone first') {
      assert.match(c.state().coreWarning, /It will be tried again/);
      assert.equal(c.retries.length, 1);
      c.installFailed(why, true);
    }
    assert.equal(c.state().coreWarning, why, order);
    assert.match(panel(c.state()), /The game is not saved offline: Quota exceeded\./);
    assert.deepEqual(c.retries, [], `${order}: another attempt would be refused the same`);
  }
  // A failure that is not storage's, or word from another project's worker, changes nothing: it is tried again.
  for (const [error, storage, from] of [['Download stalled: tiles/Chun.svg', false], [why, true, 'https://test.invalid/plateloader/sw.js']]) {
    const c = coordinator({ registered: false, coreReady: true, installs: true });
    const started = c.api.startOffline();
    await until(() => c.state().installing, 'the install runs');
    c.installFailed(error, storage, from);
    c.finishInstall('redundant');
    assert.equal(await started, null);
    assert.equal(c.state().coreWarning, 'Game download incomplete. It will be tried again; stay connected.');
    assert.deepEqual(c.retries.map(retry => retry.ms), [5000]);
  }
});

test('a repair that storage refuses says what was refused, without asking for a reconnect', async () => {
  const c = coordinator({ coreReady: true });
  await c.api.startOffline();
  c.info.coreReady = false;
  c.faults.core = { error: 'The game is not saved offline: Quota exceeded.', storage: true };
  await c.api.refreshOffline();
  assert.equal(c.state().coreWarning, 'The game is not saved offline: Quota exceeded.');
  assert.equal(c.state().coreReady, false);
  // A lost connection is still told as one.
  c.faults.core = true;
  await c.api.refreshOffline();
  assert.match(c.state().coreWarning, /Game and tile download incomplete: Connection lost.*when you reconnect or reopen the app/);
});

test('storage that refuses the trained AI leaves online play, and nobody is told to reconnect', async () => {
  // The runtime, by the worker; the network, by the page; no room for the network at all.
  for (const [options, arrange, said] of [
    [{ failAi: true, storageFailure: true }, () => {}, /^AI interrupted Online play is still available\.$/],
    [{ held: false }, c => { c.faults.network = Object.assign(new Error('The network is not saved offline: Quota exceeded.'), { name: 'NetworkStorageError' }); },
      /^The network is not saved offline: Quota exceeded\. Online play is still available\.$/],
    [{ held: false }, c => { c.navigator.storage.estimate = async () => ({ quota: 1000, usage: 1000 }); },
      /^The network is not saved offline: this device has too little free storage for it\. Online play is still available\.$/],
  ]) {
    const c = coordinator({ coreReady: true, ...options });
    arrange(c);
    await c.api.startOffline();
    assert.equal(await c.api.prepareOfflineAi(), null);
    assert.equal(c.state().phase, 'incomplete');
    assert.match(c.state().warning, said);
  }
});

test('an update already installing when the page begins to watch is reported, and saved ahead, once it has installed', async () => {
  // The browser looks for an update as the page loads, before the page has
  // asked its worker anything: no updatefound comes after that.
  const c = coordinator({ coreReady: true, aiReady: true, aiRequested: true, updating: { aiRequested: true } });
  await c.api.startOffline();
  await settle();
  assert.equal(c.state().updateReady, false);
  assert.deepEqual(c.saves, []);
  c.finishUpdate();
  await until(() => c.saves.length === 1, 'the update\'s network is saved while this version runs');
  assert.equal(c.state().updateReady, true);
  assert.deepEqual(c.calls.filter(call => call.startsWith('next:')), ['next:MAHJONG_STATUS', 'next:MAHJONG_PREPARE_AI']);
  assert.deepEqual(c.saves, [{ url: NEXT.url, expect: NEXT, requireStored: true }]);
  assert.match(panel(c.state()), /A new version is downloaded/);
});

test('a page a hard reload loaded past its worker says to reload, not that storage is unavailable', async () => {
  const c = coordinator({ coreReady: true, controlled: false, impatient: true });
  assert.equal(await c.api.startOffline(), null);
  const html = panel(c.state());
  assert.match(html, /Reload once to enable offline play/);
  assert.doesNotMatch(html, /unavailable here/);
});

test('a first visit that lost its connection and then outlasts the wait keeps nothing of the lost attempt', async () => {
  const c = coordinator({ registered: false, coreReady: true, installs: true, controlled: false, impatient: true });
  c.faults.registration = true;
  assert.equal(await c.api.startOffline(), null);
  assert.equal(c.state().phase, 'unavailable');
  // The connection returns, and the install it starts outlasts the wait.
  c.faults.registration = false;
  assert.equal(await c.api.startOffline(), null, 'the page stopped waiting');
  const { supported, phase, warning } = c.state();
  assert.deepEqual({ supported, phase, warning }, { supported: true, phase: 'checking', warning: '' });
  assert.match(panel(c.state()), /Still saving for offline play/);
});

test('optional AI download and retry touch only the AI group, not already complete base files', async () => {
  const c = coordinator({ coreReady: true, failAi: true });
  await c.api.startOffline();
  await assert.rejects(c.api.prepareOfflineAi(), /AI interrupted/);
  assert.equal(c.state().phase, 'incomplete');
  assert.equal(c.state().coreReady, true);
  c.faults.ai = false;
  await Promise.all([c.api.prepareOfflineAi(), c.api.prepareOfflineAi()]);
  assert.equal(c.state().aiReady, true);
  assert.equal(c.calls.filter(type => type === 'MAHJONG_PREPARE_AI').length, 2);
  assert.equal(c.calls.includes('MAHJONG_PREPARE_CORE'), false);
});

test('storage failure is a visible offline warning, not a prohibition on online inference', async () => {
  const c = coordinator({ coreReady: true, failAi: true, storageFailure: true });
  await c.api.startOffline();
  assert.equal(await c.api.prepareOfflineAi(), null);
  assert.equal(c.state().aiReady, false);
  assert.match(c.state().warning, /Online play is still available/);
  c.faults.ai = false;
  await c.api.prepareOfflineAi();
  assert.equal(c.state().aiReady, true);
});

test('one save: the network by the page, the runtime by the worker, and the worker told once', async () => {
  const c = coordinator({ coreReady: true, held: false });
  await c.api.startOffline();
  const progress = [];
  c.api.watchOffline(value => { if (value.phase === 'ai') progress.push(value.progress); });
  await Promise.all([c.api.prepareOfflineAi(), c.api.prepareOfflineAi()]);
  assert.deepEqual(c.calls, ['MAHJONG_STATUS', 'MAHJONG_STATUS', 'MAHJONG_PREPARE_AI', 'MAHJONG_NETWORK_SAVED']);
  assert.deepEqual(c.saves, [{ url: NETWORK.url, expect: NETWORK, requireStored: true }]);
  assert.equal(c.state().aiReady, true);
  assert.equal(c.state().phase, 'ready');
  assert.equal(c.state().progress, 100);
  assert.ok(progress.includes(47), 'half the network is half of the 95 it counts for');
  assert.deepEqual(progress, [...progress].sort((a, b) => a - b), 'the progress never goes back');
  await c.api.prepareOfflineAi();
  assert.equal(c.saves.length, 1, 'a saved network is never fetched again');
});

test('a first visit saves its network at once, beside the install, and the runtime follows when the worker is there', async () => {
  // The install is still running: no worker is active to be asked anything.
  const c = coordinator({ registered: false, coreReady: true, installs: true, held: false });
  const phases = [];
  c.api.watchOffline(value => phases.push(value.phase));
  const job = c.api.prepareOfflineAi();
  // What the opponent's worker waits for is in hand while the game is saved.
  assert.equal(await c.api.prepareNetwork(), null);
  assert.deepEqual(c.saves, [{ url: NETWORK.url, expect: NETWORK, requireStored: true }]);
  await settle();
  assert.deepEqual(c.calls, [], 'nothing waited for the worker');
  const { phase, progress, installing, aiReady } = c.state();
  assert.deepEqual({ phase, progress, installing, aiReady }, { phase: 'ai', progress: 95, installing: true, aiReady: false },
    'the save is not whole: its runtime waits for the worker');
  assert.match(panel(c.state()), /Saving AI… 95%/);
  c.finishInstall('activated');
  assert.equal((await job).aiReady, true);
  assert.deepEqual(c.calls, ['MAHJONG_STATUS', 'MAHJONG_STATUS', 'MAHJONG_PREPARE_AI', 'MAHJONG_NETWORK_SAVED']);
  assert.equal(c.saves.length, 1);
  assert.match(panel(c.state()), /Offline: game \+ AI ready/);
  // The start's own word, checking and then ready, never stood over the save's.
  const said = phases.slice(phases.indexOf('ai'));
  assert.deepEqual([...new Set(said)], ['ai', 'ready']);
});

test('the two parts of a save finish in either order, and its progress counts both', async () => {
  // The runtime first: the network's transfer is held until the worker has saved the runtime.
  const c = coordinator({ coreReady: true, held: false });
  let release;
  c.faults.transfer = () => new Promise(done => { release = done; });
  await c.api.startOffline();
  const progress = [];
  c.api.watchOffline(value => { if (value.phase === 'ai') progress.push(value.progress); });
  const job = c.api.prepareOfflineAi();
  await until(() => c.calls.includes('MAHJONG_PREPARE_AI') && c.state().progress === 5, 'the runtime is saved first');
  assert.equal(c.state().phase, 'ai', 'and the save is not whole without its network');
  assert.equal(c.calls.includes('MAHJONG_NETWORK_SAVED'), false);
  release();
  assert.equal((await job).aiReady, true);
  assert.deepEqual([...new Set(progress)], [0, 5, 52, 100]);
  assert.equal(c.calls.at(-1), 'MAHJONG_NETWORK_SAVED');
});

test('the opponent\'s worker waits for the network alone, and a network that failed can be asked again at once', async () => {
  const c = coordinator({ registered: false, coreReady: true, installs: true, held: false });
  c.faults.network = new Error('The network could not be fetched: 503');
  await assert.rejects(c.api.prepareNetwork(), /could not be fetched/);
  assert.equal(c.state().phase, 'incomplete');
  assert.match(c.state().warning, /AI download incomplete/);
  // The install still runs, and the first save's runtime still waits for it:
  // the retry starts another save all the same, and its network is fetched.
  c.faults.network = null;
  assert.equal(await c.api.prepareNetwork(), null);
  assert.equal(c.saves.length, 2);
  assert.equal(c.state().phase, 'ai');
  c.finishInstall('activated');
  await until(() => c.state().aiReady, 'the runtime follows');
  assert.equal(c.state().phase, 'ready');
});

test('the network is saved whatever becomes of the runtime, which is said once the network is in hand', async () => {
  // The owner's choice: the first Trained move comes before the bytes a
  // runtime that cannot be saved would have spared.
  const c = coordinator({ coreReady: true, held: false, failAi: true });
  await c.api.startOffline();
  const job = c.api.prepareOfflineAi(), failed = assert.rejects(job, /AI interrupted/);
  assert.equal(await c.api.prepareNetwork(), null, 'online play has its network');
  await failed;
  assert.equal(c.saves.length, 1);
  assert.equal(c.state().phase, 'incomplete');
  assert.match(c.state().warning, /AI download incomplete: AI interrupted/);
  assert.equal(c.calls.includes('MAHJONG_NETWORK_SAVED'), false);
  // The retry fetches no network again.
  c.faults.ai = false;
  assert.equal((await c.api.prepareOfflineAi()).aiReady, true);
  assert.equal(c.saves.length, 1);
});

test('a runtime owed to a worker that came late is saved when that worker takes control', async () => {
  // The install outlasts the save's wait for it, as on a slow first visit.
  const c = coordinator({ registered: false, coreReady: true, installs: true, controlled: false, impatient: true, held: false });
  const job = c.api.prepareOfflineAi();
  assert.equal(await c.api.prepareNetwork(), null);
  assert.equal(await job, null, 'no worker yet to save the runtime');
  assert.equal(c.saves.length, 1);
  assert.equal(c.state().phase, 'checking');
  assert.equal(c.calls.includes('MAHJONG_PREPARE_AI'), false);
  c.finishInstall('activated');
  c.claim();
  await until(() => c.state().aiReady, 'the page saves the runtime by itself');
  assert.ok(c.calls.includes('MAHJONG_PREPARE_AI'));
  assert.equal(c.saves.length, 1, 'and fetches no network again');
  assert.match(panel(c.state()), /Offline: game \+ AI ready/);
});

test('where nothing can be saved the page leaves the network to the opponent\'s worker', async () => {
  const c = coordinator({ held: false });
  delete c.navigator.serviceWorker;
  assert.equal(await c.api.prepareNetwork(), null);
  assert.equal(await c.api.prepareOfflineAi(), null);
  assert.deepEqual(c.saves, []);
  assert.equal(c.state().phase, 'unavailable');
});

test('a page download that fails is incomplete and a storage failure is only an offline warning', async () => {
  const c = coordinator({ coreReady: true, held: false });
  await c.api.startOffline();
  c.faults.network = new Error('The network could not be fetched: 503');
  await assert.rejects(c.api.prepareOfflineAi(), /could not be fetched/);
  assert.equal(c.state().phase, 'incomplete');
  assert.match(c.state().warning, /AI download incomplete/);
  c.faults.network = Object.assign(new Error('The network is not saved offline: quota'), { name: 'NetworkStorageError' });
  assert.equal(await c.api.prepareOfflineAi(), null);
  assert.match(c.state().warning, /Online play is still available/);
  assert.equal(c.calls.includes('MAHJONG_NETWORK_SAVED'), false);
  c.faults.network = null;
  await c.api.prepareOfflineAi();
  assert.equal(c.state().aiReady, true);
});

test('a player who asked for Trained has a new version\'s network saved once, and only online', async () => {
  const c = coordinator({ coreReady: true, aiRequested: true, held: false, online: false });
  await c.api.startOffline();
  await settle();
  assert.deepEqual(c.saves, [], 'nothing is attempted offline');
  c.navigator.onLine = true;
  c.faults.network = new Error('Connection lost');
  await c.api.refreshOffline();
  await settle();
  assert.equal(c.saves.length, 1, 'the reconnect starts it');
  assert.equal(c.state().phase, 'incomplete');
  for (let i = 0; i < 3; i++) { await c.api.refreshOffline(); await settle(); }
  assert.equal(c.saves.length, 1, 'returning to the app does not download it again');
  c.faults.network = null;
  await c.api.prepareOfflineAi();
  assert.equal(c.state().aiReady, true, 'an explicit Trained request still retries');
});

test('a player who never asked for Trained downloads nothing automatically', async () => {
  const c = coordinator({ coreReady: true, held: false });
  await c.api.startOffline(); await c.api.refreshOffline(); await settle();
  assert.deepEqual(c.saves, []);
  assert.equal(c.calls.includes('MAHJONG_PREPARE_AI'), false);
});

test('a waiting update\'s runtime and network are saved while this version runs', async () => {
  const c = coordinator({ coreReady: true, aiReady: true, aiRequested: true, waiting: { aiRequested: true } });
  await c.api.startOffline();
  await settle();
  assert.equal(c.state().updateReady, true);
  assert.deepEqual(c.calls.filter(call => call.startsWith('next:')), ['next:MAHJONG_STATUS', 'next:MAHJONG_PREPARE_AI']);
  assert.deepEqual(c.saves, [{ url: NEXT.url, expect: NEXT, requireStored: true }]);
  assert.equal(c.calls.includes('MAHJONG_NETWORK_SAVED'), false, 'the active version prunes nothing for it');
  await c.api.refreshOffline(); await settle();
  assert.equal(c.saves.length, 1, 'once per waiting update');
});

test('a waiting update\'s network waits for this version\'s own', async () => {
  const c = coordinator({ coreReady: true, aiRequested: true, held: false, waiting: { aiRequested: true } });
  await c.api.startOffline();
  await settle();
  assert.deepEqual(c.saves.map(save => save.url), [NETWORK.url, NEXT.url]);
  assert.equal(c.state().aiReady, true);
});

test('one network download at a time: a retry waits for a waiting update\'s download under way', async () => {
  // Each transfer holds a whole network, 116 MB, until it is stored.
  const c = coordinator({ coreReady: true, aiRequested: true, held: false, waiting: { aiRequested: true } });
  let running = 0, most = 0, ownTries = 0, release;
  c.faults.transfer = async url => {
    running++; most = Math.max(most, running);
    try {
      if (url === NETWORK.url && ++ownTries === 1) throw new Error('Connection lost');
      if (url === NEXT.url) await new Promise(done => { release = done; });
    } finally { running--; }
  };
  await c.api.startOffline();
  await settle();
  // This version's automatic save failed, so the update's started.
  assert.equal(c.state().phase, 'incomplete');
  assert.deepEqual(c.saves.map(save => save.url), [NETWORK.url, NEXT.url]);
  const retry = c.api.prepareOfflineAi();
  await settle();
  assert.deepEqual(c.saves.map(save => save.url), [NETWORK.url, NEXT.url], 'the retry waits its turn');
  release();
  assert.equal((await retry).aiReady, true);
  assert.deepEqual(c.saves.map(save => save.url), [NETWORK.url, NEXT.url, NETWORK.url]);
  assert.equal(most, 1);
});

test('a waiting update is left alone without a Trained request or a connection', async () => {
  for (const options of [{ waiting: { aiRequested: false } }, { waiting: { aiRequested: true }, online: false }]) {
    const c = coordinator({ coreReady: true, aiReady: true, ...options });
    await c.api.startOffline(); await settle();
    assert.deepEqual(c.saves, []);
    assert.equal(c.calls.includes('next:MAHJONG_PREPARE_AI'), false);
  }
});

test('a Trained request after a failed start tries the service worker again', async () => {
  const c = coordinator({ registered: false, coreReady: true });
  c.faults.registration = true;
  assert.equal(await c.api.startOffline(), null);
  assert.equal(await c.api.prepareOfflineAi(), null, 'still failing: online play, with its warning');
  assert.equal(c.registrations(), 2, 'a failed start is not handed back');
  c.faults.registration = false;
  const info = await c.api.prepareOfflineAi();
  assert.equal(c.registrations(), 3);
  assert.equal(info.aiReady, true);
  assert.equal(c.state().supported, true);
});

test('bytes storage refused are handed on for online play, not downloaded again', async () => {
  const c = coordinator({ coreReady: true, held: false });
  await c.api.startOffline();
  const bytes = new Uint8Array(116);
  c.faults.network = Object.assign(new Error('The network is not saved offline: quota'), { name: 'NetworkStorageError', bytes });
  const prepared = await c.api.prepareOfflineAi();
  assert.equal(prepared.aiReady, false);
  assert.equal(prepared.unstoredNetwork, bytes);
  assert.match(c.state().warning, /Online play is still available/);
  assert.equal(c.saves.length, 1);
});

test('a network storage has no room for is not downloaded only to be refused', async () => {
  const c = coordinator({ coreReady: true, held: false });
  c.navigator.storage.estimate = async () => ({ quota: 1000, usage: 1000 - NETWORK.bytes + 1 });
  await c.api.startOffline();
  assert.equal(await c.api.prepareOfflineAi(), null);
  assert.deepEqual(c.saves, [], 'no download');
  assert.ok(c.calls.includes('MAHJONG_PREPARE_AI'), 'the few megabytes of runtime are still saved');
  assert.equal(c.state().phase, 'incomplete');
  assert.match(c.state().warning, /too little free storage.*Online play is still available/);
  c.navigator.storage.estimate = async () => ({ quota: 1000, usage: 1000 - NETWORK.bytes });
  assert.equal((await c.api.prepareOfflineAi()).aiReady, true, 'room for it exactly is room');
  assert.equal(c.saves.length, 1);
});

test('the page saves the network its player picked, and names the pick to its worker and to a waiting update', async () => {
  const OLDER = { url: 'https://model.invalid/g410', version: '0.1.410', sha256: '4'.repeat(64), bytes: 118 };
  const c = coordinator({ coreReady: true, aiRequested: true, held: false, network: OLDER, pick: '0.1.410',
    waiting: { aiRequested: true } });
  await c.api.startOffline();
  await settle();
  // Its own network first, then the one the waiting worker names for the same pick.
  assert.deepEqual(c.saves, [{ url: OLDER.url, expect: OLDER, requireStored: true },
    { url: NEXT.url, expect: NEXT, requireStored: true }]);
  assert.equal(c.state().aiReady, true);
  assert.ok(c.asked.some(([call]) => call === 'next:MAHJONG_STATUS'));
  assert.ok(c.asked.some(([call]) => call === 'MAHJONG_NETWORK_SAVED'), 'its worker then keeps the picked network');
  assert.deepEqual([...new Set(c.asked.map(([, version]) => version))], ['0.1.410'], 'every message names the pick');
  // A player who picked nothing names nothing: each worker answers for its own default.
  const d = coordinator({ coreReady: true, aiRequested: true, held: false, waiting: { aiRequested: true } });
  await d.api.startOffline();
  await settle();
  assert.ok(d.asked.length > 2);
  assert.ok(d.asked.every(([, version]) => version === undefined));
  assert.deepEqual(d.saves.map(save => save.url), [NETWORK.url, NEXT.url]);
});
