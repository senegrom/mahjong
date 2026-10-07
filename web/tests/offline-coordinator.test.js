import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { MessageChannel } from 'node:worker_threads';
import { loadModule } from './fixtures/load-module.js';

const NETWORK = { url: 'https://model.invalid/g1', sha256: '1'.repeat(64), bytes: 116 };
const NEXT = { url: 'https://model.invalid/g2', sha256: '2'.repeat(64), bytes: 117 };

/** A service worker speaking the real message contract: it saves the game and
 * the runtime, reports whether the network is stored, and never fetches it. */
function fakeWorker({ calls, network, stored, info, faults, label = '' }) {
  return {
    scriptURL: 'https://test.invalid/mahjong/sw.js',
    postMessage({ type }, [port]) {
      calls.push(label + type);
      queueMicrotask(() => {
        const reply = () => ({ ...info, aiReady: info.runtime && stored.has(network.url), network });
        if (type === 'MAHJONG_PREPARE_CORE' && faults.core) port.postMessage({ error: 'Connection lost' });
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
// No browser globals or state are shared between tests.
function coordinator({ coreReady = false, aiReady = false, aiRequested = false, failCore = false, failAi = false,
  registered = true, held = true, storageFailure = false, online = true, waiting = null, installs = false } = {}) {
  const calls = [], saves = [], stored = new Set(held || aiReady ? [NETWORK.url] : []);
  const info = { coreReady, runtime: aiReady, aiRequested, hasModel: true, version: 'test' };
  const faults = { core: failCore, ai: failAi, storage: storageFailure, registration: false, network: null };
  let latest, registrations = 0;
  const active = fakeWorker({ calls, network: NETWORK, stored, info, faults });
  // With installs, the first registration's worker is still installing: it
  // activates (or fails) only when the test finishes the install.
  const changes = new Set();
  const installing = installs ? { state: 'installing', addEventListener: (_, listener) => changes.add(listener),
    removeEventListener: (_, listener) => changes.delete(listener) } : null;
  const registration = { scope: 'https://test.invalid/mahjong/', active: installs ? null : active, installing,
    addEventListener() {}, async update() {},
    waiting: waiting && fakeWorker({ calls, network: NEXT, stored, faults: {},
      info: { coreReady: true, runtime: false, hasModel: true, version: 'next', ...waiting }, label: 'next:' }) };
  function finishInstall(state = 'activated') {
    installing.state = state;
    if (state === 'activated') { registration.active = active; registration.installing = null; }
    for (const listener of [...changes]) listener();
  }
  const navigator = { onLine: online, storage: {}, serviceWorker: {
    controller: active,
    async getRegistration() { return registered ? registration : undefined; },
    async register() {
      registrations++;
      if (faults.registration) throw new Error('Connection lost');
      registered = true;
      return registration;
    },
  } };
  // offline.js's one import, network-store.js, is stood in for here.
  const api = loadModule('offline.js', { document: { baseURI: 'https://test.invalid/mahjong/' }, navigator,
    URL, MessageChannel, setTimeout, clearTimeout, caches: {}, isSecureContext: true, WeakSet,
    NETWORK, NETWORK_URL: NETWORK.url,
    // The page's own record of a saved network: a header check, never a read.
    networkIsStored: async (url = NETWORK.url) => stored.has(url),
    // The page's download: verified, then stored durably, or it throws.
    networkBytes: async ({ url = NETWORK.url, expect, requireStored, onProgress } = {}) => {
      saves.push({ url, expect: expect && { ...expect }, requireStored });
      if (faults.network) throw faults.network;
      onProgress?.({ bytes: 1, total: 2 });
      stored.add(url);
      return new Uint8Array(1);
    } }, ['startOffline', 'refreshOffline', 'prepareOfflineAi', 'watchOffline'], { 'import.meta.env.DEV': 'false' });
  api.watchOffline(value => latest = value);
  return { api, calls, info, faults, stored, saves, navigator, state: () => latest,
    registrations: () => registrations, finishInstall };
}
const settle = async () => { for (let i = 0; i < 20; i++) await setImmediate(); };

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
  assert.equal(c.state().coreLoading, false);
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
  // A Trained default's network waits for this install, which reports no
  // progress of its own; the table can only say that it is running.
  for (const ending of ['activated', 'redundant']) {
    const c = coordinator({ registered: false, coreReady: true, installs: true });
    const started = c.api.startOffline();
    await settle();
    assert.equal(c.state().installing, true, ending);
    assert.equal(c.state().phase, 'checking');
    c.finishInstall(ending);
    const result = await started;
    assert.equal(c.state().installing, false, ending);
    assert.equal(c.state().phase, ending === 'activated' ? 'ready' : 'unavailable');
    assert.equal(Boolean(result), ending === 'activated');
  }
  // A returning visit's worker is already active: nothing is installing.
  const returning = coordinator({ coreReady: true });
  const seen = [];
  returning.api.watchOffline(value => seen.push(value.installing));
  await returning.api.startOffline();
  assert.ok(seen.length > 1 && seen.every(value => value === false));
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

test('the page saves the network after the worker saves the runtime, once', async () => {
  const c = coordinator({ coreReady: true, held: false });
  await c.api.startOffline();
  const progress = [];
  c.api.watchOffline(value => { if (value.phase === 'ai') progress.push(value.progress); });
  await Promise.all([c.api.prepareOfflineAi(), c.api.prepareOfflineAi()]);
  assert.deepEqual(c.calls, ['MAHJONG_STATUS', 'MAHJONG_STATUS', 'MAHJONG_PREPARE_AI', 'MAHJONG_NETWORK_SAVED']);
  assert.deepEqual(c.saves, [{ url: NETWORK.url, expect: NETWORK, requireStored: true }]);
  assert.equal(c.state().aiReady, true);
  assert.equal(c.state().phase, 'ready');
  assert.ok(progress.includes(60), 'the network is the last four fifths of the progress');
  await c.api.prepareOfflineAi();
  assert.equal(c.saves.length, 1, 'a saved network is never fetched again');
});

test('a runtime that cannot be saved spends none of the network\'s bytes', async () => {
  const c = coordinator({ coreReady: true, held: false, failAi: true });
  await c.api.startOffline();
  await assert.rejects(c.api.prepareOfflineAi(), /AI interrupted/);
  assert.deepEqual(c.saves, []);
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
