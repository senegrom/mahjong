import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { MessageChannel } from 'node:worker_threads';

const source = (await readFile(new URL('../src/lib/offline.js', import.meta.url), 'utf8'))
  .replaceAll('import.meta.env.DEV', 'false').replaceAll('export ', '');

// Exercise the real coordinator against the service-worker message contract.
// No browser globals or state are shared between tests.
function coordinator({ coreReady = false, aiReady = false, failCore = false, failAi = false, failStrong = false, registered = true } = {}) {
  const calls = [], info = { coreReady, aiReady, strongReady: false, hasModel: true, hasStrongModel: true, version: 'test' };
  const faults = { core: failCore, ai: failAi, strong: failStrong, registration: false };
  let latest, registrations = 0;
  const active = {
    scriptURL: 'https://test.invalid/mahjong/sw.js',
    postMessage({ type }, [port]) {
      calls.push(type);
      queueMicrotask(() => {
        if (type === 'MAHJONG_PREPARE_CORE' && faults.core) port.postMessage({ error: 'Connection lost' });
        else if (type === 'MAHJONG_PREPARE_AI' && faults.ai) port.postMessage({ error: 'AI interrupted' });
        else if (type === 'MAHJONG_PREPARE_STRONG' && faults.strong) port.postMessage({ error: 'Strong interrupted' });
        else {
          if (type === 'MAHJONG_PREPARE_CORE') info.coreReady = true;
          if (type === 'MAHJONG_PREPARE_AI') info.aiReady = true;
          if (type === 'MAHJONG_PREPARE_STRONG') { info.aiReady = true; info.strongReady = true; }
          port.postMessage({ value: { ...info } });
        }
        port.close();
      });
    },
  };
  const registration = { scope: 'https://test.invalid/mahjong/', active, addEventListener() {}, async update() {} };
  const navigator = { onLine: true, storage: {}, serviceWorker: {
    controller: active,
    async getRegistration() { return registered ? registration : undefined; },
    async register() {
      registrations++;
      if (faults.registration) throw new Error('Connection lost');
      registered = true;
      return registration;
    },
  } };
  const context = vm.createContext({ document: { baseURI: 'https://test.invalid/mahjong/' }, navigator,
    URL, MessageChannel, setTimeout, clearTimeout, caches: {}, isSecureContext: true });
  vm.runInContext(source + '\nglobalThis.api = { startOffline, refreshOffline, prepareOfflineAi, watchOffline };', context);
  context.api.watchOffline(value => latest = value);
  return { api: context.api, calls, info, faults, state: () => latest, registrations: () => registrations };
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

test('a failed Strong download remains retryable when Quick is already cached', async () => {
  const c = coordinator({ coreReady: true, aiReady: true, failStrong: true });
  await c.api.startOffline();
  await assert.rejects(c.api.prepareOfflineAi('strong'), /Strong interrupted/);
  assert.equal(c.state().aiReady, true);
  assert.equal(c.state().strongReady, false);
  assert.equal(c.state().phase, 'incomplete');
  c.faults.strong = false;
  await c.api.prepareOfflineAi('strong');
  assert.equal(c.state().strongReady, true);
  assert.equal(c.state().phase, 'ready');
  assert.equal(c.calls.filter(type => type === 'MAHJONG_PREPARE_STRONG').length, 2);
  assert.equal(c.calls.includes('MAHJONG_PREPARE_CORE'), false);
});
