import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { createHash, webcrypto } from 'node:crypto';
import { serviceWorkerTemplate } from '../scripts/offline-build.mjs';
import { networkCacheName, verifiedNetworkBytes } from '../src/lib/network-transfer.js';

const template = await serviceWorkerTemplate();
const scope = 'https://test.invalid/mahjong/';
const digest = body => createHash('sha256').update(body).digest('hex');
const address = request => String(request.url ?? request);
function cacheStorageFixture() {
  const stores = new Map();
  return { async open(name) {
    if (!stores.has(name)) {
      const entries = new Map();
      stores.set(name, {
        async match(key) { return entries.get(address(key))?.clone(); },
        async put(key, value) { entries.set(address(key), value.clone()); },
        async delete(key) { return entries.delete(address(key)); },
        async keys() { return [...entries.keys()].map(url => new Request(url)); },
      });
    }
    return stores.get(name);
  } };
}
function worker(caches, generation, { broken = false } = {}) {
  const events = {}, calls = [];
  const body = `weights-${generation}`, url = `https://model.invalid/g${generation}`;
  const files = { 'index.html': `shell-${generation}`, 'ort/runtime.wasm': 'runtime' };
  const config = { version: String(generation), hasModel: true, networkCacheVersion: 2,
    network: { origin: 'https://model.invalid', object: `g${generation}`, bytes: body.length, sha256: digest(body) },
    entries: Object.entries(files).map(([url, body]) => ({ url, bytes: body.length, hash: digest(body), group: url.startsWith('ort/') ? 'ai' : 'core' })) };
  const registration = { scope, installing: null, waiting: null };
  vm.runInNewContext(template.replace('/* OFFLINE_CONFIG */ null', JSON.stringify(config)), {
    self: { registration, clients: { async claim() {} }, addEventListener(type, body) { events[type] = body; } },
    URL, Request, Response, caches, crypto: webcrypto, AbortController, AbortSignal, DOMException, setTimeout, clearTimeout,
    fetch: async request => {
      const name = String(request); calls.push(name);
      const relative = name.slice(scope.length);
      return new Response(files[relative], { status: relative in files ? 200 : 404 });
    },
  });
  async function event(name) { let pending; events[name]({ waitUntil(value) { pending = value; } }); await pending; }
  async function message(type) {
    let pending, reply;
    events.message({ data: { type }, ports: [{ postMessage(value) { if (!value.progress) reply = value; } }], waitUntil(value) { pending = value; } });
    await pending;
    if (reply.error) throw new Error(reply.error);
    return reply.value;
  }
  // The page's part (offline.js): fetch, verify and store the network with the
  // real transfer code against the same Cache Storage. This runs outside the
  // worker, which is never asked to fetch it.
  async function save() {
    const previous = { caches: globalThis.caches, fetch: globalThis.fetch };
    globalThis.caches = caches;
    globalThis.fetch = async request => { calls.push(`page:${request}`); return new Response(broken ? 'wrong bytes' : body); };
    try { await verifiedNetworkBytes({ url, expect: config.network, requireStored: true, scope }); }
    finally { globalThis.caches = previous.caches; globalThis.fetch = previous.fetch; }
  }
  return { url, config, calls, registration, install: () => event('install'), activate: () => event('activate'), save,
    download: async () => { await message('MAHJONG_PREPARE_AI'); await save(); return message('MAHJONG_NETWORK_SAVED'); },
    saved: () => message('MAHJONG_NETWORK_SAVED'), status: () => message('MAHJONG_STATUS') };
}

test('the active worker keeps its network and the previous one, pruning after a save or an activation', async () => {
  const caches = cacheStorageFixture(), cache = await caches.open(networkCacheName(scope));
  const first = worker(caches, 1); await first.install(); await first.activate(); await first.download();
  const second = worker(caches, 2); await second.install(); await second.activate(); await second.download();
  const third = worker(caches, 3); await third.install();
  assert.ok(await cache.match(first.url), 'installation cannot remove the previous retained model');
  assert.ok(await cache.match(second.url), 'active model survives installation');
  await third.activate();
  assert.ok(await cache.match(first.url), 'activating before the new network is saved removes nothing');
  assert.equal((await third.status()).aiReady, false, 'the previous network is never reported as this version\'s');
  await third.download();
  assert.equal(await cache.match(first.url), undefined);
  assert.ok(await cache.match(second.url)); assert.ok(await cache.match(third.url));
  const status = await third.status();
  assert.equal(status.aiReady, true);
  assert.deepEqual(JSON.parse(JSON.stringify(status.network)), { url: third.url, sha256: third.config.network.sha256, bytes: third.config.network.bytes });
  for (const w of [first, second, third]) {
    assert.deepEqual(w.calls.filter(call => call.includes('model.invalid')), [`page:${w.url}`], 'only the page fetches the network');
  }
});

test('a network saved while its version waits becomes current at activation', async () => {
  const caches = cacheStorageFixture(), cache = await caches.open(networkCacheName(scope));
  for (const n of [1, 2]) { const w = worker(caches, n); await w.install(); await w.activate(); await w.download(); }
  const third = worker(caches, 3); await third.install();
  // The running version's page saves the waiting update's runtime and network.
  third.registration.waiting = {};
  await third.download();
  assert.ok(await cache.match('https://model.invalid/g1'), 'nothing is pruned while an update waits');
  third.registration.waiting = null;
  await third.activate();
  assert.equal(await cache.match('https://model.invalid/g1'), undefined);
  assert.ok(await cache.match('https://model.invalid/g2'));
  assert.ok(await cache.match(third.url));
  assert.equal((await third.status()).aiReady, true);
});

test('failed model upgrade cannot replace retention metadata or remove working models', async () => {
  const caches = cacheStorageFixture(), cache = await caches.open(networkCacheName(scope));
  for (const n of [1, 2]) { const w = worker(caches, n); await w.install(); await w.activate(); await w.download(); }
  const key = new URL('__network_retention__', scope), before = await (await cache.match(key)).text();
  const bad = worker(caches, 3, { broken: true });
  await bad.install(); await bad.activate();
  await assert.rejects(bad.download(), /byte|built for/);
  await bad.saved();
  assert.equal(await (await cache.match(key)).text(), before);
  assert.ok(await cache.match('https://model.invalid/g1'));
  assert.ok(await cache.match('https://model.invalid/g2'));
  assert.equal(await cache.match(bad.url), undefined);
  assert.equal((await bad.status()).aiReady, false);
});

test('activation skips model pruning while a newer registration is waiting', async () => {
  const caches = cacheStorageFixture(), cache = await caches.open(networkCacheName(scope));
  for (const n of [1, 2]) { const w = worker(caches, n); await w.install(); await w.activate(); await w.download(); }
  const third = worker(caches, 3); await third.install(); await third.save();
  third.registration.waiting = {};
  await third.activate();
  await third.saved();
  assert.ok(await cache.match('https://model.invalid/g1'));
  assert.ok(await cache.match('https://model.invalid/g2'));
  assert.ok(await cache.match(third.url));
});
