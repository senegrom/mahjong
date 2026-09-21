import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { createHash, webcrypto } from 'node:crypto';
import { serviceWorkerTemplate } from '../scripts/offline-build.mjs';
import { networkCacheName } from '../src/lib/network-transfer.js';

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
      if (name === url) return new Response(broken ? 'wrong bytes' : body);
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
  return { url, config, calls, registration, install: () => event('install'), activate: () => event('activate'),
    download: () => message('MAHJONG_PREPARE_AI'), status: () => message('MAHJONG_STATUS') };
}

test('production worker pins first activation and prunes only after later activation', async () => {
  const caches = cacheStorageFixture(), cache = await caches.open(networkCacheName(scope));
  const first = worker(caches, 1); await first.install(); await first.activate(); await first.download();
  const second = worker(caches, 2); await second.install(); await second.activate();
  const third = worker(caches, 3); await third.install();
  assert.ok(await cache.match(first.url), 'installation cannot remove the previous retained model');
  assert.ok(await cache.match(second.url), 'active model survives installation');
  await third.activate();
  assert.equal(await cache.match(first.url), undefined);
  assert.ok(await cache.match(second.url)); assert.ok(await cache.match(third.url));
  const status = await third.status();
  assert.equal(status.aiReady, true);
  assert.deepEqual(JSON.parse(JSON.stringify(status.network)), { url: third.url, sha256: third.config.network.sha256, bytes: third.config.network.bytes });
});

test('failed model upgrade cannot replace retention metadata or remove working models', async () => {
  const caches = cacheStorageFixture(), cache = await caches.open(networkCacheName(scope));
  for (const n of [1, 2]) { const w = worker(caches, n); await w.install(); await w.activate(); await w.download(); }
  const key = new URL('__network_retention__', scope), before = await (await cache.match(key)).text();
  const bad = worker(caches, 3, { broken: true });
  await assert.rejects(bad.install(), /byte|built for/);
  assert.equal(await (await cache.match(key)).text(), before);
  assert.ok(await cache.match('https://model.invalid/g1'));
  assert.ok(await cache.match('https://model.invalid/g2'));
});

test('activation skips model pruning while a newer registration is waiting', async () => {
  const caches = cacheStorageFixture(), cache = await caches.open(networkCacheName(scope));
  for (const n of [1, 2]) { const w = worker(caches, n); await w.install(); await w.activate(); await w.download(); }
  const third = worker(caches, 3); await third.install();
  third.registration.waiting = {};
  await third.activate();
  assert.ok(await cache.match('https://model.invalid/g1'));
  assert.ok(await cache.match('https://model.invalid/g2'));
  assert.ok(await cache.match(third.url));
});
