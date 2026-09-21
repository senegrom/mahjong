import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { NETWORK_CACHE, networkCacheName, verifiedNetworkBytes, verifiedNetworkIsStored, pruneNetworkCache } from '../src/lib/network-transfer.js';
import { networkIsStored } from '../src/lib/network-store.js';

const scope = 'https://test.invalid/mahjong/';
const key = request => typeof request === 'string' ? request : request.url ?? request.href;
function cacheStorageFixture(t) {
  const old = globalThis.caches, stores = new Map();
  const caches = { async open(name) {
    if (!stores.has(name)) {
      const entries = new Map();
      stores.set(name, {
        entries,
        async match(request) { return entries.get(key(request))?.clone(); },
        async put(request, response) { entries.set(key(request), response.clone()); },
        async delete(request) { return entries.delete(key(request)); },
        async keys() { return [...entries.keys()].map(url => new Request(url)); },
      });
    }
    return stores.get(name);
  } };
  globalThis.caches = caches;
  t.after(() => { if (old === undefined) delete globalThis.caches; else globalThis.caches = old; });
  return caches;
}
function model(n) {
  const body = `model-${n}`;
  return { url: `https://models.invalid/g${n}/weights`, body,
    expect: { bytes: Buffer.byteLength(body), sha256: createHash('sha256').update(body).digest('hex') } };
}
async function store(cache, m) { await cache.put(m.url, new Response(m.body)); }
const prune = (m, options = {}) => pruneNetworkCache({ scope, ...m, ...options });

// These tests use the actual transfer/verification code and actual Response
// bodies, rather than accepting metadata as proof of a valid network.
test('scope-owned caches retain only the current and previous activated models', async t => {
  const caches = cacheStorageFixture(t), cache = await caches.open(networkCacheName(scope));
  for (let n = 1; n <= 6; n++) {
    const m = model(n);
    await store(cache, m);
    assert.equal(await prune(m), true);
    const held = (await cache.keys()).map(k => k.url).filter(url => url.startsWith('https://models.invalid/'));
    assert.deepEqual(held, [model(n - 1).url, m.url].slice(n === 1 ? 1 : 0));
  }
  // A shell-only update must not consume the previous-model retention slot.
  await prune(model(6));
  assert.ok(await cache.match(model(5).url));
});

test('the first activation can precede the optional model download', async t => {
  const caches = cacheStorageFixture(t), cache = await caches.open(networkCacheName(scope));
  assert.equal(await prune(model(1)), false);
  await store(cache, model(1));
  await store(cache, model(2));
  assert.equal(await prune(model(2)), true);
  assert.ok(await cache.match(model(1).url), 'the first model remains the rollback model');
});

test('waiting updates, failed replacements and interrupted cleanup keep working models', async t => {
  const caches = cacheStorageFixture(t), cache = await caches.open(networkCacheName(scope));
  await store(cache, model(1)); await prune(model(1));
  await store(cache, model(2)); await prune(model(2));
  await store(cache, model(3));
  assert.equal(await prune(model(3), { canPrune: () => false }), false);
  assert.ok(await cache.match(model(1).url));
  await cache.put(model(3).url, new Response('tampered'));
  assert.equal(await prune(model(3)), false);
  assert.ok(await cache.match(model(1).url));
  assert.ok(await cache.match(model(2).url));
  await store(cache, model(3));
  let checks = 0;
  assert.equal(await prune(model(3), { canPrune: () => ++checks < 3 }), false);
  assert.ok(await cache.match(model(1).url), 'no destructive operation after a new install starts');
});

test('cleanup never touches another installation or the unowned legacy cache', async t => {
  const caches = cacheStorageFixture(t), cache = await caches.open(networkCacheName(scope));
  const other = await caches.open(networkCacheName('https://test.invalid/other/'));
  const legacy = await caches.open(NETWORK_CACHE);
  await store(other, model(9)); await store(legacy, model(8));
  for (const n of [1, 2, 3]) { await store(cache, model(n)); await prune(model(n)); }
  assert.ok(await other.match(model(9).url));
  assert.ok(await legacy.match(model(8).url));
});

test('a verified legacy body migrates without fetching or deleting its shared copy', async t => {
  const caches = cacheStorageFixture(t), legacy = await caches.open(NETWORK_CACHE), m = model(1);
  await store(legacy, m);
  const oldFetch = globalThis.fetch;
  globalThis.fetch = () => { throw new Error('must not fetch a cached network'); };
  t.after(() => { globalThis.fetch = oldFetch; });
  assert.equal(await verifiedNetworkIsStored(m.url, m.expect, { scope }), true);
  let durable = false;
  const bytes = await verifiedNetworkBytes({ ...m, scope, requireStored: true, onStorage: saved => { durable = saved; } });
  assert.equal(new TextDecoder().decode(bytes), m.body);
  assert.equal(durable, true);
  assert.ok(await (await caches.open(networkCacheName(scope))).match(m.url));
  assert.ok(await legacy.match(m.url));
});

test('failed migration preserves verified legacy offline availability', async t => {
  const caches = cacheStorageFixture(t), m = model(1);
  await store(await caches.open(NETWORK_CACHE), m);
  const cache = await caches.open(networkCacheName(scope));
  cache.put = async () => { throw new Error('quota'); };
  const bytes = await verifiedNetworkBytes({ ...m, scope, requireStored: true });
  assert.equal(new TextDecoder().decode(bytes), m.body);
  assert.equal(await verifiedNetworkIsStored(m.url, m.expect, { scope }), true);
});

test('verified worker status avoids a second read only for the exact requested identity', async t => {
  const caches = cacheStorageFixture(t), m = model(1);
  let opened = 0;
  const open = caches.open;
  caches.open = async name => { opened++; return open(name); };
  const verified = { url: m.url, ...m.expect };
  assert.equal(await networkIsStored(m.url, m.expect, verified), true);
  assert.equal(opened, 0);
  for (const wrong of [{ ...verified, url: 'https://models.invalid/other' },
    { ...verified, bytes: m.expect.bytes + 1 }, { ...verified, sha256: '0'.repeat(64) }, undefined]) {
    assert.equal(await networkIsStored(m.url, m.expect, wrong), false);
  }
  assert.equal(opened, 4, 'mismatched and legacy replies require real verification');
});
