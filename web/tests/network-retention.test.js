import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { NETWORK_CACHE, networkCacheName, hasStoredNetwork, pruneNetworkCache } from '../src/lib/network-transfer.js';
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
// As verifiedNetworkBytes stores a network it has hashed: the body and its record.
async function store(cache, m) {
  await cache.put(m.url, new Response(m.body, { headers: {
    'Content-Length': String(m.expect.bytes), 'X-Mahjong-SHA256': m.expect.sha256 } }));
}
const prune = (m, options = {}) => pruneNetworkCache({ scope, ...m, ...options });
// As the worker prunes after a page has saved the model and named its pick.
const saved = (m, options = {}) => prune(m, { saved: true, ...options });

// These tests use the actual transfer/verification code and actual Response
// bodies. Retention reads the record a verified write leaves; bytes handed
// to the runtime are hashed again.
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

test('a model the player picked is kept while it is listed, whatever was saved after it; a former default nobody picked goes', async t => {
  const caches = cacheStorageFixture(t), cache = await caches.open(networkCacheName(scope));
  const held = async () => (await cache.keys()).map(k => k.url).filter(url => url.startsWith('https://models.invalid/')).sort();
  // The build lists five, its default first. 1 and 2 were picked; 3 and 4
  // were each the default once, and nobody picked them.
  const listed = [5, 1, 2, 3, 4].map(n => model(n).url);
  for (const n of [1, 2, 3, 4, 5]) { await store(cache, model(n)); await saved(model(n), { listed, picked: n < 3 }); }
  // 5 is the current model, 4 the one before it, and 1 and 2 are picks.
  assert.deepEqual(await held(), [1, 2, 4, 5].map(n => model(n).url));
  const record = await (await cache.match(new URL('__network_retention__', scope).href)).json();
  assert.deepEqual(record, { current: model(5).url, previous: model(4).url, picked: [model(1).url, model(2).url] });
  // A pick the site no longer lists goes like any other model.
  await store(cache, model(6)); await saved(model(6), { listed: [model(6).url, model(2).url] });
  assert.deepEqual(await held(), [2, 5, 6].map(n => model(n).url));
});

test('the default is kept though the player picks one model after another', async t => {
  const caches = cacheStorageFixture(t), cache = await caches.open(networkCacheName(scope));
  const listed = [9, 1, 2, 3].map(n => model(n).url);
  await store(cache, model(9)); await saved(model(9), { listed });
  for (const n of [1, 2, 3]) { await store(cache, model(n)); await saved(model(n), { listed, picked: true }); }
  // Neither current nor previous by now: a player who goes back to picking
  // nothing would otherwise fetch its 117 MB again.
  assert.ok(await cache.match(model(9).url));
  for (const n of [1, 2, 3]) assert.ok(await cache.match(model(n).url), `pick ${n}`);
});

test('a record from before picks were remembered has none, and one that cannot be read deletes nothing', async t => {
  const caches = cacheStorageFixture(t), cache = await caches.open(networkCacheName(scope));
  const key = new URL('__network_retention__', scope).href, listed = [4, 1, 2, 3].map(n => model(n).url);
  for (const n of [1, 2, 3, 4]) await store(cache, model(n));
  // As the worker before this one wrote it: every listed model was kept then.
  await cache.put(key, new Response(JSON.stringify({ current: model(3).url, previous: model(2).url })));
  assert.equal(await saved(model(4), { listed }), true);
  assert.deepEqual((await cache.keys()).map(k => k.url).filter(url => url.startsWith('https://models.invalid/')).sort(),
    [model(3).url, model(4).url]);
  assert.deepEqual((await (await cache.match(key)).json()).picked, []);
  for (const picked of ['all', [7], { 0: model(1).url }]) {
    await store(cache, model(1));
    await cache.put(key, new Response(JSON.stringify({ current: model(4).url, previous: model(3).url, picked })));
    await store(cache, model(5));
    assert.equal(await saved(model(5), { listed: [model(5).url] }), false, JSON.stringify(picked));
    assert.ok(await cache.match(model(1).url));
  }
});

test('at activation nothing the build lists goes: the page about to load may play any of it', async t => {
  const caches = cacheStorageFixture(t), cache = await caches.open(networkCacheName(scope));
  const key = new URL('__network_retention__', scope).href, listed = [3, 2, 1].map(n => model(n).url);
  const held = async () => (await cache.keys()).map(k => k.url).filter(url => url.startsWith('https://models.invalid/')).sort();
  for (const n of [0, 1, 2, 3]) await store(cache, model(n));
  // A player picked 1 when it was already saved: the page saved nothing, the
  // worker was told nothing, and the record of the worker before this one
  // names the default and the one before it.
  await cache.put(key, new Response(JSON.stringify({ current: model(3).url, previous: model(2).url })));
  // The new worker activates, perhaps on a plane. What the build no longer
  // lists goes, as it always did; 1, which the page then plays, stays.
  assert.equal(await prune(model(3), { listed }), true);
  assert.deepEqual(await held(), [1, 2, 3].map(n => model(n).url));
  // A page that saves a network names its pick, and only then is a listed
  // network nobody picked let go.
  await store(cache, model(4));
  assert.equal(await saved(model(4), { listed: [model(4).url, ...listed] }), true);
  assert.deepEqual(await held(), [3, 4].map(n => model(n).url));
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

test('a stored-network check opens only its own installation\'s cache, never the old shared one', async t => {
  const caches = cacheStorageFixture(t), m = model(1), opened = [];
  // A copy there could only be generation 36's, which no build plays, and
  // opening that cache to look would create it on every miss.
  await store(await caches.open(NETWORK_CACHE), m);
  const open = caches.open;
  caches.open = async name => { opened.push(name); return open(name); };
  assert.equal(await hasStoredNetwork(m.url, m.expect, { scope }), false);
  assert.deepEqual(opened, [networkCacheName(scope)]);
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
