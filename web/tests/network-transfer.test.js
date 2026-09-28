import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { hasStoredNetwork, verifiedNetworkBytes } from '../src/lib/network-transfer.js';

const good = Uint8Array.from([1, 2, 3, 4]);
const expect = { bytes: good.length, sha256: createHash('sha256').update(good).digest('hex') };
const url = 'https://cdn.invalid/models/g1/' + expect.sha256;
// What a verified write records beside the bytes, as Cache Storage keeps it.
const record = { 'Content-Length': String(expect.bytes), 'X-Mahjong-SHA256': expect.sha256 };
function fixture(t, { cached, headers = {}, fault, network } = {}) {
  let held = cached == null ? undefined : { bytes: cached, headers }, fetches = 0, deleted = 0, reads = 0;
  t.mock.method(globalThis, 'fetch', async (...args) => {
    fetches++; return network ? network(...args) : new Response(good);
  });
  const previous = globalThis.caches;
  globalThis.caches = { async open() {
    if (fault === 'open') throw new Error('storage open failed');
    return {
      async match() {
        if (fault === 'match') throw new Error('storage lookup failed');
        if (fault === 'body') return { async arrayBuffer() { throw new Error('unreadable body'); } };
        if (held == null) return undefined;
        const body = held.bytes;
        // Counts reads of the body, which a status check must never make.
        // No high-water mark: the stream pulls only when someone reads.
        return new Response(new ReadableStream({ pull(controller) { reads++; controller.enqueue(body); controller.close(); } },
          { highWaterMark: 0 }), { headers: held.headers });
      },
      async put(_key, response) {
        if (fault === 'put') throw new DOMException('full', 'QuotaExceededError');
        held = { bytes: new Uint8Array(await response.arrayBuffer()), headers: Object.fromEntries(response.headers) };
      },
      async delete() {
        if (fault === 'delete') throw new Error('cannot delete');
        deleted++; held = undefined;
      },
    };
  } };
  t.after(() => { if (previous === undefined) delete globalThis.caches; else globalThis.caches = previous; });
  return { fetches: () => fetches, deleted: () => deleted, held: () => held?.bytes, headers: () => held?.headers, reads: () => reads };
}

test('bytes for the runtime must have the right size AND digest; the stored record is written with them', async t => {
  const f = fixture(t, { cached: Uint8Array.from([4, 3, 2, 1]) });
  assert.equal(await hasStoredNetwork(url, expect), false, 'an entry without its record is not stored');
  assert.deepEqual(await verifiedNetworkBytes({ url, expect }), good);
  assert.equal(f.deleted(), 1, 'wrong bytes are discarded before a fetch');
  assert.equal(f.fetches(), 1);
  assert.deepEqual(f.headers(), { 'content-length': '4', 'content-type': 'application/octet-stream', 'x-mahjong-sha256': expect.sha256 });
  assert.equal(await hasStoredNetwork(url, expect), true);
  assert.deepEqual(await verifiedNetworkBytes({ url, expect }), good);
  assert.equal(f.fetches(), 1);
});

test('a stored-network check reads the record, never the 116 MB body', async t => {
  const f = fixture(t, { cached: good, headers: record });
  for (let i = 0; i < 3; i++) assert.equal(await hasStoredNetwork(url, expect), true);
  assert.equal(f.reads(), 0);
  for (const wrong of [{ ...expect, bytes: 5 }, { ...expect, sha256: '0'.repeat(64) }]) {
    assert.equal(await hasStoredNetwork(url, wrong), false);
  }
  assert.equal(f.reads(), 0);
  // Only what is handed to the runtime is read and hashed.
  assert.deepEqual(await verifiedNetworkBytes({ url, expect }), good);
  assert.equal(f.reads(), 1);
});

test('a record over the wrong bytes never reaches the runtime', async t => {
  const f = fixture(t, { cached: Uint8Array.from([4, 3, 2, 1]), headers: record, network: () => { throw new Error('offline'); } });
  assert.equal(await hasStoredNetwork(url, expect), true, 'status trusts the record of a verified write');
  await assert.rejects(verifiedNetworkBytes({ url, expect }), /offline/);
  assert.equal(f.deleted(), 1);
  assert.equal(await hasStoredNetwork(url, expect), false);
});

test('corrupt or truncated cache never runs offline', async t => {
  for (const cached of [Uint8Array.from([4, 3, 2, 1]), good.slice(1)]) await t.test(String(cached), async t => {
    fixture(t, { cached, network: () => { throw new Error('offline'); } });
    await assert.rejects(verifiedNetworkBytes({ url, expect }), /offline/);
    assert.equal(await hasStoredNetwork(url, expect), false);
  });
});

test('all cache failures allow verified online bytes, but never claim durability', async t => {
  for (const fault of ['open', 'match', 'body', 'delete', 'put']) await t.test(fault, async t => {
    const f = fixture(t, { fault, cached: good.slice(1) });
    let stored;
    assert.deepEqual(await verifiedNetworkBytes({ url, expect, onStorage: value => stored = value }), good);
    assert.equal(f.fetches(), 1);
    if (fault === 'put' || fault === 'open') {
      assert.equal(stored, false);
      await assert.rejects(verifiedNetworkBytes({ url, expect, requireStored: true }), { name: 'NetworkStorageError' });
    }
  });
});

test('downloads with mismatched bytes never enter the cache', async t => {
  const f = fixture(t, { network: () => new Response(Uint8Array.from([4, 3, 2, 1])) });
  await assert.rejects(verifiedNetworkBytes({ url, expect }), /not the one/);
  assert.equal(f.held(), undefined);
});

test('stalled headers and bodies abort, release the operation and permit retry', async t => {
  for (const headers of [true, false]) await t.test(`headers=${headers}`, async t => {
    let stalled = true, requestSignal, cancelled = false;
    const f = fixture(t, { network: (_url, options) => {
      requestSignal = options.signal;
      if (!stalled) return new Response(good);
      if (headers) return new Promise(() => {});
      return new Response(new ReadableStream({ pull() {}, cancel() { cancelled = true; } }));
    } });
    await assert.rejects(verifiedNetworkBytes({ url, expect, timeouts: { idleMs: 15, totalMs: 100 } }), /timed out/);
    assert.equal(requestSignal.aborted, true);
    if (!headers) assert.equal(cancelled, true);
    stalled = false;
    assert.deepEqual(await verifiedNetworkBytes({ url, expect }), good);
    assert.equal(f.fetches(), 2);
  });
});

test('overlong streams are cancelled before accepting or retaining excess bytes', async t => {
  let cancelled = false;
  const f = fixture(t, { network: () => new Response(new ReadableStream({
    start(controller) { controller.enqueue(new Uint8Array(5)); }, cancel() { cancelled = true; },
  })) });
  await assert.rejects(verifiedNetworkBytes({ url, expect }), /exceeded/);
  assert.equal(cancelled, true); assert.equal(f.held(), undefined);
});

test('a progressing download can outlast its idle deadline; subscriber abort still works', async t => {
  let timer, index = 0;
  fixture(t, { network: () => new Response(new ReadableStream({
    start(c) { timer = setInterval(() => { c.enqueue(good.slice(index, ++index)); if (index === 4) { clearInterval(timer); c.close(); } }, 10); },
    cancel() { clearInterval(timer); },
  })) });
  assert.deepEqual(await verifiedNetworkBytes({ url, expect, timeouts: { idleMs: 30, totalMs: 500 } }), good);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(verifiedNetworkBytes({ url, expect, signal: controller.signal }), { name: 'AbortError' });
});

test('storage that refuses a verified download hands its bytes to the caller, so nothing is fetched twice', async t => {
  const f = fixture(t, { fault: 'put' });
  const refused = await verifiedNetworkBytes({ url, expect, requireStored: true }).then(() => null, error => error);
  assert.equal(refused?.name, 'NetworkStorageError');
  assert.deepEqual(refused.bytes, good, 'the verified bytes travel with the refusal');
  assert.equal(f.fetches(), 1);
  assert.equal(await hasStoredNetwork(url, expect), false, 'and nothing claims they were stored');
});
