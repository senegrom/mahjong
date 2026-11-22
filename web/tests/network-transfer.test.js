import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { setImmediate } from 'node:timers/promises';
import v8 from 'node:v8';
import vm from 'node:vm';
import { TURN_MS, WAITING_MS, hasStoredNetwork, networkCacheName, storedNetworkBytes, verifiedNetworkBytes } from '../src/lib/network-transfer.js';
import { verifiedDownload } from '../src/lib/network-cache.js';

/** Runs the garbage collector now, so that a test can ask what is still
 * held. Node exposes it to a context made after the flag is set, which needs
 * no flag on the test runner's own command line. */
const collect = (() => { v8.setFlagsFromString('--expose-gc'); return vm.runInNewContext('gc'); })();

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
  return { fetches: () => fetches, deleted: () => deleted, held: () => held?.bytes, headers: () => held?.headers, reads: () => reads,
    heal: () => { fault = undefined; } };
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
    assert.deepEqual(await verifiedNetworkBytes({ url, expect }), good);
    assert.equal(f.fetches(), 1);
    if (fault === 'put' || fault === 'open') {
      assert.equal(await hasStoredNetwork(url, expect), false);
      await assert.rejects(verifiedNetworkBytes({ url, expect, requireStored: true }), { name: 'NetworkStorageError' });
    }
  });
});

test('a stored network that could not be read once stays for the next try, and a download replaces it', async t => {
  for (const fault of ['body', 'match']) await t.test(fault, async t => {
    // Offline, the read fails, for want of memory say: the move fails, and the network is kept.
    const f = fixture(t, { cached: good, headers: record, fault, network: () => { throw new Error('offline'); } });
    await assert.rejects(verifiedNetworkBytes({ url, expect }), /offline/);
    assert.equal(f.deleted(), 0);
    assert.deepEqual(f.held(), good);
    // The next try reads it.
    f.heal();
    assert.equal(await hasStoredNetwork(url, expect), true);
    assert.deepEqual(await verifiedNetworkBytes({ url, expect }), good);
  });
  // Online, the download that follows an unreadable entry writes over it.
  const online = fixture(t, { cached: good.slice(1), fault: 'body' });
  assert.deepEqual(await verifiedNetworkBytes({ url, expect }), good);
  assert.equal(online.fetches(), 1);
  assert.deepEqual(online.held(), good);
  online.heal();
  assert.equal(await hasStoredNetwork(url, expect), true);
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

test('a progressing download can outlast its idle deadline', async t => {
  let timer, index = 0;
  fixture(t, { network: () => new Response(new ReadableStream({
    start(c) { timer = setInterval(() => { c.enqueue(good.slice(index, ++index)); if (index === 4) { clearInterval(timer); c.close(); } }, 10); },
    cancel() { clearInterval(timer); },
  })) });
  assert.deepEqual(await verifiedNetworkBytes({ url, expect, timeouts: { idleMs: 30, totalMs: 500 } }), good);
});

test('storage that refuses a verified download hands its bytes to the caller, so nothing is fetched twice', async t => {
  const f = fixture(t, { fault: 'put' });
  const refused = await verifiedNetworkBytes({ url, expect, requireStored: true }).then(() => null, error => error);
  assert.equal(refused?.name, 'NetworkStorageError');
  assert.deepEqual(refused.bytes, good, 'the verified bytes travel with the refusal');
  assert.equal(f.fetches(), 1);
  assert.equal(await hasStoredNetwork(url, expect), false, 'and nothing claims they were stored');
});

test('the graphics card\'s reader never downloads, and deletes only what it read and found to be other bytes', async t => {
  // The stored network, verified.
  const stored = fixture(t, { cached: good, headers: record });
  assert.deepEqual(await storedNetworkBytes({ url, expect }), good);
  // None stored: none given, and none fetched.
  const none = fixture(t);
  assert.equal(await storedNetworkBytes({ url, expect }), null);
  assert.equal(none.fetches(), 0);
  // Other bytes under its name are no network, and go.
  const wrong = fixture(t, { cached: Uint8Array.from([4, 3, 2, 1]), headers: record });
  assert.equal(await storedNetworkBytes({ url, expect }), null);
  assert.equal(wrong.deleted(), 1);
  assert.equal(wrong.fetches(), 0);
  // One it could not read, for want of memory or for a fault of the storage, stays: offline play depends on it,
  // and the page's own reader may read it yet.
  for (const fault of ['body', 'match']) {
    const unread = fixture(t, { cached: good, headers: record, fault });
    await assert.rejects(storedNetworkBytes({ url, expect }), fault === 'body' ? /unreadable body/ : /storage lookup failed/);
    assert.equal(unread.deleted(), 0, fault);
    assert.deepEqual(unread.held(), good, fault);
    assert.equal(unread.fetches(), 0, fault);
  }
  assert.equal(stored.fetches(), 0);
});

test('a download holds no chunk once it is copied, however many arrive', async t => {
  // 64 chunks of 64 KB. Raced against one promise kept for the whole download,
  // every one of them stayed alive until the download ended, beside the bytes
  // they had been copied into: for a network, 117 MB twice over.
  const size = 65536, count = 64;
  const whole = new Uint8Array(size * count);
  const identity = { bytes: whole.length, sha256: createHash('sha256').update(whole).digest('hex') };
  const chunks = [];
  const body = new ReadableStream({ pull(controller) {
    if (chunks.length === count) { controller.close(); return; }
    const chunk = new Uint8Array(size);
    chunks.push(new WeakRef(chunk));
    controller.enqueue(chunk);
  } }, { highWaterMark: 0 });
  // The transport's answer as the download reads it, its body the stream itself.
  t.mock.method(globalThis, 'fetch', async () => ({ ok: true, status: 200, headers: new Headers(), body }));
  // Counted as the digest begins: every chunk has been read and copied, and
  // the download has not ended. A turn of the loop first, since a weak
  // reference holds its object until the turn that made it is over.
  let alive = null;
  const digest = crypto.subtle.digest.bind(crypto.subtle);
  t.mock.method(crypto.subtle, 'digest', async (...given) => {
    await setImmediate();
    collect();
    alive = chunks.filter(chunk => chunk.deref()).length;
    return digest(...given);
  });
  const { bytes } = await verifiedDownload('https://cdn.invalid/models/g1/chunks', identity,
    { idleMs: 5000, describe: kind => kind });
  assert.equal(chunks.length, count);
  assert.equal(bytes.length, whole.length);
  // The loop's last turn may still hold the chunk it read.
  assert.ok(alive <= 2, `${alive} of ${count} chunks were still held while the digest ran`);
});

test('a deadline ends a wait that the transport never does, at once and on every later wait', async t => {
  // A fetch that ignores its signal: the download's own promise ends the wait.
  t.mock.method(globalThis, 'fetch', () => new Promise(() => {}));
  await assert.rejects(verifiedDownload('https://cdn.invalid/models/g1/never', expect, { idleMs: 15, describe: kind => kind }), /timeout/);
  // A body that stops after its first chunk, and ignores cancellation too.
  let sent = false;
  const reader = { cancel: () => new Promise(() => {}),
    read: () => (sent ? new Promise(() => {}) : (sent = true, Promise.resolve({ done: false, value: good.slice(0, 1) }))) };
  t.mock.method(globalThis, 'fetch', async () => ({ ok: true, status: 200, headers: new Headers(), body: { getReader: () => reader } }));
  await assert.rejects(verifiedDownload('https://cdn.invalid/models/g1/stalled', expect, { idleMs: 15, totalMs: 1000, describe: kind => kind }), /timeout/);
});

/** A browser's Web Locks, as verifiedNetworkBytes asks them: one holder a
 * name at a time, the others in the order they asked, and a wait that its
 * signal ends. Installed on this process's navigator for one test. */
function webLocks(t, { refuses = false } = {}) {
  const tails = new Map(), asked = [];
  const locks = { asked, request(name, options, task = options) {
    asked.push(name);
    if (refuses) return Promise.reject(new DOMException('Locks are not allowed here', 'SecurityError'));
    const before = tails.get(name) ?? Promise.resolve();
    const granted = new Promise((resolve, reject) => {
      before.then(resolve);
      options.signal?.addEventListener('abort', () => reject(new DOMException('The wait was ended', 'AbortError')));
    });
    const held = granted.then(() => task({ name }));
    tails.set(name, Promise.allSettled([before, held]));
    return held;
  } };
  Object.defineProperty(globalThis.navigator, 'locks', { configurable: true, value: locks });
  t.after(() => { delete globalThis.navigator.locks; });
  return locks;
}
/** No Web Locks at all, as in a browser without them. */
function noWebLocks(t) {
  Object.defineProperty(globalThis.navigator, 'locks', { configurable: true, value: undefined });
  t.after(() => { delete globalThis.navigator.locks; });
}
// A transfer that takes a turn of the loop, as any real one takes far longer.
const slowly = async () => { await setImmediate(); return new Response(good); };

test('two that ask for one network at once fetch it once: the second waits its turn and reads what the first stored', async t => {
  const locks = webLocks(t), f = fixture(t, { network: slowly });
  const [first, second] = await Promise.all([verifiedNetworkBytes({ url, expect }), verifiedNetworkBytes({ url, expect, requireStored: true })]);
  assert.deepEqual([first, second], [good, good]);
  assert.equal(f.fetches(), 1);
  // One lock for the network, by the cache it is stored in and its address.
  assert.deepEqual(locks.asked, Array(2).fill(`${networkCacheName()}:${url}`));
  // Another network is not kept waiting behind it.
  const other = 'https://cdn.invalid/models/g2/' + expect.sha256;
  await Promise.all([verifiedNetworkBytes({ url: other, expect }), verifiedNetworkBytes({ url, expect })]);
  assert.equal(new Set(locks.asked).size, 2);
});

test('without Web Locks, or with a lock the browser refuses, a network is fetched as it always was', async t => {
  await t.test('no Web Locks', async t => {
    noWebLocks(t);
    const f = fixture(t, { network: slowly });
    assert.deepEqual(await Promise.all([verifiedNetworkBytes({ url, expect }), verifiedNetworkBytes({ url, expect })]), [good, good]);
    assert.equal(f.fetches(), 2, 'each for itself, as before');
  });
  await t.test('a lock refused', async t => {
    const locks = webLocks(t, { refuses: true }), f = fixture(t);
    assert.deepEqual(await verifiedNetworkBytes({ url, expect }), good);
    assert.deepEqual([locks.asked.length, f.fetches()], [1, 1]);
  });
});

test('a download that fails under the lock fails once, and lets the next one in', async t => {
  webLocks(t);
  let broken = true;
  const f = fixture(t, { network: () => (broken ? new Response('', { status: 503 }) : new Response(good)) });
  await assert.rejects(verifiedNetworkBytes({ url, expect }), /could not be fetched: 503/);
  assert.equal(f.fetches(), 1, 'its own failure is not a refused lock: it is not run again');
  broken = false;
  assert.deepEqual(await verifiedNetworkBytes({ url, expect }), good);
});

test('one that waits its turn says so, and goes ahead after three minutes whoever still holds the lock', async t => {
  const locks = webLocks(t), f = fixture(t);
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  // Another window is at the network and never finishes: suspended, say.
  void locks.request(`${networkCacheName()}:${url}`, () => new Promise(() => {}));
  let waits = 0;
  const asked = verifiedNetworkBytes({ url, expect, onWaiting: () => { waits++; } });
  for (let turn = 0; turn < 5; turn++) await setImmediate();
  assert.deepEqual([waits, f.fetches()], [0, 0], 'a short wait is not said');
  t.mock.timers.tick(WAITING_MS);
  t.mock.timers.tick(WAITING_MS);
  assert.deepEqual([waits, f.fetches()], [2, 0]);
  t.mock.timers.tick(TURN_MS - 2 * WAITING_MS);
  assert.deepEqual(await asked, good);
  assert.equal(f.fetches(), 1);
  const said = waits;
  t.mock.timers.tick(TURN_MS);
  assert.equal(waits, said, 'and says no more once it has gone ahead');
});
