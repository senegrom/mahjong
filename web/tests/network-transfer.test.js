import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { setImmediate } from 'node:timers/promises';
import v8 from 'node:v8';
import vm from 'node:vm';
import { SILENT_MS, TURN_MS, WAITING_MS, hasStoredNetwork, heldBytes, networkCacheName, storedNetworkBytes, validateNetwork,
  verifiedNetworkBytes } from '../src/lib/network-transfer.js';
import { RESUME, verifiedDownload } from '../src/lib/network-cache.js';

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
  // Twelve bytes, one every 40 ms: half a second in all against a 200 ms idle
  // deadline, a gap five times shorter than the deadline, so that a loaded
  // machine's late timer does not end it.
  const long = Uint8Array.from({ length: 12 }, (_, i) => i + 1);
  const identity = { bytes: long.length, sha256: createHash('sha256').update(long).digest('hex') };
  let timer, index = 0;
  fixture(t, { network: () => new Response(new ReadableStream({
    start(c) {
      timer = setInterval(() => { c.enqueue(long.slice(index, ++index)); if (index === long.length) { clearInterval(timer); c.close(); } }, 40);
    },
    cancel() { clearInterval(timer); },
  })) });
  assert.deepEqual(await verifiedNetworkBytes({ url, expect: identity, timeouts: { idleMs: 200, totalMs: 5000 } }), long);
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
  // The loop's last turn may still hold the chunk it read. Counted at all
  // first: a digest that never ran here would measure nothing.
  assert.equal(typeof alive, 'number', 'the digest was never asked, so nothing was counted');
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
 * name at a time, the others in the order they asked, a wait that its
 * signal ends, and `ifAvailable`, answered with no lock where another holds
 * the name or waits for it. Installed on this process's navigator for one
 * test. */
function webLocks(t, { refuses = false } = {}) {
  const tails = new Map(), pending = new Map(), asked = [];
  const locks = { asked, request(name, options, task = options) {
    asked.push(name);
    if (refuses) return Promise.reject(new DOMException('Locks are not allowed here', 'SecurityError'));
    if (options.ifAvailable && pending.get(name)) return (async () => task(null))();
    pending.set(name, (pending.get(name) ?? 0) + 1);
    const before = tails.get(name) ?? Promise.resolve();
    const granted = new Promise((resolve, reject) => {
      before.then(resolve);
      options.signal?.addEventListener('abort', () => reject(new DOMException('The wait was ended', 'AbortError')));
    });
    const held = granted.then(() => task({ name }));
    const done = () => { pending.set(name, pending.get(name) - 1); };
    held.then(done, done);
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
/** BroadcastChannel as the holder and the one waiting use it, delivering at
 * once to every other open channel of the same name, so that a test with
 * mocked timers controls when the holder is heard. `open` is every channel
 * not yet closed. Without `present`, the browser has none. */
function channels(t, { present = true } = {}) {
  const open = new Set();
  class Channel {
    constructor(name) { this.name = name; this.onmessage = null; this.sent = 0; open.add(this); }
    postMessage(data) {
      if (!open.has(this)) throw new DOMException('closed', 'InvalidStateError');
      this.sent++;
      for (const other of open) if (other !== this && other.name === this.name) other.onmessage?.({ data });
    }
    close() { open.delete(this); }
  }
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'BroadcastChannel');
  Object.defineProperty(globalThis, 'BroadcastChannel', { configurable: true, writable: true, value: present ? Channel : undefined });
  t.after(() => { if (previous) Object.defineProperty(globalThis, 'BroadcastChannel', previous); else delete globalThis.BroadcastChannel; });
  // What another window holding the lock says, and what is still listening.
  return { open, holder: name => new Channel(name), listening: () => [...open].filter(channel => channel.onmessage).length };
}
// A transfer that takes a turn of the loop, as any real one takes far longer.
const slowly = async () => { await setImmediate(); return new Response(good); };

test('two that ask for one network at once fetch it once: the second waits its turn and reads what the first stored', async t => {
  const locks = webLocks(t), f = fixture(t, { network: slowly });
  const [first, second] = await Promise.all([verifiedNetworkBytes({ url, expect }), verifiedNetworkBytes({ url, expect, requireStored: true })]);
  assert.deepEqual([first, second], [good, good]);
  assert.equal(f.fetches(), 1);
  // One lock for the network, by the cache it is stored in and its address:
  // taken at once by the first, and asked for twice by the second, which
  // found it held and waited.
  assert.deepEqual(locks.asked, Array(3).fill(`${networkCacheName()}:${url}`));
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

const lockName = `${networkCacheName()}:${url}`;
const turns = async () => { for (let turn = 0; turn < 5; turn++) await setImmediate(); };

test('without BroadcastChannel, one that waits its turn says so, and goes ahead after three minutes whoever holds the lock', async t => {
  const locks = webLocks(t), f = fixture(t);
  channels(t, { present: false });
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  // Another window is at the network and never finishes: suspended, say.
  void locks.request(lockName, () => new Promise(() => {}));
  let waits = 0;
  const asked = verifiedNetworkBytes({ url, expect, onWaiting: () => { waits++; } });
  await turns();
  assert.deepEqual([waits, f.fetches()], [1, 0], 'the wait is said as it begins');
  t.mock.timers.tick(WAITING_MS);
  t.mock.timers.tick(WAITING_MS);
  assert.deepEqual([waits, f.fetches()], [3, 0]);
  t.mock.timers.tick(TURN_MS - 2 * WAITING_MS - 1);
  await turns();
  assert.equal(f.fetches(), 0);
  t.mock.timers.tick(1);
  assert.deepEqual(await asked, good);
  assert.equal(f.fetches(), 1);
  const said = waits;
  t.mock.timers.tick(TURN_MS);
  assert.equal(waits, said, 'and says no more once it has gone ahead');
});

test('one that finds the network free takes it at once and says no wait', async t => {
  webLocks(t); channels(t);
  const f = fixture(t);
  let waits = 0;
  assert.deepEqual(await verifiedNetworkBytes({ url, expect, onWaiting: () => { waits++; } }), good);
  assert.deepEqual([waits, f.fetches()], [0, 1]);
});

test('a holder still receiving bytes is waited for past three minutes, and its wait hears it no more once over', async t => {
  const locks = webLocks(t), c = channels(t), f = fixture(t);
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  // Another window is at the network on a slow link: it says it is alive as
  // its bytes arrive, every second, here every ten.
  let finish;
  void locks.request(lockName, () => new Promise(done => { finish = done; }));
  const holder = c.holder(lockName);
  const asked = verifiedNetworkBytes({ url, expect, requireStored: true });
  await turns();
  assert.equal(c.listening(), 1, 'the wait listens for its holder');
  for (let elapsed = 0; elapsed < 2 * TURN_MS; elapsed += 10000) {
    holder.postMessage('alive');
    t.mock.timers.tick(10000);
  }
  await turns();
  assert.equal(f.fetches(), 0, 'six minutes in, the holder is still downloading, and nobody else is');
  // The holder lets go, and the wait has its turn.
  finish();
  assert.deepEqual(await asked, good);
  assert.equal(f.fetches(), 1);
  assert.equal(c.listening(), 0, 'no listener outlives its wait');
});

test('a holder that falls silent is given up on once it has said nothing for as long as a stalled download', async t => {
  const locks = webLocks(t), c = channels(t), f = fixture(t);
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  void locks.request(lockName, () => new Promise(() => {}));
  const holder = c.holder(lockName);
  const asked = verifiedNetworkBytes({ url, expect });
  await turns();
  // Alive for a minute, then suspended with the browser's window.
  for (let second = 0; second < 60; second++) { holder.postMessage('alive'); t.mock.timers.tick(1000); }
  holder.postMessage('alive');
  t.mock.timers.tick(SILENT_MS - 1);
  await turns();
  assert.equal(f.fetches(), 0, 'not before the silence limit');
  assert.ok(SILENT_MS < TURN_MS, 'sooner than the three minutes a holder used to be given');
  t.mock.timers.tick(1);
  assert.deepEqual(await asked, good);
  assert.equal(f.fetches(), 1);
  assert.equal(c.listening(), 0, 'no listener outlives its wait');
});

test('the holder says it is alive as it takes the lock and as bytes arrive, at most once a second, and then closes its channel', async t => {
  webLocks(t);
  const c = channels(t);
  t.mock.timers.enable({ apis: ['Date'] });
  // One that waits for it, as another window would.
  const listener = c.holder(lockName);
  let heard = 0;
  listener.onmessage = () => { heard++; };
  // A byte every 600 ms: heard as it takes the lock, and at 1.2 and 2.4 s.
  fixture(t, { network: () => new Response(new ReadableStream({ start() { this.index = 0; }, pull(controller) {
    t.mock.timers.tick(600);
    controller.enqueue(good.slice(this.index, ++this.index));
    if (this.index === good.length) controller.close();
  } }, { highWaterMark: 0 })) });
  assert.deepEqual(await verifiedNetworkBytes({ url, expect }), good);
  assert.equal(heard, 3);
  assert.deepEqual([...c.open], [listener], 'the holder\'s channel is closed');
});

test('a save that needs no bytes finds the network the other window stored by its record, and reads none of it', async t => {
  const locks = webLocks(t), f = fixture(t, { network: slowly });
  channels(t);
  const [first, second] = await Promise.all([verifiedNetworkBytes({ url, expect }),
    verifiedNetworkBytes({ url, expect, requireStored: true, storeOnly: true })]);
  assert.deepEqual(first, good);
  assert.equal(second, null);
  assert.equal(locks.asked.length, 3, 'the second waited its turn');
  assert.deepEqual([f.fetches(), f.reads()], [1, 0]);
});

// A packed network is stored by its host without an encoding, so a transfer
// that breaks is asked again for the rest. Its identity names the packed
// file's size and digest (storedBytes, sha256) beside the whole file's bytes.
const packed = Uint8Array.from({ length: 16 }, (_, i) => 10 + i);
const half = { bytes: 100, storedBytes: packed.length, packing: 'half', sha256: createHash('sha256').update(packed).digest('hex') };
const halfUrl = 'https://cdn.invalid/models/g2/' + half.sha256;
const quick = { pauseMs: 1, idleMs: 5000, totalMs: 20000 };
/** A body that gives these bytes and then fails as a dropped connection does. */
const breaking = (chunk, error = new TypeError('network error')) => new ReadableStream({
  start(controller) { controller.enqueue(chunk); }, pull(controller) { controller.error(error); } });
const part = (from, to, body = packed.slice(from, to)) => new Response(body,
  { status: 206, headers: { 'Content-Range': `bytes ${from}-${packed.length - 1}/${packed.length}` } });
/** The host, one answer a request in turn; `asked` is the Range of each. */
function host(t, answers, options) {
  const asked = [];
  const f = fixture(t, { ...options, network: (_url, init) => {
    const range = init.headers?.get('Range') ?? null;
    asked.push(range);
    return answers[asked.length - 1](range);
  } });
  return { ...f, asked };
}
const progressOf = () => {
  const seen = [];
  return { seen, onProgress: ({ bytes, total }) => { assert.equal(total, packed.length); seen.push(bytes); } };
};
const neverBack = seen => seen.every((bytes, at) => !at || bytes >= seen[at - 1]);

test('a packed network that breaks at several points is asked for the rest each time, and its whole digest decides', async t => {
  const h = host(t, [
    () => new Response(breaking(packed.slice(0, 5))),
    () => new Response(breaking(packed.slice(5, 9)), { status: 206, headers: { 'Content-Range': `bytes 5-15/16` } }),
    () => part(9, 16),
  ]);
  const progress = progressOf();
  assert.deepEqual(await verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: quick, requireStored: true, ...progress }), packed);
  assert.deepEqual(h.asked, [null, 'bytes=5-', 'bytes=9-']);
  assert.ok(neverBack(progress.seen), progress.seen.join());
  assert.equal(progress.seen.at(-1), packed.length);
  // Stored as the packed file it is, under the packed file's record.
  assert.deepEqual(h.held(), packed);
  assert.equal(h.headers()['content-length'], String(packed.length));
  assert.equal(await hasStoredNetwork(halfUrl, half), true);
});

test('a host that answers a range with the whole object again is read from its first byte, and progress holds', async t => {
  const h = host(t, [() => new Response(breaking(packed.slice(0, 8))), () => new Response(packed)]);
  const progress = progressOf();
  assert.deepEqual(await verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: quick, ...progress }), packed);
  assert.deepEqual(h.asked, [null, 'bytes=8-']);
  assert.ok(neverBack(progress.seen), progress.seen.join());
});

test('a 206 that starts after the bytes in hand is let go and the whole asked for; one that starts before them is used', async t => {
  await t.test('later', async t => {
    const h = host(t, [() => new Response(breaking(packed.slice(0, 5))), () => part(8, 16), () => new Response(packed)]);
    assert.deepEqual(await verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: quick }), packed);
    assert.deepEqual(h.asked, [null, 'bytes=5-', null]);
  });
  await t.test('earlier', async t => {
    const h = host(t, [() => new Response(breaking(packed.slice(0, 5))), () => part(3, 16)]);
    const progress = progressOf();
    assert.deepEqual(await verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: quick, ...progress }), packed);
    assert.deepEqual(h.asked, [null, 'bytes=5-']);
    assert.ok(neverBack(progress.seen), progress.seen.join());
  });
  await t.test('of another size, or unreadable', async t => {
    const other = () => new Response(packed.slice(5), { status: 206, headers: { 'Content-Range': 'bytes 5-15/99' } });
    const unreadable = () => new Response(packed.slice(5), { status: 206 });
    const h = host(t, [() => new Response(breaking(packed.slice(0, 5))), other, unreadable, () => new Response(packed)]);
    assert.deepEqual(await verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: quick }), packed);
    assert.deepEqual(h.asked, [null, 'bytes=5-', null, null]);
  });
});

test('a resumed transfer whose whole is other bytes fails on its digest, is not asked again, and is never stored', async t => {
  const wrong = Uint8Array.from(packed, byte => byte ^ 1);
  const h = host(t, [() => new Response(breaking(packed.slice(0, 6))), () => part(6, 16, wrong.slice(6))]);
  await assert.rejects(verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: quick }), /not the one/);
  assert.equal(h.fetches(), 2);
  assert.equal(h.held(), undefined);
});

test('a stall is resumed from where it stopped, and a transfer that keeps stalling fails as before after its attempts', async t => {
  const stalls = from => new Response(new ReadableStream({ start(controller) { controller.enqueue(packed.slice(from, from + 4)); }, pull() {
    return new Promise(() => {});
  } }), from ? { status: 206, headers: { 'Content-Range': `bytes ${from}-15/16` } } : {});
  const stalled = { ...quick, idleMs: 30 };
  const h = host(t, [() => stalls(0), () => stalls(4), () => part(8, 16)]);
  assert.deepEqual(await verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: stalled }), packed);
  assert.deepEqual(h.asked, [null, 'bytes=4-', 'bytes=8-']);
  // Stalled every time: one first request and RESUME.attempts more, then the
  // failure a stall has always been.
  const always = host(t, Array.from({ length: 10 }, () => () => new Response(new ReadableStream({ pull: () => new Promise(() => {}) }))));
  await assert.rejects(verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: stalled }), /timed out/);
  assert.equal(always.fetches(), RESUME.attempts + 1);
  assert.deepEqual(new Set(always.asked), new Set([null]), 'nothing in hand, nothing to ask a range of');
});

test('a link that breaks more often than the attempts allow goes on while every break brought new bytes', async t => {
  // Two bytes a request, eight requests: seven breaks, more than RESUME.attempts.
  const answers = Array.from({ length: 8 }, (_, index) => () => {
    const from = 2 * index, chunk = packed.slice(from, from + 2);
    const init = from ? { status: 206, headers: { 'Content-Range': `bytes ${from}-15/16` } } : {};
    return new Response(index === 7 ? chunk : breaking(chunk), init);
  });
  assert.ok(answers.length - 1 > RESUME.attempts);
  const h = host(t, answers);
  assert.deepEqual(await verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: quick }), packed);
  assert.equal(h.fetches(), 8);
});

test('a body that ends short is a break too, and the total deadline ends every attempt and pause', async t => {
  const h = host(t, [() => new Response(packed.slice(0, 7)), () => part(7, 16)]);
  assert.deepEqual(await verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: quick }), packed);
  assert.deepEqual(h.asked, [null, 'bytes=7-']);
  const slow = host(t, Array.from({ length: 10 }, () => () => new Response(breaking(packed.slice(0, 2)))));
  await assert.rejects(verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: { pauseMs: 200, idleMs: 5000, totalMs: 60 } }), /timed out/);
  assert.equal(slow.fetches(), 1, 'the deadline fell in the pause, and nothing was asked after it');
});

test('a failure that is no break is not asked again: a refusal, an overrun, a host that cannot be reached', async t => {
  for (const [answer, reason] of [[() => new Response('', { status: 503 }), /503/],
    [() => new Response(new Uint8Array(17)), /exceeded/], [() => { throw new Error('offline'); }, /offline/]]) {
    const h = host(t, [answer, () => new Response(packed)]);
    await assert.rejects(verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: quick }), reason);
    assert.equal(h.fetches(), 1, String(reason));
  }
});

test('a network stored gzipped is never asked for a range, and a break fails it as before', async t => {
  const h = host(t, [() => new Response(breaking(good.slice(0, 2))), () => new Response(good)]);
  await assert.rejects(verifiedNetworkBytes({ url, expect, timeouts: quick }), /network error/);
  assert.deepEqual(h.asked, [null]);
});

test('the holder of the lock goes on saying it is alive through a resumed transfer', async t => {
  webLocks(t);
  const c = channels(t);
  t.mock.timers.enable({ apis: ['Date'] });
  const listener = c.holder(`${networkCacheName()}:${halfUrl}`);
  let heard = 0;
  listener.onmessage = () => { heard++; };
  // Each request comes two seconds after the last.
  const later = answer => () => { t.mock.timers.tick(2000); return answer(); };
  const h = host(t, [later(() => new Response(breaking(packed.slice(0, 4)))), later(() => part(4, 16))]);
  assert.deepEqual(await verifiedNetworkBytes({ url: halfUrl, expect: half, timeouts: quick }), packed);
  assert.equal(h.fetches(), 2);
  assert.ok(heard >= 3, `heard ${heard} times: as it took the lock, and as each request's bytes came`);
});

test('a packed identity is stored, found and read back as the packed file, and an identity that misstates its packing is refused', async t => {
  fixture(t, { cached: packed, headers: { 'Content-Length': String(packed.length), 'X-Mahjong-SHA256': half.sha256 } });
  assert.equal(await hasStoredNetwork(halfUrl, half), true);
  assert.equal(await hasStoredNetwork(halfUrl, { ...half, packing: undefined, storedBytes: undefined, bytes: half.bytes }), false,
    'the whole size is not what is stored');
  assert.deepEqual(await storedNetworkBytes({ url: halfUrl, expect: half }), packed);
  for (const bad of [{ packing: 'quarter' }, { storedBytes: undefined }, { storedBytes: 100 }, { storedBytes: 0 }]) {
    assert.throws(() => validateNetwork({ ...half, ...bad }), /packing/, JSON.stringify(bad));
  }
  assert.equal(heldBytes(half), packed.length);
  assert.equal(heldBytes(expect), expect.bytes);
});
