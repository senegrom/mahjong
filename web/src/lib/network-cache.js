/** The part of the network transfer the service worker uses too, inlined into
 * sw.js at build time (scripts/offline-build.mjs), so it imports nothing: the
 * network's identity and storage errors, its installation's cache and the
 * record a verified write leaves there, retention, and the bounded, verified
 * download behind the worker's own files and the page's network alike. */
export const NETWORK_CACHE = 'mahjong-network-v1';

const size = bytes => Number.isSafeInteger(bytes) && bytes > 0 && bytes <= 512 * 1024 * 1024;

/** An identity is the size and SHA-256 of what is downloaded and stored. A
 * packed network (src/lib/network-pack.js) says so (`packing: 'half'`): its
 * digest and storedBytes are the packed file's, which is what travels and is
 * kept, and its bytes the whole file's, which its worker widens it to. A
 * network without the field, and every file of the service worker's own, is
 * downloaded and kept as it is, `bytes` long. */
export function validateNetwork(expect) {
  if (!expect || !size(expect.bytes) || !/^[0-9a-f]{64}$/.test(expect.sha256)) {
    throw new Error('Invalid trained-network size or SHA-256 identity');
  }
  if (expect.packing !== undefined && (expect.packing !== 'half' || !size(expect.storedBytes) || expect.storedBytes >= expect.bytes)) {
    throw new Error('Invalid trained-network packing');
  }
}

/** How many bytes of this identity are downloaded and stored. */
export const heldBytes = expect => (expect.packing ? expect.storedBytes : expect.bytes);

/** How often a transfer that broke or stalled is asked again for the rest,
 * in a row without a new byte, and how long it pauses before the first such
 * request: each pause in a row is twice the last, 2, 4, 8, 16 and 32 s,
 * about a minute in all, for a phone that moves from wifi to its mobile
 * network or a train in a tunnel. Only for a packed network, which its host
 * stores without an encoding, so that a byte range of it means something. */
export const RESUME = Object.freeze({ attempts: 5, pauseMs: 2000 });

/** bytes, when given, are verified bytes storage refused: usable online by
 * whoever catches this, and never a claim that anything was stored. `what`
 * names what was refused, the network unless the service worker says
 * otherwise of a file of its own: a full device refuses the game's files
 * before it ever comes to the network. */
export function storageError(cause, bytes, what = 'The network') {
  const error = new Error(`${what} is not saved offline: ${cause?.message ?? cause ?? 'storage unavailable'}`);
  error.name = 'NetworkStorageError';
  if (bytes) error.bytes = bytes;
  return error;
}

/** Every page and worker names its installation scope; a caller without one,
 * a script or a test, gets the old shared v1 name. Nothing migrates from that
 * cache any more: it could only ever hold generation 36, which no build since
 * 8 October plays. Its entries have no reliable scope ownership, so nothing
 * here deletes them. */
export function networkCacheName(scope) {
  return scope ? `mahjong-network-v2:${new URL('./', scope).href}` : NETWORK_CACHE;
}

export async function networkCache(scope) {
  try { return typeof caches === 'undefined' ? null : await caches.open(networkCacheName(scope)); }
  catch { return null; }
}

export async function digestOf(bytes) {
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join('');
}

/** Whether an entry records this exact identity. Every writer hashes the bytes
 * before storing them under these headers, as the service worker does for its
 * own files, and the object is content-addressed and immutable. The body is
 * neither read nor hashed: that is done only for bytes handed to the runtime. */
async function recorded(cache, url, expect) {
  if (!cache) return false;
  try {
    const held = await cache.match(url);
    return held?.headers.get('X-Mahjong-SHA256') === expect.sha256
      && Number(held.headers.get('Content-Length')) === heldBytes(expect);
  } catch { return false; }
}

/** Cheap enough for every status check and foreground return. */
export async function hasStoredNetwork(url, expect, { scope } = {}) {
  validateNetwork(expect);
  return recorded(await networkCache(scope), url, expect);
}

/** The bounded transfer behind every download: the network, and the service
 * worker's own files. A deadline that every chunk resets ends a transfer that
 * has stalled, while one that keeps receiving bytes goes on however slowly
 * they come; totalMs, when given, bounds the whole of it as well. The body
 * is read into exactly expect.bytes, so an overlong one is cancelled before
 * its excess is kept, and is returned with its response only when its
 * SHA-256 matches. The deadlines hold even when a mocked or broken transport
 * ignores cancellation. describe(kind, detail) words each failure for the
 * caller: 'status', 'overrun', 'size', 'digest' or 'timeout'.
 *
 * With `resume`, a transfer that has received bytes and then stalls past its
 * idle deadline, whose connection fails (a TypeError, as fetch and its body
 * throw for a network error) or whose body ends short keeps them and is
 * asked again for the rest, `Range: bytes=<received>-`, after a pause of
 * `pauseMs` that doubles with each attempt in a row that brought no new
 * byte, up to `resume` such attempts, before it fails as it would have
 * without; an attempt that brings new bytes starts the count and the pauses
 * over. One that fails before any byte arrived is never asked again: it is
 * said at once, as for a network that cannot resume. One that breaks with
 * every byte in hand asks nothing more, and the digest decides. A 206 is taken
 * from where its Content-Range says it starts, at or before the bytes in
 * hand; one that starts after them, or whose Content-Range cannot be read or
 * names another size, is let go and the next request asks for the whole. A
 * 200 is the whole object again, kept from its first byte. Progress never
 * runs backwards: it says the most ever held, and is said again as each
 * pause begins and as it ends, a sign of life for whoever waits on this
 * transfer: the longest silence is a stall's idle deadline or the longest
 * pause, whichever is longer (60 s against 32 s for a network), never the
 * two end to end. The
 * SHA-256 of the whole decides, as ever, and the total deadline covers every
 * request and pause. Only a packed network is asked so: an object stored
 * gzipped has no byte range that means anything, and its host ignores one. */
export async function verifiedDownload(url, expect, { init, idleMs, totalMs, onProgress, describe, resume = 0, pauseMs = RESUME.pauseMs }) {
  const size = heldBytes(expect);
  // Content-Length can be compressed; the expected size bounds decoded bytes.
  const bytes = new Uint8Array(size);
  let controller = null, idle, total, pause, reader = null, giveUp = null;
  // Why the wait under way ends: a deadline, or nothing yet. `overall` is the
  // total deadline's, which no further request survives.
  let stopped = null, overall = null, seen = 0, shown = 0, last = null;
  const end = error => { stopped = error; controller?.abort(error); giveUp?.(error); };
  const resetIdle = () => {
    clearTimeout(idle);
    idle = setTimeout(() => end(Object.assign(new Error(describe('timeout')), { resumable: true })), idleMs);
  };
  // Each wait has a promise of its own for the deadline to end, let go as the
  // wait ends. Raced against one promise kept for the whole download, every
  // wait stayed reachable from that promise, its answer with it: each chunk
  // read was kept until the download ended, beside the bytes it had been
  // copied into, a network's 117 MB twice over.
  const bounded = promise => new Promise((resolve, reject) => {
    if (stopped) { reject(stopped); return; }
    const ended = settle => value => { if (giveUp === reject) giveUp = null; settle(value); };
    giveUp = reject;
    promise.then(ended(resolve), ended(reject));
  });
  const letGo = () => {
    // Do not await a broken stream's cancellation promise.
    if (reader) { try { void reader.cancel().catch(() => {}); } catch { /* Already closed. */ } }
    reader = null;
    controller?.abort();
  };
  const said = () => onProgress?.({ bytes: shown, total: size });
  /** One request, for the rest where some is in hand. */
  const receive = async () => {
    controller = new AbortController(); stopped = overall;
    resetIdle();
    const request = { ...init, signal: controller.signal };
    if (seen) {
      request.headers = new Headers(init?.headers);
      request.headers.set('Range', `bytes=${seen}-`);
    }
    const response = await bounded(fetch(url, request));
    last = response;
    reader = response.body?.getReader() ?? null;
    if (response.status === 206) {
      const range = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get('Content-Range') ?? '');
      const start = range ? Number(range[1]) : NaN;
      if (!range || Number(range[3]) !== size || !(start <= seen)) {
        // Not the rest of this object: the next request asks for the whole.
        seen = 0;
        throw Object.assign(new Error(describe('status', response.status)), { resumable: true });
      }
      seen = start;
    } else if (!response.ok) throw new Error(describe('status', response.status));
    else seen = 0;
    if (reader) {
      for (;;) {
        const { done, value } = await bounded(reader.read());
        if (done) break;
        if (seen + value.byteLength > size) throw new Error(describe('overrun'));
        bytes.set(value, seen);
        seen += value.byteLength;
        if (value.byteLength) resetIdle();
        if (seen > shown) shown = seen;
        said();
      }
    } else {
      const body = new Uint8Array(await bounded(response.arrayBuffer()));
      if (seen + body.length > size) throw new Error(describe('overrun'));
      bytes.set(body, seen); seen += body.length;
      if (seen > shown) shown = seen;
    }
    if (seen !== size) throw Object.assign(new Error(describe('size', seen)), { resumable: true });
    return response;
  };
  let wake = null;
  if (totalMs) total = setTimeout(() => { overall = new Error(describe('timeout')); end(overall); wake?.(); }, totalMs);
  try {
    let response;
    // `inRow` counts the attempts in a row that brought no byte not held
    // before: one that did starts it, and the pauses, over, so that a link
    // that drops often but keeps advancing goes on.
    for (let inRow = 0, best = 0; ; inRow++) {
      try { response = await receive(); break; } catch (error) {
        letGo(); clearTimeout(idle);
        const broke = !overall && Boolean(error?.resumable || error instanceof TypeError);
        // Broken after its last byte: nothing is left to ask for. The last
        // response's headers, all its caller reads of it, came whole.
        if (broke && seen === size) { response = last; stopped = overall; break; }
        if (shown > best) { best = shown; inRow = 0; }
        if (!broke || !shown || inRow >= resume) throw overall ?? error;
      }
      said();
      await new Promise(done => { wake = done; pause = setTimeout(done, pauseMs * 2 ** inRow); });
      wake = null; clearTimeout(pause);
      if (overall) throw overall;
      said();
    }
    // Every byte is in hand: the idle deadline was the transfer's, and hashing
    // what arrived is no part of that. Only the total deadline still runs.
    clearTimeout(idle);
    if (await bounded(digestOf(bytes)) !== expect.sha256) throw new Error(describe('digest'));
    return { response, bytes };
  } finally {
    clearTimeout(idle); clearTimeout(total); clearTimeout(pause);
    letGo();
  }
}

/** Called by the active worker only: at activation, after the previous worker
 * has released its clients, and after its own page has saved this model.
 * `listed` are the addresses the worker's build lists, its default first.
 *
 * After a page has saved this model (`saved`), keep it, the previous
 * activated model, the build's default and the models the player picked.
 * `picked` says that this one is a pick, and the record remembers each for as
 * long as the build lists it, so a network a player picked is never deleted
 * because another was saved after it, and a player who picks nothing finds
 * the default still there. A former default that nobody picked goes once two
 * newer models are saved, though the site still lists it; it used to stay for
 * as long as it was listed, 117 MB for each default a player had ever had.
 *
 * At activation no page has spoken, and nothing the build lists goes: the
 * page about to load may play any of them, perhaps offline, and a pick made
 * before picks were remembered, of a network already saved, left no trace in
 * the record. Installation, waiting workers and failed replacement downloads
 * never trigger deletion. canPrune is checked again before each delete in
 * case an update starts. */
export async function pruneNetworkCache({ scope, url, expect, listed = [], saved = false, picked = false, canPrune = () => true }) {
  if (!scope || !canPrune()) return false;
  validateNetwork(expect);
  const cache = await networkCache(scope);
  if (!cache) return false;
  const key = new URL('__network_retention__', scope).href;
  try {
    const held = await cache.match(key);
    const old = held ? await held.json() : null;
    // A record written before picks were remembered has none.
    if (old && (typeof old.current !== 'string'
      || (old.previous != null && typeof old.previous !== 'string')
      || (old.picked != null && !(Array.isArray(old.picked) && old.picked.every(pick => typeof pick === 'string'))))) return false;
    if (!await recorded(cache, url, expect)) {
      // Activation precedes the optional download. Remember a first
      // identity so it can become the rollback model on the next upgrade.
      // An existing record never advances to a model that is not stored.
      if (!old && canPrune()) await cache.put(key, new Response(JSON.stringify({ current: url, previous: null })));
      return false;
    }
    const previous = old?.current === url ? old.previous : old?.current;
    const picks = [...new Set([...(old?.picked ?? []), ...(saved && picked ? [url] : [])])].filter(pick => listed.includes(pick));
    const keep = new Set([key, url, previous, ...(saved ? [listed[0], ...picks] : listed)].filter(Boolean));
    if (!canPrune()) return false;
    await cache.put(key, new Response(JSON.stringify({ current: url, previous: previous ?? null, picked: picks }), {
      headers: { 'Content-Type': 'application/json' },
    }));
    for (const entry of await cache.keys()) {
      if (!canPrune()) return false;
      if (!keep.has(entry.url)) await cache.delete(entry);
    }
    return true;
  } catch { return false; } // Cleanup must never prevent a working app activating.
}
