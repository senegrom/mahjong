/** The part of the network transfer the service worker uses too, inlined into
 * sw.js at build time (scripts/offline-build.mjs), so it imports nothing: the
 * network's identity and storage errors, its installation's cache and the
 * record a verified write leaves there, retention, and the bounded, verified
 * download behind the worker's own files and the page's network alike. */
export const NETWORK_CACHE = 'mahjong-network-v1';

export function validateNetwork(expect) {
  if (!expect || !Number.isSafeInteger(expect.bytes) || expect.bytes <= 0
      || expect.bytes > 512 * 1024 * 1024 || !/^[0-9a-f]{64}$/.test(expect.sha256)) {
    throw new Error('Invalid trained-network size or SHA-256 identity');
  }
}

/** bytes, when given, are verified bytes storage refused: usable online by
 * whoever catches this, and never a claim that anything was stored. */
export function storageError(cause, bytes) {
  const error = new Error(`The network is not saved offline: ${cause?.message ?? cause ?? 'storage unavailable'}`);
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
      && Number(held.headers.get('Content-Length')) === expect.bytes;
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
 * caller: 'status', 'overrun', 'size', 'digest' or 'timeout'. */
export async function verifiedDownload(url, expect, { init, idleMs, totalMs, onProgress, describe }) {
  const controller = new AbortController();
  let idle, total, reader, rejectAbort;
  const aborted = new Promise((_, reject) => { rejectAbort = reject; });
  const timeout = () => {
    const error = new Error(describe('timeout'));
    controller.abort(error); rejectAbort(error);
  };
  const resetIdle = () => { clearTimeout(idle); idle = setTimeout(timeout, idleMs); };
  const bounded = promise => Promise.race([promise, aborted]);
  if (totalMs) total = setTimeout(timeout, totalMs);
  resetIdle();
  try {
    const response = await bounded(fetch(url, { ...init, signal: controller.signal }));
    if (!response.ok) throw new Error(describe('status', response.status));
    // Content-Length can be compressed; the expected size bounds decoded bytes.
    const bytes = new Uint8Array(expect.bytes);
    let seen = 0;
    reader = response.body?.getReader();
    if (reader) {
      for (;;) {
        const { done, value } = await bounded(reader.read());
        if (done) break;
        if (seen + value.byteLength > expect.bytes) throw new Error(describe('overrun'));
        bytes.set(value, seen);
        seen += value.byteLength;
        if (value.byteLength) resetIdle();
        onProgress?.({ bytes: seen, total: expect.bytes });
      }
    } else {
      const body = new Uint8Array(await bounded(response.arrayBuffer()));
      if (body.length > expect.bytes) throw new Error(describe('overrun'));
      bytes.set(body); seen = body.length;
    }
    if (seen !== expect.bytes) throw new Error(describe('size', seen));
    if (await bounded(digestOf(bytes)) !== expect.sha256) throw new Error(describe('digest'));
    return { response, bytes };
  } finally {
    clearTimeout(idle); clearTimeout(total);
    // Do not await a broken stream's cancellation promise.
    if (reader) { try { void reader.cancel().catch(() => {}); } catch { /* Already closed. */ } }
    controller.abort();
  }
}

/** Called by the active worker only: at activation, after the previous worker
 * has released its clients, and after its own page has saved this model.
 * Keep this model, the previous activated model and every model at an
 * address in `listed`: the networks the worker's build lists, any of which
 * its pages may play, so a network a player picked is never deleted because
 * another was saved after it. Installation, waiting workers and failed
 * replacement downloads never trigger deletion. canPrune is checked again
 * before each delete in case an update starts. */
export async function pruneNetworkCache({ scope, url, expect, listed = [], canPrune = () => true }) {
  if (!scope || !canPrune()) return false;
  validateNetwork(expect);
  const cache = await networkCache(scope);
  if (!cache) return false;
  const key = new URL('__network_retention__', scope).href;
  try {
    const held = await cache.match(key);
    const old = held ? await held.json() : null;
    if (old && (typeof old.current !== 'string'
      || (old.previous != null && typeof old.previous !== 'string'))) return false;
    if (!await recorded(cache, url, expect)) {
      // Activation precedes the optional download. Remember a first
      // identity so it can become the rollback model on the next upgrade.
      // An existing record never advances to a model that is not stored.
      if (!old && canPrune()) await cache.put(key, new Response(JSON.stringify({ current: url, previous: null })));
      return false;
    }
    const previous = old?.current === url ? old.previous : old?.current;
    const keep = new Set([key, url, previous, ...listed].filter(Boolean));
    if (!canPrune()) return false;
    await cache.put(key, new Response(JSON.stringify({ current: url, previous: previous ?? null }), {
      headers: { 'Content-Type': 'application/json' },
    }));
    for (const entry of await cache.keys()) {
      if (!canPrune()) return false;
      if (!keep.has(entry.url)) await cache.delete(entry);
    }
    return true;
  } catch { return false; } // Cleanup must never prevent a working app activating.
}
