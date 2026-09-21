/** Shared verified, bounded model transfer. Also inlined into the service worker
 * at build time so an offline startup never imports a network dependency. */
export const NETWORK_CACHE = 'mahjong-network-v1';
export const NETWORK_TIMEOUTS = Object.freeze({ totalMs: 30 * 60 * 1000, idleMs: 60000 });

export function validateNetwork(expect) {
  if (!expect || !Number.isSafeInteger(expect.bytes) || expect.bytes <= 0
      || expect.bytes > 512 * 1024 * 1024 || !/^[0-9a-f]{64}$/.test(expect.sha256)) {
    throw new Error('Invalid trained-network size or SHA-256 identity');
  }
}

export function storageError(cause) {
  const error = new Error(`The network is not saved offline: ${cause?.message ?? cause ?? 'storage unavailable'}`);
  error.name = 'NetworkStorageError';
  return error;
}

/** New downloads belong to an installation scope. The shared v1 cache is
 * read for migration only: its old entries have no reliable scope ownership. */
export function networkCacheName(scope) {
  return scope ? `mahjong-network-v2:${new URL('./', scope).href}` : NETWORK_CACHE;
}

async function networkCache(scope) {
  try { return typeof caches === 'undefined' ? null : await caches.open(networkCacheName(scope)); }
  catch { return null; }
}

async function digestOf(bytes) {
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join('');
}

async function verifiedCached(cache, url, expect) {
  if (!cache) return null;
  try {
    const held = await cache.match(url);
    if (!held) return null;
    const bytes = new Uint8Array(await held.arrayBuffer());
    if (bytes.length === expect.bytes && await digestOf(bytes) === expect.sha256) return bytes;
  } catch { /* An unreadable cache is not a prerequisite for online play. */ }
  try { await cache.delete(url); } catch { /* Never turn a cache error into trusted bytes. */ }
  return null;
}

export async function verifiedNetworkIsStored(url, expect, { scope } = {}) {
  validateNetwork(expect);
  return Boolean(await verifiedCached(await networkCache(scope), url, expect)
    || (scope && await verifiedCached(await networkCache(), url, expect)));
}

/** Own the deadline even when a mocked/broken transport ignores cancellation.
 * Subscriber cancellation is optional; the transfer always owns total and idle
 * limits and a byte ceiling. It never retains an unbounded list of chunks. */
async function downloadNetwork(url, expect, onProgress, signal, timeouts) {
  const { totalMs, idleMs } = { ...NETWORK_TIMEOUTS, ...timeouts };
  if (![totalMs, idleMs].every(n => Number.isSafeInteger(n) && n > 0 && n <= 2 ** 31 - 1)) {
    throw new Error('Invalid network download deadline');
  }
  const controller = new AbortController();
  let idle, total, reader, rejectAbort;
  const aborted = new Promise((_, reject) => { rejectAbort = reject; });
  const fail = error => { controller.abort(error); rejectAbort(error); };
  const timeout = () => fail(new Error('Network download timed out. Reconnect and retry.'));
  const resetIdle = () => { clearTimeout(idle); idle = setTimeout(timeout, idleMs); };
  const cancel = () => fail(signal.reason ?? new DOMException('Download cancelled', 'AbortError'));
  const bounded = promise => Promise.race([promise, aborted]);
  signal?.addEventListener('abort', cancel, { once: true });
  total = setTimeout(timeout, totalMs);
  resetIdle();
  try {
    if (signal?.aborted) cancel();
    const response = await bounded(fetch(url, { mode: 'cors', signal: controller.signal }));
    if (!response.ok) throw new Error(`The network could not be fetched: ${response.status}`);
    // Content-Length can be compressed; the manifest bounds decoded bytes.
    const bytes = new Uint8Array(expect.bytes);
    let seen = 0;
    reader = response.body?.getReader();
    if (reader) {
      for (;;) {
        const { done, value } = await bounded(reader.read());
        if (done) break;
        if (seen + value.byteLength > expect.bytes) throw new Error('The network exceeded its expected byte size');
        bytes.set(value, seen);
        seen += value.byteLength;
        if (value.byteLength) resetIdle();
        onProgress?.({ bytes: seen, total: expect.bytes });
      }
    } else {
      const body = new Uint8Array(await bounded(response.arrayBuffer()));
      if (body.length > expect.bytes) throw new Error('The network exceeded its expected byte size');
      bytes.set(body); seen = body.length;
    }
    if (seen !== expect.bytes) throw new Error(`The network arrived as ${seen} bytes, not ${expect.bytes}`);
    if (await bounded(digestOf(bytes)) !== expect.sha256) {
      throw new Error('The network that arrived is not the one this page was built for');
    }
    return bytes;
  } finally {
    clearTimeout(idle); clearTimeout(total);
    signal?.removeEventListener('abort', cancel);
    // Do not await a broken stream's cancellation promise.
    if (reader) { try { void reader.cancel().catch(() => {}); } catch { /* Already closed. */ } }
    controller.abort();
  }
}

/** Verified bytes are usable online even when storage fails. Durable callers
 * explicitly require a successful cache write; they must never claim readiness
 * from a transient in-memory download. Errors never publish unverified bytes. */
export async function verifiedNetworkBytes({ url, expect, onProgress, signal,
  requireStored = false, onStorage, timeouts, scope } = {}) {
  validateNetwork(expect);
  if (signal?.aborted) throw signal.reason ?? new DOMException('Download cancelled', 'AbortError');
  const cache = await networkCache(scope);
  const held = await verifiedCached(cache, url, expect);
  if (held) { onStorage?.(true); return held; }
  // Copy a verified legacy body into the scoped cache without another fetch.
  // It is already durable even when copying fails, and is never deleted here.
  const legacy = scope && await verifiedCached(await networkCache(), url, expect);
  const bytes = legacy || await downloadNetwork(url, expect, onProgress, signal, timeouts);
  let stored = Boolean(legacy), failure;
  try {
    if (!cache) throw storageError();
    await cache.put(url, new Response(bytes.slice().buffer, {
      headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(bytes.length),
        'X-Mahjong-SHA256': expect.sha256 },
    }));
    stored = true;
  } catch (error) { failure = error; }
  onStorage?.(stored);
  if (requireStored && !stored) throw storageError(failure);
  return bytes;
}

/** Called only at activation, after the previous worker has released its
 * clients. Keep this model and the previous activated model. Installation,
 * waiting workers and failed replacement downloads never trigger deletion.
 * canPrune is checked again before each delete in case an update starts. */
export async function pruneNetworkCache({ scope, url, expect, canPrune = () => true }) {
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
    if (!await verifiedCached(cache, url, expect)) {
      // A first install can precede the optional download. Remember its
      // identity so it can become the rollback model on the next upgrade.
      // An existing record is never advanced after failed verification.
      if (!old && canPrune()) await cache.put(key, new Response(JSON.stringify({ current: url, previous: null })));
      return false;
    }
    const previous = old?.current === url ? old.previous : old?.current;
    const keep = new Set([key, url, previous].filter(Boolean));
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
