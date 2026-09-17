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

async function networkCache() {
  try { return typeof caches === 'undefined' ? null : await caches.open(NETWORK_CACHE); }
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

export async function verifiedNetworkIsStored(url, expect) {
  validateNetwork(expect);
  return Boolean(await verifiedCached(await networkCache(), url, expect));
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
  requireStored = false, onStorage, timeouts } = {}) {
  validateNetwork(expect);
  if (signal?.aborted) throw signal.reason ?? new DOMException('Download cancelled', 'AbortError');
  const cache = await networkCache();
  const held = await verifiedCached(cache, url, expect);
  if (held) { onStorage?.(true); return held; }
  const bytes = await downloadNetwork(url, expect, onProgress, signal, timeouts);
  let stored = false, failure;
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
