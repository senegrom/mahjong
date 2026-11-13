/** Shared verified, bounded model transfer. Its part the service worker uses
 * too, network-cache.js, is inlined into the worker at build time so an
 * offline startup never imports a network dependency. Only pages and their
 * workers download the network: a service-worker event is abandoned after a
 * few minutes, which a slow link needs for 100 MB. */
import { digestOf, networkCache, storageError, validateNetwork, verifiedDownload } from './network-cache.js';

export { NETWORK_CACHE, hasStoredNetwork, networkCacheName, pruneNetworkCache, validateNetwork } from './network-cache.js';
// A page has no event deadline, so a slow but progressing transfer may take
// hours; a stalled one still ends after a minute without bytes.
const NETWORK_TIMEOUTS = Object.freeze({ totalMs: 3 * 60 * 60 * 1000, idleMs: 60000 });

/** The stored network's bytes where they are the ones expected, or null.
 * An entry read whole and found to be other bytes is deleted. One that
 * could not be read at all is deleted too, for the page's own reader, which
 * then downloads the network again, unless `keepUnread`: the graphics
 * card's worker reads a second copy beside the page's, and its failure, of
 * memory for one, says nothing of the entry, which offline play depends on;
 * there the error is thrown. */
async function verifiedCached(cache, url, expect, { keepUnread = false } = {}) {
  if (!cache) return null;
  try {
    const held = await cache.match(url);
    if (!held) return null;
    const bytes = new Uint8Array(await held.arrayBuffer());
    if (bytes.length === expect.bytes && await digestOf(bytes) === expect.sha256) return bytes;
  } catch (error) {
    if (keepUnread) throw error;
    /* An unreadable cache is not a prerequisite for online play. */
  }
  try { await cache.delete(url); } catch { /* Never turn a cache error into trusted bytes. */ }
  return null;
}

async function downloadNetwork(url, expect, onProgress, timeouts) {
  const { totalMs, idleMs } = { ...NETWORK_TIMEOUTS, ...timeouts };
  if (![totalMs, idleMs].every(n => Number.isSafeInteger(n) && n > 0 && n <= 2 ** 31 - 1)) {
    throw new Error('Invalid network download deadline');
  }
  const describe = (kind, detail) => ({
    status: `The network could not be fetched: ${detail}`,
    overrun: 'The network exceeded its expected byte size',
    size: `The network arrived as ${detail} bytes, not ${expect.bytes}`,
    digest: 'The network that arrived is not the one this page was built for',
    timeout: 'Network download timed out. Reconnect and retry.',
  })[kind];
  const { bytes } = await verifiedDownload(url, expect, { init: { mode: 'cors' }, idleMs, totalMs, onProgress, describe });
  return bytes;
}

/** The verified bytes of the network stored at url in this scope's cache, or
 * null where none is: never a download, for the graphics card's worker,
 * which plays only a network already saved (gpu-batch.worker.js). An entry
 * it cannot read it leaves as it is, and throws. */
export async function storedNetworkBytes({ url, expect, scope } = {}) {
  validateNetwork(expect);
  return verifiedCached(await networkCache(scope), url, expect, { keepUnread: true });
}

/** Verified bytes are usable online even when storage fails. Durable callers
 * explicitly require a successful cache write; they must never claim readiness
 * from a transient in-memory download. Errors never publish unverified bytes. */
export async function verifiedNetworkBytes({ url, expect, onProgress,
  requireStored = false, timeouts, scope } = {}) {
  validateNetwork(expect);
  const cache = await networkCache(scope);
  const held = await verifiedCached(cache, url, expect);
  if (held) return held;
  const bytes = await downloadNetwork(url, expect, onProgress, timeouts);
  try {
    if (!cache) throw storageError();
    // A Response copies the bytes it is given, so the caller keeps its own.
    // These headers are the record hasStoredNetwork reads.
    await cache.put(url, new Response(bytes, {
      headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(bytes.length),
        'X-Mahjong-SHA256': expect.sha256 },
    }));
  } catch (error) {
    if (requireStored) throw storageError(error, bytes);
  }
  return bytes;
}
