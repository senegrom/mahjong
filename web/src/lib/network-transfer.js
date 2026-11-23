/** Shared verified, bounded model transfer. Its part the service worker uses
 * too, network-cache.js, is inlined into the worker at build time so an
 * offline startup never imports a network dependency. Only pages and their
 * workers download the network: a service-worker event is abandoned after a
 * few minutes, which a slow link needs for 100 MB. */
import { digestOf, hasStoredNetwork, networkCache, networkCacheName, storageError, validateNetwork, verifiedDownload } from './network-cache.js';

export { NETWORK_CACHE, hasStoredNetwork, networkCacheName, pruneNetworkCache, validateNetwork } from './network-cache.js';
// A page has no event deadline, so a slow but progressing transfer may take
// hours; a stalled one still ends after a minute without bytes.
const NETWORK_TIMEOUTS = Object.freeze({ totalMs: 3 * 60 * 60 * 1000, idleMs: 60000 });
/** How long one waits its turn at a network before going ahead regardless
 * (alone): SILENT_MS after its holder last said it is alive, or TURN_MS in
 * all where the browser has no BroadcastChannel to hear it by. How often the
 * holder says so (ALIVE_MS), and how often the one waiting says that it
 * waits (WAITING_MS). */
export const TURN_MS = 3 * 60 * 1000, WAITING_MS = 30000, ALIVE_MS = 1000;
// As long as the download's own idle deadline, with room for the holder's
// silent stretches beside it: reading and hashing what is stored before it
// fetches, and writing what arrived.
export const SILENT_MS = NETWORK_TIMEOUTS.idleMs + 15000;

/** A channel to the other windows and workers of this site, or null where
 * the browser has none. */
function channel(name) {
  try { return typeof BroadcastChannel === 'function' ? new BroadcastChannel(name) : null; } catch { return null; }
}

/** Runs `task` while no other window or worker of this site is at the same
 * network: under a Web Lock named `name`, held across check, download and
 * store. A second tab, or the opponent's worker beside a page that is saving
 * the network, then waits its turn and finds the network stored, where each
 * used to fetch all of it beside the other. One that has to wait tells
 * `onWaiting` at once and then every WAITING_MS, for whoever says the wait
 * and whoever watches it for a sign of life.
 *
 * The holder says it is alive, on a BroadcastChannel named after the lock,
 * as it takes the lock and then as bytes arrive, at most every ALIVE_MS:
 * `task` is handed the function that says it. One waiting goes ahead without
 * the lock once its holder has been silent for SILENT_MS: a window the
 * browser has suspended would keep it waiting for ever, and one whose
 * download stalled ends that download itself by then. A holder still
 * receiving bytes is waited for however long it takes, where a slow link
 * used to have both fetch the whole network side by side after three
 * minutes. Without BroadcastChannel nothing is heard, and the wait ends
 * after TURN_MS as it did. Where the browser has no Web Locks, or refuses
 * this one, the task runs as it always did. */
async function alone(name, task, onWaiting) {
  const locks = globalThis.navigator?.locks;
  if (typeof locks?.request !== 'function') return task(() => {});
  let began = false, limit, told, heard = null;
  const wait = new AbortController();
  const stop = () => {
    clearTimeout(limit); clearInterval(told);
    if (heard) { heard.onmessage = null; heard.close(); heard = null; }
  };
  const run = async () => {
    began = true; stop();
    const says = channel(name);
    let last = -Infinity;
    const alive = () => {
      const now = Date.now();
      if (now - last < ALIVE_MS) return;
      last = now;
      try { says?.postMessage('alive'); } catch { /* A closed channel says nothing. */ }
    };
    alive();
    try { return await task(alive); } finally { says?.close(); }
  };
  const NOT_FREE = Symbol('not free');
  try {
    // Taken at once where it is free: only a wait is said.
    const free = await locks.request(name, { ifAvailable: true }, lock => (lock ? run() : NOT_FREE));
    if (free !== NOT_FREE) return free;
    heard = channel(name);
    const patience = heard ? SILENT_MS : TURN_MS;
    const giveWay = () => { clearTimeout(limit); limit = setTimeout(() => wait.abort(), patience); };
    giveWay();
    if (heard) heard.onmessage = giveWay;
    if (onWaiting) { onWaiting(); told = setInterval(onWaiting, WAITING_MS); }
    return await locks.request(name, { signal: wait.signal }, run);
  } catch (error) {
    // The task's own failure is its caller's. A lock refused, or not had in
    // time, is no reason not to run it.
    if (began) throw error;
    return run();
  } finally { stop(); }
}

/** The stored network's bytes where they are the ones expected, or null.
 * Only an entry read whole and found to be other bytes is deleted. One that
 * could not be read at all is left as it is: a failure to read, of memory
 * for one, says nothing of the entry, which offline play depends on, and a
 * download that follows writes over it. The page's own reader then answers
 * null, and downloads the network again; with `keepUnread`, for the
 * graphics card's worker, which never downloads, the error is thrown. */
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
    return null;
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
 * from a transient in-memory download. Errors never publish unverified bytes.
 * One window or worker at a time is at a network (alone): another that asks
 * meanwhile waits, telling `onWaiting`, and then reads what the first stored.
 * `storeOnly` is for a caller that wants the network stored and not its
 * bytes, the page's save: one whose record is there already, written by the
 * window it waited for, say, is answered with null, neither read nor hashed
 * again only to be dropped. */
export async function verifiedNetworkBytes({ url, expect, onProgress, onWaiting,
  requireStored = false, storeOnly = false, timeouts, scope } = {}) {
  validateNetwork(expect);
  const cache = await networkCache(scope);
  return alone(`${networkCacheName(scope)}:${url}`, async alive => {
    if (storeOnly && await hasStoredNetwork(url, expect, { scope })) return null;
    const held = await verifiedCached(cache, url, expect);
    if (held) return held;
    const bytes = await downloadNetwork(url, expect, progress => { alive(); onProgress?.(progress); }, timeouts);
    alive();
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
  }, onWaiting);
}
