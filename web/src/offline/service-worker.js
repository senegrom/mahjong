/** Build substitutes the complete, hashed inventory here. Kept self-contained
 * so starting the service worker while offline needs no imported scripts. */
const networkTransfer = /* NETWORK_TRANSFER */ null;
const CONFIG = /* OFFLINE_CONFIG */ null;
const scope = new URL(self.registration.scope);
// GitHub Pages projects share an origin. Never touch another app's caches.
const CACHE = `mahjong-offline-v1:${scope.pathname}`;
const networkScope = scope.href;
const keyFor = entry => new URL(`__offline_content__/${entry.hash}`, scope).href;
const entries = new Map(CONFIG.entries.map(entry => [new URL(entry.url, scope).pathname, entry]));
const downloads = new Map();
const groups = new Map();
const mimeTypes = { html: 'text/html', js: 'text/javascript', mjs: 'text/javascript', css: 'text/css',
  svg: 'image/svg+xml', png: 'image/png', webp: 'image/webp', ico: 'image/x-icon',
  wasm: 'application/wasm', json: 'application/json', webmanifest: 'application/manifest+json' };

async function cached(cache, entry) {
  const response = await cache.match(keyFor(entry));
  return response?.headers.get('X-Mahjong-SHA256') === entry.hash
    && Number(response.headers.get('Content-Length')) === entry.bytes ? response : null;
}
// No fixed cap on a file: on a slow link, four 3 MB tiles share it for
// minutes, and a file cut off at a fixed time is lost whole, on every
// attempt. A file that keeps arriving goes on; one silent for a minute has
// stalled.
const IDLE_MS = 60000;
const failure = url => (kind, detail) => kind === 'status' ? `Download failed: ${url} (HTTP ${detail})`
  : kind === 'timeout' ? `Download stalled: ${url}` : `Incomplete or outdated download: ${url}`;
async function ensure(entry, durable = true) {
  let cache, hit;
  try { cache = await caches.open(CACHE); hit = await cached(cache, entry); }
  catch (error) { if (durable) throw networkTransfer.storageError(error); }
  if (hit) return hit;
  if (!downloads.has(entry.hash)) {
    const task = (async () => {
      const { response, bytes } = await networkTransfer.verifiedDownload(new URL(entry.url, scope),
        { bytes: entry.bytes, sha256: entry.hash },
        { init: { cache: 'no-store' }, idleMs: IDLE_MS, describe: failure(entry.url) });
      const type = mimeTypes[entry.url.split('.').pop()] ?? response.headers.get('Content-Type') ?? 'application/octet-stream';
      const saved = new Response(bytes, { headers: { 'Content-Type': type, 'Content-Length': String(entry.bytes),
        'X-Mahjong-SHA256': entry.hash, 'X-Content-Type-Options': 'nosniff' } });
      try {
        if (!cache) throw networkTransfer.storageError();
        await cache.put(keyFor(entry), saved.clone());
        return { response: saved, stored: true };
      } catch (error) { return { response: saved, stored: false, error }; }
    })().finally(() => downloads.delete(entry.hash));
    downloads.set(entry.hash, task);
  }
  const answer = await downloads.get(entry.hash);
  if (durable && !answer.stored) throw networkTransfer.storageError(answer.error);
  return answer.response.clone();
}
const networkUrl = network => `${network.origin}/${network.object}`;
// The build lists several networks, its default first, and a page plays the
// one its player picked. Every message from a page names that pick, which is
// read here against this build's own list, as this build's page reads it: a
// version this build does not list, or none, is its default. So a waiting
// update answers for the network its own page will play.
const networkFor = version => CONFIG.networks.find(network => network.version === version) ?? CONFIG.networks[0];
const requestedKey = () => new URL('__offline_meta__/ai-requested', scope);
// The worker never downloads the network. An install, activation or message
// event is abandoned after a few minutes (Chrome: five), and nothing of a
// half-received body is kept, so a slow link could never finish 100 MB here.
// The page saves it instead; this reads the record that save leaves, the way
// cached() reads the worker's own files, without reading or hashing a body.
const networkReady = network => networkTransfer.hasStoredNetwork(networkUrl(network), network, { scope: networkScope });

async function status(version) {
  const network = networkFor(version);
  const complete = { core: true, ai: true };
  let aiRequested = false;
  try {
    const cache = await caches.open(CACHE);
    for (const entry of CONFIG.entries) if (!(await cached(cache, entry))) complete[entry.group] = false;
    if (complete.ai && !(await networkReady(network))) complete.ai = false;
    aiRequested = Boolean(await cache.match(requestedKey()));
  } catch { complete.core = false; complete.ai = false; }
  // The identity is the network the asking page plays, saved or not: aiReady
  // says whether it is stored, and a page saving it for this build needs to
  // know which. Every build carries the runtime: the build refuses to go
  // without it.
  return { coreReady: complete.core, aiReady: complete.ai, aiRequested, hasModel: true,
    network: { url: networkUrl(network), version: network.version, sha256: network.sha256, bytes: network.bytes } };
}
async function prepare(group, progress = () => {}, version) {
  if (groups.has(group)) { await groups.get(group); return status(version); }
  // Entries with equal content are fetched once. For 'ai' these are the few
  // megabytes of runtime, never the network itself.
  const list = [...new Map(CONFIG.entries.filter(entry => entry.group === group).map(entry => [entry.hash, entry])).values()];
  const total = list.reduce((sum, entry) => sum + entry.bytes, 0);
  let next = 0, bytes = 0;
  // A lane that fails stops, and the others go on: the attempt keeps every
  // file it can finish, and the next one fetches only what is missing. The
  // first failure is reported once every lane has settled.
  const task = Promise.allSettled(Array.from({ length: Math.min(4, list.length) }, async () => {
    while (next < list.length) {
      const entry = list[next++];
      await ensure(entry);
      bytes += entry.bytes;
      progress({ bytes, total, group });
    }
  })).then(lanes => {
    const failed = lanes.find(lane => lane.status === 'rejected');
    if (failed) throw failed.reason;
  }).finally(() => groups.delete(group));
  groups.set(group, task);
  await task;
  return status(version);
}
/** The active worker keeps every network its build lists, which its pages
 * may play, the network a page has just saved (at activation, the default)
 * and the one current before it; a waiting or installing update prevents any
 * deletion. */
async function pruneNetworks(version) {
  const network = networkFor(version);
  await networkTransfer.pruneNetworkCache({
    scope: networkScope, url: networkUrl(network), expect: network, listed: CONFIG.networks.map(networkUrl),
    canPrune: () => !self.registration.installing && !self.registration.waiting,
  });
}
async function pruneObsoleteContent() {
  // Activation happens only after the old worker no longer owns live clients.
  // Old version-specific bodies are then safe to remove. Metadata is retained,
  // so a user who requested Trained once keeps that preference across upgrades.
  const cache = await caches.open(CACHE);
  const keep = new Set(CONFIG.entries.map(entry => keyFor(entry)));
  const prefix = new URL('__offline_content__/', scope).pathname;
  for (const request of await cache.keys()) {
    const url = new URL(request.url);
    if (url.pathname.startsWith(prefix) && !keep.has(request.url)) await cache.delete(request);
  }
}
self.addEventListener('install', event => {
  // The game only: an install still running when the browser's event deadline
  // passes is discarded, so the trained network is never part of one. A page
  // of this version saves it after activation when Trained is wanted (or the
  // previous version's page saves it while this one waits). Failed updates
  // leave the active worker and every verified asset intact. Normal
  // waiting-worker lifecycle: no forced upgrade during a live match.
  event.waitUntil(prepare('core'));
});
self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    // Nothing is downloaded here either. The default network, when the page
    // saved it while this version waited, becomes current; one not yet saved
    // leaves the retention record, and the previous version's recoverable
    // bytes, untouched. No network this build lists is deleted.
    await pruneNetworks();
    await pruneObsoleteContent();
    await self.clients.claim();
  })());
});
self.addEventListener('message', event => {
  const port = event.ports[0];
  const groupOf = { MAHJONG_PREPARE_AI: 'ai', MAHJONG_PREPARE_CORE: 'core' };
  if (!port || !['MAHJONG_STATUS', 'MAHJONG_NETWORK_SAVED', ...Object.keys(groupOf)].includes(event.data?.type)) return;
  const progress = value => port.postMessage({ progress: value });
  // The version the asking page's player picked; a page of a build from
  // before networks were picked names none, and plays the default.
  const version = typeof event.data.version === 'string' ? event.data.version : undefined;
  const work = (async () => {
    if (event.data.type === 'MAHJONG_STATUS') return status(version);
    if (event.data.type === 'MAHJONG_NETWORK_SAVED') { await pruneNetworks(version); return status(version); }
    if (event.data.type === 'MAHJONG_PREPARE_AI') {
      // Remembered across upgrades: later versions' pages save their own
      // network for a player who once asked for Trained offline.
      try {
        const cache = await caches.open(CACHE);
        await cache.put(requestedKey(), new Response('requested'));
      } catch (error) { throw networkTransfer.storageError(error); }
    }
    return prepare(groupOf[event.data.type], progress, version);
  })();
  event.waitUntil(work.then(value => port.postMessage({ value }))
    .catch(error => port.postMessage({ error: error.message, storage: error.name === 'NetworkStorageError'
      || ['QuotaExceededError', 'SecurityError', 'InvalidStateError'].includes(error.name) })));
});
self.addEventListener('fetch', event => {
  const request = event.request, url = new URL(request.url);
  if (!['GET', 'HEAD'].includes(request.method) || url.origin !== scope.origin || !url.pathname.startsWith(scope.pathname)) return;
  // Query strings select game preferences, not a different application shell.
  const path = url.pathname === scope.pathname ? new URL('index.html', scope).pathname : url.pathname;
  const entry = entries.get(path);
  if (!entry) return; // Unknown resources never receive HTML disguised as JS/WASM.
  event.respondWith(ensure(entry, false).then(response => request.method === 'HEAD'
    ? new Response(null, { headers: response.headers }) : response)
    .catch(() => new Response('Not saved offline. Reconnect and finish the download.', { status: 503 })));
});
// Content-addressing reuses unchanged bodies during installation; activation
// removes only hashes the newly active build can no longer reference.