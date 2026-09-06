/** Build substitutes the complete, hashed inventory here. Kept self-contained
 * so starting the service worker while offline needs no imported scripts. */
const CONFIG = /* OFFLINE_CONFIG */ null;
const scope = new URL(self.registration.scope);
// GitHub Pages projects share an origin. Never touch another app's caches.
const CACHE = `mahjong-offline-v1:${scope.pathname}`;
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
async function ensure(entry) {
  const cache = await caches.open(CACHE), hit = await cached(cache, entry);
  if (hit) return hit;
  if (!downloads.has(entry.hash)) {
    const task = (async () => {
      const response = await fetch(new URL(entry.url, scope), { cache: 'no-store', signal: AbortSignal.timeout(90000) });
      if (!response.ok) throw new Error(`Download failed: ${entry.url} (HTTP ${response.status})`);
      const body = await response.arrayBuffer();
      const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', body))]
        .map(byte => byte.toString(16).padStart(2, '0')).join('');
      // A captive portal, truncated connection or mismatched deploy must not
      // replace a good asset or be presented as a completed offline download.
      if (body.byteLength !== entry.bytes || hash !== entry.hash) throw new Error(`Incomplete or outdated download: ${entry.url}`);
      const type = mimeTypes[entry.url.split('.').pop()] ?? response.headers.get('Content-Type') ?? 'application/octet-stream';
      const saved = new Response(body, { headers: { 'Content-Type': type, 'Content-Length': String(entry.bytes),
        'X-Mahjong-SHA256': hash, 'X-Content-Type-Options': 'nosniff' } });
      await cache.put(keyFor(entry), saved);
    })().finally(() => downloads.delete(entry.hash));
    downloads.set(entry.hash, task);
  }
  await downloads.get(entry.hash);
  return cached(cache, entry);
}
async function status() {
  const cache = await caches.open(CACHE), complete = { core: true, ai: CONFIG.hasModel };
  for (const entry of CONFIG.entries) if (!(await cached(cache, entry))) complete[entry.group] = false;
  return { coreReady: complete.core, aiReady: complete.ai, hasModel: CONFIG.hasModel, version: CONFIG.version };
}
async function prepare(group, progress = () => {}) {
  if (group === 'ai' && !CONFIG.hasModel) throw new Error('No trained model is included in this version.');
  if (groups.has(group)) { await groups.get(group); return status(); }
  // Deduplicate the public runtime and the bundler's identical copy.
  const list = [...new Map(CONFIG.entries.filter(entry => entry.group === group).map(entry => [entry.hash, entry])).values()];
  const total = list.reduce((sum, entry) => sum + entry.bytes, 0);
  let next = 0, bytes = 0;
  const task = Promise.all(Array.from({ length: Math.min(4, list.length) }, async () => {
    while (next < list.length) {
      const entry = list[next++];
      await ensure(entry);
      bytes += entry.bytes;
      progress({ bytes, total, group });
    }
  })).finally(() => groups.delete(group));
  groups.set(group, task);
  await task;
  return status();
}
self.addEventListener('install', event => {
  // Failed updates leave the active worker and every verified asset intact.
  // Normal waiting-worker lifecycle: no forced upgrade during a live match.
  event.waitUntil((async () => {
    await prepare('core');
    const cache = await caches.open(CACHE);
    // An upgrade must not strand an offline Trained match on a new runtime
    // whose bytes have not finished downloading. Keep the old worker instead.
    if (CONFIG.hasModel && await cache.match(new URL('__offline_meta__/ai-requested', scope))) await prepare('ai');
  })());
});
self.addEventListener('activate', event => { event.waitUntil(self.clients.claim()); });
self.addEventListener('message', event => {
  const port = event.ports[0];
  if (!port || !['MAHJONG_STATUS', 'MAHJONG_PREPARE_AI', 'MAHJONG_PREPARE_CORE'].includes(event.data?.type)) return;
  const progress = value => port.postMessage({ progress: value });
  const work = (async () => {
    if (event.data.type === 'MAHJONG_STATUS') return status();
    if (event.data.type === 'MAHJONG_PREPARE_AI') {
      const cache = await caches.open(CACHE);
      await cache.put(new URL('__offline_meta__/ai-requested', scope), new Response('requested'));
    }
    return prepare(event.data.type === 'MAHJONG_PREPARE_AI' ? 'ai' : 'core', progress);
  })();
  event.waitUntil(work.then(value => port.postMessage({ value }))
    .catch(error => port.postMessage({ error: error.message })));
});
self.addEventListener('fetch', event => {
  const request = event.request, url = new URL(request.url);
  if (!['GET', 'HEAD'].includes(request.method) || url.origin !== scope.origin || !url.pathname.startsWith(scope.pathname)) return;
  // Query strings select game preferences, not a different application shell.
  const path = url.pathname === scope.pathname ? new URL('index.html', scope).pathname : url.pathname;
  const entry = entries.get(path);
  if (!entry) return; // Unknown resources never receive HTML disguised as JS/WASM.
  event.respondWith(ensure(entry).then(response => request.method === 'HEAD'
    ? new Response(null, { headers: response.headers }) : response)
    .catch(() => new Response('Not saved offline. Reconnect and finish the download.', { status: 503 })));
});
// Intentionally retain content-addressed bodies across upgrades. Unchanged tiles,
// models and runtimes are never downloaded again just because the UI changed.
