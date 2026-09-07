import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { createHash, webcrypto } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import vm from 'node:vm';
import { buildOffline } from '../scripts/offline-build.mjs';

const template = await readFile(new URL('../src/offline/service-worker.js', import.meta.url), 'utf8');
const sha = body => createHash('sha256').update(body).digest('hex');
const base = 'https://test.invalid/mahjong/';
const bodies = { 'index.html': '<html>Game</html>', 'app.js': 'game', 'tiles/Haku.svg': '<svg>dragon</svg>',
  'model.onnx': 'weights', 'ort/runtime.wasm': 'wasm', 'ort/runtime.mjs': 'runtime' };
const manifest = (files = bodies) => ({ version: sha(JSON.stringify(files)), hasModel: true,
  entries: Object.entries(files).map(([url, body]) => ({ url, hash: sha(body), bytes: Buffer.byteLength(body),
    group: url === 'model.onnx' || url.startsWith('ort/') ? 'ai' : 'core' })) });
function storage() {
  const stores = new Map();
  return { stores, async open(name) {
    if (!stores.has(name)) stores.set(name, new Map());
    const entries = stores.get(name);
    return { async match(request) { return entries.get(String(request.url ?? request))?.clone(); },
      async put(request, response) { entries.set(String(request.url ?? request), response.clone()); },
      async delete(request) { return entries.delete(String(request.url ?? request)); },
      async keys() { return [...entries.keys()].map(url => new Request(url)); } };
  } };
}
function worker({ files = bodies, caches = storage(), config = manifest(files), network } = {}) {
  const events = {}, counts = new Map();
  const self = { registration: { scope: base }, clients: { async claim() {} }, addEventListener: (type, fn) => events[type] = fn };
  vm.runInNewContext(template.replace('/* OFFLINE_CONFIG */ null', JSON.stringify(config)), {
    self, URL, Request, Response, Map, Set, caches, crypto: webcrypto, AbortSignal,
    fetch: async url => {
      const name = new URL(url).pathname.slice('/mahjong/'.length);
      counts.set(name, (counts.get(name) ?? 0) + 1);
      if (network) return network(name);
      return new Response(files[name], { status: name in files ? 200 : 404 });
    },
  });
  return { caches, counts,
    async install() { let work; events.install({ waitUntil(p) { work = p; } }); await work; },
    async activate() { let work; events.activate({ waitUntil(p) { work = p; } }); await work; },
    async message(type) {
      let work, reply;
      events.message({ data: { type }, ports: [{ postMessage(data) { if (!data.progress) reply = data; } }], waitUntil(p) { work = p; } });
      await work;
      if (reply.error) throw new Error(reply.error);
      return reply.value;
    },
    async get(path, method = 'GET') {
      let response;
      events.fetch({ request: new Request(new URL(path, base), { method }), respondWith(p) { response = p; } });
      return response;
    },
  };
}
const status = w => w.message('MAHJONG_STATUS');
const download = w => w.message('MAHJONG_PREPARE_AI');

test('offline install saves the shell and all graphics, with AI explicitly incomplete', async () => {
  const w = worker(); await w.install();
  assert.equal((await status(w)).coreReady, true);
  assert.equal((await status(w)).aiReady, false);
  assert.equal(w.counts.has('model.onnx'), false);
  assert.equal(await (await w.get('?opponents=neural')).text(), bodies['index.html']);
  assert.equal(await (await w.get('tiles/Haku.svg')).text(), bodies['tiles/Haku.svg']);
});
test('all AI bytes are durable before ready; simultaneous requests share downloads', async () => {
  const files = { ...bodies, 'ort/duplicate.wasm': bodies['ort/runtime.wasm'] };
  const w = worker({ files }); await w.install();
  await Promise.all([download(w), download(w), download(w)]);
  assert.equal((await status(w)).aiReady, true);
  const sum = [...w.counts].filter(([name]) => /model|ort\//.test(name)).reduce((n, [, count]) => n + count, 0);
  assert.equal(sum, 3, 'model, runtime module and one copy of the WASM');
  await download(w); assert.equal([...w.counts.values()].reduce((a, b) => a + b), 6);
});
test('a cold worker serves navigation, tiles, model and runtime without ANY network, including HEAD', async () => {
  const original = worker(); await original.install(); await download(original);
  const cold = worker({ caches: original.caches, network() { throw new Error('plane'); } });
  assert.equal((await status(cold)).aiReady, true);
  for (const [path, body] of Object.entries(bodies)) assert.equal(await (await cold.get(path)).text(), body);
  assert.equal(await (await cold.get('model.onnx', 'HEAD')).text(), '');
  assert.equal(await (await cold.get('?launched=home-screen')).text(), bodies['index.html']);
  assert.equal(cold.counts.size, 0);
});
test('truncated or captive-portal downloads never count as AI ready and retry reuses good files', async () => {
  for (const bad of ['', 'portal!']) {
    let corrupt = true;
    const w = worker({ network: name => new Response(corrupt && name === 'model.onnx' ? bad : bodies[name]) });
    await w.install(); await assert.rejects(download(w), /Incomplete or outdated/);
    assert.equal((await status(w)).aiReady, false);
    assert.equal((await status(w)).coreReady, true);
    // In-flight sibling downloads settle before retrying.
    await new Promise(done => setTimeout(done, 10));
    corrupt = false; await download(w);
    assert.equal((await status(w)).aiReady, true);
    assert.equal(w.counts.get('model.onnx'), 2);
    assert.equal(w.counts.get('ort/runtime.wasm'), 1);
  }
});
test('an app-only upgrade reuses every tile and all trained weights/runtime bytes', async () => {
  const old = worker(); await old.install(); await download(old);
  const files = { ...bodies, 'index.html': '<html>New game UI</html>' };
  const update = worker({ files, caches: old.caches }); await update.install();
  assert.equal((await status(update)).aiReady, true);
  assert.deepEqual([...update.counts.keys()], ['index.html']);
  assert.equal(await (await old.get('index.html')).text(), bodies['index.html']);
});
test('activation removes obsolete Mahjong bodies but keeps current content and metadata', async () => {
  const old = worker(); await old.install(); await download(old);
  const files = { ...bodies, 'index.html': 'new shell', 'tiles/Haku.svg': '<svg>new dragon</svg>', 'model.onnx': 'new weights' };
  const update = worker({ files, caches: old.caches }); await update.install();
  const store = update.caches.stores.get('mahjong-offline-v1:/mahjong/');
  const oldHashes = [bodies['index.html'], bodies['tiles/Haku.svg'], bodies['model.onnx']].map(sha);
  for (const hash of oldHashes) assert.equal(store.has(base + '__offline_content__/' + hash), true);
  await update.activate();
  for (const hash of oldHashes) assert.equal(store.has(base + '__offline_content__/' + hash), false);
  for (const body of new Set(Object.values(files))) assert.equal(store.has(base + '__offline_content__/' + sha(body)), true);
  assert.equal(store.has(base + '__offline_meta__/ai-requested'), true);
});
test('an interrupted AI upgrade cannot install or destroy the working offline version', async () => {
  const old = worker(); await old.install(); await download(old);
  const files = { ...bodies, 'index.html': 'new shell', 'model.onnx': 'new weights' };
  const update = worker({ files, caches: old.caches, network: name => new Response(name === 'model.onnx' ? '' : files[name], { status: name === 'model.onnx' ? 503 : 200 }) });
  await assert.rejects(update.install(), /Download failed/);
  assert.equal((await status(old)).aiReady, true);
  assert.equal(await (await old.get('model.onnx')).text(), bodies['model.onnx']);
});
test('cache eviction is detected and unrelated apps and unknown assets are untouched', async () => {
  const w = worker(); await w.install(); await download(w);
  const other = await w.caches.open('plateloader'); await other.put(base + 'sentinel', new Response('keep'));
  const store = w.caches.stores.get('mahjong-offline-v1:/mahjong/');
  store.delete(base + '__offline_content__/' + sha(bodies['model.onnx']));
  assert.equal((await status(w)).aiReady, false);
  assert.equal(await (await other.match(base + 'sentinel')).text(), 'keep');
  assert.equal(await w.get('../plateloader/'), undefined);
  assert.equal(await w.get('missing.wasm'), undefined);
});
test('quota errors cannot be misreported as successfully saved offline', async () => {
  const caches = { async open() { return { async match() {}, async put() { throw new Error('QuotaExceededError'); } }; } };
  const w = worker({ caches }); await assert.rejects(w.install(), /QuotaExceededError/);
  assert.equal((await status(w)).coreReady, false);
  await assert.rejects(download(w), /QuotaExceededError/);
});
test('production inventory classifies the external model/runtime package', async t => {
  const root = await mkdtemp(join(tmpdir(), 'mahjong-offline-build-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const files = { 'index.html': 'game', 'assets/worker-hash.js': 'worker', 'assets/riichi_bg-hash.wasm': 'engine',
    'tiles/Back.svg': '<svg/>', 'tiles/Haku.svg': '<svg>white</svg>', 'assets/white-dragon-hash.webp': 'dragon',
    'model.onnx': 'network', 'ort/ort-wasm-simd-threaded.wasm': 'wasm',
    'ort/ort-wasm-simd-threaded.mjs': 'loader' };
  for (const [path, body] of Object.entries(files)) { await mkdir(join(root, path, '..'), { recursive: true }); await writeFile(join(root, path), body); }
  const first = await buildOffline(root), second = await buildOffline(root);
  assert.deepEqual(first, second);
  assert.equal(first.entries.length, Object.keys(files).length);
  assert.equal(first.entries.filter(e => e.group === 'ai').length, 3);
  assert.ok((await readFile(join(root, 'sw.js'), 'utf8')).includes(JSON.stringify(first)));
});
