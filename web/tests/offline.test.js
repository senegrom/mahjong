import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { createHash, webcrypto } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import vm from 'node:vm';
import { buildOffline, serviceWorkerTemplate } from '../scripts/offline-build.mjs';
import { verifiedNetworkBytes } from '../src/lib/network-transfer.js';

const template = await serviceWorkerTemplate();
const sha = body => createHash('sha256').update(body).digest('hex');
const base = 'https://test.invalid/mahjong/';
const bodies = { 'index.html': '<html>Game</html>', 'app.js': 'game', 'tiles/Haku.svg': '<svg>dragon</svg>',
  'model-full.onnx': 'weights', 'ort/runtime.wasm': 'wasm', 'ort/runtime.mjs': 'runtime' };
const manifest = (files = bodies) => ({ version: sha(JSON.stringify(files)), hasModel: true,
  entries: Object.entries(files).map(([url, body]) => ({ url, hash: sha(body), bytes: Buffer.byteLength(body),
    group: url === 'model-full.onnx' || url.startsWith('ort/') ? 'ai' : 'core' })) });
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
function worker({ files = bodies, caches = storage(), config = manifest(files), network, remote = 'weights', deadlines } = {}) {
  const events = {}, counts = new Map();
  const self = { registration: { scope: base }, clients: { async claim() {} }, addEventListener: (type, fn) => events[type] = fn };
  vm.runInNewContext(template.replace('/* OFFLINE_CONFIG */ null', JSON.stringify(config)), {
    self, URL, Request, Response, Map, Set, caches, crypto: webcrypto, AbortSignal, AbortController, DOMException,
    setTimeout: deadlines ? (fn, ms) => setTimeout(fn, ms >= 60000 ? deadlines : ms) : setTimeout, clearTimeout,
    fetch: async url => {
      const address = new URL(url);
      const name = address.origin === new URL(base).origin ? address.pathname.slice('/mahjong/'.length) : address.href;
      counts.set(name, (counts.get(name) ?? 0) + 1);
      if (network) return network(name);
      if (config.network && name === `${config.network.origin}/${config.network.object}`) return new Response(remote);
      return new Response(files[name], { status: name in files ? 200 : 404 });
    },
  });
  return { caches, counts, config,
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
/** What offline.js does once the worker has saved the runtime: the page
 * fetches, verifies and stores the network with the real transfer code,
 * against the same Cache Storage, then tells the worker it is saved. */
async function pageSaves(w, body, respond = () => new Response(body)) {
  const { network } = w.config, previous = { caches: globalThis.caches, fetch: globalThis.fetch };
  let fetched = 0;
  globalThis.caches = w.caches;
  globalThis.fetch = async (...args) => { fetched++; return respond(...args); };
  try {
    await verifiedNetworkBytes({ url: `${network.origin}/${network.object}`, expect: network, requireStored: true,
      scope: w.config.networkCacheVersion === 2 ? base : undefined });
  } finally { globalThis.caches = previous.caches; globalThis.fetch = previous.fetch; }
  await w.message('MAHJONG_NETWORK_SAVED');
  return fetched;
}
const prepareAi = async (w, body) => { await download(w); return pageSaves(w, body); };

test('offline install saves the shell and all graphics, with AI explicitly incomplete', async () => {
  const w = worker(); await w.install();
  assert.equal((await status(w)).coreReady, true);
  assert.equal((await status(w)).aiReady, false);
  assert.equal(w.counts.has('model-full.onnx'), false);
  assert.equal(await (await w.get('?opponents=neural')).text(), bodies['index.html']);
  assert.equal(await (await w.get('tiles/Haku.svg')).text(), bodies['tiles/Haku.svg']);
});
test('all AI bytes are durable before ready; simultaneous requests share downloads', async () => {
  const files = { ...bodies, 'ort/duplicate.wasm': bodies['ort/runtime.wasm'] };
  const w = worker({ files }); await w.install();
  await Promise.all([download(w), download(w), download(w)]);
  assert.equal((await status(w)).aiReady, true);
  const sum = [...w.counts].filter(([name]) => /model|ort\//.test(name)).reduce((n, [, count]) => n + count, 0);
  assert.equal(sum, 3, 'the network, the runtime module and one copy of the WASM');
  await download(w); assert.equal([...w.counts.values()].reduce((a, b) => a + b), 6);
});
test('a cold worker serves navigation, tiles, model and runtime without ANY network, including HEAD', async () => {
  const original = worker(); await original.install(); await download(original);
  const cold = worker({ caches: original.caches, network() { throw new Error('plane'); } });
  assert.equal((await status(cold)).aiReady, true);
  for (const [path, body] of Object.entries(bodies)) assert.equal(await (await cold.get(path)).text(), body);
  assert.equal(await (await cold.get('model-full.onnx', 'HEAD')).text(), '');
  assert.equal(await (await cold.get('?launched=home-screen')).text(), bodies['index.html']);
  assert.equal(cold.counts.size, 0);
});
test('truncated or captive-portal downloads never count as AI ready and retry reuses good files', async () => {
  for (const bad of ['', 'portal!']) {
    let corrupt = true;
    const w = worker({ network: name => new Response(corrupt && name === 'model-full.onnx' ? bad : bodies[name]) });
    await w.install(); await assert.rejects(download(w), /Incomplete or outdated/);
    assert.equal((await status(w)).aiReady, false);
    assert.equal((await status(w)).coreReady, true);
    // In-flight sibling downloads settle before retrying.
    await new Promise(done => setTimeout(done, 10));
    corrupt = false; await download(w);
    assert.equal((await status(w)).aiReady, true);
    assert.equal(w.counts.get('model-full.onnx'), 2);
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
  const files = { ...bodies, 'index.html': 'new shell', 'tiles/Haku.svg': '<svg>new dragon</svg>', 'model-full.onnx': 'new weights' };
  const update = worker({ files, caches: old.caches }); await update.install();
  const store = update.caches.stores.get('mahjong-offline-v1:/mahjong/');
  const oldHashes = [bodies['index.html'], bodies['tiles/Haku.svg'], bodies['model-full.onnx']].map(sha);
  for (const hash of oldHashes) assert.equal(store.has(base + '__offline_content__/' + hash), true);
  assert.equal(update.counts.has('model-full.onnx'), false, 'installation saves the game alone');
  await update.activate();
  for (const hash of oldHashes) assert.equal(store.has(base + '__offline_content__/' + hash), false);
  const core = Object.entries(files).filter(([name]) => name !== 'model-full.onnx' && !name.startsWith('ort/'));
  for (const [, body] of core) assert.equal(store.has(base + '__offline_content__/' + sha(body)), true);
  assert.equal(store.has(base + '__offline_meta__/ai-requested'), true);
  // The request survives the upgrade, and the new version's AI is saved on it.
  assert.equal((await status(update)).aiRequested, true);
  assert.equal((await status(update)).aiReady, false);
  await download(update);
  assert.equal(store.has(base + '__offline_content__/' + sha('new weights')), true);
  assert.equal((await status(update)).aiReady, true);
});
test('an AI upgrade that cannot be fetched still installs the game and leaves the working version intact', async () => {
  const old = worker(); await old.install(); await download(old);
  const files = { ...bodies, 'index.html': 'new shell', 'model-full.onnx': 'new weights' };
  const update = worker({ files, caches: old.caches, network: name => new Response(name === 'model-full.onnx' ? '' : files[name], { status: name === 'model-full.onnx' ? 503 : 200 }) });
  await update.install();
  assert.equal(update.counts.has('model-full.onnx'), false);
  assert.equal((await status(old)).aiReady, true);
  assert.equal(await (await old.get('model-full.onnx')).text(), bodies['model-full.onnx']);
  await update.activate();
  await assert.rejects(download(update), /Download failed/);
  assert.equal((await status(update)).aiReady, false);
  assert.equal((await status(update)).coreReady, true);
  assert.equal(await (await update.get('index.html')).text(), 'new shell');
});
test('cache eviction is detected and unrelated apps and unknown assets are untouched', async () => {
  const w = worker(); await w.install(); await download(w);
  const other = await w.caches.open('plateloader'); await other.put(base + 'sentinel', new Response('keep'));
  const store = w.caches.stores.get('mahjong-offline-v1:/mahjong/');
  store.delete(base + '__offline_content__/' + sha(bodies['model-full.onnx']));
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
    'tiles/matisse/approved/Man7.svg': '<svg>cut-out</svg>', 'tiles/matisse/placeholders/Haku.svg': '<svg>white dragon</svg>',
    'ort/0123abcd/ort-wasm-simd-threaded.wasm': 'wasm',
    'ort/0123abcd/ort-wasm-simd-threaded.mjs': 'loader', 'ort/0123abcd/memory-budget.mjs': 'memory controls' };
  for (const [path, body] of Object.entries(files)) { await mkdir(join(root, path, '..'), { recursive: true }); await writeFile(join(root, path), body); }
  const options = { network: external('weights'), runtime: 'ort/0123abcd/' };
  await assert.rejects(buildOffline(root, { network: options.network }), /missing the AI runtime/);
  const first = await buildOffline(root, options), second = await buildOffline(root, options);
  assert.deepEqual(first, second);
  assert.equal(first.entries.length, Object.keys(files).length);
  // The trained network itself is not here: it is fetched from its bucket and
  // kept in Cache Storage, so the AI group carries only the runtime that runs it.
  assert.equal(first.entries.filter(e => e.group === 'ai').length, 3);
  assert.equal(first.hasModel, true);
  assert.equal(first.entries.some(e => e.url.endsWith('.onnx')), false);
  // Feed the generated inventory to the worker, not a hand-written approximation.
  const generated = worker({ files, config: first });
  await generated.install();
  assert.equal(generated.counts.has('ort/0123abcd/ort-wasm-simd-threaded.wasm'), false);
  await download(generated);
  assert.equal((await status(generated)).aiReady, false, 'the runtime alone is not the trained AI');
  assert.equal(await pageSaves(generated, 'weights'), 1);
  assert.equal(generated.counts.has(externalUrl(first.network)), false, 'the worker never fetches the network');
  assert.equal((await status(generated)).aiReady, true);
  assert.equal(first.entries.filter(e => e.url.startsWith('tiles/matisse/') && e.group === 'core').length, 2);
  assert.ok((await readFile(join(root, 'sw.js'), 'utf8')).includes(JSON.stringify(first)));
});

function external(body, generation = 1) {
  return { origin: 'https://model.invalid', object: `models/g${generation}/${sha(body)}`,
    sha256: sha(body), bytes: Buffer.byteLength(body) };
}
const externalUrl = model => `${model.origin}/${model.object}`;
const runtimeOnly = Object.fromEntries(Object.entries(bodies).filter(([name]) => !name.endsWith('.onnx')));
function externalWorker(body, { files = runtimeOnly, generation = 1, ...options } = {}) {
  return worker({ files, remote: body, config: { ...manifest(files), network: external(body, generation) }, ...options });
}
// Fails rather than hangs when a lifecycle event waits on the network.
function within(promise, ms = 2000) {
  let timer;
  return Promise.race([promise, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('The event waited on the network')), ms);
  })]).finally(() => clearTimeout(timer));
}

test('an upgrade whose network is slow or unreachable installs and activates with the game alone', async () => {
  const old = externalWorker('old weights'); await old.install(); await prepareAi(old, 'old weights');
  const files = { ...runtimeOnly, 'index.html': 'new app, new model' };
  const model = external('new weights', 2);
  // A network that never finishes: an install or activation that waited on
  // it would be abandoned by the browser, and the update never installed.
  const update = externalWorker('new weights', { generation: 2, files, caches: old.caches,
    network: name => name === externalUrl(model) ? new Response(new ReadableStream({ pull() {} }))
      : new Response(files[name] ?? '', { status: name in files ? 200 : 404 }) });
  await within(update.install());
  assert.equal(update.counts.has(externalUrl(model)), false);
  assert.equal((await status(old)).aiReady, true, 'the running version keeps its trained AI');
  assert.equal(await (await old.get('index.html')).text(), runtimeOnly['index.html']);
  await within(update.activate());
  assert.equal(update.counts.has(externalUrl(model)), false);
  // The old network is still stored, but it is not this version's.
  const after = await status(update);
  assert.equal(after.coreReady, true);
  assert.equal(after.aiReady, false);
  assert.equal(after.aiRequested, true);
  assert.equal(await (await update.get('index.html')).text(), 'new app, new model');
});

test('the page saves a new version\'s network after activation, and it then plays on a plane', async () => {
  const old = externalWorker('old weights'); await old.install(); await prepareAi(old, 'old weights');
  const files = { ...runtimeOnly, 'index.html': 'new app, new model' };
  const update = externalWorker('new weights', { generation: 2, files, caches: old.caches });
  await update.install(); await update.activate();
  assert.equal((await status(update)).aiReady, false);
  assert.equal(await prepareAi(update, 'new weights'), 1);
  assert.equal((await status(update)).aiReady, true);
  const cold = externalWorker('new weights', { generation: 2, files, caches: old.caches,
    network: () => { throw new Error('offline'); } });
  assert.equal((await status(cold)).aiReady, true);
  assert.equal(await (await cold.get('index.html')).text(), 'new app, new model');
  assert.equal(cold.counts.size, 0);
});

test('UI-only remote-model upgrades never redownload verified weights', async () => {
  const old = externalWorker('weights'); await old.install(); await prepareAi(old, 'weights');
  const update = externalWorker('weights', { files: { ...runtimeOnly, 'app.js': 'new UI only' }, caches: old.caches });
  await update.install(); await update.activate();
  assert.equal((await status(update)).aiReady, true);
  assert.deepEqual([...update.counts.keys()], ['app.js']);
});

test('remote cache tampering is not offline readiness, and a failed save keeps the old version', async () => {
  const old = externalWorker('weights'); await old.install(); await prepareAi(old, 'weights');
  const cache = await old.caches.open('mahjong-network-v1');
  await cache.put(externalUrl(external('other!!', 2)), new Response('wrong!!'));
  const update = externalWorker('other!!', { generation: 2, caches: old.caches });
  await update.install();
  assert.equal((await status(update)).aiReady, false, 'an entry without a verified record is not saved');
  await download(update);
  await assert.rejects(pageSaves(update, '', () => new Response('', { status: 503 })), /could not be fetched/);
  assert.equal((await status(update)).aiReady, false);
  assert.equal(await (await old.get('index.html')).text(), runtimeOnly['index.html']);
  assert.equal((await status(old)).aiReady, true);
});

test('the worker never fetches the trained network, even for a player who asked for it', async () => {
  const w = externalWorker('weights');
  await w.install();
  await download(w);
  assert.equal((await status(w)).aiRequested, true);
  assert.equal((await status(w)).aiReady, false);
  const again = externalWorker('weights', { files: { ...runtimeOnly, 'app.js': 'next UI' }, caches: w.caches });
  await again.install(); await again.activate(); await download(again);
  for (const each of [w, again]) assert.equal(each.counts.has(externalUrl(external('weights'))), false);
  assert.equal(await pageSaves(again, 'weights'), 1);
  assert.equal((await status(again)).aiReady, true);
});
