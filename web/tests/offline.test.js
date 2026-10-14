import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { createHash, webcrypto } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setImmediate } from 'node:timers/promises';
import vm from 'node:vm';
import { buildOffline, serviceWorkerTemplate } from '../scripts/offline-build.mjs';
import { networkCacheName, verifiedNetworkBytes } from '../src/lib/network-transfer.js';

const template = await serviceWorkerTemplate();
const sha = body => createHash('sha256').update(body).digest('hex');
const base = 'https://test.invalid/mahjong/';
const bodies = { 'index.html': '<html>Game</html>', 'app.js': 'game', 'tiles/Haku.svg': '<svg>dragon</svg>',
  'ort/runtime.wasm': 'wasm', 'ort/runtime.mjs': 'runtime' };
/** An inventory shaped as buildOffline writes one: the AI group is the
 * runtime alone, and the network lives in its bucket, kept by the page in
 * the installation's scoped cache. */
const manifest = (files = bodies, network = external('weights')) => ({ version: sha(JSON.stringify(files)),
  hasModel: true, networkCacheVersion: 2, network,
  entries: Object.entries(files).map(([url, body]) => ({ url, hash: sha(body), bytes: Buffer.byteLength(body),
    group: url.startsWith('ort/') ? 'ai' : 'core' })) });
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
/** Time that passes only when a test moves it, for every timer the worker
 * could use: a fixed cap on a download, AbortSignal.timeout included, would
 * end a slow transfer here as surely as it does in a browser. */
function testClock() {
  let now = 0, last = 0;
  const timers = new Map();
  const setTimeout = (fn, ms = 0) => { timers.set(++last, { at: now + ms, fn }); return last; };
  const clearTimeout = id => { timers.delete(id); };
  return { setTimeout, clearTimeout, now: () => now,
    AbortSignal: { timeout(ms) {
      const controller = new AbortController();
      setTimeout(() => controller.abort(new DOMException('The operation timed out.', 'TimeoutError')), ms);
      return controller.signal;
    } },
    advance(ms) {
      const end = now + ms;
      for (;;) {
        const [id, due] = [...timers].filter(([, timer]) => timer.at <= end).sort(([, a], [, b]) => a.at - b.at)[0] ?? [];
        if (!due) break;
        timers.delete(id); now = due.at; due.fn();
      }
      now = end;
    } };
}
/** Settles work while the clock moves on a second at a time, for as long as
 * it takes, and says how it settled. */
async function overTime(clock, work, seconds = 600) {
  let outcome;
  work.then(value => { outcome = { value }; }, error => { outcome = { error }; });
  for (let second = 0; second < seconds && !outcome; second++) {
    for (let turn = 0; turn < 10 && !outcome; turn++) await setImmediate();
    if (!outcome) clock.advance(1000);
  }
  assert.ok(outcome, `still running after ${seconds} s`);
  return outcome;
}
/** A body that arrives a byte at a time, one every `every` ms of the clock,
 * and fails, as a real one does, when its request is aborted. */
function trickle(body, clock, signal, every = 10000) {
  const bytes = new TextEncoder().encode(body);
  let sent = 0;
  return new Response(new ReadableStream({
    start(controller) {
      signal.addEventListener('abort', () => { try { controller.error(signal.reason); } catch { /* Closed. */ } });
    },
    pull(controller) {
      return new Promise(resolve => clock.setTimeout(() => {
        try {
          controller.enqueue(bytes.slice(sent, ++sent));
          if (sent === bytes.length) controller.close();
        } catch { /* Aborted meanwhile. */ }
        resolve();
      }, every));
    },
  }));
}
/** The worker's SHA-256, worked out at once. WebCrypto's digest finishes on the
 * thread pool, in real time, while the test clock moves on every ten turns: on
 * a busy machine a minute of test time could pass before a download's next
 * read, and a healthy transfer was then declared stalled. Worked out here, the
 * same digest settles in a microtask, so every wait in the worker is on the
 * test's clock. */
const workerCrypto = {
  getRandomValues: array => webcrypto.getRandomValues(array),
  subtle: { async digest(algorithm, data) {
    assert.equal(algorithm?.name ?? algorithm, 'SHA-256');
    const bytes = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
    const hash = createHash('sha256').update(bytes).digest();
    return hash.buffer.slice(hash.byteOffset, hash.byteOffset + hash.byteLength);
  } },
};
function worker({ files = bodies, caches = storage(), config = manifest(files), network, remote = 'weights', clock } = {}) {
  const events = {}, counts = new Map();
  const self = { registration: { scope: base }, clients: { async claim() {} }, addEventListener: (type, fn) => events[type] = fn };
  vm.runInNewContext(template.replace('/* OFFLINE_CONFIG */ null', JSON.stringify(config)), {
    self, URL, Request, Response, Map, Set, caches, crypto: workerCrypto, AbortController, DOMException,
    AbortSignal: clock?.AbortSignal ?? AbortSignal,
    setTimeout: clock?.setTimeout ?? setTimeout, clearTimeout: clock?.clearTimeout ?? clearTimeout,
    fetch: async (url, init) => {
      const address = new URL(url);
      const name = address.origin === new URL(base).origin ? address.pathname.slice('/mahjong/'.length) : address.href;
      counts.set(name, (counts.get(name) ?? 0) + 1);
      if (network) return network(name, init);
      if (name === externalUrl(config.network)) return new Response(remote);
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
    await verifiedNetworkBytes({ url: externalUrl(network), expect: network, requireStored: true, scope: base });
  } finally { globalThis.caches = previous.caches; globalThis.fetch = previous.fetch; }
  await w.message('MAHJONG_NETWORK_SAVED');
  return fetched;
}
const prepareAi = async (w, body) => { await download(w); return pageSaves(w, body); };

test('offline install saves the shell and all graphics, with AI explicitly incomplete', async () => {
  const w = worker(); await w.install();
  assert.equal((await status(w)).coreReady, true);
  assert.equal((await status(w)).aiReady, false);
  assert.deepEqual([...w.counts.keys()].sort(), ['app.js', 'index.html', 'tiles/Haku.svg'], 'no runtime, no network');
  assert.equal(await (await w.get('?opponents=neural')).text(), bodies['index.html']);
  assert.equal(await (await w.get('tiles/Haku.svg')).text(), bodies['tiles/Haku.svg']);
});
test('all AI bytes are durable before ready; simultaneous requests share downloads', async () => {
  const files = { ...bodies, 'ort/duplicate.wasm': bodies['ort/runtime.wasm'] };
  const w = worker({ files }); await w.install();
  await Promise.all([download(w), download(w), download(w)]);
  const sum = [...w.counts].filter(([name]) => name.startsWith('ort/')).reduce((n, [, count]) => n + count, 0);
  assert.equal(sum, 2, 'the runtime module and one copy of the WASM');
  assert.equal((await status(w)).aiReady, false, 'the runtime alone is not the trained AI');
  await pageSaves(w, 'weights');
  assert.equal((await status(w)).aiReady, true);
  await download(w); assert.equal([...w.counts.values()].reduce((a, b) => a + b), 5);
});
test('a cold worker serves navigation, tiles and the runtime without ANY network, including HEAD', async () => {
  const original = worker(); await original.install(); await prepareAi(original, 'weights');
  const cold = worker({ caches: original.caches, network() { throw new Error('plane'); } });
  assert.equal((await status(cold)).aiReady, true);
  for (const [path, body] of Object.entries(bodies)) assert.equal(await (await cold.get(path)).text(), body);
  assert.equal(await (await cold.get('ort/runtime.wasm', 'HEAD')).text(), '');
  assert.equal(await (await cold.get('?launched=home-screen')).text(), bodies['index.html']);
  assert.equal(cold.counts.size, 0);
});
test('truncated or captive-portal downloads never count as AI ready and retry reuses good files', async () => {
  for (const bad of ['', 'portal!']) {
    let corrupt = true;
    const w = worker({ network: name => new Response(corrupt && name === 'ort/runtime.wasm' ? bad : bodies[name]) });
    // The network is stored already, so only the runtime stands between.
    await w.install(); await pageSaves(w, 'weights');
    await assert.rejects(download(w), /Incomplete or outdated/);
    assert.equal((await status(w)).aiReady, false);
    assert.equal((await status(w)).coreReady, true);
    corrupt = false; await download(w);
    assert.equal((await status(w)).aiReady, true);
    assert.equal(w.counts.get('ort/runtime.wasm'), 2);
    assert.equal(w.counts.get('ort/runtime.mjs'), 1);
  }
});
test('an app-only upgrade reuses every tile, the runtime and the stored network', async () => {
  const old = worker(); await old.install(); await prepareAi(old, 'weights');
  const files = { ...bodies, 'index.html': '<html>New game UI</html>' };
  const update = worker({ files, caches: old.caches }); await update.install();
  assert.equal((await status(update)).aiReady, true);
  assert.deepEqual([...update.counts.keys()], ['index.html']);
  assert.equal(await (await old.get('index.html')).text(), bodies['index.html']);
});
test('activation removes obsolete Mahjong bodies but keeps current content and metadata', async () => {
  const old = worker(); await old.install(); await prepareAi(old, 'weights');
  const files = { ...bodies, 'index.html': 'new shell', 'tiles/Haku.svg': '<svg>new dragon</svg>', 'ort/runtime.wasm': 'new wasm' };
  const update = worker({ files, caches: old.caches }); await update.install();
  const store = update.caches.stores.get('mahjong-offline-v1:/mahjong/');
  const oldHashes = [bodies['index.html'], bodies['tiles/Haku.svg'], bodies['ort/runtime.wasm']].map(sha);
  for (const hash of oldHashes) assert.equal(store.has(base + '__offline_content__/' + hash), true);
  assert.equal(update.counts.has('ort/runtime.wasm'), false, 'installation saves the game alone');
  await update.activate();
  for (const hash of oldHashes) assert.equal(store.has(base + '__offline_content__/' + hash), false);
  const core = Object.entries(files).filter(([name]) => !name.startsWith('ort/'));
  for (const [, body] of core) assert.equal(store.has(base + '__offline_content__/' + sha(body)), true);
  assert.equal(store.has(base + '__offline_meta__/ai-requested'), true);
  // The request survives the upgrade, and the new version's AI is saved on it.
  assert.equal((await status(update)).aiRequested, true);
  assert.equal((await status(update)).aiReady, false);
  await download(update);
  assert.equal(store.has(base + '__offline_content__/' + sha('new wasm')), true);
  assert.equal((await status(update)).aiReady, true);
});
test('an AI upgrade that cannot be fetched still installs the game and leaves the working version intact', async () => {
  const old = worker(); await old.install(); await prepareAi(old, 'weights');
  const files = { ...bodies, 'index.html': 'new shell', 'ort/runtime.wasm': 'new wasm' };
  const update = worker({ files, caches: old.caches, network: name => name === 'ort/runtime.wasm'
    ? new Response('', { status: 503 }) : new Response(files[name]) });
  await update.install();
  assert.equal(update.counts.has('ort/runtime.wasm'), false);
  assert.equal((await status(old)).aiReady, true);
  assert.equal(await (await old.get('ort/runtime.wasm')).text(), bodies['ort/runtime.wasm']);
  await update.activate();
  await assert.rejects(download(update), /Download failed/);
  assert.equal((await status(update)).aiReady, false);
  assert.equal((await status(update)).coreReady, true);
  assert.equal(await (await update.get('index.html')).text(), 'new shell');
});
test('cache eviction is detected and unrelated apps and unknown assets are untouched', async () => {
  const w = worker(); await w.install(); await prepareAi(w, 'weights');
  const other = await w.caches.open('plateloader'); await other.put(base + 'sentinel', new Response('keep'));
  const store = w.caches.stores.get('mahjong-offline-v1:/mahjong/');
  assert.equal((await status(w)).aiReady, true);
  store.delete(base + '__offline_content__/' + sha(bodies['ort/runtime.wasm']));
  assert.equal((await status(w)).aiReady, false);
  assert.equal(await (await other.match(base + 'sentinel')).text(), 'keep');
  assert.equal(await w.get('../plateloader/'), undefined);
  assert.equal(await w.get('missing.wasm'), undefined);
});
test('a file that keeps arriving is saved however long it takes; one that stops for a minute ends', async () => {
  // At 10 s a byte the white dragon takes 170 s: longer than the fixed 90 s
  // the worker once allowed, which four 3 MB tiles on a slow link outlast.
  const clock = testClock(), slow = 'tiles/Haku.svg';
  const w = worker({ clock, network: (name, init) => name === slow ? trickle(bodies[name], clock, init.signal)
    : new Response(bodies[name]) });
  assert.equal((await overTime(clock, w.install())).error, undefined);
  assert.ok(clock.now() >= 170000);
  assert.equal((await status(w)).coreReady, true);
  assert.equal(await (await w.get(slow)).text(), bodies[slow]);
  // A body that stops arriving has stalled: the attempt ends a minute after
  // its last byte, not whenever the browser gives up on the event.
  const stalls = testClock();
  const stalled = worker({ clock: stalls, network: (name, init) => name !== slow ? new Response(bodies[name])
    : new Response(new ReadableStream({ start(controller) {
      controller.enqueue(new TextEncoder().encode('<svg>'));
      init.signal.addEventListener('abort', () => controller.error(init.signal.reason));
    } })) });
  assert.match((await overTime(stalls, stalled.install())).error?.message ?? '', /Download stalled: tiles\/Haku\.svg/);
  assert.ok(stalls.now() >= 60000 && stalls.now() <= 62000, `ended at ${stalls.now()} ms`);
  assert.equal((await status(stalled)).coreReady, false);
});
test('a file the page asks for before it is saved is served however slowly it arrives', async () => {
  // The opponent's worker can load the runtime while it is still being saved.
  const clock = testClock(), slow = 'ort/runtime.wasm';
  const w = worker({ clock, network: (name, init) => name === slow ? trickle(bodies[name], clock, init.signal, 30000)
    : new Response(bodies[name]) });
  const served = await overTime(clock, w.get(slow));
  assert.ok(clock.now() >= 120000);
  assert.equal(served.value.status, 200);
  assert.equal(await served.value.text(), bodies[slow]);
});
test('a file that fails costs the attempt no other file, and the next one fetches only what is missing', async () => {
  const clock = testClock(), failing = 'tiles/Haku.svg';
  let broken = true;
  const w = worker({ clock, network: (name, init) => name === failing && broken
    ? new Response('', { status: 503 }) : trickle(bodies[name], clock, init.signal, 1000) });
  const failed = await overTime(clock, w.install());
  assert.match(failed.error?.message ?? '', /Download failed: tiles\/Haku\.svg \(HTTP 503\)/);
  // Read as the install fails: a browser abandons a failed install's worker,
  // and with it any transfer still running.
  const store = w.caches.stores.get('mahjong-offline-v1:/mahjong/');
  const others = w.config.entries.filter(entry => entry.group === 'core' && entry.url !== failing);
  for (const entry of others) assert.ok(store.has(`${base}__offline_content__/${entry.hash}`), entry.url);
  broken = false;
  assert.equal((await overTime(clock, w.install())).error, undefined);
  assert.equal((await status(w)).coreReady, true);
  assert.deepEqual(Object.fromEntries(w.counts), { 'index.html': 1, 'app.js': 1, [failing]: 2 });
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
/** A build that plays the network `body`, published as this generation. */
function externalWorker(body, { files = bodies, generation = 1, ...options } = {}) {
  return worker({ files, remote: body, config: manifest(files, external(body, generation)), ...options });
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
  const files = { ...bodies, 'index.html': 'new app, new model' };
  const model = external('new weights', 2);
  // A network that never finishes: an install or activation that waited on
  // it would be abandoned by the browser, and the update never installed.
  const update = externalWorker('new weights', { generation: 2, files, caches: old.caches,
    network: name => name === externalUrl(model) ? new Response(new ReadableStream({ pull() {} }))
      : new Response(files[name] ?? '', { status: name in files ? 200 : 404 }) });
  await within(update.install());
  assert.equal(update.counts.has(externalUrl(model)), false);
  assert.equal((await status(old)).aiReady, true, 'the running version keeps its trained AI');
  assert.equal(await (await old.get('index.html')).text(), bodies['index.html']);
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
  const files = { ...bodies, 'index.html': 'new app, new model' };
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
  const update = externalWorker('weights', { files: { ...bodies, 'app.js': 'new UI only' }, caches: old.caches });
  await update.install(); await update.activate();
  assert.equal((await status(update)).aiReady, true);
  assert.deepEqual([...update.counts.keys()], ['app.js']);
});

test('remote cache tampering is not offline readiness, and a failed save keeps the old version', async () => {
  const old = externalWorker('weights'); await old.install(); await prepareAi(old, 'weights');
  const cache = await old.caches.open(networkCacheName(base));
  await cache.put(externalUrl(external('other!!', 2)), new Response('wrong!!'));
  const update = externalWorker('other!!', { generation: 2, caches: old.caches });
  await update.install();
  assert.equal((await status(update)).aiReady, false, 'an entry without a verified record is not saved');
  await download(update);
  await assert.rejects(pageSaves(update, '', () => new Response('', { status: 503 })), /could not be fetched/);
  assert.equal((await status(update)).aiReady, false);
  assert.equal(await (await old.get('index.html')).text(), bodies['index.html']);
  assert.equal((await status(old)).aiReady, true);
});

test('the worker never fetches the trained network, even for a player who asked for it', async () => {
  const w = externalWorker('weights');
  await w.install();
  await download(w);
  assert.equal((await status(w)).aiRequested, true);
  assert.equal((await status(w)).aiReady, false);
  const again = externalWorker('weights', { files: { ...bodies, 'app.js': 'next UI' }, caches: w.caches });
  await again.install(); await again.activate(); await download(again);
  for (const each of [w, again]) assert.equal(each.counts.has(externalUrl(external('weights'))), false);
  assert.equal(await pageSaves(again, 'weights'), 1);
  assert.equal((await status(again)).aiReady, true);
});
