/** A packed network (src/lib/network-pack.js) end to end, before any is
 * listed: the default network packed here, served by the model worker's own
 * code (workers/model-cdn) from a stand-in bucket on a second loopback
 * origin, and listed by a build of the site made for this check, its
 * manifest and service worker naming the packed network there and nothing
 * else changed. The page's first download is cut twice mid-body, and must
 * ask for the rest each time, by a byte range the worker answers, store the
 * packed bytes under their own digest, and play: the opponents' worker reads
 * the stored packed file and widens it before the runtime opens it. A page
 * opened again with the bucket refusing plays from what is stored.
 *
 * The float32 network is MAHJONG_NETWORK's where that names one (the
 * published file, checked against the manifest), else downloaded from the
 * bucket as the page does and checked the same way. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { offlineBuild } from './offline-build.mjs';
import { manifestSource, packedNetwork } from './publish-model-r2.mjs';
import { MANIFEST, NETWORKS } from '../src/lib/model-manifest.js';
import { verifiedNetworkBytes } from '../src/lib/network-transfer.js';
import { unpackNetwork } from '../src/lib/network-pack.js';
import worker from '../../workers/model-cdn/src/index.js';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';

const web = fileURLToPath(new URL('../', import.meta.url)), output = resolve(web, 'test-results');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

/** The default network's float32 file, checked against what the manifest
 * lists: its digest, or, where it is listed packed already, its packed one. */
async function float32Network() {
  let bytes;
  if (process.env.MAHJONG_NETWORK) bytes = new Uint8Array(await readFile(process.env.MAHJONG_NETWORK));
  else {
    bytes = await verifiedNetworkBytes({ url: `${MANIFEST.origin}/${MANIFEST.object}`, expect: MANIFEST,
      timeouts: { totalMs: 600000, idleMs: 60000 } });
    if (MANIFEST.packing) bytes = unpackNetwork(bytes, MANIFEST.bytes);
  }
  const packed = packedNetwork(bytes).packed;
  assert.equal(MANIFEST.packing ? sha(packed) : sha(bytes), MANIFEST.sha256, 'the network packed here is the listed default');
  // Only the packed file is kept: the build beside it wants the memory.
  return { size: bytes.length, packed };
}

/** The model worker's code over a bucket holding `packed` under `key`,
 * stored as a packed network is, with no content encoding. The first GETs of
 * the object are cut after the absolute offsets in `cuts`, a connection
 * dropped mid-body. `asked` records every request; `refuse` answers 503. */
function modelHost(packed, key) {
  const asked = [], cuts = [], state = { refuse: false, served: 0 };
  const uploaded = new Date('2026-11-23T00:00:00Z');
  const metadata = () => ({ httpEtag: `"${sha(packed)}"`, uploaded, size: packed.length,
    httpMetadata: { contentType: 'application/octet-stream' },
    writeHttpMetadata(headers) { headers.set('Content-Type', 'application/octet-stream'); } });
  const env = { MODELS: {
    async head(name) { return name === key ? metadata() : null; },
    async get(name, options) {
      if (name !== key) return null;
      const range = options?.range, body = range ? packed.subarray(range.offset, range.offset + range.length) : packed;
      let at = 0;
      return { ...metadata(), body: new ReadableStream({ pull(controller) {
        if (at >= body.length) { controller.close(); return; }
        controller.enqueue(body.slice(at, at + 262144)); at += 262144;
      } }) };
    },
  } };
  const server = createServer(async (request, response) => {
    const path = new URL(request.url, 'http://host.invalid').pathname;
    asked.push({ method: request.method, path, range: request.headers.range ?? null });
    if (state.refuse) { response.writeHead(503, { 'Access-Control-Allow-Origin': request.headers.origin ?? '*' }).end(); return; }
    const headers = new Headers();
    for (const name of ['origin', 'range', 'if-range', 'if-match', 'if-none-match', 'if-modified-since', 'if-unmodified-since',
      'access-control-request-method', 'access-control-request-headers']) {
      if (request.headers[name] !== undefined) headers.set(name, request.headers[name]);
    }
    const answer = await worker.fetch(new Request(`http://${request.headers.host}${request.url}`, { method: request.method, headers }), env);
    const sent = Object.fromEntries(answer.headers);
    // Where the body starts in the object, and how long it is, as Cloudflare
    // says in Content-Length.
    const part = /^bytes (\d+)-(\d+)\//.exec(sent['content-range'] ?? '');
    const start = part ? Number(part[1]) : 0, end = part ? Number(part[2]) + 1 : packed.length;
    if (answer.body) sent['content-length'] = String(end - start);
    response.writeHead(answer.status, sent);
    if (!answer.body || request.method === 'HEAD') { response.end(); return; }
    const cut = path === `/${key}` ? cuts.shift() : undefined;
    let at = start;
    for await (const chunk of answer.body) {
      const room = cut === undefined ? chunk.length : Math.max(0, cut - at);
      const piece = chunk.subarray(0, room);
      if (piece.length) { response.write(piece); state.served += piece.length; at += piece.length; }
      if (piece.length < chunk.length) {
        // The connection drops with this much sent.
        await new Promise(done => response.write('', done));
        response.socket.destroy();
        return;
      }
    }
    response.end();
  });
  return { server, asked, cuts, state };
}

/** A build of the site that lists `networks`, in `outDir`. Only the
 * manifest module, in the page and in its workers, and the offline build's
 * list differ from the site's own build. */
async function buildListing(networks, outDir, manifestFile) {
  const { build } = await import('vite');
  await writeFile(manifestFile, manifestSource(networks));
  const configure = (await import(pathToFileURL(resolve(web, 'vite.config.js')).href)).default;
  const config = await (typeof configure === 'function' ? configure({ command: 'build', mode: 'production' }) : configure);
  const listing = { name: 'packed-network-check-manifest', enforce: 'pre',
    resolveId: source => (/(^|[/\\])model-manifest\.js$/.test(source) ? manifestFile : null) };
  await build({ ...config, configFile: false, root: web, logLevel: 'warn',
    plugins: [listing, ...config.plugins.flat(Infinity).map(plugin => (plugin?.name === 'mahjong-offline' ? offlineBuild({ networks }) : plugin))],
    worker: { ...config.worker, plugins: () => [listing] },
    build: { ...config.build, outDir, emptyOutDir: true } });
}

const fixture = (() => {
  const session = new MatchSession(Game, 81, 'neural');
  try { session.advance(false); return session.snapshot(); } finally { session.dispose(); }
})();
const { check, report } = browserChecks();
const scratch = await mkdtemp(join(tmpdir(), 'mahjong-packed-network-'));
const dist = join(scratch, 'dist');
let browser, page, site, host;

async function open(context, base) {
  const opened = await context.newPage();
  opened.errors = [];
  opened.on('pageerror', error => opened.errors.push(error.message));
  await opened.setViewport({ width: 1100, height: 900 });
  await opened.setCacheEnabled(false);
  await opened.evaluateOnNewDocument((saveKey, settingsKey, snapshot) => {
    if (!localStorage.getItem(saveKey)) localStorage.setItem(saveKey, JSON.stringify(snapshot));
    if (!localStorage.getItem(settingsKey)) localStorage.setItem(settingsKey, JSON.stringify({ version: 1, difficulty: 'neural',
      opponents: snapshot.opponents, hints: true, confirmDiscards: false, shortcuts: true }));
  }, SAVE_KEY, SETTINGS_KEY, fixture);
  await opened.goto(`${base}?opponents=neural`, { waitUntil: 'domcontentloaded' });
  await opened.waitForSelector('.hand', { timeout: 120000 });
  return opened;
}
const saved = target => target.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
async function play(target, moves) {
  for (let move = 0; move < moves; move++) {
    await target.waitForFunction(() => document.querySelector('.failure, .screen, .standings, .hand button.tile:not(:disabled), .call-options button:not(:disabled)'), { timeout: 120000 });
    assert.equal(await target.$('.failure'), null, 'the trained opponents answer');
    const acted = await target.evaluate(() => {
      const next = document.querySelector('.screen .primary:not([disabled])');
      if (next) { next.click(); return true; }
      const call = document.querySelector('.call-options button[data-choice="ron"], .call-options button[data-choice="tsumo"], .call-options button[data-choice="pass"]');
      if (call) { call.click(); return true; }
      const tile = document.querySelector('.hand button.tile:not([disabled])');
      if (tile) { tile.click(); return true; }
      return false;
    });
    if (!acted) break;
    await new Promise(done => setTimeout(done, 200));
  }
  await target.waitForFunction(() => document.querySelector('.failure, .screen, .standings, .hand button.tile:not(:disabled), .call-options button:not(:disabled)'), { timeout: 120000 });
  assert.equal(await target.$('.failure'), null);
}
const opponentMoves = async target => (await saved(target)).commands.filter(command => command.type === 'opponent').length;

try {
  const { size, packed } = await float32Network();
  const packedSha = sha(packed), key = `models/g${MANIFEST.generation}/${packedSha}`;
  host = modelHost(packed, key);
  await new Promise(done => host.server.listen(0, '127.0.0.1', done));
  const hostOrigin = `http://127.0.0.1:${host.server.address().port}`;
  const listed = { ...MANIFEST, storedBytes: packed.length, packing: 'half', origin: hostOrigin, object: key, sha256: packedSha };
  console.log(`Packed ${size} bytes to ${packed.length}, SHA-256 ${packedSha}; building a site that lists it at ${hostOrigin}`);
  await buildListing([listed, ...NETWORKS.slice(1)], dist, join(scratch, 'model-manifest.js'));
  site = createServer(createFixtureHandler({ root: dist }));
  await new Promise(done => site.listen(0, '127.0.0.1', done));
  const base = `http://127.0.0.1:${site.address().port}/mahjong/`;
  browser = await launchChrome();
  await mkdir(output, { recursive: true });
  // One context for both cases, the second opening the page the first saved
  // the network for; it goes with the browser.
  const context = await browser.createBrowserContext();

  await check('a packed network cut twice mid-download asks for the rest each time, is stored packed, and plays widened', async () => {
    host.cuts.push(Math.floor(packed.length * 0.3), Math.floor(packed.length * 0.7));
    page = await open(context, base);
    await page.waitForFunction(() => /game \+ AI ready/.test(document.querySelector('[data-offline-status]')?.textContent), { timeout: 300000 });
    const gets = host.asked.filter(request => request.method === 'GET' && request.path === `/${key}`);
    console.log(`The model host was asked: ${host.asked.map(request => `${request.method} ${request.range ?? 'whole'}`).join(', ')};`
      + ` it sent ${host.state.served} bytes of ${packed.length}`);
    assert.equal(gets.length, 3, JSON.stringify(host.asked));
    assert.equal(gets[0].range, null);
    const [first, second] = gets.slice(1).map(request => Number(/^bytes=(\d+)-$/.exec(request.range)?.[1]));
    assert.ok(first > 0 && first <= packed.length * 0.3, `the first resume asks from what arrived: ${gets[1].range}`);
    assert.ok(second > first && second <= packed.length * 0.7, `and the second from further on: ${gets[2].range}`);
    assert.ok(host.state.served < packed.length * 1.05, `the object was sent about once, not again from the start: ${host.state.served} bytes`);
    // Stored as the packed file, under its record.
    const stored = await page.evaluate(async url => {
      for (const name of await caches.keys()) {
        if (!name.startsWith('mahjong-network-v2:')) continue;
        const held = await (await caches.open(name)).match(url);
        if (held) return { length: Number(held.headers.get('Content-Length')), sha256: held.headers.get('X-Mahjong-SHA256'),
          bytes: (await held.arrayBuffer()).byteLength };
      }
      return null;
    }, `${hostOrigin}/${key}`);
    assert.deepEqual(stored, { length: packed.length, sha256: packedSha, bytes: packed.length });
    const before = await opponentMoves(page);
    await play(page, 8);
    assert.ok(await opponentMoves(page) > before, 'the trained opponents moved, from the widened network');
    assert.deepEqual(page.errors, []);
    await page.screenshot({ path: resolve(output, 'packed-network-resumed.png') });
  });

  await check('a page opened again plays the stored packed network, widened again, and asks the bucket for nothing', async () => {
    assert.ok(page, 'the first case left its page');
    host.state.refuse = true;
    const asked = host.asked.length;
    await page.close();
    page = await open(context, base);
    const before = await opponentMoves(page);
    await play(page, 8);
    assert.ok(await opponentMoves(page) > before, 'the trained opponents moved');
    assert.deepEqual(host.asked.slice(asked).filter(request => request.path === `/${key}`), []);
    assert.deepEqual(page.errors, []);
  });
} finally {
  await browser?.close();
  for (const server of [site, host?.server]) if (server?.listening) await new Promise(done => server.close(done));
  await rm(scratch, { recursive: true, force: true });
  await report('packed-network browser checks', resolve(output, 'packed-network-report.json'));
}
