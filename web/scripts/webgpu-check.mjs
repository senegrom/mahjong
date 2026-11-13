/** A review's batches on the graphics card, as the page brings the card up
 * (gpu-batches.js), with the production workers of the built site, on real
 * positions: the decisions of the review check's game and of a game the
 * Club adviser played, written as Mortal's planes by the engine, asked of a
 * network exported without Softplus (neural/export.py, ExportMish).
 *
 * The WebAssembly worker answers a first batch, which loads the network,
 * then the sample, a batch of BATCH positions, timed. The card's worker
 * opens the network from this page's Cache Storage, runs the timed batch,
 * the sample's positions again and again to a hand's worth (timedOrder),
 * once to warm up and TIMED_RUNS times more, and must pass the page's own
 * gates on it: the same answers (agreement), at least twice as fast as
 * WebAssembly runs as many positions (fastEnough).
 * Then every position is asked on both, the card in batches of GPU_BATCH
 * and WebAssembly in batches of BATCH, and the check says the largest
 * difference in a weight, the share of decisions with the same best move,
 * and how long a batch takes on each.
 *
 * It runs only where Chrome gives a real WebGPU adapter: elsewhere, as in CI,
 * whose Chrome has none, it says so and passes. The network is the file
 * MAHJONG_WEBGPU_NETWORK names, else MAHJONG_NETWORK's; one whose graph does
 * not run whole on WebGPU is not one the page would send to the card, and is
 * skipped too.
 *   MAHJONG_WEBGPU_NETWORK=<network.onnx> node scripts/webgpu-check.mjs
 * Build the site first: npm run build. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { runtimeDirectory, webgpuDirectory } from './copy-runtime.mjs';
import { graphOperators, runsOnWebgpu } from './publish-model-r2.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { reviewTable } from '../src/lib/trained-move.js';
import { GPU_BATCH, BATCH } from '../src/lib/game-review.js';
import { TIMED_RUNS, WEIGHT_BOUND, agreement, fastEnough, median, softwareAdapter, timedOrder } from '../src/lib/webgpu.js';
import { networkCacheName } from '../src/lib/network-cache.js';
import { clubAdvisedGame, eachDecision, reviewCheckGame } from '../tests/fixtures/review-check-game.js';

/** The fewest decisions that must keep their best move on the card: its
 * arithmetic differs from the processor's in the last bits, so a near tie
 * may flip; anything more is a different network. */
const SAME_BEST_AT_LEAST = 0.97;
const POSITIONS = 34;

const web = fileURLToPath(new URL('../', import.meta.url)), dist = resolve(web, 'dist');
const runtime = await runtimeDirectory(web), webgpu = await webgpuDirectory(web);
assert.ok(existsSync(resolve(dist, runtime, 'ort-wasm-simd-threaded.wasm')) && existsSync(resolve(dist, webgpu, 'ort.min.mjs')),
  'build the site first: npm run build');
const chunk = pattern => readdirSync(resolve(dist, 'assets')).find(name => pattern.test(name));
const playWorker = chunk(/^policy\.worker-[^/]+\.js$/), cardWorker = chunk(/^gpu-batch\.worker-[^/]+\.js$/);
assert.ok(playWorker && cardWorker, 'the build has both workers');

// Chrome runs WebGPU headless only when told it may, and on Windows through
// Direct3D 11.
const args = ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist',
  ...(process.platform === 'win32' ? ['--use-angle=d3d11'] : [])];
const PAGE = '<!doctype html><meta charset="utf-8"><title>WebGPU check</title><body></body>';
let network = null, planes = null, masks = null, wasmFetches = 0;
const server = createServer(createFixtureHandler({ root: dist, intercept: name => {
  // This page has no service worker: every fetch of the runtime's WebAssembly reaches the server.
  if (name === `${webgpu}ort-wasm-simd-threaded.jsep.wasm`) wasmFetches++;
  if (name === 'webgpu-check.html') return { body: PAGE };
  if (name === 'webgpu-check/network.onnx') return { body: network };
  if (name === 'webgpu-check/planes.bin') return { body: planes };
  if (name === 'webgpu-check/masks.bin') return { body: masks };
  return undefined;
} }));
const { check, openContext, report } = browserChecks();
let browser;

/** The positions of every decision of the two games, as the review asks
 * them: Mortal's planes end to end, and each one's mask of legal moves. */
async function positions() {
  await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
  const rows = [];
  for (const snapshot of [reviewCheckGame(Game), clubAdvisedGame(Game, 3)]) {
    eachDecision(Game, snapshot, ({ engine, index }) => {
      const { planes: written, mask } = reviewTable(engine, index);
      rows.push({ planes: Float32Array.from(written), mask: Uint8Array.from(mask, open => (open ? 1 : 0)) });
    });
  }
  return rows;
}

try {
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  const base = `http://127.0.0.1:${server.address().port}/mahjong/`;
  browser = await launchChrome({ args });
  const context = await openContext(browser);
  const page = await context.newPage();
  await page.goto(`${base}webgpu-check.html`, { waitUntil: 'load' });
  const adapter = await page.evaluate(async () => {
    const found = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!found) return null;
    const { vendor = '', architecture = '', device = '', description = '' } = found.info ?? {};
    return { fallback: Boolean(found.info?.isFallbackAdapter || found.isFallbackAdapter), info: { vendor, architecture, device, description },
      name: [vendor, architecture, device, description].filter(Boolean).join(' ') };
  });
  const path = process.env.MAHJONG_WEBGPU_NETWORK || process.env.MAHJONG_NETWORK;
  const skip = !adapter ? 'this Chrome gives no WebGPU adapter'
    : adapter.fallback || softwareAdapter(adapter.info)
      ? `this Chrome's only WebGPU adapter (${adapter.name || 'unnamed'}) runs on the processor`
      : !path ? 'set MAHJONG_WEBGPU_NETWORK to a network exported without Softplus' : null;
  if (skip) console.log(`SKIP WebGPU against WebAssembly: ${skip}.`);
  else {
    network = readFileSync(path);
    if (!runsOnWebgpu(graphOperators(network))) {
      console.log(`SKIP WebGPU against WebAssembly: ${path} does not run whole on WebGPU; export it again with neural/export.py.`);
    } else {
      const rows = await positions();
      const width = rows[0].planes.length;
      const laid = new Float32Array(rows.length * width), legal = new Uint8Array(rows.length * rows[0].mask.length);
      rows.forEach((row, at) => { laid.set(row.planes, at * width); legal.set(row.mask, at * row.mask.length); });
      planes = Buffer.from(laid.buffer);
      masks = Buffer.from(legal.buffer);
      const expect = { bytes: network.length, sha256: createHash('sha256').update(network).digest('hex') };
      console.log(`Adapter: ${adapter.name || 'unnamed'}; ${rows.length} positions of ${width / POSITIONS} planes; network ${path}.`);
      await check('the card is brought up on a batch WebAssembly answered, passes the page\'s gates, and answers every batch as WebAssembly does', async () => {
        const result = await page.evaluate(async ({ playWorker, cardWorker, runtimeBase, webgpuBase, cacheName, expect, width, gpuBatch, wasmBatch, runs, order }) => {
          const fetched = async name => new Uint8Array(await (await fetch(`webgpu-check/${name}`)).arrayBuffer());
          const allPlanes = new Float32Array((await fetched('planes.bin')).buffer), allMasks = await fetched('masks.bin');
          const count = allPlanes.length / width, moves = allMasks.length / count;
          const url = new URL('webgpu-check/network.onnx', location.href).href, scope = new URL('./', location.href).href;
          const batch = (start, size) => ({ count: size, planes: allPlanes.slice(start * width, (start + size) * width),
            mask: Array.from(allMasks.subarray(start * moves, (start + size) * moves), allowed => allowed === 1) });
          /** A worker of the build, `ask(message)` answering a request as the
           * page reads it, and how long it took from sending to answer. */
          function open(file, onReady = () => {}) {
            const thread = new Worker(new URL(`assets/${file}`, location.href), { type: 'module' });
            const pending = new Map();
            let id = 0;
            thread.onmessage = ({ data }) => {
              if (data.ready) onReady();
              if (data.progress || data.id == null) return;
              const done = pending.get(data.id);
              pending.delete(data.id);
              if (data.error) done.reject(new Error(data.error));
              else done.resolve(data);
            };
            const ask = (message, transfer = []) => new Promise((resolve, reject) => {
              const began = performance.now();
              const sent = { id: ++id, ...message };
              pending.set(sent.id, { resolve: data => resolve({ ...data, ms: performance.now() - began }), reject });
              thread.postMessage(sent, transfer);
            });
            return { ask, close: () => thread.terminate() };
          }
          // The network saved in this page's Cache Storage, as the page saves
          // it, which is the only place the card's worker reads it from.
          const bytes = await fetched('network.onnx');
          await (await caches.open(cacheName)).put(url, new Response(bytes.slice(), { headers: { 'Content-Type': 'application/octet-stream',
            'Content-Length': String(bytes.length), 'X-Mahjong-SHA256': expect.sha256 } }));
          const wasm = open(playWorker);
          const onWasm = (start, size, extra = {}) => wasm.ask({ url, runtimeBase, scope, details: true, ...batch(start, size), ...extra },
            extra.network ? [extra.network.buffer] : []);
          const card = open(cardWorker);
          try {
            // A first batch, which loads the network, then the sample.
            await onWasm(0, wasmBatch, { network: bytes });
            const sample = await onWasm(wasmBatch, wasmBatch);
            const opened = await card.ask({ open: { url, expect, scope, webgpu: webgpuBase } }).catch(error => ({ refused: error.message }));
            if (opened.refused) return { refused: opened.refused };
            // The timed batch: the sample's positions in the page's own order (timedOrder).
            const timed = { count: order.length, planes: new Float32Array(order.length * width), mask: [] };
            order.forEach((index, at) => {
              const one = batch(wasmBatch + index, 1);
              timed.planes.set(one.planes, at * width);
              timed.mask.push(...one.mask);
            });
            const sampleRuns = [];
            for (let run = 0; run <= runs; run++) sampleRuns.push(await card.ask(timed));
            // Every position, on both.
            const all = async (ask, size) => {
              const answers = [], times = [];
              for (let start = 0; start < count; start += size) {
                const { analyses, ms } = await ask(start, Math.min(size, count - start));
                answers.push(...analyses);
                if (start + size <= count) times.push(ms);
              }
              return { answers, times };
            };
            const plain = await all(onWasm, wasmBatch);
            const onCard = await all((start, size) => card.ask(batch(start, size)), gpuBatch);
            const choices = Array.from({ length: count }, (_, row) => allMasks.subarray(row * moves, (row + 1) * moves).reduce((a, b) => a + b, 0) > 1);
            return { count, choices, sample: { answers: sample.analyses, ms: sample.ms, runs: sampleRuns.map(({ analyses, ms }) => ({ analyses, ms })) },
              plain, onCard };
          } finally { wasm.close(); card.close(); }
        }, { playWorker, cardWorker, runtimeBase: `${base}${runtime}`, webgpuBase: `${base}${webgpu}`, cacheName: networkCacheName(base),
          expect, width, gpuBatch: GPU_BATCH, wasmBatch: BATCH, runs: TIMED_RUNS, order: timedOrder(BATCH) });
        if (result.refused) assert.fail(`the card's worker would not open the network: ${result.refused}`);
        assert.equal(wasmFetches, 1, 'the runtime\'s 30 MB of WebAssembly are fetched once, and handed to the runtime');
        // The page's own gates on the timed batch.
        const { sample } = result;
        // WebAssembly's time for as many positions, at the sample's time a position.
        const order = timedOrder(BATCH), wasmTimedMs = sample.ms / BATCH * order.length;
        const differs = sample.runs.map(run => agreement(order.map(index => sample.answers[index]), run.analyses)).find(Boolean) ?? null;
        const sampleMs = median(sample.runs.slice(1).map(run => run.ms));
        console.log(`Sample of ${BATCH}: ${sample.ms.toFixed(1)} ms on WebAssembly, which makes ${wasmTimedMs.toFixed(1)} ms for ${order.length}; `
          + `on the card ${order.length} took ${sample.runs[0].ms.toFixed(1)} ms to warm up, `
          + `then a median of ${sampleMs.toFixed(1)} ms over ${TIMED_RUNS} runs; ${differs ? `disagrees: ${differs}` : 'agrees'}.`);
        assert.equal(differs, null, 'the card agrees with WebAssembly on the timed batch');
        assert.ok(fastEnough(sampleMs, wasmTimedMs), 'the card is at least twice as fast on the timed batch');
        // Every position.
        let largest = 0, same = 0, choices = 0;
        result.plain.answers.forEach((plain, row) => {
          const card = result.onCard.answers[row];
          plain.weights.forEach((weight, move) => { largest = Math.max(largest, Math.abs(card.weights[move] - weight)); });
          if (result.choices[row]) { choices++; if (card.action === plain.action) same++; }
        });
        const share = same / choices, cardMs = median(result.onCard.times), wasmMs = median(result.plain.times);
        console.log(`Same best move on ${same} of ${choices} decisions with a choice (${(100 * share).toFixed(2)}%), of ${result.count} positions.`);
        console.log(`Largest difference in a weight: ${largest.toExponential(3)} (the page's bound ${WEIGHT_BOUND}).`);
        console.log(`A batch: ${cardMs.toFixed(1)} ms for ${GPU_BATCH} positions on the card (${(GPU_BATCH * 1000 / cardMs).toFixed(0)} a second), `
          + `${wasmMs.toFixed(1)} ms for ${BATCH} on WebAssembly (${(BATCH * 1000 / wasmMs).toFixed(1)} a second).`);
        assert.ok(share >= SAME_BEST_AT_LEAST, `the same best move on ${(100 * share).toFixed(2)}% of decisions`);
        assert.ok(largest <= WEIGHT_BOUND, `a weight differs by ${largest}`);
      });
    }
  }
} finally {
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
  await report('WebGPU browser checks');
}
