/** The production worker on the graphics card against the same worker on
 * WebAssembly, on real positions: the decisions of the review check's game
 * and of a game the Club adviser played, written as Mortal's planes by the
 * engine, asked of a network exported without Softplus (neural/export.py,
 * ExportMish). It says the largest difference in the policy's logits, each
 * taken from the row's best move as WebAssembly has it (all that the weights
 * the worker answers keep, and all that a softmax or a best move reads), the
 * share of rows whose best move is the same, and how long a batch takes on
 * each runtime.
 *
 * It runs only where Chrome gives a real WebGPU adapter: elsewhere, as in CI,
 * whose Chrome has none, it says so and passes. The network is the file
 * MAHJONG_WEBGPU_NETWORK names, else MAHJONG_NETWORK's; one that still holds
 * a Softplus is not one the page would play on the card, and is skipped too.
 *   MAHJONG_WEBGPU_NETWORK=<network.onnx> node scripts/webgpu-check.mjs
 * Build the site first: npm run build. */
import assert from 'node:assert/strict';
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
import { softwareAdapter } from '../src/lib/webgpu.js';
import { clubAdvisedGame, eachDecision, reviewCheckGame } from '../tests/fixtures/review-check-game.js';

/** The fewest rows that must keep their best move, and the largest
 * difference a logit may show, on the graphics card. Its arithmetic differs
 * from the processor's in the last bits, so a near tie may flip; anything
 * more is a different network. */
const SAME_BEST_AT_LEAST = 0.97;
const LOGIT_DIFFERENCE_AT_MOST = 0.05;
const POSITIONS = 34;

const web = fileURLToPath(new URL('../', import.meta.url)), dist = resolve(web, 'dist');
const runtime = await runtimeDirectory(web), webgpu = await webgpuDirectory(web);
assert.ok(existsSync(resolve(dist, runtime, 'ort-wasm-simd-threaded.wasm')) && existsSync(resolve(dist, webgpu, 'ort.min.mjs')),
  'build the site first: npm run build');
const [workerFile] = readdirSync(resolve(dist, 'assets')).filter(name => /^policy\.worker-[^/]+\.js$/.test(name));
assert.ok(workerFile, 'the build has its policy worker');

// Chrome runs WebGPU headless only when told it may, and on Windows through
// Direct3D 11.
const args = ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist',
  ...(process.platform === 'win32' ? ['--use-angle=d3d11'] : [])];
const PAGE = '<!doctype html><meta charset="utf-8"><title>WebGPU check</title><body></body>';
let network = null, planes = null, masks = null;
const server = createServer(createFixtureHandler({ root: dist, intercept: name => {
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
      console.log(`SKIP WebGPU against WebAssembly: ${path} holds a Softplus; export it again with neural/export.py.`);
    } else {
      const rows = await positions();
      const width = rows[0].planes.length;
      const laid = new Float32Array(rows.length * width), legal = new Uint8Array(rows.length * rows[0].mask.length);
      rows.forEach((row, at) => { laid.set(row.planes, at * width); legal.set(row.mask, at * row.mask.length); });
      planes = Buffer.from(laid.buffer);
      masks = Buffer.from(legal.buffer);
      console.log(`Adapter: ${adapter.name || 'unnamed'}; ${rows.length} positions of ${width / POSITIONS} planes; network ${path}.`);
      await check('the worker on the graphics card plays the moves it plays on WebAssembly, and a batch costs about one position', async () => {
        const result = await page.evaluate(async ({ worker, runtimeBase, webgpuBase, width, gpuBatch, wasmBatch }) => {
          const fetched = async name => new Uint8Array(await (await fetch(`webgpu-check/${name}`)).arrayBuffer());
          const allPlanes = new Float32Array((await fetched('planes.bin')).buffer), allMasks = await fetched('masks.bin');
          const count = allPlanes.length / width, moves = allMasks.length / count;
          const url = new URL('webgpu-check/network.onnx', location.href).href, scope = new URL('./', location.href).href;
          /** A production worker, asked as policy.js asks it, the network
           * handed to it as verified bytes; `ask(start, size)` answers the
           * positions from start, and how long the worker took to run them. */
          async function open(onGpu) {
            const bytes = await fetched('network.onnx');
            const thread = new Worker(new URL(worker, location.href), { type: 'module' });
            const pending = new Map(), runtimes = [], refused = [];
            let id = 0, began = 0;
            thread.onmessage = ({ data }) => {
              if (data.runtime) runtimes.push(data.runtime);
              if (data.webgpuRefused) refused.push(data.webgpuRefused);
              if (data.id == null) return;
              if (data.ready) { began = performance.now(); return; }
              if (data.progress) return;
              const done = pending.get(data.id);
              pending.delete(data.id);
              if (data.error) done.reject(new Error(data.error));
              else done.resolve({ analyses: data.analyses, ms: performance.now() - began });
            };
            const ask = (start, size) => new Promise((resolve, reject) => {
              const message = { id: ++id, url, runtimeBase, scope, count: size, details: true,
                planes: allPlanes.slice(start * width, (start + size) * width),
                mask: Array.from(allMasks.subarray(start * moves, (start + size) * moves), allowed => allowed === 1) };
              if (onGpu) message.webgpu = webgpuBase;
              if (id === 1) message.network = bytes;
              pending.set(message.id, { resolve, reject });
              thread.postMessage(message, id === 1 ? [bytes.buffer] : []);
            });
            return { ask, runtimes, refused, close: () => thread.terminate() };
          }
          /** Every position's answer in batches of `size`, and each full
           * batch's time after the first, which loads the network. */
          async function all(runner, size) {
            const answers = [], times = [];
            for (let start = 0; start < count; start += size) {
              const { analyses, ms } = await runner.ask(start, Math.min(size, count - start));
              answers.push(...analyses);
              if (start && start + size <= count) times.push(ms);
            }
            return { answers, times };
          }
          const median = values => values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)];
          /** The median time of `runs` single positions, after one not timed. */
          async function single(runner, runs = 9) {
            const times = [];
            for (let run = 0; run <= runs; run++) times.push((await runner.ask(run, 1)).ms);
            return median(times.slice(1));
          }
          const gpu = await open(true);
          // A card the worker would not play on ends the check there, saying why.
          await gpu.ask(0, 1);
          if (gpu.runtimes.includes('wasm')) {
            gpu.close();
            return { refused: gpu.refused, runtimes: { gpu: [...new Set(gpu.runtimes)] } };
          }
          const wasm = await open(false);
          try {
            const onGpu = await all(gpu, gpuBatch), onWasm = await all(wasm, wasmBatch);
            const gpuOne = await single(gpu), wasmOne = await single(wasm);
            let largest = 0, largestWeight = 0, same = 0, choices = 0;
            onWasm.answers.forEach((plain, row) => {
              const card = onGpu.answers[row], best = plain.action;
              const legal = plain.weights.flatMap((weight, move) => (allMasks[row * moves + move] ? [move] : []));
              if (legal.length > 1) {
                choices++;
                if (card.action === best) same++;
              }
              for (const move of legal) {
                const relative = answer => Math.log(answer.weights[move]) - Math.log(answer.weights[best]);
                if (plain.weights[move] > 1e-30 && card.weights[move] > 1e-30) largest = Math.max(largest, Math.abs(relative(card) - relative(plain)));
                largestWeight = Math.max(largestWeight, Math.abs(card.weights[move] - plain.weights[move]));
              }
            });
            return { count, choices, same, largest, largestWeight, runtimes: { gpu: [...new Set(gpu.runtimes)], wasm: [...new Set(wasm.runtimes)] },
              ms: { gpuOne, wasmOne, gpuBatch: median(onGpu.times), wasmBatch: median(onWasm.times) } };
          } finally { gpu.close(); wasm.close(); }
        }, { worker: `assets/${workerFile}`, runtimeBase: `${base}${runtime}`, webgpuBase: `${base}${webgpu}`, width: rows[0].planes.length,
          gpuBatch: GPU_BATCH, wasmBatch: BATCH });
        if (result.refused) assert.fail(`the worker would not play on the card: ${result.refused.join('; ') || 'it did not say why'}`);
        const share = result.same / result.choices;
        console.log(`Played on: ${result.runtimes.gpu.join(', ')} and ${result.runtimes.wasm.join(', ')}.`);
        console.log(`Same best move on ${result.same} of ${result.choices} decisions with a choice (${(100 * share).toFixed(2)}%), of ${result.count} positions.`);
        console.log(`Largest difference in a logit, from the row's best move: ${result.largest.toExponential(3)}; in a weight: ${result.largestWeight.toExponential(3)}.`);
        console.log(`One position: ${result.ms.gpuOne.toFixed(1)} ms on WebGPU, ${result.ms.wasmOne.toFixed(1)} ms on WebAssembly.`);
        console.log(`A batch: ${result.ms.gpuBatch.toFixed(1)} ms for ${GPU_BATCH} positions on WebGPU (${(GPU_BATCH * 1000 / result.ms.gpuBatch).toFixed(0)} a second), `
          + `${result.ms.wasmBatch.toFixed(1)} ms for ${BATCH} on WebAssembly (${(BATCH * 1000 / result.ms.wasmBatch).toFixed(1)} a second).`);
        assert.deepEqual(result.runtimes, { gpu: ['webgpu'], wasm: ['wasm'] }, 'each worker plays where it was asked to');
        assert.ok(share >= SAME_BEST_AT_LEAST, `the same best move on ${(100 * share).toFixed(2)}% of decisions`);
        assert.ok(result.largest <= LOGIT_DIFFERENCE_AT_MOST, `a logit differs by ${result.largest}`);
      });
    }
  }
} finally {
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
  await report('WebGPU browser checks');
}
