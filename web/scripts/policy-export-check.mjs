/** Native encoder -> checked ONNX export -> actual worker -> reduced WASM.
 * Uses small synthetic checkpoints; never replaces production model artifacts.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { createFixtureHandler } from './static-fixture-server.mjs';

const web = fileURLToPath(new URL('../', import.meta.url));
const where = await mkdtemp(resolve(tmpdir(), 'mahjong-worker-export-'));
let browser, server;
const close = async () => {
  await browser?.close();
  if (server) await new Promise(done => server.close(done));
  await rm(where, { recursive: true, force: true });
};
try {
  const made = spawnSync(process.env.PYTHON || 'python', ['-m', 'neural.tests.worker_fixture', where],
    { cwd: resolve(web, '..'), encoding: 'utf8', timeout: 120_000, maxBuffer: 8 * 1024 * 1024 });
  process.stdout.write(made.stdout ?? '');
  if (made.status !== 0) throw new Error(made.error?.message || made.stderr || `export exited ${made.status}`);
  const fixtures = JSON.parse(await readFile(resolve(where, 'fixtures.json'), 'utf8'));
  await cp(resolve(web, 'node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs'), resolve(where, 'ort.mjs'));
  await mkdir(resolve(where, 'ort'));
  for (const file of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']) {
    await cp(resolve(web, 'runtime', file), resolve(where, 'ort', file));
  }
  await cp(resolve(web, 'src/lib/memory-budget.js'), resolve(where, 'ort/memory-budget.mjs'));
  for (const fixture of fixtures) {
    await cp(resolve(web, 'src/lib'), resolve(where, fixture.name, 'lib'), { recursive: true });
    const workerFile = resolve(where, fixture.name, 'lib/policy.worker.js');
    const worker = await readFile(workerFile, 'utf8');
    // Resolve the bare package import just as a bundler does. No worker logic
    // or inference/runtime function is replaced with a test double.
    assert.ok(worker.includes("from 'onnxruntime-web/wasm'"));
    await writeFile(workerFile, worker.replace("from 'onnxruntime-web/wasm'", "from '/mahjong/ort.mjs'"));
    await writeFile(resolve(where, fixture.name, 'lib/model-manifest.js'),
      `export const MANIFEST = ${JSON.stringify({ bytes: fixture.bytes, sha256: fixture.sha256, origin: '', object: 'model.onnx' })};\n`);
  }
  await writeFile(resolve(where, 'index.html'), '<!doctype html><title>Policy export regression</title>');
  server = createServer(createFixtureHandler({ root: where, publicRoot: where }));
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const executablePath = process.env.CHROME_BIN || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(executablePath, 'Set CHROME_BIN to Chrome/Chromium');
  browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  for (const fixture of fixtures) {
    const context = await browser.createBrowserContext();
    try {
      const page = await context.newPage();
      const problems = []; page.on('pageerror', error => problems.push(error.message));
      await page.goto(`${origin}/mahjong/`);
      const answers = await page.evaluate(async ({ name, cases }) => {
        const worker = new Worker(`/mahjong/${name}/lib/policy.worker.js`, { type: 'module' });
        try {
          const answers = [];
          for (let repeat = 0; repeat < 2; repeat++) {
            for (const [index, item] of cases.entries()) {
              const raw = await (await fetch(`/mahjong/${name}/${item.name}.bin`)).arrayBuffer();
              const id = repeat * cases.length + index;
              answers.push(await new Promise((done, fail) => {
                const timer = setTimeout(() => fail(new Error('policy worker timed out')), 30_000);
                worker.onerror = event => { clearTimeout(timer); fail(new Error(event.message)); };
                worker.onmessage = ({ data }) => {
                  if (data.id !== id || data.progress) return;
                  clearTimeout(timer);
                  if (data.error) fail(new Error(data.error)); else done(data);
                };
                worker.postMessage({ id, url: `${location.origin}/mahjong/${name}/model.onnx`,
                  runtimeBase: `${location.origin}/mahjong/ort/`, planes: new Float32Array(raw),
                  mask: item.mask, temperature: 0, details: true });
              }));
            }
          }
          return answers;
        } finally { worker.terminate(); }
      }, fixture);
      assert.deepEqual(problems, []);
      assert.equal(answers.length, fixture.cases.length * 2);
      for (const [index, answer] of answers.entries()) {
        const expected = fixture.cases[index % fixture.cases.length];
        assert.equal(answer.action, expected.action, `${fixture.name}/${expected.name}: reference action`);
        assert.ok(expected.mask[answer.action]);
        assert.ok(Math.abs(answer.value - expected.value) < 2e-4);
        for (let i = 0; i < expected.weights.length; i++) {
          assert.ok(Math.abs(answer.analysis.weights[i] - expected.weights[i]) < 2e-4, `policy weight ${i}`);
        }
        for (let i = 0; i < expected.hands.length; i++) {
          assert.ok(Math.abs(answer.hands[i] - expected.hands[i]) < 2e-4, `belief logit ${i}`);
        }
      }
      console.log(`PASS ${fixture.name}: ${answers.length} actual-worker inferences including both riichi stages`);
    } finally { await context.close(); }
  }
} finally { await close(); }
