/** Disposable ONNX exports through the real worker, cache and reduced WASM.
 * Run python -m neural.tests.browser_inference_fixture --out <directory> first.
 * Only the model manifest is replaced with fixture bytes/digests; no inference
 * functions, runtime operators, policy selection or input feeds are mocked.
 */
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const web = fileURLToPath(new URL('../', import.meta.url));
assert.ok(process.argv[2], 'Pass the generated fixture directory');
const folder = resolve(process.argv[2]);
const fixtures = JSON.parse(await readFile(resolve(folder, 'fixtures.json'), 'utf8'));
const executablePath = process.env.CHROME_BIN || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
assert.ok(executablePath, 'Set CHROME_BIN to a Chrome/Chromium binary');
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
let checked = 0;
try {
  for (const fixture of fixtures) {
    const files = new Map();
    files.set('/fixture.onnx', ['application/octet-stream', await readFile(resolve(folder, fixture.graph))]);
    for (const record of fixture.cases) files.set(`/${record.planes}`, ['application/octet-stream', await readFile(resolve(folder, record.planes))]);
    for (const name of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
      files.set(`/fixture-runtime/${name}`, [name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript', await readFile(resolve(web, 'runtime', name))]);
    }
    files.set('/fixture-runtime/memory-budget.mjs', ['text/javascript', await readFile(resolve(web, 'src/lib/memory-budget.js'))]);
    const manifest = { bytes: fixture.bytes, sha256: fixture.sha256, origin: '', object: 'fixture.onnx' };
    const plugin = () => ({
      name: 'disposable-inference-fixtures',
      enforce: 'pre',
      resolveId(id) { if (id === './model-manifest.js') return '\0inference-fixture-manifest'; },
      load(id) { if (id === '\0inference-fixture-manifest') return `export const MANIFEST = Object.freeze(${JSON.stringify(manifest)});`; },
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const path = new URL(req.url, 'http://local').pathname;
          if (path === '/') {
            res.setHeader('Content-Type', 'text/html');
            res.end('<!doctype html><html><head><link rel="icon" href="data:,"></head><body>Inference contract</body></html>');
          } else if (files.has(path)) {
            const [type, bytes] = files.get(path);
            res.setHeader('Content-Type', type); res.setHeader('Content-Length', bytes.length);
            res.end(bytes);
          } else next();
        });
      },
    });
    const server = await createServer({
      configFile: false, root: web, publicDir: false, logLevel: 'warn', plugins: [plugin()],
      resolve: { conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'] },
      worker: { format: 'es', plugins: () => [plugin()] },
      server: { host: '127.0.0.1', port: 0 },
    });
    const context = await browser.createBrowserContext();
    try {
      await server.listen();
      const page = await context.newPage();
      const errors = [], wasm = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('request', request => { if (new URL(request.url()).pathname.endsWith('.wasm')) wasm.push(request.url()); });
      await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/`);
      const answers = await page.evaluate(async records => {
        const { default: PolicyWorker } = await import('/src/lib/policy.worker.js?worker');
        const worker = new PolicyWorker();
        const results = [];
        try {
          // Repeat requests against the same session: inputs are disposed, the
          // session remains usable, and every stage supplies its own mask.
          for (let iteration = 0; iteration < 2; iteration++) {
            for (const [index, record] of records.entries()) {
              const id = iteration * records.length + index;
              const planes = new Float32Array(await (await fetch(`/${record.planes}`)).arrayBuffer());
              const response = await new Promise((done, reject) => {
                const timeout = setTimeout(() => reject(new Error('Worker inference timed out')), 60000);
                worker.onerror = error => { clearTimeout(timeout); reject(new Error(error.message)); };
                worker.onmessage = ({ data }) => {
                  if (data.id !== id || data.progress) return;
                  clearTimeout(timeout);
                  if (data.error) reject(new Error(data.error)); else done(data);
                };
                worker.postMessage({ id, url: new URL('/fixture.onnx', location.href).href,
                  runtimeBase: new URL('/fixture-runtime/', location.href).href,
                  planes, mask: Uint8Array.from(record.mask), temperature: 0, details: true, memoryLimitMiB: 384 });
              });
              results.push(response);
            }
          }
        } finally { worker.terminate(); }
        return results;
      }, fixture.cases);
      for (const [index, answer] of answers.entries()) {
        const record = fixture.cases[index % fixture.cases.length];
        const label = `${fixture.kind}/${record.label}`;
        assert.equal(answer.action, record.action, `${label}: PyTorch/worker action differs`);
        assert.equal(answer.analysis.action, record.action, label);
        for (let i = 0; i < 46; i++) {
          assert.ok(Math.abs(answer.analysis.weights[i] - record.weights[i]) < 1e-4, `${label}: policy ${i}`);
          if (!record.mask[i]) assert.equal(answer.analysis.weights[i], 0, label);
        }
        assert.ok(Math.abs(answer.value - record.value) < 1e-4, `${label}: value`);
        for (let i = 0; i < record.hands.length; i++) assert.ok(Math.abs(answer.hands[i] - record.hands[i]) < 1e-4, `${label}: belief ${i}`);
        checked++;
      }
      assert.deepEqual(errors, []);
      assert.ok(wasm.length > 0, 'The real WASM runtime must execute');
      assert.ok(wasm.every(url => new URL(url).pathname === '/fixture-runtime/ort-wasm-simd-threaded.wasm'),
        'Only the repository reduced runtime is permitted, not a full-runtime fallback');
      console.log(`PASS ${fixture.kind}: ordinary, call and both riichi stages, repeated in one worker`);
    } finally { await context.close(); await server.close(); }
  }
} finally { await browser.close(); }
console.log(`PASS ${checked} real browser-worker inference comparisons`);
