/** Actual mounted Watch UI + real WASM; only inference answers are controlled. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer, normalizePath } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { riichiPosition, RIICHI_SEED } from '../tests/fixtures/riichi-position.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const root = fileURLToPath(new URL('../', import.meta.url));
const cases = browserChecks();
let server, browser;
try {
  await mkdir(resolve(root, 'test-results'), { recursive: true });
  // Test-only Vite replacement: no application injection hooks or policy edits.
  // Vite hands plugins ids with forward slashes; on Windows resolve() gives
  // backslashes, so the module would never be swapped for the fixture.
  const policy = normalizePath(resolve(root, 'src/lib/policy.js'));
  server = await createServer({ root, configFile: false, logLevel: 'warn',
    plugins: [{ name: 'controlled-riichi-inference', enforce: 'pre', load(id) {
      if (id === policy) return readFileSync(resolve(root, 'tests/fixtures/riichi-policy.js'), 'utf8');
    } }, svelte()], server: { host: '127.0.0.1', port: 0 } });
  await server.listen(); browser = await launchChrome();
  for (const width of [1100, 390]) await cases.check(`riichi grouping and alternative confirmation at ${width}px`, async () => {
    const fixture = riichiPosition(Game), errors = [];
    try {
      const { match, plan, preferred, alternative, warmup } = fixture;
      const page = await (await cases.openContext(browser)).newPage();
      page.on('pageerror', error => errors.push(error.message));
      await page.setViewport({ width, height: 1000 });
      await page.evaluateOnNewDocument(({ plan, seed }) => {
        window.riichiPlan = plan; window.riichiCalls = 0; Date.now = () => seed;
      }, { plan, seed: RIICHI_SEED });
      await page.goto(`${server.resolvedUrls.local[0]}tests/fixtures/agent-defaults.html?mode=watch`, { waitUntil: 'networkidle0' });
      await page.click('[data-available]');
      await page.waitForFunction(() => document.querySelector('[aria-label="Followed agent"]').value === 'full');
      for (const label of ['Right', 'Opposite', 'Left']) await page.select(`[aria-label="${label}"]`, 'club');
      await page.click('.watch-setup .primary');
      const ready = async count => page.waitForFunction(count => window.riichiCalls === count
        && document.querySelector('.weight-row.best .choice-action:not(:disabled)'), {}, count);
      await ready(1);
      for (let step = 0; step < warmup; step++) {
        await page.click('.watch-controls > button');
        await ready(step === warmup - 1 ? plan.length : step + 2);
      }
      assert.equal(await page.$$eval('.reach-heading', rows => rows.length), 1);
      assert.equal(await page.$eval('.reach-heading meter', meter => meter.value), .55);
      assert.deepEqual(await page.$$eval('.reach-discards meter', meters => meters.map(meter => meter.value)), [.75, .25]);
      assert.equal(await page.$eval('.weight-row.best', row => row.dataset.choice), preferred.label);
      assert.equal(await page.$eval('.weight-row.best', row => getComputedStyle(row).borderColor), 'rgb(78, 163, 255)');
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'no horizontal overflow');
      const selector = `.reach-discards .choice-action[aria-label=${JSON.stringify(`Play ${alternative.label}`)}]`;
      const history = () => page.$$eval('.agent-watch .log-line', lines => lines.map(line => line.textContent));
      const before = await history();
      assert.deepEqual(before, match.events, 'browser reached the same real position');
      page.once('dialog', dialog => dialog.dismiss()); await page.click(selector);
      assert.deepEqual(await history(), before);
      assert.equal(await page.evaluate(() => window.riichiCalls), plan.length, 'cancel does not ask again');
      await page.screenshot({ path: resolve(root, `test-results/adviser-riichi-${width}.png`), fullPage: true });
      match.apply({ type: 'choose', kind: alternative.kind, tile: alternative.tile }); match.advance(false);
      page.once('dialog', dialog => dialog.accept()); await page.click(selector);
      await page.waitForFunction(expected => JSON.stringify([...document.querySelectorAll('.agent-watch .log-line')]
        .map(line => line.textContent)) === JSON.stringify(expected), {}, match.events);
      assert.deepEqual(await history(), match.events, 'confirmed alternative plays that exact tile once');
      if (match.view.phase !== 'over') await ready(plan.length + 1);
      assert.deepEqual(errors, []);
    } finally { fixture.match.dispose(); }
  });
} finally {
  await browser?.close(); await server?.close();
  await cases.report('adviser riichi checks', resolve(root, 'test-results/adviser-riichi.json'));
}
