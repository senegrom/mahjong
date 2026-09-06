/** Exercise both Vite serving modes and Svelte HMR with the real compiler.
 * Run after npm run wasm && npm run build. No test API ships in the app. */
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer, preview } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import puppeteer from 'puppeteer-core';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';

const web = fileURLToPath(new URL('../', import.meta.url));
const output = resolve(web, 'test-results');
const results = [];
await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const match = new MatchSession(Game, 81, ['neural', 'club', 'neural']);
let snapshot;
try { match.advance(false); snapshot = match.snapshot(); } finally { match.dispose(); }
const executablePath = process.env.CHROME_BIN || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
assert.ok(executablePath, 'Set CHROME_BIN to Chrome/Chromium');
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
const saved = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
const settled = page => page.waitForFunction(() => document.querySelector('.failure, .screen, .standings, .hand button:not(:disabled), .call-options button:not(:disabled)'), { timeout: 45000 });

async function check(name, run) {
  const context = await browser.createBrowserContext();
  try {
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
    await run(page);
    assert.deepEqual(errors, [], 'Browser errors or failed asset requests');
    results.push({ name, passed: true });
    console.log(`PASS ${name}`);
  } catch (error) {
    results.push({ name, passed: false, error: error.stack });
    console.error(`FAIL ${name}\n${error.stack}`);
  } finally { await context.close(); }
}

async function play(page, url, mode) {
  await page.setViewport({ width: 390, height: 844, isMobile: true, hasTouch: true });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  await page.evaluateOnNewDocument((key, settingsKey, initial) => {
    if (!localStorage.getItem(key)) localStorage.setItem(key, JSON.stringify(initial));
    localStorage.setItem(settingsKey, JSON.stringify({ version: 1, difficulty: 'club', hints: true, confirmDiscards: true }));
  }, SAVE_KEY, SETTINGS_KEY, snapshot);
  await page.goto(url, { waitUntil: 'networkidle0' });
  await page.waitForSelector('.hand');
  await settled(page);
  for (let turn = 0; turn < 8; turn++) {
    assert.equal(await page.$('.failure'), null, 'Real trained opponent failed');
    const before = await saved(page);
    if (JSON.parse(before.state)[0].phase === 'over') break;
    const call = await page.$('.call-options button[data-choice="ron"], .call-options button[data-choice="tsumo"], .call-options button[data-choice="pass"]');
    if (call) await call.click();
    else { await page.click('.hand button:not(:disabled)'); await page.click('.confirm-discard .primary'); }
    await page.waitForFunction((key, count) => JSON.parse(localStorage.getItem(key)).commands.length > count, {}, SAVE_KEY, before.commands.length);
    await settled(page);
  }
  assert.equal(await page.$('.failure'), null);
  const played = await saved(page);
  assert.ok(played.commands.filter(command => command.type === 'opponent').length >= 4, 'The actual neural worker must answer multiple turns');
  const restored = MatchSession.restore(Game, JSON.stringify(played));
  try { assert.equal(restored.stateKey(), played.state); } finally { restored.dispose(); }
  await page.reload({ waitUntil: 'networkidle0' });
  await settled(page);
  assert.equal((await saved(page)).state, played.state, 'Reload must preserve the match');
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Mobile layout overflow');
  await page.screenshot({ path: resolve(output, `toolchain-${mode}.png`), fullPage: true });
}

try {
  await mkdir(output, { recursive: true });
  const dev = await createServer({ root: web, logLevel: 'warn', server: { host: '127.0.0.1', port: 0 } });
  try {
    await dev.listen();
    await check('Vite development: actual neural worker, mobile play and saved-match reload', page => play(page, `http://127.0.0.1:${dev.httpServer.address().port}/`, 'dev'));
  } finally { await dev.close(); }

  const production = await preview({ root: web, logLevel: 'warn', preview: { host: '127.0.0.1', port: 0 } });
  try {
    await check('Vite preview: actual neural worker, mobile play and saved-match reload', page => play(page, `http://127.0.0.1:${production.httpServer.address().port}/`, 'preview'));
  } finally {
    await new Promise((done, reject) => production.httpServer.close(error => error ? reject(error) : done()));
  }

  // A disposable component checks HMR without editing application source.
  const temporary = await mkdtemp(resolve(web, '.tile-effects-hmr-'));
  let hot;
  try {
    await writeFile(resolve(temporary, 'index.html'), '<!doctype html><html><head><link rel="icon" href="data:,"></head><body><div id="app"></div><script type="module" src="/main.js"></script></body></html>');
    await writeFile(resolve(temporary, 'main.js'), 'import { mount } from "svelte"; import App from "./App.svelte"; mount(App, { target: document.getElementById("app") });');
    const component = '<script>let count = $state(0);</script><h1>before-update</h1><button onclick={() => count++}>{count}</button>';
    await writeFile(resolve(temporary, 'App.svelte'), component);
    hot = await createServer({ configFile: false, root: temporary, publicDir: false, plugins: [svelte()], logLevel: 'warn', server: { host: '127.0.0.1', port: 0 } });
    await hot.listen();
    await check('Svelte component hot reload updates the page and remains interactive', async page => {
      await page.goto(`http://127.0.0.1:${hot.httpServer.address().port}/`, { waitUntil: 'networkidle0' });
      await page.waitForSelector('button');
      await page.click('button');
      assert.equal(await page.$eval('button', element => element.textContent), '1');
      await page.evaluate(() => { window.hmrSentinel = 'same-document'; });
      await writeFile(resolve(temporary, 'App.svelte'), component.replace('before-update', 'after-update'));
      await page.waitForFunction(() => document.querySelector('h1')?.textContent === 'after-update');
      assert.equal(await page.evaluate(() => window.hmrSentinel), 'same-document');
      const count = Number(await page.$eval('button', element => element.textContent));
      await page.click('button');
      assert.equal(Number(await page.$eval('button', element => element.textContent)), count + 1);
    });
  } finally { await hot?.close(); await rm(temporary, { recursive: true, force: true }); }
} finally {
  await browser.close();
  await mkdir(output, { recursive: true });
  await writeFile(resolve(output, 'toolchain-report.json'), JSON.stringify(results, null, 2));
}
console.log(`${results.filter(result => result.passed).length}/${results.length} toolchain checks passed`);
if (results.some(result => !result.passed)) process.exitCode = 1;
