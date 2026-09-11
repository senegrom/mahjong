/** The shipped trained network, including preferences saved by older builds.
 * Model requests must use the shared inventory, never retired model filenames.
 * Exercise real inference and replay; a missing network cannot silently pass.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { createFixtureHandler } from './static-fixture-server.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { MODEL_FILES } from '../src/lib/model-package.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const match = new MatchSession(Game, 81, 'neural');
let initial;
try { match.advance(false); initial = match.snapshot(); } finally { match.dispose(); }
const web = fileURLToPath(new URL('../', import.meta.url)), dist = resolve(web, 'dist');
for (const file of Object.values(MODEL_FILES)) assert.ok(existsSync(resolve(dist, file)), `Missing trained network ${file}`);
const handler = createFixtureHandler({ root: dist, publicRoot: dist });
const downloads = [];
const server = createServer((request, response) => {
  const name = new URL(request.url, 'http://localhost').pathname.split('/').pop();
  if (request.method === 'GET' && name.endsWith('.onnx')) downloads.push(name);
  void handler(request, response);
});
const results = [];
const wait = ms => new Promise(done => setTimeout(done, ms));
let browser;

async function check(name, body) {
  try { await body(); results.push(true); console.log('PASS ' + name); }
  catch (error) { results.push(false); console.error('FAIL ' + name + '\n' + error.stack); }
}

async function open(context, base, storedModel) {
  const page = await context.newPage();
  page.errors = [];
  page.on('pageerror', error => page.errors.push(error.message));
  await page.setViewport({ width: 1100, height: 900 });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  await page.evaluateOnNewDocument((saveKey, settingsKey, snapshot, trainedModel) => {
    if (!localStorage.getItem(saveKey)) localStorage.setItem(saveKey, JSON.stringify(snapshot));
    localStorage.setItem(settingsKey, JSON.stringify({ version: 1, difficulty: 'neural', trainedModel, confirmDiscards: false }));
  }, SAVE_KEY, SETTINGS_KEY, initial, storedModel);
  await page.goto(`${base}?opponents=neural`, { waitUntil: 'networkidle0' });
  await page.waitForSelector('.hand');
  await page.waitForFunction(() => {
    const option = document.querySelector('select[aria-label="opponent strength"] option[value="neural"]');
    return option && !option.disabled;
  });
  return page;
}

async function play(page, moves) {
  for (let move = 0; move < moves; move++) {
    await page.waitForFunction(() => document.querySelector('.failure, .screen, .standings, .hand button.tile:not(:disabled), .call-options button:not(:disabled)'), { timeout: 45000 });
    assert.equal(await page.$('.failure'), null, 'The actual trained worker must answer');
    const acted = await page.evaluate(() => {
      const next = document.querySelector('.screen .primary:not([disabled])');
      if (next) { next.click(); return true; }
      const call = document.querySelector('.call-options button[data-choice="ron"], .call-options button[data-choice="tsumo"], .call-options button[data-choice="pass"]');
      if (call) { call.click(); return true; }
      const tile = document.querySelector('.hand button.tile:not([disabled])');
      if (tile) { tile.click(); return true; }
      return false;
    });
    if (!acted) break;
    await wait(200);
  }
  await page.waitForFunction(() => document.querySelector('.failure, .screen, .standings, .hand button.tile:not(:disabled), .call-options button:not(:disabled)'), { timeout: 45000 });
  assert.equal(await page.$('.failure'), null);
}

try {
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  const base = `http://127.0.0.1:${server.address().port}/mahjong/`;
  const chrome = process.env.CHROME_BIN
    || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(chrome, 'Set CHROME_BIN to a Chromium/Chrome executable');
  browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });

  // Old preference IDs are migration inputs, not additional shipped networks.
  for (const storedModel of ['full', 'quick', 'strong']) {
    await check(`${storedModel} preference uses only the shipped trained network and preserves replay`, async () => {
      const context = await browser.createBrowserContext(), before = downloads.length;
      try {
        const page = await open(context, base, storedModel);
        assert.equal(await page.$('select[aria-label="Trained opponent"]'), null,
          'One shipped network must not offer the retired network selector');
        await play(page, 12);
        const fetched = downloads.slice(before);
        assert.ok(fetched.includes(MODEL_FILES.full), 'The actual trained network was never fetched');
        assert.ok(fetched.every(name => Object.values(MODEL_FILES).includes(name)), 'A retired or unknown network was fetched');
        assert.deepEqual(page.errors, []);
        const saved = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
        assert.ok(saved.commands.filter(command => command.type === 'opponent').length >= 2,
          'The trained worker must complete real decisions, not only download');
        const restored = MatchSession.restore(Game, JSON.stringify(saved));
        try { assert.equal(restored.stateKey(), saved.state); } finally { restored.dispose(); }
      } finally { await context.close(); }
    });
  }
} finally {
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
}
console.log(`${results.filter(Boolean).length}/${results.length} trained-model browser checks passed`);
if (results.some(passed => !passed)) process.exitCode = 1;
