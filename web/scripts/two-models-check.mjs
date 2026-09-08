/**
 * The two trained opponents, against the production build.
 *
 * The game carries a small network and, when one has been trained and
 * measured stronger, a larger one beside it. A player chooses which plays,
 * and the point of the choice is that neither download happens unless it is
 * the one chosen: a phone should not have to take sixty megabytes to play
 * the quick opponent, and someone who wants the strong one should be able to
 * ask for it.
 *
 * A build with only the small network is the normal case for a checkout, so
 * the check says what it found and passes: what it must never allow is the
 * choice appearing without a second network behind it, or the large file
 * being fetched by someone who did not ask for it.
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

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const match = new MatchSession(Game, 81, 'neural');
let initial;
try { match.advance(false); initial = match.snapshot(); } finally { match.dispose(); }

const web = fileURLToPath(new URL('../', import.meta.url));
const dist = resolve(web, 'dist');
const shipped = existsSync(resolve(dist, 'model-strong.onnx'));
const handler = createFixtureHandler({ root: dist, publicRoot: dist });
const server = createServer((request, response) => void handler(request, response));
const results = [];
const wait = (ms) => new Promise((done) => setTimeout(done, ms));
let browser;

async function check(name, body) {
  try {
    await body();
    results.push(true);
    console.log('PASS ' + name);
  } catch (error) {
    results.push(false);
    console.error('FAIL ' + name + '\n' + error.stack);
  }
}

async function open(context, base) {
  const page = await context.newPage();
  page.errors = [];
  // Listening from before the first navigation: the page may ask for its
  // network while it is still loading, and a listener attached afterwards
  // would report that nothing was ever fetched.
  page.asked = [];
  page.on('request', (request) => {
    const name = new URL(request.url()).pathname.split('/').pop();
    if (name.startsWith('model') && name.endsWith('.onnx')) page.asked.push(name);
  });
  page.on('pageerror', (error) => page.errors.push(error.message));
  await page.setViewport({ width: 1100, height: 900 });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  await page.evaluateOnNewDocument((saveKey, settingsKey, snapshot) => {
    if (!localStorage.getItem(saveKey)) localStorage.setItem(saveKey, JSON.stringify(snapshot));
    localStorage.setItem(settingsKey, JSON.stringify({ version: 1, difficulty: 'neural', trainedModel: 'quick', confirmDiscards: false }));
  }, SAVE_KEY, SETTINGS_KEY, initial);
  await page.goto(`${base}?opponents=neural`, { waitUntil: 'networkidle0' });
  await page.waitForSelector('.hand');
  return page;
}

/** The choice, once the page has decided whether it has a second network.
 * That waits on the offline layer, which takes its time on a first visit. */
async function choices(page, patience = 30000) {
  const deadline = Date.now() + patience;
  while (Date.now() < deadline) {
    await page.evaluate(() => document.querySelector('details.options')?.setAttribute('open', ''));
    const found = await page.evaluate(() => {
      const select = document.querySelector('select[aria-label="Trained opponent"]');
      return select ? [...select.options].map((option) => option.value) : null;
    });
    if (found) return found;
    await wait(500);
  }
  return null;
}

/** Plays whatever the table asks for, so the opponents have to answer. */
async function play(page, moves) {
  for (let move = 0; move < moves; move += 1) {
    assert.equal(await page.$eval('body', element => element.querySelector('.failure')?.textContent?.trim() ?? null), null);
    const acted = await page.evaluate(() => {
      // A hand can end before the model switch. Keep playing so the new
      // network actually gets a turn, even when the download came through SW.
      const next = document.querySelector('.screen .primary:not([disabled])');
      if (next) { next.click(); return true; }
      const pass = [...document.querySelectorAll('.call-options button')]
        .find((button) => button.textContent.trim() === 'Pass');
      if (pass) { pass.click(); return true; }
      const tile = document.querySelector('.hand button.tile:not([disabled])');
      if (tile) { tile.click(); return true; }
      return false;
    });
    await wait(acted ? 200 : 400);
  }
}

try {
  await new Promise((done) => server.listen(0, '127.0.0.1', done));
  const base = `http://127.0.0.1:${server.address().port}/mahjong/`;
  const chrome = process.env.CHROME_BIN
    || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(chrome, 'Set CHROME_BIN to a Chromium/Chrome executable');
  browser = await puppeteer.launch({
    executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });

  await check(`the choice is offered only with a second network (this build has ${shipped ? 'one' : 'none'})`, async () => {
    const context = await browser.createBrowserContext();
    try {
      const page = await open(context, base);
      const offered = await choices(page, shipped ? 30000 : 8000);
      if (shipped) assert.deepEqual(offered, ['quick', 'strong']);
      else assert.equal(offered, null, 'a choice was offered with nothing behind it');
    } finally { await context.close(); }
  });

  await check('the quick opponent never fetches the larger network', async () => {
    const context = await browser.createBrowserContext();
    try {
      const page = await open(context, base);
      await play(page, 16);
      assert.ok(page.asked.includes('model.onnx'), 'the quick network was never fetched');
      assert.deepEqual(page.asked.filter((name) => name !== 'model.onnx'), [],
        'something other than the quick network was downloaded');
      assert.deepEqual(page.errors, []);
    } finally { await context.close(); }
  });

  if (shipped) {
    await check('choosing the stronger opponent fetches it, remembers it, and plays on', async () => {
      const context = await browser.createBrowserContext();
      try {
        const page = await open(context, base);
        assert.ok(await choices(page), 'the choice never appeared');
        await play(page, 8);
        await page.select('select[aria-label="Trained opponent"]', 'strong');
        // The larger network is tens of megabytes and answers in a couple of
        // hundred milliseconds, so this waits for it to arrive rather than
        // counting moves, which would call a slow download a missing one.
        const arrives = Date.now() + 90000;
        while (Date.now() < arrives && !page.asked.includes('model-strong.onnx')) {
          await play(page, 4);
        }
        assert.ok(page.asked.some((name) => name === 'model-strong.onnx'),
          'the stronger network was never fetched');
        const remembered = await page.evaluate(
          () => JSON.parse(localStorage.getItem('riichi.settings.v1') ?? '{}').trainedModel,
        );
        assert.equal(remembered, 'strong');
        assert.equal(await page.evaluate(
          () => document.querySelector('.failure')?.textContent?.trim() ?? null,
        ), null);
        assert.ok(await page.evaluate(
          () => document.querySelectorAll('.mine .pool .tile').length,
        ) > 0, 'the table stopped playing after the change');
        assert.deepEqual(page.errors, []);
      } finally { await context.close(); }
    });
  }
} finally {
  await browser?.close();
  if (server.listening) await new Promise((done) => server.close(done));
}
console.log(`${results.filter(Boolean).length}/${results.length} two-model browser checks passed`);
if (results.some((passed) => !passed)) process.exitCode = 1;
