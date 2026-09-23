/** Production browser checks for layout, app component boundaries and claimed-tile display. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { emptyPosition, parseTiles, recordChoice } from '../src/lib/physical-position.js';
import { emptyGuided, GUIDED_FORMAT } from '../src/lib/guided-game.js';
import { SETTINGS_KEY, SAVE_KEY } from '../src/lib/session.js';
import { tileWords } from '../src/lib/tiles.js';

const root = fileURLToPath(new URL('../', import.meta.url));
let denyUnusedArt = false;
const serve = createFixtureHandler({ root: resolve(root, 'dist'), publicRoot: resolve(root, 'dist') });
const server = createServer((request, response) => {
  if (denyUnusedArt && request.url.includes('/tiles/matisse/')) {
    response.writeHead(503, { 'Content-Type': 'text/plain', 'Cache-Control': 'no-store' });
    response.end('Artwork temporarily unavailable');
  } else serve(request, response);
});
const results = [];
let browser;
async function check(name, body) {
  const context = await browser.createBrowserContext();
  try { await body(context); results.push({ name, passed: true }); console.log(`PASS ${name}`); }
  catch (error) { results.push({ name, passed: false, error: error.stack }); console.error(`FAIL ${name}\n${error.stack}`); }
  finally { await context.close(); }
}
async function pageAt(context, mode = 'play', game = null) {
  const page = await context.newPage();
  await page.setViewport({ width: 1100, height: 1000 });
  const problems = [];
  page.on('pageerror', error => problems.push(error.message));
  const seed = await page.evaluateOnNewDocument(({ settings, guided, text }) => {
    localStorage.setItem(settings, JSON.stringify({ version: 1, difficulty: 'club', hints: true, confirmDiscards: false }));
    if (text && !localStorage.getItem(guided)) localStorage.setItem(guided, text);
  }, { settings: SETTINGS_KEY, guided: GUIDED_FORMAT.key, text: game && GUIDED_FORMAT.encode(game) });
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/?mode=${mode}`, { waitUntil: 'domcontentloaded' });
  await page.removeScriptToEvaluateOnNewDocument(seed.identifier);
  if (mode === 'play') {
    for (let step = 0; step < 10; step++) {
      await page.waitForSelector('.hand button:not(:disabled), .call-options button:not(:disabled)');
      if (await page.$('.hand button:not(:disabled)')) break;
      await page.click('.call-options button[data-choice="pass"]:not(:disabled)');
    }
    await page.waitForSelector('.hand button:not(:disabled)');
  }
  return { page, problems };
}
async function savedMatch(page) {
  await page.waitForFunction(key => Boolean(localStorage.getItem(key))
    && !document.querySelector('.failure')
    && Boolean(document.querySelector('.hand button[data-hand-index]:not(:disabled), .call-options button:not(:disabled), .play-area.ended')), {}, SAVE_KEY);
  return page.evaluate(key => localStorage.getItem(key), SAVE_KEY);
}
function calledGame(offered) {
  const p = emptyPosition();
  Object.assign(p, { turn: 3, phase: 'call', pending: offered, drawn: null, wall: 60, first_turns: false, indicators: ['7z'] });
  p.players[0].hand = [...['1m', '2m', '3m'].filter(tile => tile !== offered), ...parseTiles('456p789s11223z')];
  p.players[3].discards = [{ tile: offered, order: 0, drawn: false, riichi: false, claimed: false }];
  const choice = { kind: 'chii', tile: '1m' };
  const game = emptyGuided();
  Object.assign(game.state, { position: recordChoice(p, choice, [choice]), stage: 'decision', nextSeat: 0, needsDraw: false });
  game.past = [{ state: { ...structuredClone(game.state), position: p }, logLength: 0 }];
  game.log = ['Recorded chii'];
  return game;
}
try {
  await mkdir(resolve(root, 'test-results'), { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  const executablePath = process.env.CHROME_BIN || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(executablePath, 'Set CHROME_BIN to Chrome/Chromium');
  browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  await check('tablet breakpoint never overflows at phone, tablet, or desktop boundaries', async context => {
    const { page, problems } = await pageAt(context);
    await page.waitForSelector('.hand');
    for (const width of [320, 390, 760, 761, 768, 820, 900, 967, 968, 1024, 1280]) {
      await page.setViewport({ width, height: 1000 });
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const size = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, width: innerWidth,
        seats: [...document.querySelectorAll('main > .board > .place')].map(el => ({ left: el.getBoundingClientRect().left, right: el.getBoundingClientRect().right })) }));
      assert.ok(size.scroll <= size.width, `${width}px viewport overflows to ${size.scroll}px`);
      assert.equal(size.seats.length, 3);
      assert.ok(size.seats.every(seat => seat.left >= -1 && seat.right <= width + 1), JSON.stringify(size));
      if ([390, 820, 1024].includes(width)) await page.screenshot({ path: resolve(root, 'test-results', `app-layout-${width}.png`), fullPage: true });
    }
    assert.deepEqual(problems, []);
  });
  await check('settings bindings update the hand and preferences without changing the saved match', async context => {
    const { page, problems } = await pageAt(context);
    const saved = await savedMatch(page);
    await page.click('.options > summary');
    await page.click('.options input[type=checkbox]');
    await page.waitForFunction(() => !document.querySelector('.mine .hint'));
    await page.select('select[aria-label="Tile face"]', 'matisse');
    await page.waitForFunction(() => document.querySelectorAll('.hand .tile').length > 0
      && [...document.querySelectorAll('.hand .tile')].every(tile => tile.classList.contains('matisse')));
    await page.waitForFunction(key => {
      const settings = JSON.parse(localStorage.getItem(key));
      return settings.hints === false && settings.tileFace === 'matisse';
    }, {}, SETTINGS_KEY);
    assert.equal(await savedMatch(page), saved);
    const confirmation = '.options label:has(input[type=checkbox]):nth-of-type(3) input';
    await page.click(confirmation);
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).confirmDiscards === true, {}, SETTINGS_KEY);
    await page.reload({ waitUntil: 'networkidle0' });
    assert.equal(await savedMatch(page), saved);
    await page.click('.options > summary');
    assert.deepEqual(await page.evaluate(key => {
      const { hints, tileFace, confirmDiscards } = JSON.parse(localStorage.getItem(key));
      return { hints, tileFace, confirmDiscards };
    }, SETTINGS_KEY), { hints: false, tileFace: 'matisse', confirmDiscards: true });
    assert.equal(await page.$eval(confirmation, input => input.checked), true);
    assert.equal(await page.$eval('select[aria-label="Tile face"]', select => select.value), 'matisse');
    assert.deepEqual(problems, []);
  });
  await check('discard cancellation and preference changes clear the shared selection without playing a move', async context => {
    const { page, problems } = await pageAt(context);
    const saved = await savedMatch(page);
    await page.click('.options > summary');
    const confirmation = '.options label:has(input[type=checkbox]):nth-of-type(3) input';
    await page.click(confirmation);
    await page.click('.hand button[data-hand-index]:not(:disabled)');
    await page.waitForSelector('.confirm-discard');
    await page.click('.confirm-discard button:last-child');
    await page.waitForFunction(() => !document.querySelector('.confirm-discard'));
    await page.click('.hand button[data-hand-index]:not(:disabled)');
    await page.waitForSelector('.confirm-discard');
    await page.click(confirmation);
    await page.waitForFunction(() => !document.querySelector('.confirm-discard'));
    assert.equal(await savedMatch(page), saved);
    assert.deepEqual(problems, []);
  });
  await check('mode round trips preserve the same match and restore the bound hand element', async context => {
    const { page, problems } = await pageAt(context);
    const saved = await savedMatch(page);
    for (const index of [3, 4]) {
      await page.click(`.game-modes button:nth-child(${index})`);
      await page.waitForFunction(() => !document.querySelector('main > .board'));
      await page.click('.game-modes button:first-child');
      assert.equal(await savedMatch(page), saved);
      await page.focus('.hand');
      await page.keyboard.press('ArrowRight');
      await page.waitForSelector('.hand button[aria-pressed=true]');
      await page.keyboard.press('Escape');
      await page.waitForFunction(() => !document.querySelector('.hand button[aria-pressed=true]'));
    }
    assert.deepEqual(problems, []);
  });
  await check('inspection and custom-opponent dialogs keep active and draft state separate', async context => {
    const { page, problems } = await pageAt(context);
    const saved = await savedMatch(page);
    await page.click('.inspect');
    await page.waitForSelector('.table-dialog[open]');
    assert.equal(await page.$$eval('.inspection-grid > section', seats => seats.length), 4);
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('.table-dialog[open]'));
    await page.select('select[aria-label="opponent strength"]', 'custom');
    await page.waitForSelector('.custom-dialog[open]');
    await page.select('.opponent-fields label:first-child select', 'beginner');
    await page.click('.custom-actions button:first-child');
    await page.waitForFunction(() => !document.querySelector('.custom-dialog[open]'));
    assert.equal(await page.$eval('select[aria-label="opponent strength"]', select => select.value), 'club');
    assert.equal(await savedMatch(page), saved);
    await page.select('select[aria-label="opponent strength"]', 'custom');
    await page.waitForSelector('.custom-dialog[open]');
    assert.deepEqual(await page.$$eval('.opponent-fields select', selects => selects.map(select => select.value)), ['club', 'club', 'club']);
    await page.select('.opponent-fields label:first-child select', 'beginner');
    await page.select('.opponent-fields label:last-child select', 'beginner');
    page.on('dialog', dialog => void dialog.accept());
    await page.click('.custom-actions button:last-child');
    await page.waitForFunction(() => !document.querySelector('.custom-dialog[open]')
      && document.querySelector('select[aria-label="opponent strength"]').value === 'custom');
    await page.waitForFunction(key => {
      const settings = JSON.parse(localStorage.getItem(key));
      return JSON.stringify(settings.opponents) === JSON.stringify(['beginner', 'club', 'beginner']);
    }, {}, SETTINGS_KEY);
    await page.waitForFunction((key, previous) => localStorage.getItem(key) !== previous, {}, SAVE_KEY, saved);
    assert.notEqual(await savedMatch(page), saved);
    assert.deepEqual(problems, []);
  });
  await check('unused artwork cannot block play or falsely report offline readiness; failed switches preserve the match', async context => {
    denyUnusedArt = true;
    try {
      const { page, problems } = await pageAt(context);
      const saved = await savedMatch(page);
      assert.equal(await page.$eval('[data-core-ready]', el => el.dataset.coreReady), 'false');
      await page.click('.options > summary');
      await page.select('select[aria-label="Tile face"]', 'matisse');
      await page.waitForSelector('[data-face-error]');
      assert.equal(await page.$eval('select[aria-label="Tile face"]', el => el.value), 'classic');
      assert.equal(await page.evaluate(key => JSON.parse(localStorage.getItem(key)).tileFace, SETTINGS_KEY), 'classic');
      assert.equal(await page.$('.hand .tile.matisse'), null);
      assert.equal(await savedMatch(page), saved);
      denyUnusedArt = false;
      await page.select('select[aria-label="Tile face"]', 'matisse');
      await page.waitForSelector('.hand .tile.matisse');
      await page.waitForFunction(() => !document.querySelector('[data-face-progress]'));
      assert.equal(await page.$('[data-face-error]'), null);
      assert.equal(await savedMatch(page), saved);
      assert.deepEqual(problems, []);
    } finally { denyUnusedArt = false; }
  });
  for (const offered of ['1m', '2m', '3m']) {
    await check(`guided chii rotates ${offered} after reload and Undo removes the set`, async context => {
      const { page, problems } = await pageAt(context, 'guided', calledGame(offered));
      // Display-only tiles expose their identity through the accessible label,
      // not the data-tile attribute used by interactive hand buttons.
      const actual = () => page.$eval('.your-hand .meld .tile.rotated', el => el.getAttribute('aria-label'));
      await page.waitForSelector('.guided-controls:not(:disabled)');
      assert.equal(await actual(), tileWords(offered));
      await page.reload({ waitUntil: 'networkidle0' });
      await page.waitForSelector('.guided-controls:not(:disabled)');
      assert.equal(await actual(), tileWords(offered));
      await page.click('.guided-heading .buttons button');
      await page.waitForFunction(() => !document.querySelector('.your-hand .meld'));
      assert.deepEqual(problems, []);
    });
  }
} finally {
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
  await mkdir(resolve(root, 'test-results'), { recursive: true });
  await writeFile(resolve(root, 'test-results/cleanup.json'), JSON.stringify(results, null, 2));
}
if (results.some(result => !result.passed)) process.exitCode = 1;
