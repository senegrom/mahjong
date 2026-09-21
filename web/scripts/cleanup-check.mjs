/** Production browser checks for tablet layout and persisted claimed-tile display. */
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
import { SETTINGS_KEY } from '../src/lib/session.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const server = createServer(createFixtureHandler({ root: resolve(root, 'dist'), publicRoot: resolve(root, 'dist') }));
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
  await page.evaluateOnNewDocument(({ settings, guided, text }) => {
    localStorage.setItem(settings, JSON.stringify({ version: 1, difficulty: 'club', hints: true, confirmDiscards: false }));
    if (text && !localStorage.getItem(guided)) localStorage.setItem(guided, text);
  }, { settings: SETTINGS_KEY, guided: GUIDED_FORMAT.key, text: game && GUIDED_FORMAT.encode(game) });
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/?mode=${mode}`, { waitUntil: 'networkidle0' });
  return { page, problems };
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
    }
    assert.deepEqual(problems, []);
  });
  for (const offered of ['1m', '2m', '3m']) {
    await check(`guided chii rotates ${offered} after reload and Undo removes the set`, async context => {
      const { page, problems } = await pageAt(context, 'guided', calledGame(offered));
      const actual = () => page.$eval('.your-hand .meld .tile.rotated', el => el.dataset.tile);
      await page.waitForSelector('.guided-controls:not(:disabled)');
      assert.equal(await actual(), offered);
      await page.reload({ waitUntil: 'networkidle0' });
      await page.waitForSelector('.guided-controls:not(:disabled)');
      assert.equal(await actual(), offered);
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
