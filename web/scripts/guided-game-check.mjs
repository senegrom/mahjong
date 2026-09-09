/** Guided physical games in the production UI, using the real engine and policy. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import puppeteer from 'puppeteer-core';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { parseTiles, PHYSICAL_KEY } from '../src/lib/physical-position.js';
import { GUIDED_KEY } from '../src/lib/guided-game.js';
import { SETTINGS_KEY, SAVE_KEY } from '../src/lib/session.js';

const root = resolve(import.meta.dirname, '..'), output = resolve(root, 'test-results');
const server = createServer(createFixtureHandler({ root: resolve(root, 'dist'), publicRoot: resolve(root, 'public') }));
let browser;
const results = [];
const saved = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)).game, GUIDED_KEY);
async function stage(page, value) {
  await page.waitForSelector(`.guide-prompt[data-stage="${value}"]`);
  await page.waitForFunction(({ key, value }) => JSON.parse(localStorage.getItem(key))?.game.state.stage === value, {}, { key: GUIDED_KEY, value });
}
async function tile(page, value) { await page.click(`.guide-prompt .palette button[data-tile="${value}"]`); }
async function choice(page) { await page.waitForSelector('.record-best'); await stage(page, 'decision'); }
async function button(page, text) {
  await page.evaluate(text => {
    const button = [...document.querySelectorAll('.guided-play button')].find(b => b.textContent.trim() === text);
    if (!button || button.disabled) throw new Error(`Missing enabled button: ${text}`);
    button.click();
  }, text);
}
async function check(name, run) {
  const context = await browser.createBrowserContext();
  try { await run(context); results.push({ name, passed: true }); console.log(`PASS ${name}`); }
  catch (error) { results.push({ name, passed: false, error: error.stack }); console.error(`FAIL ${name}\n${error.stack}`); }
  finally { await context.close(); }
}
async function open(context, width = 1100) {
  const page = await context.newPage(); page.problems = [];
  page.on('pageerror', error => page.problems.push(error.message));
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1, hasTouch: width < 600 });
  await page.evaluateOnNewDocument(({ settings, physical, match }) => {
    if (!localStorage.getItem(settings)) localStorage.setItem(settings, JSON.stringify({ version: 1, difficulty: 'club', hints: true }));
    if (!localStorage.getItem(physical)) localStorage.setItem(physical, 'existing position editor draft');
    if (!localStorage.getItem(match)) localStorage.setItem(match, 'existing regular match');
  }, { settings: SETTINGS_KEY, physical: PHYSICAL_KEY, match: SAVE_KEY });
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/?mode=guided`, { waitUntil: 'networkidle0' });
  await page.waitForSelector('.guided-controls:not(:disabled)');
  return page;
}
async function setup(page, seat = '3') {
  await page.select('[aria-label="Your seat"]', seat);
  await page.click('.guide-prompt > .primary'); await stage(page, 'hand');
  for (const t of parseTiles('123m456p789s1123z')) await tile(page, t);
  assert.equal(await page.$$eval('.guide-prompt .entered .tile', e => e.length), 13);
  await page.click('.guide-prompt > .primary'); await stage(page, 'dora');
  await tile(page, '7z'); await stage(page, 'turn');
}
async function passAndContinue(page) {
  await choice(page); await page.click('.record-best'); await stage(page, 'responses');
  await page.click('.no-calls'); await stage(page, 'turn');
}
try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  const executablePath = process.env.CHROME_BIN || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(executablePath, 'Set CHROME_BIN to Chrome/Chromium');
  browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  await check('North walkthrough, automatic suggestions, cancellation, saved undo, opponent call and next hand', async context => {
    const page = await open(context);
    await setup(page);
    for (const [seat, t] of ['9m', '8m', '7m'].entries()) {
      assert.equal((await saved(page)).state.nextSeat, seat);
      await tile(page, t); await passAndContinue(page);
    }
    assert.match(await page.$eval('.guide-prompt h3', el => el.textContent), /draw/);
    await tile(page, '4z'); await choice(page);
    let game = await saved(page);
    assert.equal(game.state.position.wall, 66);
    assert.equal(game.state.position.players[3].hand.length, 14);
    assert.ok(await page.$('.held-tiles .tile.selected'));
    assert.equal(await page.$eval('.weight-row.best', el => getComputedStyle(el).borderColor), 'rgb(78, 163, 255)');
    const selector = '.weight-row:not(.best) .choice-action';
    page.once('dialog', dialog => dialog.dismiss()); await page.click(selector);
    assert.deepEqual((await saved(page)).state, game.state);
    page.once('dialog', dialog => dialog.accept()); await page.click(selector);
    await stage(page, 'responses');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).game.state.stage === 'responses', {}, GUIDED_KEY);
    const recorded = await saved(page);
    assert.equal(recorded.state.position.players[3].hand.length, 13);
    await page.reload({ waitUntil: 'networkidle0' }); await stage(page, 'responses');
    assert.deepEqual((await saved(page)).state, recorded.state);
    await button(page, 'Undo last step'); await choice(page);
    assert.deepEqual((await saved(page)).state, game.state);
    await page.screenshot({ path: resolve(output, 'guided-game-desktop.png'), fullPage: true });
    // A specific legal alternative lets East call a pon from our discard.
    const tileChoice = '.held-tiles button[data-tile="4z"]';
    page.once('dialog', dialog => dialog.accept()); await page.click(tileChoice);
    await stage(page, 'responses');
    await page.click('.guide-prompt details summary');
    await page.select('[aria-label="Who called?"]', '0');
    await page.select('[aria-label="Opponent call"]', 'pon');
    await button(page, 'Record opponent call'); await stage(page, 'turn');
    assert.equal((await saved(page)).state.nextSeat, 0);
    await tile(page, '6z'); await choice(page);
    assert.equal((await saved(page)).state.position.wall, 66, 'pon discard has no draw');
    await page.click('.end-hand summary');
    page.once('dialog', dialog => dialog.accept()); await button(page, 'Record hand result');
    await stage(page, 'over'); await button(page, 'Next hand · dealer moves'); await stage(page, 'setup');
    game = await saved(page);
    assert.equal(game.state.position.seat, 2); assert.equal(game.state.position.kyoku, 2);
    assert.ok(game.log.length > 10);
    assert.deepEqual(await page.evaluate(({ physical, match }) => [localStorage.getItem(physical), localStorage.getItem(match)], { physical: PHYSICAL_KEY, match: SAVE_KEY }), ['existing position editor draft', 'existing regular match']);
    assert.deepEqual(page.problems, []);
  });
  await check('mobile guided play shows real Strong percentages and resumes the pending decision', async context => {
    const page = await open(context, 360);
    await setup(page, '0');
    await tile(page, '4z'); await choice(page);
    await page.waitForSelector('[aria-label="Guided game adviser"] option[value="strong"]');
    await page.select('[aria-label="Guided game adviser"]', 'strong');
    await page.waitForFunction(() => document.querySelector('.recommendation strong')?.textContent.includes('Strong') && document.querySelector('.weight-row meter'), { timeout: 120000 });
    const weights = await page.$$eval('.weight-row meter', meters => meters.map(m => Number(m.value)));
    assert.ok(Math.abs(weights.reduce((a, b) => a + b, 0) - 1) < 1e-5);
    assert.equal(await page.$$eval('.held-tiles .copy-count', nodes => nodes.length), 14);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'no horizontal overflow');
    await page.screenshot({ path: resolve(output, 'guided-game-mobile.png'), fullPage: true });
    await page.reload({ waitUntil: 'networkidle0' }); await choice(page);
    assert.equal(await page.$eval('[aria-label="Guided game adviser"]', el => el.value), 'strong');
    assert.equal((await saved(page)).state.position.players[0].hand.length, 14);
    assert.deepEqual(page.problems, []);
  });
  await check('two windows stop conflicting edits and reload the newer guided prompt', async context => {
    const a = await open(context); await setup(a);
    const b = await open(context); await stage(b, 'turn');
    await tile(a, '9m'); await choice(a);
    await b.waitForFunction(() => document.querySelector('.guided-play [role="alert"]')?.textContent.includes('Another window'));
    assert.equal(await b.$eval('.guided-controls', el => el.disabled), true);
    await button(b, 'Reload saved game'); await choice(b);
    assert.equal((await saved(b)).state.position.players[0].discards[0].tile, '9m');
    assert.deepEqual(b.problems, []);
  });
} finally {
  await writeFile(resolve(output, 'guided-game.json'), JSON.stringify(results, null, 2));
  await browser?.close(); await new Promise(done => server.close(done));
}
if (results.some(result => !result.passed)) process.exitCode = 1;
