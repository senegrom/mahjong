/** Guided physical games in the production UI, using the real engine and policy. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import puppeteer from 'puppeteer-core';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { parseTiles, PHYSICAL_KEY } from '../src/lib/physical-position.js';
import { GUIDED_KEY, GUIDED_FORMAT, emptyGuided, guidedEvent } from '../src/lib/guided-game.js';
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
  // A preceding action may still be rendering its next enabled control.
  await page.waitForFunction(text => [...document.querySelectorAll('.guided-play button')]
    .some(button => button.textContent.trim() === text && !button.matches(':disabled')), {}, text);
  await page.evaluate(text => {
    const button = [...document.querySelectorAll('.guided-play button')].find(b => b.textContent.trim() === text);
    if (!button || button.matches(':disabled')) throw new Error(`Missing enabled button: ${text}`);
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
async function loadScoringFixture(page, riichi = false) {
  let game = emptyGuided();
  const p = game.state.position;
  Object.assign(p, { seat: 3, turn: 0, phase: 'call', wall: 50, first_turns: false, indicators: ['7z'], pending: '2z' });
  p.players[3].hand = parseTiles('123m456p789s1112z');
  p.players[0].discards = [{ tile: '2z', order: riichi ? 1 : 0, riichi: false, drawn: false, claimed: false }];
  if (riichi) {
    p.players[3].score = 29000; p.players[3].riichi = 'riichi'; p.riichi_sticks = 1;
    p.players[3].discards = [{ tile: '9p', order: 0, riichi: true, drawn: false, claimed: false }];
  }
  game.state.stage = 'decision'; game.state.opening = [30000,30000,30000,30000];
  game = guidedEvent(game, { type: 'choice', choice: { kind: 'ron' }, choices: [{ kind: 'ron' }] });
  await page.evaluate(({ key, text, settings }) => {
    localStorage.setItem(key, text);
    localStorage.setItem(settings, JSON.stringify({ version: 1, difficulty: 'club', hints: false }));
  }, { key: GUIDED_KEY, text: GUIDED_FORMAT.encode(game), settings: SETTINGS_KEY });
  await page.reload({ waitUntil: 'networkidle0' });
  await page.waitForSelector('.guided-controls:not(:disabled)'); await stage(page, 'over');
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
    await button(page, 'Record opponent call'); await stage(page, 'claim-response');
    await page.reload({ waitUntil: 'networkidle0' }); await stage(page, 'claim-response');
    await page.click('.no-calls'); await stage(page, 'turn');
    assert.equal((await saved(page)).state.nextSeat, 0);
    await tile(page, '6z'); await choice(page);
    assert.equal((await saved(page)).state.position.wall, 66, 'pon discard has no draw');
    await page.click('.end-hand summary');
    await page.select('.end-hand select', 'Other hand end'); // Explicit manual adjudication, not a fabricated tsumo.
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
    // Chrome may suspend animation-frame work in the background tab. Clicks
    // and waitForFunction must run in the tab a real user has activated.
    await a.bringToFront();
    await tile(a, '9m'); await choice(a);
    await b.bringToFront();
    await b.waitForFunction(() => document.querySelector('.guided-play [role="alert"]')?.textContent.includes('Another window'));
    assert.equal(await b.$eval('.guided-controls', el => el.disabled), true);
    await button(b, 'Reload saved game'); await choice(b);
    assert.equal((await saved(b)).state.position.players[0].discards[0].tile, '9m');
    assert.deepEqual(b.problems, []);
  });
  await check('scored ron preview, apply, reload, undo and next hand preserve exact balances', async context => {
    const page = await open(context);
    await loadScoringFixture(page);
    const unpaid = (await saved(page)).state;
    await button(page, 'Calculate hand settlement');
    await page.waitForSelector('[aria-label="North scored hand"]');
    assert.match(await page.$eval('[aria-label="North scored hand"]', el => el.textContent), /1 han · 40 fu/);
    assert.match(await page.$eval('[aria-label="North scored hand"]', el => el.textContent), /Round Wind Triplet/);
    assert.deepEqual((await saved(page)).state.position.players.map(p => p.score), [30000,30000,30000,30000], 'preview does not pay');
    await button(page, 'Apply settlement');
    await page.waitForFunction(key => Boolean(JSON.parse(localStorage.getItem(key)).game.state.settlement), {}, GUIDED_KEY);
    const paid = (await saved(page)).state;
    assert.deepEqual(paid.settlement.deltas, [-1300,0,0,1300]);
    assert.deepEqual(paid.position.players.map(p => p.score), [28700,30000,30000,31300]);
    await page.reload({ waitUntil: 'networkidle0' }); await stage(page, 'over');
    await page.waitForSelector('.guided-controls:not(:disabled)');
    assert.deepEqual((await saved(page)).state, paid, 'reload verifies without paying twice');
    assert.equal(await page.$$eval('.guided-result button', buttons => buttons.some(b => b.textContent === 'Apply settlement')), false);
    await button(page, 'Undo last step');
    // Undo stays on the over stage. Wait for its asynchronous, locked save,
    // not just the click or the already-present stage, before reading balances.
    await page.waitForFunction(key => {
      const state = JSON.parse(localStorage.getItem(key))?.game.state;
      return state?.stage === 'over' && state.ending && !state.settlement;
    }, {}, GUIDED_KEY);
    assert.deepEqual((await saved(page)).state.position.players.map(p => p.score), [30000,30000,30000,30000]);
    assert.deepEqual((await saved(page)).state, unpaid, 'undo restores the entire unpaid state, including the pot');
    await button(page, 'Calculate hand settlement'); await button(page, 'Apply settlement');
    await page.waitForFunction(key => Boolean(JSON.parse(localStorage.getItem(key)).game.state.settlement), {}, GUIDED_KEY);
    assert.deepEqual((await saved(page)).state, paid, 'reapplying after undo produces the same settlement exactly once');
    await page.screenshot({ path: resolve(output, 'guided-scoring-desktop.png'), fullPage: true });
    await button(page, 'Next hand · dealer moves'); await stage(page, 'setup');
    assert.deepEqual((await saved(page)).state.position.players.map(p => p.score), [30000,30000,31300,28700]);
    assert.deepEqual(page.problems, []);
  });
  await check('mobile riichi settlement requires ura and displays it with hints off', async context => {
    const page = await open(context, 360);
    await loadScoringFixture(page, true);
    await button(page, 'Calculate hand settlement');
    await page.waitForFunction(() => document.querySelector('.guided-result [role="alert"]')?.textContent.includes('ura'));
    assert.equal((await saved(page)).state.settlement, undefined);
    await page.click('[aria-label="Revealed ura-dora indicators"] button[data-tile="6z"]');
    await button(page, 'Calculate hand settlement');
    await page.waitForSelector('[aria-label="North ura-dora indicators"]');
    assert.equal(await page.$$eval('[aria-label="North ura-dora indicators"] .tile', nodes => nodes.length), 1);
    assert.match(await page.$eval('[aria-label="North scored hand"]', el => el.textContent), /Ura-dora: 0 han/);
    await button(page, 'Apply settlement');
    await page.waitForFunction(key => Boolean(JSON.parse(localStorage.getItem(key)).game.state.settlement), {}, GUIDED_KEY);
    const paid = (await saved(page)).state;
    assert.deepEqual(paid.settlement.hand_deltas, [-2600,0,0,2600]);
    assert.equal(paid.position.players[3].score, 32600);
    assert.equal(paid.position.riichi_sticks, 0);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'no horizontal overflow');
    await page.screenshot({ path: resolve(output, 'guided-scoring-mobile.png'), fullPage: true });
    await page.reload({ waitUntil: 'networkidle0' }); await stage(page, 'over');
    await page.waitForSelector('[aria-label="North ura-dora indicators"]');
    assert.deepEqual((await saved(page)).state, paid);
    assert.deepEqual(page.problems, []);
  });
} finally {
  await writeFile(resolve(output, 'guided-game.json'), JSON.stringify(results, null, 2));
  await browser?.close(); await new Promise(done => server.close(done));
}
if (results.some(result => !result.passed)) process.exitCode = 1;
