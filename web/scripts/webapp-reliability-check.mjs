/** Production-UI regressions for reconnect availability and physical Undo.
 * The HEAD probe is controlled; no trained inference or model download is needed.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import puppeteer from 'puppeteer-core';
import { MANIFEST } from '../src/lib/model-manifest.js';
import { emptyPosition, PHYSICAL_KEY } from '../src/lib/physical-position.js';
import { SAVE_KEY } from '../src/lib/session.js';
import { createFixtureHandler } from './static-fixture-server.mjs';

const root = resolve(import.meta.dirname, '..');
const output = resolve(root, 'test-results');
const networkUrl = `${MANIFEST.origin}/${MANIFEST.object}`;
const server = createServer(createFixtureHandler({ root: resolve(root, 'dist'), publicRoot: resolve(root, 'public') }));
const results = [];
let browser;
async function check(name, run) {
  const context = await browser.createBrowserContext();
  try { await run(context); results.push({ name, passed: true }); console.log(`PASS ${name}`); }
  catch (error) { results.push({ name, passed: false, error: error.stack }); console.error(`FAIL ${name}\n${error.stack}`); }
  finally { await context.close(); }
}
async function open(context, { mode = 'play', width = 1100, fixture, available = () => false } = {}) {
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewport({ width, height: 900, hasTouch: width < 600 });
  await page.setRequestInterception(true);
  page.on('request', request => {
    const answer = request.url() === networkUrl && request.method() === 'HEAD'
      ? request.respond({ status: available() ? 200 : 404, headers: { 'Access-Control-Allow-Origin': '*' }, body: '' })
      : request.continue();
    void answer.catch(() => {}); // A context can close while a request is pending.
  });
  if (fixture) await page.evaluateOnNewDocument(({ key, value, match }) => {
    if (localStorage.getItem(key) === null) localStorage.setItem(key, JSON.stringify({ version: 1, position: value }));
    if (localStorage.getItem(match) === null) localStorage.setItem(match, 'preserved regular match');
  }, { key: PHYSICAL_KEY, value: fixture, match: SAVE_KEY });
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/?mode=${mode}&opponents=club`, { waitUntil: 'networkidle0' });
  await page.waitForSelector(mode === 'physical' ? '.physical-editor:not(:disabled)' : '.hand');
  return { page, errors };
}
async function mode(page, name) {
  await page.evaluate(name => {
    const button = [...document.querySelectorAll('.game-modes button')].find(button => button.textContent.trim() === name);
    if (!button || button.disabled) throw new Error(`Unavailable mode: ${name}`);
    button.click();
  }, name);
}
const saved = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)).position, PHYSICAL_KEY);
async function expectPosition(page, expected) {
  await page.waitForFunction(({ key, expected }) => JSON.stringify(JSON.parse(localStorage.getItem(key))?.position) === JSON.stringify(expected), {}, { key: PHYSICAL_KEY, expected });
  assert.deepEqual(await saved(page), expected);
}
async function number(page, selector, value) {
  await page.focus(selector);
  await page.keyboard.down('Control'); await page.keyboard.press('KeyA'); await page.keyboard.up('Control');
  await page.keyboard.press('Backspace');
  if (value !== '') await page.keyboard.type(value);
  await page.keyboard.press('Tab');
}
async function undo(page) {
  await page.click('.physical-heading button:first-of-type');
}
try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  const executablePath = process.env.CHROME_BIN || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(executablePath, 'Set CHROME_BIN to Chrome/Chromium');
  browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  await check('reconnect enables every Trained selector without reloading or replacing the regular match', async context => {
    let reachable = false;
    const { page, errors } = await open(context, { available: () => reachable });
    assert.equal(await page.$('.opponents option[value="neural"]'), null);
    const before = await page.evaluate(key => localStorage.getItem(key), SAVE_KEY);
    reachable = true;
    await page.evaluate(() => window.dispatchEvent(new Event('online')));
    await page.waitForSelector('.opponents option[value="neural"]');
    assert.equal(await page.$eval('.opponents select', select => select.value), 'club');
    assert.equal(await page.evaluate(key => localStorage.getItem(key), SAVE_KEY), before);
    await mode(page, 'Agent watch'); await page.waitForSelector('[aria-label="Followed agent"] option[value="full"]');
    await mode(page, 'Physical agent play'); await page.waitForSelector('[aria-label="Physical play agent"] option[value="full"]');
    await mode(page, 'Guided physical game'); await page.waitForSelector('[aria-label="Guided game adviser"] option[value="full"]');
    assert.equal(await page.evaluate(key => localStorage.getItem(key), SAVE_KEY), before);
    assert.deepEqual(errors, []);
  });
  for (const width of [1100, 390]) await check(`physical edits undo independently and numeric typing stays usable at ${width}px`, async context => {
    const initial = emptyPosition(); initial.players[0].score = 31000;
    const { page, errors } = await open(context, { mode: 'physical', width, fixture: initial });
    const points = '.physical-seat:first-child .fields input[type="number"]';
    const wall = '.table-fields input[max="70"]';
    // A numeric change on its own creates one step, including all keystrokes.
    await number(page, points, '32000');
    const onlyPoints = structuredClone(initial); onlyPoints.players[0].score = 32000;
    await expectPosition(page, onlyPoints); await undo(page); await expectPosition(page, initial);
    assert.equal(await page.$eval('.physical-heading button:first-of-type', button => button.disabled), true);
    await page.click('.physical-seat:first-child .tile-entry summary');
    await page.click('.physical-seat:first-child .palette button[data-tile="1m"]');
    const tileOnly = structuredClone(initial); tileOnly.players[0].hand.push('1m');
    await expectPosition(page, tileOnly);
    await number(page, points, '32000');
    const scored = structuredClone(tileOnly); scored.players[0].score = 32000;
    await expectPosition(page, scored);
    await number(page, wall, '60');
    const counted = structuredClone(scored); counted.wall = 60;
    await expectPosition(page, counted);
    await page.click('.physical-seat:first-child .flags label:last-child input');
    const flagged = structuredClone(counted); flagged.players[0].furiten = true;
    await expectPosition(page, flagged);
    for (const expected of [counted, scored, tileOnly, initial]) { await undo(page); await expectPosition(page, expected); }
    // Native number bindings must still allow blanks and a typed minus sign.
    for (const value of ['-100', '']) {
      await number(page, points, value);
      const expected = structuredClone(initial); expected.players[0].score = value === '' ? null : -100;
      await expectPosition(page, expected); await undo(page); await expectPosition(page, initial);
    }
    await page.reload({ waitUntil: 'networkidle0' });
    await page.waitForSelector('.physical-editor:not(:disabled)');
    await expectPosition(page, initial);
    assert.equal(await page.evaluate(key => localStorage.getItem(key), SAVE_KEY), 'preserved regular match');
    assert.deepEqual(errors, []);
  });
} finally {
  await browser?.close();
  await new Promise(done => server.close(done));
  await writeFile(resolve(output, 'webapp-reliability.json'), JSON.stringify(results, null, 2));
}
console.log(`${results.filter(result => result.passed).length}/${results.length} webapp reliability checks passed`);
if (results.some(result => !result.passed)) process.exitCode = 1;
