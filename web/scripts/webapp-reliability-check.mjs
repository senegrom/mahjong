/** Production-UI regressions for reconnect availability and physical Undo.
 * The HEAD probe is controlled; no trained inference or model download is needed.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import { MANIFEST } from '../src/lib/model-manifest.js';
import { emptyPosition, parseTiles, PHYSICAL_KEY, recordDraw } from '../src/lib/physical-position.js';
import { SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { createFixtureHandler } from './static-fixture-server.mjs';

const root = resolve(import.meta.dirname, '..');
const output = resolve(root, 'test-results');
const networkUrl = `${MANIFEST.origin}/${MANIFEST.object}`;
const server = createServer(createFixtureHandler({ root: resolve(root, 'dist'), publicRoot: resolve(root, 'public') }));
const cases = browserChecks();
let browser;
const check = (name, run) => cases.check(name, async () => run(await cases.openContext(browser)));
async function open(context, { mode = 'play', width = 1100, fixture, available = () => false, countSettings = false } = {}) {
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  // Count the page's own writes of the preferences; `otherTab` writes as another tab would.
  if (countSettings) await page.evaluateOnNewDocument(key => {
    const setItem = Storage.prototype.setItem;
    window.settingsWrites = 0;
    window.otherTab = value => setItem.call(localStorage, key, value);
    Storage.prototype.setItem = function (name, value) {
      if (name === key) window.settingsWrites++;
      return setItem.call(this, name, value);
    };
  }, SETTINGS_KEY);
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
  browser = await launchChrome();
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
  await check('moves leave unchanged preferences alone, including another tab\'s change', async context => {
    const { page, errors } = await open(context, { countSettings: true });
    const before = await page.evaluate(() => window.settingsWrites);
    const otherTab = JSON.stringify({ ...JSON.parse(await page.evaluate(key => localStorage.getItem(key), SETTINGS_KEY)), hints: false });
    await page.evaluate(value => window.otherTab(value), otherTab);
    for (let move = 0; move < 4; move++) {
      await page.waitForFunction(() => document.querySelector('.failure, .screen, .hand button:not(:disabled), .call-options button:not(:disabled)'), { timeout: 45000 });
      if (await page.$('.screen, .failure')) break;
      const commands = await page.evaluate(key => JSON.parse(localStorage.getItem(key)).commands.length, SAVE_KEY);
      const call = await page.$('.call-options [data-choice=pass], .call-options [data-choice=ron], .call-options [data-choice=tsumo]');
      await (call ?? await page.$('.hand button:not(:disabled)')).click();
      if (await page.$('.confirm-discard .primary')) await page.click('.confirm-discard .primary');
      await page.waitForFunction((key, count) => JSON.parse(localStorage.getItem(key)).commands.length > count, { timeout: 45000 }, SAVE_KEY, commands);
    }
    assert.ok(await page.evaluate(key => JSON.parse(localStorage.getItem(key)).commands.length, SAVE_KEY) >= 2, 'moves were played');
    assert.equal(await page.evaluate(() => window.settingsWrites), before, 'no move rewrites the preferences');
    assert.equal(await page.evaluate(key => localStorage.getItem(key), SETTINGS_KEY), otherTab);
    assert.deepEqual(errors, []);
  });
  await check('the physical editor shows each discard\'s marks on its tile as they are ticked', async context => {
    const initial = emptyPosition(); initial.first_turns = false;
    initial.players[0].discards = [
      { tile: '3p', order: 0, drawn: true, riichi: false, claimed: false },
      { tile: '6z', order: 1, drawn: false, riichi: false, claimed: true },
      { tile: '9m', order: 2, drawn: false, riichi: false, claimed: false },
    ];
    const { page, errors } = await open(context, { mode: 'physical', fixture: initial });
    // Each tile: its name, and the border's colour and the shade on the
    // layer over its face.
    const tiles = () => page.$$eval('.physical-seat:first-child .discard-fields', rows => rows.map(row => {
      const tile = row.querySelector('.tile'), layer = getComputedStyle(tile.querySelector('.face'), '::after'), marked = layer.content !== 'none';
      return [tile.getAttribute('aria-label'), marked && layer.borderTopStyle !== 'none' ? layer.borderTopColor : 'none', marked ? layer.backgroundColor : 'none'];
    }));
    assert.deepEqual(await tiles(), [['3 circles, discarded from the draw', 'none', 'rgba(0, 0, 0, 0.28)'],
      ['green dragon, claimed', 'rgb(14, 110, 51)', 'rgba(0, 0, 0, 0)'], ['9 characters', 'none', 'none']]);
    // Ticking the boxes marks the tile beside them, as the discard row will.
    for (const name of ['From draw', 'Claimed']) await page.evaluate(name => [...document.querySelectorAll('.physical-seat:first-child .discard-fields')][2]
      .querySelectorAll('label').forEach(label => { if (label.textContent.trim() === name) label.querySelector('input').click(); }), name);
    const marked = structuredClone(initial); Object.assign(marked.players[0].discards[2], { drawn: true, claimed: true });
    await expectPosition(page, marked);
    assert.deepEqual((await tiles())[2], ['9 characters, claimed, discarded from the draw', 'rgb(14, 110, 51)', 'rgba(0, 0, 0, 0.28)']);
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
  for (const width of [1100, 390]) await check(`the physical editor's panels and palettes stay as the player leaves them at ${width}px`, async context => {
    const initial = emptyPosition();
    const { page, errors } = await open(context, { mode: 'physical', width, fixture: initial });
    const seat = place => `.physical-seat:nth-child(${place})`;
    const isOpen = selector => page.$eval(selector, details => details.open);
    // East is the analysed seat, so its panel is open. Its palette is opened
    // once and takes two tiles running: it used to shut after the first.
    const hand = `${seat(1)} .seat-content > .tile-entry`;
    await page.click(`${hand} summary`);
    for (const tile of ['1m', '2m']) await page.click(`${hand} .palette button[data-tile="${tile}"]`);
    const entered = structuredClone(initial); entered.players[0].hand = ['1m', '2m'];
    await expectPosition(page, entered);
    assert.equal(await isOpen(`${hand} details`), true, 'the palette stays open');
    // South is not analysed. Its panel is opened by its summary and a whole
    // score typed into it: it used to shut at the first key, and the rest of
    // the number was typed at the page.
    await page.click(`${seat(2)} > summary`);
    await number(page, `${seat(2)} .fields input[type="number"]`, '25000');
    const scored = structuredClone(entered); scored.players[1].score = 25000;
    await expectPosition(page, scored);
    assert.deepEqual([await isOpen(seat(1)), await isOpen(seat(2))], [true, true], 'both panels as the player left them');
    // A panel the player folds stays folded through an edit elsewhere, and
    // analysing another seat opens that seat's panel and folds none.
    await page.click(`${seat(1)} > summary`);
    await number(page, `${seat(2)} .fields input[type="number"]`, '26000');
    const again = structuredClone(scored); again.players[1].score = 26000;
    await expectPosition(page, again);
    assert.deepEqual([await isOpen(seat(1)), await isOpen(seat(2))], [false, true], 'the analysed seat stays folded');
    await page.select('[aria-label="Analyse seat"]', '2');
    const west = structuredClone(again); west.seat = 2; west.turn = 2;
    await expectPosition(page, west);
    assert.deepEqual([await isOpen(seat(1)), await isOpen(seat(2)), await isOpen(seat(3))], [false, true, true]);
    assert.deepEqual(errors, []);
  });
  await check('a refused physical move is said under the palette it was pressed in, and a number out of range by name', async context => {
    const initial = emptyPosition(); initial.players[0].hand = parseTiles('123m456p789s1123z');
    const { page, errors } = await open(context, { mode: 'physical', width: 390, fixture: initial });
    const entry = '.record-event .tile-entry', move = '.record-event .fields label:nth-child(2) select';
    // East holds thirteen tiles, so a discard before its draw is refused.
    await page.select(move, 'discard');
    await page.click(`${entry} summary`);
    await page.click(`${entry} .palette button[data-tile="1m"]`);
    await page.waitForSelector(`${entry} [role="alert"]`);
    assert.equal(await page.$eval(`${entry} [role="alert"]`, el => el.textContent), 'Record this player’s draw or call before discarding');
    assert.equal(await page.$('.physical-play .failure'), null, 'not in the line under the toolbar, a screen above');
    assert.deepEqual(await saved(page), initial, 'a refused move changes nothing');
    // The draw is taken from the same palette, and the refusal goes.
    await page.select(move, 'draw');
    await page.click(`${entry} .palette button[data-tile="4z"]`);
    const drawn = recordDraw(initial, 0, '4z');
    await expectPosition(page, drawn);
    assert.equal(await page.$(`${entry} [role="alert"]`), null);
    // A count below nought is asked for by name, not answered by the engine.
    await number(page, '.table-fields input[max="100"]', '-1');
    await expectPosition(page, { ...drawn, counters: -1 });
    await page.click('.physical-toolbar .primary');
    await page.waitForSelector('.physical-play .failure');
    assert.equal(await page.$eval('.physical-play .failure', el => el.textContent), 'The honba count must be 0 to 100');
    assert.deepEqual(errors, []);
  });
} finally {
  await browser?.close();
  await new Promise(done => server.close(done));
  await cases.report('webapp reliability checks', resolve(output, 'webapp-reliability.json'));
}
