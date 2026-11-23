/** Guided physical games in the production UI, using the real engine and policy. */
import assert from 'node:assert/strict';
import { MANIFEST } from '../src/lib/model-manifest.js';
import { createServer } from 'node:http';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import { observeNetworkRequests } from './network-requests.mjs';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { emptyPosition, parseTiles, PHYSICAL_KEY } from '../src/lib/physical-position.js';
import { GUIDED_KEY, GUIDED_FORMAT, emptyGuided, guidedEvent } from '../src/lib/guided-game.js';
import { SETTINGS_KEY, SAVE_KEY } from '../src/lib/session.js';

const root = resolve(import.meta.dirname, '..'), output = resolve(root, 'test-results');
const server = createServer(createFixtureHandler({ root: resolve(root, 'dist'), publicRoot: resolve(root, 'public') }));
let browser;
const cases = browserChecks();
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
const check = (name, run) => cases.check(name, async () => run(await cases.openContext(browser)));
async function open(context, width = 1100, { probe } = {}) {
  const page = await context.newPage(); page.problems = [];
  page.on('pageerror', error => page.problems.push(error.message));
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1, hasTouch: width < 600 });
  if (probe !== undefined) await page.evaluateOnNewDocument(({ url, available }) => {
    const realFetch = window.fetch;
    window.fetch = (input, options) => {
      const target = input instanceof Request ? input.url : String(input);
      const method = options?.method ?? (input instanceof Request ? input.method : 'GET');
      if (target === url && method.toUpperCase() === 'HEAD') {
        return Promise.resolve(new Response(null, { status: available ? 200 : 404 }));
      }
      return realFetch(input, options);
    };
  }, { url: `${MANIFEST.origin}/${MANIFEST.object}`, available: probe });
  await page.evaluateOnNewDocument(({ settings, physical, match }) => {
    if (!localStorage.getItem(settings)) localStorage.setItem(settings, JSON.stringify({ version: 2, difficulty: 'club', hints: true }));
    if (!localStorage.getItem(physical)) localStorage.setItem(physical, 'existing position editor draft');
    if (!localStorage.getItem(match)) localStorage.setItem(match, 'existing regular match');
  }, { settings: SETTINGS_KEY, physical: PHYSICAL_KEY, match: SAVE_KEY });
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/?mode=guided`, { waitUntil: 'networkidle0' });
  await page.waitForSelector('.guided-controls:not(:disabled)');
  return page;
}
const adviser = '[aria-label="Guided game adviser"]';
/** A new guided game is advised by the trained network. Checks of the guide
 * itself choose Club, so they never wait on the network's download. */
async function club(page) {
  assert.equal(await page.$eval(adviser, el => el.value), 'full', 'a new guided game starts on the trained adviser');
  await page.select(adviser, 'club');
  await page.waitForFunction(key => JSON.parse(localStorage.getItem(key))?.game.state.agent === 'club', {}, GUIDED_KEY);
}
/** Whether focus is on the prompt's heading, where each step hands it: the
 * control that was pressed has gone with the prompt it was in. */
const onHeading = page => page.evaluate(() => document.activeElement === document.querySelector('.guide-prompt h3'));
async function setup(page, seat = '3') {
  await page.select('[aria-label="Your seat"]', seat);
  await page.click('.guide-prompt > .primary'); await stage(page, 'hand');
  assert.equal(await onHeading(page), true, 'focus goes on to the starting hand\'s prompt');
  for (const t of parseTiles('123m456p789s1123z')) await tile(page, t);
  assert.equal(await page.$$eval('.guide-prompt .entered .tile', e => e.length), 13);
  await page.click('.guide-prompt > .primary'); await stage(page, 'dora');
  await tile(page, '7z'); await stage(page, 'turn');
  assert.equal(await onHeading(page), true, 'and on to the first turn\'s');
}
async function passAndContinue(page) {
  await choice(page);
  assert.equal(await onHeading(page), true, 'a discard entered, focus is on the advice\'s prompt');
  await page.click('.record-best'); await stage(page, 'responses');
  assert.equal(await onHeading(page), true, 'a move recorded, on the responses\' prompt');
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
    localStorage.setItem(settings, JSON.stringify({ version: 2, difficulty: 'club', hints: false }));
  }, { key: GUIDED_KEY, text: GUIDED_FORMAT.encode(game), settings: SETTINGS_KEY });
  await page.reload({ waitUntil: 'networkidle0' });
  await page.waitForSelector('.guided-controls:not(:disabled)'); await stage(page, 'over');
}
try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  browser = await launchChrome();
  await check('North walkthrough, automatic suggestions, cancellation, saved undo, opponent call and next hand', async context => {
    const page = await open(context);
    await club(page);
    await setup(page);
    for (const [seat, t] of ['9m', '8m', '7m'].entries()) {
      assert.equal((await saved(page)).state.nextSeat, seat);
      await tile(page, t); await passAndContinue(page);
    }
    // East's first discard, entered as the 9 characters, was the 9 circles:
    // it is put right in place, a step of its own that Undo takes back.
    const first = () => page.evaluate(key => JSON.parse(localStorage.getItem(key)).game.state.position.players[0].discards[0].tile, GUIDED_KEY);
    await page.click('.remembered .correct > summary');
    await page.select('[aria-label="Where the tile to correct is"]', '0');
    await page.select('[aria-label="The tile to correct"]', '0');
    await page.click('.correct .tile-entry summary');
    // A correction that is refused is said under the palette it was pressed in.
    await page.click('.correct .palette button[data-tile="9m"]');
    await page.waitForFunction(() => document.querySelector('.correct [role="alert"]')?.textContent === 'It is already that tile');
    assert.equal(await page.$('.guide-prompt .failure'), null);
    await page.click('.correct .palette button[data-tile="9p"]');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).game.log.at(-1) === 'Corrected East’s discard: 9 characters → 9 circles', {}, GUIDED_KEY);
    assert.equal(await first(), '9p');
    assert.equal(await page.$('.correct [role="alert"]'), null);
    assert.equal(await page.$eval('.guided-seats section:first-child .pool .tile', el => el.dataset.tile), '9p', 'the remembered table shows it');
    await button(page, 'Undo last step');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).game.state.position.players[0].discards[0].tile === '9m', {}, GUIDED_KEY);
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
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).game.state.stage === 'claim-response', {}, GUIDED_KEY);
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
  await check('a negative score keeps its minus sign while it is typed', async context => {
    const page = await open(context);
    const field = '[aria-label="Your points"]';
    // Replace the whole value the usual way: select it, then type. Before
    // the fix the lone minus emptied the field and this saved 500.
    await page.focus(field);
    await page.keyboard.down('Control'); await page.keyboard.press('KeyA'); await page.keyboard.up('Control');
    await page.keyboard.type('-500');
    await page.waitForFunction(key => {
      const position = JSON.parse(localStorage.getItem(key))?.game.state.position;
      return position?.players[position.seat].score === -500;
    }, {}, GUIDED_KEY);
    assert.equal(await page.$eval(field, el => el.value), '-500');
    assert.deepEqual(page.problems, []);
  });
  await check('a fifth copy typed into the starting hand is refused, and what was typed stays to be corrected', async context => {
    const page = await open(context);
    // Pressed by keyboard, the step hands focus to the next prompt's heading
    // and shows its ring there.
    await page.focus('.guide-prompt > .primary'); await page.keyboard.press('Enter'); await stage(page, 'hand');
    assert.deepEqual(await page.evaluate(() => {
      const heading = document.querySelector('.guide-prompt h3');
      return [document.activeElement === heading, heading.matches(':focus-visible'), heading.textContent];
    }), [true, true, 'Choose your first 13 tiles']);
    const field = '[aria-label="Your starting hand in tile notation"]';
    const entered = () => page.$$eval('.guide-prompt .entered .tile', tiles => tiles.length);
    await page.type(field, '11111m'); await page.keyboard.press('Enter');
    // Nothing was entered, so the refusal names no entry to undo.
    await page.waitForFunction(() => document.querySelector('.guide-prompt .failure')?.textContent
      === 'All four copies of 1 characters are already accounted for.');
    assert.equal(await page.$eval(field, el => el.value), '11111m', 'what was typed is kept');
    assert.equal(await entered(), 0);
    // One character taken out, it is entered, and the field is emptied.
    await page.focus(field); await page.keyboard.press('End'); await page.keyboard.press('ArrowLeft'); await page.keyboard.press('Backspace');
    assert.equal(await page.$eval(field, el => el.value), '1111m');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelectorAll('.guide-prompt .entered .tile').length === 4);
    assert.equal(await page.$eval(field, el => el.value), '');
    assert.equal(await page.$('.guide-prompt .failure'), null);
    assert.deepEqual(page.problems, []);
  });
  await check('a refused correction is said under its palette and leaves the prompt\'s own refusal on screen', async context => {
    const page = await open(context);
    // North holds all four 1 characters; East has thrown the 9 characters and
    // South is to discard.
    let game = emptyGuided('club');
    game.state.position.seat = 3;
    game = guidedEvent(game, { type: 'setup' });
    game.state.position.players[3].hand = parseTiles('1111m456p789s123z');
    for (const event of [{ type: 'hand' }, { type: 'indicator', tile: '7z' }, { type: 'discard', tile: '9m' }]) game = guidedEvent(game, event);
    game = guidedEvent(game, { type: 'choice', choice: { kind: 'pass' }, choices: [{ kind: 'pass' }] });
    game = guidedEvent(game, { type: 'continue' });
    await page.evaluate(({ key, text }) => localStorage.setItem(key, text), { key: GUIDED_KEY, text: GUIDED_FORMAT.encode(game) });
    await page.reload({ waitUntil: 'networkidle0' });
    await page.waitForSelector('.guided-controls:not(:disabled)'); await stage(page, 'turn');
    // South's 1 characters would be a fifth copy: the prompt says so.
    await tile(page, '1m');
    const refusal = 'All four copies of 1 characters are already accounted for. Correct the one entered by mistake, or undo back to it.';
    await page.waitForFunction(text => document.querySelector('.guide-prompt .failure')?.textContent === text, {}, refusal);
    // A correction refused below is said under its palette, and the prompt's line stays.
    await page.click('.remembered .correct > summary');
    await page.select('[aria-label="Where the tile to correct is"]', 'hand');
    await page.select('[aria-label="The tile to correct"]', '1m');
    await page.click('.correct .tile-entry summary');
    await page.click('.correct .palette button[data-tile="1m"]');
    await page.waitForFunction(() => document.querySelector('.correct [role="alert"]')?.textContent === 'It is already that tile');
    assert.equal(await page.$eval('.guide-prompt .failure', el => el.textContent), refusal);
    // Undone back past East's only discard, Where is the hand again, not a
    // blank seat that has no discards left, and Which is chosen afresh.
    const where = '[aria-label="Where the tile to correct is"]', which = '[aria-label="The tile to correct"]';
    await page.select(where, '0'); await page.select(which, '0');
    for (let i = 0; i < 3; i++) await button(page, 'Undo last step');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).game.state.position.players[0].discards.length === 0, {}, GUIDED_KEY);
    await page.waitForFunction(where => document.querySelector(where).value === 'hand', {}, where);
    assert.deepEqual(await page.$eval(where, el => [el.selectedIndex, el.selectedOptions[0]?.textContent]), [0, 'Your hand']);
    assert.equal(await page.$eval(which, el => el.value), '');
    assert.deepEqual(page.problems, []);
  });
  await check('mobile guided play offers every adviser and honours a saved trained preference', async context => {
    const page = await open(context, 360), modelRequests = [];
    // The network is not a file of this site any more: it comes from the
    // bucket named in the manifest, 116 MB of it, so this is the one check
    // that waits for a real download.
    const network = `${MANIFEST.origin}/${MANIFEST.object}`;
    const stopObserving = await observeNetworkRequests(context, request => {
      if (request.url === network && request.method === 'GET') modelRequests.push(request.url);
    });
    // Club first, then a saved trained preference: see club().
    await club(page);
    await setup(page, '0');
    await tile(page, '4z'); await choice(page);
    const selector = adviser;
    // A typed-in position is replayed into the events the network's encoder
    // needs, so the trained adviser is offered here as it is in a live game.
    assert.deepEqual(await page.$$eval(`${selector} option:not(:disabled)`, options => options.map(o => o.value)), ['beginner', 'club', 'full']);
    assert.equal(await page.$('.weight-row meter'), null, 'Club selection is not a trained percentage');
    assert.equal(await page.$$eval('.held-tiles .copy-count', nodes => nodes.length), 14);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'no horizontal overflow');
    const before = await saved(page);
    // A saved trained preference is kept as the adviser, and the decision it
    // was saved on is advised by the real network, not rewritten.
    before.state.agent = 'full';
    await page.evaluate(({ key, text }) => localStorage.setItem(key, text), { key: GUIDED_KEY, text: GUIDED_FORMAT.encode(before) });
    await page.reload({ waitUntil: 'networkidle0', timeout: 180000 }); await stage(page, 'decision');
    await page.waitForSelector('.guided-controls:not(:disabled)');
    assert.equal(await page.$eval(selector, el => el.value), 'full');
    assert.equal(await page.$eval(`${selector} option[value="full"]`, el => el.disabled), false);
    await page.waitForSelector('.record-best', { timeout: 90000 });
    await stopObserving();
    assert.equal(modelRequests.length, 1, 'trained advice saves exactly one verified network');
    assert.notEqual(await page.$('.weight-row meter'), null, 'a trained suggestion carries its weight');
    assert.equal(await page.$('.failure'), null);
    assert.deepEqual(await saved(page), before, 'loading must preserve the entire saved game');
    await page.select(selector, 'club'); await choice(page);
    const resumed = await saved(page);
    assert.deepEqual(resumed.state.position, before.state.position);
    assert.deepEqual(resumed.log, before.log); assert.deepEqual(resumed.past, before.past);
    assert.equal(resumed.state.agent, 'club');
    await page.screenshot({ path: resolve(output, 'guided-game-mobile.png'), fullPage: true });
    await page.reload({ waitUntil: 'networkidle0' }); await choice(page);
    assert.equal(await page.$eval(selector, el => el.value), 'club');
    assert.equal((await saved(page)).state.position.players[0].hand.length, 14);
    // The trained adviser answers more than the player's own draws: the
    // guided game stops at every opponent discard, and most of them leave
    // nothing to do but pass. South's 9m is one East cannot claim.
    await page.select(selector, 'full');
    await page.waitForSelector('.weight-row meter', { timeout: 90000 });
    await page.click('.record-best'); await stage(page, 'responses');
    await page.click('.no-calls'); await stage(page, 'turn');
    await tile(page, '9m'); await stage(page, 'decision');
    await page.waitForSelector('.weight-row meter', { timeout: 90000 });
    assert.equal(await page.$('.failure'), null, 'a pass-only decision is advised, not refused');
    assert.deepEqual(await page.$$eval('.weight-row meter', meters => meters.map(meter => meter.value)), [1]);
    // The hands nobody typed in are hidden, not empty: thirteen tiles each.
    assert.deepEqual(await page.$$eval('.belief .who em', held => held.map(entry => entry.textContent)),
      ['13 tiles', '13 tiles', '13 tiles']);
    await page.click('.record-best'); await stage(page, 'responses');
    assert.deepEqual(page.problems, []);
  });
  await check('physical editor starts on Trained when the network is there and Club preserves the table', async context => {
    const page = await open(context, 360, { probe: true }), position = emptyPosition();
    position.players[0].hand = parseTiles('123m456p789s11234z');
    position.drawn = '4z'; position.indicators = ['5z'];
    await page.evaluate(({ key, position }) => localStorage.setItem(key, JSON.stringify({ version: 1, position })), { key: PHYSICAL_KEY, position });
    await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/?mode=physical`, { waitUntil: 'networkidle0' });
    await page.waitForSelector('.physical-editor:not(:disabled)');
    const selector = '[aria-label="Physical play agent"]';
    // Draft readiness and model availability are independent. Control HEAD
    // here and wait for the actual default; the separate guided case above
    // still downloads and runs the real model.
    await page.waitForFunction(selector => document.querySelector(selector)?.value === 'full', {}, selector);
    assert.equal(await page.$eval(selector, el => el.value), 'full');
    assert.deepEqual(await page.$$eval(`${selector} option:not(:disabled)`, options => options.map(o => o.value)), ['beginner', 'club', 'full']);
    await page.select(selector, 'club');
    await page.click('.physical-toolbar .primary');
    await page.waitForSelector('.recommendation');
    assert.equal(await page.$eval('.recommendation strong', el => el.textContent), 'Club');
    assert.equal(await page.$('.weight-row meter'), null);
    assert.equal(await page.$('.physical-play .failure'), null);
    const restored = await page.evaluate(key => JSON.parse(localStorage.getItem(key)).position, PHYSICAL_KEY);
    assert.deepEqual(restored, position);
    assert.deepEqual(page.problems, []);
  });
  await check('two windows stop conflicting edits and reload the newer guided prompt', async context => {
    const a = await open(context); await club(a); await setup(a);
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
    // What the form holds is saved with the game as it is typed, so a reload
    // brings the ura back: it used to empty the form.
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).game.state.draft?.ura.join() === '6z', {}, GUIDED_KEY);
    await page.reload({ waitUntil: 'networkidle0' }); await stage(page, 'over');
    await page.waitForSelector('.guided-controls:not(:disabled)');
    assert.deepEqual(await page.$$eval('.guided-result .entered .tile', tiles => tiles.map(tile => tile.dataset.tile)), ['6z']);
    await button(page, 'Calculate hand settlement');
    await page.waitForSelector('[aria-label="North ura-dora indicators"]');
    assert.equal(await page.$$eval('[aria-label="North ura-dora indicators"] .tile', nodes => nodes.length), 1);
    assert.match(await page.$eval('[aria-label="North scored hand"]', el => el.textContent), /Ura-dora: 0 han/);
    await button(page, 'Apply settlement');
    await page.waitForFunction(key => Boolean(JSON.parse(localStorage.getItem(key)).game.state.settlement), {}, GUIDED_KEY);
    const paid = (await saved(page)).state;
    assert.deepEqual(paid.settlement.hand_deltas, [-2600,0,0,2600]);
    assert.equal(paid.draft, undefined, 'the settlement keeps its own input, and the draft goes');
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
  await browser?.close(); await new Promise(done => server.close(done));
  await cases.report('guided-game checks', resolve(output, 'guided-game.json'));
}
