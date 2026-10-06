/** Watch the actual production UI against decisions and hints from real WASM. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { WatchSession } from '../src/lib/watch-session.js';
import { SETTINGS_KEY } from '../src/lib/session.js';
import { unseenTileCounts } from '../src/lib/ui.js';
import { tileWords } from '../src/lib/tiles.js';
import { playerNames } from '../src/lib/game-log.js';
import { createFixtureHandler } from './static-fixture-server.mjs';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const web = fileURLToPath(new URL('../', import.meta.url)), output = resolve(web, 'test-results');
const server = createServer(createFixtureHandler({ root: resolve(web, 'dist'), publicRoot: resolve(web, 'public') }));
const { check, openContext, report } = browserChecks();
let browser;
const builtin = async (engine, agent) => ({ choice: engine.agent_pick(agent), choices: engine.agent_choices() });

async function open(seed, width = 1100) {
  const page = await (await openContext(browser)).newPage(); page.problems = [];
  page.on('pageerror', error => page.problems.push(error.message));
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1, hasTouch: width < 600 });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  // A saved file is kept in the page, name and text, rather than downloaded.
  await page.evaluateOnNewDocument(() => {
    const blobs = new Map(), create = URL.createObjectURL.bind(URL);
    URL.createObjectURL = object => { const url = create(object); blobs.set(url, object); return url; };
    window.savedFiles = [];
    HTMLAnchorElement.prototype.click = function () {
      const file = { name: this.download, text: null }; window.savedFiles.push(file);
      void blobs.get(this.href)?.text().then(text => { file.text = text; });
    };
  });
  await page.evaluateOnNewDocument(key => localStorage.setItem(key, JSON.stringify({ version: 1, difficulty: 'club', hints: true })), SETTINGS_KEY);
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/?mode=watch`, { waitUntil: 'networkidle0' });
  await page.waitForSelector('.watch-setup .primary:not(:disabled)');
  for (const label of ['Followed agent', 'Right', 'Opposite', 'Left']) await page.select(`select[aria-label="${label}"]`, 'club');
  await page.evaluate(value => { Date.now = () => value; }, seed);
  await page.click('.watch-setup .primary');
  await page.waitForSelector('.weight-row.best .choice-action:not(:disabled)');
  return page;
}

async function assertHints(page, watch) {
  const view = watch.match.view, counts = unseenTileCounts(view);
  const hints = new Map(watch.analysis.choices.filter(c => c.kind === 'discard').map(c => [c.tile, watch.match.engine.discard_hint(c.tile)]));
  const actual = await page.$$eval('.followed .hand-tile', elements => elements.map(element => {
    const tile = element.querySelector('button');
    return { tile: tile.dataset.tile, readiness: tile.dataset.readiness ?? null,
      dora: tile.classList.contains('dora'), count: Number(element.querySelector('.copy-count')?.textContent ?? -1),
      ring: getComputedStyle(tile).getPropertyValue('--ring').trim() };
  }));
  assert.equal(actual.length, view.seats[0].hand.length + Number(Boolean(view.seats[0].drawn)));
  for (const item of actual) {
    const shanten = hints.get(item.tile)?.shanten;
    assert.equal(item.readiness, shanten === 0 ? 'ready' : shanten === 1 ? 'one-away' : null, item.tile);
    assert.equal(item.count, counts.get(item.tile), `unseen copies of ${item.tile}`);
    assert.equal(item.dora, view.dora_types.includes(item.tile));
    if (item.dora) assert.match(item.ring, /#e2453d/);
  }
  assert.deepEqual(page.problems, []);
}

// Every seat's discard row carries the engine's record of each discard:
// thrown straight from the draw (a darker face) and claimed (see-through),
// to the eye and in the tile's name. Returns those records, row by row.
async function assertMarks(page, watch) {
  const expected = watch.match.view.seats.map(seat => seat.discards.map(discard => [Boolean(discard.drawn), discard.claimed]));
  const shown = await page.$$eval('.watch-table .pool', pools => pools.map(pool => [...pool.querySelectorAll('.tile')].map(tile => {
    const label = tile.getAttribute('aria-label');
    return { marks: [tile.classList.contains('from-draw'), tile.classList.contains('claimed')],
      words: [label.includes('discarded from the draw'), label.includes(', claimed')] };
  })));
  assert.deepEqual(shown.map(row => row.map(tile => tile.marks)), expected);
  for (const tile of shown.flat()) assert.deepEqual(tile.words, tile.marks);
  return expected;
}

try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  browser = await launchChrome();

  await check('watch combines blue recommendations with gold/silver hints and confirms alternative moves', async () => {
    const watch = new WatchSession(Game, 31, ['club', 'club', 'club', 'club'], { evaluate: builtin });
    try {
      await watch.prepare();
      const page = await open(31);
      await assertHints(page, watch);
      assert.ok(await page.$('.followed [data-readiness="ready"]'));
      assert.ok(await page.$('.followed [data-readiness="one-away"]'));
      assert.equal(await page.$eval('.weight-row.best', el => getComputedStyle(el).borderColor), 'rgb(78, 163, 255)');
      const selected = await page.$$eval('.followed .hand button.selected', els => els.map(el => ({
        tile: el.dataset.tile, ring: getComputedStyle(el).getPropertyValue('--ring'), readiness: el.dataset.readiness,
      })));
      assert.equal(selected.length, 1);
      assert.equal(selected[0].tile, watch.analysis.choice.tile);
      assert.equal(selected[0].readiness, 'ready');
      assert.match(selected[0].ring, /#4ea3ff/);
      const alternative = watch.analysis.choices.find(c => c.kind === 'discard' && c.tile !== watch.analysis.choice.tile);
      const selector = `.choice-action[aria-label=${JSON.stringify(`Play ${alternative.label}`)}]`;
      const before = await page.$eval('.followed', el => el.textContent);
      let question;
      page.once('dialog', async dialog => { question = dialog.message(); await dialog.dismiss(); });
      await page.click(selector);
      assert.match(question, /Play/);
      assert.ok(question.includes(alternative.label));
      assert.equal(await page.$eval('.followed', el => el.textContent), before);
      page.once('dialog', dialog => dialog.accept());
      await page.click(selector);
      await watch.choose(alternative, watch.analysis);
      await page.waitForFunction(() => document.querySelectorAll('.followed .pool .tile').length > 0
        && document.querySelector('.weight-row.best .choice-action:not(:disabled)'));
      assert.equal((await page.$eval('.followed .pool .tile', el => el.getAttribute('aria-label'))).split(',')[0], tileWords(alternative.tile));
      await assertHints(page, watch);
      await assertMarks(page, watch);
      await page.screenshot({ path: resolve(output, 'agent-watch-desktop.png'), fullPage: true });
    } finally { watch.dispose(); }
  });

  await check('mobile watch shows dora and copy counts, respects the hint setting, and separates opponent melds', async () => {
    const watch = new WatchSession(Game, 11, ['club', 'club', 'club', 'club'], { evaluate: builtin });
    try {
      await watch.prepare();
      const page = await open(11, 360);
      await assertHints(page, watch);
      assert.ok(await page.$('.followed .hand .dora'));
      for (let turn = 0; turn < 5; turn++) {
        const history = await page.$eval('.agent-watch > details:last-child > summary', el => el.textContent);
        await page.click('.watch-controls > button');
        await watch.step();
        await page.waitForFunction(previous => document.querySelector('.agent-watch > details:last-child > summary').textContent !== previous
          && document.querySelector('.weight-row.best .choice-action:not(:disabled)'), {}, history);
      }
      await assertHints(page, watch);
      const marks = (await assertMarks(page, watch)).flat();
      assert.ok(marks.some(([drawn]) => drawn) && marks.some(([, claimed]) => claimed),
        `the fixture must hold a discard from the draw and a claimed one: ${JSON.stringify(marks)}`);
      const gaps = await page.$$eval('.watch-table section:not(.followed) .melds', elements => elements.map(el =>
        el.getBoundingClientRect().top - el.previousElementSibling.getBoundingClientRect().bottom));
      assert.ok(gaps.length > 0, 'fixture must contain an opponent call');
      assert.ok(gaps.every(gap => gap >= 8));
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'no horizontal overflow');
      await page.screenshot({ path: resolve(output, 'agent-watch-mobile.png'), fullPage: true });
      // Dispatch the actual control even when the mobile settings drawer is closed.
      await page.evaluate(() => [...document.querySelectorAll('.option-fields label')]
        .find(el => el.textContent.includes('Hints and markings')).querySelector('input').click());
      await page.waitForFunction(() => !document.querySelector('.followed .copy-count'));
      assert.equal(await page.$('.followed [data-readiness]'), null);
      assert.equal(await page.$('.followed .hand .dora'), null);
      assert.equal(await page.$eval('.weight-row.best', el => getComputedStyle(el).borderColor), 'rgb(78, 163, 255)');
      assert.deepEqual(page.problems, []);
    } finally { watch.dispose(); }
  });

  await check('watch saves its game from East 1 once a hand is over, and never the hand being played', async () => {
    const watch = new WatchSession(Game, 23, ['club', 'club', 'club', 'club'], { evaluate: builtin });
    try {
      await watch.prepare();
      const page = await open(23);
      const button = () => page.$eval('.watch-controls [data-save-game]', el => ({ disabled: el.disabled, text: el.textContent.trim() }));
      // The page has caught up with this copy once it shows the same wall
      // and the same discards, and is waiting on the next choice or showing
      // the hand's result. A pass can add no line to the history.
      const caughtUp = () => {
        const view = watch.match.view;
        return page.waitForFunction(({ wall, discards, over }) =>
          document.querySelector('.watch-round span')?.textContent.startsWith(`${wall} tiles left`)
            && document.querySelectorAll('.watch-table .pool .tile').length === discards
            && Boolean(document.querySelector(over ? '.agent-watch .screen' : '.weight-row.best .choice-action:not(:disabled)')),
        {}, { wall: view.wall, discards: view.seats.reduce((sum, seat) => sum + seat.discards.length, 0), over: view.phase === 'over' });
      };
      // The page and this copy play the same hand move for move.
      for (let turn = 0; watch.match.view.phase !== 'over'; turn++) {
        assert.ok(turn < 150, 'the first hand ends');
        assert.deepEqual(await button(), { disabled: true, text: 'No hand has finished yet' });
        await page.click('.watch-controls > button');
        await watch.step();
        await caughtUp();
      }
      await page.waitForSelector('.agent-watch .screen [data-save-game]');
      assert.deepEqual(await button(), { disabled: false, text: 'Save game so far (1 finished hand)' });
      const labels = ['Followed agent', 'Right', 'Opposite', 'Left'].map(position => `${position} (Club)`);
      const expected = watch.match.engine.game_log(playerNames(watch.match.view, labels)) + '\n';
      const saveFrom = async selector => {
        const count = await page.evaluate(() => window.savedFiles.length);
        await page.click(selector);
        await page.waitForFunction(n => window.savedFiles.length > n && window.savedFiles.at(-1).text !== null, {}, count);
        const file = await page.evaluate(() => window.savedFiles.at(-1));
        assert.match(file.name, /^riichi-game-\d{4}-\d{2}-\d{2}-\d{6}\.mjai\.jsonl$/);
        return file.text;
      };
      assert.equal(await saveFrom('.watch-controls [data-save-game]'), expected);
      assert.equal(await saveFrom('.agent-watch .screen [data-save-game]'), expected);
      const events = expected.trimEnd().split('\n').map(line => JSON.parse(line));
      assert.deepEqual(events.map(event => event.type).filter(type => ['start_game', 'start_kyoku', 'end_kyoku', 'end_game'].includes(type)),
        ['start_game', 'start_kyoku', 'end_kyoku']);
      assert.equal(events[0].names[watch.match.view.seats[0].player], 'Followed agent (Club)');
      const deal = events[1];
      assert.deepEqual([deal.bakaze, deal.kyoku, deal.honba, deal.oya], ['E', 1, 0, 0]);
      // Dealing on puts the next hand on the table and keeps it out.
      await page.click('.watch-controls > button');
      await watch.step();
      await caughtUp();
      assert.deepEqual(await button(), { disabled: false, text: 'Save game so far (1 finished hand)' });
      assert.equal(await saveFrom('.watch-controls [data-save-game]'), expected);
      await page.screenshot({ path: resolve(output, 'agent-watch-export.png'), fullPage: true });
      assert.deepEqual(page.problems, []);
    } finally { watch.dispose(); }
  });
} finally {
  await browser?.close();
  await new Promise(done => server.close(done));
  await report('agent-watch checks', resolve(output, 'agent-watch.json'));
}
