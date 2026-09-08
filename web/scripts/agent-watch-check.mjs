/** Watch the actual production UI against decisions and hints from real WASM. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import init, { Game } from '../src/wasm/riichi.js';
import { WatchSession } from '../src/lib/watch-session.js';
import { SETTINGS_KEY } from '../src/lib/session.js';
import { unseenTileCounts } from '../src/lib/ui.js';
import { tileWords } from '../src/lib/tiles.js';
import { createFixtureHandler } from './static-fixture-server.mjs';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const web = fileURLToPath(new URL('../', import.meta.url)), output = resolve(web, 'test-results');
const server = createServer(createFixtureHandler({ root: resolve(web, 'dist'), publicRoot: resolve(web, 'public') }));
const results = [], contexts = [];
let browser;
const builtin = async (engine, agent) => ({ choice: engine.agent_pick(agent), choices: engine.agent_choices() });

async function open(seed, width = 1100) {
  const context = await browser.createBrowserContext(); contexts.push(context);
  const page = await context.newPage(); page.problems = [];
  page.on('pageerror', error => page.problems.push(error.message));
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1, hasTouch: width < 600 });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  await page.evaluateOnNewDocument(key => localStorage.setItem(key, JSON.stringify({ version: 1, difficulty: 'club', hints: true })), SETTINGS_KEY);
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/?mode=watch`, { waitUntil: 'networkidle0' });
  await page.waitForSelector('.watch-setup .primary:not(:disabled)');
  for (const label of ['Followed agent', 'Right', 'Opposite', 'Left']) await page.select(`select[aria-label="${label}"]`, 'club');
  await page.evaluate(value => { Date.now = () => value; }, seed);
  await page.click('.watch-setup .primary');
  await page.waitForSelector('.weight-row.best .choice-action:not(:disabled)');
  return page;
}

async function check(name, task) {
  try { await task(); results.push({ name, passed: true }); console.log(`PASS ${name}`); }
  catch (error) { results.push({ name, passed: false, error: error.stack }); console.error(`FAIL ${name}\n${error.stack}`); }
  finally { while (contexts.length) await contexts.pop().close(); }
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

try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  const executablePath = process.env.CHROME_BIN || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(executablePath, 'Set CHROME_BIN to Chrome/Chromium');
  browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });

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
        const history = await page.$eval('.agent-watch > details summary', el => el.textContent);
        await page.click('.watch-controls > button');
        await watch.step();
        await page.waitForFunction(previous => document.querySelector('.agent-watch > details summary').textContent !== previous
          && document.querySelector('.weight-row.best .choice-action:not(:disabled)'), {}, history);
      }
      await assertHints(page, watch);
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
} finally {
  await writeFile(resolve(output, 'agent-watch.json'), JSON.stringify(results, null, 2));
  await browser?.close();
  await new Promise(done => server.close(done));
}
if (results.some(result => !result.passed)) process.exitCode = 1;
