/** Trained is the default opponent, watch agent, review adviser and guided
 * adviser, in the production build with its real service worker. The
 * network's host is made unreachable for the whole browser and the page's
 * availability probe is answered here, so no check downloads the 116 MB
 * network: each one either never needs it or meets a download that fails,
 * which is exactly when the Club fallbacks must work. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { mkdir, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import { createFixtureHandler } from './static-fixture-server.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { GUIDED_KEY } from '../src/lib/guided-game.js';
import { parseTiles } from '../src/lib/physical-position.js';
import { MANIFEST } from '../src/lib/model-manifest.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const web = fileURLToPath(new URL('../', import.meta.url)), dist = resolve(web, 'dist'), output = resolve(web, 'test-results');
const manifest = JSON.parse(await readFile(resolve(dist, 'offline-manifest.json'), 'utf8'));
const networkUrl = `${MANIFEST.origin}/${MANIFEST.object}`;
const ai = new Set(manifest.entries.filter(entry => entry.group === 'ai').map(entry => entry.url));
const runtime = new Set([...ai].filter(url => url.endsWith('.wasm')));

// The trained AI's own files can be counted, and its runtime held back, so
// its download can be watched while it is still under way.
const requested = new Map(), waiting = new Set();
let holding = false;
const server = createServer(createFixtureHandler({ root: dist, publicRoot: dist, intercept: async name => {
  requested.set(name, (requested.get(name) ?? 0) + 1);
  if (holding && runtime.has(name)) await new Promise(done => waiting.add(done));
} }));
function release() {
  holding = false;
  for (const done of waiting) done();
  waiting.clear();
}

// A finished first hand against Club, for the review.
const finished = (() => {
  const match = new MatchSession(Game, 1, 'club');
  try {
    match.advance(false);
    for (let n = 0; n < 250 && match.view.phase !== 'over'; n++) {
      const choice = match.choices.find(c => ['ron', 'tsumo'].includes(c.kind))
        ?? match.choices.find(c => c.kind === 'pass') ?? match.choices.find(c => c.kind === 'discard') ?? match.choices[0];
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
    }
    assert.equal(match.view.phase, 'over');
    assert.ok(match.engine.review().length > 1);
    return match.snapshot();
  } finally { match.dispose(); }
})();
// What the previous build wrote on a first visit, when Club was the default.
const previousDefaults = { version: 1, difficulty: 'club', opponents: ['club', 'club', 'club'], hints: true,
  confirmDiscards: false, shortcuts: true, tileFace: 'classic', reviewAdviser: 'club' };

const cases = browserChecks();
let browser;
/** A fresh browser profile. `reachable` answers the availability probe; a
 * download of the network always fails, as it does offline. */
async function open({ mode = 'play', reachable = true, saved = {} } = {}) {
  const page = await (await cases.openContext(browser)).newPage();
  page.problems = [];
  page.on('pageerror', error => page.problems.push(error.message));
  await page.setViewport({ width: 1100, height: 900 });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  await page.evaluateOnNewDocument(({ url, reachable, saved }) => {
    for (const [key, value] of Object.entries(saved)) if (localStorage.getItem(key) === null) localStorage.setItem(key, value);
    const realFetch = window.fetch;
    window.networkDownloads = 0;
    window.fetch = (input, options) => {
      const target = input instanceof Request ? input.url : String(input);
      if (target !== url) return realFetch(input, options);
      const method = String(options?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase();
      if (method === 'HEAD') return Promise.resolve(new Response(null, { status: reachable ? 200 : 404 }));
      window.networkDownloads++;
      return Promise.reject(new TypeError('Failed to fetch'));
    };
  }, { url: networkUrl, reachable, saved });
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/${mode === 'play' ? '' : `?mode=${mode}`}`, { waitUntil: 'domcontentloaded' });
  return page;
}
const settings = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), SETTINGS_KEY);
const match = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
const prompt = page => page.$eval('#hand-help', el => el.textContent.trim());
async function clickButton(page, scope, text) {
  await page.waitForFunction((scope, text) => [...document.querySelectorAll(`${scope} button`)]
    .some(button => button.textContent.trim() === text && !button.disabled), {}, scope, text);
  await page.evaluate((scope, text) => [...document.querySelectorAll(`${scope} button`)]
    .find(button => button.textContent.trim() === text && !button.disabled).click(), scope, text);
}
const check = (name, body) => cases.check(name, async () => {
  requested.clear();
  try { await body(); } finally { release(); }
});

try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  // Nothing in this browser can reach the network's host.
  browser = await launchChrome({ args: [`--host-resolver-rules=MAP ${new URL(networkUrl).hostname} ~NOTFOUND`] });

  await check('a first visit plays Trained opponents, shows their download on the table and continues with Club when it fails', async () => {
    holding = true;
    const page = await open();
    await page.waitForSelector('.board');
    assert.equal(await page.$eval('select[aria-label="opponent strength"]', el => el.value), 'neural');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key))?.opponents?.length === 3, {}, SAVE_KEY);
    assert.deepEqual((await match(page)).opponents, ['neural', 'neural', 'neural']);
    assert.deepEqual(await settings(page), { version: 1, difficulty: 'neural', opponents: ['neural', 'neural', 'neural'],
      hints: true, confirmDiscards: false, shortcuts: true, tileFace: 'classic', reviewAdviser: 'strong' });
    // East discards first; then, or at once, a Trained opponent waits for the network.
    await page.waitForFunction(() => /downloading the trained network \d+%/.test(document.querySelector('#hand-help')?.textContent ?? '')
      || document.querySelector('.hand button[data-hand-index]:not(:disabled)'), { timeout: 60000 });
    if (!/downloading/.test(await prompt(page))) await page.click('.hand button[data-hand-index]:not(:disabled)');
    await page.waitForFunction(() => /downloading the trained network \d+%/.test(document.querySelector('#hand-help')?.textContent ?? ''), { timeout: 60000 });
    assert.match(await page.$eval('[data-offline-status]', el => el.textContent), /Saving AI… \d+%/);
    for (const deadline = Date.now() + 30000; !waiting.size;) {
      assert.ok(Date.now() < deadline, 'the trained runtime never started downloading');
      await new Promise(done => setTimeout(done, 50));
    }
    await page.screenshot({ path: resolve(output, 'trained-default-download.png'), fullPage: true });
    // The network itself cannot be had: the table offers its fallbacks.
    release();
    await page.waitForSelector('.failure .recovery-actions', { timeout: 120000 });
    assert.ok(await page.evaluate(() => window.networkDownloads) > 0, 'the network download was attempted');
    const offered = await page.$$eval('.failure .recovery-actions button', buttons => buttons.map(button => button.textContent.trim()));
    assert.ok(offered.includes('Retry trained opponent'), offered.join(', '));
    assert.ok(offered.includes('Continue with Club opponents'), offered.join(', '));
    await clickButton(page, '.failure .recovery-actions', 'Continue with Club opponents');
    await page.waitForFunction(() => !document.querySelector('.failure') && document.querySelector('.hand button[data-hand-index]:not(:disabled), .call-options button:not(:disabled)'), { timeout: 45000 });
    assert.ok((await match(page)).commands.some(command => command.type === 'club'));
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).difficulty === 'club', {}, SETTINGS_KEY);
    assert.deepEqual((await settings(page)).opponents, ['club', 'club', 'club'], 'the player\'s fallback is the table they now have');
    assert.equal(await page.$eval('select[aria-label="opponent strength"]', el => el.value), 'club');
    assert.deepEqual(page.problems, []);
  });

  await check('settings saved when Club was the default keep Club, and nothing downloads the network', async () => {
    const page = await open({ saved: { [SETTINGS_KEY]: JSON.stringify(previousDefaults) } });
    await page.waitForSelector('.hand');
    assert.equal(await page.$eval('select[aria-label="opponent strength"]', el => el.value), 'club');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key))?.opponents?.length === 3, {}, SAVE_KEY);
    assert.deepEqual((await match(page)).opponents, ['club', 'club', 'club']);
    for (let move = 0; move < 3; move++) {
      await page.waitForFunction(() => document.querySelector('.screen, .hand button[data-hand-index]:not(:disabled), .call-options button:not(:disabled)'), { timeout: 45000 });
      if (await page.$('.screen')) break;
      const before = (await match(page)).commands.length;
      await (await page.$('.call-options [data-choice=pass]') ?? await page.$('.hand button[data-hand-index]:not(:disabled)')).click();
      await page.waitForFunction((key, count) => JSON.parse(localStorage.getItem(key)).commands.length > count, { timeout: 45000 }, SAVE_KEY, before);
    }
    assert.deepEqual(await settings(page), previousDefaults);
    assert.equal(await page.evaluate(() => window.networkDownloads), 0);
    for (const url of ai) assert.equal(requested.get(url) ?? 0, 0, `Club play fetched ${url}`);
    assert.deepEqual(page.problems, []);
  });

  await check('a first review is Trained AI, and Club reviews in its place while the network is out of reach', async () => {
    const page = await open({ reachable: false, saved: { [SAVE_KEY]: JSON.stringify(finished) } });
    await page.waitForSelector('.screen .quiet'); await page.click('.screen .quiet');
    await page.waitForSelector('select[aria-label="Review adviser"]');
    assert.equal(await page.$eval('select[aria-label="Review adviser"]', el => el.value), 'strong');
    assert.equal(await page.$eval('select[aria-label="Review adviser"] option[value="strong"]', el => el.disabled), true);
    assert.match(await page.$eval('[data-review-fallback]', el => el.textContent), /Club reviews this hand/);
    assert.match(await page.$eval('.review .summary', el => el.textContent), /matched Club/);
    assert.equal(await page.$('.review [role="alert"]'), null);
    // The default is kept for when the network can be had, not replaced by the fallback.
    assert.equal((await settings(page)).reviewAdviser, 'strong');
    assert.equal(await page.evaluate(() => window.networkDownloads), 0);
    await page.screenshot({ path: resolve(output, 'trained-default-review-fallback.png'), fullPage: true });
    assert.deepEqual(page.problems, []);
  });

  await check('a Trained review that cannot load the network offers Club, which is then remembered', async () => {
    const page = await open({ saved: { [SAVE_KEY]: JSON.stringify(finished) } });
    // Opened before or after the probe answers: a review shown by Club in the
    // meantime turns to Trained AI the moment the network is reported.
    await page.waitForSelector('.screen .quiet'); await page.click('.screen .quiet');
    await page.waitForSelector('select[aria-label="Review adviser"]');
    await page.waitForSelector('.review [role="alert"] [data-review-club]', { timeout: 120000 });
    assert.ok(await page.evaluate(() => window.networkDownloads) > 0, 'the trained review asked for the network');
    assert.deepEqual(await page.$$eval('.review [role="alert"] button', buttons => buttons.map(button => button.textContent.trim())),
      ['Retry Trained review', 'Review with Club']);
    await page.click('[data-review-club]');
    await page.waitForFunction(() => document.querySelector('.review .summary')?.textContent.includes('matched Club'));
    assert.equal(await page.$eval('select[aria-label="Review adviser"]', el => el.value), 'club');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).reviewAdviser === 'club', {}, SETTINGS_KEY);
    assert.deepEqual(page.problems, []);
  });

  await check('Agent watch puts Trained in every seat and can watch the same table with Club when the network fails', async () => {
    const page = await open({ mode: 'watch' });
    const lineup = () => page.$$eval('.agent-fields select', selects => selects.map(select => select.value));
    await page.waitForFunction(() => [...document.querySelectorAll('.agent-fields select')].every(select => select.value === 'full'));
    assert.deepEqual(await lineup(), ['full', 'full', 'full', 'full']);
    await page.click('.watch-setup .primary');
    await page.waitForSelector('.agent-watch [role="alert"] [data-watch-club]', { timeout: 120000 });
    assert.ok(await page.evaluate(() => window.networkDownloads) > 0);
    await page.click('[data-watch-club]');
    await page.waitForSelector('.weight-row.best .choice-action:not(:disabled)', { timeout: 45000 });
    assert.equal(await page.$('.agent-watch [role="alert"]'), null);
    assert.deepEqual(await lineup(), ['club', 'club', 'club', 'club']);
    assert.doesNotMatch(await page.$eval('.watch-table', el => el.textContent), /Trained/);
    assert.deepEqual(page.problems, []);
  });

  await check('a new guided game is advised by Trained and offers Club when that advice cannot load', async () => {
    const page = await open({ mode: 'guided' });
    const adviser = '[aria-label="Guided game adviser"]';
    const stage = value => page.waitForSelector(`.guide-prompt[data-stage="${value}"]`);
    const tile = value => page.click(`.guide-prompt .palette button[data-tile="${value}"]`);
    await page.waitForSelector('.guided-controls:not(:disabled)');
    assert.equal(await page.$eval(adviser, el => el.value), 'full');
    await page.select('[aria-label="Your seat"]', '0');
    await page.click('.guide-prompt > .primary'); await stage('hand');
    for (const t of parseTiles('123m456p789s1123z')) await tile(t);
    await page.click('.guide-prompt > .primary'); await stage('dora');
    await tile('7z'); await stage('turn');
    await tile('4z'); await stage('decision');
    await page.waitForSelector('[data-guided-club]', { timeout: 120000 });
    assert.ok(await page.evaluate(() => window.networkDownloads) > 0);
    assert.ok(await page.$('.guide-prompt .failure'));
    await page.click('[data-guided-club]');
    await page.waitForSelector('.record-best', { timeout: 45000 });
    assert.equal(await page.$('.guide-prompt .failure'), null);
    assert.equal(await page.$eval(adviser, el => el.value), 'club');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key))?.game.state.agent === 'club', {}, GUIDED_KEY);
    assert.deepEqual(page.problems, []);
  });
} finally {
  release();
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
  await cases.report('trained-default checks', resolve(output, 'trained-default.json'));
}
