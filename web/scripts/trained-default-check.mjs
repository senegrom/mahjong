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

// The trained AI's own files can be counted. The tile faces a first visit
// does not show can be held back: the service worker's install saves them
// all before it activates, so this keeps it installing. The network's own
// download is held open by the page that asks for it (open's `hold`), so
// that it can be watched while it is still under way.
const requested = new Map(), installing = new Set();
let holdingFaces = false;
const unusedFace = name => ['tiles/matisse/', 'tiles/van-gogh/'].some(folder => name.startsWith(folder));
const server = createServer(createFixtureHandler({ root: dist, publicRoot: dist, intercept: async name => {
  requested.set(name, (requested.get(name) ?? 0) + 1);
  if (holdingFaces && unusedFace(name)) await new Promise(done => installing.add(done));
} }));
function release() {
  holdingFaces = false;
  for (const done of installing) done();
  installing.clear();
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
// What a build from when Club was the default wrote on a first visit.
const previousDefaults = { version: 1, difficulty: 'club', opponents: ['club', 'club', 'club'], hints: true,
  confirmDiscards: false, shortcuts: true, tileFace: 'classic', reviewAdviser: 'club' };
// The same choice made in this build, which stands, and writes the review's
// settings with it as a first visit has them: both hiding settings off, and
// the filter on the flagged decisions.
const chosenClub = { ...previousDefaults, version: 6, reviewHideMove: false, reviewHideTiles: false, reviewFilter: 'flagged' };

const cases = browserChecks();
let browser;
/** A fresh browser profile. `reachable` answers the availability probe; a
 * download of the network always fails, as it does offline: at once, or with
 * `hold` when the check says so (failNetwork), having been under way until
 * then. */
async function open({ mode = 'play', reachable = true, saved = {}, hold = false } = {}) {
  const page = await (await cases.openContext(browser)).newPage();
  page.problems = [];
  page.on('pageerror', error => page.problems.push(error.message));
  await page.setViewport({ width: 1100, height: 900 });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  await page.evaluateOnNewDocument(({ url, reachable, saved, hold }) => {
    for (const [key, value] of Object.entries(saved)) if (localStorage.getItem(key) === null) localStorage.setItem(key, value);
    const realFetch = window.fetch;
    window.networkDownloads = 0;
    window.heldDownloads = [];
    window.fetch = (input, options) => {
      const target = input instanceof Request ? input.url : String(input);
      if (target !== url) return realFetch(input, options);
      const method = String(options?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase();
      if (method === 'HEAD') return Promise.resolve(new Response(null, { status: reachable ? 200 : 404 }));
      window.networkDownloads++;
      const failed = new TypeError('Failed to fetch');
      return hold ? new Promise((_, reject) => window.heldDownloads.push(() => reject(failed))) : Promise.reject(failed);
    };
  }, { url: networkUrl, reachable, saved, hold });
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/${mode === 'play' ? '' : `?mode=${mode}`}`, { waitUntil: 'domcontentloaded' });
  return page;
}
const settings = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), SETTINGS_KEY);
const match = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
// The turn's prompt: what it says, with how far a download has got beside it.
const PROMPT = '.controls .prompt';
const prompt = page => page.$eval(PROMPT, el => el.textContent.trim());
// The download as every wait names it: the network's size as it is fetched.
const DOWNLOAD = `the trained network (${Math.round((MANIFEST.storedBytes ?? MANIFEST.bytes) / 1e6)} MB)`;
async function clickButton(page, scope, text) {
  await page.waitForFunction((scope, text) => [...document.querySelectorAll(`${scope} button`)]
    .some(button => button.textContent.trim() === text && !button.disabled), {}, scope, text);
  await page.evaluate((scope, text) => [...document.querySelectorAll(`${scope} button`)]
    .find(button => button.textContent.trim() === text && !button.disabled).click(), scope, text);
}
/** Waits until a Trained wait inside `scope` names the network's download,
 * which the page holds open (open's `hold`) so that it is under way. */
async function downloading(page, scope) {
  await page.waitForFunction((scope, download) => new RegExp(`downloading ${download.replace(/[()]/g, '\\$&')} \\d+%`).test(document.querySelector(scope)?.textContent ?? ''),
    { timeout: 60000 }, scope, DOWNLOAD);
  // The note is said as the save begins, a moment before the page asks for
  // the network: its turn at the network and what is stored are looked at
  // first. On a slow machine that moment is long enough to be seen, so the
  // request is waited for, not expected at once.
  await page.waitForFunction(() => window.heldDownloads.length > 0, { timeout: 30000 })
    .catch(() => assert.fail('the trained network never started downloading'));
}
/** Ends the download the page holds open: the network cannot be had. */
const failNetwork = page => page.evaluate(() => { for (const fail of window.heldDownloads.splice(0)) fail(); });
const check = (name, body) => cases.check(name, async () => {
  requested.clear();
  try { await body(); } finally { release(); }
});

try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  // Nothing in this browser can reach the network's host.
  browser = await launchChrome({ args: [`--host-resolver-rules=MAP ${new URL(networkUrl).hostname} ~NOTFOUND`] });

  await check('a first visit plays Trained opponents, shows their download on the table beside the install and continues with Club when it fails', async () => {
    holdingFaces = true;
    const page = await open({ hold: true });
    await page.waitForSelector('.board');
    assert.equal(await page.$eval('select[aria-label="opponent strength"]', el => el.value), 'neural');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key))?.opponents?.length === 3, {}, SAVE_KEY);
    assert.deepEqual((await match(page)).opponents, ['neural', 'neural', 'neural']);
    assert.deepEqual(await settings(page), { version: 6, difficulty: 'neural', opponents: ['neural', 'neural', 'neural'],
      hints: true, confirmDiscards: false, shortcuts: true, tileFace: 'classic', reviewAdviser: 'strong', reviewHideMove: false, reviewHideTiles: false,
      reviewFilter: 'flagged' });
    // East discards first; then, or at once, a Trained opponent waits. The
    // network's download does not wait for the service worker, which on a
    // first visit is still saving the game and every tile graphic: it begins
    // at once, and the table shows it rather than only "thinking".
    const download = `downloading ${DOWNLOAD}`;
    await page.waitForFunction((prompt, text) => document.querySelector(prompt)?.textContent.includes(text)
      || document.querySelector('.hand button[data-hand-index]:not(:disabled)'), { timeout: 60000 }, PROMPT, download);
    if (!(await prompt(page)).includes(download)) await page.click('.hand button[data-hand-index]:not(:disabled)');
    await downloading(page, PROMPT);
    // Only the prompt's words are a live region: how far the download has
    // got is shown beside them, outside it, so that a screen reader hears
    // the download once and not at every percent.
    assert.deepEqual(await page.$eval(PROMPT, el => [el.querySelector('[role="status"]').textContent.trim(), /^\d+%$/.test(el.querySelector('[data-figure]').textContent),
      Boolean(el.querySelector('[data-figure]').closest('[role="status"], [aria-live]'))]), [download, true, false]);
    for (const deadline = Date.now() + 30000; !installing.size;) {
      assert.ok(Date.now() < deadline, 'the service worker never reached the unused faces');
      await new Promise(done => setTimeout(done, 50));
    }
    assert.equal(await page.evaluate(async () => Boolean((await navigator.serviceWorker.getRegistration())?.active)), false,
      'the install is still running, and the network did not wait for it');
    assert.match(await page.$eval('[data-offline-status]', el => el.textContent), /Saving AI… \d+%/);
    // The runtime is the worker's to save, once it is there.
    for (const url of ai) assert.equal(requested.get(url) ?? 0, 0, `fetched before the worker was there: ${url}`);
    await page.screenshot({ path: resolve(output, 'trained-default-download.png'), fullPage: true });
    // The network itself cannot be had: the table offers its fallbacks.
    await failNetwork(page);
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

  await check('Play Club meanwhile gives a table waiting for the download to Club for the match, and Trained stays the stored choice through a reload and a new game', async () => {
    // The network's download is held open, so it is under way for as long as
    // this check runs, after the reload too.
    const page = await open({ hold: true });
    await page.waitForSelector('.board');
    // East discards first; then, or at once, a Trained opponent waits, and
    // while what it waits for is the download the table offers Club meanwhile.
    const playable = '.hand button[data-hand-index]:not(:disabled), .call-options button:not(:disabled)';
    await page.waitForFunction(() => document.querySelector('[data-club-meanwhile], .hand button[data-hand-index]:not(:disabled)'), { timeout: 60000 });
    if (!await page.$('[data-club-meanwhile]')) await page.click('.hand button[data-hand-index]:not(:disabled)');
    await page.waitForSelector('[data-club-meanwhile]', { timeout: 60000 });
    assert.ok((await prompt(page)).includes(`downloading ${DOWNLOAD}`), 'the wait is the download');
    const chosen = await settings(page);
    assert.deepEqual([chosen.difficulty, chosen.opponents], ['neural', ['neural', 'neural', 'neural']]);
    await page.screenshot({ path: resolve(output, 'trained-default-meanwhile.png'), fullPage: true });
    // The wait holds nobody in Play either: the other modes open while the
    // download goes on, and on coming back the match is there as it was,
    // still waiting, with the offer.
    const waitingMatch = JSON.stringify(await match(page));
    assert.deepEqual(await page.$$eval('.game-modes button', buttons => buttons.map(button => button.disabled)), [false, false, false, false],
      'no mode is held by the waiting match');
    await page.click('.game-modes button:nth-child(2)');
    await page.waitForSelector('.agent-watch');
    assert.equal(await page.$('[data-club-meanwhile]'), null);
    assert.equal(await page.$eval('.game-modes button:nth-child(2)', button => button.getAttribute('aria-pressed')), 'true');
    await page.click('.game-modes button:first-child');
    await page.waitForSelector('[data-club-meanwhile]', { timeout: 60000 });
    assert.equal(JSON.stringify(await match(page)), waitingMatch, 'the match is as it was');
    assert.ok((await prompt(page)).includes(`downloading ${DOWNLOAD}`), 'and still waits for the download');
    await page.click('[data-club-meanwhile]');
    await page.waitForFunction(playable => !document.querySelector('[data-club-meanwhile]') && document.querySelector(playable), { timeout: 45000 }, playable);
    assert.equal(await page.$('.failure'), null, 'nothing failed, and nothing asks to be recovered');
    // The match is Club's from here, and shows it; its Trained seats were
    // given up by one recorded command, as after a failure.
    assert.equal((await match(page)).commands.filter(command => command.type === 'club').length, 1);
    // The select says Club stands in, in an entry of its own, rather than
    // showing Club as if it were the player's choice.
    assert.deepEqual(await page.$eval('select[aria-label="opponent strength"]', el => [el.value, el.selectedOptions[0].textContent.trim()]), ['meanwhile', 'Club meanwhile']);
    assert.deepEqual(await page.$$eval('.place .opponent-type', badges => badges.map(badge => badge.dataset.controller)), ['club', 'club', 'club']);
    // The button went as it was pressed: the hand has focus, not the page.
    assert.ok(await page.evaluate(() => document.querySelector('.hand').contains(document.activeElement)), 'focus goes on to the hand');
    // The table the player chose is still what is stored.
    assert.deepEqual(await settings(page), chosen);
    const before = (await match(page)).commands.length;
    await (await page.$('.call-options [data-choice=pass]') ?? await page.$('.hand button[data-hand-index]:not(:disabled)')).click();
    await page.waitForFunction((key, count) => JSON.parse(localStorage.getItem(key)).commands.length > count, { timeout: 45000 }, SAVE_KEY, before);
    assert.ok(!(await match(page)).commands.some(command => command.type === 'opponent'), 'no Trained move is played or waited for');
    assert.deepEqual(await settings(page), chosen, 'nor does a move store the stand-in');
    // Loaded again, nothing saved marks the stand-in: the match says Club and
    // the stored choice Trained, and the page keeps both as they are.
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForSelector(`${playable}, .screen`, { timeout: 60000 });
    assert.equal(await page.$eval('select[aria-label="opponent strength"]', el => el.value), 'meanwhile');
    assert.deepEqual(await settings(page), chosen);
    // A new game while the network is still not saved keeps the stand-in.
    page.on('dialog', dialog => void dialog.accept());
    const seed = (await match(page)).seed;
    await page.click('.restart');
    await page.waitForFunction((key, seed) => JSON.parse(localStorage.getItem(key)).seed !== seed, {}, SAVE_KEY, seed);
    assert.deepEqual((await match(page)).opponents, ['club', 'club', 'club']);
    assert.deepEqual(await settings(page), chosen);
    assert.equal(await page.$('.failure'), null);
    assert.equal(await page.$eval('select[aria-label="opponent strength"]', el => el.value), 'meanwhile');
    // The player's own pick of Club, though Club plays already, is a pick: it
    // deals Club, is stored, ends the stand-in, and sticks through a new game.
    const picked = (await match(page)).seed;
    await page.select('select[aria-label="opponent strength"]', 'club');
    await page.waitForFunction((key, seed) => JSON.parse(localStorage.getItem(key)).seed !== seed, {}, SAVE_KEY, picked);
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).difficulty === 'club', {}, SETTINGS_KEY);
    assert.deepEqual((await settings(page)).opponents, ['club', 'club', 'club']);
    await page.waitForFunction(() => document.querySelector('select[aria-label="opponent strength"]').value === 'club');
    assert.equal(await page.$('select[aria-label="opponent strength"] option[value="meanwhile"]'), null, 'no stand-in is said any more');
    const clubSeed = (await match(page)).seed;
    await page.click('.restart');
    await page.waitForFunction((key, seed) => JSON.parse(localStorage.getItem(key)).seed !== seed, {}, SAVE_KEY, clubSeed);
    assert.deepEqual((await match(page)).opponents, ['club', 'club', 'club']);
    assert.deepEqual((await settings(page)).opponents, ['club', 'club', 'club']);
    assert.equal(await page.$eval('select[aria-label="opponent strength"]', el => el.value), 'club');
    assert.deepEqual(page.problems, []);
  });

  await check('settings saved before generation 810 meet Trained once and keep everything else', async () => {
    const page = await open({ saved: { [SETTINGS_KEY]: JSON.stringify({ ...previousDefaults, hints: false, tileFace: 'van-gogh' }) } });
    await page.waitForSelector('.board');
    assert.equal(await page.$eval('select[aria-label="opponent strength"]', el => el.value), 'neural');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key))?.version === 6, {}, SETTINGS_KEY);
    assert.deepEqual(await settings(page), { version: 6, difficulty: 'neural', opponents: ['neural', 'neural', 'neural'],
      hints: false, confirmDiscards: false, shortcuts: true, tileFace: 'van-gogh', reviewAdviser: 'strong', reviewHideMove: false, reviewHideTiles: false,
      reviewFilter: 'flagged' });
    assert.deepEqual(page.problems, []);
  });

  await check('Club chosen in this build is kept, and nothing downloads the network', async () => {
    const page = await open({ saved: { [SETTINGS_KEY]: JSON.stringify(chosenClub) } });
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
    assert.deepEqual(await settings(page), chosenClub);
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
    // The note's words arrive once the review has been drawn, for a screen
    // reader to hear them.
    await page.waitForFunction(() => /Club reviews this hand/.test(document.querySelector('[data-review-fallback]')?.textContent ?? ''));
    assert.match(await page.$eval('.review .summary', el => el.textContent), /matched Club/);
    assert.equal(await page.$('.review [role="alert"]'), null);
    // The default is kept for when the network can be had, not replaced by the fallback.
    assert.equal((await settings(page)).reviewAdviser, 'strong');
    assert.equal(await page.evaluate(() => window.networkDownloads), 0);
    await page.screenshot({ path: resolve(output, 'trained-default-review-fallback.png'), fullPage: true });
    assert.deepEqual(page.problems, []);
  });

  await check('a Trained review shows the download it waits for and, when that fails, offers Club, which is then remembered', async () => {
    const page = await open({ hold: true, saved: { [SAVE_KEY]: JSON.stringify(finished) } });
    // Opened before or after the probe answers: a review shown by Club in the
    // meantime turns to Trained AI the moment the network is reported.
    await page.waitForSelector('.screen .quiet'); await page.click('.screen .quiet');
    await page.waitForSelector('select[aria-label="Review adviser"]');
    await downloading(page, '.review');
    // The status line says only that Trained AI is reviewing; the count and
    // the download's note, which change too often to be read out, are beside it.
    assert.match(await page.$eval('.review', el => el.textContent), /Trained AI is reviewing your decisions…\s*0 of \d+ · downloading the trained network/);
    assert.equal(await page.$eval('.review [data-review-progress]', el => el.textContent.trim()), 'Trained AI is reviewing your decisions…');
    assert.match(await page.$eval('.review [data-review-count]', el => el.textContent.trim()), /^0 of \d+ · downloading the trained network/);
    await failNetwork(page);
    await page.waitForSelector('.review [role="alert"] [data-review-club]', { timeout: 120000 });
    assert.ok(await page.evaluate(() => window.networkDownloads) > 0, 'the trained review asked for the network');
    assert.deepEqual(await page.$$eval('.review [role="alert"] button', buttons => buttons.map(button => button.textContent.trim())),
      ['Retry Trained review', 'Review with Club']);
    // Either button goes from the page with the failure it answers, so focus
    // goes on to the review's heading rather than dropping to the page.
    // Retry, pressed by keyboard: the retry fails as the first try did, and
    // says so in an alert of its own.
    const onHeading = () => page.evaluate(() => document.activeElement?.matches('.review h2') ?? false);
    await page.$eval('.review [role="alert"]', el => { el.dataset.first = ''; });
    await page.focus('.review [data-review-retry]');
    await page.keyboard.press('Enter');
    assert.equal(await onHeading(), true, 'Retry Trained review hands focus on to the heading');
    await page.waitForSelector('.review [role="alert"]:not([data-first]) [data-review-club]', { timeout: 120000 });
    assert.equal(await onHeading(), true, 'focus stays there as the retry fails');
    // Review with Club, pressed by pointer.
    await page.click('[data-review-club]');
    await page.waitForFunction(() => document.querySelector('.review .summary')?.textContent.includes('matched Club'));
    assert.equal(await onHeading(), true, 'Review with Club hands focus on to the heading');
    assert.equal(await page.$eval('select[aria-label="Review adviser"]', el => el.value), 'club');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).reviewAdviser === 'club', {}, SETTINGS_KEY);
    assert.deepEqual(page.problems, []);
  });

  await check('Agent watch puts Trained in every seat, shows its download and can watch the same table with Club when the network fails', async () => {
    const page = await open({ hold: true, mode: 'watch' });
    const lineup = () => page.$$eval('.agent-fields select', selects => selects.map(select => select.value));
    await page.waitForFunction(() => {
      const seats = [...document.querySelectorAll('.agent-fields select')];
      return seats.length === 4 && seats.every(select => select.value === 'full');
    });
    assert.deepEqual(await lineup(), ['full', 'full', 'full', 'full']);
    await page.click('.watch-setup .primary');
    await downloading(page, '.agent-watch');
    assert.match(await page.$eval('.agent-watch', el => el.textContent), /The agents are thinking… downloading the trained network/);
    await failNetwork(page);
    await page.waitForSelector('.agent-watch [role="alert"] [data-watch-club]', { timeout: 120000 });
    assert.ok(await page.evaluate(() => window.networkDownloads) > 0);
    await page.click('[data-watch-club]');
    await page.waitForSelector('.weight-row.best .choice-action:not(:disabled)', { timeout: 45000 });
    assert.equal(await page.$('.agent-watch [role="alert"]'), null);
    assert.deepEqual(await lineup(), ['club', 'club', 'club', 'club']);
    assert.doesNotMatch(await page.$eval('.watch-table', el => el.textContent), /Trained/);
    assert.deepEqual(page.problems, []);
  });

  await check('a new guided game is advised by Trained, shows its download and offers Club when that advice cannot load', async () => {
    const page = await open({ hold: true, mode: 'guided' });
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
    await downloading(page, '.guide-prompt');
    assert.match(await page.$eval('.guide-prompt', el => el.textContent), /Your adviser is thinking… downloading the trained network/);
    await failNetwork(page);
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
