/** Review a restored hand with the actual production UI and trained worker.
 * Each question Trained AI hesitates over is asked again of the renamed game,
 * which must be the renamed game (renamed-requests.mjs), and judged by the
 * mean of the two answers (src/lib/suits.js). */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { MANIFEST } from '../src/lib/model-manifest.js';
import { average, hesitates } from '../src/lib/suits.js';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { renamedRequests, watchRenamedRequests } from './renamed-requests.mjs';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const web = fileURLToPath(new URL('../', import.meta.url)), dist = resolve(web, 'dist'), output = resolve(web, 'test-results');
const networkUrl = `${MANIFEST.origin}/${MANIFEST.object}`;
const match = new MatchSession(Game, 1, 'club');
let snapshot, notes, first, choices, translations, reachTranslations, masks, reachMasks;
try {
  match.advance(false);
  for (let n = 0; n < 250 && match.view.phase !== 'over'; n++) {
    const choice = match.choices.find(c => ['ron', 'tsumo'].includes(c.kind))
      ?? match.choices.find(c => c.kind === 'pass') ?? match.choices.find(c => c.kind === 'discard') ?? match.choices[0];
    match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
  }
  assert.equal(match.view.phase, 'over');
  snapshot = match.snapshot(); notes = match.engine.review();
  assert.ok(notes.length > 1);
  first = { planes: Array.from(match.engine.review_observation_mortal(0)), mask: Array.from(match.engine.review_mask_mortal(0)) };
  choices = notes.map((_, index) => match.engine.review_choices(index));
  translations = notes.map((_, index) => Array.from({ length: 46 }, (_, action) =>
    match.engine.review_action_from_mortal(index, action, false)));
  reachTranslations = notes.map((_, index) => Array.from({ length: 34 }, (_, action) =>
    match.engine.review_action_from_mortal(index, action, true)));
  masks = notes.map((_, index) => Array.from(match.engine.review_mask_mortal(index)));
  reachMasks = notes.map((_, index) => Array.from(match.engine.review_mask_after_reach(index)));
} finally { match.dispose(); }

const server = createServer(createFixtureHandler({ root: dist, publicRoot: dist }));
const cases = browserChecks();
let browser;
const saved = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
const select = 'select[aria-label="Review adviser"]';
const percent = weight => weight > 0 && weight < .001 ? '<0.1%' : `${(weight * 100).toFixed(1)}%`;

async function openReview(page) {
  await page.waitForSelector('.screen .quiet');
  await page.click('.screen .quiet');
  await page.waitForSelector(select);
}

async function strongResults(page) {
  await page.waitForFunction(() => document.querySelector('.review[aria-busy="false"] .policy-preference'), { timeout: 90000 });
  await page.evaluate(() => [...document.querySelectorAll('.review .tabs button')]
    .find(button => button.textContent.includes('Every decision'))?.click());
  await page.waitForFunction(count => document.querySelectorAll('.review li').length === count, {}, notes.length);
  const actual = await page.$$eval('.review li', elements => elements.map(el => ({
    played: el.querySelector('.played .what').textContent,
    advised: el.querySelector('.advised .what')?.textContent ?? el.querySelector('.played .what').textContent,
    weight: el.querySelector('.policy-preference strong').textContent,
    agreed: el.classList.contains('agreed'),
  })));
  const inference = await page.evaluate(() => window.reviewAnswers);
  let cursor = 0, again = 0;
  // A question's answer, and where it hesitates the mean of it and the
  // renamed game's, the next answer.
  const asked = mask => {
    const answer = inference[cursor++];
    assert.ok(answer, 'each historical decision needs a real trained answer');
    if (!hesitates(answer.weights)) return answer;
    const renamed = inference[cursor++];
    assert.ok(renamed, 'a question Trained AI hesitates over is asked again of the renamed game');
    again++;
    return { ...answer, ...average(answer.weights, renamed.weights, mask) };
  };
  for (let index = 0; index < notes.length; index++) {
    const answer = asked(masks[index]);
    const afterReach = answer.action === 37;
    const action = afterReach ? asked(reachMasks[index]).action : answer.action;
    assert.ok(Number.isInteger(action), 'a reach must also answer which tile is discarded');
    const chosen = (afterReach ? reachTranslations : translations)[index][action];
    const preferred = choices[index].find(c => c.index === chosen);
    assert.ok(preferred);
    assert.equal(actual[index].played, notes[index].played);
    assert.equal(actual[index].advised, actual[index].agreed ? notes[index].played : preferred.label);
    // Independently sum the real worker's aliases for this historical move.
    const weight = afterReach ? answer.weights[37]
      : answer.weights.reduce((sum, value, action) => sum
        + (action !== 37 && translations[index][action] === chosen ? value : 0), 0);
    assert.equal(actual[index].weight, percent(Math.min(1, weight)));
    assert.equal(actual[index].agreed, preferred.kind === notes[index].played_kind
      && (preferred.tile ?? null) === (notes[index].played_tile ?? null));
  }
  assert.equal(cursor, inference.length, 'no extra or missing review answers');
  assert.equal(await page.$('.review .numbers'), null);
  const requests = await page.evaluate(() => window.reviewRequests);
  assert.equal(requests.length, cursor);
  assert.ok(requests.every(request => request.url === networkUrl));
  assert.deepEqual({ planes: requests[0].planes, mask: requests[0].mask }, first);
  // Every question asked again was the renamed game's, right after its own.
  assert.deepEqual(await renamedRequests(page), { checked: again, faults: [], awaited: false });
  assert.deepEqual(await saved(page), snapshot);
  return again;
}

try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  browser = await launchChrome();
  await cases.check('Club and Trained AI review: real percentages, retry, cached results and remembered choice on mobile', async () => {
    const page = await (await cases.openContext(browser)).newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.setViewport({ width: 1100, height: 900 });
    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
    await watchRenamedRequests(page);
    await page.evaluateOnNewDocument((saveKey, settingsKey, initial) => {
      if (!localStorage.getItem(saveKey)) localStorage.setItem(saveKey, JSON.stringify(initial));
      if (!localStorage.getItem(settingsKey)) localStorage.setItem(settingsKey, JSON.stringify({
        version: 2, difficulty: 'club', trainedModel: 'full', reviewAdviser: 'club', hints: true,
      }));
      window.reviewRequests = []; window.reviewAnswers = [];
      const NativeWorker = window.Worker;
      window.Worker = class extends NativeWorker {
        constructor(...args) {
          super(...args);
          this.addEventListener('message', ({ data }) => {
            if (data.analysis) window.reviewAnswers.push(data.analysis);
            window.renamedRequests.answered(data);
          });
        }
        postMessage(data, ...args) {
          if (data.details) {
            // Exercise the actual UI failure/retry path once; all successful
            // results below still come from the real shipped model and worker.
            if (!sessionStorage.getItem('review-failure-tested')) {
              sessionStorage.setItem('review-failure-tested', 'yes');
              queueMicrotask(() => this.dispatchEvent(new MessageEvent('message', {
                data: { id: data.id, error: 'Review connection interrupted. Try again.' },
              })));
              return;
            }
            window.reviewRequests.push({ url: data.url,
              ...(window.reviewRequests.length ? {} : { planes: Array.from(data.planes), mask: Array.from(data.mask) }),
            });
            window.renamedRequests.posted(data);
          }
          super.postMessage(data, ...args);
        }
      };
    }, SAVE_KEY, SETTINGS_KEY, snapshot);
    await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/`, { waitUntil: 'networkidle0' });
    await openReview(page);
    assert.equal(await page.$eval(select, el => el.value), 'club');
    assert.match(await page.$eval('.review .summary', el => el.textContent), /matched Club/);
    if (notes.some(note => !note.agreed)) assert.ok(await page.$('.review .numbers'));
    await page.waitForFunction(() => !document.querySelector('select[aria-label="Review adviser"] option[value="strong"]').disabled);
    await page.select(select, 'strong');
    await page.waitForSelector('.review .retry', { timeout: 90000 });
    assert.match(await page.$eval('.review [role="alert"]', el => el.textContent), /Review connection interrupted/);
    await page.click('.review .retry');
    // However many questions the network hesitates over, none included for a
    // network sure of the whole hand, each was paired with its renamed game
    // and the answers averaged (strongResults); the count is only said.
    console.log(`  Trained AI asked ${await strongResults(page)} of the hand's questions again, renamed`);
    await page.screenshot({ path: resolve(output, 'adviser-review-desktop.png'), fullPage: true });
    await page.select(select, 'club');
    await page.waitForFunction(() => document.querySelector('.review .summary')?.textContent.includes('matched Club'));
    assert.equal(await page.$('.policy-preference'), null);
    await page.select(select, 'strong');
    await strongResults(page); // No additional worker requests for a cached hand.
    const preferences = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), SETTINGS_KEY);
    assert.equal(preferences.reviewAdviser, 'strong'); assert.equal(preferences.difficulty, 'club');
    assert.equal(Object.hasOwn(preferences, 'trainedModel'), false, 'retired network preferences are no longer persisted');
    await page.setViewport({ width: 360, height: 800, hasTouch: true });
    await page.reload({ waitUntil: 'networkidle0' });
    await openReview(page);
    assert.equal(await page.$eval(select, el => el.value), 'strong');
    await strongResults(page);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'no horizontal overflow');
    await page.screenshot({ path: resolve(output, 'adviser-review-mobile.png'), fullPage: true });
    assert.deepEqual(errors, []);
  });
} finally {
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
  await cases.report('adviser review checks', resolve(output, 'adviser-review.json'));
}
