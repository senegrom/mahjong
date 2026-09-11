/** Review a restored hand with the actual production UI and trained worker. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { MODEL_FILES } from '../src/lib/model-package.js';
import { createFixtureHandler } from './static-fixture-server.mjs';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const web = fileURLToPath(new URL('../', import.meta.url)), dist = resolve(web, 'dist'), output = resolve(web, 'test-results');
const trainedShipped = existsSync(resolve(dist, MODEL_FILES.full));
const match = new MatchSession(Game, 1, 'club');
let snapshot, notes, first, choices, translations, reachTranslations;
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
} finally { match.dispose(); }

const server = createServer(createFixtureHandler({ root: dist, publicRoot: dist }));
const results = [];
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
  let cursor = 0;
  for (let index = 0; index < notes.length; index++) {
    const answer = inference[cursor++];
    assert.ok(answer, 'each historical decision needs a real trained answer');
    const afterReach = answer.action === 37;
    const action = afterReach ? inference[cursor++]?.action : answer.action;
    assert.ok(Number.isInteger(action), 'a reach must also answer which tile is discarded');
    const chosen = (afterReach ? reachTranslations : translations)[index][action];
    const preferred = choices[index].find(c => c.index === chosen);
    assert.ok(preferred);
    assert.equal(actual[index].played, notes[index].played);
    assert.equal(actual[index].advised, actual[index].agreed ? notes[index].played : preferred.label);
    assert.equal(actual[index].weight, percent(answer.weights[answer.action]));
    assert.equal(actual[index].agreed, preferred.kind === notes[index].played_kind
      && (preferred.tile ?? null) === (notes[index].played_tile ?? null));
  }
  assert.equal(cursor, inference.length, 'no extra or missing review answers');
  assert.equal(await page.$('.review .numbers'), null);
  const requests = await page.evaluate(() => window.reviewRequests);
  assert.equal(requests.length, cursor);
  assert.ok(requests.every(request => request.url.endsWith(`/${MODEL_FILES.full}`)));
  assert.deepEqual({ planes: requests[0].planes, mask: requests[0].mask }, first);
  assert.deepEqual(await saved(page), snapshot);
}

try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  const executablePath = process.env.CHROME_BIN || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(executablePath, 'Set CHROME_BIN to Chrome/Chromium');
  browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const context = await browser.createBrowserContext();
  const name = 'Club and Trained AI review: real percentages, retry, cached results and remembered choice on mobile';
  try {
    const page = await context.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.setViewport({ width: 1100, height: 900 });
    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
    await page.evaluateOnNewDocument((saveKey, settingsKey, initial) => {
      if (!localStorage.getItem(saveKey)) localStorage.setItem(saveKey, JSON.stringify(initial));
      if (!localStorage.getItem(settingsKey)) localStorage.setItem(settingsKey, JSON.stringify({
        version: 1, difficulty: 'club', trainedModel: 'full', reviewAdviser: 'club', hints: true,
      }));
      window.reviewRequests = []; window.reviewAnswers = [];
      const NativeWorker = window.Worker;
      window.Worker = class extends NativeWorker {
        constructor(...args) {
          super(...args);
          this.addEventListener('message', ({ data }) => { if (data.analysis) window.reviewAnswers.push(data.analysis); });
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
    if (trainedShipped) {
      await page.waitForFunction(() => !document.querySelector('select[aria-label="Review adviser"] option[value="strong"]').disabled);
      await page.select(select, 'strong');
      await page.waitForSelector('.review .retry', { timeout: 90000 });
      assert.match(await page.$eval('.review [role="alert"]', el => el.textContent), /Review connection interrupted/);
      await page.click('.review .retry');
      await strongResults(page);
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
    } else assert.equal(await page.$eval(`${select} option[value="strong"]`, el => el.disabled), true);
    assert.deepEqual(errors, []);
    results.push({ name, passed: true }); console.log(`PASS ${name}`);
  } catch (error) {
    results.push({ name, passed: false, error: error.stack }); console.error(`FAIL ${name}\n${error.stack}`);
  } finally { await context.close(); }
} finally {
  await writeFile(resolve(output, 'adviser-review.json'), JSON.stringify(results, null, 2));
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
}
if (results.some(result => !result.passed)) process.exitCode = 1;
