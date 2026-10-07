/** Real mounted agent modes with deterministically delayed availability. */
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import { emptyGuided, guidedEvent, GUIDED_FORMAT, GUIDED_KEY } from '../src/lib/guided-game.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const { check: run, openContext, report, results } = browserChecks();
let server, browser;
const physical = '[aria-label="Physical play agent"]';
const adviser = '[aria-label="Guided game adviser"]';
const lineup = page => page.$$eval('.agent-fields select', controls => controls.map(control => control.value));
async function available(page) {
  await page.click('[data-available]');
  await page.waitForFunction(() => Boolean(document.querySelector('option[value="full"]:not(:disabled)')));
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}
const ready = { watch: '.agent-fields select', guided: '.guided-controls:not(:disabled)', physical: '.physical-editor:not(:disabled)' };
async function check(name, mode, body, { saved } = {}) {
  const problems = [];
  const passed = await run(name, async () => {
    const page = await (await openContext(browser)).newPage();
    page.on('pageerror', error => problems.push(error.message));
    page.on('console', message => { if (['warn', 'error'].includes(message.type())) problems.push(message.text()); });
    await page.setViewport({ width: 1100, height: 1000 });
    if (saved) await page.evaluateOnNewDocument(entries => {
      for (const [key, value] of entries) if (localStorage.getItem(key) === null) localStorage.setItem(key, value);
    }, Object.entries(saved));
    await page.goto(`${server.resolvedUrls.local[0]}tests/fixtures/agent-defaults.html?mode=${mode}`, { waitUntil: 'networkidle0' });
    await page.waitForSelector(ready[mode]);
    await body(page);
    assert.deepEqual(problems, []);
  });
  if (!passed) results.at(-1).problems = problems;
}
const guided = page => page.evaluate(key => JSON.parse(localStorage.getItem(key))?.game, GUIDED_KEY);
try {
  await mkdir(resolve(root, 'test-results'), { recursive: true });
  server = await createServer({ root, configFile: false, plugins: [svelte()], logLevel: 'warn', server: { host: '127.0.0.1', port: 0 } });
  await server.listen();
  browser = await launchChrome();
  await check('unavailable model keeps built-in controls enabled without offering Trained', 'physical', async page => {
    await page.click('[data-unavailable]');
    assert.equal(await page.$eval(physical, el => el.value), 'club');
    assert.deepEqual(await page.$$eval(`${physical} option:not(:disabled)`, options => options.map(option => option.value)), ['beginner', 'club']);
    await page.select(physical, 'beginner');
    assert.equal(await page.$eval('.physical-toolbar .primary', button => button.disabled), false);
    assert.equal(await page.$(`${physical} option[value="full"]`), null);
  });
  await check('late availability selects the trained default only while Physical adviser is untouched', 'physical', async page => {
    assert.equal(await page.$eval(physical, el => el.value), 'club');
    await available(page);
    assert.equal(await page.$eval(physical, el => el.value), 'full');
    await page.select(physical, 'beginner');
    await page.click('[data-unavailable]');
    await available(page);
    assert.equal(await page.$eval(physical, el => el.value), 'beginner');
  });
  await check('explicit Physical Club selection survives a later trained result', 'physical', async page => {
    await page.select(physical, 'club');
    await available(page);
    assert.equal(await page.$eval(physical, el => el.value), 'club');
  });
  await check('starting a Club analysis pins that adviser even when the entered hand is incomplete', 'physical', async page => {
    await page.click('.physical-toolbar .primary');
    await page.waitForSelector('.physical-play .failure');
    await available(page);
    assert.equal(await page.$eval(physical, el => el.value), 'club');
    assert.ok(await page.$('.physical-play .failure'), 'a late probe must not erase the analysis result');
  });
  await check('late availability makes every seat of an untouched Watch lineup Trained', 'watch', async page => {
    assert.deepEqual(await lineup(page), ['club', 'club', 'club', 'club']);
    await available(page);
    assert.deepEqual(await lineup(page), ['full', 'full', 'full', 'full']);
    await page.click('[data-unavailable]');
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.deepEqual(await lineup(page), ['full', 'full', 'full', 'full'], 'a Trained lineup, once offered, stays');
  });
  await check('explicit Watch draft changes survive a late probe', 'watch', async page => {
    await page.select('.agent-fields label:nth-child(2) select', 'beginner');
    await available(page);
    assert.deepEqual(await lineup(page), ['club', 'beginner', 'club', 'club']);
  });
  await check('an already started watched game never silently switches agents', 'watch', async page => {
    await page.click('.watch-setup .primary');
    await page.waitForSelector('.watch-table');
    await available(page);
    assert.deepEqual(await lineup(page), ['club', 'club', 'club', 'club']);
    assert.doesNotMatch(await page.$eval('.watch-table', el => el.textContent), /Trained/);
  });
  await check('a new guided game is advised by Trained, shown even before the network is known', 'guided', async page => {
    // Offered as the guide's own adviser while availability is unknown, but
    // not offered as a fresh choice until the network is there.
    assert.equal(await page.$eval(adviser, el => el.value), 'full');
    assert.equal(await page.$eval(`${adviser} option[value="full"]`, el => el.disabled), true);
    await available(page);
    assert.equal(await page.$eval(adviser, el => el.value), 'full');
    assert.deepEqual(await page.$$eval(`${adviser} option:not(:disabled)`, options => options.map(option => option.value)), ['beginner', 'club', 'full']);
  });
  await check('a guide saved with Club keeps Club, and a new guided game keeps it too', 'guided', async page => {
    await available(page);
    assert.equal(await page.$eval(adviser, el => el.value), 'club');
    page.on('dialog', dialog => void dialog.accept());
    await page.evaluate(() => [...document.querySelectorAll('.guided-heading button')].find(button => button.textContent.trim() === 'New guided game').click());
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key))?.game.log.length === 0, {}, GUIDED_KEY);
    const game = await guided(page);
    assert.equal(game.state.agent, 'club');
    assert.equal(game.state.stage, 'setup');
    assert.equal(await page.$eval(adviser, el => el.value), 'club');
  }, { saved: { [GUIDED_KEY]: GUIDED_FORMAT.encode(guidedEvent(emptyGuided('club'), { type: 'setup' })) } });
} finally {
  await browser?.close();
  await server?.close();
  await report('agent-default checks', resolve(root, 'test-results/agent-defaults.json'));
}
