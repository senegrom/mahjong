/** Real mounted agent modes with deterministically delayed availability. */
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import puppeteer from 'puppeteer-core';

const root = fileURLToPath(new URL('../', import.meta.url));
const results = [];
let server, browser;
const physical = '[aria-label="Physical play agent"]';
const lineup = page => page.$$eval('.agent-fields select', controls => controls.map(control => control.value));
async function available(page) {
  await page.click('[data-available]');
  await page.waitForFunction(() => Boolean(document.querySelector('option[value="full"]')));
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}
async function check(name, mode, body) {
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  const problems = [];
  let failure;
  page.on('pageerror', error => problems.push(error.message));
  page.on('console', message => { if (['warn', 'error'].includes(message.type())) problems.push(message.text()); });
  try {
    await page.setViewport({ width: 1100, height: 1000 });
    await page.goto(`${server.resolvedUrls.local[0]}tests/fixtures/agent-defaults.html?mode=${mode}`, { waitUntil: 'networkidle0' });
    await page.waitForSelector(mode === 'watch' ? '.agent-fields select' : '.physical-editor:not(:disabled)');
    await body(page);
    assert.deepEqual(problems, []);
  } catch (error) { failure = error; }
  finally {
    try { await context.close(); } catch (error) { failure ??= error; }
  }
  results.push({ name, passed: !failure, ...(failure ? { error: failure.stack, problems } : {}) });
  console[failure ? 'error' : 'log'](`${failure ? 'FAIL' : 'PASS'} ${name}${failure ? `\n${failure.stack}` : ''}`);
}
try {
  await mkdir(resolve(root, 'test-results'), { recursive: true });
  server = await createServer({ root, configFile: false, plugins: [svelte()], logLevel: 'warn', server: { host: '127.0.0.1', port: 0 } });
  await server.listen();
  const executablePath = process.env.CHROME_BIN || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(executablePath, 'Set CHROME_BIN to Chrome/Chromium');
  browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
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
  await check('late availability upgrades only an untouched Watch lineup', 'watch', async page => {
    assert.deepEqual(await lineup(page), ['club', 'club', 'beginner', 'club']);
    await available(page);
    assert.deepEqual(await lineup(page), ['full', 'full', 'beginner', 'club']);
  });
  await check('explicit Watch draft changes survive a late probe', 'watch', async page => {
    await page.select('.agent-fields label:nth-child(2) select', 'beginner');
    await available(page);
    assert.deepEqual(await lineup(page), ['club', 'beginner', 'beginner', 'club']);
  });
  await check('an already started watched game never silently switches agents', 'watch', async page => {
    await page.click('.watch-setup .primary');
    await page.waitForSelector('.watch-table');
    await available(page);
    assert.deepEqual(await lineup(page), ['club', 'club', 'beginner', 'club']);
    assert.doesNotMatch(await page.$eval('.watch-table', el => el.textContent), /Trained/);
  });
} finally {
  await browser?.close();
  await server?.close();
  await writeFile(resolve(root, 'test-results/agent-defaults.json'), JSON.stringify(results, null, 2));
}
if (results.some(result => !result.passed)) process.exitCode = 1;
