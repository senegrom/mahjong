/** Mount real components under a reactive parent; never execute sliced markup. */
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
const state = page => page.$eval('[data-parent-state]', el => JSON.parse(el.textContent));
const openSettings = async page => {
  await page.click('.settings-trigger');
  await page.waitForSelector('.settings-dialog[open]');
  assert.equal(await page.$eval('.settings-dialog', el => el.matches(':modal')), true);
};
async function check(name, viewport, body) {
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  const problems = [];
  let failure;
  page.on('pageerror', error => problems.push(error.message));
  page.on('console', message => { if (['warn', 'error'].includes(message.type())) problems.push(message.text()); });
  try {
    await page.setViewport(viewport);
    await page.goto(`${server.resolvedUrls.local[0]}tests/fixtures/app-components.html`, { waitUntil: 'networkidle0' });
    await page.waitForSelector('[data-parent-state]');
    await body(page);
    assert.deepEqual(problems, [], 'Mounted components must not emit ownership or runtime warnings');
  } catch (error) {
    failure = error;
    await page.screenshot({ path: resolve(root, 'test-results', `components-${results.length}.png`), fullPage: true }).catch(() => {});
    await writeFile(resolve(root, 'test-results', `components-${results.length}.html`), await page.content()).catch(() => {});
  } finally {
    try { await context.close(); } catch (error) { failure ??= error; }
  }
  results.push({ name, passed: !failure, ...(failure ? { error: failure.stack, problems } : {}) });
  console[failure ? 'error' : 'log'](`${failure ? 'FAIL' : 'PASS'} ${name}${failure ? `\n${failure.stack}` : ''}`);
}
const desktop = { width: 1100, height: 1000 }, phone = { width: 390, height: 844 };
try {
  await mkdir(resolve(root, 'test-results'), { recursive: true });
  server = await createServer({ root, configFile: false, plugins: [svelte()], logLevel: 'warn', server: { host: '127.0.0.1', port: 0 } });
  await server.listen();
  const executablePath = process.env.CHROME_BIN || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(executablePath, 'Set CHROME_BIN to Chrome/Chromium');
  browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  await check('mounted bindings, context and action callbacks reach the parent', desktop, async page => {
    assert.equal((await state(page)).handBound, true);
    await page.click('.confirm-discard button:last-child');
    await page.waitForFunction(() => !document.querySelector('.confirm-discard'));
    assert.equal((await state(page)).selected, null);
    await page.click('.hand button[data-hand-index="1"]');
    await page.waitForSelector('.confirm-discard');
    assert.equal((await state(page)).selected, 1);
    await page.click('.options > summary');
    const boxes = await page.$$('.options input[type=checkbox]');
    await boxes[1].click();
    assert.equal((await state(page)).confirmDiscards, false);
    assert.equal((await state(page)).selected, null);
    await boxes[0].click();
    await page.waitForFunction(() => !document.querySelector('.mine .hint'));
    assert.equal((await state(page)).hints, false);
    await page.select('select[aria-label="Tile face"]', 'matisse');
    await page.waitForSelector('.hand .tile.matisse');
    assert.equal((await state(page)).tileFace, 'matisse');
    await page.click('.call-options button[data-choice="riichi"]');
    assert.ok((await state(page)).callbacks.includes('riichi'));
    await page.click('.game-modes button:nth-child(3)');
    assert.equal((await state(page)).mode, 'physical');
  });
  await check('real offline status reacts to a parent update and invokes download once', desktop, async page => {
    await page.click('.offline-settings > summary');
    assert.match(await page.$eval('[data-ai-status]', el => el.textContent), /Interrupted fixture download/);
    await page.click('[data-download-ai]');
    await page.waitForFunction(() => document.querySelector('[data-offline-status]').textContent.includes('game + AI ready'));
    assert.equal(await page.$('[data-download-ai]'), null);
    assert.deepEqual((await state(page)).callbacks, ['download']);
  });
  await check('native compact modal traps focus, blocks the hand and restores its opener', phone, async page => {
    await openSettings(page);
    assert.equal(await page.evaluate(() => document.activeElement.hasAttribute('data-close-settings')), true);
    await page.evaluate(() => document.querySelector('.hand button').focus());
    assert.equal(await page.evaluate(() => document.querySelector('.settings-dialog').contains(document.activeElement)), true);
    for (const expanded of [false, true]) {
      if (expanded) {
        await page.click('.options > summary');
        await page.click('.offline-settings > summary');
      }
      for (const reverse of [false, true]) {
        if (reverse) await page.keyboard.down('Shift');
        try {
          for (let i = 0; i < 18; i++) {
            await page.keyboard.press('Tab');
            assert.equal(await page.evaluate(() => document.querySelector('.settings-dialog').contains(document.activeElement)), true,
              `${reverse ? 'Shift+Tab' : 'Tab'} escaped the modal (expanded=${expanded})`);
          }
        } finally { if (reverse) await page.keyboard.up('Shift'); }
      }
    }
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('.settings-dialog[open]'));
    assert.equal(await page.evaluate(() => document.activeElement.matches('.settings-trigger')), true);
    assert.equal((await state(page)).settingsOpen, false);
  });
  await check('responsive changes preserve preferences, release modality and remove listeners on unmount', phone, async page => {
    await openSettings(page);
    await page.click('.options > summary');
    await page.click('.options input[type=checkbox]');
    await page.setViewport(desktop);
    await page.waitForFunction(() => !document.querySelector(':modal') && document.activeElement.matches('.options > summary'));
    assert.equal((await state(page)).hints, false);
    assert.equal((await state(page)).settingsOpen, false);
    await page.setViewport(phone);
    await page.waitForFunction(() => document.activeElement.matches('.settings-trigger'));
    await openSettings(page);
    assert.equal(await page.$eval('.options input[type=checkbox]', el => el.checked), false);
    await page.evaluate(() => window.unmountFixture());
    assert.equal(await page.$('dialog'), null);
    await page.setViewport(desktop);
  });
  await check('settings-to-opponent dialog handoff keeps a single modal and independent draft', phone, async page => {
    await openSettings(page);
    await page.click('.edit-table');
    await page.waitForSelector('.custom-dialog[open]');
    assert.equal(await page.$$eval(':modal', els => els.length), 1);
    assert.equal(await page.$eval('.settings-dialog', el => el.open), false);
    await page.select('.opponent-fields select', 'beginner');
    await page.click('.custom-actions button:first-child');
    assert.deepEqual((await state(page)).opponents, ['club', 'club', 'club']);
    await page.waitForFunction(() => document.activeElement.matches('.settings-trigger'));
    await openSettings(page);
    await page.click('.edit-table');
    assert.deepEqual(await page.$$eval('.opponent-fields select', els => els.map(el => el.value)), ['club', 'club', 'club']);
    await page.keyboard.press('Escape');
  });
  await check('scoped wait counts and drawn gaps retain their effective styles without control leakage', desktop, async page => {
    for (const [width, gap] of [[1100, 18], [820, 18], [390, 14]]) {
      await page.setViewport({ width, height: 1000 });
      const style = await page.evaluate(() => {
        const count = document.querySelector('.wait .remaining'), tile = document.querySelector('.wait .tile');
        const drawn = document.querySelector('.hand-tile[data-hand-drawn=true]'), button = drawn.querySelector('button');
        return { position: getComputedStyle(count).position, below: count.getBoundingClientRect().top >= tile.getBoundingClientRect().bottom - 1,
          gap: parseFloat(getComputedStyle(drawn).marginInlineStart), padding: getComputedStyle(button).padding,
          margin: parseFloat(getComputedStyle(button).marginInlineStart), control: button.classList.contains('app-control') };
      });
      assert.deepEqual(style, { position: 'static', below: true, gap, padding: '0px', margin: 0, control: false });
      await page.screenshot({ path: resolve(root, 'test-results', `components-style-${width}.png`), fullPage: true });
    }
  });
} finally {
  await browser?.close();
  await server?.close();
  await writeFile(resolve(root, 'test-results/components.json'), JSON.stringify(results, null, 2));
}
if (results.some(result => !result.passed)) process.exitCode = 1;
