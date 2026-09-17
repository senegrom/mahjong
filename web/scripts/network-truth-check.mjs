// Does the page play what the network says?
//
// Every trained decision is recorded as the page made it: the planes it sent,
// the legality mask, and the move it took. The same planes are then put to the
// same checkpoint in PyTorch (`neural/tests/replay_browser_decisions.py`), and
// the two must name the same move. A network served the wrong planes, the
// wrong mask or the wrong graph disagrees here rather than in a person's game.
//
//   node web/scripts/network-truth-check.mjs [out.json] [decisions]
import { createServer } from 'node:http';
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';

const web = fileURLToPath(new URL('../', import.meta.url));
const out = resolve(process.argv[2] ?? resolve(web, 'test-results', 'network-truth.json'));
const wanted = Number(process.argv[3] ?? 12);
const handler = createFixtureHandler({ root: resolve(web, 'dist'), publicRoot: resolve(web, 'dist') });
const server = createServer((request, response) => void handler(request, response));
await new Promise(done => server.listen(0, '127.0.0.1', done));

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_BIN,
  args: ['--no-sandbox', '--enable-features=SharedArrayBuffer'],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewport({ width: 1100, height: 900 });
  await page.evaluateOnNewDocument((saveKey, settingsKey) => {
    localStorage.setItem(settingsKey, JSON.stringify({
      version: 1, difficulty: 'custom', opponents: ['neural', 'neural', 'neural'],
      trainedModel: 'full', hints: true,
    }));
    localStorage.removeItem(saveKey);
    window.asked = [];
    const NativeWorker = window.Worker;
    const base64 = buffer => {
      const bytes = new Uint8Array(buffer);
      let text = '';
      for (let at = 0; at < bytes.length; at += 8192) {
        text += String.fromCharCode(...bytes.subarray(at, at + 8192));
      }
      return btoa(text);
    };
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.pending = new Map();
        this.addEventListener('message', ({ data }) => {
          const asked = this.pending.get(data.id);
          if (!asked || data.action === undefined) return;
          this.pending.delete(data.id);
          window.asked.push({ ...asked, action: data.action, value: data.value ?? null });
        });
      }
      postMessage(data, ...args) {
        if (data?.planes) {
          // The buffer is transferred away, so it is copied before it goes.
          this.pending.set(data.id, {
            planes: base64(data.planes.slice().buffer),
            mask: Array.from(data.mask, allowed => (allowed ? 1 : 0)),
          });
        }
        super.postMessage(data, ...args);
      }
    };
  }, SAVE_KEY, SETTINGS_KEY);

  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/`, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.hand', { timeout: 180000 });
  // The three trained seats answer on their own; the human seat is played by
  // taking the first legal discard, which is not the subject here.
  const deadline = Date.now() + 10 * 60 * 1000;
  while (await page.evaluate(() => window.asked.length) < wanted && Date.now() < deadline) {
    const failure = await page.$('.failure');
    if (failure) throw new Error(await page.$eval('.failure', el => el.textContent.trim()));
    const screen = await page.$('.screen .buttons .primary');
    if (screen) { await screen.click(); await page.waitForTimeout?.(200); continue; }
    const choice = await page.$('.call-options button:not(:disabled)');
    if (choice) { await choice.click(); continue; }
    const tile = await page.$('.hand button:not(:disabled)');
    if (tile) {
      await tile.click();
      const confirm = await page.$('.confirm-discard .primary');
      if (confirm) await confirm.click();
      continue;
    }
    await new Promise(done => setTimeout(done, 250));
  }
  const asked = await page.evaluate(() => window.asked);
  await writeFile(out, JSON.stringify({ decisions: asked, errors }, null, 1), 'utf8');
  console.log(`recorded ${asked.length} trained decisions to ${out}` +
    (errors.length ? `, with ${errors.length} page errors` : ''));
  if (!asked.length) throw new Error('the trained seats were never asked anything');
} finally {
  await browser.close();
  server.close();
}
