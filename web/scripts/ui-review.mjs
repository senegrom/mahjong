/**
 * Photographs the page's main states for a review: the guide open, the
 * table mid-hand with the keyboard marker up, a moment with choices to
 * make, the score screen, its review, and the final standings. Plays at
 * real speed in a real browser as play-check does, throwing the drawn tile
 * and passing on claims, and reports console errors, failed requests and
 * faces that did not load.
 *
 * Usage: node scripts/ui-review.mjs [url] [--out dir] [--seconds 240]
 */
import { mkdir } from 'node:fs/promises';
import puppeteer from 'puppeteer-core';

const CHROME = process.env.CHROME_BIN ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const args = process.argv.slice(2);
const url = args.find((arg) => !arg.startsWith('--')) ?? 'http://127.0.0.1:8732/';
const out = args.includes('--out') ? args[args.indexOf('--out') + 1] : 'review-shots';
const seconds = args.includes('--seconds') ? Number(args[args.indexOf('--seconds') + 1]) : 240;
await mkdir(out, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--disable-gpu', '--no-sandbox'],
});
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 900 });
  const problems = [];
  page.on('console', (message) => {
    if (message.type() === 'error') problems.push(message.text());
  });
  page.on('pageerror', (error) => problems.push(String(error)));
  page.on('requestfailed', (request) => {
    const reason = request.failure()?.errorText ?? '';
    if (!reason.includes('ERR_ABORTED')) problems.push(`request failed: ${request.url()} (${reason})`);
  });
  page.on('response', (response) => {
    if (response.status() >= 400) problems.push(`${response.status()} ${response.url()}`);
  });
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  // The page preloads every tile face, so the network is busy for a while
  // on a slow connection; the table appearing is what matters.
  await page.waitForSelector('.hand, .failure', { timeout: 120000 });
  await pause(500);

  // The guide, open.
  await page.click('.guide summary');
  await pause(300);
  await page.screenshot({ path: `${out}/guide.png`, clip: { x: 0, y: 0, width: 1100, height: 520 } });
  await page.click('.guide summary');
  await pause(200);

  const read = () =>
    page.evaluate(() => ({
      prompt: document.querySelector('.prompt')?.textContent?.trim() ?? '',
      buttons: [...document.querySelectorAll('.controls button')].map((b) => b.textContent.trim()),
      ended: !!document.querySelector('.screen h2'),
      standings: !!document.querySelector('[aria-label="final standings"]'),
      called: [...document.querySelectorAll('.called')].map((s) => s.textContent.trim()),
    }));

  const taken = new Set();
  const shot = async (name, clip) => {
    if (taken.has(name)) return;
    taken.add(name);
    await page.screenshot({ path: `${out}/${name}.png`, ...(clip ? { clip } : {}) });
    console.log('shot', name);
  };

  const started = Date.now();
  let turns = 0;
  while (Date.now() - started < seconds * 1000) {
    const now = await read();
    if (now.standings) {
      await shot('standings');
      break;
    }
    if (now.ended) {
      await shot('score');
      const review = now.buttons.find((b) => /review/i.test(b));
      if (review && !taken.has('review')) {
        await page.evaluate((label) => {
          [...document.querySelectorAll('.controls button')].find((b) => b.textContent.trim() === label)?.click();
        }, review);
        await pause(600);
        await shot('review');
      }
      const next = now.buttons.find((b) => /next|new game|again/i.test(b));
      if (next) {
        await page.evaluate((label) => {
          [...document.querySelectorAll('.controls button')].find((b) => b.textContent.trim() === label)?.click();
        }, next);
        await pause(500);
      }
      continue;
    }
    if (now.called.length && !taken.has('called')) {
      await shot('called');
    }
    if (now.buttons.length) {
      await shot('choices', { x: 0, y: 440, width: 1100, height: 460 });
      const pass = now.buttons.find((b) => /^pass$/i.test(b));
      if (pass) {
        await page.evaluate(() => {
          [...document.querySelectorAll('.controls button')].find((b) => /^pass$/i.test(b.textContent.trim()))?.click();
        });
        await pause(300);
        continue;
      }
    }
    if (now.prompt.includes('Your turn')) {
      if (turns === 4 && !taken.has('table')) {
        await page.keyboard.press('ArrowLeft');
        await page.keyboard.press('ArrowLeft');
        await pause(250);
        await shot('table');
      }
      await page.keyboard.press('0');
      turns += 1;
      await pause(250);
      continue;
    }
    await pause(150);
  }

  const unloaded = await page.evaluate(() =>
    [...document.images].filter((image) => !image.complete || image.naturalWidth === 0).map((image) => image.getAttribute('src')),
  );
  const fetches = await page.evaluate(() =>
    performance
      .getEntriesByType('resource')
      .filter((entry) => entry.name.includes('/tiles/'))
      .map((entry) => ({ file: entry.name.split('/').pop(), start: Math.round(entry.startTime), ms: Math.round(entry.duration) })),
  );
  fetches.sort((a, b) => b.ms - a.ms);
  console.log('tile fetches:', fetches.length, 'slowest:', JSON.stringify(fetches.slice(0, 5)));
  const late = fetches.filter((f) => f.start + f.ms > 30000).length;
  console.log('faces that arrived after 30 s:', late);
  console.log('turns played:', turns);
  console.log('images not loaded:', JSON.stringify(unloaded));
  console.log('problems:', JSON.stringify(problems));
  console.log('ui review done');
} finally {
  await browser.close();
}
