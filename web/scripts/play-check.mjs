/**
 * Plays the published game in a real browser and reports what happened.
 *
 * A headless screenshot with a virtual clock is not enough to test this: the
 * clock races ahead of the work, so a network that is still loading looks
 * like one that has hung. This drives an actual browser at real speed,
 * watches the console, and can take a picture at the end.
 *
 * Usage: node scripts/play-check.mjs [url] [--shot path.png] [--seconds 60]
 */
import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import puppeteer from 'puppeteer-core';

const CHROME =
  process.env.CHROME_BIN ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe';

const args = process.argv.slice(2);
const url = args.find((arg) => !arg.startsWith('--')) ?? 'http://127.0.0.1:8732/';
const shot = args.includes('--shot') ? args[args.indexOf('--shot') + 1] : null;
const seconds = args.includes('--seconds') ? Number(args[args.indexOf('--seconds') + 1]) : 45;

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--disable-gpu', '--hide-scrollbars', '--no-sandbox'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 800 });

  const problems = [];
  page.on('console', (message) => {
    if (message.type() === 'error') problems.push(message.text());
  });
  page.on('pageerror', (error) => problems.push(String(error)));
  page.on('requestfailed', (request) =>
    problems.push(`request failed: ${request.url()}`),
  );
  page.on('response', (response) => {
    if (response.status() >= 400) problems.push(`${response.status()} ${response.url()}`);
  });

  await page.goto(url, { waitUntil: 'networkidle2', timeout: 60000 });

  // Play a few tiles, so the opponents actually have to answer.
  const read = () =>
    page.evaluate(() => ({
      failure: document.querySelector('.failure')?.textContent?.trim() ?? null,
      prompt: document.querySelector('.prompt')?.textContent?.trim() ?? null,
      hand: document.querySelectorAll('.hand .tile').length,
      log: [...document.querySelectorAll('.log p')].map((p) => p.textContent.trim()),
      opponents: document.querySelector('select')?.value ?? null,
      myTurn: (document.querySelector('.prompt')?.textContent ?? '').includes('Your turn'),
    }));

  const deadline = Date.now() + seconds * 1000;
  let state = await read();
  let played = 0;
  while (Date.now() < deadline && played < 6) {
    state = await read();
    if (state.failure) break;
    if (state.myTurn) {
      const clicked = await page.evaluate(() => {
        const tile = document.querySelector('.hand button.tile:not([disabled])');
        if (!tile) return false;
        tile.click();
        return true;
      });
      if (clicked) played += 1;
    }
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
  state = await read();
  console.log('discards played:', played);

  if (shot) {
    await mkdir(dirname(shot), { recursive: true });
    await page.screenshot({ path: shot });
  }

  console.log('opponents:', state?.opponents);
  console.log('tiles in hand:', state?.hand);
  console.log('prompt:', state?.prompt);
  console.log('failure:', state?.failure ?? 'none');
  console.log('log:');
  for (const line of (state?.log ?? []).slice(0, 6)) console.log('  ' + line);
  if (problems.length) {
    console.log('console errors:');
    for (const problem of problems.slice(0, 5)) console.log('  ' + problem);
  }
  process.exitCode = state?.failure ? 1 : 0;
} finally {
  await browser.close();
}
