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
const phone = args.includes('--phone');

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--disable-gpu', '--hide-scrollbars', '--no-sandbox'],
});

try {
  const page = await browser.newPage();
  await page.setViewport(
    phone
      ? { width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true }
      : { width: 1100, height: 800 },
  );

  const problems = [];
  page.on('console', (message) => {
    if (message.type() === 'error') problems.push(message.text());
  });
  page.on('pageerror', (error) => problems.push(String(error)));
  // A request the browser cuts short as the page closes is not a fault. The
  // trained model is fetched lazily in a worker, so it is usually the one
  // still in flight at the end, and reporting it teaches you to ignore this
  // list. A real answer of 400 or worse is caught below.
  page.on('requestfailed', (request) => {
    const reason = request.failure()?.errorText ?? '';
    if (reason.includes('ERR_ABORTED')) return;
    problems.push(`request failed: ${request.url()} (${reason})`);
  });
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
      calls: document.querySelectorAll('.controls button').length > 0,
      ended: document.querySelector('.screen h2')?.textContent?.trim() ?? null,
      yaku: [...document.querySelectorAll('.yaku li')].map((li) => li.textContent.trim()),
      safe: document.querySelector('.safe-note')?.textContent?.trim() ?? null,
      safeMarks: document.querySelectorAll('.hand .safe').length,
      standings: [...document.querySelectorAll('.standings tbody tr')].map((row) =>
        [...row.children].map((cell) => cell.textContent.trim()).join(' '),
      ),
      finished: document.querySelector('.standings h2')?.textContent?.trim() ?? null,
      called: [...document.querySelectorAll('.called')].map((n) => n.textContent.trim()),
    }));

  const wholeGame = args.includes('--whole-game');
  const calls = [];
  const toEnd = args.includes('--to-end') || wholeGame;
  const limit = wholeGame ? 6000 : toEnd ? 400 : 6;
  const deadline = Date.now() + seconds * 1000;
  let state = await read();
  let played = 0;
  while (Date.now() < deadline && played < limit) {
    state = await read();
    if (state.failure) break;
    if (state.finished) break;
    if (state.ended) {
      if (!wholeGame) break;
      // On to the next hand, so a whole game can be played out.
      await page.evaluate(() => {
        const next = [...document.querySelectorAll('button')].find((button) =>
          ['Next hand', 'Play again'].includes(button.textContent.trim()),
        );
        if (next && next.textContent.trim() === 'Next hand') next.click();
      });
      await new Promise((resolve) => setTimeout(resolve, 200));
      continue;
    }
    if (state.called?.length) calls.push(...state.called);
    if (state.myTurn) {
      const clicked = await page.evaluate(() => {
        const tile = document.querySelector('.hand button.tile:not([disabled])');
        if (!tile) return false;
        tile.click();
        return true;
      });
      if (clicked) played += 1;
    } else if (state.calls) {
      // Decline every claim, so the hand runs on rather than stalling.
      await page.evaluate(() => {
        const buttons = [...document.querySelectorAll('.controls button')];
        const pass = buttons.find((button) => button.textContent.trim() === 'Pass');
        if (pass) pass.click();
      });
    }
    await new Promise((resolve) => setTimeout(resolve, toEnd ? 120 : 700));
  }
  state = await read();
  console.log('discards played:', played);
  if (calls.length) console.log('calls announced:', [...new Set(calls)].join(', '));
  if (state.safe) console.log('safe hint:', state.safe, `(${state.safeMarks} marked)`);
  if (state.ended) console.log('hand ended:', state.ended);
  if (state.finished) {
    console.log('game over:', state.finished);
    for (const row of state.standings) console.log('  ' + row);
  }

  if (shot) {
    await mkdir(dirname(shot), { recursive: true });
    await page.screenshot({ path: shot, fullPage: phone });
  }

  // A page that scrolls sideways on a phone is a page nobody can play.
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  if (overflow > 0) console.log('horizontal overflow:', overflow + 'px');

  console.log('opponents:', state?.opponents);
  console.log('tiles in hand:', state?.hand);
  console.log('prompt:', state?.prompt);
  console.log('failure:', state?.failure ?? 'none');
  if (state?.yaku?.length) {
    console.log('yaku:');
    for (const line of state.yaku) console.log('  ' + line);
  }
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
