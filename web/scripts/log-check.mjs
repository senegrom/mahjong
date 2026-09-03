// Plays a hand out, presses the button a person would press, and reads what
// lands on disk. The log is only useful if it leaves the page.
import puppeteer from 'puppeteer-core';
import { mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const URL = process.argv[2] ?? 'http://127.0.0.1:8732/?opponents=club';
const LIMIT = Number(process.argv[3] ?? 180) * 1000;
const DOWNLOADS = mkdtempSync(join(process.env.CLAUDE_CODE_TMPDIR ?? tmpdir(), 'riichi-log-'));

const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: 'new',
  args: ['--disable-gpu', '--no-sandbox'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 900 });
  const session = await page.createCDPSession();
  await session.send('Page.setDownloadBehavior', {
    behavior: 'allow',
    downloadPath: DOWNLOADS,
  });
  await page.goto(URL, { waitUntil: 'networkidle2' });

  const began = Date.now();
  let over = false;
  while (Date.now() - began < LIMIT && !over) {
    over = await page.evaluate(
      () => !!document.querySelector('[aria-label="how the hand ended"]'),
    );
    if (over) break;
    const mine = await page.evaluate(
      () => (document.querySelector('.prompt')?.textContent ?? '').includes('Your turn'),
    );
    if (mine) {
      await page.keyboard.press('1');
    } else {
      await page.evaluate(() => {
        const pass = [...document.querySelectorAll('.controls button')].find(
          (b) => b.textContent.trim() === 'Pass',
        );
        pass?.click();
      });
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  if (!over) throw new Error('the hand did not end in time');

  const pressed = await page.evaluate(() => {
    const button = [...document.querySelectorAll('button')].find((b) =>
      b.textContent.includes('Save this hand'),
    );
    if (!button) return false;
    button.click();
    return true;
  });
  if (!pressed) throw new Error('there is no way to save the hand');

  let saved = [];
  for (let wait = 0; wait < 40 && saved.length === 0; wait += 1) {
    await new Promise((r) => setTimeout(r, 150));
    saved = readdirSync(DOWNLOADS).filter((name) => name.endsWith('.jsonl'));
  }
  if (saved.length === 0) throw new Error('pressing the button saved nothing');

  const text = readFileSync(join(DOWNLOADS, saved[0]), 'utf-8');
  const lines = text.split('\n').filter((line) => line.trim());
  const kinds = {};
  for (const line of lines) {
    const event = JSON.parse(line);
    kinds[event.type] = (kinds[event.type] ?? 0) + 1;
  }

  console.log(`saved as: ${saved[0]}`);
  console.log(`events: ${lines.length}`);
  console.log(
    Object.entries(kinds)
      .sort((a, b) => b[1] - a[1])
      .map(([kind, count]) => `${kind} ${count}`)
      .join(', '),
  );
  if (kinds.start_kyoku !== 1) throw new Error('the log does not open the hand once');
  if (kinds.end_kyoku !== 1) throw new Error('the log does not close the hand once');
  if (!kinds.dahai) throw new Error('a played hand has discards in it');
  console.log('log check passed');
} finally {
  await browser.close();
  rmSync(DOWNLOADS, { recursive: true, force: true });
}
