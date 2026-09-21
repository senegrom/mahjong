// Manual diagnostic: play a running site through a hand and inspect its review.
import puppeteer from 'puppeteer-core';

const URL = process.argv[2] ?? 'http://127.0.0.1:8732/?opponents=club';
const LIMIT = Number(process.argv[3] ?? 180) * 1000;
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_BIN ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true, args: ['--disable-gpu', '--no-sandbox'],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 900 });
  const problems = [], missing = [];
  page.on('pageerror', error => problems.push(String(error)));
  page.on('response', response => { if (response.status() >= 400) missing.push(response.url()); });
  page.on('console', message => {
    const text = message.text();
    if (message.type() !== 'error') return;
    // Actual HTTP failures are asserted separately, without any model exemptions.
    if (text.includes('404') || text.includes('Failed to load resource')) return;
    problems.push(text);
  });
  await page.goto(URL, { waitUntil: 'networkidle2' });
  const state = () => page.evaluate(() => ({
    myTurn: (document.querySelector('.prompt')?.textContent ?? '').includes('Your turn'),
    over: !!document.querySelector('[aria-label="how the hand ended"]'),
    calls: [...document.querySelectorAll('.controls button')].map(button => button.textContent.trim()),
  }));
  const began = Date.now();
  let discards = 0;
  while (Date.now() - began < LIMIT) {
    const now = await state();
    if (now.over) break;
    if (now.myTurn) {
      await page.focus('.hand'); await page.keyboard.press('1'); discards++;
      await new Promise(resolve => setTimeout(resolve, 350));
    } else if (now.calls.includes('Pass')) {
      await page.evaluate(() => [...document.querySelectorAll('.controls button')]
        .find(button => button.textContent.trim() === 'Pass')?.click());
      await new Promise(resolve => setTimeout(resolve, 250));
    } else await new Promise(resolve => setTimeout(resolve, 250));
  }
  if (!(await state()).over) throw new Error(`the hand did not end within ${LIMIT / 1000}s`);
  const opened = await page.evaluate(() => {
    const words = ['Review final hand', 'View table / my hand', 'Look at my hand again'];
    const button = [...document.querySelectorAll('button')].find(button => words.some(text => button.textContent.includes(text)));
    if (!button) return false;
    button.click(); return true;
  });
  if (!opened) throw new Error('there was no button to open the review');
  await new Promise(resolve => setTimeout(resolve, 600));
  const review = await page.evaluate(() => {
    const panel = document.querySelector('[aria-label="your decisions this hand"]');
    return panel && {
      summary: panel.querySelector('.summary, .clean, .empty')?.textContent.trim() ?? '',
      entries: panel.querySelectorAll('li').length, tables: panel.querySelectorAll('table.numbers').length,
      reasons: [...panel.querySelectorAll('.why')].map(el => el.textContent.trim()),
      numbers: [...panel.querySelectorAll('table.numbers td')].map(el => el.textContent.trim()),
    };
  });
  if (!review?.summary) throw new Error('the review said nothing or never appeared');
  console.log(JSON.stringify({ discards, review, problems, missing }, null, 2));
  if (problems.length) throw new Error('the page reported errors');
  if (missing.length) throw new Error(`unexpected missing files: ${missing.join(', ')}`);
  console.log('review check passed');
} finally { await browser.close(); }
