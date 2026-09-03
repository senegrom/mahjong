// Plays a hand to its end and opens the review, which is the only way to
// find out whether the numbers behind it survive the trip into the page.
import puppeteer from 'puppeteer-core';

const URL = process.argv[2] ?? 'http://127.0.0.1:8732/?opponents=club';
const LIMIT = Number(process.argv[3] ?? 180) * 1000;

const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: 'new',
  args: ['--disable-gpu', '--no-sandbox'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 900 });
  const problems = [];
  const missing = [];
  page.on('pageerror', (error) => problems.push(String(error)));
  // Only a real answer counts: a request cut short when the page closes is
  // not a missing file.
  page.on('response', (response) => {
    if (response.status() >= 400) missing.push(response.url());
  });
  page.on('console', (message) => {
    // A missing trained model is not a fault: the page offers that tier
    // only when the file is there, and falls back to the heuristic bots.
    const text = message.text();
    if (message.type() !== 'error') return;
    if (text.includes('404') || text.includes('Failed to load resource')) return;
    problems.push(text);
  });
  await page.goto(URL, { waitUntil: 'networkidle2' });

  const state = () =>
    page.evaluate(() => ({
      myTurn: (document.querySelector('.prompt')?.textContent ?? '').includes('Your turn'),
      over: !!document.querySelector('[aria-label="how the hand ended"]'),
      calls: [...document.querySelectorAll('.controls button')].map((b) =>
        b.textContent.trim(),
      ),
    }));

  const began = Date.now();
  let discards = 0;
  while (Date.now() - began < LIMIT) {
    const now = await state();
    if (now.over) break;
    if (now.myTurn) {
      await page.keyboard.press('1');
      discards += 1;
      await new Promise((r) => setTimeout(r, 350));
    } else if (now.calls.some((label) => label === 'Pass')) {
      // Decline claims so the hand runs to its natural end.
      await page.evaluate(() => {
        const pass = [...document.querySelectorAll('.controls button')].find(
          (b) => b.textContent.trim() === 'Pass',
        );
        pass?.click();
      });
      await new Promise((r) => setTimeout(r, 250));
    } else {
      await new Promise((r) => setTimeout(r, 250));
    }
  }

  const ended = await state();
  if (!ended.over) throw new Error(`the hand did not end within ${LIMIT / 1000}s`);

  const opened = await page.evaluate(() => {
    const button = [...document.querySelectorAll('button')].find((b) =>
      b.textContent.includes('Look at my hand again'),
    );
    if (!button) return false;
    button.click();
    return true;
  });
  if (!opened) throw new Error('there was no button to open the review');
  await new Promise((r) => setTimeout(r, 600));

  const review = await page.evaluate(() => {
    const panel = document.querySelector('[aria-label="your decisions this hand"]');
    if (!panel) return null;
    return {
      summary: panel.querySelector('.summary, .clean, .empty')?.textContent.trim() ?? '',
      entries: panel.querySelectorAll('li').length,
      tables: panel.querySelectorAll('table.numbers').length,
      reasons: [...panel.querySelectorAll('.why')].map((p) => p.textContent.trim()),
      numbers: [...panel.querySelectorAll('table.numbers td')].map((td) =>
        td.textContent.trim(),
      ),
    };
  });
  if (!review) throw new Error('the review panel never appeared');

  console.log(`discards played: ${discards}`);
  console.log(`review says: ${review.summary}`);
  console.log(`entries: ${review.entries}, with ${review.tables} number tables`);
  if (review.reasons.length) console.log(`first reason: ${review.reasons[0]}`);
  if (review.numbers.length) {
    console.log(`first numbers: ${review.numbers.slice(0, 6).join(' | ')}`);
  }
  const unexpected = missing;
  console.log(`page errors: ${problems.length ? problems.join('; ') : 'none'}`);
  console.log(
    `missing files: ${missing.length ? missing.map((u) => u.split('/').pop()).join(', ') : 'none'}`,
  );

  if (!review.summary) throw new Error('the review said nothing');
  if (problems.length) throw new Error('the page reported errors');
  if (unexpected.length) throw new Error(`unexpected missing files: ${unexpected.join(', ')}`);
  console.log('review check passed');
} finally {
  await browser.close();
}
