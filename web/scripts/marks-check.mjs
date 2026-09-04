/**
 * Photographs the marks on the player's hand: the keyboard marker brought
 * up with an arrow key, the drawn tile's ring, dora and safe tiles, and the
 * stripes where they coincide. The page is played at real speed in a real
 * browser, as play-check does, and two pictures are taken: the marker on
 * the tile just drawn, and after two presses to the left.
 *
 * Usage: node scripts/marks-check.mjs [url] [--shot path.png]
 */
import puppeteer from 'puppeteer-core';

const CHROME = process.env.CHROME_BIN ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const args = process.argv.slice(2);
const url = args.find((arg) => !arg.startsWith('--')) ?? 'http://127.0.0.1:8732/';
const shot = args.includes('--shot') ? args[args.indexOf('--shot') + 1] : 'marks.png';

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--disable-gpu', '--no-sandbox'],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 820 });
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
  await page.goto(url, { waitUntil: 'networkidle2', timeout: 60000 });

  const myTurn = () =>
    page.evaluate(() => (document.querySelector('.prompt')?.textContent ?? '').includes('Your turn'));
  const waitForTurn = async () => {
    for (let i = 0; i < 600; i++) {
      if (await myTurn()) return true;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    return false;
  };

  // Throw a few tiles first, so there are discards and safe tiles about.
  for (let round = 0; round < 3; round++) {
    if (!(await waitForTurn())) break;
    await page.keyboard.press('0');
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  if (!(await waitForTurn())) throw new Error('never got a turn');

  const rings = () =>
    page.evaluate(() =>
      [...document.querySelectorAll('.hand button.tile')].map((button) => ({
        tile: button.getAttribute('aria-label'),
        ring: getComputedStyle(button).getPropertyValue('--ring').trim(),
        selected: button.classList.contains('selected'),
      })),
    );

  console.log('before any arrow:', JSON.stringify((await rings()).filter((r) => r.selected)));
  await page.keyboard.press('ArrowLeft');
  await new Promise((resolve) => setTimeout(resolve, 200));
  const first = await rings();
  console.log('after one arrow:', JSON.stringify(first.filter((r) => r.selected)));
  await page.screenshot({ path: shot.replace(/\.png$/, '-drawn.png'), clip: { x: 0, y: 470, width: 1100, height: 240 } });

  await page.keyboard.press('ArrowLeft');
  await page.keyboard.press('ArrowLeft');
  await new Promise((resolve) => setTimeout(resolve, 200));
  const moved = await rings();
  console.log('after three arrows:', JSON.stringify(moved.filter((r) => r.selected)));
  console.log('rings on the hand:', JSON.stringify(moved.filter((r) => r.ring && r.ring !== 'none').map((r) => `${r.tile}: ${r.ring.slice(0, 60)}`)));
  await page.screenshot({ path: shot.replace(/\.png$/, '-moved.png'), clip: { x: 0, y: 470, width: 1100, height: 240 } });
  await page.screenshot({ path: shot });
  const unloaded = await page.evaluate(() =>
    [...document.images]
      .filter((image) => !image.complete || image.naturalWidth === 0)
      .map((image) => image.getAttribute('src')),
  );
  console.log('images not loaded:', JSON.stringify(unloaded));
  const fetches = await page.evaluate(() =>
    performance
      .getEntriesByType('resource')
      .filter((entry) => entry.name.includes('/tiles/'))
      .map((entry) => ({ file: entry.name.split('/').pop(), ms: Math.round(entry.duration) }))
      .sort((a, b) => b.ms - a.ms)
      .slice(0, 6),
  );
  console.log('slowest tile fetches:', JSON.stringify(fetches));
  console.log('problems:', JSON.stringify(problems));
  console.log('marks check done');
} finally {
  await browser.close();
}
