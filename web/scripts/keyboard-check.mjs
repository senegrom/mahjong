// Plays by keyboard alone, which is the path a person who cannot use a
// mouse depends on, and which nothing has exercised until now.
import puppeteer from 'puppeteer-core';

const URL = process.argv[2] ?? 'http://127.0.0.1:8732/?opponents=club';

const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: 'new',
  args: ['--disable-gpu', '--no-sandbox'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 800 });
  await page.goto(URL, { waitUntil: 'networkidle2' });

  const read = () =>
    page.evaluate(() => ({
      hand: [...document.querySelectorAll('.hand .tile')].map((t) =>
        t.getAttribute('aria-label'),
      ),
      myTurn: (document.querySelector('.prompt')?.textContent ?? '').includes('Your turn'),
      discards: document.querySelectorAll('.mine .pool .tile').length,
    }));

  let played = 0;
  for (let attempt = 0; attempt < 60 && played < 4; attempt += 1) {
    const before = await read();
    if (before.myTurn) {
      await page.keyboard.press('1');
      await new Promise((resolve) => setTimeout(resolve, 600));
      const after = await read();
      if (after.discards > before.discards) played += 1;
      else throw new Error('pressing 1 did not discard the first tile');
    } else {
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
  }

  // Everything a screen reader needs: tiles named, regions labelled.
  const labels = await page.evaluate(() => ({
    tiles: [...document.querySelectorAll('.hand .tile')].every((t) =>
      (t.getAttribute('aria-label') ?? '').length > 3,
    ),
    regions: [...document.querySelectorAll('[aria-label]')].length,
    liveRegions: document.querySelectorAll('[aria-live]').length,
    focusable: document.querySelectorAll('button:not([disabled]), select, input').length,
  }));

  console.log(`discarded by keyboard: ${played}`);
  console.log(`every tile named: ${labels.tiles}`);
  console.log(`labelled regions: ${labels.regions}, live regions: ${labels.liveRegions}`);
  console.log(`reachable controls: ${labels.focusable}`);
  if (played < 4) throw new Error('keyboard play did not work');
  if (!labels.tiles) throw new Error('a tile has no name for a screen reader');
  console.log('keyboard and labelling check passed');
} finally {
  await browser.close();
}
