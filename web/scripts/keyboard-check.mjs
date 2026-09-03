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

  // Three ways in, all of which a person might use: the number keys, the
  // arrow keys with Enter, and zero for the tile just drawn. The numbers
  // stop at nine and a hand is fourteen tiles, so the arrows are the only
  // way to reach the rest.
  const ways = ['1', 'arrows', '0'];
  let played = 0;
  let marked = 0;
  for (let attempt = 0; attempt < 90 && played < ways.length; attempt += 1) {
    const before = await read();
    if (!before.myTurn) {
      await new Promise((resolve) => setTimeout(resolve, 400));
      continue;
    }
    const way = ways[played % ways.length];
    if (way === 'arrows') {
      // Walk to the far end of the hand, which is past where the number
      // keys reach, and throw what is under the marker.
      for (let step = 0; step < 13; step += 1) await page.keyboard.press('ArrowRight');
      await new Promise((resolve) => setTimeout(resolve, 150));
      const lit = await page.evaluate(
        () => document.querySelectorAll('.hand .tile.selected').length,
      );
      if (lit !== 1) throw new Error(`the arrow keys marked ${lit} tiles, not one`);
      marked += 1;
      await page.keyboard.press('Enter');
    } else {
      await page.keyboard.press(way);
    }
    await new Promise((resolve) => setTimeout(resolve, 600));
    const after = await read();
    if (after.discards > before.discards) played += 1;
    else throw new Error(`${way} did not discard anything`);
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

  console.log(`discarded by keyboard: ${played} (${ways.join(', ')})`);
  console.log(`turns where the arrow keys marked exactly one tile: ${marked}`);
  console.log(`every tile named: ${labels.tiles}`);
  console.log(`labelled regions: ${labels.regions}, live regions: ${labels.liveRegions}`);
  console.log(`reachable controls: ${labels.focusable}`);
  if (played < ways.length) throw new Error('keyboard play did not work');
  if (!labels.tiles) throw new Error('a tile has no name for a screen reader');
  console.log('keyboard and labelling check passed');
} finally {
  await browser.close();
}
