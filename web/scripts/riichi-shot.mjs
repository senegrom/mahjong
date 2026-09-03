// Plays until somebody declares riichi, then looks at the sideways tile.
import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: 'new', args: ['--disable-gpu', '--no-sandbox'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1200, height: 900, deviceScaleFactor: 2 });
await page.goto(process.argv[2] ?? 'http://127.0.0.1:8732/?opponents=club', { waitUntil: 'networkidle2' });

let found = null;
for (let step = 0; step < 200 && !found; step += 1) {
  found = await page.evaluate(() => {
    const tile = document.querySelector('.pool .tile.rotated, .tile.rotated');
    if (!tile) return null;
    const box = tile.getBoundingClientRect();
    const pool = tile.closest('.pool') ?? tile.parentElement;
    const poolBox = pool.getBoundingClientRect();
    return {
      w: Math.round(box.width), h: Math.round(box.height),
      insidePool: box.left >= poolBox.left - 0.5 && box.right <= poolBox.right + 0.5,
      overflowsPage: box.right > document.documentElement.clientWidth + 0.5,
    };
  });
  if (found) break;
  await page.evaluate(() => {
    const pass = [...document.querySelectorAll('.controls button')].find(b => b.textContent.trim() === 'Pass');
    pass?.click();
  });
  await page.keyboard.press('1');
  await new Promise(r => setTimeout(r, 260));
}

if (!found) {
  console.log('no riichi was declared in this game, nothing to look at');
} else {
  console.log(JSON.stringify(found));
  if (found.overflowsPage) throw new Error('a sideways tile pushes the page wide');
  console.log(found.insidePool ? 'the sideways tile sits inside its row' : 'it hangs out of its row');
}
await browser.close();
