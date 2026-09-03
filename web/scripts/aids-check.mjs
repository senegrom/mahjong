// The learning aids are only useful if the numbers on them are right, so
// this reads them out of a real game and checks them against each other.
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
  await page.goto(URL, { waitUntil: 'networkidle2' });

  const read = () =>
    page.evaluate(() => ({
      myTurn: (document.querySelector('.prompt')?.textContent ?? '').includes('Your turn'),
      over: !!document.querySelector('[aria-label="how the hand ended"]'),
      waits: [...document.querySelectorAll('.wait')].map((node) => ({
        tile: node.querySelector('.tile')?.getAttribute('aria-label') ?? '',
        left: Number(node.querySelector('.left')?.textContent.trim() ?? -1),
      })),
      doraNote: document.querySelector('.dora-note')?.textContent.trim() ?? '',
      doraTiles: document.querySelectorAll('.hand .tile.dora').length,
      handSize: document.querySelectorAll('.hand .tile').length,
    }));

  const began = Date.now();
  let sawWaits = 0;
  let sawDora = 0;
  const problems = [];

  while (Date.now() - began < LIMIT) {
    const now = await read();
    if (now.over) break;

    for (const wait of now.waits) {
      sawWaits += 1;
      if (wait.left < 0 || wait.left > 4) {
        problems.push(`a wait says ${wait.left} left, which is not possible`);
      }
      if (!wait.tile) problems.push('a wait has no tile');
    }

    if (now.doraNote) {
      sawDora += 1;
      const said = Number(now.doraNote.split(' ')[0]);
      // The note counts dora in hand and in called sets; the marks are on
      // the concealed tiles alone, so the note is never the smaller number.
      if (now.doraTiles > said) {
        problems.push(`${now.doraTiles} tiles marked dora but the note says ${said}`);
      }
    }

    if (now.myTurn) {
      await page.keyboard.press('1');
      await new Promise((r) => setTimeout(r, 320));
    } else {
      await page.evaluate(() => {
        const pass = [...document.querySelectorAll('.controls button')].find(
          (b) => b.textContent.trim() === 'Pass',
        );
        pass?.click();
      });
      await new Promise((r) => setTimeout(r, 260));
    }
  }

  console.log(`waits seen with a count: ${sawWaits}`);
  console.log(`turns showing a dora count: ${sawDora}`);
  console.log(`problems: ${problems.length ? [...new Set(problems)].join('; ') : 'none'}`);
  if (problems.length) throw new Error('an aid showed something impossible');
  // This plays by throwing the leftmost tile, so it rarely reaches a wait.
  // Saying so beats passing on having watched nothing: the wait counter is
  // checked properly in the engine's own tests.
  if (sawWaits === 0) console.log('note: never reached a wait, so those counts went unseen');
  if (sawDora === 0) throw new Error('no dora was ever shown, which cannot be right');
  console.log('aids check passed');
} finally {
  await browser.close();
}
