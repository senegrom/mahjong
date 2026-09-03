// Plays a whole hanchan in the browser and reads the final standings.
//
// Every other check stops at the end of one hand. A game is East round then
// South round, the deal moving on or repeating, counters and riichi bets
// carried between hands, and the winner bonus at the end. None of that is
// exercised by one hand, and it is what a person actually sits down to play.
import puppeteer from 'puppeteer-core';

const URL = process.argv[2] ?? 'http://127.0.0.1:8732/?opponents=club';
const LIMIT = Number(process.argv[3] ?? 600) * 1000;

const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: 'new',
  args: ['--disable-gpu', '--hide-scrollbars', '--no-sandbox'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 850 });
  const problems = [];
  page.on('pageerror', (error) => problems.push(String(error)));
  page.on('console', (message) => {
    const text = message.text();
    if (message.type() !== 'error') return;
    if (text.includes('404') || text.includes('Failed to load resource')) return;
    problems.push(text);
  });
  await page.goto(URL, { waitUntil: 'networkidle2' });

  const state = () =>
    page.evaluate(() => {
      const buttons = [...document.querySelectorAll('.controls button')].map((b) =>
        b.textContent.trim(),
      );
      return {
        standings: !!document.querySelector('[aria-label="final standings"]'),
        handOver: !!document.querySelector('[aria-label="how the hand ended"]'),
        // Whether a tile can be thrown, rather than whether a line of text
        // says so: the two came apart once already.
        myTurn: [...document.querySelectorAll('.hand button.tile')].some((b) => !b.disabled),
        round: document.querySelector('.round')?.textContent.trim() ?? '',
        buttons,
        wide:
          document.documentElement.scrollWidth - document.documentElement.clientWidth,
      };
    });

  const began = Date.now();
  let hands = 0;
  let discards = 0;
  let widest = 0;
  const rounds = new Set();
  const spent = { handOver: 0, myTurn: 0, calls: 0, waiting: 0 };

  while (Date.now() - began < LIMIT) {
    const now = await state();
    widest = Math.max(widest, now.wide);
    if (now.round) rounds.add(now.round);
    if (now.standings) break;

    if (now.handOver) {
      spent.handOver += 1;
      hands += 1;
      const moved = await page.evaluate(() => {
        const next = [...document.querySelectorAll('button')].find(
          (b) => b.textContent.includes('Next hand') || b.textContent.includes('Play again'),
        );
        if (!next) return false;
        next.click();
        return true;
      });
      if (!moved) throw new Error('the hand ended with no way to carry on');
      await new Promise((r) => setTimeout(r, 400));
      continue;
    }

    if (now.myTurn) {
      spent.myTurn += 1;
      // Take a win whenever it is offered, otherwise throw the first tile
      // that can actually go. Pressing the first key regardless spins on a
      // hand where that tile is barred, which is what a swap-call does.
      const acted = await page.evaluate(() => {
        const win = [...document.querySelectorAll('.controls button')].find(
          (b) => b.textContent.trim() === 'Win',
        );
        if (win) {
          win.click();
          return 'won';
        }
        const tile = [...document.querySelectorAll('.hand button.tile')].find(
          (button) => !button.disabled,
        );
        if (!tile) return null;
        tile.click();
        return 'discarded';
      });
      if (acted === 'discarded') discards += 1;
      if (!acted) {
        // Nothing can be thrown and no win is offered: let the page settle
        // rather than hammering it.
        await new Promise((r) => setTimeout(r, 200));
      }
    } else if (now.buttons.length) {
      spent.calls += 1;
      await page.evaluate(() => {
        const buttons = [...document.querySelectorAll('.controls button')];
        const win = buttons.find((b) => b.textContent.trim() === 'Win');
        const pass = buttons.find((b) => b.textContent.trim() === 'Pass');
        (win ?? pass)?.click();
      });
    }
    if (!now.handOver && !now.myTurn && !now.buttons.length) spent.waiting += 1;
    await new Promise((r) => setTimeout(r, 90));
  }

  const end = await state();
  const standings = end.standings
    ? await page.evaluate(() =>
        [...document.querySelectorAll('[aria-label="final standings"] tbody tr')].map((row) =>
          [...row.querySelectorAll('td')].map((cell) => cell.textContent.trim()),
        ),
      )
    : null;

  console.log(`hands played: ${hands}`);
  console.log(`discards: ${discards}`);
  console.log(`rounds seen: ${[...rounds].join(', ') || '(none read)'}`);
  console.log(`widest horizontal overflow: ${widest}px`);
  console.log(`page errors: ${problems.length ? problems.join('; ') : 'none'}`);
  // What the loop spent its turns on, which is how a stall gets found.
  console.log(`loops: ${JSON.stringify(spent)}`);

  if (!standings) throw new Error(`the game did not finish within ${LIMIT / 1000}s`);
  console.log('final standings:');
  for (const row of standings) console.log(`  ${row.join('  ')}`);

  if (standings.length !== 4) throw new Error('four players finish a game');
  // The winner bonus is zero-sum, so the results must cancel.
  const totals = standings.map((row) => Number(row[4].replace(/[+,]/g, '')));
  const sum = totals.reduce((a, b) => a + b, 0);
  console.log(`results sum to ${sum}`);
  if (Math.abs(sum) > 1) throw new Error(`the results do not cancel: ${sum}`);
  if (widest > 0) throw new Error(`the page scrolled sideways by ${widest}px`);
  if (problems.length) throw new Error('the page reported errors');
  console.log('game check passed');
} finally {
  await browser.close();
}
