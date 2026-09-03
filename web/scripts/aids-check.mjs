// The learning aids are only useful if their numbers are right, so this
// works out the answer independently and compares.
//
// The first version waited for a dora to turn up in the hand and checked
// the count looked sensible, which passed when nothing appeared and failed
// when a game happened not to deal one. This derives the dora from the
// indicator on the table, the way the rules do, and checks the hand against
// it every turn, including the turns where the answer is none.
import puppeteer from 'puppeteer-core';

const URL = process.argv[2] ?? 'http://127.0.0.1:8732/?opponents=club';
const LIMIT = Number(process.argv[3] ?? 180) * 1000;

const SUITS = { characters: 'm', circles: 'p', bamboo: 's' };
const HONOURS = [
  'east wind',
  'south wind',
  'west wind',
  'north wind',
  'white dragon',
  'green dragon',
  'red dragon',
];

/** A tile from the words a screen reader is given, as "5 circles" or "east wind". */
function fromWords(words) {
  const honour = HONOURS.indexOf(words);
  if (honour >= 0) return `${honour + 1}z`;
  const [rank, suit] = words.split(' ');
  const letter = SUITS[suit];
  if (!letter || !Number(rank)) return null;
  return `${rank}${letter}`;
}

/**
 * What an indicator points at. Nine points back to one in the same suit,
 * the winds run east to south to west to north to east, and the dragons
 * white to green to red to white.
 */
function doraOf(tile) {
  if (!tile) return null;
  const rank = Number(tile[0]);
  const suit = tile[1];
  if (suit !== 'z') return `${rank === 9 ? 1 : rank + 1}${suit}`;
  if (rank <= 4) return `${rank === 4 ? 1 : rank + 1}z`;
  return `${rank === 7 ? 5 : rank + 1}z`;
}

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
      myTurn: [...document.querySelectorAll('.hand button.tile')].some((b) => !b.disabled),
      over: !!document.querySelector('[aria-label="how the hand ended"]'),
      indicators: [...document.querySelectorAll('.dora .tile')].map((t) =>
        t.getAttribute('aria-label'),
      ),
      hand: [...document.querySelectorAll('.hand .tile')].map((t) => ({
        words: t.getAttribute('aria-label') ?? '',
        marked: t.classList.contains('dora'),
      })),
      waits: [...document.querySelectorAll('.wait')].map((node) => ({
        tile: node.querySelector('.tile')?.getAttribute('aria-label') ?? '',
        left: Number(node.querySelector('.remaining')?.textContent.trim() ?? -1),
      })),
      note: document.querySelector('.dora-note')?.textContent.trim() ?? '',
    }));

  const began = Date.now();
  let turnsChecked = 0;
  let turnsWithDora = 0;
  let waitsSeen = 0;
  const problems = [];

  while (Date.now() - began < LIMIT) {
    const now = await read();
    if (now.over) break;

    // What the rules say the dora are, worked out here rather than asked of
    // the page. A tile several indicators point at is dora several times.
    const dora = now.indicators.map((words) => doraOf(fromWords(words))).filter(Boolean);
    if (dora.length === now.indicators.length && now.hand.length) {
      let expected = 0;
      for (const tile of now.hand) {
        const named = fromWords(tile.words.replace(', dora', ''));
        const times = dora.filter((each) => each === named).length;
        expected += times;
        if (times > 0 !== tile.marked) {
          problems.push(
            `${tile.words} is ${times > 0 ? '' : 'not '}dora but ${tile.marked ? 'is' : 'is not'} marked`,
          );
        }
      }
      const said = now.note ? Number(now.note.split(' ')[0]) : 0;
      // The note counts called sets too, so it is never the smaller number.
      if (said < expected) {
        problems.push(`the note says ${said} dora where the hand shows ${expected}`);
      }
      turnsChecked += 1;
      if (expected > 0) turnsWithDora += 1;
    }

    for (const wait of now.waits) {
      waitsSeen += 1;
      if (wait.left < 0 || wait.left > 4) {
        problems.push(`a wait says ${wait.left} left, which is not possible`);
      }
    }

    if (now.myTurn) {
      await page.evaluate(() => {
        const tile = [...document.querySelectorAll('.hand button.tile')].find(
          (button) => !button.disabled,
        );
        tile?.click();
      });
    } else {
      await page.evaluate(() => {
        const pass = [...document.querySelectorAll('.controls button')].find(
          (b) => b.textContent.trim() === 'Pass',
        );
        pass?.click();
      });
    }
    await new Promise((r) => setTimeout(r, 260));
  }

  console.log(`turns checked against the indicator: ${turnsChecked}`);
  console.log(`of those, turns holding a dora: ${turnsWithDora}`);
  console.log(`waits seen with a count: ${waitsSeen}`);
  console.log(`problems: ${problems.length ? [...new Set(problems)].slice(0, 5).join('; ') : 'none'}`);

  if (problems.length) throw new Error('an aid disagreed with the rules');
  // Holding no dora is a real answer, so this asks for turns examined
  // rather than for a dora to have turned up.
  if (turnsChecked < 8) throw new Error(`only ${turnsChecked} turns were examined`);
  if (waitsSeen === 0) console.log('note: never reached a wait, so those counts went unseen');
  console.log('aids check passed');
} finally {
  await browser.close();
}
