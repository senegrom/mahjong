/** The whole-game review on the production build, with the trained worker.
 *
 * A finished match is restored from its save and reviewed hand by hand by the
 * shipped network. Every batch the page sends is captured with the worker's
 * answer, and the flags, their order and the agreement figure are worked out
 * again here from those answers alone. A flagged decision is then replayed
 * from the deal by keyboard, chosen again, and Trained AI's answer read back,
 * with keyboard focus checked after every button the replay offers; the saved
 * match must be exactly as it was, and the game must enter the trend once,
 * under the version of Trained AI that judged it. A reviewed hand is replayed
 * whole, by keyboard and pointer, and at every step every player's tiles must
 * be the ones the engine held, face up, each hand a list named by its seat,
 * with the board as it stood and nothing offering to hide them, fitting the
 * screen at desktop, tablet and phone sizes, an opponent's tiles on one row,
 * or on two where one would draw them under 18 pixels wide; the hand's list
 * of decisions and the
 * review's own list are filtered to the flagged decisions, those where
 * Trained AI chose differently or every one, saying how many each holds, and
 * a decision that was not flagged opens without a retry. A list the player
 * opens or folds stays as they left it: a hand's list of decisions, folded on
 * a phone and in a replay opened to choose again, as its steps are pressed by
 * touch and by keyboard while later hands are reviewed, focus staying on the
 * step's button; a slow device's list of hands as the review goes on; a
 * hand's events across a step with none; and the trend's games across a
 * replay. A flagged answer to
 * another's discard, from a game whose player calls everything, is chosen
 * again from the moves its prompt points to, by keyboard. While the review goes on, a flagged
 * decision that later hands rank out of the leading five keeps keyboard focus,
 * in the list, while the player is in another window, and back from its
 * replay, and the list says where it went in words that stay true as the list
 * grows, from a status line that is in the page before it speaks. On a phone,
 * where the final result covers the standings, the review must be reachable
 * from the result itself by touch and come back to the standings; held
 * sideways, from the standings beside the result; and from the standings' own
 * button under the result, by keyboard, it must leave the page as the
 * result's own button does, with the result's chip bringing it back, focus
 * and all.
 *
 * Before the end, any hand's result opens the review of the game so far,
 * restored from a save waiting at that result. It must ask Trained AI about
 * the hand just played first, then about the rest from the first deal, every
 * group once; lead with that hand, whose own row replays it and takes focus
 * back from the replay; keep the game out of the trend; and come back to the
 * result as it was, with focus on the button that opened it, by keyboard and
 * pointer at desktop and phone sizes, where the sheet must be up again with
 * no chip, and held sideways, where the button shows beside the table. Next
 * hand on the last hand stops it, and the standings' review must then ask
 * about no hand it reviewed before; a game every hand of which was reviewed
 * at the last result must join the trend as Next hand ends it, once. Next
 * hand on an earlier hand, pressed as the review runs, stops it too: nothing
 * is asked while the next hand is played in the page, and that hand's result
 * must open the review leading with it, asking about it first and then only
 * about the hands left waiting, none it had reviewed.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { BATCH, FLAG_PLAYED_BELOW, FLAG_PREFERRED_FROM, LEADING, filterSaid, scanMatch, tableSaid } from '../src/lib/game-review.js';
import { OPPONENT_LABELS, OPPONENT_POSITIONS } from '../src/lib/opponents.js';
import { SEAT_NAMES, tileWords } from '../src/lib/tiles.js';
import { TREND_KEY } from '../src/lib/review-trend.js';
import { MANIFEST } from '../src/lib/model-manifest.js';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { runtimeDirectory } from './copy-runtime.mjs';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const web = fileURLToPath(new URL('../', import.meta.url)), dist = resolve(web, 'dist'), output = resolve(web, 'test-results');
assert.ok(existsSync(resolve(dist, `${await runtimeDirectory()}ort-wasm-simd-threaded.wasm`)), 'build the site first: npm run build');
const MORTAL_REACH = 37;
const networkUrl = `${MANIFEST.origin}/${MANIFEST.object}`;

/** A whole game against Beginner players, by a player who takes every win,
 * declares every riichi and otherwise throws the tile just drawn: plenty for
 * Trained AI to disagree with. With `calls` the player also calls every pon
 * and chii offered, where Trained AI mostly passes. */
function finishedGame(seed, { calls = false } = {}) {
  const match = new MatchSession(Game, seed, 'beginner');
  try {
    match.advance(false);
    for (let step = 0; step < 5000 && !match.over; step++) {
      if (match.engine.hand_is_over()) { match.apply({ type: 'next' }); match.advance(false); continue; }
      const choices = match.choices, drawn = match.view.seats[0].drawn;
      const choice = choices.find(c => ['ron', 'tsumo', 'riichi'].includes(c.kind))
        ?? (calls ? choices.find(c => ['pon', 'chii'].includes(c.kind)) : undefined) ?? choices.find(c => c.kind === 'pass')
        ?? choices.find(c => c.kind === 'discard' && c.tile === drawn) ?? choices.find(c => c.kind === 'discard') ?? choices[0];
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
    }
    assert.ok(match.over);
    return match.snapshot();
  } finally { match.dispose(); }
}
const snapshot = finishedGame(5), callingGame = finishedGame(5, { calls: true });
const hands = scanMatch(Game, snapshot);
/** The match as it was saved after its first `count` commands: waiting at a
 * hand's result where `count` is that hand's end. */
function savedAfter(finished, count) {
  const match = new MatchSession(Game, finished.seed, finished.opponents);
  try {
    match.advance(false);
    for (const command of finished.commands.slice(0, count)) { match.apply(command); match.advance(false); }
    return match.snapshot();
  } finally { match.dispose(); }
}
// The tile-throwing game waiting at the result of its third hand, and at the
// result of its last, before the Next hand that ends it.
const midway = savedAfter(snapshot, hands[2].end), lastResult = savedAfter(snapshot, hands.at(-1).end);
// What the engine says about every decision, read off a replay at each
// hand's end: its legal moves, both of Mortal's masks, and how each of
// Mortal's moves translates back.
const legalCount = mask => Array.from(mask).filter(Boolean).length;
const facts = hands.map(hand => {
  const match = new MatchSession(Game, snapshot.seed, snapshot.opponents);
  try {
    match.advance(false);
    for (const command of snapshot.commands.slice(0, hand.end)) { match.apply(command); match.advance(false); }
    const engine = match.engine, notes = engine.review();
    return notes.map((note, index) => {
      const mask = Array.from(engine.review_mask_mortal(index));
      const reachMask = mask[MORTAL_REACH] ? Array.from(engine.review_mask_after_reach(index)) : null;
      const choices = engine.review_choices(index).map(choice => ({ ...choice,
        mortal: choice.index == null ? -1 : engine.mortal_action_of(choice.index) }));
      return { note, mask, reachMask, choices, forced: legalCount(mask) === 1,
        plain: Array.from({ length: 46 }, (_, action) => engine.review_action_from_mortal(index, action, false)),
        declared: Array.from({ length: 34 }, (_, action) => engine.review_action_from_mortal(index, action, true)) };
    });
  } finally { match.dispose(); }
});

/** Walks the batches the review sent, asking about the hands of `order` in
 * that order: a hand at a time, eight decisions a group, forced moves
 * unasked, then each group's declarations. Every batch must be the group it
 * stands for, asked from the decisions' own positions, and answered. Says the
 * network's answers to each decision, and how many batches were walked, so
 * that a caller can tell whether any was missing or extra. */
function walkBatches(batches, order) {
  const first = new Map(), second = new Map();
  let cursor = 0;
  for (const hand of order) {
    const decisions = facts[hand];
    for (let start = 0; start < decisions.length; start += BATCH) {
      const group = decisions.map((_, index) => index).slice(start, start + BATCH).filter(index => !decisions[index].forced);
      if (!group.length) continue;
      const batch = batches[cursor++];
      assert.equal(batch?.count, group.length, `hand ${hand}, group from ${start}`);
      assert.ok(batch.answers, 'every batch was answered');
      group.forEach((index, slot) => {
        assert.deepEqual(batch.mask.slice(slot * 46, (slot + 1) * 46), decisions[index].mask, 'the decision was asked from its own position');
        first.set(`${hand}:${index}`, batch.answers[slot]);
      });
      const reaching = group.filter((_, slot) => batch.answers[slot].action === MORTAL_REACH);
      const asked = reaching.filter(index => legalCount(decisions[index].reachMask) > 1);
      if (asked.length) {
        const named = batches[cursor++];
        assert.equal(named?.count, asked.length);
        asked.forEach((index, slot) => second.set(`${hand}:${index}`, named.answers[slot]));
      }
      for (const index of reaching.filter(index => !asked.includes(index))) {
        const only = decisions[index].reachMask.indexOf(1);
        second.set(`${hand}:${index}`, { action: only, weights: decisions[index].reachMask });
      }
    }
  }
  return { first, second, walked: cursor };
}

/** Each decision judged from the network's answers alone, as the review is
 * documented to judge it. */
function judged(fact, first, second) {
  const same = (choice, kind, tile) => choice.kind === kind && (choice.tile ?? null) === (tile ?? null);
  const weight = choice => {
    if (choice.mortal < 0) return null;
    if (choice.mortal === MORTAL_REACH) return first.weights[MORTAL_REACH];
    return fact.plain[choice.mortal] === choice.index ? first.weights[choice.mortal] : 0;
  };
  const conditional = choice => {
    if (!second) return null;
    const action = fact.declared.findIndex(index => index === choice.index);
    return action >= 0 ? second.weights[action] : 0;
  };
  const advisedIndex = first.action === MORTAL_REACH ? fact.declared[second.action] : fact.plain[first.action];
  const advised = fact.choices.find(choice => choice.index === advisedIndex);
  const played = fact.choices.find(choice => same(choice, fact.note.played_kind, fact.note.played_tile));
  assert.ok(advised && played);
  const tile = played.kind === 'riichi' && advised.kind === 'riichi';
  const playedWeight = tile ? conditional(played) : weight(played), preferred = tile ? conditional(advised) : weight(advised);
  const agreed = same(played, advised.kind, advised.tile);
  // How the answer states the network's own weight: a riichi by both stages.
  const said = advised.kind === 'riichi'
    ? `${percent(weight(advised))} to declare riichi · discard given riichi: ${percent(conditional(advised))}` : percent(weight(advised));
  return { advised, played, agreed, said, stage: tile ? 'tile' : 'move', playedWeight, preferred,
    flagged: !agreed && !fact.forced && playedWeight != null && playedWeight < FLAG_PLAYED_BELOW && preferred >= FLAG_PREFERRED_FROM };
}

const server = createServer(createFixtureHandler({ root: dist, publicRoot: dist }));
const cases = browserChecks();
let browser;
const percent = weight => weight > 0 && weight < .001 ? '<0.1%' : `${(weight * 100).toFixed(1)}%`;
/** A picture of the screen as it stands on a phone, and of the whole page
 * elsewhere. The table's felt is fixed to the screen (app.css), and a
 * picture of the whole page paints it only behind the part on the screen,
 * leaving the rest washed out; so for such a picture the felt scrolls with
 * the page. */
async function shot(page, name) {
  const path = resolve(output, `${name}.png`);
  if (phoneSized(page)) return page.screenshot({ path });
  const felt = await page.addStyleTag({ content: 'body { background-attachment: scroll !important; }' });
  try {
    return await page.screenshot({ path, fullPage: true });
  } finally {
    await felt.evaluate(style => style.remove());
  }
}
const overflow = page => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1);
/** Whether the page is the size of a phone, held either way, as the site's
 * layouts tell one. */
const phoneSized = page => page.viewport().width <= 760 || page.viewport().height <= 500;
/** Two frames after a short wait: long enough for a list's toggle event and
 * whatever the page does with it. */
const settle = page => page.evaluate(() => new Promise(done => setTimeout(() => requestAnimationFrame(() => requestAnimationFrame(done)), 50)));
/** How many of the review's batches the worker has answered. */
const answered = page => page.evaluate(() => window.reviewBatches.filter(batch => batch.answers).length);
/** Waits until the worker has answered `more` batches beyond `count`. */
const answeredBeyond = (page, count, more = 1) => page.waitForFunction((count, more) =>
  window.reviewBatches.filter(batch => batch.answers).length >= count + more, { timeout: 120000, polling: 100 }, count, more);
/** Taps the element at its middle, brought to the middle of the screen
 * first, clear of the game's bar along the top of a phone's screen. */
async function tapAt(page, selector) {
  await page.$eval(selector, element => element.scrollIntoView({ block: 'center' }));
  await page.tap(selector);
}
/** Closes a page's context before the case ends, so that a case going
 * through several sizes keeps one page open at a time. */
async function closePage(page) {
  const context = page.browserContext();
  cases.contexts.splice(cases.contexts.indexOf(context), 1);
  await context.close();
}
/** Whether a tap at the middle of the element would reach it, rather than
 * whatever lies over it. */
const tappable = (page, selector) => page.$eval(selector, element => {
  const box = element.getBoundingClientRect();
  const hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
  return box.width > 0 && Boolean(hit) && (hit === element || element.contains(hit));
});

/** The page with a finished match restored, the tile-throwing game unless
 * another is `saved`. `touch` makes it a phone, set before the page loads
 * since turning touch on reloads it; without `network` the page finds
 * Trained AI out of reach and never downloads it. `answerDelay` holds each
 * batch the review sends that many milliseconds before the worker has it, as
 * a slower device would take, and `trend` is the games the trend holds from
 * before. `ready` is what the page is waited for by: the standings' review,
 * or for a match saved at a hand's result, that result's. */
async function openPage(width, height, { touch = false, network = true, saved = snapshot, answerDelay = 0, trend = null,
  ready = '.standings [data-review-game]' } = {}) {
  const page = await (await cases.openContext(browser)).newPage();
  page.errors = [];
  page.on('pageerror', error => page.errors.push(error.message));
  await page.setViewport({ width, height, isMobile: touch, hasTouch: touch });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  if (!network) {
    await page.setRequestInterception(true);
    page.on('request', request => {
      const answer = request.url() === networkUrl
        ? request.respond({ status: 404, headers: { 'Access-Control-Allow-Origin': '*' }, body: '' }) : request.continue();
      void answer.catch(() => {}); // A context can close while a request is pending.
    });
  }
  await page.evaluateOnNewDocument((saveKey, settingsKey, saved, answerDelay, trendKey, trend) => {
    if (!localStorage.getItem(saveKey)) localStorage.setItem(saveKey, JSON.stringify(saved));
    if (!localStorage.getItem(settingsKey)) localStorage.setItem(settingsKey, JSON.stringify({ version: 2, difficulty: 'beginner', hints: true }));
    if (trend && !localStorage.getItem(trendKey)) localStorage.setItem(trendKey, JSON.stringify({ version: 1, games: trend }));
    // Every batch the review sends, and the worker's answer to it.
    window.reviewBatches = [];
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', ({ data }) => {
          const batch = data.analyses && window.reviewBatches.find(entry => entry.id === data.id);
          if (batch) batch.answers = data.analyses.map(({ action, weights }) => ({ action, weights: Array.from(weights) }));
        });
      }
      postMessage(data, ...rest) {
        if (data?.count) {
          window.reviewBatches.push({ id: data.id, count: data.count, mask: Array.from(data.mask) });
          if (answerDelay) {
            setTimeout(() => super.postMessage(data, ...rest), answerDelay);
            return;
          }
        }
        super.postMessage(data, ...rest);
      }
    };
  }, SAVE_KEY, SETTINGS_KEY, saved, answerDelay, TREND_KEY, trend);
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/`, { waitUntil: 'networkidle0' });
  await page.waitForSelector(ready);
  return page;
}

/** A place as the list says it: 1st, 2nd, 3rd, 4th, ... 11th, 21st. */
const nth = number => `${number}${[11, 12, 13].includes(number % 100) ? 'th' : { 1: 'st', 2: 'nd', 3: 'rd' }[number % 10] ?? 'th'}`;

/** The flagged decisions listed, in the order shown. */
const listed = page => page.$$eval('[data-game-review] [data-flagged]', items => items.map(item => item.dataset.flagged));
const focusedDecision = page => page.evaluate(() => {
  const active = document.activeElement;
  return active?.dataset.open ?? (!active || active === document.body ? 'the page' : active.outerHTML.slice(0, 80));
});

/** Opens the review and has it go through every hand unheld, however slow
 * the device: Continue asks for every hand, as a player who wants all of it
 * would. Both are pressed without moving focus, which stays where the review
 * put it. The review is opened from `from`: the standings' button by
 * default, pressed where it is, which on a phone held upright is under the
 * result's sheet, or a result's own. */
async function reviewWholeGame(page, from = '.standings [data-review-game]') {
  await page.$eval(from, button => button.click());
  await page.waitForSelector('[data-game-review] [data-stop-review]');
  await page.$eval('[data-game-review] [data-stop-review]', button => button.click());
  await page.waitForSelector('[data-game-review] [data-continue]');
  await page.$eval('[data-game-review] [data-continue]', button => button.click());
}

/** The flagged decision listed last of the leading few, once the first hands
 * reviewed have filled them: the one decisions reviewed later push down. */
async function lastOfLeading(page) {
  await page.waitForFunction(count => document.querySelectorAll('[data-game-review] [data-flagged]').length >= count,
    { timeout: 300000, polling: 50 }, LEADING);
  return (await listed(page))[LEADING - 1];
}

/** The player turns to another window or tab, or comes back from it. The page
 * loses focus, but the element that had it keeps it and has it again when the
 * window comes back: Chrome sends that element a blur and a focusout with no
 * element taking focus, while document.activeElement stays on it, and focus
 * and focusin on the way back, as a real tab switch shows. A headless page
 * cannot be kept in view while another window has focus, so the events are
 * sent here as the window sends them. */
const windowAway = (page, back = false) => page.evaluate(back => {
  const active = document.activeElement;
  active.dispatchEvent(new FocusEvent(back ? 'focus' : 'blur', { relatedTarget: null }));
  active.dispatchEvent(new FocusEvent(back ? 'focusin' : 'focusout', { bubbles: true, relatedTarget: null }));
}, back);

/** Opens the review, has it go through the whole game, and puts keyboard
 * focus on the flagged decision listed last of the leading five once the
 * first hands have filled them; with `away`, the player then turns to another
 * window. Waits until a later hand ranks a decision that departed further
 * above it, out of the leading five, or focus leaves it, or the review ends,
 * and says which decision it was. */
async function focusUntilRankedOut(page, { away = false } = {}) {
  await reviewWholeGame(page);
  const key = await lastOfLeading(page);
  await page.focus(`[data-game-review] [data-open="${key}"]`);
  if (away) await windowAway(page);
  await page.waitForFunction((key, leading) => {
    const active = document.activeElement, keys = [...document.querySelectorAll('[data-game-review] [data-flagged]')].map(item => item.dataset.flagged);
    return active?.dataset.open !== key || keys.indexOf(key) >= leading
      || /^Every hand is reviewed/.test(document.querySelector('[data-game-review] .headline')?.textContent ?? '');
  }, { timeout: 300000, polling: 50 }, key, LEADING);
  return key;
}

/** What the status line says of a flagged decision, read at one moment with
 * the list, must be true of the list at that moment: the place it names, any
 * count of the flagged decisions it gives, and that every one is shown. Says
 * what was read. */
async function saysWhereItIs(page, key) {
  const now = await page.evaluate(key => {
    const keys = [...document.querySelectorAll('[data-game-review] [data-flagged]')].map(item => item.dataset.flagged);
    return { rank: keys.indexOf(key), count: keys.length, note: document.querySelector('[data-game-review] [data-moved]')?.textContent ?? '',
      toggle: document.querySelector('[data-game-review] [data-every-flag]')?.textContent ?? '' };
  }, key);
  const said = /^Decisions reviewed since departed further from Trained AI: this one is now (\d+)(?:st|nd|rd|th)(?: of (\d+))?(, and every flagged decision is shown)?\.$/
    .exec(now.note);
  assert.ok(said, `the status says where the decision went: "${now.note}"`);
  assert.equal(Number(said[1]), now.rank + 1, `the place it names is where the decision is now: "${now.note}"`);
  if (said[2] !== undefined) assert.equal(Number(said[2]), now.count, `a count it names is how many are flagged now: "${now.note}"`);
  if (said[3]) assert.equal(now.toggle, `Show the first ${LEADING}`, 'every flagged decision is shown, as it says');
  return now;
}

/** The replay's overview as the page draws it: each opponent's hand, a list
 * named by its seat, and the player's own, with the names of their tiles,
 * the narrowest tile, how many rows the tiles take and how many are on the
 * first, whether every tile is on
 * the screen and whether the list spills out of its seat; whether anything
 * is face down or
 * offers to show or hide the hands; and the step, what is said of the table,
 * the wall and what each seat was doing. */
const overviewAt = page => page.evaluate(() => {
  const box = element => element.getBoundingClientRect();
  const hand = list => {
    const tiles = [...list.querySelectorAll('.tile')], seat = list.closest('.seat, .mine');
    const row = tile => Math.round(box(tile).bottom);
    return { label: list.getAttribute('aria-label'), seat: list.dataset.seatHand ?? null,
      names: tiles.map(tile => tile.getAttribute('aria-label')), tileWidth: Math.min(...tiles.map(tile => box(tile).width)),
      rows: new Set(tiles.map(row)).size, firstRow: tiles.filter(tile => row(tile) === row(tiles[0])).length,
      spills: list.scrollWidth > list.clientWidth + 1 ? `${list.scrollWidth} in ${list.clientWidth}` : '',
      offscreen: tiles.filter(tile => box(tile).left < -0.5 || box(tile).right > innerWidth + 0.5).map(tile => tile.getAttribute('aria-label')),
      // The rest of the seat: its discards, its sets and its riichi.
      discards: seat.querySelectorAll('.pool .tile').length, melds: seat.querySelectorAll('.meld').length,
      riichi: Boolean(seat.querySelector('.stick, .riichi')) };
  };
  const overview = document.querySelector('[data-decision] [data-overview]');
  return { opponents: [...overview.querySelectorAll('ul.revealed')].map(hand), own: hand(overview.querySelector('.hand[role="list"]')),
    // A concealed quad shows two of its tiles face down, as at a table; a
    // hand shows none.
    facedown: overview.querySelectorAll('.held, .back, :is(ul.revealed, .hand) [aria-label="face-down tile"]').length,
    // A control that offers to show or hide tiles or hands. The hand's own
    // tiles are buttons too, and a safe one is announced "not guaranteed
    // against undeclared hands", so tiles are not counted, and a control
    // counts only where it offers to show, hide or reveal.
    toggles: [...document.querySelectorAll('[data-decision] :is(button, input, select):not(.tile)')]
      .filter(control => control.type === 'checkbox'
        || /\b(show|hide|reveal|peek)\b[^.]*\b(tiles?|hands?)\b/i.test(`${control.textContent} ${control.getAttribute('aria-label') ?? ''}`)).length,
    fits: document.documentElement.scrollWidth <= innerWidth + 1,
    step: document.querySelector('[data-decision] [data-step]').textContent,
    said: overview.querySelector('[data-table-said]').textContent,
    wall: Number(overview.querySelector('.wall strong').textContent),
    notes: [...overview.querySelectorAll('.note')].map(note => note.textContent) };
});
/** A concealed hand as its tiles are named, the tile just drawn last. */
const named = held => [...held.hand.map(tileWords), ...(held.drawn ? [`${tileWords(held.drawn)}, just drawn`] : [])];
/** Tile names without the marks they go on to give, such as dora. */
const unmarked = names => names.map(name => name.split(', ').filter((part, index) => !index || part === 'just drawn').join(', '));

/** Says what the overview shows, having checked it shows, for this step of a
 * hand, every player's tiles as the engine held them, face up, each list
 * named by its seat and keeping to its seat and the screen, with the board as
 * it stood. */
async function overviewIsRight(page, frame, step, steps) {
  const shown = await overviewAt(page), seats = frame.view.seats;
  assert.equal(shown.step, `Your decision ${step + 1} of ${steps}`);
  assert.equal(shown.opponents.length, 3, "three opponents' hands");
  for (const list of shown.opponents) {
    const slot = seats.findIndex(seat => seat.seat === list.seat);
    assert.ok(slot > 0, `an opponent's seat: ${list.seat}`);
    assert.equal(list.label, `${SEAT_NAMES[list.seat]}'s tiles, ${OPPONENT_POSITIONS[slot - 1]} (${OPPONENT_LABELS[seats[slot].controller]})`);
    assert.deepEqual(unmarked(list.names), named(frame.hands[slot]), `${list.seat}'s tiles at step ${step + 1}`);
    assert.deepEqual([list.discards, list.melds, list.riichi], [seats[slot].discards.length, seats[slot].melds.length, seats[slot].riichi],
      `${list.seat}'s discards, sets and riichi at step ${step + 1}`);
  }
  assert.equal(shown.own.label, `${SEAT_NAMES[seats[0].seat]}'s tiles, yours`);
  assert.deepEqual(unmarked(shown.own.names), named(frame.hands[0]), `the player's own tiles at step ${step + 1}`);
  assert.deepEqual([shown.own.discards, shown.own.melds, shown.own.riichi], [seats[0].discards.length, seats[0].melds.length, seats[0].riichi],
    `the player's discards, sets and riichi at step ${step + 1}`);
  // Every tile of every hand is on the screen. An opponent's hand keeps to
  // its seat too, on one row, which a hand is read along, or where one row
  // would draw its tiles under 18 pixels wide, on two of seven, so that they
  // are larger: at every size checked they are 18 pixels or wider. (The
  // player's own hand lets the drawn tile's gap run into its end padding,
  // as in play, so only its tiles are held to the screen.)
  for (const list of [...shown.opponents, shown.own]) {
    assert.deepEqual(list.offscreen, [], `${list.label}: tiles off the screen`);
  }
  for (const list of shown.opponents) {
    assert.equal(list.spills, '', `${list.label} spills out of its seat`);
    assert.ok(list.tileWidth >= 18 && (list.rows === 1 || (list.rows === 2 && list.firstRow === 7)),
      `${list.label}: ${list.rows} rows, ${list.firstRow} on the first, of tiles ${list.tileWidth} pixels wide`);
  }
  assert.equal(shown.facedown, 0, 'nothing is face down');
  assert.equal(shown.toggles, 0, 'nothing offers to show or hide the hands');
  assert.ok(shown.fits, 'the page does not scroll sideways');
  // The table in a sentence, and whether Trained AI flagged the move there.
  assert.ok(shown.said.startsWith(tableSaid(frame.view)), shown.said);
  assert.match(shown.said.slice(tableSaid(frame.view).length), /^( Trained AI flagged your move here\.)?$/);
  assert.equal(shown.wall, frame.view.wall);
  assert.ok(shown.notes.includes(frame.view.phase === 'act' ? 'To play' : 'To answer'), `whose move it was: ${shown.notes.join(', ')}`);
  return shown;
}

/** Waits for every hand to be reviewed, of the game so far or of the whole
 * game, going on past a slow device's hold. */
async function reviewed(page) {
  const deadline = Date.now() + 300000;
  while (Date.now() < deadline) {
    const state = await page.evaluate(() => ({
      headline: document.querySelector('[data-game-review] .headline')?.textContent ?? '',
      rest: Boolean(document.querySelector('[data-game-review] [data-rest]')),
    }));
    if (/^Every hand (so far )?is reviewed/.test(state.headline)) return;
    assert.doesNotMatch(state.headline, /The review stopped/, state.headline);
    if (state.rest) await page.click('[data-game-review] [data-rest]');
    await new Promise(done => setTimeout(done, 500));
  }
  throw new Error('The whole-game review did not finish in five minutes');
}

/** The tile-throwing game's decisions, by the saved command that made each. */
const decisionAt = new Map(hands.flatMap(hand => hand.frames.map(frame => [frame.command, frame])));
/** Whether the page has saved `count` commands of its match, and shows what
 * `selector` names, if anything. */
const savedTo = (key, count, selector) => JSON.parse(localStorage.getItem(key))?.commands.length === count
  && (!selector || Boolean(document.querySelector(selector)));

/** Plays `hand` of the tile-throwing game in the page from its deal, by
 * pointer, pressing in order the moves its saved commands hold, as the
 * replays here apply them: a discard on its tile in the hand, and any other
 * move on its own button among the calls, the one in the place the engine
 * offers it. Each is pressed once the page has saved every command before it
 * and offers the move. The opponents are the engine's own, which play as each
 * move is applied, so every command a hand saves is a move of the player's.
 * Waits for the hand's result. */
async function playHand(page, hand) {
  for (let at = hands[hand.index - 1].end + 1; at < hand.end; at++) {
    const command = snapshot.commands[at], frame = decisionAt.get(at);
    assert.ok(command.type === 'choose' && frame, `command ${at} is a move of the player's`);
    const call = frame.choices.filter(choice => choice.kind !== 'discard')
      .findIndex(choice => choice.kind === command.kind && (choice.tile ?? null) === command.tile);
    const move = command.kind === 'discard' ? `.play-area .hand button[data-tile="${command.tile}"]:not(:disabled)`
      : `.play-area .call-options button:nth-child(${call + 1})[data-choice="${command.kind}"]:not(:disabled)`;
    await page.waitForFunction(savedTo, { timeout: 15000, polling: 50 }, SAVE_KEY, at, move);
    await page.$eval(move, button => button.click());
  }
  await page.waitForFunction(savedTo, { timeout: 15000, polling: 50 }, SAVE_KEY, hand.end, '.play-area .screen');
}

try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  browser = await launchChrome();
  await cases.check('a finished match is reviewed whole, ranked from the real answers, replayed and chosen again', async () => {
    const page = await openPage(1100, 900);
    await page.click('.standings [data-review-game]');
    await page.waitForSelector('[data-game-review]');
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title'); // The review takes focus.
    assert.equal(await page.$eval('.match-view', el => getComputedStyle(el).display), 'none', 'the table steps aside');
    // Waiting on Stop: when the review ends, the button goes, and focus moves
    // on to the flagged decisions rather than to the page.
    await page.waitForSelector('[data-game-review] [data-stop-review]');
    await page.focus('[data-game-review] [data-stop-review]');
    await reviewed(page);
    await page.waitForFunction(() => document.activeElement?.matches('[data-game-review] h3'), { timeout: 5000 });
    assert.equal(await page.evaluate(() => document.activeElement.textContent), 'Where you departed most from Trained AI');

    // Every batch, in the order the review asks: a finished game's from its
    // first deal.
    const batches = await page.evaluate(() => window.reviewBatches);
    const { first, second, walked } = walkBatches(batches, facts.map((_, hand) => hand));
    assert.equal(walked, batches.length, 'no batch is missing or extra');

    const verdicts = facts.flatMap((decisions, hand) => decisions.map((fact, index) => {
      const key = `${hand}:${index}`;
      if (fact.forced) return { key, forced: true };
      return { key, order: 0, ...judged(fact, first.get(key), second.get(key)) };
    }));
    verdicts.forEach((verdict, order) => { verdict.order = order; });
    const counted = verdicts.filter(verdict => !verdict.forced);
    const matched = counted.filter(verdict => verdict.agreed).length;
    assert.match(await page.$eval('[data-trend] .count', el => el.textContent), new RegExp(`^${matched} of ${counted.length} decisions\\.`));
    const flagged = verdicts.filter(verdict => verdict.flagged)
      .sort((a, b) => (b.preferred - b.playedWeight) - (a.preferred - a.playedWeight) || a.order - b.order);
    assert.ok(flagged.length > 0, 'Trained AI flags some of a tile-throwing game');
    const shown = await page.$$eval('[data-game-review] [data-flagged]', items => items.map(item => ({
      key: item.dataset.flagged, weight: item.querySelector('.weight').textContent })));
    assert.deepEqual(shown.map(item => item.key), flagged.slice(0, LEADING).map(verdict => verdict.key), 'flagged and ranked by the gap');
    shown.forEach((item, index) => {
      const { stage, playedWeight, preferred } = flagged[index];
      assert.equal(item.weight, `Trained AI gave ${stage === 'tile' ? 'this tile, once riichi was declared,' : 'this move'} `
        + `${percent(playedWeight)}, and its own choice ${percent(preferred)}.`);
    });
    await shot(page, 'game-review-list');

    // The game enters the trend once, with the figure shown and the version
    // of Trained AI that judged it: the default, as no other was picked.
    const trend = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), TREND_KEY);
    assert.equal(trend.games.length, 1);
    assert.deepEqual({ ...trend.games[0], at: 0 }, { id: `${snapshot.seed}:${snapshot.commands.length}`, network: MANIFEST.version,
      at: 0, matched, decisions: counted.length, flagged: flagged.length });
    assert.equal(await page.$eval('[data-trend] .label', el => el.textContent), `Matched Trained AI ${MANIFEST.version}`);

    // The highest-ranked flagged decision with two of the player's decisions
    // before it in its hand, so there is a hand to step through, opened
    // where it was made.
    const lead = flagged.find(verdict => Number(verdict.key.split(':')[1]) >= 2);
    assert.ok(lead, 'a flagged decision late enough in its hand to replay');
    const [hand, index] = lead.key.split(':').map(Number), frames = hands[hand].frames;
    if (!shown.some(item => item.key === lead.key)) {
      await page.evaluate(() => [...document.querySelectorAll('[data-game-review] button')].find(b => /^Show all/.test(b.textContent)).click());
    }
    await page.click(`[data-game-review] [data-open="${lead.key}"]`);
    await page.waitForSelector('[data-decision]');
    assert.equal(await page.$eval('#decision-title', el => el.textContent), `${hands[hand].name}, your decision ${index + 1}`);
    const status = () => page.$eval('[data-decision] .where [data-step]', el => el.textContent);
    assert.equal(await status(), `Your decision ${index + 1} of ${index + 1}, the one flagged`);
    assert.equal(await page.$('[data-answer]'), null, "the answer waits for the player's choice");
    const wall = () => page.$eval('[data-decision] .wall strong', el => Number(el.textContent));
    assert.equal(await wall(), frames[index].view.wall, 'the table as it was at the decision');

    // Replayed from the deal by keyboard, a decision of the player's at a
    // time. A step that disables the button pressed hands focus on, never to
    // the page: at the deal to Next, at the decision to the hand.
    const press = async label => {
      await page.evaluate(text => [...document.querySelectorAll('[data-decision] .steps button')].find(b => b.textContent === text).focus(), label);
      await page.keyboard.press('Enter');
    };
    const focusOn = target => page.waitForFunction(target => {
      const active = document.activeElement;
      return target === 'hand' ? active?.matches('[data-decision] .hand') : active?.getAttribute('aria-label') === target;
    }, { timeout: 5000 }, target);
    await press('Previous');
    assert.equal(await status(), `Your decision ${index} of ${index + 1}`);
    await focusOn('Previous decision');
    await press('From the deal');
    assert.equal(await status(), `Your decision 1 of ${index + 1}`);
    assert.equal(await wall(), frames[0].view.wall);
    await focusOn('Next decision');
    const said = await page.$eval('[data-decision] #hand-help', el => el.textContent.trim());
    assert.match(said, /^Your move: .+\. (Trained AI chose (the same|differently)|It was the only move)\.$/);
    // The step says the move made there as well as where it is, so stepping
    // by ear tells the hand too.
    assert.equal(await page.$eval('[data-decision] .where', el => el.textContent), `Your decision 1 of ${index + 1}. ${said}`);
    await page.keyboard.press('Enter');
    assert.equal(await status(), `Your decision 2 of ${index + 1}`);
    assert.equal(await wall(), frames[1].view.wall);
    await focusOn('Next decision');
    await press('Previous');
    assert.equal(await status(), `Your decision 1 of ${index + 1}`);
    await focusOn('Next decision');
    await press('This decision');
    assert.equal(await status(), `Your decision ${index + 1} of ${index + 1}, the one flagged`);
    await focusOn('hand');
    // The hand says what to do through its description, the prompt.
    assert.match(await page.$eval('[data-decision] .hand', el => document.getElementById(el.getAttribute('aria-describedby')).textContent),
      /^What would you play here\? Choose again before you see/);
    await press('Previous');
    await focusOn('Previous decision');
    await press('Next');
    assert.equal(await status(), `Your decision ${index + 1} of ${index + 1}, the one flagged`);
    await focusOn('hand');
    assert.equal(await page.$('[data-answer]'), null);
    // Chosen again: another move than the game's, a discard where there is
    // one, otherwise a move that names no tile, such as a pass.
    const fact = facts[hand][index];
    const others = fact.choices.filter(choice => !(choice.kind === lead.played.kind && (choice.tile ?? null) === (lead.played.tile ?? null)));
    const retry = others.find(choice => choice.kind === 'discard') ?? others.find(choice => !choice.tile) ?? others[0];
    if (retry.kind === 'discard') await page.click(`[data-decision] .hand button[data-tile="${retry.tile}"]:not(:disabled)`);
    else {
      const chosen = await page.evaluate((kind, tile) => {
        const button = [...document.querySelectorAll('[data-decision] [data-retry]')]
          .find(entry => entry.dataset.retry === kind && (!tile || entry.querySelector(`[data-tile="${tile}"]`)));
        button?.click();
        return Boolean(button);
      }, retry.kind, retry.tile ?? null);
      assert.ok(chosen, `the ${retry.kind} is offered again`);
    }
    await page.waitForSelector('[data-answer]');
    await page.waitForFunction(() => document.activeElement?.id === 'answer-title'); // The answer takes focus.
    const verdict = await page.$$eval('[data-answer] .verdicts li', items => items.map(item => item.textContent.trim()));
    assert.match(verdict[0], new RegExp(`^This time: ${retry.label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} · `));
    assert.equal(verdict.at(-1), `Trained AI: ${lead.advised.label} · ${lead.said}`, "Trained AI's answer, from the worker");
    const marks = await page.$$eval('[data-answer] .row', rows => rows.map(row => ({ move: row.dataset.move,
      marks: [...row.querySelectorAll('.mark')].map(mark => mark.textContent) })));
    assert.ok(marks.find(row => row.move === lead.advised.label)?.marks.includes("Trained AI's choice"));
    assert.ok(marks.find(row => row.move === lead.played.label)?.marks.includes('Your move in the game'));
    assert.ok(marks.find(row => row.move === retry.label)?.marks.includes('Your choice now'));
    if (!lead.played.kind.match(/pass|pon|chii|kan|ron/) && !fact.note.call_tile) {
      assert.ok(await page.$('[data-answer] .numbers'), 'a turn compares what each move leaves');
    }
    await shot(page, 'game-review-answer');
    // Again from the same position, by keyboard: the answer goes with the
    // button pressed, and the hand takes focus to be chosen from again.
    await page.focus('[data-answer] [data-again]');
    await page.keyboard.press('Enter');
    assert.equal(await page.$('[data-answer]'), null);
    await focusOn('hand');
    assert.match(await page.$eval('[data-decision] #hand-help', el => el.textContent), /Choose again before you see/);
    // The answer outright, by keyboard: the button goes, and the answer's
    // heading takes focus.
    await page.focus('[data-decision] [data-reveal]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.id === 'answer-title', { timeout: 5000 });
    assert.equal(await page.$('[data-decision] [data-reveal]'), null);
    // Then back to the list where it was opened.
    await page.click('[data-decision] [data-back]');
    // Focus returns to the decision opened.
    await page.waitForFunction(key => document.activeElement?.dataset.open === key, {}, lead.key);

    // The answer leads with five moves and those it marks. A decision with
    // more hides the rest behind a button that stays to hide them again, so
    // focus stays on it both ways.
    const wide = flagged.find(verdict => {
      const [at, number] = verdict.key.split(':').map(Number);
      return facts[at][number].choices.length >= 8;
    });
    assert.ok(wide, 'a flagged decision with more moves than the answer leads with');
    if (!await page.$(`[data-game-review] [data-open="${wide.key}"]`)) {
      await page.evaluate(() => [...document.querySelectorAll('[data-game-review] button')].find(b => /^Show all/.test(b.textContent)).click());
    }
    await page.click(`[data-game-review] [data-open="${wide.key}"]`);
    await page.waitForSelector('[data-decision] [data-reveal]');
    await page.focus('[data-decision] [data-reveal]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.id === 'answer-title', { timeout: 5000 });
    const rows = () => page.$$eval('[data-answer] .row', items => items.length);
    const leading = await rows();
    await page.focus('[data-answer] [data-every-move]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.matches('[data-answer] [data-every-move]')
      && document.activeElement.textContent === 'Show the leading moves', { timeout: 5000 });
    assert.ok(await rows() > leading, 'every move is shown');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.matches('[data-answer] [data-every-move]')
      && /^Show every move \(\d+ more\)$/.test(document.activeElement.textContent), { timeout: 5000 });
    assert.equal(await rows(), leading);
    await page.click('[data-decision] [data-back]');
    await page.waitForFunction(key => document.activeElement?.dataset.open === key, {}, wide.key);

    // On a phone, the list and a decision fit the screen. Only the size
    // changes: emulating touch would reload the page, and the review with it.
    await page.setViewport({ width: 360, height: 780 });
    assert.ok(await page.$('[data-game-review] [data-flagged]'), 'the review is still open');
    assert.ok(await overflow(page), 'the list fits a phone');
    await shot(page, 'game-review-phone');
    await page.click(`[data-game-review] [data-open="${lead.key}"]`);
    await page.waitForSelector('[data-decision]');
    assert.ok(await overflow(page), 'the replayed table fits a phone');
    // The decision's heading is brought to the top of the screen, below the
    // game's bar, which stays there on a phone.
    await page.waitForFunction(() => {
      const bar = document.querySelector('header.bar').getBoundingClientRect();
      const heading = document.querySelector('#decision-title').getBoundingClientRect();
      return heading.top >= bar.bottom && heading.bottom <= innerHeight;
    });
    await shot(page, 'game-review-phone-decision');
    await page.click('[data-decision] [data-back]');
    await page.waitForSelector('[data-game-review] [data-flagged]');

    // Back to the standings, as they were; the saved match is untouched.
    await page.evaluate(() => scrollTo(0, 0));
    await page.click('[data-game-review] [data-close-review]');
    await page.waitForFunction(() => !document.querySelector('[data-game-review]') && document.activeElement?.matches('.standings [data-review-game]'));
    assert.notEqual(await page.$eval('.match-view', el => getComputedStyle(el).display), 'none');
    const { _storage, ...kept } = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
    assert.deepEqual(kept, snapshot, 'the save is unchanged');
    assert.deepEqual(page.errors, []);
  });

  await cases.check('a reviewed hand replays from its deal with every hand face up and the board at every step, at desktop, tablet and phone sizes, by keyboard and pointer', async () => {
    const page = await openPage(1100, 900);
    await reviewWholeGame(page);
    await reviewed(page);
    // Of the hands with a flagged decision, the one with the most on the
    // table: sets called and riichi declared.
    await page.evaluate(() => [...document.querySelectorAll('[data-game-review] [data-every-flag]')].find(b => /^Show all/.test(b.textContent))?.click());
    const flaggedHands = new Set((await listed(page)).map(key => Number(key.split(':')[0])));
    const busy = hand => hand.frames.reduce((sum, frame) => sum
      + frame.view.seats.reduce((count, seat) => count + seat.melds.length + (seat.riichi ? 1 : 0), 0), 0);
    const chosen = hands.filter(hand => hand.frames.length >= 4 && flaggedHands.has(hand.index)).sort((a, b) => busy(b) - busy(a))[0];
    assert.ok(chosen, 'a hand of four decisions or more with one flagged');
    const { frames } = chosen, focused = () => page.evaluate(() => {
      const active = document.activeElement;
      return !active || active === document.body ? 'the page' : active.getAttribute('aria-label') ?? active.textContent.trim();
    });
    // By keyboard: the list of hands, unless a slow device's review opened
    // it already, then the hand's own button.
    if (!await page.$eval('[data-game-review] .hands', details => details.open)) {
      await page.focus('[data-game-review] .hands summary');
      await page.keyboard.press('Enter');
    }
    await page.waitForSelector(`[data-game-review] [data-replay-hand="${chosen.index}"]`, { visible: true });
    await page.focus(`[data-game-review] [data-replay-hand="${chosen.index}"]`);
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.id === 'decision-title', { timeout: 5000 });
    assert.equal(await page.$eval('#decision-title', el => el.textContent), `${chosen.name}, decision by decision`);
    // Every step by keyboard, from the deal to the hand's last decision. A
    // step keeps focus on Next, and the last hands it to Previous.
    await page.evaluate(() => [...document.querySelectorAll('[data-decision] .steps button')].find(b => b.textContent === 'Next').focus());
    for (const [step, frame] of frames.entries()) {
      await page.waitForFunction(text => document.querySelector('[data-decision] [data-step]')?.textContent === text, { timeout: 5000 },
        `Your decision ${step + 1} of ${frames.length}`);
      await overviewIsRight(page, frame, step, frames.length);
      assert.equal(await focused(), step < frames.length - 1 ? 'Next decision' : 'Previous decision', `focus after step ${step + 1}`);
      if (step < frames.length - 1) await page.keyboard.press('Enter');
    }
    await shot(page, 'review-overview-1100x900');

    // By pointer, through the hand's list of decisions, open beside the
    // table on a wide screen: a step's button keeps focus as it shows it.
    const target = Math.floor(frames.length / 2);
    await page.click(`[data-decision] [data-step-to="${target}"]`);
    await overviewIsRight(page, frames[target], target, frames.length);
    assert.equal(await page.evaluate(() => document.activeElement?.dataset.stepTo), String(target));
    assert.equal(await page.$eval(`[data-decision] [data-step-to="${target}"]`, el => el.getAttribute('aria-current')), 'step');
    // A flagged decision reached in a hand replayed whole offers its retry,
    // by keyboard: Choose again here gives the hand focus, as a turn does,
    // with every hand still in view, and the answer outright its heading.
    const flaggedStep = Number(await page.$$eval('[data-decision] [data-step-to]', buttons =>
      buttons.find(button => button.querySelector('.verdict')?.textContent === 'Flagged')?.dataset.stepTo));
    await page.click(`[data-decision] [data-step-to="${flaggedStep}"]`);
    await page.waitForSelector('[data-decision] [data-offer]');
    assert.match(await page.$eval('[data-decision] [data-offer] p', el => el.textContent), /^Flagged: Trained AI gave /);
    await page.focus('[data-decision] [data-choose-here]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.matches('[data-decision] .hand'), { timeout: 5000 });
    assert.match(await page.$eval('[data-decision] .hand', el => document.getElementById(el.getAttribute('aria-describedby')).textContent),
      /^What would you play here\? Choose again before you see Trained AI's answer/);
    await overviewIsRight(page, frames[flaggedStep], flaggedStep, frames.length);
    await page.focus('[data-decision] [data-reveal]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.id === 'answer-title', { timeout: 5000 });
    assert.ok(await page.$('[data-decision] [data-answer]'), "Trained AI's answer");
    await page.click(`[data-decision] [data-step-to="${target}"]`);
    // Its filter, by keyboard: the moves where Trained AI chose differently,
    // said as the list changes, and then every decision again.
    const filterOption = option => `[data-decision] [data-decisions] [data-filter-option="${option}"]`;
    const counted = async option => Number(/\((\d+)\)$/.exec(await page.$eval(filterOption(option), el => el.textContent.trim()))[1]);
    const counts = { flagged: await counted('flagged'), differed: await counted('differed'), every: await counted('every') };
    assert.equal(counts.every, frames.length, 'the list holds every decision of the hand');
    await page.focus(filterOption('differed'));
    await page.keyboard.press('Enter');
    await page.waitForFunction(selector => document.querySelector(selector)?.getAttribute('aria-pressed') === 'true', { timeout: 5000 }, filterOption('differed'));
    assert.equal(await page.$eval('[data-decision] [data-decisions] [data-filter-said]', el => el.textContent),
      filterSaid('differed', counts, { scope: ' in this hand', ranked: false }));
    const verdicts = await page.$$eval('[data-decision] [data-step-to] .verdict', items => items.map(item => item.textContent));
    assert.equal(verdicts.length, counts.differed);
    assert.ok(verdicts.every(verdict => ['Flagged', 'Trained AI differed'].includes(verdict)), verdicts.join(', '));
    assert.equal(verdicts.filter(verdict => verdict === 'Flagged').length, counts.flagged);
    assert.ok(await page.evaluate(selector => document.activeElement?.matches(selector), filterOption('differed')), 'focus stays on the filter');
    // Focus leaving the filter quiets the line, which spoke of the list as it was.
    await page.focus('[data-decision] [data-back]');
    await page.waitForFunction(() => document.querySelector('[data-decision] [data-decisions] [data-filter-said]')?.textContent === '', { timeout: 5000 });
    await page.click(filterOption('every'));
    assert.equal(await page.$$eval('[data-decision] [data-step-to]', items => items.length), frames.length);

    // The same overview at each size, the hands fitting the screen, and a
    // step by pointer at each. Only the size changes: emulating touch would
    // reload the page, and the review with it. A phone's picture is of the
    // screen, so there are two: the table, and the player's own seat.
    let step = target;
    for (const [width, height] of [[1366, 768], [800, 1280], [390, 844], [320, 568], [844, 390]]) {
      await page.setViewport({ width, height });
      await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
      await overviewIsRight(page, frames[step], step, frames.length);
      const forward = step < frames.length - 1;
      step += forward ? 1 : -1;
      await page.$eval('[data-decision] .steps', el => el.scrollIntoView({ block: 'center' }));
      await page.click(`[data-decision] .steps button[aria-label="${forward ? 'Next decision' : 'Previous decision'}"]`);
      await page.waitForFunction(text => document.querySelector('[data-decision] [data-step]')?.textContent === text, { timeout: 5000 },
        `Your decision ${step + 1} of ${frames.length}`);
      await overviewIsRight(page, frames[step], step, frames.length);
      assert.notEqual(await focused(), 'the page', `focus stays in the replay at ${width}x${height}`);
      await page.$eval('[data-decision] #overview-title', el => el.scrollIntoView({ block: 'start' }));
      await shot(page, `review-overview-${width}x${height}`);
      if (phoneSized(page)) {
        // The player's seat from its top, just below the game's bar.
        await page.$eval('[data-decision] [data-overview] .mine', el =>
          scrollBy(0, el.getBoundingClientRect().top - document.querySelector('header.bar').getBoundingClientRect().bottom - 8));
        await shot(page, `review-overview-${width}x${height}-yours`);
      }
    }

    // Back to the list, by pointer: focus returns to the hand's own button.
    await page.setViewport({ width: 1100, height: 900 });
    await page.click('[data-decision] [data-back]');
    await page.waitForFunction(index => document.activeElement?.dataset.replayHand === String(index), { timeout: 5000 }, chosen.index);
    assert.deepEqual(page.errors, []);
  });

  await cases.check("a hand's list of decisions, folded on a phone and to choose again, stays as the player leaves it, with focus on the step's button pressed, as the step changes and later hands are reviewed, by touch and keyboard", async () => {
    // A page loaded at each size, one at a time. On a phone the list starts
    // folded under the steps; at 1100 by 900 it starts folded in a replay
    // opened to choose again. Each batch reaches the worker a second late, so
    // that later hands are still being reviewed while the list is used,
    // however fast the machine.
    for (const [width, height] of [[390, 844], [320, 568], [844, 390], [1100, 900]]) {
      const size = `${width}x${height}`, page = await openPage(width, height, { touch: true, answerDelay: 1000 });
      await reviewWholeGame(page);
      if (phoneSized(page)) {
        // The first hand reviewed, replayed whole from the list of hands.
        const hand = await (await page.waitForFunction(() => document.querySelector('[data-game-review] [data-replay-hand]')?.dataset.replayHand ?? false,
          { timeout: 300000, polling: 100 })).jsonValue();
        if (!await page.$eval('[data-game-review] .hands', details => details.open)) await tapAt(page, '[data-game-review] .hands summary');
        await tapAt(page, `[data-game-review] [data-replay-hand="${hand}"]`);
      } else {
        // A flagged decision with two of the player's decisions before it.
        const key = await (await page.waitForFunction(() => {
          const all = document.querySelector('[data-game-review] [data-every-flag]');
          if (all && /^Show all/.test(all.textContent)) all.click();
          return [...document.querySelectorAll('[data-game-review] [data-flagged]')].map(item => item.dataset.flagged)
            .find(key => Number(key.split(':')[1]) >= 2) ?? false;
        }, { timeout: 300000, polling: 100 })).jsonValue();
        await tapAt(page, `[data-game-review] [data-open="${key}"]`);
      }
      const list = '[data-decision] [data-decisions]';
      await page.waitForSelector(list);
      /** Whether the list is open, and the step whose button has focus, or
       * what else has it. */
      const state = () => page.evaluate(list => {
        const active = document.activeElement;
        return { open: document.querySelector(list).open, focus: active?.dataset.stepTo
          ?? (!active || active === document.body ? 'the page' : active.getAttribute('aria-label') ?? active.textContent.trim()) };
      }, list);
      const onStep = index => page.waitForFunction((list, index) => document.querySelector(`${list} [data-step-to="${index}"]`)
        ?.getAttribute('aria-current') === 'step', { timeout: 5000 }, list, index);
      assert.equal((await state()).open, false, `the list starts folded at ${size}`);
      // By touch: the list opened, then a step's button.
      await tapAt(page, `${list} summary`);
      await page.waitForFunction(list => document.querySelector(list).open, { timeout: 5000 }, list);
      await settle(page);
      await tapAt(page, `${list} [data-step-to="1"]`);
      await onStep(1);
      await settle(page);
      assert.deepEqual(await state(), { open: true, focus: '1' }, `a step's button tapped at ${size}`);
      // By keyboard: another step's button.
      await page.focus(`${list} [data-step-to="0"]`);
      await page.keyboard.press('Enter');
      await onStep(0);
      await settle(page);
      assert.deepEqual(await state(), { open: true, focus: '0' }, `a step's button pressed by keyboard at ${size}`);
      // Later hands are reviewed meanwhile, and nothing is pressed: the list
      // stays open, and focus on the button.
      await answeredBeyond(page, await answered(page), 2);
      await settle(page);
      assert.deepEqual(await state(), { open: true, focus: '0' }, `later hands reviewed at ${size}`);
      // Folded by touch, it stays folded as the step changes and the review
      // goes on.
      await tapAt(page, `${list} summary`);
      await page.waitForFunction(list => !document.querySelector(list).open, { timeout: 5000 }, list);
      await tapAt(page, '[data-decision] .steps button[aria-label="Next decision"]');
      await onStep(1);
      await answeredBeyond(page, await answered(page));
      await settle(page);
      assert.equal((await state()).open, false, `the list stays folded at ${size}`);
      assert.deepEqual(page.errors, []);
      await closePage(page);
    }
  });

  await cases.check("the review's other lists stay as the player leaves them: a slow device's list of hands as the review goes on, a hand's events across a step with none, and the trend's games across a replay, by keyboard and pointer", async () => {
    // Each batch reaches the worker four and a half seconds late, over half a
    // second a position: a device slow enough to hold after each hand, for
    // which the list of hands opens. A game reviewed before this one puts the
    // trend's table of games on the page from the first hand.
    const page = await openPage(1100, 900, { answerDelay: 4500,
      trend: [{ id: 'earlier', network: MANIFEST.version, at: Date.UTC(2026, 9, 1, 12), matched: 60, decisions: 100, flagged: 4 }] });
    // What the page did not keep as the player left it, said together at the
    // end, so that each list is checked whatever the others do.
    const changed = [];
    const stays = async (selector, open, what) => {
      await settle(page);
      if (await page.$eval(selector, details => details.open) !== open) changed.push(what);
    };
    const handList = '[data-game-review] .hands', games = '[data-game-review] [data-trend] details.games';
    await page.click('.standings [data-review-game]');
    await page.waitForSelector('[data-game-review] [data-next-hand]', { timeout: 300000 });
    assert.equal(await page.$eval(handList, details => details.open), true, 'the list of hands opens for the hold');
    // Folded by keyboard, it stays folded, also while the next hand is
    // reviewed and at the hold after it.
    await page.focus(`${handList} summary`);
    await page.keyboard.press('Enter');
    await stays(handList, false, 'the list of hands opened again as it was folded');
    const before = await answered(page);
    await page.focus('[data-game-review] [data-next-hand]');
    await page.keyboard.press('Enter');
    await answeredBeyond(page, before);
    await stays(handList, false, 'the list of hands opened again as the review went on');
    await page.waitForSelector('[data-game-review] [data-next-hand]', { timeout: 300000 });
    await stays(handList, false, "the list of hands opened again at the review's next hold");

    // The trend's table of games, opened by pointer.
    await page.click(`${games} summary`);
    await page.waitForFunction(games => document.querySelector(games).open, { timeout: 5000 }, games);
    // A reviewed hand with a step that nothing happened before, between two
    // that had events, replayed whole by pointer: its events, folded at the
    // step before, stay folded at the step after.
    const done = await page.$$eval('[data-game-review] [data-replay-hand]', buttons => buttons.map(button => Number(button.dataset.replayHand)));
    const quiet = hands.filter(hand => done.includes(hand.index)).flatMap(hand => hand.frames.map((_, step) => ({ hand, step })))
      .find(({ hand, step }) => step > 0 && step < hand.frames.length - 1 && !hand.frames[step].lines.length
        && hand.frames[step - 1].lines.length && hand.frames[step + 1].lines.length);
    assert.ok(quiet, `a reviewed hand with a step that nothing happened before, among hands ${done.join(', ')}`);
    if (!await page.$eval(handList, details => details.open)) await page.click(`${handList} summary`);
    await page.click(`[data-game-review] [data-replay-hand="${quiet.hand.index}"]`);
    const lines = '[data-decision] details.lines', next = '[data-decision] .steps button[aria-label="Next decision"]';
    const onStep = step => page.waitForFunction(text => document.querySelector('[data-decision] [data-step]')?.textContent === text,
      { timeout: 5000 }, `Your decision ${step + 1} of ${quiet.hand.frames.length}`);
    await page.click(`[data-decision] [data-decisions] [data-step-to="${quiet.step - 1}"]`);
    await onStep(quiet.step - 1);
    await page.click(`${lines} summary`);
    await page.waitForFunction(lines => !document.querySelector(lines).open, { timeout: 5000 }, lines);
    await page.click(next);
    await onStep(quiet.step);
    assert.equal(await page.$(lines), null, 'no events since the previous decision');
    await page.click(next);
    await onStep(quiet.step + 1);
    await stays(lines, false, "a hand's events opened again after a step with none");
    // Back to the list by keyboard: focus returns to the hand's own button,
    // and the trend's games are open still.
    await page.focus('[data-decision] [data-back]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(index => document.activeElement?.dataset.replayHand === String(index), { timeout: 5000 }, quiet.hand.index);
    await stays(games, true, "the trend's games folded after a replay");
    assert.deepEqual(changed, [], 'every list stays as the player left it');
    assert.deepEqual(page.errors, []);
  });

  await cases.check("the list shows the flagged decisions, those where Trained AI chose differently or every one, says how many each holds, and opens any of them, by keyboard and pointer", async () => {
    const page = await openPage(1100, 900);
    await reviewWholeGame(page);
    await reviewed(page);
    const option = name => `[data-game-review] [data-filter] [data-filter-option="${name}"]`;
    const counted = async name => Number(/\((\d+)\)$/.exec(await page.$eval(option(name), el => el.textContent.trim()))[1]);
    const counts = { flagged: await counted('flagged'), differed: await counted('differed'), every: await counted('every') };
    const total = Number(/^Every hand is reviewed: (\d+) decisions?\.$/.exec(await page.$eval('[data-game-review] .headline', el => el.textContent))[1]);
    assert.equal(counts.every, total, 'every decision of the game');
    assert.ok(counts.flagged > 0 && counts.differed > counts.flagged && counts.every > counts.differed, JSON.stringify(counts));
    assert.equal(await page.$eval(option('flagged'), el => el.getAttribute('aria-pressed')), 'true', 'the list opens on the flagged decisions');
    const items = () => page.$$eval('[data-game-review] [data-listed]', entries => entries.map(entry => ({ key: entry.dataset.listed,
      flagged: entry.hasAttribute('data-flagged'), said: entry.innerText.replace(/\s+/g, ' ').trim() })));
    const showAll = () => page.evaluate(() => [...document.querySelectorAll('[data-game-review] [data-every-flag]')].find(b => /^Show all/.test(b.textContent))?.click());
    const heading = () => page.$eval('[data-game-review] h3', el => el.textContent);
    await showAll();
    assert.equal((await items()).length, counts.flagged);

    // By pointer: those where Trained AI chose differently, the flagged ones
    // marked among them, said as the list changes, with focus on the button.
    await page.click(option('differed'));
    await page.waitForFunction(selector => document.querySelector(selector)?.getAttribute('aria-pressed') === 'true', { timeout: 5000 }, option('differed'));
    assert.equal(await heading(), 'Where Trained AI chose differently');
    assert.equal(await page.$eval('[data-game-review] [data-filter-said]', el => el.textContent), filterSaid('differed', counts, { leading: LEADING }));
    assert.ok(await page.evaluate(selector => document.activeElement?.matches(selector), option('differed')));
    assert.equal((await items()).length, Math.min(LEADING, counts.differed), 'the leading few, the rest behind Show all');
    await showAll();
    const different = await items();
    assert.equal(different.length, counts.differed);
    assert.equal(different.filter(item => item.flagged).length, counts.flagged);
    assert.ok(different.filter(item => item.flagged).every(item => / · Flagged /.test(item.said) && /Replay and choose again$/.test(item.said)));
    assert.ok(different.filter(item => !item.flagged).every(item => /Trained AI: /.test(item.said) && / Replay$/.test(item.said)));

    // A decision that was not flagged opens on its step with the position,
    // the move and Trained AI's view, and no retry; back by keyboard, focus
    // returns to it.
    const plain = different.find(item => !item.flagged);
    const [handIndex, index] = plain.key.split(':').map(Number), frame = hands[handIndex].frames[index];
    await page.click(`[data-game-review] [data-open="${plain.key}"]`);
    await page.waitForSelector('[data-decision] [data-view]');
    assert.equal(await page.$eval('#decision-title', el => el.textContent), `${hands[handIndex].name}, your decision ${index + 1}`);
    await overviewIsRight(page, frame, index, hands[handIndex].frames.length);
    assert.equal(await page.$('[data-decision] :is([data-reveal], [data-again], [data-retry], [data-choose-here], [data-answer])'), null, 'no retry');
    assert.match(await page.$eval('[data-decision] #hand-help', el => el.textContent.trim()), /^Your move: .+\. Trained AI chose differently\.$/);
    await shot(page, 'review-overview-plain-decision');
    await page.focus('[data-decision] [data-back]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(key => document.activeElement?.dataset.open === key, { timeout: 5000 }, plain.key);
    assert.equal(await heading(), 'Where Trained AI chose differently', 'the list is as it was left');

    // A flagged one is chosen again, with every hand in view all the while.
    const flaggedItem = different.find(item => item.flagged);
    const [flaggedHand, flaggedIndex] = flaggedItem.key.split(':').map(Number);
    await page.click(`[data-game-review] [data-open="${flaggedItem.key}"]`);
    await page.waitForSelector('[data-decision] [data-reveal]');
    const choosing = await overviewAt(page);
    assert.equal(choosing.step, `Your decision ${flaggedIndex + 1} of ${flaggedIndex + 1}, the one flagged`);
    assert.equal(choosing.opponents.length, 3);
    choosing.opponents.forEach(list => {
      const slot = hands[flaggedHand].frames[flaggedIndex].view.seats.findIndex(seat => seat.seat === list.seat);
      assert.deepEqual(unmarked(list.names), named(hands[flaggedHand].frames[flaggedIndex].hands[slot]));
    });
    assert.equal(await page.$('[data-decision] [data-answer]'), null, "the answer waits for the player's choice");
    await page.click('[data-decision] [data-back]');
    await page.waitForFunction(key => document.activeElement?.dataset.open === key, { timeout: 5000 }, flaggedItem.key);

    // By keyboard: every decision, in the order played, each saying how it went.
    await page.focus(option('every'));
    await page.keyboard.press('Enter');
    await page.waitForFunction(selector => document.querySelector(selector)?.getAttribute('aria-pressed') === 'true', { timeout: 5000 }, option('every'));
    assert.equal(await heading(), 'Every decision you made');
    assert.equal(await page.$eval('[data-game-review] [data-filter-said]', el => el.textContent), filterSaid('every', counts, { leading: LEADING }));
    // The screen as it shows the first few: every one of them makes a page
    // too long to keep a picture of.
    await page.screenshot({ path: resolve(output, 'review-overview-every-decision.png') });
    await showAll();
    const every = await items();
    assert.deepEqual(every.map(item => item.key), hands.flatMap(hand => hand.frames.map((_, step) => `${hand.index}:${step}`)));
    assert.ok(every.every(item => /(Trained AI chose the same\.|It was the only move\.|Trained AI gave .+|Trained AI: .+) Replay( and choose again)?$/.test(item.said)));
    // Focus moving on quiets the line, which spoke of the list as it was.
    await page.focus('[data-game-review] h3');
    await page.waitForFunction(() => document.querySelector('[data-game-review] [data-filter-said]')?.textContent === '', { timeout: 5000 });
    assert.deepEqual(page.errors, []);
  });

  await cases.check("the review's own buttons hand focus on as they change, by keyboard", async () => {
    const page = await openPage(1100, 900);
    await page.click('.standings [data-review-game]');
    const focused = selector => page.waitForFunction(selector => document.activeElement?.matches(selector), { timeout: 15000 }, selector);
    // Stop becomes Continue, and Continue Stop, under the player's hand.
    await page.waitForSelector('[data-game-review] [data-stop-review]');
    await page.focus('[data-game-review] [data-stop-review]');
    await page.keyboard.press('Enter');
    await focused('[data-game-review] [data-continue]');
    await page.keyboard.press('Enter');
    await focused('[data-game-review] [data-stop-review]');
    await page.keyboard.press('Enter');
    await focused('[data-game-review] [data-continue]');
    // A hand reviewed on its own loses its button once it is under review;
    // focus stays with the list of hands.
    await page.focus('[data-game-review] .hands summary');
    await page.keyboard.press('Enter');
    await page.waitForSelector('[data-game-review] [data-review-hand]');
    await page.focus('[data-game-review] [data-review-hand]');
    await page.keyboard.press('Enter');
    await focused('[data-game-review] .hands summary');
    assert.deepEqual(page.errors, []);
  });

  await cases.check('a flagged decision keeps keyboard focus while hands reviewed later rank it out of the leading five, and the list says where it is as it grows', async () => {
    const page = await openPage(1100, 900);
    const key = await focusUntilRankedOut(page);
    assert.equal(await focusedDecision(page), key, 'focus stays on the decision the player is on');
    const keys = await listed(page), rank = keys.indexOf(key);
    assert.ok(rank >= LEADING, `hands reviewed later rank ${key} out of the leading ${LEADING}: ${keys.join(', ')}`);
    // It is listed still, as every flagged decision is, and the status says
    // where it went and why the list grew.
    assert.equal(await page.$eval('[data-game-review] [data-every-flag]', el => el.textContent), `Show the first ${LEADING}`);
    assert.equal(await page.$eval('[data-game-review] [data-moved]', el => el.getAttribute('role')), 'status');
    await saysWhereItIs(page, key);
    await shot(page, 'game-review-moved');
    // The hands reviewed after that flag more decisions, most of them below
    // it, while focus stays: what the status says stays true of the list.
    await page.waitForFunction(count => document.querySelectorAll('[data-game-review] [data-flagged]').length > count,
      { timeout: 300000, polling: 50 }, keys.length);
    assert.equal(await focusedDecision(page), key);
    const now = await saysWhereItIs(page, key);
    assert.ok(now.count > keys.length, 'the list has grown since it said so');
    // The place, and no count. Should a later hand move it again before this
    // is read, the line says the new place without repeating that every
    // flagged decision is shown, which it said as the list grew.
    assert.match(now.note, new RegExp(`^Decisions reviewed since departed further from Trained AI: this one is now ${nth(now.rank + 1)}`
      + '(, and every flagged decision is shown)?\\.$'));
    // Focus moving on clears the note, which spoke of the decision it left.
    await page.focus('[data-game-review] [data-every-flag]');
    await page.waitForFunction(() => document.querySelector('[data-game-review] [data-moved]')?.textContent === '', { timeout: 5000 });
    assert.deepEqual(page.errors, []);
  });

  await cases.check('a flagged decision keeps keyboard focus while the player is in another window and hands reviewed meanwhile rank it out of the leading five', async () => {
    const page = await openPage(1100, 900);
    const key = await focusUntilRankedOut(page, { away: true });
    // The window took focus from the page, not from the decision, so its
    // button is kept as the list moves, and the player comes back to it.
    assert.equal(await focusedDecision(page), key, 'focus stays on the decision the player is on');
    const keys = await listed(page), rank = keys.indexOf(key);
    assert.ok(rank >= LEADING, `hands reviewed meanwhile rank ${key} out of the leading ${LEADING}: ${keys.join(', ')}`);
    await saysWhereItIs(page, key);
    // Back in the window, the decision has focus again, and the status still
    // says where it is.
    await windowAway(page, true);
    assert.equal(await focusedDecision(page), key);
    await saysWhereItIs(page, key);
    assert.deepEqual(page.errors, []);
  });

  await cases.check('back from a decision the review ranked out of the leading five while it was open, focus returns to it and the list says where it went, from a status line in the page all along', async () => {
    const page = await openPage(1100, 900);
    await reviewWholeGame(page);
    const key = await lastOfLeading(page);
    await page.click(`[data-game-review] [data-open="${key}"]`);
    await page.waitForSelector('[data-decision]');
    // A screen reader hears a status only once it is in the accessibility
    // tree. So the line that says where the decision went waits in the page,
    // silent, while the decision is open, rather than coming back with the
    // list already speaking.
    const line = await page.$('[data-game-review] [data-moved]');
    assert.ok(line, 'the status line is in the page while a decision is open');
    assert.equal(await line.evaluate(el => el.textContent), '', 'and silent');
    assert.equal((await page.accessibility.snapshot({ root: line, interestingOnly: false }))?.role, 'status', 'and in the accessibility tree');
    // The review goes on behind the decision until every hand is reviewed,
    // which enters the game in the trend.
    await page.waitForFunction(trendKey => Boolean(localStorage.getItem(trendKey)), { timeout: 300000, polling: 250 }, TREND_KEY);
    await page.focus('[data-decision] [data-back]');
    await page.keyboard.press('Enter');
    await page.waitForSelector('[data-game-review] [data-flagged]');
    await page.waitForFunction(key => document.activeElement?.dataset.open === key, { timeout: 5000 }, key).catch(() => {});
    assert.equal(await focusedDecision(page), key, 'focus returns to the decision opened');
    const keys = await listed(page), rank = keys.indexOf(key);
    assert.ok(rank >= LEADING, `the hands reviewed while it was open rank ${key} out of the leading ${LEADING}: ${keys.join(', ')}`);
    // The line that waited is the one that speaks, not one rebuilt with the
    // list.
    assert.ok(await line.evaluate(el => el.isConnected), 'the status line that waited is still in the page');
    assert.equal(await line.evaluate(el => el.textContent),
      `Decisions reviewed since departed further from Trained AI: this one is now ${nth(rank + 1)}, and every flagged decision is shown.`);
    assert.deepEqual(page.errors, []);
  });

  await cases.check("an answer to another's discard is chosen again from the moves the hand's prompt points to, by keyboard", async () => {
    // The tile-throwing player passes every call, and Trained AI seldom
    // gives a pass under 5%, so this game's player calls everything instead.
    const page = await openPage(1100, 900, { saved: callingGame });
    await page.click('.standings [data-review-game]');
    await page.waitForSelector('[data-game-review]');
    await reviewed(page);
    await page.evaluate(() => [...document.querySelectorAll('[data-game-review] button')].find(b => /^Show all/.test(b.textContent))?.click());
    const key = await page.$$eval('[data-game-review] [data-flagged]', items =>
      items.find(item => /^Response to/.test(item.querySelector('.context')?.textContent.trim() ?? ''))?.dataset.flagged);
    assert.ok(key, 'a flagged answer to a discard');
    await page.click(`[data-game-review] [data-open="${key}"]`);
    await page.waitForSelector('[data-decision] [data-retry]');
    // No tile in the hand can be thrown, so the hand's description, the
    // prompt, sends the player to the moves below it, which Tab reaches next.
    assert.equal(await page.$$eval('[data-decision] .hand button:not(:disabled)', buttons => buttons.length), 0, 'no tile to throw');
    const prompt = () => page.$eval('[data-decision] .hand', el => document.getElementById(el.getAttribute('aria-describedby')).textContent);
    assert.match(await prompt(), /^What would you play here\? Choose again before you see Trained AI's answer: choose a move below\.$/);
    await page.focus('[data-decision] .hand');
    await page.keyboard.press('Tab');
    await page.waitForFunction(() => document.activeElement?.matches('[data-decision] [data-retry]'), { timeout: 5000 });
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.id === 'answer-title', { timeout: 5000 });
    assert.match(await page.$eval('[data-answer] .verdicts li', el => el.textContent.trim()), /^This time: /);
    // Choose again gives the hand focus, as at a turn, and its prompt says
    // again where the choice is.
    await page.focus('[data-answer] [data-again]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.matches('[data-decision] .hand'), { timeout: 5000 });
    assert.match(await prompt(), /: choose a move below\.$/);
    assert.deepEqual(page.errors, []);
  });

  await cases.check('on a phone the final result, which covers the standings, opens the review, and the review comes back to the standings', async () => {
    for (const [width, height] of [[390, 844], [320, 568], [844, 390]]) {
      // Only the way in and out is looked at here, so Trained AI is out of
      // reach and the review only waits for it.
      const page = await openPage(width, height, { touch: true, network: false });
      const sheet = '.screen [data-review-game]', standing = '.standings [data-review-game]';
      if (width > height) {
        // Held sideways, the result sits beside the standings rather than
        // over them, so it does not repeat their button, and theirs can be
        // tapped once scrolled to.
        assert.notEqual(await page.$eval('.screen', el => getComputedStyle(el).position), 'fixed');
        assert.equal(await page.$eval(sheet, el => getComputedStyle(el).display), 'none');
        await page.$eval(standing, el => el.scrollIntoView({ block: 'center' }));
        assert.ok(await tappable(page, standing), `the standings' button can be tapped at ${width}x${height}`);
        await page.tap(standing);
        await page.waitForSelector('[data-game-review]');
        await page.waitForFunction(() => document.activeElement?.id === 'game-review-title');
        await page.tap('[data-game-review] [data-close-review]');
        await page.waitForFunction(selector => !document.querySelector('[data-game-review]') && document.activeElement?.matches(selector), {}, standing);
        assert.ok(await tappable(page, standing), `the standings' button is in view again at ${width}x${height}`);
        await shot(page, `game-review-phone-standings-${width}x${height}`);
        assert.deepEqual(page.errors, []);
        continue;
      }
      assert.equal(await page.$eval('.screen', el => getComputedStyle(el).position), 'fixed', 'the result is a sheet over the standings');
      assert.equal(await page.$eval(sheet, el => el.innerText.trim()), 'Review game', 'the short label, beside Save hand and Save game');
      assert.ok(await tappable(page, sheet), `the result's own button can be tapped at ${width}x${height}`);
      await page.tap(sheet);
      await page.waitForSelector('[data-game-review]');
      await page.waitForFunction(() => document.activeElement?.id === 'game-review-title');
      await page.tap('[data-game-review] [data-close-review]');
      // The sheet stepped aside, as for the final hand's review, so the
      // standings are what the review comes back to, with focus on their
      // own button, in view and free to tap.
      await page.waitForFunction(selector => !document.querySelector('[data-game-review]') && document.activeElement?.matches(selector), {}, standing);
      assert.equal(await page.$eval('.screen', el => getComputedStyle(el).display), 'none');
      assert.match(await page.$eval('.result-chip', el => el.textContent), /^Show result/);
      assert.ok(await tappable(page, standing), `the standings' button can be tapped at ${width}x${height}`);
      await shot(page, `game-review-phone-standings-${width}x${height}`);
      // A picture of the whole page could leave the page scrolled elsewhere:
      // in Chrome 155 at 320x568 it put the button under the result's chip,
      // and the tap showed the result instead. The picture is of the screen
      // now, and the button is brought back first all the same.
      await page.$eval(standing, el => el.scrollIntoView({ block: 'center' }));
      assert.ok(await tappable(page, standing), `the standings' button can be tapped again at ${width}x${height}`);
      await page.tap(standing);
      await page.waitForSelector('[data-game-review]');
      assert.deepEqual(page.errors, []);
    }
  });

  await cases.check("on a phone, the standings' own button under the result sheet, reached by keyboard, opens the review as the sheet's does and comes back to the standings uncovered, and the result's chip brings the result back with focus", async () => {
    for (const [width, height] of [[390, 844], [320, 568]]) {
      const page = await openPage(width, height, { network: false });
      const standing = '.standings [data-review-game]';
      assert.equal(await page.$eval('.screen', el => getComputedStyle(el).position), 'fixed', 'the result is a sheet over the standings');
      // The sheet takes focus as it appears; the standings' buttons come
      // before it, so Shift+Tab reaches them although the sheet covers them.
      await page.waitForFunction(() => document.activeElement?.matches('.screen'));
      for (let presses = 0; presses < 6 && !await page.evaluate(selector => document.activeElement?.matches(selector), standing); presses++) {
        await page.keyboard.down('Shift');
        await page.keyboard.press('Tab');
        await page.keyboard.up('Shift');
      }
      assert.ok(await page.evaluate(selector => document.activeElement?.matches(selector), standing), `Shift+Tab reaches the standings' button at ${width}x${height}`);
      await page.keyboard.press('Enter');
      await page.waitForFunction(() => document.activeElement?.id === 'game-review-title');
      await page.keyboard.press('Tab');
      assert.ok(await page.evaluate(() => document.activeElement?.matches('[data-game-review] [data-close-review]')), 'Back to the standings follows the heading');
      await page.keyboard.press('Enter');
      // The same as from the sheet's own button: the sheet has stepped aside
      // for the review, so the standings' button that took focus back is in
      // view and uncovered, and the result is a tap away on its chip.
      await page.waitForFunction(selector => !document.querySelector('[data-game-review]') && document.activeElement?.matches(selector), {}, standing);
      assert.equal(await page.$eval('.screen', el => getComputedStyle(el).display), 'none', `the sheet stepped aside at ${width}x${height}`);
      assert.match(await page.$eval('.result-chip', el => el.textContent), /^Show result/);
      assert.ok(await tappable(page, standing), `the standings' button is uncovered at ${width}x${height}`);
      await shot(page, `game-review-phone-keyboard-${width}x${height}`);
      // The chip, which follows the standings' buttons, brings the result
      // back by keyboard too. It goes as it does, so focus goes to the
      // result, as when the result first appeared, and not to the page.
      const onChip = () => page.evaluate(() => document.activeElement?.matches('.result-chip'));
      for (let presses = 0; presses < 6 && !await onChip(); presses++) await page.keyboard.press('Tab');
      assert.ok(await onChip(), `Tab reaches the result's chip at ${width}x${height}`);
      await page.keyboard.press('Enter');
      await page.waitForFunction(() => getComputedStyle(document.querySelector('.screen')).display !== 'none', { timeout: 5000 });
      await page.waitForFunction(() => document.activeElement?.matches('.screen'), { timeout: 5000 }).catch(() => {});
      assert.equal(await page.evaluate(() => {
        const active = document.activeElement;
        return active?.matches('.screen') ? 'the result' : !active || active === document.body ? 'the page' : active.outerHTML.slice(0, 80);
      }), 'the result', `the chip hands focus to the result at ${width}x${height}`);
      assert.deepEqual(page.errors, []);
    }
  });

  await cases.check('after a hand, its result opens the review of the game so far, which asks Trained AI about that hand first and leads with it, keeps the game out of the trend, and comes back to the result, by keyboard and pointer at desktop and phone sizes', async () => {
    const button = '.screen [data-review-game]', latest = hands[2], where = page => `at ${page.viewport().width}x${page.viewport().height}`;
    const focused = (page, selector) => page.evaluate(selector => Boolean(document.activeElement?.matches(selector)), selector);
    /** Back from the review, which goes, to the result's button, which has
     * focus again, with the table and the result shown as they were. */
    const backOnResult = page => page.waitForFunction(selector => !document.querySelector('[data-game-review]')
      && document.activeElement?.matches(selector), { timeout: 5000 }, button);
    // A game reviewed before this one, for the trend to say when this one
    // joins it.
    let page = await openPage(1100, 900, { saved: midway, ready: button,
      trend: [{ id: 'earlier', network: MANIFEST.version, at: Date.UTC(2026, 9, 1, 12), matched: 60, decisions: 100, flagged: 4 }] });
    // The result takes focus as it appears; its button is pressed after it,
    // by keyboard.
    await page.waitForFunction(() => document.activeElement?.matches('.screen'));
    assert.equal(await page.$eval(button, el => el.textContent.trim()), 'Review game so far');
    await page.focus(button);
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    assert.equal(await page.$eval('#game-review-title', el => el.textContent), 'Your game so far, looked at again');
    assert.equal(await page.$eval('.match-view', el => getComputedStyle(el).display), 'none', 'the table and the result step aside');
    assert.equal(await page.$eval('[data-game-review] [data-close-review]', el => el.textContent.trim()), 'Back to the result');
    // It leads with the hand just played.
    assert.equal(await page.$eval('[data-game-review] [data-latest-hand]', el => Number(el.dataset.latestHand)), latest.index);
    assert.equal(await page.$eval('[data-game-review] [data-latest-hand] .name', el => el.textContent), latest.name);
    await reviewed(page);

    // Every batch: the hand just played first, then the others from the
    // first deal, each group once.
    const batches = await page.evaluate(() => window.reviewBatches);
    assert.equal(walkBatches(batches, [latest.index, 0, 1]).walked, batches.length, 'no batch is missing or extra');
    const total = hands.slice(0, latest.index + 1).reduce((sum, hand) => sum + hand.frames.length, 0);
    assert.equal(await page.$eval('[data-game-review] .headline', el => el.textContent),
      `Every hand so far is reviewed: ${total} decision${total === 1 ? '' : 's'}.`);
    // The game so far is not a game for the trend, which says when it joins.
    const trend = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), TREND_KEY);
    assert.deepEqual(trend.games.map(game => game.id), ['earlier'], 'the game so far is not in the trend');
    assert.match(await page.$eval('[data-trend]', el => el.textContent), /This game joins your trend once the game is over and every hand is reviewed\./);
    await shot(page, 'game-review-so-far');

    // The hand just played, replayed whole from its own row by keyboard,
    // with every hand face up, and back to the row's button.
    await page.focus('[data-game-review] [data-replay-latest]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.id === 'decision-title', { timeout: 5000 });
    assert.equal(await page.$eval('#decision-title', el => el.textContent), `${latest.name}, decision by decision`);
    await overviewIsRight(page, latest.frames[0], 0, latest.frames.length);
    await page.focus('[data-decision] [data-back]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.matches('[data-game-review] [data-replay-latest]'), { timeout: 5000 });

    // Back to the result as it was, with focus on the button that opened the
    // review, and the save untouched.
    await page.focus('[data-game-review] [data-close-review]');
    await page.keyboard.press('Enter');
    await backOnResult(page);
    assert.notEqual(await page.$eval('.match-view', el => getComputedStyle(el).display), 'none', 'the table and the result are back');
    assert.ok(await tappable(page, button), 'the button is in view, uncovered');
    const { _storage, ...save } = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
    assert.deepEqual(save, midway, 'the save is unchanged');
    assert.deepEqual(page.errors, []);
    await closePage(page);

    // On a phone the result is a sheet over the table, its button "Review
    // game", and the review comes back to the sheet, up as it was left, with
    // no chip and focus on that button. Only the way in and out is looked at
    // here, so Trained AI is out of reach and the review only waits for it.
    for (const [width, height] of [[390, 844], [320, 568]]) {
      page = await openPage(width, height, { touch: true, network: false, saved: midway, ready: button });
      assert.equal(await page.$eval('.screen', el => getComputedStyle(el).position), 'fixed', `the result is a sheet ${where(page)}`);
      assert.equal(await page.$eval(button, el => el.innerText.trim()), 'Review game', `the short label ${where(page)}`);
      assert.ok(await tappable(page, button), `the sheet's button can be tapped ${where(page)}`);
      await page.tap(button);
      await page.waitForSelector('[data-game-review]');
      await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
      await page.tap('[data-game-review] [data-close-review]');
      await backOnResult(page);
      assert.notEqual(await page.$eval('.screen', el => getComputedStyle(el).display), 'none', `the sheet is up again ${where(page)}`);
      assert.equal(await page.$('.result-chip'), null, `with no chip ${where(page)}`);
      assert.ok(await tappable(page, button), `the sheet's button is uncovered ${where(page)}`);
      await shot(page, `game-review-so-far-${width}x${height}`);
      assert.deepEqual(page.errors, []);
      await closePage(page);
    }

    // By keyboard on a phone: the sheet has focus as it appears, Tab reaches
    // its button, and Back to the result follows the review's heading.
    page = await openPage(390, 844, { network: false, saved: midway, ready: button });
    await page.waitForFunction(() => document.activeElement?.matches('.screen'));
    for (let presses = 0; presses < 20 && !await focused(page, button); presses++) await page.keyboard.press('Tab');
    assert.ok(await focused(page, button), "Tab reaches the sheet's button at 390x844");
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    await page.keyboard.press('Tab');
    assert.ok(await focused(page, '[data-game-review] [data-close-review]'), 'Back to the result follows the heading');
    await page.keyboard.press('Enter');
    await backOnResult(page);
    assert.notEqual(await page.$eval('.screen', el => getComputedStyle(el).display), 'none', 'the sheet is up again at 390x844');
    assert.ok(await tappable(page, button), "the sheet's button is uncovered at 390x844");
    assert.deepEqual(page.errors, []);
    await closePage(page);

    // Held sideways the result sits beside the table rather than over it,
    // and offers the review all the same: unlike the final hand's buttons
    // for the whole game, which the standings beside it repeat, nothing else
    // offers this one.
    page = await openPage(844, 390, { touch: true, network: false, saved: midway, ready: button });
    assert.notEqual(await page.$eval('.screen', el => getComputedStyle(el).position), 'fixed');
    assert.notEqual(await page.$eval(button, el => getComputedStyle(el).display), 'none', 'the button is shown at 844x390');
    await page.$eval(button, el => el.scrollIntoView({ block: 'center' }));
    assert.ok(await tappable(page, button), 'the button can be tapped at 844x390');
    await page.tap(button);
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    await page.tap('[data-game-review] [data-close-review]');
    await backOnResult(page);
    assert.ok(await tappable(page, button), 'the button is in view again at 844x390');
    await shot(page, 'game-review-so-far-844x390');
    assert.deepEqual(page.errors, []);
  });

  await cases.check("a review of the game so far goes on into the whole game's: Next hand stops it, the hands it reviewed are not asked again, and the game joins the trend once, as it ends", async () => {
    const button = '.screen [data-review-game]', standing = '.standings [data-review-game]';
    const id = `${snapshot.seed}:${snapshot.commands.length}`;
    const trendIds = page => page.evaluate(key => (JSON.parse(localStorage.getItem(key)) ?? { games: [] }).games.map(game => game.id), TREND_KEY);
    const sent = page => page.evaluate(() => window.reviewBatches.length);
    /** Back to the last hand's result by keyboard, and its Next hand, which
     * ends the game and hands focus on to the standings' heading. */
    async function backAndOn(page) {
      await page.focus('[data-game-review] [data-close-review]');
      await page.keyboard.press('Enter');
      await page.waitForFunction(selector => !document.querySelector('[data-game-review]') && document.activeElement?.matches(selector),
        { timeout: 5000 }, button);
      await page.focus('.screen .primary');
      await page.keyboard.press('Enter');
      await page.waitForFunction(() => document.activeElement?.matches('.standings h2'), { timeout: 5000 });
    }

    // Each batch reaches the worker a second late, so that the review is
    // still under way at Next hand, however fast the machine, and it goes
    // through the hands unheld, as a player who asks for all of them has it.
    let page = await openPage(1100, 900, { saved: lastResult, ready: button, answerDelay: 1000 });
    await page.waitForFunction(() => document.activeElement?.matches('.screen'));
    await reviewWholeGame(page, button);
    // The hand just played is asked about first: once it is reviewed, its
    // row offers its replay while the hands before it are reviewed.
    await page.waitForSelector('[data-game-review] [data-replay-latest]', { timeout: 300000 });
    assert.ok(await page.$('[data-game-review] [data-stop-review]'), 'the review goes on to the earlier hands');
    await backAndOn(page);
    // The deal that ended the game stopped the review, which asks nothing
    // more until it is opened again.
    const stopped = await sent(page);
    await new Promise(done => setTimeout(done, 2500));
    assert.equal(await sent(page), stopped, 'no batch is sent while the review is shut');
    assert.deepEqual(await trendIds(page), [], 'a game not reviewed whole is not in the trend');
    await page.click(standing);
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    assert.equal(await page.$eval('#game-review-title', el => el.textContent), 'Your game, looked at again');
    assert.equal(await page.$('[data-game-review] [data-latest-hand]'), null, 'a finished game leads with no hand');
    // The hands reviewed before Next hand, read before the review could
    // finish another, a batch taking a second to reach the worker: none of
    // them is asked about again, and every other hand is, once, from the
    // first deal.
    const done = await page.$$eval('[data-game-review] [data-replay-hand]', buttons => buttons.map(entry => Number(entry.dataset.replayHand)));
    assert.ok(done.includes(hands.at(-1).index), `the last hand was reviewed before the game ended: ${done.join(', ')}`);
    await reviewed(page);
    const after = (await page.evaluate(() => window.reviewBatches)).slice(stopped);
    const rest = hands.map(hand => hand.index).filter(index => !done.includes(index));
    assert.equal(walkBatches(after, rest).walked, after.length, `only hands ${rest.join(', ')} are asked about, each once`);
    assert.deepEqual(await trendIds(page), [id], 'the game joins the trend once, whole');
    assert.equal(await page.$eval('[data-game-review] [data-close-review]', el => el.textContent.trim()), 'Back to the standings');
    await page.click('[data-game-review] [data-close-review]');
    await page.waitForFunction(selector => !document.querySelector('[data-game-review]') && document.activeElement?.matches(selector),
      { timeout: 5000 }, standing);
    assert.deepEqual(page.errors, []);
    await closePage(page);

    // Every hand reviewed at the last hand's result: the game joins the
    // trend as Next hand ends it, before any review is opened, and the
    // standings' review asks about nothing again.
    page = await openPage(1100, 900, { saved: lastResult, ready: button });
    await page.waitForFunction(() => document.activeElement?.matches('.screen'));
    await page.click(button);
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    await reviewed(page);
    assert.deepEqual(await trendIds(page), [], 'the game so far is not in the trend');
    await backAndOn(page);
    assert.deepEqual(await trendIds(page), [id], 'the game joins the trend as it ends');
    const asked = await sent(page);
    await page.click(standing);
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    assert.match(await page.$eval('[data-game-review] .headline', el => el.textContent), /^Every hand is reviewed: \d+ decisions?\.$/);
    await new Promise(done => setTimeout(done, 500));
    assert.equal(await sent(page), asked, 'nothing is asked again');
    assert.deepEqual(await trendIds(page), [id], 'and the game is in the trend once');
    assert.deepEqual(page.errors, []);
  });

  await cases.check("a review of the game so far goes on at the next hand's result: Next hand stops it as it runs, and that result's review asks about the hand just played first, then only about the hands left waiting, none it had reviewed", async () => {
    const button = '.screen [data-review-game]', sent = page => page.evaluate(() => window.reviewBatches.length);
    // The hand played in the page: the first after the third that asks
    // Trained AI about a decision, so that batches of its own lead what its
    // result's review sends. The review is opened at the result of the hand
    // before it.
    const next = hands.find(hand => hand.index > 2 && facts[hand.index].some(fact => !fact.forced));
    assert.ok(next, 'a hand after the third asks Trained AI about a decision');
    // Each batch reaches the worker a second late, so that the review is
    // still under way at Next hand, however fast the machine, and it goes
    // through the hands unheld, as a player who asks for all of them has it.
    const page = await openPage(1100, 900, { saved: savedAfter(snapshot, hands[next.index - 1].end), ready: button, answerDelay: 1000 });
    await page.waitForFunction(() => document.activeElement?.matches('.screen'));
    await reviewWholeGame(page, button);
    // A hand is reviewed, and the review goes on to the others.
    await page.waitForSelector('[data-game-review] [data-replay-hand]', { timeout: 300000 });
    assert.ok(await page.$('[data-game-review] [data-stop-review]'), 'the review goes on to the other hands');
    // Back to the result by keyboard, and its Next hand while the review
    // runs, which stops it.
    await page.focus('[data-game-review] [data-close-review]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(selector => !document.querySelector('[data-game-review]') && document.activeElement?.matches(selector),
      { timeout: 5000 }, button);
    await page.focus('.screen .primary');
    await page.keyboard.press('Enter');
    const stopped = await sent(page);
    // The next hand, played in the page as the game played it: nothing is
    // asked meanwhile, and the match is saved at its result as the game was.
    await playHand(page, next);
    assert.equal(await sent(page), stopped, 'no batch is sent while the review is shut');
    const { _storage, ...save } = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
    assert.deepEqual(save, savedAfter(snapshot, next.end), 'the hand was played as the game played it');
    // That hand's result opens the review again, which leads with that hand.
    // The hands reviewed before Next hand are read before the review could
    // finish another, a batch taking a second to reach the worker.
    await page.click(button);
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    assert.equal(await page.$eval('[data-game-review] [data-latest-hand]', el => Number(el.dataset.latestHand)), next.index,
      'the review leads with the hand just played');
    assert.equal(await page.$eval('[data-game-review] [data-latest-hand] .name', el => el.textContent), next.name);
    const done = await page.$$eval('[data-game-review] [data-replay-hand]', buttons => buttons.map(entry => Number(entry.dataset.replayHand)));
    assert.ok(done.length > 0 && !done.includes(next.index), `hands reviewed before Next hand, and not the hand played since: ${done.join(', ')}`);
    await reviewed(page);
    // Every batch sent since walks the hand just played first, then the
    // hands left waiting from the first deal, each group once, asked from its
    // decisions' own positions: none of a hand reviewed before Next hand is
    // sent again.
    const after = (await page.evaluate(() => window.reviewBatches)).slice(stopped);
    const rest = [next.index, ...hands.slice(0, next.index).map(hand => hand.index).filter(index => !done.includes(index))];
    assert.equal(walkBatches(after, rest).walked, after.length, `only hands ${rest.join(', ')} are asked about, in that order, each once`);
    assert.deepEqual(page.errors, []);
  });
} finally {
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
  await cases.report('whole-game review checks', resolve(output, 'game-review.json'));
}
