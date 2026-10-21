/** The game's review on the production build, with the trained worker: one
 * screen, a step bar over the decision on show, and the list of hands and
 * decisions beside it on a wide screen, or folded above it.
 *
 * A finished match is restored from its save and reviewed hand by hand by the
 * shipped network, its final hand first, the hand the review opens on. Every
 * batch the page sends is captured with the worker's answer, and the flags and
 * the agreement figure are worked out again here from those answers alone:
 * the list's flagged lines must be those decisions, hand by hand in the order
 * played, each naming both moves, and the step bar must give both moves with
 * the weights the worker gave them, Trained AI's tile tagged and ringed in the
 * hand and its view tagged too. Previous and Next step through the flagged
 * decisions across hands by keyboard, keeping focus, handing it on at either
 * end, the list keeping the line of the decision on show in view; Esc goes
 * back to the standings, focus on the button that opened the review; the
 * saved match must be exactly as it was, and the game must enter the trend
 * once, under the version of Trained AI that judged it.
 *
 * A hand is walked under All, by the arrow key and by Next, and at every step
 * every player's tiles must be the ones the engine held, face up, each hand a
 * list named by its seat, with the board as it stood and nothing offering to
 * hide them, at desktop, tablet and phone sizes: from tablet width up the
 * dense table's opponents in a row under its centre and the player's tiles in
 * one row, an opponent's tiles on one row or on two where one would draw them
 * under 18 pixels wide, the player's seat first on a phone held upright, the
 * step bar on the screen, flush under the game's bar on a phone, as soon as
 * the screen changes size and at the step pressed, and the player's tiles,
 * scrolled to, clear of it; no focus lands under the bar, tabbing through the
 * decision or stepping by the list, nor moves the page as it lands on the
 * bar's own buttons; and where the open list's box runs up under the bars,
 * Previous brings each line it steps to out from under them.
 * The filter keeps the flagged decisions, those where Trained AI differed or
 * every one, saying how many each holds; Next and Previous visit only what it
 * keeps, a decision it leaves out stays shown, dim, and the filter is kept
 * across a reload. A list the player opens or folds stays as they left it:
 * the folded list, which shows the line on show as it opens, as steps are
 * pressed by touch and by keyboard while later hands are reviewed, focus
 * staying on the line pressed; a slow device's list, which opens by itself
 * once, with a Review in each header still to review; a hand's events across
 * a step with none and across hands, and the trend. On a wide screen the list
 * is a column of its own on the left, after the decision in the page, which
 * stays on the screen and keeps still what the player is looking at, the
 * page not moving: a line pressed while later hands are reviewed, a filter
 * pressed at its top, and an earlier hand scrolled to while a later one is
 * reviewed. The review's buttons hand focus on as they go, and Next, waiting
 * for the answers of a hand the player went to, keeps its focus and then goes
 * on. A flagged answer to another's discard, from a game whose player calls
 * everything, says in the step bar whose tile it answers, which that seat's
 * row outlines; with Trained AI's move kept back, it is answered by a call or
 * Pass picked by keyboard, kept as Now. The review's two settings, "Hide
 * Trained AI's move" and "Hide other players' tiles", are the same in Options
 * and in the review's header, kept across a reload, and on a phone in the
 * settings alone. With both on, a decision keeps the move and the other
 * players' tiles back, every legal discard a tile to press and every other
 * move a button under the hand, until Show answer, s, a digit or a tile
 * pressed reveals it, a pick kept as Now and the racks turning over where
 * they lay, focus going on to Next as the control used goes; on a phone its
 * live hand is tabbed through clear of the bars; each reveal stays through
 * steps, Back and opening again, and is gone after a reload; with the tiles
 * alone kept back, Show tiles turns them over. The keys step, change hand,
 * show what a decision keeps back, move along and pick from its live hand, 0
 * the tile just drawn, and go back, and do nothing under a dialog, in a
 * field, with a modifier held or with the shortcuts off. On a phone, where
 * the final result covers the standings, the review must be reachable from
 * the result itself by touch and come back to the standings; held sideways,
 * from the standings beside the result; and from the standings' own button
 * under the result, by keyboard, it must leave the page as the result's own
 * button does, with the result's chip bringing it back, focus and all.
 *
 * Before the end, any hand's result opens the review of the game so far,
 * restored from a save waiting at that result. It must open on the hand just
 * played, at its first decision, and ask Trained AI about it first, then about
 * the rest from the first deal, every group once; keep the game out of the
 * trend; come back to the result as it was, with focus on the button that
 * opened it, by keyboard and pointer at desktop and phone sizes, where the
 * sheet must be up again with no chip, and held sideways, where the button
 * shows beside the table; and open again where the player left it. Next hand
 * on the last hand stops it, and the standings' review must then open where
 * the player left it and ask about no hand it reviewed before; a game every
 * hand of which was reviewed at the last result must join the trend as Next
 * hand ends it, once. Next hand on an earlier hand, pressed as the review
 * runs, stops it too: nothing is asked while the next hand is played in the
 * page, and that hand's result must open the review on it, asking about it
 * first and then only about the hands left waiting, none it had reviewed.
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
import { BATCH, FILTER_LABELS, FLAG_PLAYED_BELOW, FLAG_PREFERRED_FROM, filterSaid, moveToken, scanMatch, tableSaid } from '../src/lib/game-review.js';
import { OPPONENT_LABELS, OPPONENT_POSITIONS } from '../src/lib/opponents.js';
import { SEAT_NAMES, tileWords } from '../src/lib/tiles.js';
import { callLabel, quadOnOffer } from '../src/lib/ui.js';
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
const hands = scanMatch(Game, snapshot), callingHands = scanMatch(Game, callingGame);
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
 * before. `settings` is the preferences the page first loads with, Beginner
 * opponents and hints unless given. `ready` is what the page is waited for
 * by: the standings' review, or for a match saved at a hand's result, that
 * result's. */
async function openPage(width, height, { touch = false, network = true, saved = snapshot, answerDelay = 0, trend = null,
  settings = { version: 2, difficulty: 'beginner', hints: true }, ready = '.standings [data-review-game]' } = {}) {
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
  await page.evaluateOnNewDocument((saveKey, settingsKey, saved, answerDelay, trendKey, trend, settings) => {
    if (!localStorage.getItem(saveKey)) localStorage.setItem(saveKey, JSON.stringify(saved));
    if (!localStorage.getItem(settingsKey)) localStorage.setItem(settingsKey, JSON.stringify(settings));
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
  }, SAVE_KEY, SETTINGS_KEY, saved, answerDelay, TREND_KEY, trend, settings);
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/`, { waitUntil: 'networkidle0' });
  await page.waitForSelector(ready);
  return page;
}

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

/** Where the step bar says a step of a hand is: "East 4 · decision 12 of 15". */
const placeOf = (hand, step) => `${hand.name} · decision ${step + 1} of ${hand.frames.length}`;
/** Where the step bar says the review is. */
const stepText = page => page.$eval('[data-game-review] [data-step]', el => el.textContent);
/** Waits until the step bar says the review is at `step` of `hand`. */
const onPlace = (page, hand, step) => page.waitForFunction(text => document.querySelector('[data-game-review] [data-step]')?.textContent === text,
  { timeout: 5000 }, placeOf(hand, step));
/** What has keyboard focus, as a failure says it. */
const focusedName = page => page.evaluate(() => {
  const active = document.activeElement;
  return !active || active === document.body ? 'the page' : active.getAttribute('aria-label') ?? active.textContent.trim().slice(0, 60);
});
/** Whether keyboard focus is on what `selector` names. */
const focusOn = (page, selector) => page.evaluate(selector => Boolean(document.activeElement?.matches(selector)), selector);
/** Brings what `selector` names to the middle of the screen, clear of the
 * bars that stay at its top, and clicks it. */
async function press(page, selector) {
  await page.$eval(selector, element => element.scrollIntoView({ block: 'center' }));
  await page.click(selector);
}
/** Opens the list of hands and decisions, by pointer, where it is folded
 * above the decision. */
async function openList(page) {
  if (!await page.$('details[data-review-index]:not([open])')) return;
  await press(page, 'details[data-review-index] > summary');
  await page.waitForFunction(() => document.querySelector('details[data-review-index]').open, { timeout: 5000 });
  await settle(page);
}
/** A decision's line in the list, by its key, "hand:step". */
const line = key => `[data-review-index] [data-step-to="${key}"]`;
/** Goes to a decision by its line in the list, by pointer: of the
 * tile-throwing game unless `game` is another game's hands. */
async function stepTo(page, key, game = hands) {
  await openList(page);
  await press(page, line(key));
  const [hand, step] = key.split(':').map(Number);
  await onPlace(page, game[hand], step);
}
/** Whether the list's line of the decision on show is in view, as the list
 * keeps it: in the box the list scrolls in, the column on a wide screen or the
 * open list's box of hands, and in the part of that box the player can see,
 * on the screen and below the bars stuck over the top of the screen there:
 * the game's on a phone, and the step bar where it is stuck. Where that part
 * is too short to hold the line with room around it, the line need only be in
 * it; and where the box is scrolled as far as it goes towards the line, the
 * list has done all its own scrolling can, as for the last line of a box
 * that runs on below the screen's foot: only the page could show more, which
 * the list never moves. True where no box scrolls the line; otherwise where
 * the line is. */
const lineInView = page => page.evaluate(() => {
  const item = document.querySelector('[data-review-index] [aria-current="step"]');
  if (!item) return 'no line on show';
  let box = item.parentElement;
  while (box && box !== document.body && !(/auto|scroll/.test(getComputedStyle(box).overflowY) && box.scrollHeight > box.clientHeight)) box = box.parentElement;
  if (!box || box === document.body || !box.getClientRects().length) return true;
  const edge = box.getBoundingClientRect(), at = item.getBoundingClientRect();
  // Each bar stuck where the one before it ends, from the top of the screen,
  // that runs over the box.
  let covered = 0;
  for (const bar of [document.querySelector('header.bar'), document.querySelector('[data-game-review] [data-steps]')]) {
    const rect = bar?.getBoundingClientRect();
    if (rect && getComputedStyle(bar).position === 'sticky' && rect.top <= covered + 1 && rect.left < edge.right && rect.right > edge.left) {
      covered = Math.max(covered, rect.bottom);
    }
  }
  const onScreen = edge.bottom > covered && edge.top < innerHeight;
  const top = onScreen ? Math.max(edge.top, covered) : edge.top, bottom = onScreen ? Math.min(edge.bottom, innerHeight) : edge.bottom;
  const inside = at.top >= top - 0.5 && at.bottom <= bottom + 0.5;
  const crowded = bottom - top < at.height + 16 && at.top < bottom && at.bottom > top;
  const spent = (at.bottom > bottom && box.scrollTop >= box.scrollHeight - box.clientHeight - 1) || (at.top < top && box.scrollTop <= 0);
  return inside || crowded || spent ? true
    : `${Math.round(at.top)} to ${Math.round(at.bottom)} in ${Math.round(top)} to ${Math.round(bottom)}, scrolled ${Math.round(box.scrollTop)} of ${box.scrollHeight - box.clientHeight}`;
});
/** The filter's button for one filter. */
const option = name => `[data-game-review] [data-filter-option="${name}"]`;
/** Chooses a filter by its button, by pointer. */
async function choose(page, name) {
  await openList(page);
  await press(page, option(name));
  await page.waitForFunction(selector => document.querySelector(selector)?.getAttribute('aria-pressed') === 'true', { timeout: 5000 }, option(name));
}

/** The decision's overview as the page draws it: each opponent's hand, a list
 * named by its seat, and the player's own, with the names of their tiles,
 * the narrowest tile, how many rows the tiles take and how many are on the
 * first, whether every tile is on the screen and whether the list spills out
 * of its seat; whether anything is face down or offers to show or hide the
 * hands; and the step bar's place and what it says of the step, the wall and
 * what each seat was doing. */
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
  const board = box(overview.querySelector('.board')), mine = box(overview.querySelector('.mine'));
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
    // The table's layout: whether the screen takes the dense one (761 pixels
    // wide or more, over 500 high), where each opponent's seat begins and the
    // centre ends, whether the player's seat comes first, as on a phone held
    // upright, and the height of the table and the player's seat together.
    dense: innerWidth >= 761 && innerHeight > 500,
    seatTops: [...overview.querySelectorAll('.board > .place')].map(place => Math.round(box(place).top)),
    centreBottom: Math.round(box(overview.querySelector('.board > .centre')).bottom),
    seatFirst: mine.top < board.top,
    height: Math.round(Math.max(board.bottom, mine.bottom) - Math.min(board.top, mine.top)),
    step: document.querySelector('[data-game-review] [data-step]').textContent,
    said: document.querySelector('[data-game-review] [data-step-said]').textContent,
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
async function overviewIsRight(page, hand, step) {
  const frame = hand.frames[step], shown = await overviewAt(page), seats = frame.view.seats;
  assert.equal(shown.step, placeOf(hand, step));
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
  // From 761 pixels wide at every height over 500 the table is the dense
  // one: the centre a bar across the top, the three opponents in one row
  // under it.
  if (shown.dense) {
    assert.equal(new Set(shown.seatTops).size, 1, `the opponents in one row: ${shown.seatTops.join(', ')}`);
    assert.ok(shown.centreBottom <= Math.min(...shown.seatTops), `the centre above them: ${shown.centreBottom}, ${shown.seatTops.join(', ')}`);
    // The player's tiles keep one row too, drawn as large as the stage has
    // room for, beside the list on a wide screen (DecisionReplay).
    assert.equal(shown.own.rows, 1, `the player's tiles in one row, ${shown.own.tileWidth} pixels wide`);
  }
  // The player's seat comes first on a phone held upright, where it is the
  // first thing under the step bar, and after the table everywhere else.
  const { width, height } = page.viewport();
  assert.equal(shown.seatFirst, width <= 760 && height >= width, `the player's seat ${shown.seatFirst ? 'before' : 'after'} the table at ${width}x${height}`);
  // The step, the table in a sentence and both moves, and whether Trained AI
  // flagged the move there, in the one line said as the step changes.
  const lead = `${hand.name}, decision ${step + 1} of ${hand.frames.length}. ${tableSaid(frame.view)} You: `;
  assert.ok(shown.said.startsWith(lead), `${shown.said} begins ${lead}`);
  assert.match(shown.said.slice(lead.length), /\. (Trained AI agreed|Only move|Trained AI: .+)\.( Flagged\.)?$/, shown.said);
  assert.equal(shown.wall, frame.view.wall);
  assert.ok(shown.notes.includes(frame.view.phase === 'act' ? 'To play' : 'To answer'), `whose move it was: ${shown.notes.join(', ')}`);
  return shown;
}

/** Waits until the page has stopped moving: the scroll, where the stage and
 * the player's tiles lie and how tall the page is, the same over three
 * frames running. A change of the screen's size takes the page and the
 * browser a few frames to settle, the browser's own scroll that keeps what
 * was in view in view (scroll anchoring) among them. Gives up after two
 * seconds, leaving what is measured next to say what moved. */
const steady = page => page.evaluate(() => new Promise(done => {
  const read = () => [scrollY, document.documentElement.scrollHeight,
    ...['[data-game-review] [data-stage]', '[data-decision] .mine .hand .tile'].map(selector => Math.round(document.querySelector(selector)?.getBoundingClientRect().top ?? 0))].join();
  let last = read(), same = 0, frames = 0;
  const frame = () => {
    const now = read();
    same = now === last ? same + 1 : 0;
    last = now;
    if (same >= 3 || ++frames >= 120) done();
    else requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}));

/** Checks that the stage fits the screen, the player's tiles scrolled to as
 * a focused control is: the step bar on the screen, stuck flush under the
 * game's bar where that stays at the top of a phone's screen, or at the top
 * of the screen, once the page is scrolled past it, and the tiles clear of
 * it; where the step bar stays in its place, the tiles clear of the game's
 * bar. Says what was measured, once the page has been drawn after the
 * scroll, as the player sees it.
 *
 * The page is let settle first, and the tiles scrolled to again once it has:
 * at 390 by 844, just after the screen changed size, CI found the page 374
 * pixels on from where the scroll had put it a few frames before, as if the
 * browser's own scroll for the change of size had come after the check's.
 * Where the page moves after the first scroll, the log says how far; what
 * the first scroll found is said too, should the second fail. */
async function stageFits(page) {
  const tiles = '[data-decision] .mine .hand .tile';
  const where = () => page.evaluate(selector => ({ scrolled: Math.round(scrollY),
    tile: Math.round(document.querySelector(selector).getBoundingClientRect().top),
    stage: Math.round(document.querySelector('[data-game-review] [data-stage]').getBoundingClientRect().top) }), tiles);
  await steady(page);
  await page.$eval(tiles, tile => tile.scrollIntoView({ block: 'start' }));
  const first = await where();
  await settle(page);
  await steady(page);
  const then = await where();
  await page.$eval(tiles, tile => tile.scrollIntoView({ block: 'start' }));
  await settle(page);
  const shown = await page.evaluate(() => {
    const tile = document.querySelector('[data-decision] .mine .hand .tile');
    const bar = document.querySelector('header.bar'), steps = document.querySelector('[data-game-review] [data-steps]');
    const barSticks = getComputedStyle(bar).position === 'sticky', edge = steps.getBoundingClientRect();
    return { sticks: getComputedStyle(steps).position === 'sticky', top: barSticks ? Math.round(bar.getBoundingClientRect().bottom) : 0,
      stage: Math.round(document.querySelector('[data-game-review] [data-stage]').getBoundingClientRect().top),
      steps: { top: Math.round(edge.top), bottom: Math.round(edge.bottom) }, tile: Math.round(tile.getBoundingClientRect().top), height: innerHeight,
      scrolled: Math.round(scrollY) };
  });
  const size = `${page.viewport().width}x${page.viewport().height}`, said = `${JSON.stringify(shown)}; the first scroll ${JSON.stringify(first)}, `
    + `settled ${JSON.stringify(then)}`;
  if (then.tile !== first.tile || then.scrolled !== first.scrolled) console.log(`the page moved after the first scroll to the tiles at ${size}: ${said}`);
  if (shown.sticks) {
    assert.ok(shown.steps.top >= shown.top - 1 && shown.steps.bottom <= shown.height, `the step bar on the screen at ${size}: ${said}`);
    if (shown.stage < shown.top) assert.ok(Math.abs(shown.steps.top - shown.top) <= 1, `the step bar stuck flush at the top at ${size}: ${said}`);
    assert.ok(shown.tile >= shown.steps.bottom, `the player's tiles clear of the step bar at ${size}: ${said}`);
  } else {
    assert.ok(shown.tile >= shown.top, `the player's tiles clear of the game's bar at ${size}: ${said}`);
  }
  return shown;
}

/** Tabs on from the step bar's last button until focus leaves the stage,
 * checking that each control it lands on is on the screen and clear of the
 * step bar. Says how many it landed on. */
async function tabsClear(page) {
  const size = `${page.viewport().width}x${page.viewport().height}`;
  await page.evaluate(() => [...document.querySelectorAll('[data-game-review] [data-steps] button:not(:disabled)')].at(-1).focus());
  let landed = 0;
  for (let presses = 0; presses < 200; presses++) {
    await page.keyboard.press('Tab');
    await settle(page);
    const at = await page.evaluate(() => {
      const active = document.activeElement;
      if (!document.querySelector('[data-game-review] [data-stage]').contains(active)) return null;
      const box = active.getBoundingClientRect(), steps = document.querySelector('[data-game-review] [data-steps]').getBoundingClientRect();
      return { name: active.getAttribute('aria-label') ?? active.textContent.trim().slice(0, 40), top: box.top, bottom: box.bottom,
        edge: steps.bottom, height: innerHeight };
    });
    if (!at) break;
    landed++;
    assert.ok(at.top >= at.edge - 0.5, `${at.name} at ${at.top}, under the step bar to ${at.edge}, at ${size}`);
    assert.ok(at.bottom <= at.height + 0.5, `${at.name} below the screen at ${size}`);
  }
  assert.ok(landed > 0, `Tab goes on into the stage at ${size}`);
  return landed;
}

/** What the decision on show keeps back, as the page shows it: what the step
 * bar's reveal says, if it has one; whether Trained AI's view is there; each
 * opponent's rack, by its name, how many of its tiles are face down, whether
 * any tile names itself in an attribute, and where the rack lies in the table
 * and how large it is; how tall the table is and where its top lies on the
 * screen; in the player's hand, the tiles ringed, whether a tile is tagged AI,
 * which tile is tagged Now, the tiles a press picks and the moves under the
 * hand, by name, and what the hand is described by; the step bar's line and
 * the step's status; and the list's line of the decision, by name. */
const shownAt = page => page.evaluate(() => {
  const decision = document.querySelector('[data-decision]'), board = decision.querySelector('.board').getBoundingClientRect();
  const tags = [...decision.querySelectorAll('.mine .hand .hand-tile')].map(tile => tile.querySelector('.tag')?.textContent ?? '');
  return {
    reveal: document.querySelector('[data-game-review] [data-reveal]')?.textContent.trim() ?? null,
    view: Boolean(decision.querySelector('[data-view]')),
    racks: [...decision.querySelectorAll('ul.revealed')].map(list => {
      const box = list.getBoundingClientRect();
      return { label: list.getAttribute('aria-label'), tiles: list.querySelectorAll('.tile').length,
        down: list.querySelectorAll('.tile[aria-label="face-down tile"]').length, named: list.querySelectorAll('[data-tile]').length,
        names: [...list.querySelectorAll('.tile')].map(tile => tile.getAttribute('aria-label')),
        box: [box.top - board.top, box.left - board.left, box.width, box.height].map(value => Math.round(value * 2) / 2) };
    }),
    board: Math.round(board.height),
    top: Math.round(board.top * 2) / 2,
    ringed: decision.querySelectorAll('.mine .hand .tile.selected').length,
    ai: tags.some(tag => /\bAI\b/.test(tag)),
    now: tags.findIndex(tag => /\bNow\b/.test(tag)),
    live: [...decision.querySelectorAll('.mine .hand button.tile:not(:disabled)')].map(tile => tile.dataset.tile),
    picks: [...decision.querySelectorAll('[data-pick]')].map(button => button.getAttribute('aria-label')),
    help: decision.querySelector('#hand-help')?.textContent.trim() ?? '',
    line: document.querySelector('[data-game-review] [data-move-line]').textContent.replace(/\s+/g, ' ').trim(),
    said: document.querySelector('[data-game-review] [data-step-said]').textContent,
    listed: document.querySelector('[data-review-index] [aria-current="step"]')?.getAttribute('aria-label') ?? null,
  };
});
/** Checks that the decision on show, `place` of the tile-throwing game unless
 * `game` is another game's hands, keeps Trained AI's move back, and the other
 * players' tiles where `tiles` says so: Show answer in the step bar, no view,
 * no ring or AI over a tile, every legal discard a tile to press and every
 * other move a button under the hand in play's words, the prompt, and the
 * move said hidden in the step's line and the list's. Says what it saw. */
async function keptBack(page, place, { tiles = true, game = hands } = {}) {
  const shown = await shownAt(page), frame = game[place.hand].frames[place.step], me = frame.view.seats[0];
  const at = `at ${place.hand}:${place.step}`;
  assert.equal(shown.reveal, 'Show answer', at);
  assert.equal(shown.view, false, `no view ${at}`);
  assert.deepEqual([shown.ringed, shown.ai], [0, false], `no ring or AI over a tile ${at}`);
  const legal = tile => frame.choices.some(choice => choice.kind === 'discard' && choice.tile === tile);
  assert.deepEqual(shown.live, [...me.hand, ...(me.drawn ? [me.drawn] : [])].filter(legal), `the live tiles ${at}`);
  const robbing = Boolean(quadOnOffer(frame.view));
  assert.deepEqual(shown.picks, frame.choices.filter(choice => choice.kind !== 'discard').map(choice => callLabel(choice, robbing)), `the other moves ${at}`);
  assert.equal(shown.help, 'Pick a move, or Show answer.');
  assert.match(shown.line, /AI \?\s*Show answer(\s*·\s*Flagged)?$/, shown.line);
  assert.match(shown.said, /\. Trained AI's move hidden\.( Flagged\.)?$/, shown.said);
  assert.match(shown.listed, /\. Trained AI's move hidden\.( Flagged\.)?$/, shown.listed);
  assert.equal(shown.racks.length, 3);
  for (const rack of shown.racks) {
    assert.equal(rack.label.endsWith(', face down'), tiles, rack.label);
    if (tiles) assert.deepEqual([rack.down, rack.named], [rack.tiles, 0], `${rack.label}: every tile face down, none named ${at}`);
  }
  return shown;
}
/** Checks that the decision on show, `place`, shows everything again, its
 * racks turned over in place: each in the same place in the table, the same
 * size, as they were `before` (shownAt), and the table where it was on the
 * screen, with the tiles the engine held. Says what it saw. */
async function shownAgain(page, place, before = null) {
  const shown = await shownAt(page), frame = hands[place.hand].frames[place.step], seats = frame.view.seats;
  const at = `at ${place.hand}:${place.step}`;
  assert.equal(shown.reveal, null, `nothing left to show ${at}`);
  assert.equal(shown.view, !facts[place.hand][place.step].forced, `Trained AI's view ${at}`);
  assert.doesNotMatch(shown.said, /hidden/, shown.said);
  assert.doesNotMatch(shown.listed, /hidden/, shown.listed);
  for (const rack of shown.racks) {
    const slot = seats.findIndex(seat => `${SEAT_NAMES[seat.seat]}'s tiles` === rack.label.split(',')[0]);
    assert.deepEqual(unmarked(rack.names), named(frame.hands[slot]), `${rack.label} ${at}`);
  }
  if (before) {
    assert.deepEqual(shown.racks.map(rack => rack.box), before.racks.map(rack => rack.box), `each rack where it was, as large, ${at}`);
    assert.equal(shown.board, before.board, `the table as tall ${at}`);
    assert.ok(Math.abs(shown.top - before.top) <= 1, `the table where it was on the screen ${at}: ${before.top}, then ${shown.top}`);
  }
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
    if (/^All hands reviewed/.test(state.headline)) return;
    assert.doesNotMatch(state.headline, /^Review stopped/, state.headline);
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
  await cases.check("a finished match is reviewed whole, its final hand first, flagged from the real answers hand by hand, stepped through by keyboard with Trained AI's move on show, and left by Esc", async () => {
    const page = await openPage(1100, 900);
    await page.click('.standings [data-review-game]');
    await page.waitForSelector('[data-game-review]');
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title'); // The review takes focus.
    assert.equal(await page.$eval('#game-review-title', el => el.textContent), 'Whole game');
    assert.equal(await page.$eval('.match-view', el => getComputedStyle(el).display), 'none', 'the table steps aside');
    // It opens on the final hand the player made a decision in, at its first
    // decision, which is asked about first.
    const final = hands.findLast(hand => hand.frames.length);
    assert.equal(await stepText(page), placeOf(final, 0));
    // Waiting on Stop: when the review ends, the button goes, and focus moves
    // on to Next, or Previous at the end, rather than to the page.
    await page.waitForSelector('[data-game-review] [data-stop-review]');
    await page.focus('[data-game-review] [data-stop-review]');
    await reviewed(page);
    await page.waitForFunction(() => document.activeElement?.matches('[data-steps] :is([data-next], [data-previous])'), { timeout: 5000 });

    // Every batch, in the order the review asks: a finished game's final
    // hand first, then the rest from the first deal.
    const batches = await page.evaluate(() => window.reviewBatches);
    const indices = hands.map(hand => hand.index);
    const { first, second, walked } = walkBatches(batches, [indices.at(-1), ...indices.slice(0, -1)]);
    assert.equal(walked, batches.length, 'no batch is missing or extra');

    const verdicts = facts.flatMap((decisions, hand) => decisions.map((fact, index) => {
      const key = `${hand}:${index}`;
      if (fact.forced) return { key, hand, index, forced: true };
      return { key, hand, index, ...judged(fact, first.get(key), second.get(key)) };
    }));
    const counted = verdicts.filter(verdict => !verdict.forced);
    const matched = counted.filter(verdict => verdict.agreed).length;
    assert.match(await page.$eval('[data-trend] .count', el => el.textContent), new RegExp(`^${matched} of ${counted.length} decisions? with a choice$`));
    // The list's flagged lines are the decisions flagged from the answers,
    // hand by hand in the order played, each named by both moves.
    const flagged = verdicts.filter(verdict => verdict.flagged);
    assert.ok(flagged.length >= 2, 'Trained AI flags some of a tile-throwing game');
    const shown = await page.$$eval('[data-review-index] [data-flagged]', lines => lines.map(line => ({ key: line.dataset.flagged,
      label: line.getAttribute('aria-label') })));
    assert.deepEqual(shown.map(item => item.key), flagged.map(verdict => verdict.key), 'flagged, hand by hand in the order played');
    shown.forEach((item, at) => {
      const { hand, index, played, advised } = flagged[at], view = hands[hand].frames[index].view;
      const offered = view.phase === 'call' && view.pending_discard ? `On the ${tileWords(view.pending_discard)}. ` : '';
      assert.equal(item.label, `Decision ${index + 1}. ${offered}You: ${played.label}. Trained AI: ${advised.label}. Flagged.`);
    });
    await openList(page);
    await shot(page, 'game-review-list');

    // The game enters the trend once, with the figure shown and the version
    // of Trained AI that judged it: the default, as no other was picked.
    const trend = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), TREND_KEY);
    assert.equal(trend.games.length, 1);
    assert.deepEqual({ ...trend.games[0], at: 0 }, { id: `${snapshot.seed}:${snapshot.commands.length}`, network: MANIFEST.version,
      at: 0, matched, decisions: counted.length, flagged: flagged.length });
    assert.ok((await page.$eval('[data-trend] > summary', el => el.textContent.trim())).startsWith(`Matched Trained AI ${MANIFEST.version}: `));

    // Every flagged decision by keyboard, from the last back to the first by
    // Previous and on again by Next, across hands. Each keeps focus until an
    // end disables it, which hands focus to the other, never to the page,
    // and each step says where it is, the table and both moves.
    const said = () => page.$eval('[data-game-review] [data-step-said]', el => el.textContent);
    const focusLabel = label => page.waitForFunction(label => document.activeElement?.getAttribute('aria-label') === label, { timeout: 5000 }, label);
    await stepTo(page, flagged.at(-1).key);
    await page.focus('[data-game-review] [data-previous]');
    // The list keeps the line of the decision on show in view as it goes.
    for (let at = flagged.length - 2; at >= 0; at--) {
      await page.keyboard.press('Enter');
      const { hand, index } = flagged[at], frames = hands[hand].frames;
      await onPlace(page, hands[hand], index);
      assert.ok((await said()).startsWith(`${hands[hand].name}, decision ${index + 1} of ${frames.length}. ${tableSaid(frames[index].view)} You: `));
      assert.match(await said(), / Flagged\.$/);
      await focusLabel(at ? 'Previous flagged decision' : 'Next flagged decision');
      assert.equal(await lineInView(page), true, `the line of ${flagged[at].key} in view`);
    }
    for (let at = 1; at < flagged.length; at++) {
      await page.keyboard.press('Enter');
      await onPlace(page, hands[flagged[at].hand], flagged[at].index);
      await focusLabel(at < flagged.length - 1 ? 'Next flagged decision' : 'Previous flagged decision');
      assert.equal(await lineInView(page), true, `the line of ${flagged[at].key} in view`);
    }

    // A flagged decision with two of the player's decisions before it in its
    // hand, gone to from its line, with Trained AI's move on show.
    const lead = flagged.find(verdict => verdict.index >= 2) ?? flagged[0];
    await stepTo(page, lead.key);
    await page.waitForSelector('[data-decision] [data-view]');
    const frames = hands[lead.hand].frames, index = lead.index;
    assert.equal(await page.$eval('[data-decision] .wall strong', el => Number(el.textContent)), frames[index].view.wall, 'the table as it was');
    // The step bar says both moves with the weights the worker gave them,
    // the tiles given riichi where both declared it, and the flag.
    const given = lead.stage === 'tile' ? ' given riichi' : '';
    assert.equal(await said(), `${hands[lead.hand].name}, decision ${index + 1} of ${frames.length}. ${tableSaid(frames[index].view)} `
      + `You: ${lead.played.label}, ${percent(lead.playedWeight)}${given}. Trained AI: ${lead.advised.label}, ${percent(lead.preferred)}${given}. Flagged.`,
    "both moves, from the worker's answer");
    assert.match(await page.$eval('[data-game-review] [data-move-line]', el => el.textContent.trim()), /Flagged$/);
    // Trained AI's view tags its choice and the move made as the hand does,
    // and for a turn compares what each leaves, folded until it is opened.
    const tags = await page.$$eval('[data-view] .row[data-move]', rows => rows.map(row => ({ move: row.dataset.move,
      tag: row.querySelector('.tag')?.textContent ?? '' })));
    assert.equal(tags.find(row => row.move === lead.advised.label)?.tag, 'AI', JSON.stringify(tags));
    assert.equal(tags.find(row => row.move === lead.played.label)?.tag, 'You', JSON.stringify(tags));
    const fact = facts[lead.hand][index];
    if (!lead.played.kind.match(/pass|pon|chii|kan|ron/) && !fact.note.call_tile) {
      assert.equal(await page.$eval('[data-view] details.numbers', details => details.open), false, 'a turn compares what each move leaves');
    }
    // The hand puts words over the tile each threw, on a line every tile
    // keeps, and rings Trained AI's alone, as Agent watch rings the agent's.
    const tagged = await page.$$eval('[data-decision] .mine .hand-tile', tiles => tiles.map(tile => ({
      tag: tile.querySelector('.tag')?.textContent ?? null, ringed: tile.querySelector('.tile').classList.contains('selected'),
      name: tile.querySelector('.tile').getAttribute('aria-label') })));
    assert.ok(tagged.length && tagged.every(tile => tile.tag !== null), 'a line for the words over every tile');
    const throws = move => ['discard', 'riichi'].includes(move.kind);
    assert.equal(tagged.filter(tile => tile.ringed).length, throws(lead.advised) ? 1 : 0, "Trained AI's tile alone is ringed");
    if (throws(lead.advised)) {
      const ringed = tagged.find(tile => tile.ringed);
      assert.match(ringed.tag, /\bAI$/);
      assert.ok(ringed.name.startsWith(tileWords(lead.advised.tile)) && ringed.name.endsWith("Trained AI's choice"), ringed.name);
    }
    if (throws(lead.played)) assert.equal(tagged.filter(tile => /^You\b/.test(tile.tag)).length, 1, 'the tile the player threw');
    await shot(page, 'game-review-decision');

    // Trained AI's view leads with five moves and those it marks. A decision
    // with more hides the rest behind a button that stays to hide them again,
    // so focus stays on it both ways; a step to a decision with fewer takes
    // the button away, and focus goes on to Next rather than to the page.
    // Under every decision, so that Next goes to the decision after it.
    await choose(page, 'every');
    const all = facts.flatMap((decisions, hand) => decisions.map((entry, at) => ({ hand, at, entry })));
    const wide = all.find(({ hand, entry }, place) => !entry.forced && entry.choices.length >= 8 && hand === all[place + 1]?.hand
      && all[place + 1].entry.choices.length <= 5);
    assert.ok(wide, 'a decision with more moves than the view leads with, before one with fewer');
    await stepTo(page, `${wide.hand}:${wide.at}`);
    await page.waitForSelector('[data-decision] [data-view] [data-every-move]');
    const rows = () => page.$$eval('[data-view] .row', items => items.length);
    const leading = await rows();
    await page.focus('[data-view] [data-every-move]');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.matches('[data-view] [data-every-move]')
      && document.activeElement.textContent === 'Show fewer moves', { timeout: 5000 });
    assert.ok(await rows() > leading, 'every move is shown');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.matches('[data-view] [data-every-move]')
      && /^Show \d+ more moves?$/.test(document.activeElement.textContent), { timeout: 5000 });
    assert.equal(await rows(), leading);
    await page.keyboard.press('ArrowRight');
    await onPlace(page, hands[wide.hand], wide.at + 1);
    await page.waitForFunction(() => document.activeElement?.matches('[data-steps] :is([data-next], [data-previous])'), { timeout: 5000 })
      .catch(async () => assert.fail(`focus on Next, or Previous at the end, once the button went, not on ${await focusedName(page)}`));

    // On a phone the review fits the screen. Only the size changes: emulating
    // touch would reload the page, and the review with it.
    await page.setViewport({ width: 360, height: 780 });
    await settle(page);
    assert.ok(await overflow(page), 'the review fits a phone');
    await shot(page, 'game-review-phone');
    await page.setViewport({ width: 1100, height: 900 });

    // Esc goes back to the standings, as they were, focus on the button that
    // opened the review; the saved match is untouched.
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('[data-game-review]') && document.activeElement?.matches('.standings [data-review-game]'),
      { timeout: 5000 });
    assert.notEqual(await page.$eval('.match-view', el => getComputedStyle(el).display), 'none');
    const { _storage, ...kept } = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
    assert.deepEqual(kept, snapshot, 'the save is unchanged');
    assert.deepEqual(page.errors, []);
  });

  await cases.check("every step of a hand shows every hand face up and the board as it stood, walked by the arrow key and by Next at desktop, tablet and phone sizes, the step bar on the screen, nothing focused under it, and the open list's line on show stepped out from under it", async () => {
    const page = await openPage(1100, 900);
    await reviewWholeGame(page);
    await reviewed(page);
    // Of the hands with a flagged decision, the one with the most on the
    // table: sets called and riichi declared.
    const flaggedHands = new Set(await page.$$eval('[data-review-index] [data-flagged]', lines => lines.map(line => Number(line.dataset.flagged.split(':')[0]))));
    const busy = hand => hand.frames.reduce((sum, frame) => sum
      + frame.view.seats.reduce((count, seat) => count + seat.melds.length + (seat.riichi ? 1 : 0), 0), 0);
    const chosen = hands.filter(hand => hand.frames.length >= 4 && flaggedHands.has(hand.index)).sort((a, b) => busy(b) - busy(a))[0];
    assert.ok(chosen, 'a hand of four decisions or more with one flagged');
    const { frames } = chosen;
    /** The overview at a step, checked, its figures kept by the size of the
     * screen: how wide the opponents' tiles and the player's are drawn, and
     * how tall the table and the player's seat are together. They are said
     * at the end, for the UI notes to give. */
    const figures = new Map();
    const measured = async step => {
      const shown = await overviewIsRight(page, chosen, step), { width, height } = page.viewport(), size = `${width}x${height}`;
      if (!figures.has(size)) figures.set(size, { opponents: [], own: [], height: [] });
      const kept = figures.get(size);
      kept.opponents.push(...shown.opponents.map(list => list.tileWidth));
      kept.own.push(shown.own.tileWidth);
      kept.height.push(shown.height);
      return shown;
    };
    // Every decision, from the hand's name in the list, under All.
    await choose(page, 'every');
    const name = `[data-review-index] [data-hand-row="${chosen.index}"]`;
    await press(page, name);
    // By the arrow key, every step of the hand. The key leaves focus where it
    // was, on the hand's name.
    for (const step of frames.keys()) {
      await onPlace(page, chosen, step);
      await measured(step);
      if (step < frames.length - 1) await page.keyboard.press('ArrowRight');
    }
    assert.ok(await focusOn(page, name), `the key leaves focus where it was, not on ${await focusedName(page)}`);
    await shot(page, 'review-overview-1100x900');
    // By Next from the hand's first decision, which keeps focus to the
    // hand's last.
    await press(page, name);
    await onPlace(page, chosen, 0);
    await page.focus('[data-game-review] [data-next]');
    for (let step = 1; step < frames.length; step++) {
      await page.keyboard.press('Enter');
      await onPlace(page, chosen, step);
      assert.ok(await focusOn(page, '[data-game-review] [data-next]'), `focus on Next at step ${step + 1}, not on ${await focusedName(page)}`);
    }
    await stageFits(page);
    // Tabbing on from the step bar lands on nothing under it.
    await tabsClear(page);

    // The same overview at each size, the hands fitting the screen, and a
    // step by pointer at each. Only the size changes: emulating touch would
    // reload the page, and the review with it. A phone's picture is of the
    // screen, so there are two: the table, and the player's own seat.
    let step = frames.length - 1;
    for (const [width, height] of [[1440, 900], [1366, 768], [800, 1280], [390, 844], [320, 568], [844, 390]]) {
      await page.setViewport({ width, height });
      await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
      await overviewIsRight(page, chosen, step);
      // The stage fits the screen as soon as the screen changes, before any
      // press: narrowed to a phone's, the step bar sticks under the game's
      // bar. Measured in the same task as the scroll to the player's tiles,
      // CI found the bar at the top of the screen under it, and a frame
      // after that scroll, as stageFits measures now, where it belongs.
      // Should it fail, the picture shows what the player saw.
      await stageFits(page).catch(async error => {
        await shot(page, `review-resized-${width}x${height}`);
        throw error;
      });
      const forward = step < frames.length - 1;
      step += forward ? 1 : -1;
      // Brought to the middle of the screen first, clear of the game's bar,
      // which stays over the page on a phone held either way.
      await page.$eval('[data-game-review] [data-steps]', el => el.scrollIntoView({ block: 'center' }));
      await page.click(`[data-game-review] ${forward ? '[data-next]' : '[data-previous]'}`);
      await onPlace(page, chosen, step);
      await overviewIsRight(page, chosen, step);
      assert.notEqual(await focusedName(page), 'the page', `focus stays in the review at ${width}x${height}`);
      // And at the step pressed, which can make the step bar taller.
      await stageFits(page);
      await page.$eval('[data-decision] [data-overview]', el => el.scrollIntoView({ block: 'start' }));
      await shot(page, `review-overview-${width}x${height}`);
      // The step bar is stuck at the top of the screen now, over the table,
      // and its own buttons are on the screen with it: keyboard focus that
      // lands on them, as it does when handed on at either end or tabbed
      // back to, moves nothing. On a screen 500 pixels high or less the bar
      // stays in its place, and a button focused there is brought into view
      // below the game's bar, which stays at the top. Focus is taken off
      // first, so that it moves onto each button: the one pressed above has
      // it already, and focusing an element that has focus does nothing,
      // scrolling included.
      for (const button of ['[data-previous]', '[data-next]']) {
        const selector = `[data-game-review] [data-steps] ${button}:not(:disabled)`;
        if (!await page.$(selector)) continue;
        await page.evaluate(() => document.activeElement?.blur());
        const before = await page.evaluate(() => scrollY);
        await page.focus(selector);
        await settle(page);
        const after = await page.evaluate(selector => ({ scrolled: scrollY, landed: document.activeElement === document.querySelector(selector),
          sticks: getComputedStyle(document.querySelector('[data-game-review] [data-steps]')).position === 'sticky',
          top: document.querySelector(selector).getBoundingClientRect().top,
          below: document.querySelector('header.bar').getBoundingClientRect().bottom }), selector);
        assert.ok(after.landed, `focus on ${button} at ${width}x${height}`);
        if (after.sticks) assert.equal(after.scrolled, before, `focus on ${button} moved the page at ${width}x${height}`);
        else assert.ok(after.top >= after.below, `${button} at ${after.top}, under the game's bar to ${after.below}, at ${width}x${height}`);
      }
      if (phoneSized(page)) {
        // The player's seat from its top, just below the bars that stay at
        // the top of the screen: the game's, and the step bar under it where
        // that stays too.
        await page.$eval('[data-decision] [data-overview] .mine', el => {
          const bar = document.querySelector('header.bar').getBoundingClientRect().bottom;
          const steps = document.querySelector('[data-game-review] [data-steps]');
          const top = getComputedStyle(steps).position === 'sticky' ? bar + steps.offsetHeight : bar;
          scrollBy(0, el.getBoundingClientRect().top - top - 8);
        });
        await shot(page, `review-overview-${width}x${height}-yours`);
      }
      // Every step of the hand at this size, by its line in the list,
      // measured. Each line is brought to the top of the screen first, just
      // under the step bar where that stays, as a line focused there is: it
      // keeps focus, and stays clear of the bar also where its step makes the
      // bar taller over the page.
      await openList(page);
      for (const at of frames.keys()) {
        await page.$eval(line(`${chosen.index}:${at}`), el => el.scrollIntoView({ block: 'start' }));
        await page.click(line(`${chosen.index}:${at}`));
        await onPlace(page, chosen, at);
        await measured(at);
        await settle(page);
        const held = await page.evaluate(() => {
          const steps = document.querySelector('[data-game-review] [data-steps]'), active = document.activeElement;
          return { focus: active.dataset.stepTo ?? null, top: Math.round(active.getBoundingClientRect().top),
            sticks: getComputedStyle(steps).position === 'sticky', edge: Math.round(steps.getBoundingClientRect().bottom),
            inStage: document.querySelector('[data-game-review] [data-stage]').contains(active) };
        });
        assert.equal(held.focus, `${chosen.index}:${at}`, `focus on the line of step ${at + 1} at ${width}x${height}`);
        if (held.sticks && held.inStage) {
          assert.ok(held.top >= held.edge, `the line of step ${at + 1} at ${held.top}, under the step bar to ${held.edge}, at ${width}x${height}`);
        }
      }
      step = frames.length - 1;
      // On a phone, tabbing on from the step bar through the open list, under
      // the flagged decisions, so that Tab goes through its lines and on into
      // the decision, lands on nothing under the bars.
      if (width === 390) {
        await choose(page, 'flagged');
        await tabsClear(page);
        await choose(page, 'every');
        // With the open list's box running up under the bars, Previous
        // brings each line it steps to out from under them, by scrolling
        // the box: the page is scrolled so that the box's top is 50 pixels
        // under the step bar, and the box so that the line of the hand's
        // last step, on show, is just below the bar.
        await page.$eval(line(`${chosen.index}:${frames.length - 1}`), item => {
          const box = item.closest('.hands'), steps = document.querySelector('[data-game-review] [data-steps]');
          const clear = document.querySelector('header.bar').getBoundingClientRect().bottom + steps.offsetHeight;
          scrollBy(0, box.getBoundingClientRect().top - (clear - 50));
          box.scrollTop += item.getBoundingClientRect().top - (clear + 8);
        });
        await settle(page);
        assert.equal(await lineInView(page), true, `the line of the hand's last step just below the bars at ${width}x${height}`);
        await page.focus('[data-game-review] [data-previous]');
        for (let at = frames.length - 2; at >= 0; at--) {
          await page.keyboard.press('Enter');
          await onPlace(page, chosen, at);
          await settle(page);
          assert.equal(await lineInView(page), true, `the line of step ${at + 1}, stepped back to, clear of the bars at ${width}x${height}`);
          step = at;
        }
      }
    }
    // The figures at each size, from every step of the hand: the narrowest
    // and widest tile of the opponents' and of the player's, and the least
    // and most height the table and the player's seat took together.
    const span = values => `${Math.min(...values)} to ${Math.max(...values)}`;
    for (const [size, kept] of figures) {
      console.log(`review overview at ${size}: opponents' tiles ${span(kept.opponents.map(Math.round))} px wide, the player's `
        + `${span(kept.own.map(Math.round))}; the table and the player's seat ${span(kept.height)} px tall`);
    }
    assert.deepEqual(page.errors, []);
  });

  await cases.check('the list of hands shows the line on show as it opens, stays as the player leaves it, folded under the step bar or open, with focus on the line pressed, as the step changes and later hands are reviewed, by touch and keyboard, and on a wide screen keeps still what the player looks at, a line or a filter pressed or an earlier hand, the page still', async () => {
    // A page loaded at each size, one at a time, where the list is folded
    // above the decision. Each batch reaches the worker a second late, so
    // that later hands are still being reviewed while the list is used,
    // however fast the machine.
    for (const [width, height] of [[390, 844], [320, 568], [844, 390], [800, 1280]]) {
      const size = `${width}x${height}`, page = await openPage(width, height, { touch: true, answerDelay: 1000 });
      await reviewWholeGame(page);
      const list = 'details[data-review-index]';
      /** Whether the list is open, and the decision whose line has focus, or
       * what else has it. */
      const state = () => page.evaluate(list => {
        const active = document.activeElement;
        return { open: document.querySelector(list).open, focus: active?.dataset.stepTo
          ?? (!active || active === document.body ? 'the page' : active.getAttribute('aria-label') ?? active.textContent.trim()) };
      }, list);
      const current = key => page.waitForFunction(selector => document.querySelector(selector)?.getAttribute('aria-current') === 'step',
        { timeout: 5000 }, line(key));
      assert.equal((await state()).open, false, `the list starts folded at ${size}`);
      // By touch: the list opened, every decision chosen, then a line of the
      // first hand with two decisions. Opened, the list shows the line of the
      // decision on show, the final hand's, at the foot of a box of every
      // hand's heading, rather than the top of the box.
      const first = hands.find(hand => hand.frames.length >= 2).index;
      await tapAt(page, `${list} > summary`);
      await page.waitForFunction(list => document.querySelector(list).open, { timeout: 5000 }, list);
      await settle(page);
      assert.equal(await lineInView(page), true, `the line on show as the list opens at ${size}`);
      await tapAt(page, option('every'));
      await page.waitForFunction(selector => document.querySelector(selector)?.getAttribute('aria-pressed') === 'true', { timeout: 5000 }, option('every'));
      await tapAt(page, line(`${first}:1`));
      await current(`${first}:1`);
      await settle(page);
      assert.deepEqual(await state(), { open: true, focus: `${first}:1` }, `a line tapped at ${size}`);
      // By keyboard: another line.
      await page.focus(line(`${first}:0`));
      await page.keyboard.press('Enter');
      await current(`${first}:0`);
      await settle(page);
      assert.deepEqual(await state(), { open: true, focus: `${first}:0` }, `a line pressed by keyboard at ${size}`);
      // Later hands are reviewed meanwhile, and nothing is pressed: the list
      // stays open, and focus on the line.
      await answeredBeyond(page, await answered(page), 2);
      await settle(page);
      assert.deepEqual(await state(), { open: true, focus: `${first}:0` }, `later hands reviewed at ${size}`);
      // Folded by touch, it stays folded as the step changes and the review
      // goes on.
      await tapAt(page, `${list} > summary`);
      await page.waitForFunction(list => !document.querySelector(list).open, { timeout: 5000 }, list);
      await tapAt(page, '[data-game-review] [data-next]');
      await current(`${first}:1`);
      await answeredBeyond(page, await answered(page));
      await settle(page);
      assert.equal((await state()).open, false, `the list stays folded at ${size}`);
      assert.deepEqual(page.errors, []);
      await closePage(page);
    }

    // On a wide screen the list is a column of its own on the left, after the
    // decision in the page, and stays on the screen as the page scrolls. Each
    // batch reaches the worker two seconds late, so that hands are still to
    // be reviewed when the column is scrolled to an earlier one.
    const page = await openPage(1366, 768, { answerDelay: 2000 });
    await reviewWholeGame(page);
    const rail = 'aside[data-review-index]';
    assert.equal(await page.$('details[data-review-index]'), null, 'no folded list beside the column');
    assert.deepEqual(await page.evaluate(() => {
      const column = document.querySelector('aside[data-review-index]'), stage = document.querySelector('[data-game-review] [data-stage]');
      return { left: column.getBoundingClientRect().right <= stage.getBoundingClientRect().left,
        after: Boolean(stage.compareDocumentPosition(column) & Node.DOCUMENT_POSITION_FOLLOWING), position: getComputedStyle(column).position };
    }), { left: true, after: true, position: 'sticky' });
    // A line of the final hand, which is asked about first, pressed by
    // keyboard: a flagged one, which the hands reviewed after it draw lines
    // above, or under every decision its first. It keeps focus and its place
    // in the column while the first hand is reviewed, and the page stays
    // where it was.
    const last = hands.findLast(hand => hand.frames.length);
    const next = hands.find(hand => hand.index !== last.index && facts[hand.index].some(fact => !fact.forced));
    await page.waitForSelector(`${rail} [data-hand="${last.index}"][data-status="done"]`, { timeout: 300000 });
    let key = await page.$eval(rail, (column, hand) => column.querySelector(`[data-step-to^="${hand}:"][data-flagged]`)?.dataset.stepTo ?? null, last.index);
    if (!key) {
      await choose(page, 'every');
      key = `${last.index}:0`;
    }
    await page.focus(line(key));
    await page.keyboard.press('Enter');
    await page.waitForFunction(selector => document.querySelector(selector)?.getAttribute('aria-current') === 'step', { timeout: 5000 }, line(key));
    await settle(page);
    const where = () => page.evaluate(selector => {
      const column = document.querySelector('aside[data-review-index]').getBoundingClientRect(), item = document.querySelector(selector);
      const box = item.getBoundingClientRect();
      return { focused: document.activeElement === item, shown: box.top >= column.top && box.bottom <= column.bottom,
        offset: box.top - column.top, scrolled: scrollY };
    }, line(key));
    const before = await where();
    assert.ok(before.focused && before.shown, `the line pressed has focus and is in view: ${JSON.stringify(before)}`);
    await page.waitForSelector(`${rail} [data-hand="${next.index}"][data-status="done"]`, { timeout: 300000 });
    await settle(page);
    const after = await where();
    assert.ok(after.focused && after.shown, `the line keeps focus and stays in view: ${JSON.stringify(after)}`);
    assert.ok(Math.abs(after.offset - before.offset) <= 1, `the line stays where it was in the column: ${before.offset} then ${after.offset}`);
    assert.equal(after.scrolled, before.scrolled, 'the page does not move');

    // The column at its top, each filter pressed in turn by keyboard: every
    // decision draws lines under the filter, above the line on show, and the
    // fewer take them away again, but the button pressed keeps focus where it
    // was in the column, in view, and the page does not move.
    await page.$eval(rail, column => { column.scrollTop = 0; });
    const pressed = () => page.evaluate(() => {
      const column = document.querySelector('aside[data-review-index]'), active = document.activeElement;
      const edge = column.getBoundingClientRect(), box = active.getBoundingClientRect();
      return { filter: active.dataset.filterOption ?? null, shown: box.top >= edge.top && box.bottom <= edge.bottom,
        offset: Math.round(box.top - edge.top), scrolled: scrollY };
    });
    for (const name of ['flagged', 'every', 'differed']) {
      await page.focus(option(name));
      const was = await pressed();
      assert.ok(was.filter === name && was.shown, `${name} focused in view at the column's top: ${JSON.stringify(was)}`);
      await page.keyboard.press('Enter');
      await page.waitForFunction(selector => document.querySelector(selector)?.getAttribute('aria-pressed') === 'true', { timeout: 5000 }, option(name));
      await settle(page);
      assert.deepEqual(await pressed(), was, `${name} pressed: the button where it was in the column, the page still`);
    }

    // Scrolled to the first hand, nothing in the column focused, every
    // decision listed, which holds every hand's lines: a later hand reviewed
    // meanwhile loses its state and its Review below the first hand, whose
    // heading stays where it was in the column, the page still.
    await page.focus(option('every'));
    await page.keyboard.press('Enter');
    await page.waitForFunction(selector => document.querySelector(selector)?.getAttribute('aria-pressed') === 'true', { timeout: 5000 }, option('every'));
    const heading = `${rail} [data-hand="${hands.find(hand => hand.frames.length).index}"] .head`;
    await page.$eval(heading, head => {
      const column = head.closest('aside');
      column.scrollTop += head.getBoundingClientRect().top - column.getBoundingClientRect().top - 4;
      document.activeElement?.blur();
    });
    await settle(page);
    const headingAt = () => page.$eval(heading, head => ({
      offset: Math.round(head.getBoundingClientRect().top - head.closest('aside').getBoundingClientRect().top), scrolled: scrollY }));
    const held = await headingAt();
    const reviewedSoFar = `${rail} [data-hand][data-status="done"]`, count = await page.$$eval(reviewedSoFar, items => items.length);
    assert.ok(await page.$(`${rail} [data-hand]:not([data-status="done"])`), 'a hand is still to be reviewed');
    await page.waitForFunction((selector, count) => document.querySelectorAll(selector).length > count, { timeout: 300000, polling: 100 },
      reviewedSoFar, count);
    await settle(page);
    assert.deepEqual(await headingAt(), held, 'the first hand stays where it was in the column as a later hand is reviewed, the page still');

    // Scrolled down the decision, the column stays at the top of the screen,
    // where the two panes still have room for it there; measured a frame
    // after the scroll, as a sticky box is placed (stageFits).
    await page.evaluate(() => scrollBy(0, document.querySelector('[data-game-review] .panes').getBoundingClientRect().top + 200));
    await settle(page);
    const stuck = await page.evaluate(() => {
      const column = document.querySelector('aside[data-review-index]'), panes = column.parentElement, box = column.getBoundingClientRect();
      return { top: Math.round(box.top), room: panes.getBoundingClientRect().bottom >= 8 + box.height };
    });
    if (stuck.room) assert.equal(stuck.top, 8, 'the column stays at the top of the screen');
    await shot(page, 'game-review-wide');
    assert.deepEqual(page.errors, []);
  });

  await cases.check("on a slow device the list opens by itself once, offering each hand still to review, and the review's lists stay as the player leaves them: the list as the review goes on, a hand's events across a step with none and across hands, and the trend", async () => {
    // Each batch reaches the worker four and a half seconds late, over half a
    // second a position: a device slow enough to hold after each hand, for
    // which the list opens. A game reviewed before this one gives the trend a
    // table of games. A tablet's width, where the list is folded above the
    // decision.
    const page = await openPage(800, 1280, { answerDelay: 4500,
      trend: [{ id: 'earlier', network: MANIFEST.version, at: Date.UTC(2026, 9, 1, 12), matched: 60, decisions: 100, flagged: 4 }] });
    // What the page did not keep as the player left it, said together at the
    // end, so that each list is checked whatever the others do.
    const changed = [];
    const stays = async (selector, open, what) => {
      await settle(page);
      if (await page.$eval(selector, details => details.open) !== open) changed.push(what);
    };
    const list = 'details[data-review-index]', trendTile = '[data-game-review] details[data-trend]';
    await page.click('.standings [data-review-game]');
    await page.waitForSelector('[data-game-review] [data-next-hand]', { timeout: 300000 });
    assert.equal(await page.$eval(list, details => details.open), true, 'the list opens for the hold');
    // Each hand not reviewed yet offers to be reviewed next, in its header.
    const offered = await page.$$eval('[data-review-index] [data-review-hand]', buttons => buttons.map(button => button.getAttribute('aria-label')));
    assert.ok(offered.length && offered.every(label => /^Review: \w+ \d/.test(label)), offered.join(', '));
    // Folded by keyboard, it stays folded, also while the next hand is
    // reviewed and at the hold after it.
    await page.focus(`${list} > summary`);
    await page.keyboard.press('Enter');
    await stays(list, false, 'the list opened again as it was folded');
    const before = await answered(page);
    await page.focus('[data-game-review] [data-next-hand]');
    await page.keyboard.press('Enter');
    await answeredBeyond(page, before);
    await stays(list, false, 'the list opened again as the review went on');
    await page.waitForSelector('[data-game-review] [data-next-hand]', { timeout: 300000 });
    await stays(list, false, "the list opened again at the review's next hold");

    // The trend, opened by pointer in the list, which the player opens again.
    await openList(page);
    await press(page, `${trendTile} > summary`);
    await page.waitForFunction(selector => document.querySelector(selector).open, { timeout: 5000 }, trendTile);
    // A step that nothing happened before, between two that had events, gone
    // to under every decision, which holds the hands not reviewed yet too:
    // its events, which start folded, opened at the step before, stay open at
    // the step after, and at another hand's.
    const quiet = hands.flatMap(hand => hand.frames.map((_, step) => ({ hand, step })))
      .find(({ hand, step }) => step > 0 && step < hand.frames.length - 1 && !hand.frames[step].lines.length
        && hand.frames[step - 1].lines.length && hand.frames[step + 1].lines.length);
    assert.ok(quiet, 'a hand with a step that nothing happened before');
    const elsewhere = hands.find(hand => hand.index !== quiet.hand.index && hand.frames.length > 1 && hand.frames[1].lines.length);
    assert.ok(elsewhere, 'another hand with events before its second decision');
    await choose(page, 'every');
    await stepTo(page, `${quiet.hand.index}:${quiet.step - 1}`);
    const lines = '[data-decision] details.lines', next = '[data-game-review] [data-next]';
    assert.equal(await page.$eval(lines, details => details.open), false, "a hand's events start folded");
    await press(page, `${lines} > summary`);
    await page.waitForFunction(lines => document.querySelector(lines).open, { timeout: 5000 }, lines);
    await press(page, next);
    await onPlace(page, quiet.hand, quiet.step);
    assert.equal(await page.$(lines), null, 'no events since the previous decision');
    await press(page, next);
    await onPlace(page, quiet.hand, quiet.step + 1);
    await stays(lines, true, "a hand's events folded again after a step with none");
    await stepTo(page, `${elsewhere.index}:1`);
    await stays(lines, true, "a hand's events folded again in another hand");
    await stays(trendTile, true, 'the trend folded again across steps and hands');
    await stays(list, true, 'the list folded again across steps and hands');
    assert.deepEqual(changed, [], 'every list stays as the player left it');
    assert.deepEqual(page.errors, []);
  });

  await cases.check('the filter keeps the flagged decisions, those where Trained AI differed or every one, says how many each holds, has Next visit only those across hands, keeps a decision it leaves out in view, dim, and is kept across a reload', async () => {
    const page = await openPage(1100, 900);
    await reviewWholeGame(page);
    await reviewed(page);
    await openList(page);
    const counted = async name => Number(/\((\d+)\)$/.exec(await page.$eval(option(name), el => el.textContent.trim()))[1]);
    const counts = { flagged: await counted('flagged'), differed: await counted('differed'), every: await counted('every') };
    const total = Number(/^All hands reviewed: (\d+) decisions?\.$/.exec(await page.$eval('[data-game-review] .headline', el => el.textContent))[1]);
    assert.equal(counts.every, total, 'every decision of the game');
    assert.ok(counts.flagged > 0 && counts.differed > counts.flagged && counts.every > counts.differed, JSON.stringify(counts));
    assert.equal(await page.$eval(option('flagged'), el => el.getAttribute('aria-pressed')), 'true', 'the review opens on the flagged decisions');
    /** The decisions the list has a line for that its filter keeps, and
     * whether each says it is flagged. */
    const kept = () => page.$$eval('[data-review-index] [data-step-to]:not(.outside)', lines => lines.map(line => ({ key: line.dataset.stepTo,
      flagged: line.hasAttribute('data-flagged'), said: line.innerText.replace(/\s+/g, ' ').trim() })));
    const summary = async () => (await page.$('details[data-review-index]'))
      ? page.$eval('details[data-review-index] > summary', el => el.textContent.trim()) : null;
    assert.equal((await kept()).length, counts.flagged);

    // By pointer: those where Trained AI chose differently, said as the
    // filter changes, with focus on its button; the flagged ones say so.
    await press(page, option('differed'));
    await page.waitForFunction(selector => document.querySelector(selector)?.getAttribute('aria-pressed') === 'true', { timeout: 5000 }, option('differed'));
    assert.equal(await page.$eval('[data-game-review] [data-filter-said]', el => el.textContent), filterSaid('differed', counts));
    assert.ok(await focusOn(page, option('differed')), 'focus stays on the button pressed');
    const folded = await summary();
    if (folded !== null) assert.equal(folded, `Hands · ${FILTER_LABELS.differed} (${counts.differed})`);
    const different = await kept();
    assert.equal(different.length, counts.differed);
    assert.equal(different.filter(item => item.flagged).length, counts.flagged);
    assert.ok(different.every(item => item.flagged === /Flagged$/.test(item.said)), 'the flagged ones say so');

    // Next visits only those, across hands, in the order played, by keyboard,
    // from the first; at the last it hands focus to Previous.
    const at = key => key.split(':').map(Number);
    await stepTo(page, different[0].key);
    await page.focus('[data-game-review] [data-next]');
    for (const item of different.slice(1)) {
      await page.keyboard.press('Enter');
      const [hand, step] = at(item.key);
      await onPlace(page, hands[hand], step);
    }
    await page.waitForFunction(() => document.activeElement?.matches('[data-steps] [data-previous]'), { timeout: 5000 });
    assert.equal(await page.$eval('[data-game-review] [data-next]', el => el.disabled), true, 'Next is disabled at the last');

    // A decision the flagged ones leave out, on show as the filter changes,
    // stays in view and in the list, dim, current, until the player moves on.
    const plain = different.find(item => !item.flagged);
    await stepTo(page, plain.key);
    await choose(page, 'flagged');
    const [plainHand, plainStep] = at(plain.key);
    assert.equal(await stepText(page), placeOf(hands[plainHand], plainStep), 'the decision on show stays');
    assert.deepEqual(await page.$eval(line(plain.key), el => [el.classList.contains('outside'), el.getAttribute('aria-current')]), [true, 'step']);
    await overviewIsRight(page, hands[plainHand], plainStep);
    await shot(page, 'review-overview-plain-decision');
    // Moved on, it is no longer listed.
    await page.$eval('[data-game-review] [data-steps]', el => el.scrollIntoView({ block: 'center' }));
    await page.click(`[data-game-review] ${(await page.$('[data-game-review] [data-next]:not(:disabled)')) ? '[data-next]' : '[data-previous]'}`);
    await page.waitForFunction(selector => !document.querySelector(selector), { timeout: 5000 }, line(plain.key));

    // By keyboard: every decision, in the order played, hand by hand.
    await page.focus(option('every'));
    await page.keyboard.press('Enter');
    await page.waitForFunction(selector => document.querySelector(selector)?.getAttribute('aria-pressed') === 'true', { timeout: 5000 }, option('every'));
    assert.equal(await page.$eval('[data-game-review] [data-filter-said]', el => el.textContent), filterSaid('every', counts));
    assert.deepEqual((await kept()).map(item => item.key), hands.flatMap(hand => hand.frames.map((_, step) => `${hand.index}:${step}`)));
    // Focus moving on quiets the line, which spoke of the filter as it was.
    await page.focus('[data-game-review] [data-close-review]');
    await page.waitForFunction(() => document.querySelector('[data-game-review] [data-filter-said]')?.textContent === '', { timeout: 5000 });

    // The filter is kept with the other preferences: after a reload, the
    // review opens on the decisions it was left on.
    await choose(page, 'differed');
    await page.waitForFunction(key => JSON.parse(localStorage.getItem(key)).reviewFilter === 'differed', { timeout: 5000 }, SETTINGS_KEY);
    await page.reload({ waitUntil: 'networkidle0' });
    await page.waitForSelector('.standings [data-review-game]');
    await page.click('.standings [data-review-game]');
    await page.waitForSelector('[data-game-review]');
    assert.equal(await page.$eval(option('differed'), el => el.getAttribute('aria-pressed')), 'true', 'the filter after a reload');
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
    // A hand reviewed on its own loses its Review once it is under review;
    // focus goes to the hand's name, beside where the button was.
    await openList(page);
    await page.waitForSelector('[data-review-index] [data-review-hand]');
    const hand = await page.$eval('[data-review-index] [data-review-hand]', button => button.dataset.reviewHand);
    await page.focus(`[data-review-index] [data-review-hand="${hand}"]`);
    await page.keyboard.press('Enter');
    await focused(`[data-review-index] [data-hand-row="${hand}"]`);
    assert.deepEqual(page.errors, []);
  });

  await cases.check("Next, on a hand still to be reviewed, keeps its focus as the review reaches the hand and it waits for the hand's answers, does nothing meanwhile, and then goes on", async () => {
    // Each batch reaches the worker two seconds late, so that the player can
    // go to a hand before the review reaches it.
    const page = await openPage(1100, 900, { answerDelay: 2000 });
    await reviewWholeGame(page);
    const rail = 'aside[data-review-index]';
    // The final hand, asked about first, gives Next a decision to go on to
    // under those where Trained AI differed.
    const last = hands.findLast(hand => hand.frames.length);
    await page.waitForSelector(`${rail} [data-hand="${last.index}"][data-status="done"]`, { timeout: 300000 });
    await choose(page, 'differed');
    // A hand that has a decision to put to Trained AI, waiting behind another
    // that does, gone to by its name in the list.
    const waiting = await page.$$eval(`${rail} [data-hand][data-status="waiting"]`, items => items.map(item => Number(item.dataset.hand)));
    const target = waiting.filter(index => facts[index].some(fact => !fact.forced))[1];
    assert.ok(target !== undefined, `a hand waiting behind another: ${waiting.join(', ')}`);
    await page.focus(`${rail} [data-hand-row="${target}"]`);
    await page.keyboard.press('Enter');
    await onPlace(page, hands[target], 0);
    await page.focus('[data-game-review] [data-next]');
    /** Whether Next has focus, is disabled, and waits, said to be disabled. */
    const next = () => page.$eval('[data-game-review] [data-next]', button => ({ focused: document.activeElement === button,
      disabled: button.disabled, waiting: button.getAttribute('aria-disabled') === 'true' }));
    // As the review reaches the hand, Next waits for its answers, said to be
    // disabled, and keeps focus; a press meanwhile goes nowhere.
    await page.waitForSelector(`${rail} [data-hand="${target}"][data-status="reviewing"]`, { timeout: 300000 });
    assert.deepEqual(await next(), { focused: true, disabled: false, waiting: true }, "Next waits for the hand's answers, keeping focus");
    await page.keyboard.press('Enter');
    await settle(page);
    assert.equal(await stepText(page), placeOf(hands[target], 0), 'Next waiting goes nowhere');
    // Answered, Next goes on from where the player was, focus still on it.
    await page.waitForSelector(`${rail} [data-hand="${target}"][data-status="done"]`, { timeout: 300000 });
    await settle(page);
    assert.deepEqual(await next(), { focused: true, disabled: false, waiting: false }, 'Next goes on once the hand is answered, focus still on it');
    await page.keyboard.press('Enter');
    await page.waitForFunction(text => document.querySelector('[data-game-review] [data-step]')?.textContent !== text, { timeout: 5000 },
      placeOf(hands[target], 0));
    const [hand, step] = (await page.$eval(`${rail} [aria-current="step"]`, item => item.dataset.stepTo)).split(':').map(Number);
    assert.ok(hand > target || (hand === target && step > 0), `Next goes on, to ${hand}:${step}`);
    assert.deepEqual(page.errors, []);
  });

  await cases.check("the review's two settings are offered in Options and, from tablet width, in the review's header, the same settings in both, kept across a reload, and on a phone in the settings alone", async () => {
    // Only the settings are looked at here, so Trained AI is out of reach.
    let page = await openPage(1100, 900, { network: false });
    await page.click('.standings [data-review-game]');
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    /** Each setting as Options and the review's header have it. */
    const boxes = () => page.evaluate(() => Object.fromEntries(['move', 'tiles'].flatMap(which => [
      [`options ${which}`, document.querySelector(`.options [data-hide-${which}]`).checked],
      [`header ${which}`, document.querySelector(`[data-game-review] [data-hide-${which}]`).checked]])));
    const all = on => ({ 'options move': on, 'header move': on, 'options tiles': on, 'header tiles': on });
    const stored = (move, tiles) => page.waitForFunction((key, move, tiles) => {
      const record = JSON.parse(localStorage.getItem(key));
      return record.reviewHideMove === move && record.reviewHideTiles === tiles;
    }, { timeout: 5000 }, SETTINGS_KEY, move, tiles);
    assert.deepEqual(await boxes(), all(false), 'both off at first');
    // The header names them without "in reviews", between the heading and Back.
    assert.deepEqual(await page.$$eval('[data-game-review] header label', labels => labels.map(label => label.textContent.trim())),
      ["Hide Trained AI's move", "Hide other players' tiles"]);
    // One turned on in the header by keyboard, the other in Options by pointer.
    await page.focus('[data-game-review] [data-hide-move]');
    await page.keyboard.press('Space');
    await press(page, '.options > summary');
    await page.waitForFunction(() => document.querySelector('.options').open, { timeout: 5000 });
    assert.deepEqual(await page.$$eval('.options label:has(input[type=checkbox])', labels => labels.map(label => label.textContent.trim()).slice(-2)),
      ["Hide Trained AI's move in reviews", "Hide other players' tiles in reviews"], 'after the other checkboxes in Options');
    await press(page, '.options [data-hide-tiles]');
    await stored(true, true);
    assert.deepEqual(await boxes(), all(true), 'the two places agree');
    // Kept across a reload, as the other preferences are.
    await page.reload({ waitUntil: 'networkidle0' });
    await page.waitForSelector('.standings [data-review-game]');
    await page.click('.standings [data-review-game]');
    await page.waitForSelector('[data-game-review]');
    assert.deepEqual(await boxes(), all(true), 'after a reload');
    // Turned off again, both from the header.
    await press(page, '[data-game-review] [data-hide-move]');
    await press(page, '[data-game-review] [data-hide-tiles]');
    await stored(false, false);
    assert.deepEqual(await boxes(), all(false));
    assert.deepEqual(page.errors, []);
    await closePage(page);

    // On a phone the header leaves them to the settings, which hold them as
    // they were stored.
    page = await openPage(390, 844, { network: false, settings: { version: 3, difficulty: 'beginner', hints: true, reviewHideTiles: true } });
    await page.$eval('.standings [data-review-game]', button => button.click());
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    assert.equal(await page.$eval('[data-game-review] .hiding', el => getComputedStyle(el).display), 'none', 'not in the header on a phone');
    await page.click('.settings-trigger');
    await page.waitForFunction(() => document.querySelector('dialog.settings-dialog[open]'), { timeout: 5000 });
    await page.click('dialog.settings-dialog .options > summary');
    await page.waitForFunction(() => document.querySelector('dialog.settings-dialog .options').open, { timeout: 5000 });
    assert.deepEqual(await page.evaluate(() => ['move', 'tiles'].map(which => document.querySelector(`dialog.settings-dialog [data-hide-${which}]`).checked)),
      [false, true], 'in the settings on a phone');
    assert.deepEqual(page.errors, []);
  });

  await cases.check("with both settings on, a decision keeps Trained AI's move and the other players' tiles back until asked, by Show answer, s, a tile's digit or the tile itself, a pick kept as Now, the racks turning over in place, each reveal kept through steps, Back and opening again, and gone after a reload", async () => {
    const hiding = { version: 3, difficulty: 'beginner', hints: true, reviewHideMove: true, reviewHideTiles: true };
    const page = await openPage(1100, 900, { settings: hiding });
    await reviewWholeGame(page);
    await reviewed(page);
    // Every decision, so that the decisions below are stepped to wherever
    // they are flagged or not.
    await choose(page, 'every');
    const keyOf = place => `${place.hand}:${place.step}`;
    // The decisions with a choice, in the order played, and where one is a
    // turn, the place in the hand of its first legal discard among the
    // hand's first nine tiles, which a digit picks.
    const choiceful = hands.flatMap(hand => hand.frames.map((_, step) => ({ hand: hand.index, step })))
      .filter(({ hand, step }) => !facts[hand][step].forced);
    const digitOf = ({ hand, step }) => {
      const frame = hands[hand].frames[step];
      return frame.view.phase !== 'act' ? -1
        : frame.view.seats[0].hand.slice(0, 9).findIndex(tile => frame.choices.some(choice => choice.kind === 'discard' && choice.tile === tile));
    };
    const [byButton, byKey] = choiceful;
    const [byDigit, byTile] = choiceful.filter(place => place !== byButton && place !== byKey && digitOf(place) >= 0);
    assert.ok(byButton && byKey && byDigit && byTile, 'four decisions with a choice, two of them turns');
    /** The tile a pick of the hand's `index`th tile is tagged on, as the
     * replay places a move: the tile just drawn where it is the one. */
    const nowAt = (place, tile) => {
      const me = hands[place.hand].frames[place.step].view.seats[0], held = [...me.hand, ...(me.drawn ? [me.drawn] : [])];
      return tile === me.drawn ? held.length - 1 : held.indexOf(tile);
    };
    const stepFocused = () => page.waitForFunction(() => document.activeElement?.matches('[data-steps] :is([data-next], [data-previous])'),
      { timeout: 5000 }).catch(async () => assert.fail(`focus on Next, or Previous at the end, not on ${await focusedName(page)}`));
    const answered = () => page.waitForFunction(() => !document.querySelector('[data-game-review] [data-reveal]'), { timeout: 5000 });

    // Show answer, by keyboard: the answer and the tiles shown together, the
    // racks turned over where they lay, and focus handed on to Next as the
    // button goes, never to the page. It is pressed at the top of the page,
    // where the step bar lies in its place rather than stuck: no scroll could
    // keep the table still there, had the bar changed height as the button
    // went. Enter is held down: its repeats, which reach Next once it has
    // focus, press nothing, so the review stays on the answer just shown.
    await stepTo(page, keyOf(byButton));
    await keptBack(page, byButton);
    await shot(page, 'game-review-hidden');
    await page.evaluate(() => scrollTo(0, 0));
    await settle(page);
    assert.ok(await page.$eval('[data-game-review] [data-steps]', bar => bar.getBoundingClientRect().top > 1), 'the step bar in its place at the top of the page');
    const first = await shownAt(page);
    await page.focus('[data-game-review] [data-reveal]');
    await page.keyboard.down('Enter');
    await answered();
    await shownAgain(page, byButton, first);
    await stepFocused();
    await page.keyboard.down('Enter');
    await page.keyboard.down('Enter');
    await page.keyboard.up('Enter');
    await settle(page);
    assert.equal(await stepText(page), placeOf(hands[byButton.hand], byButton.step), 'a held Enter does not step on past the answer');
    await shot(page, 'game-review-answered');
    // The same by s, from the decision's line in the list, which keeps focus.
    await stepTo(page, keyOf(byKey));
    const second = await keptBack(page, byKey);
    await page.keyboard.press('s');
    await answered();
    await shownAgain(page, byKey, second);
    assert.ok(await focusOn(page, line(keyOf(byKey))), `s leaves focus where it was, not on ${await focusedName(page)}`);
    // A tile picked by its digit, as play's digits throw one: the decision is
    // answered, the pick kept as Now in the step bar, over its tile, in
    // Trained AI's view and in the step's line, and focus stays on the line.
    await stepTo(page, keyOf(byDigit));
    await keptBack(page, byDigit);
    const digit = digitOf(byDigit), tile = hands[byDigit.hand].frames[byDigit.step].view.seats[0].hand[digit];
    await page.keyboard.press(String(digit + 1));
    await answered();
    const picked = await shownAgain(page, byDigit);
    const label = hands[byDigit.hand].frames[byDigit.step].choices.find(choice => choice.kind === 'discard' && choice.tile === tile).label;
    assert.match(picked.line, /^You\b.*·\s*Now\b.*·\s*(Trained )?AI\b/, picked.line);
    assert.equal(picked.now, nowAt(byDigit, tile), 'Now over the tile picked');
    assert.ok(picked.said.includes(` Now: ${label}`), picked.said);
    assert.match(await page.$eval(`[data-view] .row[data-move="${label}"] .tag`, el => el.textContent), /\bNow\b/);
    assert.ok(await focusOn(page, line(keyOf(byDigit))), `the digit leaves focus where it was, not on ${await focusedName(page)}`);
    // A tile pressed by pointer: the same, and focus goes on to Next as the
    // tile goes back to being a picture.
    await stepTo(page, keyOf(byTile));
    await keptBack(page, byTile);
    const pressed = hands[byTile.hand].frames[byTile.step].view.seats[0].hand[digitOf(byTile)];
    await press(page, `[data-decision] .mine .hand button.tile[data-tile="${pressed}"]:not(:disabled)`);
    await answered();
    assert.equal((await shownAgain(page, byTile)).now, nowAt(byTile, pressed), 'Now over the tile pressed');
    await stepFocused();

    // Each reveal is kept: away by the step bar and back, the tile pressed
    // still Now.
    const away = await page.$('[data-game-review] [data-next]:not(:disabled)') ? '[data-next]' : '[data-previous]';
    const back = away === '[data-next]' ? '[data-previous]' : '[data-next]';
    await page.focus(`[data-game-review] ${away}`);
    await page.keyboard.press('Enter');
    await page.waitForFunction(text => document.querySelector('[data-game-review] [data-step]')?.textContent !== text, { timeout: 5000 },
      placeOf(hands[byTile.hand], byTile.step));
    await page.focus(`[data-game-review] ${back}`);
    await page.keyboard.press('Enter');
    await onPlace(page, hands[byTile.hand], byTile.step);
    assert.equal((await shownAgain(page, byTile)).now, nowAt(byTile, pressed), 'still Now after a step away and back');
    // Back to the standings and in again: the review opens where it was left,
    // and every decision revealed stays so.
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('[data-game-review]'), { timeout: 5000 });
    await page.click('.standings [data-review-game]');
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    assert.equal(await stepText(page), placeOf(hands[byTile.hand], byTile.step), 'opened again where it was left');
    assert.equal((await shownAgain(page, byTile)).now, nowAt(byTile, pressed));
    for (const place of [byButton, byKey, byDigit]) {
      await stepTo(page, keyOf(place));
      await shownAgain(page, place);
    }

    // The other players' tiles alone kept back: at a decision not revealed
    // yet, Trained AI's move is on show and Show tiles turns the racks over
    // where they lay, by pointer, handing focus on as the button goes.
    await press(page, '[data-game-review] [data-hide-move]');
    const tilesOnly = choiceful.find(place => ![byButton, byKey, byDigit, byTile].includes(place));
    await stepTo(page, keyOf(tilesOnly));
    await page.$eval('[data-game-review] [data-reveal]', button => button.scrollIntoView({ block: 'center' }));
    await settle(page);
    const covered = await shownAt(page);
    assert.equal(covered.reveal, 'Show tiles');
    assert.equal(covered.view, true, "Trained AI's view on show");
    assert.ok(covered.racks.every(rack => rack.label.endsWith(', face down') && rack.down === rack.tiles), 'the racks face down');
    await page.click('[data-game-review] [data-reveal]');
    await answered();
    await shownAgain(page, tilesOnly, covered);
    await stepFocused();
    await press(page, '[data-game-review] [data-hide-move]');

    // On a phone held upright, a decision kept back: a tile of its live hand
    // that had focus as the screen turned keeps it, the seat drawn afresh
    // before the table; its live tiles and the moves under it are tabbed
    // through clear of the bars, and nothing scrolls sideways.
    const phoneAt = choiceful.find(place => ![byButton, byKey, byDigit, byTile, tilesOnly].includes(place) && digitOf(place) >= 0);
    const handTile = index => `[data-decision] .mine .hand button.tile[data-hand-index="${index}"]`;
    await stepTo(page, keyOf(phoneAt));
    await page.focus(handTile(digitOf(phoneAt)));
    await page.setViewport({ width: 390, height: 844 });
    await settle(page);
    await page.waitForFunction(selector => document.activeElement?.matches(selector), { timeout: 5000 }, handTile(digitOf(phoneAt)))
      .catch(async () => assert.fail(`the tile keeps focus as the seat moves before the table, not ${await focusedName(page)}`));
    assert.ok((await overviewAt(page)).seatFirst, "the player's seat drawn before the table");
    await keptBack(page, phoneAt);
    if (await page.$('details[data-review-index][open]')) {
      await press(page, 'details[data-review-index] > summary');
      await page.waitForFunction(() => !document.querySelector('details[data-review-index]').open, { timeout: 5000 });
    }
    await tabsClear(page);
    assert.ok(await overflow(page), 'the review fits a phone with the hand live');
    await shot(page, 'game-review-hidden-phone');
    // Answered there by a tile pressed, the racks turn over where they lay,
    // and the table stays where it was on the screen as the line under the
    // hand goes from between the player's seat and the table; so it does at
    // an answer to a discard, whose moves under the hand go with the line.
    // Each is answered with the player's seat just under the bars, as the
    // player scrolls to it, the table below it on the screen.
    const seatUnderBars = async () => {
      await page.$eval('[data-decision] [data-overview] .mine', seat => {
        const bars = document.querySelector('header.bar').getBoundingClientRect().bottom
          + document.querySelector('[data-game-review] [data-steps]').offsetHeight;
        scrollBy(0, seat.getBoundingClientRect().top - bars - 8);
      });
      await settle(page);
    };
    await seatUnderBars();
    const upright = await shownAt(page);
    await page.click(handTile(digitOf(phoneAt)));
    await answered();
    await shownAgain(page, phoneAt, upright);
    const callAt = choiceful.find(place => ![byButton, byKey, byDigit, byTile, tilesOnly, phoneAt].includes(place)
      && hands[place.hand].frames[place.step].view.phase === 'call');
    assert.ok(callAt, 'an answer to a discard to reveal on a phone');
    await stepTo(page, keyOf(callAt));
    await keptBack(page, callAt);
    await press(page, 'details[data-review-index] > summary');
    await page.waitForFunction(() => !document.querySelector('details[data-review-index]').open, { timeout: 5000 });
    await seatUnderBars();
    const offered = await shownAt(page);
    await page.click('[data-decision] [data-pick]');
    await answered();
    await shownAgain(page, callAt, offered);

    // On a screen 500 pixels high or less the step bar stays in its place,
    // above the table. A tile of the live hand pressed by pointer answers the
    // decision and leaves the page where it was, the tile on the screen with
    // its answer, focus going on to Next out of sight above it; pressed by
    // keyboard, the page moves to show Next, below the game's bar.
    const [byPointerLow, byKeyLow] = choiceful
      .filter(place => ![byButton, byKey, byDigit, byTile, tilesOnly, phoneAt].includes(place) && digitOf(place) >= 0);
    assert.ok(byPointerLow && byKeyLow, 'two more turns to answer on a low screen');
    await page.setViewport({ width: 844, height: 390 });
    await settle(page);
    /** Where something lies on the screen, the focused control unless
     * `selector` names it, and how far down the game's bar covers it. */
    const lies = (selector = null) => page.evaluate(selector => {
      const box = (selector ? document.querySelector(selector) : document.activeElement).getBoundingClientRect();
      const bar = document.querySelector('header.bar');
      return { top: Math.round(box.top), bottom: Math.round(box.bottom), height: innerHeight,
        covered: getComputedStyle(bar).position === 'sticky' ? Math.round(bar.getBoundingClientRect().bottom) : 0 };
    }, selector);
    const seen = where => where.top >= where.covered - 1 && where.bottom <= where.height + 1;
    await stepTo(page, keyOf(byPointerLow));
    await keptBack(page, byPointerLow);
    const pointed = handTile(digitOf(byPointerLow));
    await page.$eval(pointed, tile => tile.scrollIntoView({ block: 'center' }));
    await settle(page);
    await page.click(pointed);
    await answered();
    await stepFocused();
    await settle(page);
    const tileThen = await lies(pointed), focusThen = await lies();
    assert.ok(seen(tileThen), `the tile pressed by pointer still on the screen at 844x390: ${JSON.stringify(tileThen)}`);
    assert.ok(focusThen.bottom <= focusThen.covered + 1, `the page not moved up to Next at 844x390: ${JSON.stringify(focusThen)}`);
    await stepTo(page, keyOf(byKeyLow));
    await keptBack(page, byKeyLow);
    await page.focus(handTile(digitOf(byKeyLow)));
    await page.keyboard.press('Enter');
    await answered();
    await stepFocused();
    await settle(page);
    const focusNow = await lies();
    assert.ok(seen(focusNow), `pressed by keyboard, the page moves to show Next at 844x390: ${JSON.stringify(focusNow)}`);
    await page.setViewport({ width: 1100, height: 900 });
    await settle(page);

    // Nothing of it is saved: after a reload the review asks again.
    await page.reload({ waitUntil: 'networkidle0' });
    await page.waitForSelector('.standings [data-review-game]');
    await reviewWholeGame(page);
    await page.waitForSelector(`[data-review-index] [data-hand="${byButton.hand}"][data-status="done"]`, { timeout: 300000 });
    await stepTo(page, keyOf(byButton));
    await keptBack(page, byButton);
    assert.deepEqual(page.errors, []);
  });

  await cases.check("an answer to another's discard says in the step bar whose tile it answers, outlines that tile in the seat's row, and names both moves, stepped to by keyboard, and with Trained AI's move kept back is answered by a call or Pass picked by keyboard, kept as Now", async () => {
    // The tile-throwing player passes every call, and Trained AI seldom
    // gives a pass under 5%, so this game's player calls everything instead.
    const page = await openPage(1100, 900, { saved: callingGame });
    await page.click('.standings [data-review-game]');
    await page.waitForSelector('[data-game-review]');
    await reviewed(page);
    // A flagged line named for the tile it answers.
    const key = await page.$$eval('[data-review-index] [data-flagged]', lines =>
      lines.find(line => /^Decision \d+\. On the /.test(line.getAttribute('aria-label')))?.dataset.flagged);
    assert.ok(key, 'a flagged answer to a discard');
    const [handIndex, index] = key.split(':').map(Number), hand = callingHands[handIndex], view = hand.frames[index].view;
    await stepTo(page, key, callingHands);
    await page.waitForSelector('[data-decision] [data-view]');
    /** What the step bar shows of the moves, and the tile on offer in the
     * row of the seat that let it go, by name, if any. */
    const shown = () => page.evaluate(() => ({
      line: document.querySelector('[data-game-review] [data-move-line]').textContent.replace(/\s+/g, ' ').trim(),
      said: document.querySelector('[data-game-review] [data-step-said]').textContent,
      outlined: [...document.querySelectorAll('[data-decision] [data-overview] .seat .offered .tile')].map(tile => tile.getAttribute('aria-label')),
    }));
    // Whose tile the player answered, in the bar and in the step's line; a
    // quad offered to be robbed holds every copy of its tile, so no row has
    // it to outline.
    const robbing = Boolean(quadOnOffer(view)), check = async () => {
      const now = await shown();
      assert.match(now.line, new RegExp(`^On ${SEAT_NAMES[view.pending_from]}'s${robbing ? ' kan' : ''} ?: You `), now.line);
      assert.match(now.line, / · Flagged$/);
      assert.ok(now.said.startsWith(`${hand.name}, decision ${index + 1} of ${hand.frames.length}. ${tableSaid(view)} You: `), now.said);
      assert.match(now.said, /\. Trained AI: .+\. Flagged\.$/);
      assert.equal(now.outlined.length, robbing ? 0 : 1, now.outlined.join(', '));
      if (!robbing) {
        assert.ok(now.outlined[0].startsWith(tileWords(view.pending_discard)) && now.outlined[0].endsWith(', the tile on offer'), now.outlined[0]);
      }
    };
    await check();
    // No tile in the hand was thrown at the answer, so none is tagged or ringed.
    assert.equal(await page.$$eval('[data-decision] .hand button:not(:disabled)', buttons => buttons.length), 0, 'no tile to throw');
    assert.equal(await page.$$eval('[data-decision] .hand :is(.tag b, .tile.selected)', marks => marks.length), 0, 'no tile tagged or ringed');
    // Away and back by keyboard, Previous or Next keeping focus: the same.
    const away = (await page.$('[data-game-review] [data-previous]:not(:disabled)')) ? '[data-previous]' : '[data-next]';
    const back = away === '[data-previous]' ? '[data-next]' : '[data-previous]';
    await page.focus(`[data-game-review] ${away}`);
    await page.keyboard.press('Enter');
    await page.waitForFunction(text => document.querySelector('[data-game-review] [data-step]')?.textContent !== text, { timeout: 5000 },
      placeOf(hand, index));
    await page.focus(`[data-game-review] ${back}`);
    await page.keyboard.press('Enter');
    await onPlace(page, hand, index);
    await check();
    assert.deepEqual(page.errors, []);
    await closePage(page);

    // With Trained AI's move kept back the same decision is put to the
    // player: no tile to throw, and each call and Pass a button under the
    // hand in play's words. Another move than the one played, pressed by
    // keyboard, is kept as Now and shown with the answer, and focus goes on
    // to Next as the buttons go.
    const quiz = await openPage(1100, 900, { saved: callingGame, settings: { version: 3, difficulty: 'beginner', hints: true, reviewHideMove: true } });
    await quiz.click('.standings [data-review-game]');
    await quiz.waitForSelector('[data-game-review]');
    await reviewed(quiz);
    await stepTo(quiz, key, callingHands);
    await keptBack(quiz, { hand: handIndex, step: index }, { tiles: false, game: callingHands });
    const { played, choices } = hand.frames[index];
    const choice = choices.find(entry => entry.kind !== 'discard' && !(entry.kind === played.kind && (entry.tile ?? null) === (played.tile ?? null)));
    await quiz.focus(`[data-decision] [data-pick][aria-label="${callLabel(choice, robbing)}"]`);
    await quiz.keyboard.press('Enter');
    await quiz.waitForFunction(() => !document.querySelector('[data-game-review] [data-reveal]'), { timeout: 5000 });
    const answer = await shownAt(quiz);
    assert.match(answer.line, new RegExp(`·\\s*Now ${moveToken(choice.kind, choice.tile ?? null).word}\\b`), answer.line);
    assert.ok(answer.said.includes(` Now: ${choice.label}`), answer.said);
    assert.deepEqual([answer.picks.length, answer.view], [0, true], 'the buttons go with the question, and the view comes');
    await quiz.waitForFunction(() => document.activeElement?.matches('[data-steps] :is([data-next], [data-previous])'), { timeout: 5000 })
      .catch(async () => assert.fail(`focus on Next, or Previous at the end, as the buttons went, not on ${await focusedName(quiz)}`));
    assert.deepEqual(quiz.errors, []);
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

  await cases.check('after a hand, its result opens the review of the game so far on the hand just played, which it asks Trained AI about first, keeps the game out of the trend, comes back to the result and opens again where the player left it, by keyboard and pointer at desktop and phone sizes', async () => {
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
    assert.equal(await page.$eval('#game-review-title', el => el.textContent), 'Game so far');
    assert.equal(await page.$eval('.match-view', el => getComputedStyle(el).display), 'none', 'the table and the result step aside');
    assert.equal(await page.$eval('[data-game-review] [data-close-review]', el => el.textContent.trim()), 'Back to the result');
    // It opens on the hand just played, at its first decision.
    assert.equal(await stepText(page), placeOf(latest, 0));
    await reviewed(page);

    // Every batch: the hand just played first, then the others from the
    // first deal, each group once.
    const batches = await page.evaluate(() => window.reviewBatches);
    assert.equal(walkBatches(batches, [latest.index, 0, 1]).walked, batches.length, 'no batch is missing or extra');
    const total = hands.slice(0, latest.index + 1).reduce((sum, hand) => sum + hand.frames.length, 0);
    assert.equal(await page.$eval('[data-game-review] .headline', el => el.textContent),
      `All hands reviewed: ${total} decision${total === 1 ? '' : 's'}.`);
    // The game so far is not a game for the trend, which says when it joins.
    const trend = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), TREND_KEY);
    assert.deepEqual(trend.games.map(game => game.id), ['earlier'], 'the game so far is not in the trend');
    assert.match(await page.$eval('[data-trend]', el => el.textContent), /Joins your trend once the game is over and fully reviewed\./);
    // The hand just played, with every hand face up, where the review opened.
    await overviewIsRight(page, latest, 0);
    await shot(page, 'game-review-so-far');

    // The player moves on, back a decision under every decision, by the key.
    await choose(page, 'every');
    await page.keyboard.press('ArrowLeft');
    await page.waitForFunction(text => document.querySelector('[data-game-review] [data-step]')?.textContent !== text, { timeout: 5000 },
      placeOf(latest, 0));
    const left = await stepText(page);

    // Back to the result as it was, with focus on the button that opened the
    // review, and the save untouched; open again, the review is where the
    // player left it.
    await page.focus('[data-game-review] [data-close-review]');
    await page.keyboard.press('Enter');
    await backOnResult(page);
    assert.notEqual(await page.$eval('.match-view', el => getComputedStyle(el).display), 'none', 'the table and the result are back');
    assert.ok(await tappable(page, button), 'the button is in view, uncovered');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    assert.equal(await stepText(page), left, 'opened again where the player left it');
    await page.focus('[data-game-review] [data-close-review]');
    await page.keyboard.press('Enter');
    await backOnResult(page);
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

  await cases.check("a review of the game so far goes on into the whole game's: Next hand stops it, the review opens again where the player left it, the hands it reviewed are not asked again, and the game joins the trend once, as it ends", async () => {
    const button = '.screen [data-review-game]', standing = '.standings [data-review-game]';
    const id = `${snapshot.seed}:${snapshot.commands.length}`;
    const trendIds = page => page.evaluate(key => (JSON.parse(localStorage.getItem(key)) ?? { games: [] }).games.map(game => game.id), TREND_KEY);
    const sent = page => page.evaluate(() => window.reviewBatches.length);
    const reviewedHands = page => page.$$eval('[data-review-index] [data-hand][data-status="done"]', items => items.map(item => Number(item.dataset.hand)));
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
    // The hand just played is asked about first: once it is reviewed, the
    // review goes on to the hands before it.
    await page.waitForSelector(`[data-review-index] [data-hand="${hands.at(-1).index}"][data-status="done"]`, { timeout: 300000 });
    assert.ok(await page.$('[data-game-review] [data-stop-review]'), 'the review goes on to the earlier hands');
    const left = await stepText(page);
    await backAndOn(page);
    // The deal that ended the game stopped the review, which asks nothing
    // more until it is opened again.
    const stopped = await sent(page);
    await new Promise(done => setTimeout(done, 2500));
    assert.equal(await sent(page), stopped, 'no batch is sent while the review is shut');
    assert.deepEqual(await trendIds(page), [], 'a game not reviewed whole is not in the trend');
    await page.click(standing);
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    assert.equal(await page.$eval('#game-review-title', el => el.textContent), 'Whole game');
    // The game ended without a hand more to look at, so the review opens
    // where the player left it.
    assert.equal(await stepText(page), left, 'it opens where the player left it');
    // The hands reviewed before Next hand, read before the review could
    // finish another, a batch taking a second to reach the worker: none of
    // them is asked about again, and every other hand is, once, from the
    // first deal.
    const done = await reviewedHands(page);
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
    assert.match(await page.$eval('[data-game-review] .headline', el => el.textContent), /^All hands reviewed: \d+ decisions?\.$/);
    await new Promise(done => setTimeout(done, 500));
    assert.equal(await sent(page), asked, 'nothing is asked again');
    assert.deepEqual(await trendIds(page), [id], 'and the game is in the trend once');
    assert.deepEqual(page.errors, []);
  });

  await cases.check("a review of the game so far goes on at the next hand's result: Next hand stops it as it runs, and that result's review opens on the hand just played, asks about it first, then only about the hands left waiting, none it had reviewed", async () => {
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
    await page.waitForSelector('[data-review-index] [data-hand][data-status="done"]', { timeout: 300000 });
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
    // That hand's result opens the review again, on that hand, at its first
    // decision. The hands reviewed before Next hand are read before the
    // review could finish another, a batch taking a second to reach the
    // worker.
    await page.click(button);
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    assert.equal(await stepText(page), placeOf(next, 0), 'the review opens on the hand just played');
    const done = await page.$$eval('[data-review-index] [data-hand][data-status="done"]', items => items.map(item => Number(item.dataset.hand)));
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

  await cases.check('the keys step, change hand, show what a decision keeps back, pick from its live hand and go back, and do nothing under a dialog, in a field, with a modifier held or with the shortcuts off', async () => {
    // Trained AI's move kept back, so that s and the digits have a decision
    // to answer; the arrows step all the same.
    let page = await openPage(1100, 900, { settings: { version: 3, difficulty: 'beginner', hints: true, reviewHideMove: true } });
    await reviewWholeGame(page);
    await reviewed(page);
    // Every decision, so that each key's step is known: a hand of three
    // decisions or more, and the next hand with one.
    await choose(page, 'every');
    const from = hands.find(hand => hand.frames.length >= 3 && hands.some(other => other.index > hand.index && other.frames.length));
    const after = hands.find(hand => hand.index > from.index && hand.frames.length);
    await stepTo(page, `${from.index}:1`);
    const key = async (name, shift = false) => {
      if (shift) await page.keyboard.down('Shift');
      await page.keyboard.press(name);
      if (shift) await page.keyboard.up('Shift');
    };
    // ← and → step as Previous and Next do; with Shift, to the hand after or
    // before, at its first decision the filter keeps.
    await key('ArrowRight');
    await onPlace(page, from, 2);
    await key('ArrowLeft');
    await onPlace(page, from, 1);
    await key('ArrowRight', true);
    await onPlace(page, after, 0);
    await key('ArrowLeft', true);
    await onPlace(page, from, 0);
    // Nothing with Ctrl, Alt or Meta held, or in a field. The keys are sent
    // as the page would have them, without the browser's own use of them,
    // such as Alt+← going back a page.
    const still = async (what, send, ...values) => {
      const before = await stepText(page);
      await page.evaluate(send, ...values);
      await settle(page);
      assert.equal(await stepText(page), before, what);
    };
    for (const modifier of ['ctrlKey', 'altKey', 'metaKey']) {
      await still(`→ with ${modifier}`, modifier => document.body.dispatchEvent(new KeyboardEvent('keydown',
        { key: 'ArrowRight', [modifier]: true, bubbles: true, cancelable: true })), modifier);
    }
    await still('→ in a select', () => document.querySelector('select').dispatchEvent(new KeyboardEvent('keydown',
      { key: 'ArrowRight', bubbles: true, cancelable: true })));
    // Under the table's dialog of discards the arrow does not step, s shows
    // nothing, and Esc closes the dialog, not the review.
    await press(page, '[data-decision] .inspect');
    await page.waitForFunction(() => document.querySelector('dialog[open]'), { timeout: 5000 });
    const shownBefore = (await shownAt(page)).reveal;
    await key('ArrowRight');
    await key('s');
    await settle(page);
    assert.equal(await stepText(page), placeOf(from, 0), '→ under a dialog');
    assert.equal((await shownAt(page)).reveal, shownBefore, 's under a dialog');
    await key('Escape');
    await page.waitForFunction(() => !document.querySelector('dialog[open]'), { timeout: 5000 });
    assert.ok(await page.$('[data-game-review]'), 'Esc closed the dialog, not the review');

    // At a turn that keeps its answer back, whose tile just drawn may be
    // thrown: s and the digits do nothing with a modifier held or in a field;
    // in the live hand ← and → move along the tiles, as in play, rather than
    // step; and 0 picks the tile just drawn, as play's 0 throws it, kept as
    // Now, focus going on to Next as the tile it was on becomes a picture
    // again.
    const drawnAt = hands.flatMap(hand => hand.frames.map((frame, step) => ({ hand: hand.index, step, frame })))
      .find(({ hand, step, frame }) => {
        const drawn = frame.view.seats[0].drawn, discards = frame.choices.filter(choice => choice.kind === 'discard');
        return !facts[hand][step].forced && frame.view.phase === 'act' && drawn && discards.length >= 2 && discards.some(choice => choice.tile === drawn);
      });
    assert.ok(drawnAt, 'a turn whose tile just drawn may be thrown');
    await stepTo(page, `${drawnAt.hand}:${drawnAt.step}`);
    await keptBack(page, drawnAt, { tiles: false });
    const kept = async (what, send, ...values) => {
      await page.evaluate(send, ...values);
      await settle(page);
      assert.equal((await shownAt(page)).reveal, 'Show answer', what);
    };
    for (const [name, modifier] of [['s', 'ctrlKey'], ['s', 'altKey'], ['0', 'metaKey'], ['1', 'ctrlKey']]) {
      await kept(`${name} with ${modifier}`, (name, modifier) => document.body.dispatchEvent(new KeyboardEvent('keydown',
        { key: name, [modifier]: true, bubbles: true, cancelable: true })), name, modifier);
    }
    await kept('s in a select', () => document.querySelector('select').dispatchEvent(new KeyboardEvent('keydown',
      { key: 's', bubbles: true, cancelable: true })));
    const tileFocused = () => page.evaluate(() => Number(document.activeElement?.dataset.handIndex ?? -1));
    await page.focus('[data-decision] .mine .hand button.tile:not(:disabled)');
    const firstTile = await tileFocused();
    await key('ArrowRight');
    assert.ok(await tileFocused() > firstTile, '→ moves along the live hand');
    assert.equal(await stepText(page), placeOf(hands[drawnAt.hand], drawnAt.step), '→ in the live hand does not step');
    await key('ArrowLeft');
    assert.equal(await tileFocused(), firstTile, '← moves back along it');
    await key('0');
    await page.waitForFunction(() => !document.querySelector('[data-game-review] [data-reveal]'), { timeout: 5000 });
    assert.equal((await shownAt(page)).now, drawnAt.frame.view.seats[0].hand.length, 'Now over the tile just drawn');
    await page.waitForFunction(() => document.activeElement?.matches('[data-steps] :is([data-next], [data-previous])'), { timeout: 5000 })
      .catch(async () => assert.fail(`focus on Next, or Previous at the end, as the tile went, not on ${await focusedName(page)}`));

    // Esc goes back to the standings, focus on the button that opened the
    // review.
    await key('Escape');
    await page.waitForFunction(() => !document.querySelector('[data-game-review]') && document.activeElement?.matches('.standings [data-review-game]'),
      { timeout: 5000 });
    assert.deepEqual(page.errors, []);
    await closePage(page);

    // Under the settings' dialog on a phone, not a touch one, the keys are
    // left to the dialog: the arrow does not step, s does not turn the other
    // players' tiles over, kept back here, and Esc closes the dialog. Once
    // it is closed, s turns them over. Only the keys are looked at here, so
    // Trained AI is out of reach, and the hands are not reviewed: only the
    // tiles are kept back.
    page = await openPage(390, 844, { network: false, settings: { version: 3, difficulty: 'beginner', hints: true, reviewHideTiles: true } });
    await page.$eval('.standings [data-review-game]', button => button.click());
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    await choose(page, 'every');
    const before = await stepText(page);
    assert.equal((await shownAt(page)).reveal, 'Show tiles');
    await page.click('.settings-trigger');
    await page.waitForFunction(() => document.querySelector('dialog.settings-dialog[open]'), { timeout: 5000 });
    await key('ArrowLeft');
    await key('s');
    await settle(page);
    assert.equal(await stepText(page), before, '← under the settings');
    await key('Escape');
    await page.waitForFunction(() => !document.querySelector('dialog[open]'), { timeout: 5000 });
    assert.ok(await page.$('[data-game-review]'), 'Esc closed the settings, not the review');
    assert.equal((await shownAt(page)).reveal, 'Show tiles', 's under the settings');
    await key('s');
    await page.waitForFunction(() => !document.querySelector('[data-game-review] [data-reveal]'), { timeout: 5000 });
    assert.ok((await shownAt(page)).racks.every(rack => !rack.label.endsWith(', face down') && rack.down === 0), 's turns the tiles over');
    assert.deepEqual(page.errors, []);
    await closePage(page);

    // With the shortcuts off, as with play's, no key does anything.
    page = await openPage(1100, 900, { network: false,
      settings: { version: 2, difficulty: 'beginner', hints: true, shortcuts: false, reviewHideTiles: true } });
    await page.click('.standings [data-review-game]');
    await page.waitForFunction(() => document.activeElement?.id === 'game-review-title', { timeout: 5000 });
    await choose(page, 'every');
    const at = await stepText(page);
    for (const name of ['ArrowLeft', 'ArrowRight', 'Escape', 's']) {
      await key(name);
      await settle(page);
      assert.ok(await page.$('[data-game-review]'), `${name} with the shortcuts off`);
      assert.equal(await stepText(page), at, `${name} with the shortcuts off`);
      assert.equal((await shownAt(page)).reveal, 'Show tiles', `${name} with the shortcuts off`);
    }
    assert.deepEqual(page.errors, []);
  });
} finally {
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
  await cases.report('whole-game review checks', resolve(output, 'game-review.json'));
}
