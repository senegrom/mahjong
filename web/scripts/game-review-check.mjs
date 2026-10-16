/** The whole-game review on the production build, with the trained worker.
 *
 * A finished match is restored from its save and reviewed hand by hand by the
 * shipped network. Every batch the page sends is captured with the worker's
 * answer, and the flags, their order and the agreement figure are worked out
 * again here from those answers alone. A flagged decision is then replayed
 * from the deal by keyboard, chosen again, and Trained AI's answer read back,
 * with keyboard focus checked after every button the replay offers; the saved
 * match must be exactly as it was, and the game must enter the trend once,
 * under the version of Trained AI that judged it. A flagged answer to another's
 * discard, from a game whose player calls everything, is chosen again from the
 * moves its prompt points to, by keyboard. While the review goes on, a flagged
 * decision that later hands rank out of the leading five keeps keyboard focus,
 * in the list and back from its replay, and the list says where it went. On a
 * phone, where the final result covers the standings, the review must be
 * reachable from the result itself by touch and come back to the standings;
 * held sideways, from the standings beside the result; and from the
 * standings' own button under the result, by keyboard, it must leave the page
 * as the result's own button does.
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
import { BATCH, FLAG_PLAYED_BELOW, FLAG_PREFERRED_FROM, LEADING, scanMatch } from '../src/lib/game-review.js';
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
const shot = (page, name) => page.screenshot({ path: resolve(output, `${name}.png`), fullPage: true });
const overflow = page => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1);
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
 * Trained AI out of reach and never downloads it. */
async function openPage(width, height, { touch = false, network = true, saved = snapshot } = {}) {
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
  await page.evaluateOnNewDocument((saveKey, settingsKey, saved) => {
    if (!localStorage.getItem(saveKey)) localStorage.setItem(saveKey, JSON.stringify(saved));
    if (!localStorage.getItem(settingsKey)) localStorage.setItem(settingsKey, JSON.stringify({ version: 2, difficulty: 'beginner', hints: true }));
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
        if (data?.count) window.reviewBatches.push({ id: data.id, count: data.count, mask: Array.from(data.mask) });
        super.postMessage(data, ...rest);
      }
    };
  }, SAVE_KEY, SETTINGS_KEY, saved);
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/`, { waitUntil: 'networkidle0' });
  await page.waitForSelector('.standings [data-review-game]');
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

/** Opens the review and has it go through the whole game unheld, however
 * slow the device: Continue asks for the whole game, as a player who wants
 * all of it would. Both are pressed without moving focus, which stays where
 * the review put it. */
async function reviewWholeGame(page) {
  await page.click('.standings [data-review-game]');
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

/** Waits for every hand to be reviewed, going on past a slow device's hold. */
async function reviewed(page) {
  const deadline = Date.now() + 300000;
  while (Date.now() < deadline) {
    const state = await page.evaluate(() => ({
      headline: document.querySelector('[data-game-review] .headline')?.textContent ?? '',
      rest: Boolean(document.querySelector('[data-game-review] [data-rest]')),
    }));
    if (state.headline.startsWith('Every hand is reviewed')) return;
    assert.doesNotMatch(state.headline, /The review stopped/, state.headline);
    if (state.rest) await page.click('[data-game-review] [data-rest]');
    await new Promise(done => setTimeout(done, 500));
  }
  throw new Error('The whole-game review did not finish in five minutes');
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

    // Every batch, in the order the review asks: a hand at a time, eight
    // decisions a group, forced moves unasked, then each group's declarations.
    const batches = await page.evaluate(() => window.reviewBatches);
    const first = new Map(), second = new Map();
    let cursor = 0;
    facts.forEach((decisions, hand) => {
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
    });
    assert.equal(cursor, batches.length, 'no batch is missing or extra');

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

  await cases.check('a flagged decision keeps keyboard focus while hands reviewed later rank it out of the leading five, and the list says so', async () => {
    const page = await openPage(1100, 900);
    await reviewWholeGame(page);
    const key = await lastOfLeading(page);
    await page.focus(`[data-game-review] [data-open="${key}"]`);
    // Until a later hand ranks a decision that departed further above it,
    // out of the leading five, or focus leaves it, or the review ends.
    await page.waitForFunction((key, leading) => {
      const active = document.activeElement, keys = [...document.querySelectorAll('[data-game-review] [data-flagged]')].map(item => item.dataset.flagged);
      return active?.dataset.open !== key || keys.indexOf(key) >= leading
        || /^Every hand is reviewed/.test(document.querySelector('[data-game-review] .headline')?.textContent ?? '');
    }, { timeout: 300000, polling: 50 }, key, LEADING);
    assert.equal(await focusedDecision(page), key, 'focus stays on the decision the player is on');
    const keys = await listed(page), rank = keys.indexOf(key);
    assert.ok(rank >= LEADING, `hands reviewed later rank ${key} out of the leading ${LEADING}: ${keys.join(', ')}`);
    // It is listed still, as every flagged decision is, and the status says
    // where it went and why the list grew.
    assert.equal(await page.$eval('[data-game-review] [data-every-flag]', el => el.textContent), `Show the first ${LEADING}`);
    assert.equal(await page.$eval('[data-game-review] [data-moved]', el => el.getAttribute('role')), 'status');
    assert.equal(await page.$eval('[data-game-review] [data-moved]', el => el.textContent),
      `Decisions reviewed since departed further from Trained AI: this one is now ${nth(rank + 1)} of ${keys.length}, and every flagged decision is shown.`);
    // Focus moving on clears the note, which spoke of the decision it left.
    await page.focus('[data-game-review] [data-every-flag]');
    await page.waitForFunction(() => document.querySelector('[data-game-review] [data-moved]')?.textContent === '', { timeout: 5000 });
    assert.deepEqual(page.errors, []);
  });

  await cases.check('back from a decision the review ranked out of the leading five while it was open, focus returns to it and the list says where it went', async () => {
    const page = await openPage(1100, 900);
    await reviewWholeGame(page);
    const key = await lastOfLeading(page);
    await page.click(`[data-game-review] [data-open="${key}"]`);
    await page.waitForSelector('[data-decision]');
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
    assert.equal(await page.$eval('[data-game-review] [data-moved]', el => el.textContent),
      `Decisions reviewed since departed further from Trained AI: this one is now ${nth(rank + 1)} of ${keys.length}, and every flagged decision is shown.`);
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
      // The full-page screenshot can leave the page scrolled elsewhere: in
      // Chrome 155 at 320x568 it put the button under the result's chip, and
      // the tap showed the result instead. So it is brought back first.
      await page.$eval(standing, el => el.scrollIntoView({ block: 'center' }));
      assert.ok(await tappable(page, standing), `the standings' button can be tapped again at ${width}x${height}`);
      await page.tap(standing);
      await page.waitForSelector('[data-game-review]');
      assert.deepEqual(page.errors, []);
    }
  });

  await cases.check("on a phone, the standings' own button under the result sheet, reached by keyboard, opens the review as the sheet's does and comes back to the standings uncovered", async () => {
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
      assert.deepEqual(page.errors, []);
    }
  });
} finally {
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
  await cases.report('whole-game review checks', resolve(output, 'game-review.json'));
}
