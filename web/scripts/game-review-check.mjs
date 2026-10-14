/** The whole-game review on the production build, with the trained worker.
 *
 * A finished match is restored from its save and reviewed hand by hand by the
 * shipped network. Every batch the page sends is captured with the worker's
 * answer, and the flags, their order and the agreement figure are worked out
 * again here from those answers alone. The leading decision is then replayed
 * from the deal, chosen again, and Trained AI's answer read back; the saved
 * match must be exactly as it was, and the game must enter the trend once.
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
import { createFixtureHandler } from './static-fixture-server.mjs';
import { runtimeDirectory } from './copy-runtime.mjs';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const web = fileURLToPath(new URL('../', import.meta.url)), dist = resolve(web, 'dist'), output = resolve(web, 'test-results');
assert.ok(existsSync(resolve(dist, `${await runtimeDirectory()}ort-wasm-simd-threaded.wasm`)), 'build the site first: npm run build');
const MORTAL_REACH = 37;

/** A whole game against Beginner players, by a player who takes every win,
 * declares every riichi and otherwise throws the tile just drawn: plenty for
 * Trained AI to disagree with. */
function finishedGame(seed) {
  const match = new MatchSession(Game, seed, 'beginner');
  try {
    match.advance(false);
    for (let step = 0; step < 5000 && !match.over; step++) {
      if (match.engine.hand_is_over()) { match.apply({ type: 'next' }); match.advance(false); continue; }
      const choices = match.choices, drawn = match.view.seats[0].drawn;
      const choice = choices.find(c => ['ron', 'tsumo', 'riichi'].includes(c.kind)) ?? choices.find(c => c.kind === 'pass')
        ?? choices.find(c => c.kind === 'discard' && c.tile === drawn) ?? choices.find(c => c.kind === 'discard') ?? choices[0];
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
    }
    assert.ok(match.over);
    return match.snapshot();
  } finally { match.dispose(); }
}
const snapshot = finishedGame(5);
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

async function openPage(width, height) {
  const page = await (await cases.openContext(browser)).newPage();
  page.errors = [];
  page.on('pageerror', error => page.errors.push(error.message));
  await page.setViewport({ width, height });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
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
  }, SAVE_KEY, SETTINGS_KEY, snapshot);
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/`, { waitUntil: 'networkidle0' });
  await page.waitForSelector('.standings [data-review-game]');
  return page;
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
    await reviewed(page);

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

    // The game enters the trend once, with the figure shown.
    const trend = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), TREND_KEY);
    assert.equal(trend.games.length, 1);
    assert.deepEqual({ ...trend.games[0], at: 0 }, { id: `${snapshot.seed}:${snapshot.commands.length}`, at: 0,
      matched, decisions: counted.length, flagged: flagged.length });

    // The leading decision, opened where it was made.
    const lead = flagged[0], [hand, index] = lead.key.split(':').map(Number), frames = hands[hand].frames;
    await page.click(`[data-game-review] [data-open="${lead.key}"]`);
    await page.waitForSelector('[data-decision]');
    const status = () => page.$eval('[data-decision] .where', el => el.textContent);
    assert.equal(await status(), `Your decision ${index + 1} of ${index + 1}, the one flagged`);
    assert.equal(await page.$('[data-answer]'), null, "the answer waits for the player's choice");
    const wall = () => page.$eval('[data-decision] .wall strong', el => Number(el.textContent));
    assert.equal(await wall(), frames[index].view.wall, 'the table as it was at the decision');
    // Replayed from the deal, a decision of the player's at a time.
    if (index > 0) {
      await page.evaluate(() => [...document.querySelectorAll('[data-decision] .steps button')].find(b => b.textContent === 'From the deal').click());
      assert.equal(await status(), `Your decision 1 of ${index + 1}`);
      assert.equal(await wall(), frames[0].view.wall);
      assert.match(await page.$eval('[data-decision] #hand-help', el => el.textContent), /Your move: /);
      await page.click('[data-decision] [aria-label="Next decision"]');
      assert.equal(await status(), `Your decision 2 of ${index + 1}${index === 1 ? ', the one flagged' : ''}`);
      assert.equal(await wall(), frames[1].view.wall);
      if (index > 1) await page.evaluate(() => [...document.querySelectorAll('[data-decision] .steps button')].find(b => b.textContent === 'This decision').click());
    }
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
    // Again from the same position, then back to the list where it was opened.
    await page.click('[data-answer] [data-again]');
    assert.equal(await page.$('[data-answer]'), null);
    assert.match(await page.$eval('[data-decision] #hand-help', el => el.textContent), /Choose again before you see/);
    await page.click('[data-decision] [data-back]');
    // Focus returns to the decision opened.
    await page.waitForFunction(key => document.activeElement?.dataset.open === key, {}, lead.key);

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
    await page.waitForFunction(() => !document.querySelector('[data-game-review]') && document.activeElement?.matches('[data-review-game]'));
    assert.notEqual(await page.$eval('.match-view', el => getComputedStyle(el).display), 'none');
    const { _storage, ...kept } = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
    assert.deepEqual(kept, snapshot, 'the save is unchanged');
    assert.deepEqual(page.errors, []);
  });
} finally {
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
  await cases.report('whole-game review checks', resolve(output, 'game-review.json'));
}
