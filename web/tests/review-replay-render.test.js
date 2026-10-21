import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { render } from 'svelte/server';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { text } from './fixtures/markup-text.js';
import { MatchReview, filterCounts, moveToken, moveWeight, sameMove, stepSaid } from '../src/lib/game-review.js';
import { OPPONENT_LABELS, OPPONENT_POSITIONS } from '../src/lib/opponents.js';
import { SEAT_NAMES, tileWords } from '../src/lib/tiles.js';
import { callLabel, percent, quadOnOffer } from '../src/lib/ui.js';
import { componentLoader } from './fixtures/load-component.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const component = componentLoader();
const DecisionReplay = await component('DecisionReplay.svelte');
const ReviewStepBar = await component('ReviewStepBar.svelte');
const ReviewIndex = await component('ReviewIndex.svelte');

/** A whole game against Club players, by a player who takes every win,
 * declares every riichi, calls every set and quad offered, and otherwise
 * throws the tile just drawn: hands that shrink by calls on both sides. */
function finishedGame(seed) {
  const match = new MatchSession(Game, seed, 'club');
  try {
    match.advance(false);
    for (let step = 0; step < 5000 && !match.over; step++) {
      if (match.engine.hand_is_over()) { match.apply({ type: 'next' }); match.advance(false); continue; }
      const choices = match.choices, drawn = match.view.seats[0].drawn;
      const choice = choices.find(c => ['ron', 'tsumo', 'riichi', 'concealed-kan', 'extended-kan', 'kan', 'pon', 'chii'].includes(c.kind))
        ?? choices.find(c => c.kind === 'discard' && c.tile === drawn) ?? choices.find(c => c.kind === 'discard') ?? choices[0];
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
    }
    assert.ok(match.over, 'the game reaches its standings');
    return match.snapshot();
  } finally { match.dispose(); }
}

/** A stand-in network sure of the last legal move its mask allows. */
function decisive(mask) {
  const legal = Array.from(mask).flatMap((allowed, index) => (allowed ? [index] : []));
  const action = legal.at(-1);
  return { action, weights: Array.from(mask, (allowed, index) => (!allowed ? 0 : index === action
    ? (legal.length === 1 ? 1 : 0.95) : 0.05 / (legal.length - 1))) };
}

const review = new MatchReview(Game, finishedGame(9), { analyzeBatch: async positions => positions.map(position => decisive(position.mask)) });
review.review();
while (review.busy) await new Promise(resolve => setImmediate(resolve));
assert.ok(review.complete, 'every hand is reviewed');
// The hand with the most decisions among those with both a flagged decision
// and one that was not.
const hand = review.hands.filter(entry => entry.decisions.some(decision => decision.judgement?.flagged)
  && entry.decisions.some(decision => !decision.forced && !decision.judgement?.flagged))
  .sort((a, b) => b.frames.length - a.frames.length)[0];
assert.ok(hand, 'a hand with flagged decisions and others');
const flaggedAt = hand.decisions.findIndex(decision => decision.judgement?.flagged);
const plainAt = hand.decisions.findIndex(decision => !decision.forced && !decision.judgement?.flagged);
const replay = props => render(DecisionReplay, { props: { hand, hints: true, ...props } }).body;
/** The review's step bar at a step of the hand, as the review draws it there. */
const stepBar = (step, values = {}) => render(ReviewStepBar, { props: { hand, step, decision: hand.decisions[step], previous: null, next: null,
  onstep() {}, ...values } }).body;

/** The names of the tiles shown in a piece of markup, in order. */
const tileNames = markup => [...markup.matchAll(/class="tile [^"]*"[^>]*aria-label="([^"]*)"/g)].map(match => match[1]);
/** The markup of the hands, each as the list a screen reader moves to: the
 * three opponents' lists, then the player's own. Each says it is a list in
 * so many words (`role="list"`), as Safari does not tell VoiceOver that a
 * list whose markers are taken away is one. */
function handsIn(html) {
  const opponents = [...html.matchAll(/<ul class="revealed[^"]*" role="list" aria-label="([^"]*)" data-seat-hand="(\w+)">([^]*?)<\/ul>/g)]
    .map(([, label, seat, inner]) => ({ label, seat, tiles: tileNames(inner) }));
  const own = /<div class="hand[^"]*"[^>]*role="list" aria-label="([^"]*)"[^>]*>([^]*?)<div class="own-discards/.exec(html);
  return { opponents, own: own && { label: own[1], markup: own[2].split('class="my-melds')[0], tiles: tileNames(own[2].split('class="my-melds')[0]) } };
}
/** A concealed hand as its tiles are named, the tile just drawn last and
 * saying so, without the marks a name goes on to give, such as dora. */
const named = held => [...held.hand.map(tileWords), ...(held.drawn ? [`${tileWords(held.drawn)}, just drawn`] : [])];
const unmarked = names => names.map(name => name.split(', ').filter((part, index) => !index || part === 'just drawn').join(', '));
/** The player's own tiles as the replay tags them: the words over each, and
 * whether it is ringed as Trained AI's choice. */
function tagged(html) {
  const { markup } = handsIn(html).own;
  const words = [...markup.matchAll(/<span class="tag[^"]*" aria-hidden="true">([^]*?)<\/span>/g)].map(match => text(match[1]));
  const ringed = [...markup.matchAll(/<button class="(tile [^"]*)"/g)].map(match => /\bselected\b/.test(match[1]));
  return words.map((word, index) => ({ word, ringed: ringed[index] }));
}
/** The tags a step's moves give the player's tiles: "You" over the tile
 * the player threw, "Now" over one they picked, "AI" over Trained AI's, the
 * tile just drawn where it is the one, and only Trained AI's ringed. A
 * decision whose answer is kept back gives Trained AI's none. */
function tagsOf(frame, decision, pick = null) {
  const me = frame.view.seats[0], tiles = [...me.hand, ...(me.drawn ? [me.drawn] : [])];
  const place = move => (move && ['discard', 'riichi'].includes(move.kind) ? (move.tile === me.drawn ? tiles.length - 1 : tiles.indexOf(move.tile)) : -1);
  const yours = place(frame.played), picked = place(pick);
  const trained = decision && !decision.forced ? place({ kind: decision.advised_kind, tile: decision.advised_tile }) : -1;
  return tiles.map((_, index) => ({ word: [index === yours && 'You', index === picked && 'Now', index === trained && 'AI'].filter(Boolean).join(' · '),
    ringed: index === trained }));
}
/** Each of the player's own tiles, whether a press picks it: not disabled. */
function liveTiles(html) {
  return [...handsIn(html).own.markup.matchAll(/<button class="tile [^"]*"([^>]*)>/g)].map(([, rest]) => !/(?:^|\s)disabled=""/.test(rest));
}
/** The buttons under the hand for the moves other than a discard: each
 * move's kind and what it is called. */
const picks = html => [...html.matchAll(/<button type="button" class="app-control[^"]*" data-pick="([^"]+)" aria-label="([^"]*)"/g)]
  .map(([, kind, label]) => ({ kind, label }));
/** Every decision of the review, with its hand, its step and its frame. */
const everyDecision = review.hands.flatMap(entry => entry.frames.map((frame, step) => ({ entry, step, frame, decision: entry.decisions[step] })));
/** The row of Trained AI's view for a move of a decision, by the engine's
 * words for it, and the words it is tagged with; null where it has none. */
function viewRow(html, decision, move) {
  const label = decision.weighted.find(choice => sameMove(choice, move.kind, move.tile)).label;
  const row = [...html.matchAll(/<div class="row[^"]*" role="listitem"[^>]*data-move="([^"]*)">([^]*?)<\/div>/g)].find(([, said]) => said === label);
  return row && { inner: row[2], tag: text(/<span class="tag[^"]*" aria-hidden="true">([^]*?)<\/span>/.exec(row[2])[1]) };
}
/** The step bar's line of moves, in words, and the tiles it shows. */
function moveLine(html) {
  const line = /<p class="moves[^"]*" data-move-line="">([^]*?)<\/p>/.exec(html)[1];
  return { said: text(line), tiles: tileNames(line) };
}
/** The line a decision's moves take in the step bar, as its words read: the
 * tiles are pictures, and say nothing here. */
function expectedLine(decision, frame) {
  const word = move => (move.word ? ` ${move.word}` : '');
  const weight = ({ weight: value }) => (value == null ? '' : ` ${percent(value)}`);
  const view = frame.view;
  const on = view.phase === 'call' ? `On ${SEAT_NAMES[view.pending_from]}'s${quadOnOffer(view) ? ' kan' : ''} : ` : '';
  const you = moveToken(frame.played.kind, frame.played.tile), yours = moveWeight(decision, frame.played);
  if (decision.forced) return `${on}You${word(you)} · Only move`;
  if (decision.agreed) return `${on}You${word(you)}${weight(yours)}${yours.given ? ', given riichi' : ''} · Trained AI agreed`;
  const advised = { kind: decision.advised_kind, tile: decision.advised_tile ?? null }, theirs = moveWeight(decision, advised);
  return `${on}You${word(you)}${weight(yours)} · Trained AI${word(moveToken(advised.kind, advised.tile))}${weight(theirs)}`
    + `${theirs.given ? ', given riichi' : ''}${decision.judgement?.flagged ? ' · Flagged' : ''}`;
}

test('every step of a reviewed hand shows every hand face up, each a list named by its seat, both moves tagged, and nothing that hides them', () => {
  for (const [index, frame] of hand.frames.entries()) {
    const html = replay({ step: index }), decision = hand.decisions[index];
    const { opponents, own } = handsIn(html), seats = frame.view.seats;
    assert.equal(opponents.length, 3, `three opponents' hands at step ${index}`);
    // MatchTable draws Opposite, then Left, then Right.
    for (const { label, seat, tiles } of opponents) {
      const slot = seats.findIndex(entry => entry.seat === seat);
      assert.ok(slot > 0);
      assert.equal(label, `${SEAT_NAMES[seat]}'s tiles, ${OPPONENT_POSITIONS[slot - 1]} (${OPPONENT_LABELS[seats[slot].controller]})`);
      assert.deepEqual(unmarked(tiles), named(frame.hands[slot]), `${seat}'s tiles at step ${index}`);
      assert.equal(tiles.length, seats[slot].hand_size);
    }
    assert.equal(own.label, `${SEAT_NAMES[seats[0].seat]}'s tiles, yours`);
    assert.deepEqual(unmarked(own.tiles), named(frame.hands[0]));
    // Nothing turns them face down: no backs, and no control to show them
    // or to show the answer, which is on show, nor to pick a move.
    assert.doesNotMatch(html, /class="held|class="back|type="checkbox"|data-peek|data-reveal|data-pick/);
    assert.doesNotMatch(text(html), /(Show|Hide) (the )?(other players'|every|all) (tiles|hands)|Show answer|Pick a move/i);
    assert.ok(liveTiles(html).every(live => !live), `no tile to pick at step ${index}`);
    assert.match(html, new RegExp(`<div class="wall[^"]*"><strong[^>]*>${frame.view.wall}</strong>`));
    for (const indicator of frame.view.dora_indicators) assert.ok(html.includes(`data-tile="${indicator}"`));
    // Whose turn it is, in words: the player to play or to answer.
    assert.match(text(html), frame.view.phase === 'act' ? /To play/ : /To answer/);
    // The player's tile and Trained AI's tagged, Trained AI's ringed, and
    // Trained AI's view of the move wherever it was asked.
    assert.deepEqual(tagged(html), tagsOf(frame, decision), `the tags at step ${index}`);
    assert.equal(/data-view=""/.test(html), !decision.forced, `Trained AI's view at step ${index}`);
    // The decision alone: the step bar and the list are the review's, above
    // it, and it has no heading or way back of its own.
    assert.doesNotMatch(html, /data-steps|data-step-to|data-decisions|data-back|decision-title|Back to the list/);
  }
});

test("with the other players' tiles kept back, each opponent's rack holds as many tiles face down, named as such and nothing more, and the player's own stay face up", () => {
  for (const [index, frame] of hand.frames.entries()) {
    const html = replay({ step: index, tilesHidden: true }), seats = frame.view.seats;
    const racks = [...html.matchAll(/<ul class="revealed[^"]*" role="list" aria-label="([^"]*)" data-seat-hand="(\w+)">([^]*?)<\/ul>/g)];
    assert.equal(racks.length, 3, `three racks at step ${index}`);
    for (const [, label, seat, inner] of racks) {
      const slot = seats.findIndex(entry => entry.seat === seat);
      assert.equal(label, `${SEAT_NAMES[seat]}'s tiles, ${OPPONENT_POSITIONS[slot - 1]} (${OPPONENT_LABELS[seats[slot].controller]}), face down`);
      assert.deepEqual(tileNames(inner), Array(seats[slot].hand_size).fill('face-down tile'), `${seat}'s rack at step ${index}`);
      // Nothing of a tile is left to give it away: no tile in an attribute,
      // and every name and title only that it is face down.
      assert.doesNotMatch(inner, /data-tile="|dora|ringed/);
      assert.deepEqual([...new Set([...inner.matchAll(/title="([^"]*)"/g)].map(match => match[1]))], ['face-down tile']);
    }
    assert.deepEqual(unmarked(handsIn(html).own.tiles), named(frame.hands[0]), `the player's own tiles at step ${index}`);
  }
});

test("with Trained AI's move kept back, a decision is put to the player: no view, no ring or AI over a tile, every legal discard live, the other moves under the hand, and the prompt", () => {
  for (const [index, frame] of hand.frames.entries()) {
    if (hand.decisions[index].forced) continue;
    const html = replay({ step: index, moveHidden: true });
    assert.doesNotMatch(html, /data-view=""/, `no view at step ${index}`);
    // The player's own move keeps its word; Trained AI's has neither ring
    // nor word.
    assert.deepEqual(tagged(html), tagsOf(frame, null), `the tags at step ${index}`);
    // Every legal discard is a tile to press, as in play, and no other tile.
    const me = frame.view.seats[0], legal = tile => frame.choices.some(choice => choice.kind === 'discard' && choice.tile === tile);
    assert.deepEqual(liveTiles(html), [...me.hand, ...(me.drawn ? [me.drawn] : [])].map(legal), `the live tiles at step ${index}`);
    // Every other move is a button under the hand, in play's words.
    const robbing = Boolean(quadOnOffer(frame.view));
    assert.deepEqual(picks(html), frame.choices.filter(choice => choice.kind !== 'discard')
      .map(choice => ({ kind: choice.kind, label: callLabel(choice, robbing) })), `the other moves at step ${index}`);
    assert.match(html, /<p id="hand-help" class="prompt[^"]*">Pick a move, or Show answer\.<\/p>/);
    assert.doesNotMatch(text(html), /Trained AI's choice|Trained AI's view/);
  }
});

test("an answer to another's discard, its answer kept back, offers each call and Pass under the hand in play's words, and no tile", () => {
  const call = everyDecision.find(({ frame, decision }) => frame.view.phase === 'call' && !decision.forced);
  assert.ok(call, 'an answer to a discard that was a choice');
  const html = render(DecisionReplay, { props: { hand: call.entry, step: call.step, hints: true, moveHidden: true } }).body;
  const robbing = Boolean(quadOnOffer(call.frame.view)), offered = call.frame.choices.filter(choice => choice.kind !== 'discard');
  assert.deepEqual(picks(html), offered.map(choice => ({ kind: choice.kind, label: callLabel(choice, robbing) })));
  assert.ok(picks(html).some(({ kind, label }) => kind === 'pass' && label === 'Pass'), 'Pass among them');
  assert.ok(liveTiles(html).every(live => !live), 'no tile is thrown in answer to a discard');
  // Picked, the answer is shown, the pick tagged Now in Trained AI's view.
  const choice = offered.find(entry => !sameMove(entry, call.decision.advised_kind, call.decision.advised_tile)) ?? offered[0];
  const pick = { kind: choice.kind, tile: choice.tile ?? null };
  const shown = render(DecisionReplay, { props: { hand: call.entry, step: call.step, hints: true, pick } }).body;
  assert.match(shown, /data-view=""/);
  assert.doesNotMatch(shown, /data-pick|Pick a move/);
  const row = viewRow(shown, call.decision, pick);
  assert.ok(row, `the pick's row: ${choice.kind}`);
  assert.match(row.tag, /\bNow\b/);
});

test("a move picked before the answer was shown is tagged Now over its tile, in Trained AI's view and in a column of the numbers where the review measured it", () => {
  // A flagged turn, where the review measured every move, with a legal
  // discard of a tile neither the player nor Trained AI threw.
  const other = ({ frame, decision }) => frame.choices.find(choice => choice.kind === 'discard'
    && choice.tile !== decision.played_tile && choice.tile !== decision.advised_tile);
  const target = everyDecision.find(entry => entry.frame.view.phase === 'act' && entry.decision.judgement?.flagged && other(entry));
  assert.ok(target, 'a flagged turn with a third discard');
  const pick = { kind: 'discard', tile: other(target).tile };
  const html = render(DecisionReplay, { props: { hand: target.entry, step: target.step, hints: true, pick } }).body;
  assert.deepEqual(tagged(html), tagsOf(target.frame, target.decision, pick));
  assert.ok(tagged(html).some(tile => tile.word === 'Now'), 'Now over the tile picked');
  const row = viewRow(/<section class="view[^"]*"[^>]*data-view="">([^]*?)<\/section>/.exec(html)[1], target.decision, pick);
  assert.ok(row, "the pick's row is listed");
  assert.equal(row.tag, 'Now');
  assert.ok(row.inner.includes(', your pick'), 'heard as the pick');
  // What each move leaves gives a column to each move the review measured,
  // the pick between the player's and Trained AI's.
  const heads = markup => [...(/<details class="numbers[^"]*"[^>]*>([^]*?)<\/details>/.exec(markup)?.[1] ?? '')
    .matchAll(/<th scope="col"[^>]*>([^<]*)/g)].map(match => match[1]).filter(Boolean);
  const measured = move => target.decision.measures?.some(measure => sameMove(measure, move.kind, move.tile));
  const advised = { kind: target.decision.advised_kind, tile: target.decision.advised_tile ?? null };
  const played = { kind: target.decision.played_kind, tile: target.decision.played_tile ?? null };
  assert.deepEqual(heads(html), [['You', played], ['Now', pick], ['Trained AI', advised]].filter(([, move]) => measured(move)).map(([title]) => title));
  assert.ok(heads(html).includes('Now'), 'a flagged turn measures every move');
  // A pick the review did not measure has no column, and the others keep theirs.
  const unmeasured = { ...target.entry, decisions: target.entry.decisions.map((entry, at) => (at !== target.step ? entry
    : { ...entry, measures: entry.measures.filter(measure => !sameMove(measure, pick.kind, pick.tile)) })) };
  const without = render(DecisionReplay, { props: { hand: unmeasured, step: target.step, hints: true, pick } }).body;
  assert.deepEqual(heads(without), heads(html).filter(title => title !== 'Now'));
});

test("the step bar gives each decision of a reviewed hand, the move made beside Trained AI's with the weights the review gave them, and says it in one line", () => {
  for (const [index, frame] of hand.frames.entries()) {
    const html = stepBar(index), decision = hand.decisions[index];
    assert.match(html, new RegExp(`<p class="place[^"]*" data-step="">${hand.name} · <span class="long[^"]*">decision </span>`
      + `${index + 1} of ${hand.frames.length}</p>`));
    assert.equal(moveLine(html).said, expectedLine(decision, frame), `the move line at step ${index}`);
    assert.ok(html.includes(`role="status" data-step-said="">${stepSaid({ hand, step: index, decision })}</p>`), `the status at step ${index}`);
  }
  // A flagged decision gives both moves' tiles, and the flag after them.
  const decision = hand.decisions[flaggedAt], frame = hand.frames[flaggedAt], line = moveLine(stepBar(flaggedAt));
  assert.match(line.said, / · Flagged$/);
  assert.deepEqual(line.tiles, [...moveToken(frame.played.kind, frame.played.tile).tiles,
    ...moveToken(decision.advised_kind, decision.advised_tile ?? null).tiles].map(tileWords));
});

test("a flagged decision shows Trained AI's move over its tile and in its view, and offers nothing to choose again", () => {
  const html = replay({ step: flaggedAt });
  assert.match(html, /data-view=""/);
  assert.match(html, /<h3 id="view-title"[^>]*>Trained AI's view<\/h3>/);
  // Nothing of choosing again: no offer, no retry, no answer to wait for.
  assert.doesNotMatch(html, /data-answer=|data-offer=|data-reveal=|data-retry=|data-again=|data-choose-here=|>First<|>This decision</);
  assert.doesNotMatch(text(html), /Choose again|Show answer|Trained AI's answer/);
});

test("on a phone held upright the player's seat comes before the table, the first thing under the step bar, and after it everywhere else", () => {
  const order = html => (html.indexOf('class="mine') < html.indexOf('class="board') ? 'seat first' : 'table first');
  // Measuring nothing, a server render draws it as a larger screen does.
  assert.equal(order(replay({ step: flaggedAt })), 'table first');
  // A phone held upright, as the decision reads its media query while it draws.
  globalThis.matchMedia = query => ({ matches: query === '(max-width: 760px) and (orientation: portrait)', addEventListener() {}, removeEventListener() {} });
  try {
    const html = replay({ step: flaggedAt });
    assert.equal(order(html), 'seat first');
    // Drawn there rather than moved: the seat and what describes its hand come
    // once, before the table, in the order a screen reader goes through them.
    assert.equal(html.match(/<section class="mine/g).length, 1);
    assert.equal(html.match(/id="hand-help"/g).length, 1);
    assert.ok(html.indexOf('id="hand-help"') < html.indexOf('class="board'));
    assert.deepEqual(tagged(html), tagsOf(hand.frames[flaggedAt], hand.decisions[flaggedAt]), 'tagged as anywhere');
  } finally { delete globalThis.matchMedia; }
});

test("a decision that was not flagged shows the move and Trained AI's view, said in words for the hand, with no bullets and no explainer", () => {
  const html = replay({ step: plainAt });
  assert.match(html, /data-view=""/);
  assert.doesNotMatch(moveLine(stepBar(plainAt)).said, /Flagged/);
  // The view lists the moves by weight, marking Trained AI's choice and the
  // move made, without the lines that said them over again or the line that
  // said what a percentage is, which the tile guide says.
  assert.match(text(html), /Trained AI's choice/);
  assert.doesNotMatch(html, /class="verdicts|class="help/);
  assert.doesNotMatch(text(html), /In the game:|This time|Percentages are preferences/);
  // The hand is described by both moves in words.
  assert.match(html, /<p id="hand-help" class="visually-hidden[^"]*">(On the [^<.]+\. )?You: [^<]+\. (Trained AI agreed|Trained AI: [^<]+)\.<\/p>/);
  assert.equal(handsIn(html).opponents.length, 3);
});

test("Trained AI's view gives each move as its tile or word, tagged as the hand is, and keeps the events and its numbers folded, a column only for each move measured", () => {
  let tables = 0;
  for (const entry of review.hands) {
    for (const [index, decision] of entry.decisions.entries()) {
      const html = render(DecisionReplay, { props: { hand: entry, step: index, hints: true } }).body;
      // What happened since the previous decision starts folded.
      assert.doesNotMatch(html, /<details class="lines[^"]*" open/);
      if (decision.forced) continue;
      const view = /<section class="view[^"]*"[^>]*data-view="">([^]*?)<\/section>/.exec(html)[1];
      const played = decision.weighted.find(choice => choice.kind === decision.played_kind && (choice.tile ?? null) === (decision.played_tile ?? null));
      // Each row: the move as a picture for the eye and in the engine's words
      // for a screen reader, its weight, and the words it is tagged with.
      const rows = [...view.matchAll(/<div class="row[^"]*" role="listitem"[^>]*data-move="([^"]*)">([^]*?)<\/div>/g)]
        .map(([, move, inner]) => ({ move, inner, tag: text(/<span class="tag[^"]*" aria-hidden="true">([^]*?)<\/span>/.exec(inner)[1]) }));
      assert.ok(rows.length >= 1);
      for (const row of rows) {
        assert.match(row.inner, /<span class="move[^"]*"><span aria-hidden="true">/, row.move);
        assert.ok(row.inner.includes(`<span class="visually-hidden">${row.move}`), row.move);
        assert.doesNotMatch(row.inner, /<meter[^>]*aria-label/, 'the bar says no more than the percentage');
        const words = [row.move === played.label && 'You', row.move === decision.advised && 'AI'].filter(Boolean);
        assert.equal(row.tag, words.join(' · '), `${row.move} at ${entry.name}, decision ${index + 1}`);
      }
      assert.doesNotMatch(view, /class="marks|In the game/);
      // What each move leaves, folded, with a column for the player's move
      // and Trained AI's each where it was measured.
      const advised = { kind: decision.advised_kind, tile: decision.advised_tile ?? null };
      const measured = [{ title: 'You', move: { kind: decision.played_kind, tile: decision.played_tile ?? null } },
        ...(decision.agreed ? [] : [{ title: 'Trained AI', move: advised }])]
        .filter(({ move }) => decision.measures?.some(measure => measure.kind === move.kind && measure.tile === move.tile));
      const numbers = /<details class="numbers[^"]*"( open="")?>([^]*?)<\/details>/.exec(view);
      if (measured.length < 2) {
        assert.equal(numbers, null, `no table at ${entry.name}, decision ${index + 1}`);
        continue;
      }
      tables++;
      assert.equal(numbers[1], undefined, 'folded');
      assert.match(numbers[2], /<summary[^>]*>What each move leaves<\/summary>/);
      assert.deepEqual([...numbers[2].matchAll(/<th scope="col"[^>]*>([^<]*)/g)].map(match => match[1]).filter(Boolean),
        measured.map(({ title }) => title));
    }
  }
  assert.ok(tables > 0, 'some decision compares what each move leaves');
});

test("the list gives each decision of a reviewed hand a line with the player's move and Trained AI's, its name saying both in words", () => {
  const decisions = review.hands.flatMap(entry => entry.decisions);
  const counts = { ...filterCounts(decisions), every: decisions.length };
  const html = render(ReviewIndex, { props: { hands: review.hands, place: { hand: hand.index, step: flaggedAt }, filter: 'every', counts,
    hints: true, trainedAvailable: true, onplace() {}, onreview() {} } }).body;
  // The hand's own list: a line a decision, a button to it, giving its
  // number, the player's move in tiles and words, then Trained AI's after an
  // arrow, or a tick where it agreed, or that the move was the only one, and
  // Flagged where the list keeps more than the flagged decisions. Its name
  // says the same in the engine's words.
  const lines = [...html.matchAll(/<button type="button" class="line[^"]*" data-step-to="(\d+):(\d+)"([^>]*)>([^]*?)<\/button>/g)]
    .map(([, handIndex, step, rest, inner]) => ({ hand: Number(handIndex), step: Number(step), label: /aria-label="([^"]*)"/.exec(rest)[1],
      current: /aria-current="step"/.test(rest), shown: text(inner), tiles: tileNames(inner) }));
  assert.equal(lines.length, decisions.length, 'every decision of the game, every hand reviewed');
  assert.deepEqual(lines.filter(line => line.current).map(line => [line.hand, line.step]), [[hand.index, flaggedAt]], 'the decision on show is current');
  const items = lines.filter(line => line.hand === hand.index);
  assert.deepEqual(items.map(item => item.step), hand.frames.map((_, step) => step));
  for (const { step, label, shown, tiles } of items) {
    const decision = hand.decisions[step], frame = hand.frames[step], you = moveToken(frame.played.kind, frame.played.tile);
    const ai = decision.forced || decision.agreed ? null : moveToken(decision.advised_kind, decision.advised_tile ?? null);
    const offered = frame.view.phase === 'call' ? 'on ' : '';
    const ending = decision.forced ? 'Only move' : decision.agreed ? '✓' : `→${ai.word ? ` ${ai.word}` : ''}`;
    assert.equal(shown, `${step + 1} ${offered}${you.word ? `${you.word} ` : ''}${ending}${decision.judgement?.flagged ? ' Flagged' : ''}`,
      `the line of decision ${step + 1}`);
    assert.deepEqual(unmarked(tiles), [...(offered ? [frame.view.pending_discard] : []), ...you.tiles, ...(ai?.tiles ?? [])]
      .map(tileWords), `the tiles of decision ${step + 1}`);
    const answer = decision.forced ? 'Only move.' : decision.agreed ? 'Trained AI agreed.' : `Trained AI: ${decision.advised}.`;
    assert.ok(label.startsWith(`Decision ${step + 1}. `) && label.endsWith(`${answer}${decision.judgement?.flagged ? ' Flagged.' : ''}`), label);
    assert.match(label, /You: [a-z]/, label);
  }
});
