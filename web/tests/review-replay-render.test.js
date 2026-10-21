import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { render } from 'svelte/server';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { text } from './fixtures/markup-text.js';
import { MatchReview, moveToken, moveWeight, stepSaid } from '../src/lib/game-review.js';
import { OPPONENT_LABELS, OPPONENT_POSITIONS } from '../src/lib/opponents.js';
import { SEAT_NAMES, tileWords } from '../src/lib/tiles.js';
import { percent, quadOnOffer } from '../src/lib/ui.js';
import { componentLoader } from './fixtures/load-component.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const DecisionReplay = await componentLoader()('DecisionReplay.svelte');

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
const replay = props => render(DecisionReplay, { props: { hand, hints: true, onback() {}, ...props } }).body;

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
 * the player threw, "AI" over Trained AI's, the tile just drawn where it is
 * the one, and only Trained AI's ringed. */
function tagsOf(frame, decision) {
  const me = frame.view.seats[0], tiles = [...me.hand, ...(me.drawn ? [me.drawn] : [])];
  const place = move => (['discard', 'riichi'].includes(move.kind) ? (move.tile === me.drawn ? tiles.length - 1 : tiles.indexOf(move.tile)) : -1);
  const yours = place(frame.played), trained = decision && !decision.forced ? place({ kind: decision.advised_kind, tile: decision.advised_tile }) : -1;
  return tiles.map((_, index) => ({ word: [index === yours && 'You', index === trained && 'AI'].filter(Boolean).join(' · '), ringed: index === trained }));
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

test('every step of a replayed hand shows every hand face up, each a list named by its seat, both moves tagged, and no control hides them', () => {
  for (const [index, frame] of hand.frames.entries()) {
    const html = replay({ start: index, whole: true }), decision = hand.decisions[index];
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
    // or to show the answer, which is on show.
    assert.doesNotMatch(html, /class="held|class="back|type="checkbox"|data-peek|data-reveal/);
    assert.doesNotMatch(text(html), /(Show|Hide) (the )?(other players'|every|all) (tiles|hands)|Show answer/i);
    // The step bar says the step, the table and both moves as the step changes.
    assert.ok(html.includes(`role="status" data-step-said="">${stepSaid({ hand, step: index, decision })}</p>`), `the status at step ${index}`);
    assert.equal(moveLine(html).said, expectedLine(decision, frame), `the move line at step ${index}`);
    assert.match(html, new RegExp(`<div class="wall[^"]*"><strong[^>]*>${frame.view.wall}</strong>`));
    for (const indicator of frame.view.dora_indicators) assert.ok(html.includes(`data-tile="${indicator}"`));
    // Whose turn it is, in words: the player to play or to answer.
    assert.match(text(html), frame.view.phase === 'act' ? /To play/ : /To answer/);
    // The player's tile and Trained AI's tagged, Trained AI's ringed, and
    // Trained AI's view of the move wherever it was asked.
    assert.deepEqual(tagged(html), tagsOf(frame, decision), `the tags at step ${index}`);
    assert.equal(/data-view=""/.test(html), !decision.forced, `Trained AI's view at step ${index}`);
  }
});

test("a flagged decision shows Trained AI's move beside the player's, in the step bar, over its tile and in its view, and offers nothing to choose again", () => {
  const decision = hand.decisions[flaggedAt], frame = hand.frames[flaggedAt];
  const html = replay({ start: flaggedAt });
  // Where it is, and both moves, the flag after them.
  assert.match(html, new RegExp(`<p class="place[^"]*" data-step="">${hand.name} · <span class="long[^"]*">decision </span>`
    + `${flaggedAt + 1} of ${hand.frames.length}</p>`));
  const line = moveLine(html), advised = moveToken(decision.advised_kind, decision.advised_tile ?? null);
  assert.equal(line.said, expectedLine(decision, frame));
  assert.match(line.said, / · Flagged$/);
  assert.deepEqual(line.tiles, [...moveToken(frame.played.kind, frame.played.tile).tiles, ...advised.tiles].map(tileWords));
  assert.match(html, /<strong class="flag[^"]*">Flagged<\/strong>/);
  assert.match(html, /data-view=""/);
  assert.match(text(html), /Trained AI's view/);
  // Nothing of choosing again: no offer, no retry, no answer to wait for.
  assert.doesNotMatch(html, /data-answer=|data-offer=|data-reveal=|data-retry=|data-again=|data-choose-here=|>First<|>This decision</);
  assert.doesNotMatch(text(html), /Choose again|Show answer|Trained AI's answer/);
  // The replay goes on past it: every decision of the hand is a step, and
  // Next goes on unless the decision is the hand's last.
  assert.equal([...html.matchAll(/data-step-to="(\d+)"/g)].length, hand.frames.length);
  assert.match(html, new RegExp(`data-step-to="${flaggedAt}"[^>]*aria-current="step"`));
  assert.equal(/<button[^>]*\bdata-next\b[^>]*\bdisabled\b/.test(html), flaggedAt === hand.frames.length - 1);
});

test("a decision that was not flagged shows the move and Trained AI's view, said in words for the hand, with no bullets and no explainer", () => {
  const decision = hand.decisions[plainAt], frame = hand.frames[plainAt];
  const html = replay({ start: plainAt });
  assert.match(html, new RegExp(`data-step="">${hand.name} · <span class="long[^"]*">decision </span>${plainAt + 1} of ${hand.frames.length}</p>`));
  assert.equal(moveLine(html).said, expectedLine(decision, frame));
  assert.doesNotMatch(moveLine(html).said, /Flagged/);
  assert.match(html, /data-view=""/);
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

test("the hand's list gives every decision, each with the player's move and Trained AI's, by the filter, its name saying both in words", () => {
  const html = replay({ start: flaggedAt, whole: true });
  assert.match(html, /<h3 id="decision-title"[^>]*>[^<]+, whole hand<\/h3>/);
  // Each line a button to its step: its number, the player's move in tiles
  // and words, then Trained AI's after an arrow, or a tick where it agreed,
  // or that the move was the only one; Flagged where this list shows every
  // decision. Its name says the same in the engine's words.
  const items = [...html.matchAll(/<button class="app-control[^"]*" data-step-to="(\d+)"[^>]*aria-label="([^"]*)"[^>]*>([^]*?)<\/button>/g)]
    .map(([, step, label, inner]) => ({ step: Number(step), label, shown: text(inner), tiles: tileNames(inner) }));
  assert.equal(items.length, hand.frames.length);
  assert.match(html, /<div class="decision-list[^"]*">[^]*?<ol[^>]*\brole="list"[^>]*>(?:<!--[^>]*-->)*<li[^>]*><button class="app-control[^"]*" data-step-to="0"/);
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
  // Its filter opens on every decision and counts each choice.
  const filter = /role="group" aria-label="Which of this hand's decisions to list"[^>]*>([^]*?)<p class="visually-hidden[^"]*" role="status" data-filter-said=""><\/p>/.exec(html);
  assert.ok(filter, 'the filter, with its status line in the page from the start');
  const options = [...filter[1].matchAll(/aria-pressed="(true|false)" data-filter-option="(\w+)">([^]*?)<\/button>/g)]
    .map(([, pressed, option, label]) => [option, pressed, text(label)]);
  const flagged = hand.decisions.filter(decision => decision.judgement?.flagged).length;
  const different = hand.decisions.filter(decision => !decision.agreed && !decision.forced).length;
  assert.deepEqual(options, [['flagged', 'false', `Flagged (${flagged})`], ['differed', 'false', `Trained AI differed (${different})`],
    ['every', 'true', `Every decision (${hand.frames.length})`]]);
  // The Flagged filter's title says what flags a decision, here as in the
  // whole game's list.
  assert.match(filter[1], /<button[^>]*\btitle="Your move under 5%, Trained AI's choice 60% or more"[^>]*\bdata-filter-option="flagged"/);
});
