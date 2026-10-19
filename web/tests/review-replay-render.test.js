import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { render } from 'svelte/server';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { MatchReview, tableSaid } from '../src/lib/game-review.js';
import { OPPONENT_LABELS, OPPONENT_POSITIONS } from '../src/lib/opponents.js';
import { SEAT_NAMES, tileWords } from '../src/lib/tiles.js';
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

/** The text of a piece of markup, its tags and the renderer's markers gone,
 * a space where each tag was, as the page lays its elements apart. */
const text = markup => markup.replace(/<!--[^]*?-->/g, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
/** The names of the tiles shown in a piece of markup, in order. */
const tileNames = markup => [...markup.matchAll(/class="tile [^"]*"[^>]*aria-label="([^"]*)"/g)].map(match => match[1]);
/** The markup of the hands, each as the list a screen reader moves to: the
 * three opponents' lists, then the player's own. */
function handsIn(html) {
  const opponents = [...html.matchAll(/<ul class="revealed[^"]*" aria-label="([^"]*)" data-seat-hand="(\w+)">([^]*?)<\/ul>/g)]
    .map(([, label, seat, inner]) => ({ label, seat, tiles: tileNames(inner) }));
  const own = /<div class="hand[^"]*"[^>]*role="list" aria-label="([^"]*)"[^>]*>([^]*?)<div class="own-discards/.exec(html);
  return { opponents, own: own && { label: own[1], tiles: tileNames(own[2].split('class="my-melds')[0]) } };
}
/** A concealed hand as its tiles are named, the tile just drawn last and
 * saying so, without the marks a name goes on to give, such as dora. */
const named = held => [...held.hand.map(tileWords), ...(held.drawn ? [`${tileWords(held.drawn)}, just drawn`] : [])];
const unmarked = names => names.map(name => name.split(', ').filter((part, index) => !index || part === 'just drawn').join(', '));

test('every step of a replayed hand shows every hand face up, each a list named by its seat, and no control hides them', () => {
  for (const [index, frame] of hand.frames.entries()) {
    const html = replay({ start: index, whole: true });
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
    // Nothing turns them face down: no backs, and no control to show them.
    assert.doesNotMatch(html, /class="held|class="back|type="checkbox"|data-peek/);
    assert.doesNotMatch(text(html), /(Show|Hide) (the )?(other players'|every|all) (tiles|hands)/i);
    // The board and the step, and the line that says the table as it changes.
    assert.match(html, new RegExp(`<h4 id="overview-title"[^>]*>The table at your decision ${index + 1}</h4>`));
    assert.match(html, new RegExp(`role="status" data-table-said="">${tableSaid(frame.view).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}</p>`));
    assert.match(html, new RegExp(`<div class="wall[^"]*"><strong[^>]*>${frame.view.wall}</strong>`));
    for (const indicator of frame.view.dora_indicators) assert.ok(html.includes(`data-tile="${indicator}"`));
    // Whose turn it is, in words: the player to play or to answer.
    assert.match(text(html), frame.view.phase === 'act' ? /To play/ : /To answer/);
  }
});

test('a flagged decision opened to choose again waits for the choice with every hand in view, and goes no further', () => {
  const html = replay({ start: flaggedAt, retry: true });
  const { opponents, own } = handsIn(html);
  assert.equal(opponents.length, 3);
  assert.deepEqual(unmarked(own.tiles), named(hand.frames[flaggedAt].hands[0]));
  assert.match(html, new RegExp(`<span data-step="">Your decision ${flaggedAt + 1} of ${flaggedAt + 1}, the one flagged</span>`));
  assert.match(html, /What would you play here\? Choose again before you see Trained AI's answer/);
  assert.match(html, /data-reveal=""/);
  assert.doesNotMatch(html, /data-answer=|data-view=|Trained AI's view/, "Trained AI's answer waits");
  // The hand's list goes as far as the decision, which is the step shown.
  const steps = [...html.matchAll(/data-step-to="(\d+)"/g)].map(match => Number(match[1]));
  assert.deepEqual(steps, Array.from({ length: flaggedAt + 1 }, (_, step) => step));
  assert.match(html, new RegExp(`data-step-to="${flaggedAt}" aria-current="step"`));
});

test('a decision that was not flagged shows the move and Trained AI\'s view, and has no retry', () => {
  const html = replay({ start: plainAt });
  const decision = hand.decisions[plainAt];
  assert.match(html, new RegExp(`<span data-step="">Your decision ${plainAt + 1} of ${hand.frames.length}</span>`));
  assert.match(html, /data-view=""/);
  assert.match(text(html), /Trained AI's view/);
  assert.ok(text(html).includes(`In the game: ${decision.played}`));
  assert.ok(text(html).includes(`Trained AI: ${decision.advised}`));
  assert.match(html, /<p id="hand-help" class="prompt[^"]*">Your move: [^<]+\. (Trained AI chose (the same|differently)|It was the only move)\.<\/p>/);
  const retry = /[^]{0,160}(data-again=|data-reveal=|data-retry=|data-choose-here=|What would you play here)[^]{0,80}/.exec(html);
  assert.equal(retry?.[0] ?? null, null, 'no retry');
  assert.equal(handsIn(html).opponents.length, 3);
});

test('another flagged decision of the hand offers its retry or its answer, and the hand lists every decision by the filter', () => {
  const html = replay({ start: flaggedAt, whole: true });
  assert.match(html, /<h3 id="decision-title"[^>]*>[^<]+, decision by decision<\/h3>/);
  assert.doesNotMatch(html, />This decision</, 'a hand opened whole has no decision of its own');
  assert.match(html, /data-offer=""/);
  assert.match(html, /data-choose-here=""/);
  assert.match(html, /data-reveal=""/);
  assert.doesNotMatch(html, /data-answer=|data-view=/);
  assert.match(text(html), /Flagged: Trained AI gave (this move|this tile, once riichi was declared,) (&lt;)?[\d.]+%, and its own choice [\d.]+%\./);
  // The list holds every decision of the hand, each saying the move and
  // whether Trained AI agreed, the flagged ones saying so.
  const items = [...html.matchAll(/<button class="app-control[^"]*" data-step-to="(\d+)"[^>]*>([^]*?)<\/button>/g)]
    .map(([, step, inner]) => ({ step: Number(step), said: text(inner) }));
  assert.equal(items.length, hand.frames.length);
  for (const { step, said } of items) {
    const decision = hand.decisions[step];
    const verdict = decision.judgement?.flagged ? 'Flagged' : decision.forced ? 'Only move' : decision.agreed ? 'Trained AI agreed' : 'Trained AI differed';
    assert.ok(said.startsWith(`Decision ${step + 1} `) && said.endsWith(verdict), said);
  }
  // Its filter opens on every decision and counts each choice.
  const filter = /role="group" aria-label="Which of this hand's decisions to list"[^>]*>([^]*?)<p class="visually-hidden[^"]*" role="status" data-filter-said=""><\/p>/.exec(html);
  assert.ok(filter, 'the filter, with its status line in the page from the start');
  const options = [...filter[1].matchAll(/aria-pressed="(true|false)" data-filter-option="(\w+)">([^]*?)<\/button>/g)]
    .map(([, pressed, option, label]) => [option, pressed, text(label)]);
  const flagged = hand.decisions.filter(decision => decision.judgement?.flagged).length;
  const different = hand.decisions.filter(decision => !decision.agreed && !decision.forced).length;
  assert.deepEqual(options, [['flagged', 'false', `Flagged (${flagged})`], ['differed', 'false', `Trained AI chose differently (${different})`],
    ['every', 'true', `Every decision (${hand.frames.length})`]]);
});
