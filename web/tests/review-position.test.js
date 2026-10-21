import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis } from '../src/wasm/riichi.js';
import { eachDecision, reviewCheckGame } from './fixtures/review-check-game.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

const snapshot = reviewCheckGame(Game);
const decisions = visit => eachDecision(Game, snapshot, visit);
const WINDS = ['east', 'south', 'west', 'north'];

/** Two of the engine's answers are the same, value for value. */
const same = (a, b) => a.length === b.length && Array.from(a).every((value, at) => value === b[at]);
/** The planes of thirty-four on which two observations differ. */
function planesDiffering(want, got) {
  const planes = new Set();
  for (let at = 0; at < want.length; at++) if (want[at] !== got[at]) planes.add(Math.floor(at / 34));
  return [...planes];
}
/** Trained AI's reading of a typed-in table: Mortal's planes, from the
 * table replayed as the hand it describes. */
function planesOf(position) {
  const analysis = new PhysicalAnalysis(position);
  try { return analysis.agent_observation_mortal(); } finally { analysis.free(); }
}
/** The moves a seat could make, as the review lists them. */
const moves = choices => choices.map(({ index, kind, tile }) => ({ index, kind, tile: tile ?? null }));

test('every decision of the review check\'s game, written as a position, asks Trained AI what the review asked', () => {
  const wrong = [];
  let count = 0, quads = 0;
  decisions(({ hand, index, engine, position }) => {
    count++;
    quads += Number(position.players.some(player => player.melds.some(meld => Number.isInteger(meld.order))));
    const analysis = new PhysicalAnalysis(position);
    try {
      const at = `${hand.name}, decision ${index + 1}`;
      const want = engine.review_observation_mortal(index), got = analysis.agent_observation_mortal();
      if (!same(want, got)) wrong.push(`${at}: planes ${planesDiffering(want, got).slice(0, 8).join(', ')}`);
      const mask = engine.review_mask_mortal(index);
      if (!same(mask, analysis.agent_mask_mortal())) wrong.push(`${at}: mask`);
      // A riichi is two questions; the second is asked of a copy declared.
      if (mask[37] && !(same(engine.review_observation_after_reach(index), analysis.agent_observation_after_reach())
        && same(engine.review_mask_after_reach(index), analysis.agent_mask_after_reach()))) wrong.push(`${at}: after a riichi`);
      assert.deepEqual(moves(analysis.agent_choices()), moves(engine.review_choices(index)), at);
    } finally { analysis.free(); }
  });
  assert.equal(count, 139, 'the review check\'s game has 139 decisions');
  assert.deepEqual(wrong, []);
  // North's concealed quad in East 2 · 1 honba stood through thirteen of
  // them; the twelve after the turn that declared it read wrong while a
  // typed-in table could not say when that was.
  assert.equal(quads, 13);
});

test('a written position holds what the deciding player saw, with the game\'s seating, and nothing hidden', () => {
  let count = 0;
  decisions(({ view, engine, index, position }) => {
    // The seating numbers each seat's player as the view does.
    assert.deepEqual(position.seating, WINDS.map(wind => view.seats.find(seat => seat.seat === wind).player));
    const mine = view.seats[0];
    assert.equal(position.seat, WINDS.indexOf(mine.seat));
    assert.deepEqual([...position.players[position.seat].hand].sort(), [...mine.hand, ...(mine.drawn ? [mine.drawn] : [])].sort());
    position.players.forEach((player, seat) => {
      const shown = view.seats.find(entry => entry.seat === WINDS[seat]);
      if (seat !== position.seat) assert.deepEqual(player.hand, [], 'another seat\'s tiles stay hidden');
      assert.deepEqual(player.discards.map(({ tile, drawn, riichi, claimed }) => ({ tile, drawn, riichi, claimed })), shown.discards);
      assert.equal(player.score, shown.score);
    });
    assert.deepEqual(position.indicators, view.dora_indicators);
    assert.equal(position.wall, view.wall);
    // Written again it is the same: nothing in it depends on being read.
    assert.deepEqual(engine.review_position(index), position);
    count++;
  });
  assert.equal(count, 139);
});

test('a typed-in table without a seating is numbered as the game numbered its players at that hand', () => {
  let tied = 0, renumbered = 0;
  decisions(({ engine, index, position }) => {
    const scores = position.players.map(player => player.score);
    if (new Set(scores).size === 4 || position.kyoku === 1) return;
    // The deal moves on one seat a hand, so the hand number gives the seating.
    const { seating, ...typed } = position;
    assert.deepEqual(seating, [0, 1, 2, 3].map(seat => (seat + position.kyoku - 1) % 4));
    const want = engine.review_observation_mortal(index);
    assert.ok(same(planesOf(typed), want));
    tied++;
    // Numbered from the hand's own dealer instead, players on the same points
    // rank by the wrong numbers, and only the four rank planes can change.
    const differing = planesDiffering(want, planesOf({ ...typed, seating: [0, 1, 2, 3] }));
    assert.ok(differing.every(plane => plane >= 15 && plane < 19), String(differing));
    renumbered += Number(differing.length > 0);
  });
  assert.ok(tied > 20, `only ${tied} decisions with tied points after the first hand`);
  assert.ok(renumbered > 0, 'no tie was broken differently');
});

test('a written position is refused where it names a seating no game has or a quad\'s turn its seat never took', () => {
  let refused = false;
  decisions(({ position }) => {
    if (refused || !position.players.some(player => player.melds.some(meld => Number.isInteger(meld.order)))) return;
    assert.throws(() => planesOf({ ...position, seating: [0, 1, 1, 3] }), /seating/);
    const owner = position.players.findIndex(player => player.melds.some(meld => Number.isInteger(meld.order)));
    const other = position.players.find((player, seat) => seat !== owner && player.discards.length);
    const moved = structuredClone(position);
    moved.players[owner].melds.find(meld => Number.isInteger(meld.order)).order = other.discards[0].order;
    assert.throws(() => planesOf(moved), /kan records its turn/);
    refused = true;
  });
  assert.ok(refused);
});
