import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis } from '../src/wasm/riichi.js';
import { clubAdvisedGame, eachDecision, reviewCheckGame } from './fixtures/review-check-game.js';

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

test('a written position holds what the deciding player saw, and nothing hidden', () => {
  let count = 0;
  decisions(({ view, engine, index, position }) => {
    // Who sits where is not written: the hand number names each seat's player
    // as the view does, the deal having moved on one seat a hand.
    assert.equal('seating' in position, false);
    assert.deepEqual([0, 1, 2, 3].map(seat => (seat + position.kyoku - 1) % 4),
      WINDS.map(wind => view.seats.find(seat => seat.seat === wind).player));
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

test('players on the same points after the first hand rank as the game ranked them', () => {
  let tied = 0;
  decisions(({ engine, index, position }) => {
    const scores = position.players.map(player => player.score);
    if (new Set(scores).size === 4 || position.kyoku === 1) return;
    // Mortal breaks a tie by player number from the game's first dealer, which
    // the hand number gives, so the rank planes are the review's own too.
    assert.ok(same(planesOf(position), engine.review_observation_mortal(index)));
    tied++;
  });
  assert.ok(tied > 20, `only ${tied} decisions with tied points after the first hand`);
});

test('the riichi\'s second question of a written position is the review\'s own, in the first hand and after it', () => {
  const wrong = [];
  let asked = 0, later = 0;
  eachDecision(Game, clubAdvisedGame(Game, 5), ({ hand, index, engine, position }) => {
    if (!engine.review_mask_mortal(index)[37]) return;
    const analysis = new PhysicalAnalysis(position);
    try {
      if (!same(engine.review_observation_after_reach(index), analysis.agent_observation_after_reach())
        || !same(engine.review_mask_after_reach(index), analysis.agent_mask_after_reach())) wrong.push(`${hand.name}, decision ${index + 1}`);
    } finally { analysis.free(); }
    asked++;
    // From a round's second hand on, a seat's player number is not its own.
    later += Number(position.kyoku > 1);
  });
  assert.deepEqual(wrong, []);
  assert.ok(later > 0, `${asked} riichi questions, none after a round's first hand`);
});

test('a table typed in at any hand number asks the riichi\'s second question of its own declaration', () => {
  const seat = hand => ({ hand, melds: [], discards: [], score: 25000, riichi: 'none', ippatsu: false, furiten: false });
  // East on the first turn, as Guided play and Physical play enter a table,
  // ready to declare with fourteen tiles of fourteen kinds: the 5 bamboo just
  // drawn makes 3-4-5 bamboo, and the 3 or the 4 circles goes.
  const table = kyoku => ({ seat: 0, turn: 0, phase: 'act', round: 0, kyoku, counters: 0, riichi_sticks: 0, wall: 69,
    indicators: ['7z'], players: [seat(['1m', '2m', '3m', '6m', '7m', '8m', '3p', '4p', '6p', '7p', '8p', '3s', '4s', '5s']),
      seat([]), seat([]), seat([])],
    drawn: '5s', pending: null, pending_kind: 'discard', just_claimed: null, after_quad: false, first_turns: true });
  const declared = kyoku => {
    const analysis = new PhysicalAnalysis(table(kyoku));
    try {
      assert.equal(analysis.agent_mask_mortal()[37], 1, `riichi is offered in East ${kyoku}`);
      return planesDiffering(analysis.agent_observation_mortal(), analysis.agent_observation_after_reach());
    } finally { analysis.free(); }
  };
  // The declaration changes what East reads in the same planes at every hand
  // number, as East's own, whichever player number East has.
  const first = declared(1);
  assert.ok(first.length > 0);
  for (const kyoku of [2, 3, 4]) assert.deepEqual(declared(kyoku), first, `East ${kyoku}`);
});

test('a written position is refused where it names a quad\'s turn its seat never took', () => {
  let refused = false;
  decisions(({ position }) => {
    if (refused || !position.players.some(player => player.melds.some(meld => Number.isInteger(meld.order)))) return;
    const owner = position.players.findIndex(player => player.melds.some(meld => Number.isInteger(meld.order)));
    const other = position.players.find((player, seat) => seat !== owner && player.discards.length);
    const moved = structuredClone(position);
    moved.players[owner].melds.find(meld => Number.isInteger(meld.order)).order = other.discards[0].order;
    assert.throws(() => planesOf(moved), /kan records its turn/);
    refused = true;
  });
  assert.ok(refused);
});
