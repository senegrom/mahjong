import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { scanMatch } from '../src/lib/game-review.js';
import { handTables } from '../src/lib/review-insight-policy.js';
import { analysisInput, emptyPosition, parsePhysical, parseTiles, recordChoice, recordDiscard, recordDraw } from '../src/lib/physical-position.js';
import { emptyGuided, guidedEvent } from '../src/lib/guided-game.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

// Two facts of Mortal's replay that a typed-in table knows as it records the
// move, and that a game played here carries: the tile a seat drew on the
// turn it declared riichi (`riichi_draw`), and the turn of a quad a seat
// declared itself (a set's `order`, that of the discard the turn ends with).

/** East to act with 1m drawn, ready on 1m-4m once 9p goes. */
function readyAfterDraw() {
  const position = emptyPosition();
  position.players[0].hand = parseTiles('234567m11p456s99p');
  position.players[0].hand.splice(position.players[0].hand.indexOf('9p'), 1);
  position.players[0].hand.push('1m');
  Object.assign(position, { drawn: '1m', first_turns: false, wall: 50 });
  return position;
}

test('a riichi recorded from a known hand keeps the tile drawn that turn, and one from an unknown hand keeps none', () => {
  const p = readyAfterDraw();
  p.players[0].hand = parseTiles('2345678m11p456s9p1m');
  const declared = recordDiscard(p, 0, '9p', true);
  assert.equal(declared.players[0].riichi_draw, '1m');
  assert.equal(recordDiscard(p, 0, '9p').players[0].riichi_draw, undefined, 'a plain discard records none');
  // An opponent whose hand nobody entered: its draw is not known.
  const unknown = emptyPosition();
  Object.assign(unknown, { turn: 3, phase: 'call', pending: '5z', first_turns: false, wall: 50 });
  assert.equal(recordDiscard(unknown, 0, '9p', true).players[0].riichi_draw, undefined);
});

test('a quad a seat declares records the turn its discard will end, at the table and in a guided game', () => {
  const p = emptyPosition();
  p.players[1].discards.push({ tile: '9m', order: 0, drawn: true, riichi: false, claimed: false });
  p.players[0].hand = parseTiles('1111m234p567s555z');
  p.players[0].hand.push('2z');
  Object.assign(p, { drawn: '2z', first_turns: false, wall: 50 });
  p.players[2].discards.push({ tile: '1z', order: 4, drawn: true, riichi: false, claimed: false });
  const concealed = recordChoice(p, { kind: 'concealed-kan', tile: '1m' }, [{ kind: 'concealed-kan', tile: '1m' }]);
  assert.equal(concealed.players[0].melds[0].order, 5);
  // The replacement, then the discard that ends the turn carries that number.
  const ended = recordDiscard(recordDraw(concealed, 0, '7z'), 0, '7z');
  assert.equal(ended.players[0].discards.at(-1).order, 5);
  const added = structuredClone(p);
  added.players[0].hand = parseTiles('234p567s555z2z1m');
  added.players[0].melds.push({ kind: 'pon', tile: '1m', from: 1, claimed_tile: '1m' });
  added.players[1].discards[0] = { tile: '1m', order: 0, drawn: true, riichi: false, claimed: true };
  const extended = recordChoice(added, { kind: 'extended-kan', tile: '1m' }, [{ kind: 'extended-kan', tile: '1m' }]);
  assert.equal(extended.players[0].melds[0].order, 5);
  // An opponent's quad, told at a guided table.
  let game = emptyGuided('club');
  game.state.position.seat = 3;
  game.state.position.players[3].hand = parseTiles('123m456p789s1123z');
  for (const event of [{ type: 'setup' }, { type: 'hand' }, { type: 'indicator', tile: '7z' }]) game = guidedEvent(game, event);
  game = guidedEvent(game, { type: 'kan', kind: 'concealed-kan', tile: '2s' });
  assert.equal(game.state.position.players[0].melds[0].order, 0, 'the first discard of the hand is still to come');
});

test('the engine is told a riichi draw for the seat analysed alone, and a quad turn only while it still holds', () => {
  const p = readyAfterDraw();
  p.players[0].hand = parseTiles('2345678m11p456s9p1m');
  const declared = recordDiscard(p, 0, '9p', true);
  assert.equal(analysisInput(declared).players[0].riichi_draw, '1m', 'East is analysed');
  // South answers East's discard: East's draw is East's to know.
  const south = { ...structuredClone(declared), seat: 1 };
  assert.equal(analysisInput(south).players[0].riichi_draw, undefined);
  assert.equal(south.players[0].riichi_draw, '1m', 'kept in the table for East\'s own decisions');
  // A riichi taken back by an edit takes its draw with it.
  const taken = structuredClone(declared);
  taken.players[0].riichi = 'none';
  assert.equal(analysisInput(taken).players[0].riichi_draw, undefined);
  // A quad's turn goes with the discard it names, or with the set's kind.
  const quad = emptyPosition();
  quad.players[0].melds.push({ kind: 'concealed-kan', tile: '1m', from: 0, order: 3 });
  quad.players[0].discards.push({ tile: '9p', order: 3, drawn: true, riichi: false, claimed: false });
  assert.equal(analysisInput(quad).players[0].melds[0].order, 3);
  const removed = structuredClone(quad);
  removed.players[0].discards = [];
  assert.equal(analysisInput(removed).players[0].melds[0].order, undefined, 'its discard removed in the editor');
  const renamed = structuredClone(quad);
  renamed.players[0].melds[0].kind = 'pon';
  assert.equal(analysisInput(renamed).players[0].melds[0].order, undefined, 'no longer a quad of its own');
  // The turn still being played: the replacement held, or the quad open to robbery.
  const playing = structuredClone(removed);
  Object.assign(playing, { after_quad: true, phase: 'act', turn: 0 });
  playing.players[1].discards.push({ tile: '1z', order: 2, drawn: true, riichi: false, claimed: false });
  assert.equal(analysisInput(playing).players[0].melds[0].order, 3);
  assert.deepEqual(quad, analysisInput(structuredClone(quad)), 'nothing else changes');
});

test('drafts saved before these facts read as before, and a draft that holds them is checked where it is read', () => {
  const draft = position => JSON.stringify({ version: 1, position });
  const old = emptyPosition();
  old.players[0].melds.push({ kind: 'concealed-kan', tile: '1m', from: 0 });
  assert.ok(parsePhysical(draft(old)));
  const recorded = structuredClone(old);
  recorded.players[0].melds[0].order = 7;
  recorded.players[0].riichi_draw = '5p';
  assert.deepEqual(parsePhysical(draft(recorded)), recorded);
  for (const spoil of [p => { p.players[0].melds[0].order = -1; }, p => { p.players[0].melds[0].order = 1.5; }, p => { p.players[0].melds[0].order = '7'; },
    p => { p.players[0].riichi_draw = '0m'; }, p => { p.players[0].riichi_draw = 5; }]) {
    const bad = structuredClone(recorded);
    spoil(bad);
    assert.equal(parsePhysical(draft(bad)), null, JSON.stringify(bad.players[0]));
  }
});

/** A game the player plays with Club's moves, declaring every riichi it can. */
function clubGame(seed) {
  const match = new MatchSession(Game, seed, 'club');
  try {
    match.advance(false);
    for (let step = 0; step < 5000 && !match.over; step++) {
      if (match.engine.hand_is_over()) { match.apply({ type: 'next' }); match.advance(false); continue; }
      const choices = match.choices, pick = match.engine.agent_pick('club');
      const choice = choices.find(c => c.kind === 'riichi') ?? choices.find(c => c.kind === pick.kind && (c.tile ?? null) === (pick.tile ?? null)) ?? choices[0];
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
    }
    return match.snapshot();
  } finally { match.dispose(); }
}

/** A game the player plays throwing the tile drawn, calling every set and quad offered. */
function callingGame(seed) {
  const match = new MatchSession(Game, seed, 'beginner');
  try {
    match.advance(false);
    for (let step = 0; step < 5000 && !match.over; step++) {
      if (match.engine.hand_is_over()) { match.apply({ type: 'next' }); match.advance(false); continue; }
      const choices = match.choices, drawn = match.view.seats[0].drawn;
      const choice = choices.find(c => ['ron', 'tsumo', 'riichi'].includes(c.kind))
        ?? choices.find(c => ['concealed-kan', 'extended-kan', 'kan', 'pon', 'chii'].includes(c.kind))
        ?? choices.find(c => c.kind === 'discard' && c.tile === drawn) ?? choices.find(c => c.kind === 'discard') ?? choices[0];
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
    }
    return match.snapshot();
  } finally { match.dispose(); }
}

const planes = position => {
  const analysis = new PhysicalAnalysis(position);
  try { return Array.from(analysis.agent_observation_mortal()); } finally { analysis.free(); }
};
const same = (a, b) => a.length === b.length && a.every((value, at) => value === b[at]);

/** Each hand of `game` in which the player made a move of `kind`: the hand's
 * tables, as the engine writes them, and where the move was made. */
function movesOf(game, kinds) {
  return scanMatch(Game, game).flatMap(hand => {
    const at = hand.frames.findIndex(frame => kinds.includes(frame.played.kind));
    return at < 0 ? [] : [{ tables: handTables(Game, game, hand), at, played: hand.frames[at].played }];
  });
}

test("a riichi recorded at the table gives Mortal the played game's planes at every later decision, as it did not before", () => {
  let later = 0, differed = 0;
  for (const { tables, at, played } of movesOf(clubGame(1), ['riichi'])) {
    const recorded = recordChoice(tables[at], played, [played]);
    for (const position of tables.slice(at + 1)) {
      // The later table as the page holds it: the played game's, with what
      // the page itself recorded at the riichi in place of the game's own.
      const page = structuredClone(position), seat = page.players[page.seat];
      assert.equal(seat.riichi_draw, recorded.players[recorded.seat].riichi_draw, 'the tile the played game recorded');
      delete seat.riichi_draw;
      const unrecorded = planes(page);
      seat.riichi_draw = recorded.players[recorded.seat].riichi_draw;
      assert.ok(same(planes(analysisInput(page)), planes(position)));
      later++;
      differed += same(unrecorded, planes(position)) ? 0 : 1;
    }
  }
  assert.ok(later > 3 && differed > 0, `${differed} of ${later} later decisions read otherwise without it`);
});

test("a quad recorded at the table gives Mortal the played game's planes, at its replacement and at every later decision", () => {
  let later = 0, differed = 0;
  for (const { tables, at, played } of movesOf(callingGame(21), ['concealed-kan', 'extended-kan'])) {
    const recorded = recordChoice(tables[at], played, [played]), seat = recorded.seat;
    const turns = recorded.players[seat].melds.map(meld => meld.order);
    // The replacement drawn: the turn is still being played, and the engine takes it.
    const next = tables[at + 1];
    assert.equal(next.phase, 'act');
    const replaced = analysisInput(recordDraw({ ...recorded, indicators: next.indicators }, seat, next.drawn));
    assert.ok(same(planes(replaced), planes(next)));
    for (const position of tables.slice(at + 1)) {
      assert.deepEqual(position.players[seat].melds.slice(0, turns.length).map(meld => meld.order ?? null), turns.map(turn => turn ?? null));
      const page = structuredClone(position);
      page.players[seat].melds.forEach(meld => { delete meld.order; });
      const unrecorded = planes(page);
      page.players[seat].melds.forEach((meld, index) => { if (turns[index] != null) meld.order = turns[index]; else delete meld.order; });
      assert.ok(same(planes(analysisInput(page)), planes(position)));
      later++;
      differed += same(unrecorded, planes(position)) ? 0 : 1;
    }
  }
  assert.ok(later > 3 && differed > 0, `${differed} of ${later} later decisions read otherwise without it`);
});
