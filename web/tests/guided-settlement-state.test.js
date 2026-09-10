import test from 'node:test';
import assert from 'node:assert/strict';
import { emptyGuided, guidedEvent, editGuided, undoGuided, parseGuided, GUIDED_FORMAT } from '../src/lib/guided-game.js';
import { parseTiles } from '../src/lib/physical-position.js';

// These tests exercise the ledger boundary, not scoring. Actual Rust/WASM
// scoring fixtures live in guided-settlement.test.js.
function beforeRon() {
  const g = emptyGuided(), p = g.state.position;
  p.seat = 3; p.turn = 0; p.phase = 'call'; p.wall = 50; p.first_turns = false;
  p.indicators = ['7z']; p.pending = '2z';
  p.players[3].hand = parseTiles('123m456p789s1112z');
  p.players[0].discards = [{ tile: '2z', order: 0, riichi: false, claimed: false, drawn: false }];
  g.state.stage = 'decision'; g.state.opening = [30000, 30000, 30000, 30000];
  return g;
}
const ron = g => guidedEvent(g, { type: 'choice', choice: { kind: 'ron' }, choices: [{ kind: 'ron' }] });
function result() {
  return { kind: 'ron', before: [30000,30000,30000,30000], after: [28700,30000,30000,31300],
    deltas: [-1300,0,0,1300], sticks_before: 0, sticks_after: 0, repeat: false, next_counters: 0,
    refunded_riichi: null, tenpai: [], winners: [{ seat: 3, winning_tile: '2z', hand: parseTiles('123m456p789s11122z'),
      han: 1, fu: 40, yaku: [{ name: 'Round Wind Triplet', han: 1, yakuman: false }],
      dora: 0, ura_dora: 0, indicators: ['7z'], ura_indicators: [], limit: null, hand_payment: 1300,
      fu_detail: [['Base', 20], ['Concealed ron', 10], ['Set of 1z', 8], ['Wait', 2]] }] };
}
const settle = (g, score = result) => guidedEvent(g, { type: 'settle', input: { winners: [3] } }, undefined, score);

test('ron retains the actionable scoring position and waits for settlement', () => {
  const g = beforeRon(), ended = ron(g);
  assert.equal(ended.state.stage, 'over');
  assert.equal(ended.state.position.phase, 'over');
  assert.equal(ended.state.ending.position.phase, 'call');
  assert.equal(ended.state.ending.position.pending, '2z');
  assert.deepEqual(ended.state.position.players.map(p => p.score), [30000,30000,30000,30000]);
  assert.deepEqual(undoGuided(ended), g);
  assert.throws(() => guidedEvent(ended, { type: 'next-hand', repeat: false }), /settlement first/);
});

test('settlement applies the engine ledger once and saves the whole-hand changes', () => {
  const ended = ron(beforeRon()), paid = settle(ended);
  assert.deepEqual(paid.state.position.players.map(p => p.score), result().after);
  assert.deepEqual(paid.state.settlement.hand_deltas, result().deltas);
  assert.equal(paid.state.position.riichi_sticks, 0);
  assert.throws(() => settle(paid), /already been settled/);
  assert.deepEqual(undoGuided(paid), ended);
  assert.deepEqual(undoGuided(settle(undoGuided(paid))), ended);
});

test('a failed or missing scorer changes neither the ledger nor history', () => {
  const ended = ron(beforeRon()), copy = structuredClone(ended);
  assert.throws(() => settle(ended, () => { throw new Error('No yaku'); }), /No yaku/);
  assert.throws(() => guidedEvent(ended, { type: 'settle' }), /not ready/);
  assert.deepEqual(ended, copy);
});

test('unbalanced, stale, or malformed results fail closed', () => {
  const ended = ron(beforeRon());
  for (const change of [r => { r.after[0]++; }, r => { r.deltas[3]++; },
    r => { r.before[0]++; }, r => { r.sticks_after = 1; }, r => { r.next_counters = 1; },
    r => { r.winners = []; }, r => { r.winners[0].yaku = []; }]) {
    const r = result(); change(r);
    assert.throws(() => settle(ended, () => r), /invalid/);
  }
  assert.deepEqual(ended.state.position.players.map(p => p.score), result().before);
});

test('next hand rotates settled balances and cannot override dealer continuation', () => {
  const paid = settle(ron(beforeRon()));
  assert.throws(() => guidedEvent(paid, { type: 'next-hand', repeat: true }), /continuation/);
  const next = guidedEvent(paid, { type: 'next-hand', repeat: false });
  assert.deepEqual(next.state.position.players.map(p => p.score), [30000,30000,31300,28700]);
  assert.equal(next.state.position.seat, 2);
  assert.equal(next.state.position.riichi_sticks, 0);
  assert.equal(next.state.ending, undefined);
  assert.deepEqual(undoGuided(next), paid);
});

test('settled results survive reload without paying again, including undo snapshots', () => {
  const paid = settle(ron(beforeRon()));
  const restored = parseGuided(GUIDED_FORMAT.encode(paid), result);
  assert.deepEqual(restored, paid);
  assert.throws(() => settle(restored), /already been settled/);
  const next = guidedEvent(paid, { type: 'next-hand', repeat: false });
  assert.deepEqual(parseGuided(GUIDED_FORMAT.encode(next), result), next);
});

test('altered saved yaku and balanced but incorrect payments are rejected on engine verification', () => {
  for (const mutate of [r => { r.winners[0].yaku[0].name = 'Invented'; }, r => {
    r.after[0] -= 100; r.after[3] += 100; r.deltas[0] -= 100; r.deltas[3] += 100;
  }]) {
    const paid = settle(ron(beforeRon()));
    mutate(paid.state.settlement);
    paid.state.settlement.hand_deltas = paid.state.settlement.deltas;
    paid.state.position.players.forEach((p, i) => { p.score = paid.state.settlement.after[i]; });
    assert.equal(parseGuided(GUIDED_FORMAT.encode(paid), result), null);
  }
});

test('scored and pending tables cannot be manually edited; adviser selection remains available', () => {
  for (const g of [ron(beforeRon()), settle(ron(beforeRon()))]) {
    assert.throws(() => editGuided(g, s => { s.position.players[0].score = 1; }), /Undo/);
    assert.throws(() => editGuided(g, s => { s.ending.position.counters++; }), /Undo/);
    assert.equal(editGuided(g, s => { s.agent = 'beginner'; }).state.agent, 'beginner');
  }
});

test('automatic exhaustion retains the final discard before clearing the display position', () => {
  let g = beforeRon(); g.state.position.wall = 0; g.state.stage = 'responses';
  g = guidedEvent(g, { type: 'continue' });
  assert.equal(g.state.ending.kind, 'draw');
  assert.equal(g.state.ending.position.pending, '2z');
  assert.equal(g.state.ending.position.turn, 0);
  assert.equal(g.state.ending.position.wall, 0);
  assert.equal(g.state.stage, 'over');
  assert.throws(() => guidedEvent(g, { type: 'next-hand', repeat: false }), /settlement first/);
});

test('legacy manually settled saves stay manual and are never paid again', () => {
  const g = beforeRon(); g.state.stage = 'over'; g.state.position.phase = 'over'; g.state.result = 'Old hand';
  g.state.position.players[3].score = 31300;
  const read = parseGuided(GUIDED_FORMAT.encode(g), () => { throw new Error('Must not rescore a legacy hand'); });
  assert.deepEqual(read, g);
  assert.equal(guidedEvent(read, { type: 'next-hand', repeat: false }).state.position.players[2].score, 31300);
});

test('manual adjudication remains explicitly separate from automatic scoring', () => {
  let g = guidedEvent(beforeRon(), { type: 'finish', kind: 'manual', result: 'Other hand end' });
  g = editGuided(g, s => { s.position.players[0].score = 25000; });
  assert.throws(() => settle(g), /manual table settlement/);
  assert.equal(guidedEvent(g, { type: 'next-hand', repeat: true }).state.position.players[0].score, 25000);
});

test('invalid winner timing and premature exhaustive draws are refused', () => {
  const g = beforeRon();
  assert.throws(() => guidedEvent(g, { type: 'finish', kind: 'ron', result: 'Ron', winner: 0 }), /another player/);
  assert.throws(() => guidedEvent(g, { type: 'finish', kind: 'tsumo', result: 'Tsumo', winner: 3 }), /actual draw/);
  assert.throws(() => guidedEvent(g, { type: 'finish', kind: 'draw', result: 'Exhaustive draw' }), /final discard/);
});
