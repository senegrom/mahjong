import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { settle_physical } from '../src/wasm/riichi.js';
import { emptyPosition, parseTiles } from '../src/lib/physical-position.js';
import { emptyGuided, guidedEvent, parseGuided, GUIDED_FORMAT, undoGuided } from '../src/lib/guided-game.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const thrown = (tile, order = 0, extra = {}) => ({ tile, order, drawn: false, riichi: false, claimed: false, ...extra });
function ron() {
  const p = emptyPosition();
  Object.assign(p, { seat: 3, turn: 0, phase: 'call', wall: 50, first_turns: false, indicators: ['7z'], pending: '2z' });
  p.players[3].hand = parseTiles('123m456p789s1112z');
  p.players[0].discards = [thrown('2z')];
  return { kind: 'ron', winners: [3], position: p, nextSeat: 0, needsDraw: false };
}
const score = (ending, input = {}) => settle_physical(ending, { winners: ending.winners, ...input });
const names = r => r.winners[0].yaku.map(y => y.name);
const hands = (seat, tiles) => Array.from({ length: 4 }, (_, i) => i === seat ? parseTiles(tiles) : []);
function tsumo(seat = 3) {
  const e = ron(), p = e.position;
  p.players[seat].hand = parseTiles('123m456p789s11122z');
  if (seat !== 3) p.players[3].hand = [];
  Object.assign(p, { seat, turn: seat, phase: 'act', drawn: '2z', pending: null });
  e.kind = 'tsumo'; e.winners = [seat]; e.nextSeat = seat;
  return e;
}
function riichi(e, seat = 3) {
  const p = e.position;
  p.players[seat].riichi = 'riichi'; p.players[seat].score -= 1000; p.riichi_sticks++;
  p.players[seat].discards = [thrown('9p', 0, { riichi: true })];
  p.players[0].discards[0].order = 1;
  return e;
}

test('real scorer: 1 han 40 fu North ron pays 1300 and records the yaku', () => {
  const r = score(ron());
  assert.deepEqual(r.deltas, [-1300,0,0,1300]);
  assert.deepEqual([r.winners[0].han, r.winners[0].fu], [1,40]);
  assert.deepEqual(names(r), ['Round Wind Triplet']);
  assert.equal(r.repeat, false);
});

test('real scorer: honba and carried riichi pot are paid, and the pot clears', () => {
  const e = ron(); e.position.counters = 2; e.position.riichi_sticks = 3;
  const r = score(e);
  assert.deepEqual(r.deltas, [-1900,0,0,4900]);
  assert.equal(r.sticks_after, 0);
  assert.equal(r.next_counters, 0);
});

test('real scorer: nondealer and dealer tsumo use the right individual payments', () => {
  assert.deepEqual(score(tsumo()).deltas, [-1300,-700,-700,2700]);
  const e = tsumo(0); e.position.round = 1;
  const r = score(e);
  assert.deepEqual(r.deltas, [3900,-1300,-1300,-1300]);
  assert.equal(r.repeat, true); assert.equal(r.next_counters, 1);
});

test('real scorer: seven pairs stays 25 fu and pinfu tsumo stays 20 fu', () => {
  const e = ron(), p = e.position;
  p.players[3].hand = parseTiles('1122m3344p5566s1z'); p.pending = '1z'; p.players[0].discards[0].tile = '1z';
  let r = score(e);
  assert.deepEqual([r.winners[0].han, r.winners[0].fu, r.deltas[3]], [2,25,1600]);
  const t = tsumo();
  // 123m 456m 234p 567s + pair 22p, with 5s completing a two-sided wait.
  t.position.players[3].hand = parseTiles('123456m23422p567s'); t.position.drawn = '5s';
  r = score(t);
  assert.deepEqual([r.winners[0].han, r.winners[0].fu], [2,20]);
  assert.deepEqual(r.deltas, [-700,-400,-400,1500]);
});

test('real scorer: a ron winning tile counts toward dora exactly once', () => {
  const e = ron(); e.position.indicators = ['1z'];
  const r = score(e);
  assert.equal(r.winners[0].dora, 2);
  assert.equal(r.winners[0].han, 3);
  assert.deepEqual(r.deltas, [-5200,0,0,5200]);
});

test('real scorer: riichi requires actual ura, including zero-bonus indicators', () => {
  const e = riichi(ron());
  assert.throws(() => score(e), /ura/);
  const zero = score(e, { ura: ['6z'] });
  assert.equal(zero.winners[0].ura_dora, 0);
  assert.deepEqual(zero.winners[0].ura_indicators, ['6z']);
  assert.equal(zero.after[3], 32600);
  const bonus = score(e, { ura: ['9m'] });
  assert.equal(bonus.winners[0].ura_dora, 1);
  assert.equal(bonus.winners[0].han, 3);
  assert.equal(bonus.after[3], 35200);
  assert.throws(() => score(ron(), { ura: ['6z'] }), /only revealed/);
});

test('real scorer: riichi declaration-discard ron refunds the rejected bet', () => {
  const e = ron(), p = e.position;
  p.players[0].riichi = 'riichi'; p.players[0].score = 29000;
  p.players[0].discards[0].riichi = true; p.riichi_sticks = 1;
  const r = score(e);
  assert.deepEqual(r.after, [28700,30000,30000,31300]);
  assert.deepEqual(r.deltas, [-300,0,0,1300]);
  assert.equal(r.refunded_riichi, 0); assert.equal(r.sticks_after, 0);
});

test('real scorer: multi-ron orders winners and returns each winner’s own riichi bet', () => {
  const e = riichi(ron()), p = e.position;
  p.indicators = ['6z']; p.players[3].discards[0].order = 0;
  p.players[1].riichi = 'riichi'; p.players[1].score = 29000;
  p.players[1].discards = [thrown('8p', 1, { riichi: true })];
  p.players[0].discards[0].order = 2; p.riichi_sticks = 4;
  const r = score(e, { winners: [3,1], hands: hands(1, '234m567p234s5552z'), ura: ['8m'], confirmed_no_furiten: true });
  assert.deepEqual(r.winners.map(w => w.seat), [1,3]);
  assert.deepEqual(r.after, [24200,35200,30000,32600]);
  assert.equal(r.sticks_after, 0);
  assert.throws(() => score(e, { winners: [3,3], ura: ['8m'] }), /once/);
});

test('real scorer: opponents reveal their hand; unseen tiles are never invented', () => {
  const e = ron(); e.winners = [2];
  e.position.players[3].hand = parseTiles('456m123p456s2346z');
  assert.throws(() => score(e), /Reveal/);
  assert.throws(() => score(e, { hands: hands(2, '123m456p789s1112z') }), /furiten/);
  const r = score(e, { hands: hands(2, '123m456p789s1112z'), confirmed_no_furiten: true });
  assert.deepEqual(r.deltas, [-1300,0,1300,0]);
  assert.equal(e.position.players[2].hand.length, 0, 'scoring is read-only');
});

test('real scorer: an opponent’s last live tsumo is counted once and gets Haitei', () => {
  const e = ron(), p = e.position; e.kind = 'tsumo'; e.winners = [1]; e.nextSeat = 1; e.needsDraw = true;
  Object.assign(p, { phase: 'draw', turn: 1, pending: null, wall: 1 });
  p.players[3].hand = parseTiles('456m123p456s2346z');
  const r = score(e, { hands: hands(1, '123m456p789s1112z'), winning_tile: '2z' });
  assert.ok(names(r).includes('Under the Sea'));
  assert.deepEqual(r.deltas, [-2600,5200,-1300,-1300]);
  assert.equal(p.wall, 1, 'the input wall is not mutated');
});

test('real scorer: no-yaku, furiten, and impossible fifth copies are refused', () => {
  const e = ron(), p = e.position;
  p.players[3].hand = parseTiles('123m456p789s45s11z'); p.pending = '6s';
  p.players[0].discards[0].tile = '6s'; p.indicators = ['5s'];
  assert.throws(() => score(e), /legal ron/);
  const f = ron(); f.position.players[3].discards = [thrown('2z', 0)]; f.position.players[0].discards[0].order = 1;
  assert.throws(() => score(f), /furiten/);
  const t = tsumo(); t.position.players[3].furiten = true;
  assert.ok(score(t).deltas[3] > 0, 'furiten does not bar tsumo');
  const bad = riichi(ron()); bad.position.indicators = ['1z'];
  assert.throws(() => score(bad, { ura: ['1z'] }), /four copies/);
});

test('real scorer: first-turn blessings are retained and yakuman ignores dora', () => {
  const e = ron(); e.position.first_turns = true;
  const r = score(e);
  assert.ok(names(r).includes('Blessing of Man')); assert.equal(r.deltas[3], 8000);
  const t = tsumo(0); t.position.first_turns = true; t.position.players[0].discards = [];
  const heaven = score(t);
  assert.ok(names(heaven).includes('Blessing of Heaven'));
  assert.equal(heaven.winners[0].limit, 'yakuman'); assert.equal(heaven.winners[0].dora, 0);
  assert.deepEqual(heaven.deltas, [48000,-16000,-16000,-16000]);
});

test('real scorer: robbing an added kan keeps ippatsu and does not reveal an extra indicator', () => {
  const e = ron(), p = e.position;
  p.players[3].hand = parseTiles('123m456p789s11z34m');
  p.players[3].riichi = 'riichi'; p.players[3].ippatsu = true; p.players[3].score = 29000;
  p.players[3].discards = [thrown('2z', 1, { riichi: true })]; p.riichi_sticks = 1;
  p.players[0].discards = []; p.players[0].melds = [{ kind: 'extended-kan', tile: '5m', from: 1 }];
  p.players[1].discards = [thrown('5m', 0, { claimed: true })];
  p.pending = '5m'; p.pending_kind = 'extended-kan'; p.after_quad = true; p.indicators = ['6z'];
  const r = score(e, { ura: ['6z'] });
  assert.ok(names(r).includes('Robbing a Quad')); assert.ok(names(r).includes('Ippatsu'));
  assert.equal(r.winners[0].han, 3); assert.equal(r.winners[0].indicators.length, 1);
});

test('real scorer: replacement draw gets Rinshan, not Haitei, and all kan indicators count', () => {
  const e = tsumo(), p = e.position;
  p.players[3].melds = [{ kind: 'concealed-kan', tile: '1m', from: 0 }];
  p.players[3].hand = parseTiles('456p789s55522z');
  p.after_quad = true; p.wall = 0; p.indicators = ['6z', '8m'];
  const r = score(e);
  assert.ok(names(r).includes('After a Quad'));
  assert.ok(!names(r).includes('Under the Sea'));
  assert.equal(r.winners[0].indicators.length, 2);
});

for (const kind of ['tsumo', 'ron']) test(`real scorer: responsibility payments on ${kind} use the actual feeder`, () => {
  const e = kind === 'tsumo' ? tsumo() : ron(), p = e.position;
  p.counters = 2; p.indicators = ['8m'];
  p.players[3].melds = ['5z','6z','7z'].map((tile, i) => ({ kind: 'pon', tile, from: i + 1 }));
  p.players[3].hand = parseTiles(kind === 'tsumo' ? '123m22p' : '123m2p');
  p.players[0].discards = [thrown('5z', 0, { claimed: true })];
  p.players[1].discards = [thrown('6z', 1, { claimed: true })];
  p.players[2].discards = [thrown('7z', 2, { claimed: true })];
  if (kind === 'tsumo') p.drawn = '2p';
  else { p.pending = '2p'; p.players[0].discards.push(thrown('2p', 3)); }
  const r = score(e);
  assert.ok(names(r).includes('Big Three Dragons'));
  assert.deepEqual(r.deltas, kind === 'tsumo' ? [0,0,-32600,32600] : [-16600,0,-16000,32600]);
});

for (let n = 0; n <= 4; n++) test(`real scorer: exhaustive draw with ${n} tenpai declarations`, () => {
  const e = ron(), p = e.position; e.kind = 'draw'; e.winners = [];
  p.wall = 0; p.pending = '6z'; p.players[0].discards[0].tile = '6z'; p.riichi_sticks = 2; p.counters = 1;
  const tiles = ['123456789m123p1z', '123456789p123s2z', '123456789s456m3z', '789m789p789s234m4z'];
  p.players[3].hand = parseTiles(tiles[3]);
  const r = score(e, { tenpai: [0,1,2,3].map(i => i < n), hands: tiles.map((t, i) => i < n && i !== 3 ? parseTiles(t) : []) });
  const expected = n === 0 || n === 4 ? [0,0,0,0] : [0,1,2,3].map(i => i < n ? 3000 / n : -3000 / (4 - n));
  assert.deepEqual(r.deltas, expected); assert.equal(r.sticks_after, 2);
  assert.equal(r.repeat, n > 0); assert.equal(r.next_counters, 2);
});

test('real scorer: guided application, save verification, undo and next hand reconcile', () => {
  const e = riichi(ron());
  let g = emptyGuided(); g.state.position = e.position; g.state.stage = 'decision'; g.state.opening = [30000,30000,30000,30000];
  g = guidedEvent(g, { type: 'choice', choice: { kind: 'ron' }, choices: [{ kind: 'ron' }] });
  const before = structuredClone(g);
  g = guidedEvent(g, { type: 'settle', input: { winners: [3], ura: ['6z'] } }, undefined, settle_physical);
  assert.deepEqual(g.state.settlement.hand_deltas, [-2600,0,0,2600]);
  assert.deepEqual(parseGuided(GUIDED_FORMAT.encode(g), settle_physical), g);
  assert.deepEqual(undoGuided(g), before);
  const next = guidedEvent(g, { type: 'next-hand', repeat: false });
  assert.equal(next.state.position.riichi_sticks, 0); assert.equal(next.state.position.players[2].score, 32600);
});
