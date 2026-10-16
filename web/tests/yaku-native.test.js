/** The real WASM settlement result is what both result screens explain. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, settle_physical } from '../src/wasm/riichi.js';
import { emptyPosition, parseTiles } from '../src/lib/physical-position.js';
import { setTiles } from '../src/lib/guided-game.js';
import { MatchSession } from '../src/lib/session.js';
import { tilesFor } from '../src/lib/yaku.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

const thrown = (tile, order, claimed = false) => ({ tile, order, drawn: false, riichi: false, claimed });
const kan = tile => ({ kind: 'concealed-kan', tile, from: 0 });
const pon = (tile, from) => ({ kind: 'pon', tile, from });

/** South's win, scored by the real engine, as the guided result shows it:
 * East discards a ron, or South draws a tsumo. A called set is claimed from
 * the seat it names, and each quad turns over an indicator of its own. */
function win(standing, winning, by = 'ron', melds = []) {
  const p = emptyPosition();
  const quads = melds.filter(meld => meld.kind.includes('kan')).length;
  Object.assign(p, { seat: 1, turn: by === 'ron' ? 0 : 1, phase: by === 'ron' ? 'call' : 'act',
    wall: 50, first_turns: false, indicators: ['7z', ...['3m', '4m', '5m', '6m'].slice(0, quads)],
    pending: by === 'ron' ? winning : null, drawn: by === 'tsumo' ? winning : null });
  p.players[1].hand = parseTiles(standing);
  p.players[1].melds = melds;
  const open = melds.filter(meld => meld.from !== 0);
  open.forEach((meld, order) => p.players[(1 + meld.from) % 4].discards.push(thrown(meld.tile, order, true)));
  if (by === 'tsumo') p.players[1].hand.push(winning);
  else p.players[0].discards.push(thrown(winning, open.length));
  const result = settle_physical({ kind: by, winners: [1], position: p, nextSeat: p.turn, needsDraw: false }, { winners: [1] });
  const winner = { ...result.winners[0], melds: melds.map(meld => ({ kind: meld.kind, tiles: setTiles(meld) })), by };
  assert.equal(winner.hand.length, 13 - 3 * melds.length, 'the separately drawn winning tile is not duplicated');
  const places = [...winner.scoring.sets, winner.scoring.pair].flatMap(g => g.tiles.map(t => t.at));
  assert.equal(places.length, 14 + quads); assert.equal(new Set(places).size, 14 + quads);
  return winner;
}
/** The tiles a yaku lights, read from where the screen draws them. */
function selected(winner, name) {
  const yaku = winner.yaku.find(y => y.name === name);
  assert.ok(yaku, `${name} must have been awarded by the scorer`);
  return tilesFor(yaku, winner).map(at => {
    const [where, index, slot] = at.split(':');
    return where === 'won' ? winner.winning_tile
      : where === 'hand' ? winner.hand[Number(index)] : winner.melds[Number(index)].tiles[Number(slot)];
  });
}

test('seat and round wind buttons point to different actual triplets', () => {
  const winner = win('123m789p6s111222z', '6s');
  assert.deepEqual(selected(winner, 'Seat Wind Triplet'), ['2z', '2z', '2z']);
  assert.deepEqual(selected(winner, 'Round Wind Triplet'), ['1z', '1z', '1z']);
});

test('ron cannot highlight its winning triplet as concealed; tsumo can', () => {
  const ron = win('11m222p333s44455z', '1m');
  const tiles = selected(ron, 'Three Concealed Triplets');
  assert.equal(tiles.length, 9); assert.ok(!tiles.includes('1m'));
  assert.equal(selected(win('11m222p333s44455z', '1m', 'tsumo'), 'Four Concealed Triplets').length, 12);
});

test('four identical sequences count as two pairs of sequences', () => {
  const winner = win('111122223333m5p', '5p');
  assert.equal(selected(winner, 'Twice Pure Double Sequence').length, 12);
});

test('two dragon triplets retain unique render identities and individual highlights', () => {
  const winner = win('123m789p4s555666z', '4s');
  const dragons = winner.yaku.filter(y => y.name === 'Dragon Triplet');
  assert.equal(dragons.length, 2);
  assert.equal(new Set(winner.yaku.map(y => y.id)).size, winner.yaku.length);
  for (const yaku of dragons) {
    assert.equal(tilesFor(yaku, winner).length, 3);
    assert.ok(tilesFor(yaku, winner).every(at => winner.hand[Number(at.split(':')[1])] === yaku.tile));
  }
});

test('ambiguous tiles follow the reading the scorer chose, not another way to split them', () => {
  const winner = win('111222333m456p5s', '5s');
  assert.equal(winner.scoring.sets.filter(g => g.kind === 'triplet').length, 3);
  assert.equal(selected(winner, 'Three Concealed Triplets').length, 9);
});

// Every yaku made of particular sets lights the sets the scorer scored, and
// no other tile: South's standing tiles, the winning tile, South's declared
// or called sets, and the tiles that make the yaku.
for (const [name, standing, winning, melds, made] of [
  ['Pure Double Sequence', '223344m678p11s55z', '5z', [], '223344m'],
  ['Mixed Triple Sequence', '345m345p345s11z99m', '9m', [], '345m345p345s'],
  ['Pure Straight', '123456789p22z55s', '5s', [], '123456789p'],
  ['Triple Triplet', '222m222p22s456m99p', '2s', [], '222m222p222s'],
  ['Three Quads', '456s7z', '7z', [kan('1m'), kan('9p'), kan('2s')], '1111m9999p2222s'],
  ['Four Quads', '7z', '7z', [kan('1m'), kan('9p'), kan('2s'), kan('5z')], '1111m9999p2222s5555z'],
  ['Dragon Triplet', '234m567p11s23s', '4s', [pon('6z', 2)], '666z'],
  ['Little Three Dragons', '555z666z77z123m56p', '4p', [], '55566677z'],
  ['Big Three Dragons', '555z666z777z12m55p', '3m', [], '555666777z'],
  ['Little Four Winds', '111z222z333z44z23m', '1m', [], '11122233344z'],
  ['Big Four Winds', '111z222z333z444z5m', '5m', [], '111222333444z'],
]) test(`${name} lights the sets that make it${melds.length ? ', declared or called ones included' : ''}`, () => {
  assert.deepEqual(selected(win(standing, winning, 'ron', melds), name).sort(), parseTiles(made).sort());
});

test('real finished-match replay upgrades pre-attribution saves without changing game state', () => {
  const match = new MatchSession(Game, 3, 'club');
  try {
    match.advance(false);
    for (let step = 0; step < 250 && match.view.phase !== 'over'; step++) {
      const choices = match.choices;
      const choice = choices.find(c => ['ron', 'tsumo'].includes(c.kind))
        ?? choices.find(c => c.kind === 'riichi') ?? choices.find(c => c.kind === 'pass')
        ?? choices.find(c => c.kind === 'discard' && c.tile === match.view.seats[0].drawn)
        ?? choices.find(c => c.kind === 'discard') ?? choices[0];
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
    }
    assert.ok(match.view.outcome.wins.length);
    const current = match.snapshot(), old = { ...current, format: 5 }, state = JSON.parse(old.state);
    for (const winner of state[0].outcome.wins) {
      delete winner.scoring;
      for (const yaku of winner.yaku) { delete yaku.id; delete yaku.tile; }
    }
    old.state = JSON.stringify(state);
    const restored = MatchSession.restore(Game, JSON.stringify(old));
    try { assert.deepEqual(restored.snapshot(), current); } finally { restored.dispose(); }
    state[0].outcome.wins[0].han++;
    assert.throws(() => MatchSession.restore(Game, JSON.stringify({ ...old, state: JSON.stringify(state) })), /does not match/);
  } finally { match.dispose(); }
});
