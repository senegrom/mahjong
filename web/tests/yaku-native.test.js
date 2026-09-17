/** The real WASM settlement result is what both result screens explain. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { settle_physical } from '../src/wasm/riichi.js';
import { emptyPosition, parseTiles } from '../src/lib/physical-position.js';
import { tilesFor } from '../src/lib/yaku.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

function win(standing, winning, by = 'ron') {
  const p = emptyPosition();
  Object.assign(p, { seat: 1, turn: by === 'ron' ? 0 : 1, phase: by === 'ron' ? 'call' : 'act',
    wall: 50, first_turns: false, indicators: ['7z'], pending: by === 'ron' ? winning : null,
    drawn: by === 'tsumo' ? winning : null });
  p.players[1].hand = parseTiles(standing);
  if (by === 'tsumo') p.players[1].hand.push(winning);
  else p.players[0].discards = [{ tile: winning, order: 0, drawn: false, riichi: false, claimed: false }];
  const result = settle_physical({ kind: by, winners: [1], position: p, nextSeat: p.turn, needsDraw: false }, { winners: [1] });
  const winner = { ...result.winners[0], melds: [], by };
  assert.equal(winner.hand.length, 13, 'the separately drawn winning tile is not duplicated');
  const places = [...winner.scoring.sets, winner.scoring.pair].flatMap(g => g.tiles.map(t => t.at));
  assert.equal(places.length, 14); assert.equal(new Set(places).size, 14);
  return winner;
}
function selected(winner, name) {
  const yaku = winner.yaku.find(y => y.name === name);
  assert.ok(yaku, `${name} must have been awarded by the scorer`);
  const locations = tilesFor(yaku, winner);
  return locations.map(at => at === 'won' ? winner.winning_tile : winner.hand[Number(at.split(':')[1])]);
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

test('ambiguous tiles follow the selected score, not the first JavaScript decomposition', () => {
  const winner = win('111222333m456p5s', '5s');
  assert.equal(winner.scoring.sets.filter(g => g.kind === 'triplet').length, 3);
  assert.equal(selected(winner, 'Three Concealed Triplets').length, 9);
});
