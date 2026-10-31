/** A game's planes with characters and circles renamed, as
 * neural/tests/test_suits.py renames them, for checking that what the page
 * sends as the renamed game's planes is the renamed game and not the game.
 *
 * Every tile column is renamed, and so is every row that names a tile: the
 * tiles each discard needs (rows 891-958, a row a discard) and the red fives
 * (a row a suit). The renamed game's own encoding, which the engine writes,
 * is the game's planes so renamed on every row but those of the efficiency
 * lookahead (LOOKAHEAD), which breaks ties by tile index, characters before
 * circles; and it is never the game's own planes. */
import assert from 'node:assert/strict';
import { MOVES, hesitates, swapMoves } from '../../src/lib/suits.js';

const PLANES = 1012, TILES = 34;
const rows = count => Array.from({ length: count }, (_, row) => row);

/** Tile kind `k` of a game is kind `KINDS[k]` of the renamed game. */
export const KINDS = Object.freeze(MOVES.slice(0, TILES));
/** The row of a game's planes each row of the renamed game's is read from. */
export const TILE_ROWS = Object.freeze((() => {
  const read = rows(PLANES);
  for (const start of [891, 925]) KINDS.forEach((kind, k) => { read[start + k] = start + kind; });
  for (const [a, b] of [[4, 5], [32, 33], ...[0, 1, 2, 3].map(k => [723 + 7 * k + 4, 723 + 7 * k + 5])]) [read[a], read[b]] = [b, a];
  return read;
})());
/** The efficiency lookahead's rows: its best expected points, the discard
 * with the most tiles needed, and its tables by imagined turn. */
export const LOOKAHEAD = Object.freeze([889, 890, ...rows(PLANES).slice(959)]);
/** Every other row, where the renamed game's planes are the game's renamed. */
export const EXACT = Object.freeze(rows(PLANES).filter(row => !LOOKAHEAD.includes(row)));

/** A game's planes, flat, row by row, with characters and circles renamed. */
export function renamedPlanes(planes) {
  const renamed = new Float32Array(PLANES * TILES);
  for (let row = 0; row < PLANES; row++) {
    for (let tile = 0; tile < TILES; tile++) renamed[row * TILES + tile] = planes[TILE_ROWS[row] * TILES + KINDS[tile]];
  }
  return renamed;
}

/** What is wrong with `renamed` as the renamed game's planes of the game
 * whose planes are `plain`, both flat; null where nothing is. */
export function renamingFault(plain, renamed) {
  if (plain.length !== PLANES * TILES || renamed.length !== PLANES * TILES) return `${renamed.length} values, not Mortal's ${PLANES} planes`;
  const expected = renamedPlanes(plain);
  for (const row of EXACT) {
    for (let tile = 0; tile < TILES; tile++) {
      const at = row * TILES + tile;
      if (renamed[at] !== expected[at]) return `row ${row}, tile ${tile}: ${renamed[at]}, not the game's ${expected[at]} renamed`;
    }
  }
  return Array.prototype.some.call(renamed, (value, at) => value !== plain[at]) ? null : 'the game\'s own planes';
}

/** A network that knows the symmetry exactly, asked as analyzeBatch is: a
 * request of the game's tables answered by `answer(planes, mask)`, each in
 * turn, and the request after one with tables Trained AI hesitates over
 * taken for those tables' renamed games, in order, each checked to be the
 * renamed game's planes and moves, and answered as the plain answer renamed,
 * so that the mean is the plain answer. `requests` notes each request's size
 * and whether it was of renamed games. */
export function mirroringNetwork(answer) {
  let hesitant = [];
  const requests = [];
  const analyzeBatch = async positions => {
    if (hesitant.length) {
      const asked = hesitant;
      hesitant = [];
      requests.push({ size: positions.length, renamed: true });
      assert.equal(positions.length, asked.length, 'the renamed game of each table hesitated over, and of no other');
      return positions.map(({ planes, mask }, row) => {
        assert.equal(renamingFault(asked[row].planes, planes), null, 'the renamed game\'s planes');
        assert.deepEqual(Array.from(mask, Number), swapMoves(Array.from(asked[row].mask, Number)), 'the renamed game\'s moves');
        const { action, weights } = asked[row].reply;
        return { action: action == null ? null : MOVES[action], weights: swapMoves(weights) };
      });
    }
    requests.push({ size: positions.length, renamed: false });
    const replies = positions.map(({ planes, mask }) => answer(planes, mask));
    hesitant = positions.flatMap(({ planes, mask }, row) => (hesitates(replies[row].weights) ? [{ planes: Float32Array.from(planes), mask, reply: replies[row] }] : []));
    return replies;
  };
  return { analyzeBatch, requests };
}
