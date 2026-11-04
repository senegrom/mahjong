import { emptyPosition, parseTiles } from '../../src/lib/physical-position.js';

/** The last hand's tables of riichi_core::all_last's tests, typed in as
 * Physical play takes a table: South 4, thirty tiles left, the south wind
 * the indicator, nothing thrown before, and nobody's tiles but those of
 * `seat`, the seat asked, which holds `hand`, `drawn` among them; the points
 * by seat. */
export function southFour({ seat = 3, hand, drawn = null, scores }) {
  const position = emptyPosition();
  Object.assign(position, { seat, turn: seat, round: 1, kyoku: 4, wall: 30, indicators: ['2z'], drawn, first_turns: false });
  position.players.forEach((player, at) => { player.score = scores[at]; });
  position.players[seat].hand = parseTiles(hand);
  return position;
}

/** North to move on 23456m 456p 678s 33s with the 9p drawn, the 9p its one
 * riichi tile; East on 20,000, South on 36,000 + `gap`, West on 28,000 -
 * `gap` and North on 36,000. Thrown undeclared, the 9p keeps North first
 * through every win while `gap` is under 1,000. */
export const probe = gap => southFour({ hand: '23456m456p678s33s9p', drawn: '9p', scores: [20000, 36000 + gap, 28000 - gap, 36000] });

/** North to move on 123456m 456p 678s 33s with the 1m drawn, which wins:
 * pinfu and menzen tsumo, which leaves North alone in last while West holds
 * 12,000 or more. Every tile is a riichi tile, and short. */
export const tsumo = west => southFour({ hand: '123456m456p678s33s', drawn: '1m', scores: [52000 - west, 38000, west, 10000] });

/** West has just thrown the 1m, which North, on 23456m 456p 678s 33s, may
 * chii or ron: pinfu, which leaves North alone in last while West holds more
 * than 12,000. */
export function ron(west) {
  const position = southFour({ hand: '23456m456p678s33s', scores: [52000 - west, 38000, west, 10000] });
  Object.assign(position, { turn: 2, phase: 'call', pending: '1m' });
  position.players[2].discards.push({ tile: '1m', order: 0, drawn: false, riichi: false, claimed: false });
  return position;
}
