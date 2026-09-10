import { TILES, parsePhysical } from './physical-position.js';

const seat = n => Number.isInteger(n) && n >= 0 && n < 4;
const vector = a => Array.isArray(a) && a.length === 4 && a.every(Number.isSafeInteger);
const tiles = a => Array.isArray(a) && a.length <= 18 && a.every(t => TILES.includes(t));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sum = a => a.reduce((total, n) => total + n, 0);
const seats = a => Array.isArray(a) && a.length <= 4 && a.every(seat) && new Set(a).size === a.length;
const count = n => Number.isInteger(n) && n >= 0 && n <= 100;

/** Keep the last actionable position, not the already-over display position.
 * A ron tile remains at its source; each winning hand uses it for scoring
 * without manufacturing extra physical copies for multiple winners. */
export function rememberEnding(state, kind, winners = []) {
  if (!['ron', 'tsumo', 'draw', 'manual'].includes(kind) || !seats(winners)) throw new Error('Choose a valid hand result');
  state.ending = { kind, winners, position: structuredClone(state.position), nextSeat: state.nextSeat, needsDraw: state.needsDraw };
  delete state.settlement;
}

export function validSettlement(result, ending) {
  return result && result.kind === ending.kind
    && vector(result.before) && vector(result.after) && vector(result.deltas)
    && result.deltas.every((n, i) => n === result.after[i] - result.before[i])
    && count(result.sticks_before) && count(result.sticks_after)
    && result.sticks_before === ending.position.riichi_sticks
    && same(result.before, ending.position.players.map(p => p.score))
    && sum(result.after) + result.sticks_after * 1000 === sum(result.before) + result.sticks_before * 1000
    && typeof result.repeat === 'boolean' && Number.isInteger(result.next_counters)
    && result.next_counters === (ending.kind === 'draw' || result.repeat ? ending.position.counters + 1 : 0)
    && Array.isArray(result.winners) && result.winners.length <= 3
    && seats(result.winners.map(w => w.seat)) && seats(result.tenpai)
    && (result.refunded_riichi == null || seat(result.refunded_riichi))
    && (result.hand_deltas == null || vector(result.hand_deltas))
    && result.winners.every(w => TILES.includes(w.winning_tile) && tiles(w.hand)
      && Number.isInteger(w.han) && w.han >= 0 && Number.isInteger(w.fu) && w.fu >= 0
      && Number.isInteger(w.dora) && w.dora >= 0 && Number.isInteger(w.ura_dora) && w.ura_dora >= 0
      && (w.limit == null || ['mangan', 'haneman', 'baiman', 'sanbaiman', 'yakuman'].includes(w.limit))
      && tiles(w.indicators) && w.indicators.length <= 5 && tiles(w.ura_indicators) && w.ura_indicators.length <= 5
      && Number.isInteger(w.hand_payment) && w.hand_payment >= 0
      && Array.isArray(w.yaku) && w.yaku.length > 0 && w.yaku.length <= 50
      && w.yaku.every(y => typeof y.name === 'string' && y.name.length < 100 && Number.isInteger(y.han) && y.han >= 0 && typeof y.yakuman === 'boolean')
      && Array.isArray(w.fu_detail) && w.fu_detail.length <= 20
      && w.fu_detail.every(f => Array.isArray(f) && f.length === 2 && typeof f[0] === 'string' && f[0].length < 100 && Number.isInteger(f[1]) && f[1] >= 0))
    && (ending.kind === 'draw'
      ? result.winners.length === 0 && result.repeat === result.tenpai.includes(0) && result.sticks_after === result.sticks_before
      : result.winners.length > 0 && (ending.kind !== 'tsumo' || result.winners.length === 1)
        && ending.winners.every(i => result.winners.some(w => w.seat === i))
        && result.repeat === result.winners.some(w => w.seat === 0) && result.tenpai.length === 0 && result.sticks_after === 0);
}

/** State validation is also used for old undo snapshots. Legacy finished
 * hands lack an ending and remain explicitly manual; never replay them. */
export function validEndingState(state) {
  if (state.opening != null && !vector(state.opening)) return false;
  const ending = state.ending;
  if (ending == null) return state.settlement == null;
  if (state.stage !== 'over' || !['ron', 'tsumo', 'draw', 'manual'].includes(ending.kind)
    || !seats(ending.winners) || !seat(ending.nextSeat) || typeof ending.needsDraw !== 'boolean'
    || !parsePhysical(JSON.stringify({ version: 1, position: ending.position }))) return false;
  if (ending.kind === 'manual') return state.settlement == null;
  if (state.settlement == null) return same(state.position.players.map(p => p.score), ending.position.players.map(p => p.score))
    && state.position.riichi_sticks === ending.position.riichi_sticks;
  return validSettlement(state.settlement, ending)
    && same(state.position.players.map(p => p.score), state.settlement.after)
    && state.position.riichi_sticks === state.settlement.sticks_after
    && (state.opening == null || same(state.settlement.hand_deltas, state.settlement.after.map((n, i) => n - state.opening[i])));
}

/** Only the injected Rust scorer supplies payouts. The UI never submits
 * trusted balances, cached previews, han, fu, or arbitrary payment amounts. */
export function applySettlement(state, input, score) {
  if (!state.ending || state.ending.kind === 'manual') throw new Error('This hand needs a manual table settlement');
  if (state.settlement) throw new Error('This hand has already been settled');
  if (!validEndingState(state)) throw new Error('The hand changed before settlement. Undo the last step.');
  if (typeof score !== 'function') throw new Error('The scoring engine is not ready');
  const result = structuredClone(score(structuredClone(state.ending), structuredClone(input)));
  if (!validSettlement(result, state.ending)) throw new Error('The scoring result is invalid; no points were changed');
  result.hand_deltas = state.opening ? result.after.map((n, i) => n - state.opening[i]) : null;
  result.input = structuredClone(input);
  state.position.players.forEach((p, i) => { p.score = result.after[i]; });
  state.position.riichi_sticks = result.sticks_after;
  if (result.refunded_riichi != null) {
    const p = state.position.players[result.refunded_riichi];
    p.riichi = 'none'; p.ippatsu = false;
    if (p.discards.at(-1)) p.discards.at(-1).riichi = false;
  }
  state.settlement = result;
  return `Settlement · ${result.deltas.map(n => `${n >= 0 ? '+' : ''}${n}`).join(' / ')} · riichi pot ${result.sticks_after}`;
}

/** Recompute saved results on read, without applying them. A balanced but
 * altered ledger or yaku list is not an authoritative saved settlement. */
export function verifySettlement(state, score) {
  if (!state.settlement) return true;
  if (!state.settlement.input || typeof score !== 'function') return false;
  const actual = score(structuredClone(state.ending), structuredClone(state.settlement.input));
  return validSettlement(actual, state.ending)
    && Object.keys(actual).every(key => same(actual[key], state.settlement[key]));
}
