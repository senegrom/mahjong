import { trainedName } from './trained-network.js';

/** Configuration is in fixed physical order from the human: right, opposite,
 * left. Wind labels rotate between hands; these controller assignments do not.
 */
export const OPPONENT_LABELS = Object.freeze({ beginner: 'Beginner', club: 'Club', neural: 'Trained' });
/** An opponent as the table names it: "Club", or "Trained 0.1.3000" with the
 * version of the network this page plays. */
export const opponentName = type => type === 'neural' ? trainedName() : OPPONENT_LABELS[type];
export const OPPONENT_TYPES = Object.freeze(/** @type {(keyof typeof OPPONENT_LABELS)[]} */ (Object.keys(OPPONENT_LABELS)));
export const OPPONENT_POSITIONS = Object.freeze(['Right', 'Opposite', 'Left']);
/** Who a player meets when nothing has been chosen: the trained network at
 * every seat. Its download starts with the first match that needs it. */
export const DEFAULT_OPPONENT = 'neural';

export function normalizeOpponents(value) {
  if (OPPONENT_TYPES.includes(value)) return [value, value, value];
  if (!Array.isArray(value) || value.length !== 3 || !Array.from(value).every(type => OPPONENT_TYPES.includes(type))) {
    throw new Error('Invalid opponents: choose Beginner, Club or Trained for each of the three players');
  }
  return [...value];
}

export function opponentPreset(opponents) {
  return opponents.every(type => type === opponents[0]) ? opponents[0] : 'custom';
}

/** The same controllers in the same seats: an equal copy is not a change. */
export function sameOpponents(a, b) {
  return a.length === b.length && a.every((type, index) => type === b[index]);
}

// A stand-in: Club in a Trained opponent's seat for one match, taken by a
// player who would rather play on than wait for the network to download
// (Play's "Play Club meanwhile"). The match is Club's from then on, and the
// page shows it so, but the table the player chose stays their preference,
// so that the next game is against Trained once the network is there. The
// stand-in ends when the player picks opponents themself. Nothing marks it in
// what is saved: after a reload it is known by the two tables alone.

/** The table a match stands in for, or null where it plays no stand-in:
 * `preferred` is the table the player chose, as their preferences hold it,
 * or null where they hold none, and `playing` the match's. They differ, and
 * only where the player chose Trained and the match plays Club. Any other
 * difference is the match's own table, which then becomes the preference, as
 * it always did. */
export function standInFor(preferred, playing) {
  if (!preferred || preferred.length !== playing.length || sameOpponents(preferred, playing)) return null;
  return preferred.every((type, seat) => type === playing[seat] || (type === 'neural' && playing[seat] === 'club')) ? [...preferred] : null;
}

/** The table a new game is dealt with while a stand-in holds for `chosen`:
 * that table once the trained network is at hand (trainedAtHand), and until
 * then the stand-in again, Club in each of its Trained seats. */
export function tableAfterStandIn(chosen, atHand) {
  return chosen.map(type => (type === 'neural' && !atHand ? 'club' : type));
}

/** Whether a game against the trained network would begin without the wait
 * a stand-in is taken for. The network is saved on this device (`offline`
 * says aiReady); or, in a browser that cannot save it for offline play, it
 * can be reached (`available`) and its worker is not in the middle of
 * loading it (`loading`, the worker's own note): there it is held by the
 * worker alone, and fetched only when a Trained opponent first asks. */
export function trainedAtHand(available, offline, loading = '') {
  if (offline?.aiReady) return true;
  return offline?.supported === false && Boolean(available) && !loading;
}
