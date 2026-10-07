/** Configuration is in fixed physical order from the human: right, opposite,
 * left. Wind labels rotate between hands; these controller assignments do not.
 */
export const OPPONENT_LABELS = Object.freeze({ beginner: 'Beginner', club: 'Club', neural: 'Trained' });
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
