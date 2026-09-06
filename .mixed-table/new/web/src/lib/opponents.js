/** Configuration is in fixed physical order from the human: right, opposite,
 * left. Wind labels rotate between hands; these controller assignments do not.
 */
export const OPPONENT_LABELS = Object.freeze({ beginner: 'Beginner', club: 'Club', neural: 'Trained' });
export const OPPONENT_TYPES = Object.freeze(Object.keys(OPPONENT_LABELS));
export const OPPONENT_POSITIONS = Object.freeze(['Right', 'Opposite', 'Left']);

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
