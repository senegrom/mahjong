/** Translate policy mass, not just one representative action, to actual moves.
 * Red-five aliases name the ordinary five in this ruleset. A reach remains a
 * declaration probability shared by its candidate discards; the second-stage
 * conditional tile probabilities are not independent first-stage moves.
 * Selection still uses the network's original best action, as in training.
 */
export const MORTAL_REACH = 37;

export function weightsByChoice(engine, choices, weights, fromMortal) {
  if (!weights || weights.length !== 46
      || Array.from(weights).some(weight => !Number.isFinite(weight) || weight < 0 || weight > 1)) {
    throw new Error('The trained network returned invalid policy weights');
  }
  const mass = new Map();
  for (let action = 0; action < weights.length; action++) {
    if (action === MORTAL_REACH || weights[action] === 0) continue;
    const index = fromMortal(action);
    if (index >= 0) mass.set(index, (mass.get(index) ?? 0) + weights[action]);
  }
  return choices.map(entry => {
    const action = entry.index == null ? -1 : engine.mortal_action_of(entry.index);
    if (action < 0) return { ...entry, weight: null };
    const weight = action === MORTAL_REACH ? weights[MORTAL_REACH] : (mass.get(entry.index) ?? 0);
    if (weight > 1 + 1e-6) throw new Error('The trained network returned invalid move probability');
    return { ...entry, weight: Math.min(1, weight) };
  });
}
