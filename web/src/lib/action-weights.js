/** Translate the policy's weights over Mortal's actions to our moves. Each of
 * our moves has at most one Mortal action, so a weight is never summed.
 * Mortal's red-five actions mean nothing in this ruleset, which has no red
 * fives: they are never opened, so they carry no mass. A reach remains a
 * declaration probability shared by its candidate discards; the second-stage
 * conditional tile probabilities are not independent first-stage moves.
 * Selection still uses the network's original best action, as in training.
 */
export const MORTAL_REACH = 37;

function massByChoice(weights, fromMortal) {
  if (!weights || weights.length !== 46
      || Array.from(weights).some(weight => !Number.isFinite(weight) || weight < 0 || weight > 1)) {
    throw new Error('The trained network returned invalid policy weights');
  }
  const mass = new Map();
  for (let action = 0; action < weights.length; action++) {
    if (action === MORTAL_REACH || weights[action] === 0) continue;
    const index = fromMortal(action);
    if (index >= 0) mass.set(index, weights[action]);
  }
  return mass;
}

/** Keep first-stage weights for callers that compare declarations. Conditional
 * discard weights are additional presentation data, never a selection rule.
 * `reach` contains only the answer to an already requested second stage. */
export function weightsByChoice(engine, choices, weights, fromMortal, reach = null) {
  const mass = massByChoice(weights, fromMortal);
  const conditional = reach ? massByChoice(reach.weights, reach.fromMortal) : null;
  return choices.map(entry => {
    const action = entry.index == null ? -1 : engine.mortal_action_of(entry.index);
    if (action < 0) return { ...entry, weight: null };
    if (action !== MORTAL_REACH) return { ...entry, weight: mass.get(entry.index) ?? 0 };
    return { ...entry, weight: weights[MORTAL_REACH],
      conditionalWeight: conditional ? (conditional.get(entry.index) ?? 0) : null };
  });
}

/** One first-stage row per decision, with riichi's conditional discards nested.
 * This never reorders the input array, mutates choices or picks a move. */
export function groupPolicyChoices(choices) {
  const rows = [];
  let reach = null;
  for (const choice of choices) {
    if (choice.kind !== 'riichi') { rows.push({ choice }); continue; }
    if (!reach) { reach = { weight: choice.weight, choices: [] }; rows.push(reach); }
    reach.choices.push(choice);
  }
  if (reach) reach.choices.sort((a, b) => (b.conditionalWeight ?? -1) - (a.conditionalWeight ?? -1));
  return rows;
}
