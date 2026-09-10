import { analyzePolicy } from './policy.js';
import { readBeliefs, readValue } from './beliefs.js';
import { MORTAL_REACH, weightsOverOurMoves } from './mortal-space.js';

export const AGENTS = Object.freeze({ beginner: 'Beginner', club: 'Club', full: 'Trained' });
export const WINDS = Object.freeze(['East', 'South', 'West', 'North']);
export const isTrained = agent => agent === 'full';
export const controller = agent => isTrained(agent) ? 'neural' : agent;

export async function evaluateAgent(engine, agent, signal) {
  const choices = engine.agent_choices();
  if (!choices.length) throw new Error('This seat has no decision to make yet');
  if (isTrained(agent)) {
    // The network reads Mortal's planes, which are built from events. A
    // game in progress has them; a position typed in is replayed into them
    // first, which the engine does and which fails loudly if the position
    // describes a hand that cannot have happened.
    if (typeof engine.agent_observation_mortal !== 'function') {
      throw new Error('The trained network is not available for this table');
    }
    const mask = engine.agent_mask_mortal();
    let { action, weights, value, hands } = await analyzePolicy(
      engine.agent_observation_mortal(), mask, signal, agent,
    );
    // Asked whenever a reach is legal, not only when it wins: the tiles it
    // would throw are what turns one weight into a weight for each riichi.
    const after = mask[MORTAL_REACH]
      ? await analyzePolicy(engine.agent_observation_after_reach(), engine.agent_mask_after_reach(), signal, agent)
      : null;
    const afterReach = action === MORTAL_REACH;
    if (afterReach) action = after.action;
    const index = engine.agent_action_from_mortal(action, afterReach);
    const choice = choices.find(entry => entry.index === index);
    if (!choice) throw new Error('The agent did not return a legal choice');
    const spread = weightsOverOurMoves(weights, mask, after,
      (entry, reach) => engine.agent_action_from_mortal(entry, reach));
    return { agent, choice, kind: 'policy',
      // What the network makes of the position beside the move: what it
      // thinks the hand is worth, and what it thinks the other three hold.
      value: readValue(value),
      beliefs: hands ? readBeliefs(hands, concealedCounts(engine)) : null,
      choices: choices
        .map(entry => ({ ...entry, weight: entry.index == null ? null : spread.get(entry.index) ?? 0 }))
        .sort((a, b) => (b.weight ?? -1) - (a.weight ?? -1) || (a.index ?? 99) - (b.index ?? 99)) };
  }
  if (!(agent in AGENTS)) throw new Error('Unknown agent');
  const choice = engine.agent_pick(agent);
  return { agent, choice, kind: 'selection', choices: choices.map(entry => ({ ...entry,
    weight: entry.kind === choice.kind && (entry.tile ?? null) === (choice.tile ?? null) ? 1 : 0,
  })).sort((a, b) => b.weight - a.weight || (a.index ?? 99) - (b.index ?? 99)) };
}

/** How many concealed tiles each of the other three seats holds, which
 * turns a guessed share of a hand into a guessed number of copies. An
 * engine too old to say leaves the shares to speak for themselves. */
function concealedCounts(engine) {
  if (typeof engine.concealed_counts !== 'function') return null;
  const counts = Array.from(engine.concealed_counts());
  return counts.length === 3 ? counts : null;
}
