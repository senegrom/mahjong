import { analyzePolicy } from './policy.js';

export const AGENTS = Object.freeze({ beginner: 'Beginner', club: 'Club', full: 'Trained' });
export const WINDS = Object.freeze(['East', 'South', 'West', 'North']);
export const isTrained = agent => agent === 'full';
export const controller = agent => isTrained(agent) ? 'neural' : agent;

/** Mortal's reach. It names no tile: a declaration is asked again which
 * tile it discards, from a position that already knows about it. */
const MORTAL_REACH = 37;

/** Weights over Mortal's moves, read against our own choices. Every riichi
 * discard is the one reach, so they all show the weight it was given. */
function weightsByChoice(engine, choices, weights) {
  return choices.map((entry) => {
    if (entry.index == null) return { ...entry, weight: null };
    const action = engine.mortal_action_of(entry.index);
    return { ...entry, weight: action < 0 ? null : weights[action] };
  });
}

export async function evaluateAgent(engine, agent, signal) {
  const choices = engine.agent_choices();
  if (!choices.length) throw new Error('This seat has no decision to make yet');
  if (isTrained(agent)) {
    // The network reads Mortal's planes, which are built from everything
    // that has happened. A table reconstructed from a position alone has no
    // history to build them from, so it cannot be asked.
    if (typeof engine.agent_observation_mortal !== 'function') {
      throw new Error('The trained network needs a game in progress, not a position on its own');
    }
    let { action, weights } = await analyzePolicy(
      engine.agent_observation_mortal(), engine.agent_mask_mortal(), signal, agent,
    );
    let afterReach = false;
    if (action === MORTAL_REACH) {
      afterReach = true;
      ({ action } = await analyzePolicy(
        engine.agent_observation_after_reach(), engine.agent_mask_after_reach(), signal, agent,
      ));
    }
    const index = engine.agent_action_from_mortal(action, afterReach);
    const choice = choices.find(entry => entry.index === index);
    if (!choice) throw new Error('The agent did not return a legal choice');
    return { agent, choice, kind: 'policy',
      choices: weightsByChoice(engine, choices, weights)
        .sort((a, b) => (b.weight ?? -1) - (a.weight ?? -1) || (a.index ?? 99) - (b.index ?? 99)) };
  }
  if (!(agent in AGENTS)) throw new Error('Unknown agent');
  const choice = engine.agent_pick(agent);
  return { agent, choice, kind: 'selection', choices: choices.map(entry => ({ ...entry,
    weight: entry.kind === choice.kind && (entry.tile ?? null) === (choice.tile ?? null) ? 1 : 0,
  })).sort((a, b) => b.weight - a.weight || (a.index ?? 99) - (b.index ?? 99)) };
}
