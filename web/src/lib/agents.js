import { analyzePolicy } from './policy.js';
import { weightsByChoice, MORTAL_REACH } from './action-weights.js';

export const AGENTS = Object.freeze({ beginner: 'Beginner', club: 'Club', full: 'Trained' });
export const WINDS = Object.freeze(['East', 'South', 'West', 'North']);
export const isTrained = agent => agent === 'full';
export const controller = agent => isTrained(agent) ? 'neural' : agent;

/** Model availability is not encoder availability. Position-only engines
 * cannot reproduce the complete history required by the trained policy. */
export function supportsTrainedAgent(engine) {
  return ['agent_observation_mortal', 'agent_mask_mortal',
    'agent_observation_after_reach', 'agent_mask_after_reach',
    'agent_action_from_mortal', 'mortal_action_of']
    .every(name => typeof engine?.[name] === 'function');
}
export const TRAINED_HISTORY_REQUIRED = 'Trained advice requires a complete in-app game history. '
  + 'For manually entered tables, choose Beginner or Club. Trained play remains available in Play, Watch and hand review.';

export async function evaluateAgent(engine, agent, signal) {
  const choices = engine.agent_choices();
  if (!choices.length) throw new Error('This seat has no decision to make yet');
  if (isTrained(agent)) {
    // The network reads Mortal's planes, which are built from everything
    // that has happened. A table reconstructed from a position alone has no
    // history to build them from, so it cannot be asked.
    if (!supportsTrainedAgent(engine)) {
      throw new Error(TRAINED_HISTORY_REQUIRED);
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
      choices: weightsByChoice(engine, choices, weights, action => engine.agent_action_from_mortal(action, false))
        .sort((a, b) => (b.weight ?? -1) - (a.weight ?? -1) || (a.index ?? 99) - (b.index ?? 99)) };
  }
  if (!(agent in AGENTS)) throw new Error('Unknown agent');
  const choice = engine.agent_pick(agent);
  return { agent, choice, kind: 'selection', choices: choices.map(entry => ({ ...entry,
    weight: entry.kind === choice.kind && (entry.tile ?? null) === (choice.tile ?? null) ? 1 : 0,
  })).sort((a, b) => b.weight - a.weight || (a.index ?? 99) - (b.index ?? 99)) };
}
