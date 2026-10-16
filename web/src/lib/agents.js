import { analyzePolicy } from './policy.js';
import { readBeliefs, readValue } from './beliefs.js';
import { weightsByChoice, MORTAL_REACH } from './action-weights.js';
import { trainedName } from './trained-network.js';

export const AGENTS = Object.freeze({ beginner: 'Beginner', club: 'Club', full: 'Trained' });
export const isTrained = agent => agent === 'full';
/** An agent as a mode names it where it is picked, followed or advises:
 * "Club", or "Trained 0.1.810" with the version of the network this page plays. */
export const agentName = agent => isTrained(agent) ? trainedName() : AGENTS[agent];

export async function evaluateAgent(engine, agent, signal) {
  const choices = engine.agent_choices();
  if (!choices.length) throw new Error('This seat has no decision to make yet');
  if (isTrained(agent)) {
    let { action, weights, value, hands } = await analyzePolicy(
      engine.agent_observation_mortal(), engine.agent_mask_mortal(), signal,
    );
    let afterReach = false, reachWeights = null;
    if (action === MORTAL_REACH) {
      afterReach = true;
      ({ action, weights: reachWeights } = await analyzePolicy(
        engine.agent_observation_after_reach(), engine.agent_mask_after_reach(), signal,
      ));
    }
    const index = engine.agent_action_from_mortal(action, afterReach);
    const choice = choices.find(entry => entry.index === index);
    if (!choice) throw new Error('The agent did not return a legal choice');
    return { agent, choice, kind: 'policy',
      // What the network makes of the position beside the move: what it
      // thinks the hand is worth, and what it thinks the other three hold.
      // How many concealed tiles each of them holds turns a guessed share of
      // a hand into a guessed number of copies.
      value: readValue(value),
      beliefs: hands ? readBeliefs(hands, Array.from(engine.concealed_counts())) : null,
      choices: weightsByChoice(engine, choices, weights, action => engine.agent_action_from_mortal(action, false),
        reachWeights ? { weights: reachWeights, fromMortal: action => engine.agent_action_from_mortal(action, true) } : null)
        .sort((a, b) => (b.weight ?? -1) - (a.weight ?? -1) || (a.index ?? 99) - (b.index ?? 99)) };
  }
  if (!(agent in AGENTS)) throw new Error('Unknown agent');
  const choice = engine.agent_pick(agent);
  return { agent, choice, kind: 'selection', choices: choices.map(entry => ({ ...entry,
    weight: entry.kind === choice.kind && (entry.tile ?? null) === (choice.tile ?? null) ? 1 : 0,
  })).sort((a, b) => b.weight - a.weight || (a.index ?? 99) - (b.index ?? 99)) };
}
