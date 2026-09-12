import { analyzePolicy } from './policy.js';
import { readBeliefs, readValue } from './beliefs.js';
import { weightsByChoice, MORTAL_REACH } from './action-weights.js';

export const AGENTS = Object.freeze({ beginner: 'Beginner', club: 'Club', full: 'Trained' });
export const WINDS = Object.freeze(['East', 'South', 'West', 'North']);
export const isTrained = agent => agent === 'full';
export const controller = agent => isTrained(agent) ? 'neural' : agent;

/** Whether an engine can be asked the trained network's questions at all:
 * the observation and mask in Mortal's space, the second question a reach
 * asks, and the translation back into our moves.
 *
 * A game in progress has all of them. So does a position typed into the
 * guided or physical table, since the engine replays it into the events
 * Mortal's encoder needs (see `mortal_log.rs`); an engine that lacks any
 * of the six is refused here rather than being asked half a question. */
export function supportsTrainedAgent(engine) {
  return ['agent_observation_mortal', 'agent_mask_mortal',
    'agent_observation_after_reach', 'agent_mask_after_reach',
    'agent_action_from_mortal', 'mortal_action_of']
    .every(name => typeof engine?.[name] === 'function');
}
export const TRAINED_HISTORY_REQUIRED = 'Trained advice needs an engine that can build the network\'s '
  + 'observation for this table. Choose Beginner or Club here; Trained play remains available in Play, Watch and hand review.';

export async function evaluateAgent(engine, agent, signal) {
  const choices = engine.agent_choices();
  if (!choices.length) throw new Error('This seat has no decision to make yet');
  if (isTrained(agent)) {
    if (!supportsTrainedAgent(engine)) {
      throw new Error(TRAINED_HISTORY_REQUIRED);
    }
    let { action, weights, value, hands } = await analyzePolicy(
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
      // What the network makes of the position beside the move: what it
      // thinks the hand is worth, and what it thinks the other three hold.
      value: readValue(value),
      beliefs: hands ? readBeliefs(hands, concealedCounts(engine)) : null,
      choices: weightsByChoice(engine, choices, weights, action => engine.agent_action_from_mortal(action, false))
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
