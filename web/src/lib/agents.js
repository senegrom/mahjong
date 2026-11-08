import { analyzePolicy } from './policy.js';
import { readBeliefs, readValue } from './beliefs.js';
import { weightsByChoice } from './action-weights.js';
import { trainedName } from './trained-network.js';
import { agentTable, trainedMove } from './trained-move.js';

export const AGENTS = Object.freeze({ beginner: 'Beginner', club: 'Club', full: 'Trained' });
export const isTrained = agent => agent === 'full';
/** An agent as a mode names it where it is picked, followed or advises:
 * "Club", or "Trained 0.1.3000" with the version of the network this page plays. */
export const agentName = agent => isTrained(agent) ? trainedName() : AGENTS[agent];

export async function evaluateAgent(engine, agent, signal) {
  const choices = engine.agent_choices();
  if (!choices.length) throw new Error('This seat has no decision to make yet');
  if (isTrained(agent)) {
    // Trained AI's move, as it is asked everywhere (trained-move.js): each
    // question of the renamed game too where it hesitates, its answer then
    // the mean of the two (suits.js), and the last hand's rules over that.
    // The move is translated as a riichi only where it is one: a riichi the
    // rules throw undeclared is the plain discard of its tile.
    const { first, second, final, ruled } = await trainedMove(analyzePolicy, agentTable(engine), signal);
    const index = engine.agent_action_from_mortal(final.action, final.afterReach);
    const choice = choices.find(entry => entry.index === index);
    if (!choice) throw new Error('The agent did not return a legal choice');
    return { agent, choice, kind: 'policy',
      // Which of the last hand's rules made the move, if any did.
      ruled: ruled ? { rules: ruled.rules } : null,
      // What the network makes of the position beside the move: what it
      // expects the reward it learned from to come to (valueWords names that
      // reward), and what it thinks the other three hold, both the plain
      // answer's.
      // How many concealed tiles each of them holds turns a guessed share of
      // a hand into a guessed number of copies.
      value: readValue(first.value),
      beliefs: first.hands ? readBeliefs(first.hands, Array.from(engine.concealed_counts())) : null,
      // The network's own weights, a riichi's tiles weighed wherever its
      // tile was asked: a move a rule made keeps its own weight.
      choices: weightsByChoice(engine, choices, first.weights, action => engine.agent_action_from_mortal(action, false),
        second ? { weights: second.weights, fromMortal: action => engine.agent_action_from_mortal(action, true) } : null)
        .sort((a, b) => (b.weight ?? -1) - (a.weight ?? -1) || (a.index ?? 99) - (b.index ?? 99)) };
  }
  if (!(agent in AGENTS)) throw new Error('Unknown agent');
  const choice = engine.agent_pick(agent);
  return { agent, choice, kind: 'selection', choices: choices.map(entry => ({ ...entry,
    weight: entry.kind === choice.kind && (entry.tile ?? null) === (choice.tile ?? null) ? 1 : 0,
  })).sort((a, b) => b.weight - a.weight || (a.index ?? 99) - (b.index ?? 99)) };
}
