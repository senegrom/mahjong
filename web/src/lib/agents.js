import { analyzePolicy } from './policy.js';

export const AGENTS = Object.freeze({ beginner: 'Beginner', club: 'Club', quick: 'Trained · Quick', strong: 'Trained · Strong' });
export const WINDS = Object.freeze(['East', 'South', 'West', 'North']);
export const isTrained = agent => agent === 'quick' || agent === 'strong';
export const controller = agent => isTrained(agent) ? 'neural' : agent;

export async function evaluateAgent(engine, agent, signal) {
  const choices = engine.agent_choices();
  if (!choices.length) throw new Error('This seat has no decision to make yet');
  if (isTrained(agent)) {
    const { action, weights } = await analyzePolicy(engine.agent_observation(), engine.agent_mask(), signal, agent);
    const choice = choices.find(entry => entry.index === action);
    if (!choice) throw new Error('The agent did not return a legal choice');
    return { agent, choice, kind: 'policy', choices: choices.map(entry => ({ ...entry, weight: entry.index == null ? null : weights[entry.index] }))
      .sort((a, b) => (b.weight ?? -1) - (a.weight ?? -1) || (a.index ?? 99) - (b.index ?? 99)) };
  }
  if (!(agent in AGENTS)) throw new Error('Unknown agent');
  const choice = engine.agent_pick(agent);
  return { agent, choice, kind: 'selection', choices: choices.map(entry => ({ ...entry,
    weight: entry.kind === choice.kind && (entry.tile ?? null) === (choice.tile ?? null) ? 1 : 0,
  })).sort((a, b) => b.weight - a.weight || a.index - b.index) };
}
