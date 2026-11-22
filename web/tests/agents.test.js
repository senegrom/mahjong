import test from 'node:test';
import assert from 'node:assert/strict';
import { AGENT_IDS, TRAINED_AGENT } from '../src/lib/agent-ids.js';
import { sameChoice } from '../src/lib/action-weights.js';
import { emptyGuided } from '../src/lib/guided-game.js';

// The agents module reaches the policy client, which reads the page's base
// address as it loads; node has no page, so it is given one.
globalThis.document ??= { baseURI: 'https://test.invalid/mahjong/' };
const { AGENTS, evaluateAgent, isTrained } = await import('../src/lib/agents.js');

test('the agents the page names are the ones a saved guided game and a watched table know', () => {
  // agents.js cannot be imported where there is no page, so the modules that
  // run without one read the list from agent-ids.js: the two must agree.
  assert.deepEqual(Object.keys(AGENTS), [...AGENT_IDS]);
  assert.deepEqual(AGENT_IDS.filter(isTrained), [TRAINED_AGENT]);
  for (const agent of AGENT_IDS) assert.equal(emptyGuided(agent).state.agent, agent);
});

test('two choices are the same move by kind and tile, a pass with no tile or a null one alike', () => {
  assert.equal(sameChoice({ kind: 'pass' }, { kind: 'pass', tile: null }), true);
  assert.equal(sameChoice({ kind: 'discard', tile: '1m' }, { kind: 'discard', tile: '1m', label: 'discards the 1 characters' }), true);
  assert.equal(sameChoice({ kind: 'discard', tile: '1m' }, { kind: 'riichi', tile: '1m' }), false);
  assert.equal(sameChoice({ kind: 'discard', tile: '1m' }, { kind: 'discard', tile: '2m' }), false);
});

test('a built-in agent lists the unscored kans after the ranked choices, in a stable order', async () => {
  const choices = [
    { index: 5, kind: 'discard', tile: '1m', label: 'discard' },
    { index: null, kind: 'concealed-kan', tile: '9s', label: 'kan' },
    { index: 2, kind: 'discard', tile: '2m', label: 'discard' },
    { index: null, kind: 'extended-kan', tile: '3p', label: 'kan' },
  ];
  const engine = { agent_choices: () => choices, agent_pick: () => ({ kind: 'discard', tile: '2m' }) };
  const result = await evaluateAgent(engine, 'club');
  assert.deepEqual(result.choices.map(entry => [entry.kind, entry.tile, entry.weight]), [
    ['discard', '2m', 1], ['discard', '1m', 0], ['concealed-kan', '9s', 0], ['extended-kan', '3p', 0],
  ]);
  assert.ok(result.choices.every(entry => !Number.isNaN(entry.weight)));
});
