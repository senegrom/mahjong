import test from 'node:test';
import assert from 'node:assert/strict';

// The agents module reaches the policy client, which reads the page's base
// address as it loads; node has no page, so it is given one.
globalThis.document ??= { baseURI: 'https://test.invalid/mahjong/' };
const { evaluateAgent } = await import('../src/lib/agents.js');

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
