import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis } from '../src/wasm/riichi.js';
import { emptyPosition, parseTiles } from '../src/lib/physical-position.js';

// Loading policy.js resolves URLs, but an unsupported adapter must never
// construct a worker or download the trained model.
globalThis.document = { baseURI: 'https://example.invalid/mahjong/' };
const { evaluateAgent, supportsTrainedAgent, TRAINED_HISTORY_REQUIRED } = await import('../src/lib/agents.js');
await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

test('trained capability distinguishes the actual live and position-only adapters', () => {
  assert.equal(supportsTrainedAgent(Game.prototype), true);
  assert.equal(supportsTrainedAgent(PhysicalAnalysis.prototype), false);
  assert.equal(supportsTrainedAgent(null), false);
  assert.equal(supportsTrainedAgent({}), false);
});

test('trained capability requires both riichi stages and action translation', () => {
  const names = ['agent_observation_mortal', 'agent_mask_mortal',
    'agent_observation_after_reach', 'agent_mask_after_reach',
    'agent_action_from_mortal', 'mortal_action_of'];
  const complete = Object.fromEntries(names.map(name => [name, () => {}]));
  assert.equal(supportsTrainedAgent(complete), true);
  for (const name of names) {
    assert.equal(supportsTrainedAgent({ ...complete, [name]: undefined }), false, name);
    assert.equal(supportsTrainedAgent({ ...complete, [name]: true }), false, name);
  }
});

test('a physical position rejects trained inference without falling back or changing legal choices', async () => {
  const position = emptyPosition();
  position.players[0].hand = parseTiles('123m456p789s11234z');
  position.drawn = '4z'; position.indicators = ['5z'];
  const original = structuredClone(position), engine = new PhysicalAnalysis(position);
  try {
    const choices = engine.agent_choices();
    await assert.rejects(evaluateAgent(engine, 'full'), { message: TRAINED_HISTORY_REQUIRED });
    assert.deepEqual(engine.agent_choices(), choices);
    for (const agent of ['beginner', 'club']) {
      const answer = await evaluateAgent(engine, agent);
      assert.equal(answer.agent, agent);
      assert.equal(answer.kind, 'selection');
      assert.ok(choices.some(choice => choice.kind === answer.choice.kind && choice.tile === answer.choice.tile));
    }
    assert.deepEqual(position, original);
  } finally { engine.free(); }
});
