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

test('trained capability is met by the live game and by a typed-in position alike', () => {
  // A position typed into the guided or physical table is replayed into
  // the events Mortal's encoder needs (mortal_log.rs), so the position-only
  // engine answers the same six questions the live game does.
  assert.equal(supportsTrainedAgent(Game.prototype), true);
  assert.equal(supportsTrainedAgent(PhysicalAnalysis.prototype), true);
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

test('a physical position builds the trained observation and keeps its legal choices', async () => {
  const position = emptyPosition();
  position.players[0].hand = parseTiles('123m456p789s11234z');
  position.drawn = '4z'; position.indicators = ['5z'];
  const original = structuredClone(position), engine = new PhysicalAnalysis(position);
  try {
    const choices = engine.agent_choices();
    // The observation the network would be asked, built from the replay:
    // Mortal's planes over 34 tiles, and a mask over Mortal's 46 moves.
    const planes = engine.agent_observation_mortal();
    assert.equal(planes.length % 34, 0);
    assert.ok(planes.length / 34 > 900, 'not Mortal planes');
    assert.equal(engine.agent_mask_mortal().length, 46);
    assert.deepEqual(engine.agent_choices(), choices);
    // An engine that cannot answer the six questions is still refused by
    // name rather than being asked half of one.
    await assert.rejects(evaluateAgent({ agent_choices: () => choices }, 'full'), { message: TRAINED_HISTORY_REQUIRED });
    for (const agent of ['beginner', 'club']) {
      const answer = await evaluateAgent(engine, agent);
      assert.equal(answer.agent, agent);
      assert.equal(answer.kind, 'selection');
      assert.ok(choices.some(choice => choice.kind === answer.choice.kind && choice.tile === answer.choice.tile));
    }
    assert.deepEqual(position, original);
  } finally { engine.free(); }
});
