import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';
import { weightsByChoice, groupPolicyChoices, MORTAL_REACH } from '../src/lib/action-weights.js';
import { readBeliefs, readValue } from '../src/lib/beliefs.js';
import { reviewWithStrong } from '../src/lib/review-policy.js';
import { loadModule } from './fixtures/load-module.js';
import { riichiPosition } from './fixtures/riichi-position.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-7, `${actual} != ${expected}`);

test('real WASM live riichi weights retain both stages and the original selected action', async () => {
  const { match, first, second, preferred, alternative } = riichiPosition(Game);
  try {
    const original = match.snapshot(); let calls = 0;
    const expected = [match.engine.agent_observation_mortal(), match.engine.agent_observation_after_reach()];
    const signal = new AbortController().signal;
    const analyzePolicy = async (planes, mask, passed) => {
      assert.equal(passed, signal);
      const answer = [first, second][calls]; assert.ok(answer, 'no extra inference');
      assert.deepEqual(planes, expected[calls++]);
      assert.deepEqual(Array.from(mask, Boolean), answer.mask);
      return answer;
    };
    const { evaluateAgent } = loadModule('agents.js', {
      analyzePolicy, weightsByChoice, MORTAL_REACH, readBeliefs, readValue,
    }, ['evaluateAgent']);
    const result = await evaluateAgent(match.engine, 'full', signal);
    assert.equal(calls, 2);
    assert.equal(result.choice.index, preferred.index);
    assert.equal(result.value, -1.25);
    const group = groupPolicyChoices(result.choices).find(row => row.choices);
    close(group.weight, .55);
    close(group.choices.find(c => c.index === preferred.index).conditionalWeight, .75);
    close(group.choices.find(c => c.index === alternative.index).conditionalWeight, .25);
    close(group.choices.reduce((sum, choice) => sum + choice.conditionalWeight, 0), 1);
    close(groupPolicyChoices(result.choices).reduce((sum, row) => sum + (row.weight ?? row.choice?.weight ?? 0), 0), 1);
    assert.ok(.55 * .75 < .45, 'joint ranking would wrongly replace the selected riichi');
    assert.deepEqual(match.snapshot(), original, 'analysis cannot mutate the real game');
  } finally { match.dispose(); }
});

test('real WASM historical review maps populated conditional weights to two different riichi tiles', async () => {
  const { match, plan, alternative, preferred } = riichiPosition(Game);
  try {
    // Record the non-preferred riichi; review must distinguish the declaration
    // from the tile, with real historical translation rather than a stub.
    const index = match.engine.review().length;
    match.apply({ type: 'choose', kind: alternative.kind, tile: alternative.tile });
    match.advance(false);
    const before = match.snapshot(), notes = match.engine.review(); let calls = 0;
    const rows = await reviewWithStrong(match.engine, notes, async (_planes, mask) => {
      const answer = plan[calls++]; assert.ok(answer, 'no extra historical inference');
      assert.deepEqual(Array.from(mask, Boolean), answer.mask);
      return answer;
    });
    assert.equal(calls, notes.length + 1, 'exactly one conditional question');
    const row = rows[index];
    assert.equal(row.advised_kind, 'riichi'); assert.equal(row.advised_tile, preferred.tile);
    assert.equal(row.played_tile, alternative.tile); assert.equal(row.agreed, false);
    close(row.preferred_weight, .55); close(row.played_weight, .55);
    close(row.preferred_conditional_weight, .75); close(row.played_conditional_weight, .25);
    assert.deepEqual(match.snapshot(), before);
  } finally { match.dispose(); }
});
