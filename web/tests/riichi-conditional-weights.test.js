import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';
import { weightsByChoice, groupPolicyChoices, MORTAL_REACH } from '../src/lib/action-weights.js';
import { readBeliefs, readValue } from '../src/lib/beliefs.js';
import { reviewWithStrong } from '../src/lib/review-policy.js';
import { averagedAnswer } from '../src/lib/suits.js';
import { loadModule } from './fixtures/load-module.js';
import { renamingFault } from './fixtures/renamed-planes.js';
import { riichiPosition } from './fixtures/riichi-position.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-7, `${actual} != ${expected}`);

test('real WASM live riichi weights retain both stages and the original selected action', async () => {
  const { match, first, firstRenamed, second, secondRenamed, declared, tile, actions, preferred, alternative } = riichiPosition(Game);
  try {
    const original = match.snapshot(); let calls = 0;
    // Each question as the engine writes it, and, Trained AI hesitating over
    // both, the renamed game's after it.
    const engine = match.engine, expected = [engine.agent_observation_mortal(), engine.agent_observation_mortal_renamed(),
      engine.agent_observation_after_reach(), engine.agent_observation_after_reach_renamed()];
    assert.equal(renamingFault(expected[0], expected[1]), null, 'the renamed game\'s planes');
    assert.equal(renamingFault(expected[2], expected[3]), null, 'the renamed game\'s planes, its reach declared');
    const signal = new AbortController().signal;
    const analyzePolicy = async (planes, mask, passed) => {
      assert.equal(passed, signal);
      const answer = [first, firstRenamed, second, secondRenamed][calls]; assert.ok(answer, 'no extra inference');
      assert.deepEqual(planes, expected[calls++]);
      assert.deepEqual(Array.from(mask, Boolean), answer.mask);
      return answer;
    };
    const { evaluateAgent } = loadModule('agents.js', {
      analyzePolicy, averagedAnswer, weightsByChoice, MORTAL_REACH, readBeliefs, readValue,
    }, ['evaluateAgent']);
    const result = await evaluateAgent(engine, 'full', signal);
    assert.equal(calls, 4);
    assert.equal(result.choice.index, preferred.index);
    assert.equal(result.value, -1.25);
    const group = groupPolicyChoices(result.choices).find(row => row.choices);
    close(group.weight, declared.weights[MORTAL_REACH]);
    close(group.choices.find(c => c.index === preferred.index).conditionalWeight, tile.weights[actions[1]]);
    close(group.choices.find(c => c.index === alternative.index).conditionalWeight, tile.weights[actions[0]]);
    close(group.choices.reduce((sum, choice) => sum + choice.conditionalWeight, 0), 1);
    close(groupPolicyChoices(result.choices).reduce((sum, row) => sum + (row.weight ?? row.choice?.weight ?? 0), 0), 1);
    // Ranked by joint probability, the riichi and its tile would lose to the
    // plain discard the fixture also offers.
    const plain = Math.max(...declared.weights.filter((_, action) => action !== MORTAL_REACH));
    assert.ok(declared.weights[MORTAL_REACH] * tile.weights[tile.action] < plain,
      'joint ranking would wrongly replace the selected riichi');
    assert.deepEqual(match.snapshot(), original, 'analysis cannot mutate the real game');
  } finally { match.dispose(); }
});

test('real WASM historical review maps populated conditional weights to two different riichi tiles', async () => {
  const { match, plan, declared, tile, actions, alternative, preferred } = riichiPosition(Game);
  try {
    // Record the non-preferred riichi; review must distinguish the declaration
    // from the tile, with real historical translation rather than a stub.
    const index = match.engine.review().length;
    match.apply({ type: 'choose', kind: alternative.kind, tile: alternative.tile });
    match.advance(false);
    const before = match.snapshot(), notes = match.engine.review(); let calls = 0;
    const renamed = [match.engine.review_observation_mortal_renamed(index), match.engine.review_observation_after_reach_renamed(index)];
    const rows = await reviewWithStrong(match.engine, notes, async (planes, mask) => {
      const answer = plan[calls++]; assert.ok(answer, 'no extra historical inference');
      assert.deepEqual(Array.from(mask, Boolean), answer.mask);
      // The two renamed questions are asked of the renamed game as it was.
      if (calls === notes.length + 1) assert.deepEqual(planes, renamed[0]);
      if (calls === notes.length + 3) assert.deepEqual(planes, renamed[1]);
      return answer;
    });
    assert.equal(calls, notes.length + 3, 'one conditional question, and each of the two asked again renamed');
    const row = rows[index];
    assert.equal(row.advised_kind, 'riichi'); assert.equal(row.advised_tile, preferred.tile);
    assert.equal(row.played_tile, alternative.tile); assert.equal(row.agreed, false);
    close(row.preferred_weight, declared.weights[MORTAL_REACH]); close(row.played_weight, declared.weights[MORTAL_REACH]);
    close(row.preferred_conditional_weight, tile.weights[actions[1]]); close(row.played_conditional_weight, tile.weights[actions[0]]);
    assert.deepEqual(match.snapshot(), before);
  } finally { match.dispose(); }
});
