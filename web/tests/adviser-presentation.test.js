import test from 'node:test';
import assert from 'node:assert/strict';
import { weightsByChoice, groupPolicyChoices, MORTAL_REACH } from '../src/lib/action-weights.js';
import { reviewWithStrong } from '../src/lib/review-policy.js';
import { loadModule } from './fixtures/load-module.js';

const weights = entries => Object.freeze(Array.from({ length: 46 }, (_, index) => entries[index] ?? 0));
const choices = Object.freeze([
  { index: 0, kind: 'discard', tile: '1m', label: 'Discard 1m' },
  { index: 1, kind: 'discard', tile: '2m', label: 'Discard 2m' },
  { index: 2, kind: 'discard', tile: '3m', label: 'Discard 3m' },
  { index: 34, kind: 'riichi', tile: '1m', label: 'Riichi 1m' },
  { index: 35, kind: 'riichi', tile: '2m', label: 'Riichi 2m' },
  { index: null, kind: 'concealed-kan', tile: '7z', label: 'Extra kan' },
].map(Object.freeze));

function engine() {
  const planes = new Float32Array([1]), after = new Float32Array([2]);
  const mask = Array.from({ length: 46 }, (_, index) => [0, 1, 2, 37].includes(index));
  const afterMask = Array.from({ length: 46 }, (_, index) => [0, 1].includes(index));
  const from = (action, reached) => reached ? ([0, 1].includes(action) ? action + 34 : -1)
    : ([0, 1, 2].includes(action) ? action : -1);
  return {
    planes, after, mask, afterMask,
    agent_choices: () => choices,
    agent_observation_mortal: () => planes, agent_mask_mortal: () => mask,
    agent_observation_after_reach: () => after, agent_mask_after_reach: () => afterMask,
    agent_action_from_mortal: from,
    mortal_action_of: index => [34, 35].includes(index) ? MORTAL_REACH : index,
    review_choices: () => choices,
    review_observation_mortal: () => planes, review_mask_mortal: () => mask,
    review_observation_after_reach: () => after, review_mask_after_reach: () => afterMask,
    review_action_from_mortal: (_index, action, reached) => from(action, reached),
  };
}
function analyzer(responses) {
  const calls = [];
  return { calls, async analyze(...args) {
    calls.push(args);
    assert.ok(calls.length <= responses.length, 'display must not request extra inference');
    return responses[calls.length - 1];
  } };
}
const evaluator = analyzePolicy => loadModule('agents.js', {
  analyzePolicy, weightsByChoice, MORTAL_REACH,
  readValue: value => value ?? null, readBeliefs: () => null,
}, ['evaluateAgent']).evaluateAgent;
const note = choice => ({ turn: 1, played: choice.label, played_kind: choice.kind, played_tile: choice.tile });

test('one declaration row has separate conditional discards; neither distribution is multiplied', async () => {
  const e = engine(), signal = new AbortController().signal;
  const a = analyzer([
    { action: 37, weights: weights({ 37: .8, 2: .2 }), value: -1.25 },
    { action: 1, weights: weights({ 0: .25, 1: .75 }), value: 99 },
  ]);
  const result = await evaluator(a.analyze)(e, 'full', signal);
  assert.strictEqual(result.choice, choices[4], 'second-stage action, not first matching riichi row');
  assert.equal(result.value, -1.25, 'keep the original position value, not the post-declaration value');
  assert.equal(a.calls.length, 2);
  assert.strictEqual(a.calls[0][0], e.planes); assert.strictEqual(a.calls[0][1], e.mask);
  assert.strictEqual(a.calls[1][0], e.after); assert.strictEqual(a.calls[1][1], e.afterMask);
  assert.ok(a.calls.every(call => call[2] === signal));
  const before = JSON.stringify(result.choices), rows = groupPolicyChoices(result.choices);
  const groups = rows.filter(row => row.choices);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].weight, .8);
  assert.deepEqual(groups[0].choices.map(choice => choice.conditionalWeight), [.75, .25]);
  assert.equal(rows.reduce((sum, row) => sum + (row.choice?.weight ?? row.weight ?? 0), 0), 1);
  assert.equal(groups[0].choices.reduce((sum, choice) => sum + choice.conditionalWeight, 0), 1);
  assert.equal(JSON.stringify(result.choices), before, 'sorting display rows must not mutate command choices');
  assert.ok(groups[0].choices.every(choice => result.choices.includes(choice)), 'confirmation receives original choice objects');
});

test('a lower joint probability cannot override the policy\'s two-stage choice', async () => {
  const e = engine(), a = analyzer([
    { action: 37, weights: weights({ 37: .4, 2: .35, 1: .25 }) },
    { action: 1, weights: weights({ 0: .4, 1: .6 }) },
  ]);
  const result = await evaluator(a.analyze)(e, 'full');
  assert.ok(.4 * .6 < .35);
  assert.strictEqual(result.choice, choices[4], 'do not replace the chosen riichi with the 35% plain discard');
  groupPolicyChoices(result.choices);
  assert.strictEqual(result.choice, choices[4]);
  assert.equal(a.calls.length, 2);
});

test('display aggregation never reranks the selected backend action', async () => {
  const e = engine();
  // A synthetic many-to-one translator exercises this contract. Current EMA
  // play masks out red-five aliases; this fixture does not reopen them.
  e.agent_action_from_mortal = action => action === 34 ? 0 : action;
  const a = analyzer([{ action: 2, weights: weights({ 0: .3, 34: .3, 2: .4 }) }]);
  const result = await evaluator(a.analyze)(e, 'full');
  assert.strictEqual(result.choice, choices[2]);
  assert.equal(result.choices[0].weight, .6);
  assert.equal(result.choices[0].tile, '1m');
  groupPolicyChoices(result.choices);
  assert.strictEqual(result.choice, choices[2]);
  assert.equal(a.calls.length, 1);
});

test('riichi not chosen stays unevaluated, not zero or an invented conditional distribution', async () => {
  const e = engine(), a = analyzer([{ action: 2, weights: weights({ 37: .2, 2: .8 }) }]);
  const result = await evaluator(a.analyze)(e, 'full');
  assert.strictEqual(result.choice, choices[2]);
  assert.equal(a.calls.length, 1);
  const group = groupPolicyChoices(result.choices).find(row => row.choices);
  assert.equal(group.weight, .2);
  assert.ok(group.choices.every(choice => choice.conditionalWeight === null));
});

test('zero conditional weights remain scored zero; unrelated extra kans remain unscored', async () => {
  const e = engine(), a = analyzer([
    { action: 37, weights: weights({ 37: 1 }) },
    { action: 1, weights: weights({ 1: 1 }) },
  ]);
  const result = await evaluator(a.analyze)(e, 'full');
  assert.equal(result.choices.find(choice => choice.index === 34).conditionalWeight, 0);
  assert.equal(result.choices.find(choice => choice.index === null).weight, null);
});

test('conditional distributions do not leak into the next advisory request', async () => {
  const e = engine(), a = analyzer([
    { action: 37, weights: weights({ 37: 1 }) },
    { action: 1, weights: weights({ 1: 1 }) },
    { action: 2, weights: weights({ 2: 1 }) },
  ]);
  const evaluate = evaluator(a.analyze);
  await evaluate(e, 'full');
  const result = await evaluate(e, 'full');
  assert.equal(a.calls.length, 3);
  assert.ok(result.choices.filter(choice => choice.kind === 'riichi').every(choice => choice.conditionalWeight === null));
});

test('built-in advisers keep their sampled choice and make no network requests', async () => {
  const e = engine(); let picks = 0;
  e.agent_pick = agent => { assert.equal(agent, 'club'); picks++; return choices[1]; };
  const a = analyzer([]), result = await evaluator(a.analyze)(e, 'club');
  assert.equal(result.kind, 'selection');
  assert.strictEqual(result.choice, choices[1]);
  assert.equal(result.choices[0].weight, 1);
  assert.equal(picks, 1); assert.equal(a.calls.length, 0);
});

test('historical riichi review keeps declaration and conditional weights separate for both tiles', async () => {
  const e = engine(), a = analyzer([
    { action: 37, weights: weights({ 37: .8, 2: .2 }) },
    { action: 1, weights: weights({ 0: .25, 1: .75 }) },
  ]);
  const [result] = await reviewWithStrong(e, [note(choices[3])], a.analyze);
  assert.equal(result.advised_kind, 'riichi'); assert.equal(result.advised_tile, '2m');
  assert.equal(result.agreed, false);
  assert.equal(result.preferred_weight, .8); assert.equal(result.played_weight, .8);
  assert.equal(result.preferred_conditional_weight, .75);
  assert.equal(result.played_conditional_weight, .25);
  assert.equal(a.calls.length, 2);
});

test('historical riichi that was played but not advised does not trigger another inference', async () => {
  const e = engine(), a = analyzer([{ action: 2, weights: weights({ 37: .2, 2: .8 }) }]);
  const [result] = await reviewWithStrong(e, [note(choices[3])], a.analyze);
  assert.equal(result.advised_kind, 'discard'); assert.equal(result.agreed, false);
  assert.equal(result.played_weight, .2); assert.equal(result.played_conditional_weight, null);
  assert.equal(result.preferred_conditional_weight, null); assert.equal(a.calls.length, 1);
});

test('cancelled reviews never ask the conditional question or publish a late result', async () => {
  const e = engine(), owner = new AbortController(); let calls = 0;
  await assert.rejects(reviewWithStrong(e, [note(choices[3])], async () => {
    calls++; owner.abort(); return { action: 37, weights: weights({ 37: 1 }) };
  }, { signal: owner.signal }), { name: 'AbortError' });
  assert.equal(calls, 1);
});

test('invalid conditional weights are rejected without weakening first-stage validation', () => {
  const e = engine(), first = weights({ 37: 1 });
  for (const value of [NaN, Infinity, -.1, 1.1]) {
    assert.throws(() => weightsByChoice(e, choices, first, e.agent_action_from_mortal,
      { weights: weights({ 0: value }), fromMortal: action => e.agent_action_from_mortal(action, true) }), /invalid/);
  }
  assert.throws(() => weightsByChoice(e, choices, first, e.agent_action_from_mortal,
    { weights: weights({ 0: .6, 1: .6 }), fromMortal: () => 34 }), /invalid conditional/);
});
