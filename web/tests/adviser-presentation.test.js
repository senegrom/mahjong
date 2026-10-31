import test from 'node:test';
import assert from 'node:assert/strict';
import { weightsByChoice, groupPolicyChoices, MORTAL_REACH } from '../src/lib/action-weights.js';
import { reviewWithStrong } from '../src/lib/review-policy.js';
import { MOVES, average, averagedAnswer, swapMoves } from '../src/lib/suits.js';
import { loadModule } from './fixtures/load-module.js';

const weights = entries => Object.freeze(Array.from({ length: 46 }, (_, index) => entries[index] ?? 0));
/** The renamed game's answer that is `answer` renamed, as a network that
 * knew the symmetry exactly would give it: the mean is then `answer`. */
const mirrored = answer => ({ action: MOVES[answer.action], weights: swapMoves(answer.weights) });
const close = (actual, expected, label) => assert.ok(Math.abs(actual - expected) < 1e-12, `${label}: ${actual}, not ${expected}`);
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
  // The renamed game's planes, other than the game's own.
  const renamed = new Float32Array([11]), afterRenamed = new Float32Array([12]);
  const mask = Array.from({ length: 46 }, (_, index) => [0, 1, 2, 37].includes(index));
  const afterMask = Array.from({ length: 46 }, (_, index) => [0, 1].includes(index));
  const from = (action, reached) => reached ? ([0, 1].includes(action) ? action + 34 : -1)
    : ([0, 1, 2].includes(action) ? action : -1);
  return {
    planes, after, renamed, afterRenamed, mask, afterMask,
    agent_choices: () => choices,
    agent_observation_mortal: () => planes, agent_mask_mortal: () => mask,
    agent_observation_after_reach: () => after, agent_mask_after_reach: () => afterMask,
    agent_observation_mortal_renamed: () => renamed, agent_observation_after_reach_renamed: () => afterRenamed,
    agent_action_from_mortal: from,
    mortal_action_of: index => [34, 35].includes(index) ? MORTAL_REACH : index,
    review_choices: () => choices,
    review_observation_mortal: () => planes, review_mask_mortal: () => mask,
    review_observation_after_reach: () => after, review_mask_after_reach: () => afterMask,
    review_observation_mortal_renamed: () => renamed, review_observation_after_reach_renamed: () => afterRenamed,
    review_action_from_mortal: (_index, action, reached) => from(action, reached),
  };
}
/** A declaration and its tile, each hesitant, with the renamed game's
 * answers, in its own moves, where 3m is called 3p (move 11) and 1m and 2m
 * are 1p and 2p: the mean declares again, and throws 2m again, at weights of
 * its own. */
const declared = {
  first: { action: 37, weights: weights({ 37: .8, 2: .2 }), value: -1.25 },
  firstRenamed: { action: 37, weights: weights({ 37: .7, [MOVES[2]]: .3 }), value: 5 },
  second: { action: 1, weights: weights({ 0: .25, 1: .75 }), value: 99 },
  secondRenamed: { action: MOVES[1], weights: weights({ [MOVES[0]]: .35, [MOVES[1]]: .65 }), value: 98 },
};
const sequence = ({ first, firstRenamed, second, secondRenamed }) => [first, firstRenamed, second, secondRenamed];
function analyzer(responses) {
  const calls = [];
  return { calls, async analyze(...args) {
    calls.push(args);
    assert.ok(calls.length <= responses.length, 'display must not request extra inference');
    return responses[calls.length - 1];
  } };
}
const evaluator = analyzePolicy => loadModule('agents.js', {
  analyzePolicy, averagedAnswer, weightsByChoice, MORTAL_REACH,
  readValue: value => value ?? null, readBeliefs: () => null,
}, ['evaluateAgent']).evaluateAgent;
const note = choice => ({ turn: 1, played: choice.label, played_kind: choice.kind, played_tile: choice.tile });

test('one declaration row has separate conditional discards; neither distribution is multiplied', async () => {
  const e = engine(), signal = new AbortController().signal;
  const a = analyzer(sequence(declared));
  const result = await evaluator(a.analyze)(e, 'full', signal);
  assert.strictEqual(result.choice, choices[4], 'second-stage action, not first matching riichi row');
  assert.equal(result.value, -1.25, 'keep the original position value, not the post-declaration value');
  // Each question as the engine writes it, then, Trained AI hesitating, the
  // renamed game's, its moves renamed.
  assert.equal(a.calls.length, 4);
  assert.strictEqual(a.calls[0][0], e.planes); assert.strictEqual(a.calls[0][1], e.mask);
  assert.strictEqual(a.calls[1][0], e.renamed); assert.deepEqual(a.calls[1][1], swapMoves(e.mask));
  assert.strictEqual(a.calls[2][0], e.after); assert.strictEqual(a.calls[2][1], e.afterMask);
  assert.strictEqual(a.calls[3][0], e.afterRenamed); assert.deepEqual(a.calls[3][1], swapMoves(e.afterMask));
  assert.ok(a.calls.every(call => call[2] === signal));
  // The weights shown are each question's mean.
  const declaration = average(declared.first.weights, declared.firstRenamed.weights, e.mask).weights;
  const tile = average(declared.second.weights, declared.secondRenamed.weights, e.afterMask).weights;
  const before = JSON.stringify(result.choices), rows = groupPolicyChoices(result.choices);
  const groups = rows.filter(row => row.choices);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].weight, declaration[37]);
  assert.notEqual(declaration[37], .8, 'the renamed game moved the declaration\'s weight');
  assert.deepEqual(groups[0].choices.map(choice => choice.conditionalWeight), [tile[1], tile[0]]);
  close(rows.reduce((sum, row) => sum + (row.choice?.weight ?? row.weight ?? 0), 0), 1, 'the first stage');
  close(groups[0].choices.reduce((sum, choice) => sum + choice.conditionalWeight, 0), 1, 'the tiles');
  assert.equal(JSON.stringify(result.choices), before, 'sorting display rows must not mutate command choices');
  assert.ok(groups[0].choices.every(choice => result.choices.includes(choice)), 'confirmation receives original choice objects');
});

test('a lower joint probability cannot override the policy\'s two-stage choice', async () => {
  const first = { action: 37, weights: weights({ 37: .4, 2: .35, 1: .25 }) };
  const second = { action: 1, weights: weights({ 0: .4, 1: .6 }) };
  // Ranked by joint probability, the riichi and its tile would lose to the
  // likeliest plain discard.
  const plain = Math.max(...first.weights.filter((_, action) => action !== MORTAL_REACH));
  assert.ok(first.weights[MORTAL_REACH] * second.weights[second.action] < plain);
  const e = engine(), a = analyzer([first, mirrored(first), second, mirrored(second)]);
  const result = await evaluator(a.analyze)(e, 'full');
  assert.strictEqual(result.choice, choices[4], 'do not replace the chosen riichi with the 35% plain discard');
  groupPolicyChoices(result.choices);
  assert.strictEqual(result.choice, choices[4]);
  assert.equal(a.calls.length, 4);
});

test('riichi not chosen stays unevaluated, not zero or an invented conditional distribution', async () => {
  const first = { action: 2, weights: weights({ 37: .2, 2: .8 }) };
  const e = engine(), a = analyzer([first, mirrored(first)]);
  const result = await evaluator(a.analyze)(e, 'full');
  assert.strictEqual(result.choice, choices[2]);
  assert.equal(a.calls.length, 2, 'the declaration not chosen, its tile is not asked');
  const group = groupPolicyChoices(result.choices).find(row => row.choices);
  close(group.weight, .2, 'a mirrored renamed answer leaves the weight as it was');
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
  const e = engine(), a = analyzer(sequence(declared));
  const [result] = await reviewWithStrong(e, [note(choices[3])], a.analyze);
  assert.equal(result.advised_kind, 'riichi'); assert.equal(result.advised_tile, '2m');
  assert.equal(result.agreed, false);
  // Judged as the adviser shows it: each question's mean, asked the same way.
  const declaration = average(declared.first.weights, declared.firstRenamed.weights, e.mask).weights;
  const tile = average(declared.second.weights, declared.secondRenamed.weights, e.afterMask).weights;
  assert.equal(result.preferred_weight, declaration[37]); assert.equal(result.played_weight, declaration[37]);
  assert.equal(result.preferred_conditional_weight, tile[1]);
  assert.equal(result.played_conditional_weight, tile[0]);
  assert.deepEqual(a.calls.map(([planes, mask]) => [planes, Array.from(mask)]),
    [[e.planes, e.mask], [e.renamed, swapMoves(e.mask)], [e.after, e.afterMask], [e.afterRenamed, swapMoves(e.afterMask)]]);
});

test('historical riichi that was played but not advised does not trigger another inference', async () => {
  const e = engine(), a = analyzer([{ action: 2, weights: weights({ 37: .05, 2: .95 }) }]);
  const [result] = await reviewWithStrong(e, [note(choices[3])], a.analyze);
  assert.equal(result.advised_kind, 'discard'); assert.equal(result.agreed, false);
  assert.equal(result.played_weight, .05); assert.equal(result.played_conditional_weight, null);
  assert.equal(result.preferred_conditional_weight, null); assert.equal(a.calls.length, 1, 'a sure answer is not asked again either');
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
});
