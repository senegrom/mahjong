import test from 'node:test';
import assert from 'node:assert/strict';
import { GATE, MOVES, average, averagedAnswer, averagedAnswers, hesitates, swapMoves } from '../src/lib/suits.js';
import { policyWeights } from '../src/lib/policy-weights.js';

const range = (from, to) => Array.from({ length: to - from }, (_, at) => from + at);
const mask = open => Array.from({ length: 46 }, (_, move) => open.includes(move));
/** The softmax over the legal moves of the mean of two answers' logits, the
 * renamed one renamed back, as neural/suits.py's averaging player has it. */
function meanOfLogits(plain, renamed, legal) {
  const mean = plain.map((value, move) => (value + renamed[MOVES[move]]) / 2);
  const best = Math.max(...mean.filter((_, move) => legal[move]));
  const raised = mean.map((value, move) => (legal[move] ? Math.exp(value - best) : 0));
  const total = raised.reduce((sum, value) => sum + value, 0);
  return { action: mean.findIndex((value, move) => legal[move] && value === best), weights: raised.map(value => value / total) };
}
const close = (actual, expected, label) => actual.forEach((value, at) => assert.ok(Math.abs(value - expected[at]) < 1e-12,
  `${label}: move ${at} weighs ${value}, not ${expected[at]}`));

test('Mortal\'s moves are renamed as neural/suits.py renames them, and renamed twice they are themselves', () => {
  assert.deepEqual(MOVES, [...range(9, 18), ...range(0, 9), ...range(18, 34), 35, 34, ...range(36, 46)]);
  assert.ok(Object.isFrozen(MOVES));
  assert.deepEqual(MOVES.map(move => MOVES[move]), range(0, 46));
  // 1m and 1p trade places, 9m and 9p, and the red fives 5mr and 5pr; 1s-9s,
  // the honours, 5sr, riichi, the calls, a win, a draw and pass stay.
  for (const [a, b] of [[0, 9], [8, 17], [34, 35]]) assert.deepEqual([MOVES[a], MOVES[b]], [b, a]);
  for (const move of [...range(18, 34), ...range(36, 46)]) assert.equal(MOVES[move], move);
  const values = range(0, 46).map(move => move * 10);
  assert.deepEqual(swapMoves(values).slice(0, 18), [...range(9, 18), ...range(0, 9)].map(move => move * 10));
  assert.deepEqual(swapMoves(swapMoves(values)), values);
  // A mask of the engine's, a Uint8Array, renames as a list.
  const open = Uint8Array.from(mask([0, 8, 20, 34, 37, 45]), Number);
  assert.deepEqual(swapMoves(open), Array.from(mask([9, 17, 20, 35, 37, 45]), Number));
  assert.throws(() => swapMoves(new Float32Array(34)), /wrong number of choices/);
});

test('only an answer whose top weight is below 0.9 hesitates, and an only move never does', () => {
  assert.equal(GATE, 0.9);
  const weights = top => [top, 1 - top, ...Array(44).fill(0)];
  assert.equal(hesitates(weights(0.8999999)), true);
  assert.equal(hesitates(weights(0.9)), false, 'the gate is strict');
  assert.equal(hesitates(weights(0.95)), false);
  assert.equal(hesitates([...Array(45).fill(0), 1]), false);
  // The worker's own weights, from logits: the top weight is one over the
  // sum of each legal move's exp(logit - best), as suits.top_weight has it.
  const legal = mask([3, 4]), logits = Array(46).fill(0);
  logits[3] = Math.log(9.01);
  assert.equal(hesitates(policyWeights(logits, legal).weights), false, '9.01 to one is over 0.9');
  logits[3] = Math.log(8.99);
  assert.equal(hesitates(policyWeights(logits, legal).weights), true);
  assert.throws(() => hesitates(undefined), /no weights/);
});

test('the average of two answers\' weights is the softmax of the mean of their logits, the renamed one renamed back', () => {
  // Hand-made logits over a table of discards of characters, circles and
  // bamboo, a riichi and a pass, the renamed game's answer its own.
  const legal = mask([0, 4, 9, 13, 17, 22, 31, 37, 45]);
  const plain = range(0, 46).map(move => Math.sin(move * 1.7) * 3);
  const renamed = range(0, 46).map(move => Math.cos(move * 0.9) * 2.5);
  // The renamed game's mask is the game's renamed: its answer is asked there.
  const renamedMask = swapMoves(legal);
  const fromPlain = policyWeights(plain, legal), fromRenamed = policyWeights(renamed, renamedMask);
  const averaged = average(fromPlain.weights, fromRenamed.weights, legal), expected = meanOfLogits(plain, renamed, legal);
  assert.equal(averaged.action, expected.action);
  close(averaged.weights, expected.weights, 'the mean of the logits');
  assert.ok(Math.abs(averaged.weights.reduce((sum, weight) => sum + weight, 0) - 1) < 1e-12);
  assert.ok(averaged.weights.every((weight, move) => legal[move] || weight === 0), 'an illegal move weighs nothing');
  // The two answers disagree: the plain one prefers another move than the mean.
  assert.notEqual(fromPlain.action, averaged.action, 'the hand-made answers must disagree');
  // A renamed answer that is the plain one renamed back changes nothing.
  const same = average(fromPlain.weights, swapMoves(fromPlain.weights), legal);
  assert.equal(same.action, fromPlain.action);
  close(same.weights, fromPlain.weights, 'the plain answer');
  assert.throws(() => average(fromPlain.weights, fromRenamed.weights.slice(0, 34), legal), /wrong number of choices/);
  assert.throws(() => average(Array(46).fill(0), Array(46).fill(0), legal), /invalid weights/);
});

test('of moves the mean weighs alike the first is best, as numpy\'s argmax takes it', () => {
  // 1s and 9s (18 and 26) are not renamed: at equal logits in both answers
  // the mean ties them, and the first wins.
  const legal = mask([18, 26]), logits = Array(46).fill(0);
  logits[18] = logits[26] = 1.25;
  const tied = average(policyWeights(logits, legal).weights, policyWeights(logits, swapMoves(legal)).weights, legal);
  assert.deepEqual([tied.action, tied.weights[18], tied.weights[26]], [18, 0.5, 0.5]);
  // 1m (0) leads in the plain answer and 1p (9) by as much in the renamed
  // game, where 1m is called 1p: the mean ties them, and 1m wins.
  const both = mask([0, 9]), plain = Array(46).fill(0), renamed = Array(46).fill(0);
  plain[0] = 2; plain[9] = 1;
  renamed[MOVES[9]] = 2; renamed[MOVES[0]] = 1;
  const averaged = average(policyWeights(plain, both).weights, policyWeights(renamed, swapMoves(both)).weights, both);
  assert.equal(averaged.action, 0);
  assert.ok(Math.abs(averaged.weights[0] - averaged.weights[9]) < 1e-15);
  assert.equal(meanOfLogits(plain, renamed, both).action, 0);
});

/** An asker that answers each position from `answers`, by its planes, and
 * notes every request. */
function asker(answers) {
  const requests = [];
  return { requests, ask: async positions => {
    requests.push(positions.map(({ planes, mask: open }) => ({ planes, mask: Array.from(open) })));
    return positions.map(({ planes }) => answers.get(planes));
  } };
}

test('a question Trained AI hesitates over is asked again of the renamed game, its mask renamed, and answered by the mean', async () => {
  const legal = mask([0, 9, 45]), sure = mask([2, 45]);
  const plain = { action: 0, weights: weightsOf({ 0: 0.6, 9: 0.3, 45: 0.1 }), value: 0.25, hands: 'plain belief' };
  const renamed = { action: 0, weights: weightsOf({ 0: 0.7, 9: 0.2, 45: 0.1 }), value: -1, hands: 'renamed belief' };
  const confident = { action: 2, weights: weightsOf({ 2: 0.95, 45: 0.05 }) };
  const read = [];
  const positions = [
    { planes: 'first', mask: legal, renamed: () => { read.push('first'); return 'first renamed'; } },
    { planes: 'second', mask: sure, renamed: () => { read.push('second'); return 'second renamed'; } },
  ];
  const { requests, ask } = asker(new Map([['first', plain], ['second', confident], ['first renamed', renamed]]));
  const [first, second] = await averagedAnswers(ask, positions);
  // The plain request as it came, then the hesitant position alone, renamed.
  assert.deepEqual(requests, [[{ planes: 'first', mask: Array.from(legal) }, { planes: 'second', mask: Array.from(sure) }],
    [{ planes: 'first renamed', mask: swapMoves(legal) }]]);
  assert.deepEqual(read, ['first'], 'a confident position\'s renamed planes are never read');
  assert.equal(second, confident);
  // 1m at 0.6 and 0.2 (the renamed game's 1p), 1p at 0.3 and 0.7: the mean
  // prefers 1p, and the value and belief stay the plain answer's.
  const expected = average(plain.weights, renamed.weights, legal);
  assert.deepEqual(first, { ...plain, action: 9, weights: expected.weights });
  assert.equal(expected.action, 9);
  // Asked one at a time, the same.
  const single = asker(new Map([['first', plain], ['first renamed', renamed]]));
  const calls = [];
  const answer = await averagedAnswer(async (planes, open, signal) => { calls.push([planes, Array.from(open), signal]); return (await single.ask([{ planes, mask: open }]))[0]; },
    positions[0], 'signal');
  assert.deepEqual(calls, [['first', Array.from(legal), 'signal'], ['first renamed', swapMoves(legal), 'signal']]);
  assert.deepEqual(answer, first);
});

test('a sure question is asked once, and a question stopped before the renamed game is asked reads nothing more', async () => {
  const legal = mask([0, 45]);
  const { requests, ask } = asker(new Map([['plain', { action: 0, weights: weightsOf({ 0: 0.9, 45: 0.1 }) }]]));
  const answer = await averagedAnswer((planes, open) => ask([{ planes, mask: open }]).then(([one]) => one),
    { planes: 'plain', mask: legal, renamed: () => assert.fail('a sure question asks once') });
  assert.equal(answer.action, 0);
  assert.equal(requests.length, 1);
  assert.deepEqual(await averagedAnswers(ask, []), [], 'nothing asked, nothing answered');
  // Stopped while the plain question was answered: the engine the renamed
  // planes come from may be gone, so they are not read.
  const stop = new AbortController();
  const hesitant = { action: 0, weights: weightsOf({ 0: 0.5, 45: 0.5 }) };
  await assert.rejects(averagedAnswer(async () => { stop.abort(); return hesitant; },
    { planes: 'plain', mask: legal, renamed: () => assert.fail('read after the stop') }, stop.signal), { name: 'AbortError' });
});

function weightsOf(entries) {
  return Array.from({ length: 46 }, (_, move) => entries[move] ?? 0);
}
