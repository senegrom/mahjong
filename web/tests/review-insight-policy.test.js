import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis, review_facts } from '../src/wasm/riichi.js';
import { MORTAL_REACH, weightsByChoice } from '../src/lib/action-weights.js';
import { Asking, scanMatch } from '../src/lib/game-review.js';
import { PASS, saidLines, sameMove } from '../src/lib/review-insight.js';
import { anyGate, handTables, questionOf, reviewInsight, trainedAsk } from '../src/lib/review-insight-policy.js';
import { recordedAsk, recordedQuestions } from './fixtures/recorded-insight.js';
import { eachDecision, reviewCheckGame } from './fixtures/review-check-game.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

const entry = recordedQuestions(Game).find(question => question.id === 'e3-5');
const turn = () => new Promise(resolve => setImmediate(resolve));
async function until(ready, what) {
  for (let count = 0; count < 1000 && !ready(); count++) await turn();
  assert.ok(ready(), what);
}

/** Trained AI's weights over Mortal's forty-six moves for a table, from a
 * recorded answer: each of the engine's moves given its recorded weight, and
 * a move the record does not keep none. */
function mortalWeights(table, moves) {
  const analysis = new PhysicalAnalysis(table);
  try {
    const choices = analysis.agent_choices();
    const planes = analysis.agent_observation_mortal(), mask = analysis.agent_mask_mortal();
    const weights = Array.from(mask, (open, action) => {
      if (!open) return 0;
      const choice = action === MORTAL_REACH ? { kind: 'riichi', tile: null }
        : choices.find(listed => listed.index === analysis.agent_action_from_mortal(action, false));
      return moves.find(move => choice && sameMove(move, choice))?.weight ?? 0;
    });
    return { planes, mask, weights };
  } finally { analysis.free(); }
}
/** The moves that carry weight, in tile order. */
const weighed = moves => moves.filter(move => move.weight > 0).sort((a, b) => (a.tile ?? '').localeCompare(b.tile ?? ''));
const planesKey = (planes, mask) => `${Array.from(mask).join('')}:${Array.from(planes).flatMap((value, at) => (value ? [`${at}=${value}`] : [])).join(',')}`;

/** The policy worker as the recorded run answered East 3 #5, keyed by the
 * planes the engine writes for each table the record answers, so that every
 * table goes through the page's own encoding and back. `sizes` notes each
 * batch; `onBatch(sent)` hears each before it is answered, with how many
 * came before it. */
function recordedNetwork({ onBatch = () => {} } = {}) {
  const byPlanes = new Map();
  for (const [key, answer] of recordedAsk(entry).answers) {
    const { planes, mask, weights } = mortalWeights(JSON.parse(key), answer.moves);
    byPlanes.set(planesKey(planes, mask), weights);
  }
  const sizes = [];
  const analyzeBatch = async (positions, signal) => {
    onBatch(sizes.length);
    sizes.push(positions.length);
    await turn();
    if (signal?.aborted) throw new DOMException('Review changed', 'AbortError');
    return positions.map(({ planes, mask }) => {
      const weights = byPlanes.get(planesKey(planes, mask));
      if (!weights) throw new Error('A table the recorded run never asked');
      return { action: weights.indexOf(Math.max(...weights)), weights };
    });
  };
  return { analyzeBatch, sizes };
}

/** East 3 #5 as the review keeps it: the player's discard and Trained AI's,
 * and every move's weight read into the listed choices, as the review reads
 * the network's answer (weightsByChoice). */
function reviewedDecision() {
  const analysis = new PhysicalAnalysis(entry.position);
  try {
    const { weights } = mortalWeights(entry.position, entry.weights);
    return { played_kind: 'discard', played_tile: entry.played.tile, advised_kind: 'discard', advised_tile: entry.preferred.tile,
      weighted: weightsByChoice(analysis, analysis.agent_choices(), weights, action => analysis.agent_action_from_mortal(action, false)) };
  } finally { analysis.free(); }
}

const insightOf = (network, asking = new Asking(), notes = new Asking()) => ({ asking, notes,
  insight: reviewInsight({ PhysicalAnalysis, review_facts, analyzeBatch: network.analyzeBatch, asking, notes }) });

test('a hand\'s tables are the positions the engine writes at the hand\'s end, read from a replay of their own', () => {
  const snapshot = reviewCheckGame(Game), hands = scanMatch(Game, snapshot), written = [];
  eachDecision(Game, snapshot, ({ hand, position }) => { written[hand.index] = [...(written[hand.index] ?? []), position]; });
  for (const hand of hands) assert.deepEqual(handTables(Game, snapshot, hand), written[hand.index], hand.name);
  assert.equal(written.flat().length, 139);
});

test('Trained AI is asked through analyzeBatch in passes of at most 32, and its answers read back into the engine\'s moves', async () => {
  const network = recordedNetwork(), { insight } = insightOf(network);
  const question = questionOf(reviewedDecision(), entry.position);
  // The review's weights come back as moves, the same as the recorded run's.
  assert.deepEqual(weighed(question.weights), weighed(entry.weights));
  const why = await insight.why(question);
  assert.deepEqual(network.sizes, [2], 'the unedited table and the furiten taken away');
  const search = await insight.search(question);
  assert.deepEqual(network.sizes, [2, PASS, 7]);
  assert.deepEqual([...saidLines(why), ...saidLines(search.said)], [
    'Partly: You had already thrown a 7 characters, so a wait on it would be furiten. Trained AI puts 93.0% on tiles you had already thrown.',
    'One change and it throws your west wind:',
    '- A faster hand (7 of 7 tried): if your white dragon had been a 2 circles (your hand a step closer to waiting and one safe tile fewer): your west wind 99.6%, the 7 characters 0.2%.',
    '- A hand worth more (4 of 14 tried): if the dora indicator had been the 8 bamboo, not the 9 bamboo, making the 9 bamboo dora (2 more dora in your hand): your west wind 94.3%, the 7 characters 2.0%.',
    '- What is safe (2 of 9 tried): if South had declared riichi with their last discard, the west wind (a second riichi and the 7 characters no longer safe): your west wind 70.8%, the 7 characters <0.1%.',
    'Not moved by the score (±16,000) or the round.',
  ]);
});

test('a search shuts the review\'s gate while it asks, and opens it however it ends', async () => {
  let shutWhileAsking = null;
  const network = recordedNetwork({ onBatch: () => { shutWhileAsking ??= review(); } });
  const { insight, asking, notes } = insightOf(network);
  // The review of the game waits at the gate of the notes and of the search.
  const review = anyGate(notes, asking);
  assert.equal(review(), null);
  const progress = [];
  await insight.search(questionOf(reviewedDecision(), entry.position), { onProgress: step => progress.push(step) });
  assert.ok(shutWhileAsking instanceof Promise, 'shut while the search asked');
  assert.equal(review(), null, 'open once it has ended');
  assert.deepEqual(progress.at(-1), { asked: 39, planned: 39 });
  // A search that fails ends its spell too.
  const failing = insightOf({ analyzeBatch: async () => { throw new Error('The trained opponent did not answer in time'); } });
  await assert.rejects(failing.insight.search(questionOf(reviewedDecision(), entry.position)), /did not answer in time/);
  assert.equal(anyGate(failing.notes, failing.asking)(), null);
});

test('a search waits while the hand\'s notes ask, and goes on where it was', async () => {
  let stopNotes = null;
  const network = recordedNetwork({ onBatch: sent => { if (sent === 0) stopNotes = notes.begin(); } });
  const { insight, notes } = insightOf(network);
  // The notes begin to ask while the first pass is under way: the second
  // waits until they stop, then goes on.
  const search = insight.search(questionOf(reviewedDecision(), entry.position));
  await until(() => stopNotes, 'the first pass is sent');
  for (let count = 0; count < 20; count++) await turn();
  assert.deepEqual(network.sizes, [PASS], 'no pass while the notes ask');
  stopNotes();
  const found = await search;
  assert.deepEqual(network.sizes, [PASS, 7]);
  assert.equal(found.status, 'searched');
});

test('a search stopped while it waits for the notes stops at once, and ends its spell', async () => {
  const network = recordedNetwork(), { insight, notes, asking } = insightOf(network), stop = new AbortController();
  const stopNotes = notes.begin();
  const search = insight.search(questionOf(reviewedDecision(), entry.position), { signal: stop.signal });
  await until(() => asking.gate(), 'the search holds its spell');
  stop.abort();
  await assert.rejects(search, { name: 'AbortError' });
  assert.deepEqual([network.sizes, asking.gate()], [[], null]);
  stopNotes();
});

test('a table the rules refuse is answered as refused, and never sent', async () => {
  // East 3 #5 with a fifth white dragon in West's river.
  const fifth = structuredClone(entry.position);
  fifth.players[2].discards.push(...['5z', '5z', '5z', '5z'].map((tile, at) => ({ tile, order: 100 + at, drawn: false, riichi: false, claimed: false })));
  const network = recordedNetwork();
  const answers = await trainedAsk({ PhysicalAnalysis, analyzeBatch: network.analyzeBatch })([fifth, entry.position]);
  assert.deepEqual(network.sizes, [1]);
  assert.match(answers[0].refused, /More than four copies/);
  assert.deepEqual(weighed(answers[1].moves), weighed(entry.weights));
});

test('a search needs a spell of asking apart from the notes\'', () => {
  const shared = new Asking();
  assert.throws(() => reviewInsight({ PhysicalAnalysis, review_facts, analyzeBatch: async () => [], asking: shared, notes: shared }), /apart from the notes/);
});
