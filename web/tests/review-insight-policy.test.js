import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis, review_facts } from '../src/wasm/riichi.js';
import { MORTAL_REACH, weightsByChoice } from '../src/lib/action-weights.js';
import { Asking, scanMatch } from '../src/lib/game-review.js';
import { PASS, movesOf, sameMove } from '../src/lib/review-insight.js';
import { anyGate, countedBatches, handTables, passFor, questionOf, reviewInsight, trainedAsk } from '../src/lib/review-insight-policy.js';
import { saidLines } from './fixtures/insight-panel.js';
import { recordedAsk, recordedQuestions } from './fixtures/recorded-insight.js';
import { eachDecision, reviewCheckGame } from './fixtures/review-check-game.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

const questions = recordedQuestions(Game, PhysicalAnalysis), entry = questions.find(question => question.id === 'e3-5');
/** East 3 #5's search: the unedited table and its 35 changes, in passes of
 * the review's batch. */
const SEARCH = [PASS, PASS, PASS, PASS, 4];
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

/** The policy worker as the recorded run answered East 3 #5, or the questions
 * named (`asked`), keyed by the planes the engine writes for each table the
 * record answers, so that every table goes through the page's own encoding
 * and back. `sizes` notes each batch; `onBatch(sent)` hears each before it is
 * answered, with how many came before it. A pass is never cancelled: the
 * worker runs one it has begun to its end. */
function recordedNetwork({ onBatch = () => {}, asked = [entry] } = {}) {
  const byPlanes = new Map();
  for (const question of asked) {
    for (const [key, answer] of recordedAsk(question).answers) {
      const { planes, mask, weights } = mortalWeights(JSON.parse(key), answer.moves);
      byPlanes.set(planesKey(planes, mask), weights);
    }
  }
  const sizes = [];
  const analyzeBatch = async positions => {
    onBatch(sizes.length);
    sizes.push(positions.length);
    await turn();
    return positions.map(({ planes, mask }) => {
      const weights = byPlanes.get(planesKey(planes, mask));
      if (!weights) throw new Error('A table the recorded run never asked');
      return { action: weights.indexOf(Math.max(...weights)), weights };
    });
  };
  return { analyzeBatch, sizes };
}

/** East 3 #5, or another recorded question, as the review keeps it: the
 * player's discard and Trained AI's, and every move's weight read into the
 * listed choices, as the review reads the network's answer (weightsByChoice). */
function reviewedDecision(question = entry) {
  const analysis = new PhysicalAnalysis(question.position);
  try {
    const { weights } = mortalWeights(question.position, question.weights);
    return { played_kind: 'discard', played_tile: question.played.tile, advised_kind: 'discard', advised_tile: question.preferred.tile,
      weighted: weightsByChoice(analysis, analysis.agent_choices(), weights, action => analysis.agent_action_from_mortal(action, false)) };
  } finally { analysis.free(); }
}

/** The reasoning asked of `network`, with the spells of asking, of the notes
 * and of the worker's batches it keeps to. */
const insightOf = (network, asking = new Asking(), notes = new Asking(), batches = new Asking()) => ({ asking, notes, batches,
  insight: reviewInsight({ PhysicalAnalysis, review_facts, analyzeBatch: network.analyzeBatch, asking, notes, batches }) });

test('a hand\'s tables are the positions the engine writes at the hand\'s end, read from a replay of their own', () => {
  const snapshot = reviewCheckGame(Game), hands = scanMatch(Game, snapshot), written = [];
  eachDecision(Game, snapshot, ({ hand, position }) => { written[hand.index] = [...(written[hand.index] ?? []), position]; });
  for (const hand of hands) assert.deepEqual(handTables(Game, snapshot, hand), written[hand.index], hand.name);
  assert.equal(written.flat().length, 139);
});

test('Trained AI is asked through analyzeBatch in passes of the review\'s batch, and its answers read back into the engine\'s moves', async () => {
  const network = recordedNetwork(), { insight } = insightOf(network);
  const question = questionOf(reviewedDecision(), entry.position);
  // The review's weights come back as moves, the same as the recorded run's.
  assert.deepEqual(weighed(question.weights), weighed(entry.weights));
  const why = await insight.why(question);
  assert.deepEqual(network.sizes, [2], 'the unedited table and the furiten taken away');
  const search = await insight.search(question);
  assert.deepEqual(network.sizes, [2, ...SEARCH]);
  assert.deepEqual([...saidLines(why), ...saidLines(search.said)], [
    'Partly: You had already thrown a 7 characters, so a wait on it would be furiten. Trained AI puts 93.0% on tiles you had already thrown.',
    'One change and it throws your west wind:',
    '- A faster hand (7 of 7 tried): if your white dragon had been a 2 circles (your hand a step closer to waiting and one safe tile fewer): your west wind 99.6%, the 7 characters 0.2%.',
    '- A hand worth more (4 of 14 tried): if the dora indicator had been the 8 bamboo, not the 9 bamboo, making the 9 bamboo dora (2 more dora in your hand): your west wind 94.3%, the 7 characters 2.0%.',
    '- What is safe (2 of 8 tried): if South had declared riichi with their last discard, the west wind (a second riichi and the 7 characters no longer safe): your west wind 70.8%, the 7 characters <0.1%.',
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
  assert.deepEqual(progress.at(-1), { asked: 36, planned: 36 });
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
  assert.deepEqual(network.sizes, SEARCH);
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

test('a search needs spells of asking apart from the notes\' and from the worker\'s batches', () => {
  const shared = new Asking(), own = new Asking(), ask = { PhysicalAnalysis, review_facts, analyzeBatch: async () => [] };
  assert.throws(() => reviewInsight({ ...ask, asking: shared, notes: shared, batches: own }), /apart from the notes/);
  for (const batches of [undefined, shared]) {
    assert.throws(() => reviewInsight({ ...ask, asking: shared, notes: own, batches }), /batches need spells of their own/);
  }
});

/** The policy worker as it runs requests, as policy.worker.js runs them: one
 * at a time, in turn, each to its end once begun, whatever is cancelled, and
 * one still queued dropped when its signal stops it. `answer()` lets the
 * request under way finish, answered as the recorded run answered the
 * questions named (`asked`), and `fail(error)` fails it, as a worker reset
 * fails every request. `sent` counts the requests it was given, and
 * `deepest` the most it held at once. */
function workerOfPasses({ asked } = {}) {
  const recorded = recordedNetwork({ asked }), waiting = [];
  let deepest = 0;
  return {
    get sent() { return recorded.sizes.length + waiting.length; },
    get deepest() { return deepest; },
    analyzeBatch: (positions, signal) => new Promise((resolve, reject) => {
      const request = { positions, resolve, reject };
      waiting.push(request);
      deepest = Math.max(deepest, waiting.length);
      signal?.addEventListener('abort', () => {
        if (waiting.indexOf(request) < 1) return;
        waiting.splice(waiting.indexOf(request), 1);
        reject(new DOMException('Match changed', 'AbortError'));
      }, { once: true });
    }),
    answer: () => { const pass = waiting.shift(); recorded.analyzeBatch(pass.positions).then(pass.resolve, pass.reject); },
    fail: error => waiting.shift().reject(error),
  };
}

test('the reasoning\'s passes wait their turn in the page, one with the worker at a time, and a question stopped while it waits sends nothing', async () => {
  // The three recorded decisions whose Why lines ask Trained AI, opened one
  // after another, each Why line stopped as the next decision opens, on a
  // device slow enough that the first pass is still with the worker. Passes
  // sent as each Why line asked would wait in the worker's queue, where
  // their deadlines run, and run however stopped, being sent without a
  // signal: the stopped ones then kept the last one waiting.
  const probing = ['e3-5', 's3-4', 'e2b-14'].map(id => questions.find(question => question.id === id));
  const worker = workerOfPasses({ asked: probing }), { insight } = insightOf(worker), stops = [], lines = [];
  for (const question of probing) {
    stops.at(-1)?.abort();
    stops.push(new AbortController());
    lines.push(insight.why(questionOf(reviewedDecision(question), question.position), { signal: stops.at(-1).signal })
      .then(why => saidLines(why), error => error.name));
    for (let count = 0; count < 20; count++) await turn();
  }
  assert.equal(worker.sent, 1, 'the first Why line\'s pass, and no other');
  worker.answer();
  await until(() => worker.sent === 2, 'the last Why line sends its pass once the first is answered');
  worker.answer();
  assert.deepEqual(await Promise.all(lines), ['AbortError', 'AbortError', [
    'Why: You had already thrown an 8 bamboo, so a wait on it would be furiten. Trained AI puts 99.2% on tiles you had already thrown. If your second and ninth discards had each been a 1 characters, not the 8 bamboo: the 2 bamboo 66.4%, your 6 bamboo <0.1%, the 8 bamboo <0.1%.',
  ]]);
  assert.deepEqual([worker.sent, worker.deepest], [2, 1]);
});

test('a pass waits for the review\'s batch already with the worker, so that what is sent next waits behind one request at most', async () => {
  // The review's batch is with the worker when the search is pressed. The
  // search's first pass, sent behind the batch, would keep whatever came
  // next behind both: the notes' first question behind sixteen positions.
  const worker = workerOfPasses(), batches = new Asking(), { insight, notes, asking } = insightOf(worker, new Asking(), new Asking(), batches);
  const reviewBatch = countedBatches(worker.analyzeBatch, batches), review = anyGate(notes, asking);
  const analysis = new PhysicalAnalysis(entry.position);
  const table = { planes: analysis.agent_observation_mortal(), mask: analysis.agent_mask_mortal() };
  analysis.free();
  const reviewed = reviewBatch([table]);
  const search = insight.search(questionOf(reviewedDecision(), entry.position));
  for (let count = 0; count < 20; count++) await turn();
  assert.deepEqual([worker.sent, review() instanceof Promise], [1, true], 'the search waits for the batch, and the review for the search');
  worker.answer();
  assert.equal((await reviewed).length, 1);
  for (let pass = 0; pass < SEARCH.length; pass++) {
    await until(() => worker.sent === pass + 2, 'the next pass is with the worker');
    worker.answer();
  }
  assert.equal((await search).status, 'searched');
  assert.deepEqual([worker.sent, worker.deepest, review(), batches.gate()], [SEARCH.length + 1, 1, null, null]);
});

test('a search stopped while its pass is with the worker holds the review back until the worker has answered that pass', async () => {
  // The worker runs the pass to its end whatever is cancelled, and a batch
  // the review sent meanwhile would wait behind it, its deadline running.
  const worker = workerOfPasses(), { insight, notes, asking } = insightOf(worker), review = anyGate(notes, asking), stop = new AbortController();
  const search = insight.search(questionOf(reviewedDecision(), entry.position), { signal: stop.signal });
  await until(() => worker.sent === 1, 'the first pass is with the worker');
  stop.abort();
  await assert.rejects(search, { name: 'AbortError' });
  assert.ok(review() instanceof Promise, 'the review waits while the pass runs');
  worker.answer();
  await until(() => review() === null, 'the review goes on once the pass is answered');
  assert.equal(worker.sent, 1, 'nothing more was asked');
  // A pass that fails, as every request fails when the worker is reset, lets
  // the review go on too, and nobody is left to hear the failure.
  const failing = workerOfPasses(), second = insightOf(failing), stopped = new AbortController();
  const searching = second.insight.search(questionOf(reviewedDecision(), entry.position), { signal: stopped.signal });
  await until(() => failing.sent === 1, 'the first pass is with the worker');
  stopped.abort();
  await assert.rejects(searching, { name: 'AbortError' });
  assert.ok(anyGate(second.notes, second.asking)() instanceof Promise);
  failing.fail(new Error('The trained opponent did not answer in time'));
  await until(() => anyGate(second.notes, second.asking)() === null, 'the review goes on once the pass has failed');
});

test('a pass is the review\'s batch, or fewer where a question sent behind a whole batch would run out of its 20 seconds', async () => {
  // At the review's measured time a position: none measured yet, desktop
  // Chromium's 157 ms, the slow threshold, and slower still.
  assert.deepEqual([null, 157, 500, 1875, 2000, 2500, 7500, 20000].map(passFor), [PASS, PASS, PASS, PASS, 7, 6, 2, 1]);
  const network = recordedNetwork(), asking = new Asking(), notes = new Asking(), batches = new Asking();
  const insight = reviewInsight({ PhysicalAnalysis, review_facts, analyzeBatch: network.analyzeBatch, asking, notes, batches, positionMs: () => 2500 });
  const found = await insight.search(questionOf(reviewedDecision(), entry.position));
  assert.deepEqual([found.status, network.sizes], ['searched', [6, 6, 6, 6, 6, 6]]);
});

test('a table where the seat may both add to its pon and declare a quad is read back as the review reads it', async () => {
  // Mortal has one kan, which the engine gives to the first kan open, so the
  // review lists the added kan at no weight. The answer read back lists it
  // too, or the unedited table would never match the review's.
  const { position, weights } = JSON.parse(readFileSync(new URL('./fixtures/review-insight-dual-kan.json', import.meta.url), 'utf8'));
  const analysis = new PhysicalAnalysis(position);
  let review;
  try {
    review = movesOf(weightsByChoice(analysis, analysis.agent_choices(), weights, action => analysis.agent_action_from_mortal(action, false)));
  } finally { analysis.free(); }
  const [answer] = await trainedAsk({ PhysicalAnalysis, analyzeBatch: async positions => positions.map(() => ({ weights })) })([position]);
  assert.deepEqual(answer.moves, review);
  assert.deepEqual(review.filter(move => /kan$/.test(move.kind)).map(move => [move.kind, move.tile, move.weight > 0.99]),
    [['concealed-kan', '1s', true], ['extended-kan', '9m', false]]);
});
