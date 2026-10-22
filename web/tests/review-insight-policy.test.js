import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis, review_facts } from '../src/wasm/riichi.js';
import { MORTAL_REACH, weightsByChoice } from '../src/lib/action-weights.js';
import { Asking, scanMatch } from '../src/lib/game-review.js';
import { PASS, movesOf, sameMove, tablesFirstAsked } from '../src/lib/review-insight.js';
import { MatchInsight, anyGate, countedBatches, handTables, passFor, questionOf, reasonedAbout, reviewInsight, trainedAsk }
  from '../src/lib/review-insight-policy.js';
import { saidLines } from './fixtures/insight-panel.js';
import { recordedAsk, recordedQuestions } from './fixtures/recorded-insight.js';
import { eachDecision, reviewCheckGame } from './fixtures/review-check-game.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

const questions = recordedQuestions(Game, PhysicalAnalysis), entry = questions.find(question => question.id === 'e3-5');
/** East 3 #5's search: the unedited table and its 35 changes, in passes of
 * the review's batch. */
const SEARCH = [PASS, PASS, PASS, PASS, 4];
/** East 3 #5's Why line, and what its search finds, as the panel reads them
 * (saidLines), on the recorded run's answers. */
const WHY_SAID = 'Partly: You had already thrown a 7 characters, so a wait on it would be furiten. Trained AI puts 93.0% on tiles you had already thrown.';
const SEARCH_SAID = [
  'One change and it throws your west wind:',
  '- A faster hand (7 of 7 tried): if your white dragon had been a 2 circles (your hand a step closer to waiting and one safe tile fewer): your west wind 99.6%, the 7 characters 0.2%.',
  '- A hand worth more (4 of 14 tried): if the dora indicator had been the 8 bamboo, not the 9 bamboo, making the 9 bamboo dora (2 more dora in your hand): your west wind 94.3%, the 7 characters 2.0%.',
  '- What is safe (2 of 8 tried): if South had declared riichi with their last discard, the west wind (a second riichi and the 7 characters no longer safe): your west wind 70.8%, the 7 characters <0.1%.',
  'Not moved by the score (±16,000) or the round.',
];
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
  assert.deepEqual([...saidLines(why), ...saidLines(search.said)], [WHY_SAID, ...SEARCH_SAID]);
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
 * one still queued dropped when its signal stops it; a request stopped by its
 * signal fails at once either way, as policy.js fails it. `answer()` lets the
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
        if (waiting.indexOf(request) > 0) waiting.splice(waiting.indexOf(request), 1);
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

test('a review batch stopped by its signal holds the passes back until the worker has answered it, and one stopped before it is sent is not sent', async () => {
  // The player stops the review with its Stop button while its batch runs,
  // and opens a decision whose Why line asks. The worker runs the batch to
  // its end, so the Why line's pass, sent at once, would wait behind it.
  const worker = workerOfPasses(), batches = new Asking(), { insight } = insightOf(worker, new Asking(), new Asking(), batches);
  const reviewBatch = countedBatches(worker.analyzeBatch, batches), stop = new AbortController();
  const analysis = new PhysicalAnalysis(entry.position);
  const table = { planes: analysis.agent_observation_mortal(), mask: analysis.agent_mask_mortal() };
  analysis.free();
  const reviewed = reviewBatch([table], stop.signal);
  stop.abort();
  await assert.rejects(reviewed, { name: 'AbortError' });
  const why = insight.why(questionOf(reviewedDecision(), entry.position));
  for (let count = 0; count < 20; count++) await turn();
  assert.deepEqual([worker.sent, batches.gate() instanceof Promise], [1, true], 'the stopped batch runs on, and the Why line waits');
  worker.answer();
  await until(() => worker.sent === 2, 'the Why line sends its pass once the batch is answered');
  worker.answer();
  assert.equal((await why).label, 'Partly');
  assert.deepEqual([worker.deepest, batches.gate()], [1, null]);
  await assert.rejects(reviewBatch([table], stop.signal), { name: 'AbortError' });
  assert.equal(worker.sent, 2, 'nothing sent for a batch stopped before it went');
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

/** The review check's game as its review holds it once Trained AI has
 * answered the recorded decisions named (reviewedDecision), every other
 * decision of the game not reviewed yet. `read` counts how often its saved
 * commands are read, which a hand's tables are replayed from. */
function reviewOf(asked = [entry]) {
  const snapshot = reviewCheckGame(Game), hands = scanMatch(Game, snapshot).map(hand => ({ ...hand, decisions: hand.frames.map(() => null) }));
  for (const question of asked) hands[question.record.hand].decisions[question.record.decision] = reviewedDecision(question);
  let read = 0;
  return { review: { hands, snapshot: { ...snapshot, get commands() { read++; return snapshot.commands; } } }, get read() { return read; } };
}
/** A recorded decision's key in the review, "hand:step". */
const keyOf = question => `${question.record.hand}:${question.record.decision}`;

test('the reasoning speaks only of a discard where Trained AI threw another tile', () => {
  const own = reviewedDecision();
  assert.equal(reasonedAbout(own), true);
  for (const [what, decision] of [['not reviewed yet', null], ['agreed', { ...own, advised_tile: own.played_tile }], ['an only move', { ...own, forced: true }],
    ['a call', { ...own, played_kind: 'pon', played_tile: null }], ['a riichi declared', { ...own, played_kind: 'riichi' }],
    ['a riichi advised', { ...own, advised_kind: 'riichi', advised_tile: null }]]) assert.equal(reasonedAbout(decision), false, what);
});

test("a match's reasoning writes a hand's tables once, says nothing of a decision it does not reason about, and keeps a Why line once had", async () => {
  const network = recordedNetwork(), { insight: reasoning } = insightOf(network), held = reviewOf(), heard = [];
  const insight = new MatchInsight(Game, held.review, reasoning, { onChange: () => heard.push(insight.at(keyOf(entry))?.why?.state ?? null) });
  // The decision before it in its hand, agreed with, and the one after it,
  // not reviewed: nothing to say, and no hand replayed to say it.
  const hand = held.review.hands[entry.record.hand], step = entry.record.decision;
  hand.decisions[step - 1] = { ...hand.decisions[step], advised_tile: hand.decisions[step].played_tile };
  assert.deepEqual([insight.at(`${hand.index}:${step - 1}`), insight.at(`${hand.index}:${step + 1}`), held.read], [null, null, 0]);
  // The search asks first the tables the estimate counts, on the table the
  // engine writes for the decision; nothing is had yet.
  const question = questionOf(hand.decisions[step], entry.position);
  assert.deepEqual(insight.at(keyOf(entry)), { why: null, search: null, tables: tablesFirstAsked(question), offered: true });
  assert.ok(insight.at(keyOf(entry)).tables >= 36, 'at least the unedited table and the 35 changes the search asks');
  // Shown, its Why line is worked out, asking Trained AI once, and kept: shown
  // again and stepped away from, it asks nothing more.
  insight.explain(keyOf(entry));
  assert.equal(insight.at(keyOf(entry)).why.state, 'asking');
  await until(() => insight.at(keyOf(entry)).why.state === 'done', 'the Why line is had');
  insight.explain(keyOf(entry));
  insight.leave(keyOf(entry));
  assert.deepEqual(saidLines(insight.at(keyOf(entry)).why.result), [WHY_SAID]);
  assert.deepEqual(network.sizes, [2], 'the unedited table and the furiten taken away, asked once');
  assert.equal(held.read, 1, "the hand's tables are written once");
  assert.deepEqual(heard, ['asking', 'done'], 'every change is heard');
});

test('a search shows how far it has got, a step away or Cancel stops it and forgets it, the review waiting only for the pass the worker has, and one had is kept', async () => {
  const worker = workerOfPasses(), { insight: reasoning, notes, asking } = insightOf(worker), review = anyGate(notes, asking);
  const insight = new MatchInsight(Game, reviewOf().review, reasoning), key = keyOf(entry);
  const searching = () => insight.at(key).search;
  // Pressed, it is under way at once, the unedited table and its 35 changes
  // planned, and its first pass with the worker.
  insight.search(key);
  assert.deepEqual(searching(), { state: 'searching', asked: 0, planned: 36, status: undefined, said: undefined, failure: undefined });
  await until(() => worker.sent === 1, 'the first pass is with the worker');
  worker.answer();
  await until(() => searching().asked === PASS, 'the first pass is answered');
  await until(() => worker.sent === 2, 'the second pass is with the worker');
  // A step away forgets it at once. The review waits for the pass the worker
  // runs to its end anyway, and then goes on; nothing more is asked.
  insight.leave(key);
  assert.equal(searching(), null);
  assert.ok(review() instanceof Promise, 'the review waits for the pass with the worker');
  worker.answer();
  await until(() => review() === null, 'the review goes on once that pass is answered');
  for (let count = 0; count < 20; count++) await turn();
  assert.equal(worker.sent, 2, 'nothing more is asked');
  // Cancel does as much.
  insight.search(key);
  await until(() => worker.sent === 3, 'the first pass of the search asked again is with the worker');
  insight.cancel(key);
  assert.equal(searching(), null);
  worker.answer();
  await until(() => review() === null, 'the review goes on');
  // Asked again and answered to its end, what it found is kept, through a
  // step away, and Cancel has nothing to stop.
  insight.search(key);
  for (const [pass] of SEARCH.entries()) {
    await until(() => worker.sent === 4 + pass, `pass ${pass + 1} is with the worker`);
    worker.answer();
  }
  await until(() => searching().state === 'done', 'the search is had');
  insight.leave(key);
  insight.cancel(key);
  assert.equal(searching().status, 'searched');
  assert.deepEqual(saidLines(searching().said), SEARCH_SAID);
  assert.deepEqual([worker.sent, review()], [3 + SEARCH.length, null]);
});

test('stopping the reasoning stops every Why line and search under way, and one that failed is said so and asked again', async () => {
  const later = questions.find(question => question.id === 's3-4');
  const worker = workerOfPasses({ asked: [entry, later] }), { insight: reasoning, notes, asking } = insightOf(worker);
  const insight = new MatchInsight(Game, reviewOf([entry, later]).review, reasoning), review = anyGate(notes, asking);
  insight.explain(keyOf(entry));
  insight.search(keyOf(later));
  await until(() => worker.sent === 1, "the Why line's pass is with the worker");
  insight.stop();
  assert.deepEqual([insight.at(keyOf(entry)).why, insight.at(keyOf(later)).search], [null, null], 'both forgotten');
  worker.answer();
  await until(() => review() === null, 'the review goes on once the pass is answered');
  assert.equal(worker.sent, 1, 'the search never asked');

  // A worker that fails every request, as one reset fails them.
  const failing = insightOf({ analyzeBatch: async () => { throw new Error('The trained opponent did not answer in time'); } });
  const broken = new MatchInsight(Game, reviewOf().review, failing.insight), key = keyOf(entry);
  broken.explain(key);
  broken.search(key);
  await until(() => broken.at(key).why.state === 'failed' && broken.at(key).search.state === 'failed', 'both fail');
  assert.deepEqual(broken.at(key), { why: { state: 'failed', result: null }, tables: broken.at(key).tables, offered: true,
    search: { state: 'failed', asked: undefined, planned: undefined, status: undefined, said: undefined, failure: 'The trained opponent did not answer in time' } });
  broken.explain(key);
  assert.equal(broken.at(key).why.state, 'asking', 'a failed Why line is asked again as it is shown again');
  await until(() => broken.at(key).why.state === 'failed', 'and fails again');
});

test("the search is not offered below the margin floor, nor where the Why line finds that Trained AI does not answer the decision's table as the review recorded", async () => {
  // East 2 · 2 honba #14 is chosen by a margin of -9.2.
  const sure = questions.find(question => question.id === 'e2b-14');
  const network = recordedNetwork({ asked: [entry, sure] }), { insight: reasoning } = insightOf(network), held = reviewOf([entry, sure]);
  const insight = new MatchInsight(Game, held.review, reasoning);
  assert.deepEqual([insight.at(keyOf(sure)).tables, insight.at(keyOf(sure)).offered], [0, false]);
  insight.explain(keyOf(sure));
  await until(() => insight.at(keyOf(sure)).why.state === 'done', 'its Why line is had all the same');
  assert.equal(insight.at(keyOf(sure)).why.result.label, 'Why');
  // East 3 #5 with a weight the review recorded off by a hundredth: the Why
  // line's probe asks the unedited table first, and Trained AI answers it
  // otherwise, so no change could be compared.
  const decision = held.review.hands[entry.record.hand].decisions[entry.record.decision];
  decision.weighted = decision.weighted.map(choice => (choice.kind === 'discard' && choice.tile === entry.played.tile
    ? { ...choice, weight: choice.weight + 0.01 } : choice));
  assert.equal(insight.at(keyOf(entry)).offered, true);
  insight.explain(keyOf(entry));
  await until(() => insight.at(keyOf(entry)).why.state === 'done', 'the Why line is had');
  assert.equal(insight.at(keyOf(entry)).why.result.verdict, 'unlike the review');
  assert.equal(insight.at(keyOf(entry)).offered, false);
});

test('a hand that cannot be replayed to write its tables says nothing of its decisions, and asks nothing', () => {
  const held = reviewOf(), sizes = [];
  // A saved first move that no table could hold.
  const { commands } = held.review.snapshot;
  held.review.snapshot = { ...held.review.snapshot, commands: [{ type: 'choose', kind: 'discard', tile: '9z' }, ...commands.slice(1)] };
  const insight = new MatchInsight(Game, held.review, insightOf({ analyzeBatch: async positions => { sizes.push(positions.length); return []; } }).insight);
  assert.equal(insight.at(keyOf(entry)), null);
  insight.explain(keyOf(entry));
  insight.search(keyOf(entry));
  assert.deepEqual([insight.at(keyOf(entry)), sizes], [null, []]);
});
