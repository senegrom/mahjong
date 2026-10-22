import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { parse } from 'svelte/compiler';
import { render } from 'svelte/server';
import { readBeliefs } from '../src/lib/beliefs.js';
import { Asking } from '../src/lib/game-review.js';
import { anyGate, countedBatches } from '../src/lib/review-insight-policy.js';
import { reviewFallbackNote } from '../src/lib/ui.js';
import { componentLoader } from './fixtures/load-component.js';

// Render the production components, substituting only the browser-only
// policy worker (SSR never executes the analysis effect).
const Review = await componentLoader({
  'policy.js': 'data:text/javascript,export const analyzePolicy = () => { throw new Error("unexpected SSR inference"); };',
})('Review.svelte');
const review = notes => render(Review, { props: { notes } }).body;

test('a missed ron shows the offered tile and responder advice without invented discard metrics', () => {
  const html = review([{
    turn: 12, played: 'passes', played_kind: 'pass', played_tile: null,
    call_tile: '3m', call_from: 'East', advised: 'wins on the discard', advised_tile: null,
    agreed: false, reason: 'the adviser would take the available win',
    shanten_played: null, shanten_advised: null, acceptance_played: null, acceptance_advised: null,
  }]);
  assert.match(html, /Response to/);
  assert.match(html, /aria-label="3 characters"/);
  assert.match(html, /from East/);
  assert.match(html, /passes/);
  assert.match(html, /wins on the discard/);
  assert.match(html, /the adviser would take the available win/);
  assert.doesNotMatch(html, /<table|Hand left|Tiles that improve it/);
});

test('turn action reviews retain their readiness and improvement comparisons', () => {
  const html = review([{
    turn: 4, played: 'discards the 3 characters', played_kind: 'discard', played_tile: '3m',
    advised: 'discards the 9 circles', advised_tile: '9p', agreed: false,
    shanten_played: 1, shanten_advised: 0, acceptance_played: 8, acceptance_advised: 4,
    danger_played: 'quiet', danger_advised: 'quiet', reason: 'keeps a ready hand',
  }]);
  assert.doesNotMatch(html, /Response to/);
  assert.match(html, /<table/);
  assert.match(html, /Hand left/);
  assert.match(html, /1 from waiting/);
  assert.match(html, /Tiles that improve it/);
});

const differing = [{
  turn: 4, played: 'discards the 3 characters', played_kind: 'discard', played_tile: '3m',
  advised: 'discards the 9 circles', advised_tile: '9p', agreed: false,
  shanten_played: 1, shanten_advised: 0, acceptance_played: 8, acceptance_advised: 4,
  danger_played: 'quiet', danger_advised: 'quiet', reason: 'keeps a ready hand',
}];

// The note that says why Club reviews is a status line, which comes into the
// page with the review and waits there, empty, until the review has been
// drawn, so that a screen reader hears its words. A server render is the
// review as it is first drawn, so it shows the line without them; the words
// are the note's own (reviewFallbackNote), and a browser check sees them
// arrive in the page.
const silentNote = /<p [^>]*data-review-fallback[^>]*><\/p>/;

test('Trained AI is the default adviser, and Club reviews in its place while the network cannot be had', () => {
  // No adviser chosen, no network: Trained stays selected, Club's review is shown, and the page says why.
  // The engine is only handed on, never run, in a server render.
  const html = render(Review, { props: { notes: differing, engine: {} } }).body;
  assert.match(html, /<option value="strong"[^>]*selected/);
  assert.match(html, silentNote);
  assert.match(reviewFallbackNote({}), /not available right now, so Club reviews this hand/);
  assert.match(html, /keeps a ready hand/);
  assert.match(html, /<table/);
  assert.match(html, /matched Club/);
  assert.doesNotMatch(html, /Trained AI is reviewing|policy-preference/);
  // No hand left to ask about (another window took the match over): Club
  // reviews it, never an error, and the note does not blame the network.
  for (const trainedAvailable of [true, false]) {
    const noEngine = render(Review, { props: { notes: differing, trainedAvailable } }).body;
    assert.match(noEngine, silentNote);
    assert.doesNotMatch(noEngine, /role="alert"/);
  }
  assert.match(reviewFallbackNote(null), /can no longer be put to Trained AI, so Club reviews it/);
  assert.doesNotMatch(reviewFallbackNote(null), /not available right now/);
});

test('a chosen Club review says nothing about the network', () => {
  const html = render(Review, { props: { notes: differing, adviser: 'club' } }).body;
  assert.match(html, /<option value="club"[^>]*selected/);
  assert.match(html, silentNote);
  assert.match(html, /keeps a ready hand/);
});

test("with Trained AI's move hidden, the hand's notes keep the adviser's answers behind Show answers", () => {
  // Club's notes, which a server render shows as they are first drawn: the
  // move made at each decision, and nothing of what Club would have done
  // instead, its reason or its numbers, until the button is pressed.
  const hidden = render(Review, { props: { notes: differing, adviser: 'club', hideAnswers: true } }).body;
  assert.match(hidden, /<button class="answers[^"]*" data-show-answers[^>]*>Show answers<\/button>/);
  assert.match(hidden, /discards the 3 characters/);
  assert.match(hidden, /aria-label="3 characters"/);
  assert.doesNotMatch(hidden, /instead of|discards the 9 circles|9 circles|keeps a ready hand|<table|Hand left/);
  // How many decisions matched, and the filter with its counts, say how far
  // the player went wrong, not what was better: they stay.
  assert.match(hidden, /matched Club/);
  assert.match(hidden, /Where it differs \(1\)/);
  // Without the setting the notes are as they were, with no such button.
  const shown = render(Review, { props: { notes: differing, adviser: 'club' } }).body;
  assert.match(shown, /instead of/);
  assert.match(shown, /keeps a ready hand/);
  assert.match(shown, /<table/);
  assert.doesNotMatch(shown, /data-show-answers|Show answers/);
});

test('the belief panel names honours as the page names tiles, and marks their dora', async () => {
  const agents = 'data:text/javascript,export const AGENTS = { full: "Trained" }; export const agentName = agent => AGENTS[agent];';
  const AgentWeights = await componentLoader({ 'agents.js': agents })('AgentWeights.svelte');
  // The belief head in Mortal's order: index 31 is its P, the white dragon.
  const hands = new Float32Array(3 * 34).fill(-5);
  hands[31] = 5; hands[27] = 4;
  const choice = { kind: 'discard', tile: '1m', label: 'Discard 1 characters', weight: 1 };
  const html = render(AgentWeights, { props: { dora: ['5z'], analysis: {
    agent: 'full', kind: 'policy', value: 0.1, choice, choices: [choice],
    beliefs: readBeliefs(hands, [13, 13, 13]),
  } } }).body;
  assert.match(html, /aria-label="white dragon, dora"/);
  assert.match(html, /aria-label="east wind"/);
  assert.doesNotMatch(html, /NaN|Front\.svg/, 'no kind is shown as an unknown blank tile');
});

// A server render never runs the review's effects, and no unit test has a
// browser to mount the review in. So the effect that asks Trained AI is run
// here from the component's own source, as Svelte runs it: once as the review
// opens or is asked again, and its cleanup as the review is closed or asked
// again. The component's state, the questions and `untrack` are stand-ins.
const reviewSource = readFileSync(new URL('../src/lib/Review.svelte', import.meta.url), 'utf8');
function find(node, predicate) {
  if (!node || typeof node !== 'object') return null;
  if (predicate(node)) return node;
  for (const value of Object.values(node)) {
    const found = find(value, predicate);
    if (found) return found;
  }
  return null;
}
const askingEffect = (() => {
  const call = find(parse(reviewSource, { modern: true }).instance.content, node => node.type === 'CallExpression'
    && node.callee?.name === '$effect' && reviewSource.slice(node.start, node.end).includes('reviewWithStrong'));
  assert.ok(call, 'the review asks Trained AI from an effect');
  return reviewSource.slice(call.arguments[0].start, call.arguments[0].end);
})();
const turn = () => new Promise(resolve => setImmediate(resolve));

/** The review's effect on its state, Trained AI chosen and two decisions to
 * ask about unless `state` says otherwise, telling `onask` by beginning a
 * spell of the page's `asking`. Each request is held until the test answers
 * or fails it, and notes whether the gate was already shut as it was sent;
 * `told` notes, each time the review said it was asking, whether it said so
 * untracked. `run()` runs the effect and answers its cleanup. */
function askingNotes(asking, state = {}) {
  const requests = [];
  const context = vm.createContext({
    notes: [{ turn: 1 }, { turn: 2 }], engine: {}, retry: 0, reviewer: 'strong', cached: null,
    trainedNotes: null, completed: 0, reviewing: false, failure: '', AbortController, told: [], untracking: false,
    untrack: read => {
      context.untracking = true;
      try { return read(); } finally { context.untracking = false; }
    },
    onask: () => {
      context.told.push(context.untracking);
      return asking.begin();
    },
    analyzePolicy: () => { throw new Error('The stand-in for the questions asks nothing itself'); },
    reviewWithStrong: (_engine, _notes, _analyze, { signal }) => new Promise((answer, fail) => {
      requests.push({ signal, shut: asking.gate() !== null, answer, fail });
    }),
    ...state,
  });
  const effect = vm.runInContext(`(${askingEffect})`, context);
  return { context, requests, run: () => effect() };
}

test("the hand's notes shut the review's gate before their first question, and open it once, however they stop", async () => {
  // Answered: shut, untracked, before the first question is sent, and open
  // again as the answers come.
  let asking = new Asking(), notes = askingNotes(asking);
  const close = notes.run();
  assert.deepEqual(notes.context.told, [true], 'told once, untracked');
  assert.equal(notes.requests.length, 1);
  assert.equal(notes.requests[0].shut, true, 'shut before the first question is sent');
  const rows = [{ agreed: true }];
  notes.requests[0].answer(rows);
  await turn();
  assert.equal(asking.gate(), null, 'open as the answers come');
  assert.equal(notes.context.trainedNotes, rows);
  assert.equal(notes.context.reviewing, false);
  // Closed later, they say nothing more, which would open the gate on
  // another spell under way meanwhile.
  const other = asking.begin();
  close();
  assert.notEqual(asking.gate(), null, 'the notes said once that they stopped');
  other();

  // A failure, or no answer in time, which fails every request the worker
  // holds: open as it comes, and said.
  for (const error of [new Error('The opponent worker failed'), new Error('The trained opponent did not answer in time')]) {
    asking = new Asking();
    notes = askingNotes(asking);
    notes.run();
    notes.requests[0].fail(error);
    await turn();
    assert.equal(asking.gate(), null, error.message);
    assert.equal(notes.context.failure, error.message);
  }

  // Closed before the answers come, as when the next hand is dealt, Play is
  // left or Club is chosen: open at once, the question dropped, and nothing
  // more said as its refusal comes.
  asking = new Asking();
  notes = askingNotes(asking);
  notes.run()();
  assert.equal(asking.gate(), null, 'open as the notes are closed');
  assert.equal(notes.requests[0].signal.aborted, true, 'the question is dropped');
  const later = asking.begin();
  notes.requests[0].fail(new DOMException('Match changed', 'AbortError'));
  await turn();
  assert.notEqual(asking.gate(), null, 'nothing more is said');
  assert.equal(notes.context.failure, '');
  later();

  // Asked again, as Svelte does when what they review changes: the cleanup,
  // then the next run, which shuts the gate afresh, the first run's late
  // refusal leaving it shut.
  asking = new Asking();
  notes = askingNotes(asking);
  notes.run()();
  notes.run();
  assert.deepEqual(notes.context.told, [true, true]);
  assert.equal(notes.requests[1].shut, true);
  notes.requests[0].fail(new DOMException('Match changed', 'AbortError'));
  await turn();
  assert.notEqual(asking.gate(), null, "the first run's end leaves the next run's gate shut");
  notes.requests[1].answer([]);
  await turn();
  assert.equal(asking.gate(), null);

  // Nothing asked, so nothing said: Club chosen, no decision to ask about,
  // or answers already had for the same decisions.
  const decisions = [{ turn: 1 }], engine = {};
  for (const state of [{ reviewer: 'club' }, { notes: [] }, { notes: decisions, engine, cached: { source: decisions, engine, rows: [] } }]) {
    asking = new Asking();
    notes = askingNotes(asking, state);
    notes.run();
    assert.deepEqual({ told: notes.context.told, requests: notes.requests.length, gate: asking.gate() }, { told: [], requests: 0, gate: null },
      JSON.stringify(Object.keys(state)));
  }
});

test("the page gives the review of the game so far the gate that the hand's notes and the review's reasoning shut, and counts its batches with the reasoning's passes", async () => {
  // The page's own wiring, from its source: what it tells the notes to call
  // as they begin to ask, the gate it gives the review and what the review's
  // batches go through, each run on the spells the page counts; and the
  // spells it gives the reasoning, the same ones.
  const appSource = readFileSync(new URL('../src/App.svelte', import.meta.url), 'utf8');
  const app = parse(appSource, { modern: true }), script = app.instance.content;
  const declared = name => find(script, node => node.type === 'VariableDeclarator' && node.id.name === name)?.init;
  for (const name of ['notesAsking', 'insightAsking', 'workerBatches']) assert.equal(declared(name)?.callee.name, 'Asking', `the page counts ${name}`);
  const onask = find(app.fragment, node => node.type === 'Component' && node.name === 'Review')
    ?.attributes.find(attribute => attribute.name === 'onask')?.value?.expression;
  const options = find(script, node => node.type === 'NewExpression' && node.callee.name === 'MatchReview')?.arguments[2]?.properties;
  const option = name => options?.find(property => property.key?.name === name)?.value;
  const [gate, batch] = [option('gate'), option('analyzeBatch')];
  assert.ok(onask && gate && batch, 'the notes are told what to call as they ask, and the review is given a gate and a way to ask');
  const given = find(script, node => node.type === 'CallExpression' && node.callee.name === 'reviewInsight')?.arguments[0]?.properties;
  assert.deepEqual(['asking', 'notes', 'batches'].map(name => given?.find(property => property.key?.name === name)?.value?.name),
    ['insightAsking', 'notesAsking', 'workerBatches'], "the reasoning asks in spells of its own, waiting for the notes and for the worker's batches");
  const notesAsking = new Asking(), insightAsking = new Asking(), workerBatches = new Asking();
  let answer = null;
  const analyzeBatch = () => new Promise(resolve => { answer = resolve; });
  const context = vm.createContext({ notesAsking, insightAsking, workerBatches, anyGate, countedBatches, analyzeBatch });
  const run = node => vm.runInContext(`(${appSource.slice(node.start, node.end)})`, context);
  const [ask, shut] = [onask, gate].map(run);
  assert.equal(shut(), null, 'open while nothing asks');
  const stop = ask();
  assert.ok(shut() instanceof Promise, 'shut while the notes ask');
  stop();
  assert.equal(shut(), null, 'open once they stop');
  const end = insightAsking.begin();
  assert.ok(shut() instanceof Promise, 'shut while the reasoning asks');
  end();
  assert.equal(shut(), null, 'open once it stops');
  // The review's batches go through the page's counted way of asking, so
  // that the reasoning's passes wait while one is with the worker.
  const reviewBatch = run(batch.type === 'Identifier' ? declared(batch.name) : batch);
  const asked = reviewBatch([]);
  assert.ok(workerBatches.gate() instanceof Promise, "the review's batch is counted while it is with the worker");
  answer([]);
  assert.deepEqual(await asked, []);
  assert.equal(workerBatches.gate(), null, 'and no longer once answered');
});

test("with Trained AI's move hidden, the hand's notes edge every decision alike, agreed or not, until the answers are shown", () => {
  // The notes keep the answers back: their rows take the edge that neither
  // agrees nor differs, and without the setting they do not.
  const hidden = render(Review, { props: { notes: differing, adviser: 'club', hideAnswers: true } }).body;
  assert.match(hidden, /<section class="review\b[^"]*\bhiding\b/);
  const shown = render(Review, { props: { notes: differing, adviser: 'club' } }).body;
  assert.doesNotMatch(shown, /<section class="review\b[^"]*\bhiding\b/);
  // A server render lists only the decisions where the adviser differs, so
  // the green edge of one where it agreed, listed under Every decision, is
  // read from the component's own rule for it, as Svelte runs it.
  const rule = find(parse(reviewSource, { modern: true }).fragment, node => node.type === 'ClassDirective' && node.name === 'agreed');
  assert.ok(rule, 'a decision the adviser agreed with has an edge of its own');
  const agreedEdge = (note, hiding) => vm.runInNewContext(`(${reviewSource.slice(rule.expression.start, rule.expression.end)})`, { note, hiding });
  assert.equal(agreedEdge({ agreed: true }, true), false, 'not while the answers are kept back');
  assert.equal(agreedEdge({ agreed: true }, false), true, 'once they are shown');
  assert.equal(agreedEdge({ agreed: false }, false), false);
});
