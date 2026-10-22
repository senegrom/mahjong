import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { render } from 'svelte/server';
import init, { Game, PhysicalAnalysis, review_facts } from '../src/wasm/riichi.js';
import { scanMatch } from '../src/lib/game-review.js';
import { explainWhy, sameMove, tablesAtMost, tablesFirstAsked, whatWouldChangeItsMind } from '../src/lib/review-insight.js';
import { componentLoader } from './fixtures/load-component.js';
import { saidLines } from './fixtures/insight-panel.js';
import { text } from './fixtures/markup-text.js';
import { recordedAsk, recordedQuestions } from './fixtures/recorded-insight.js';
import { reviewCheckGame } from './fixtures/review-check-game.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const component = componentLoader();
const ReviewInsight = await component('ReviewInsight.svelte');
const DecisionReplay = await component('DecisionReplay.svelte');

const questions = recordedQuestions(Game, PhysicalAnalysis);
const entry = questions.find(question => question.id === 'e3-5'), sure = questions.find(question => question.id === 'e2b-14');

/** A recorded decision as the review keeps it: every move open on its table,
 * in the engine's words, with the weight Trained AI gave it in the recorded
 * run, the player's discard and Trained AI's. */
function decisionOf(question) {
  const analysis = new PhysicalAnalysis(question.position);
  try {
    const weighted = analysis.agent_choices().map(choice => ({ ...choice,
      weight: question.weights.find(move => sameMove(move, { kind: choice.kind, tile: choice.kind === 'riichi' ? null : choice.tile }))?.weight ?? 0 }));
    const advised = weighted.find(choice => choice.kind === 'discard' && choice.tile === question.preferred.tile);
    return { played_kind: 'discard', played_tile: question.played.tile, advised_kind: 'discard', advised_tile: question.preferred.tile,
      advised: advised.label, agreed: false, forced: false, weighted };
  } finally { analysis.free(); }
}

/** East 3 of the review check's game, its fifth decision reviewed as the
 * recorded run answered it and the rest not yet. */
const east3 = scanMatch(Game, reviewCheckGame(Game))[entry.record.hand];
const step = entry.record.decision;
const handWith = decision => ({ ...east3, decisions: east3.frames.map((_, at) => (at === step ? decision : null)) });
const hand = handWith(decisionOf(entry));

/** What the reasoning has for East 3 #5 and East 2 · 2 honba #14 on the
 * recorded run's answers: their Why lines, the first partly explained and the
 * second explained, and what the search found for the first. */
const why = await explainWhy({ ...entry, facts: review_facts, ask: recordedAsk(entry).ask });
const plainWhy = await explainWhy({ ...sure, facts: review_facts, ask: recordedAsk(sure).ask });
const found = await whatWouldChangeItsMind({ ...entry, facts: review_facts, ask: recordedAsk(entry).ask });
/** East 3 #5 with its Why line had and nothing searched: its search asks 36
 * tables first and 64 at most. */
const had = { why: { state: 'done', result: why }, search: null, tables: tablesFirstAsked(entry), most: tablesAtMost(entry), offered: true };

/** The match's reasoning stood in for: `at` answers the same at every
 * decision, and every decision it is asked about is noted. A server render
 * runs no effect, so nothing else of it is asked. */
function standIn(state) {
  const asked = [];
  return { asked, onChange: () => {}, at: key => { asked.push(key); return state; }, explain() {}, leave() {}, search() {}, cancel() {} };
}
const panel = (state, props = {}) => render(ReviewInsight, { props: { insight: standIn(state), hand, step, positionMs: 157, ...props } }).body;

/** Text as the page holds it, the characters the markup writes as entities
 * written out again. */
const decoded = written => written.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
/** A piece of markup's words as they read, without the space the markup
 * reader leaves where a tag stood before a colon or a full stop. */
const words = markup => decoded(text(markup)).replace(/ ([:.,])/g, '$1');
/** An attribute's value as the page holds it, or null where it has none. */
const attribute = (markup, name) => {
  const value = new RegExp(`${name}="([^"]*)"`).exec(markup)?.[1];
  return value == null ? null : decoded(value);
};
const whyLine = html => /<p class="why[^"]*"[^>]*>([^]*?)<\/p>/.exec(html);
const button = html => /<button type="button" class="app-control ask[^"]*"([^>]*)>([^]*?)<\/button>/.exec(html);
const statusLine = html => /<p class="visually-hidden[^"]*" role="status" data-insight-said="">([^<]*)<\/p>/.exec(html)?.[1] ?? null;

test("a decision where Trained AI threw another tile ends with its Why line and, under it, the button, which says about how long the search takes", () => {
  const html = panel(had);
  assert.match(html, /<div class="insight[^"]*" data-insight="" data-settled="">/);
  assert.match(whyLine(html)[0], /data-why="partly explained"/);
  assert.equal(words(whyLine(html)[1]), saidLines(why)[0], 'the verdict and the fact, in the reasoning\'s words');
  assert.equal(saidLines(why)[0], 'Partly: You had already thrown a 7 characters, so a wait on it would be furiten. Trained AI puts 93.0% on tiles you had already thrown.');
  // The button follows the Why line, and says about how long at the review's
  // 157 ms a position, from the 36 tables the search asks first to the 64 it
  // asks at most, a screen reader hearing it in words.
  const [, rest, inner] = button(html);
  assert.deepEqual([had.tables, had.most], [36, 64]);
  assert.ok(html.indexOf('data-why=') < html.indexOf('data-insight-search'));
  assert.equal(text(inner), 'What would change its mind? · about 6–10 s');
  assert.equal(attribute(rest, 'aria-label'), 'What would change its mind? About 6 to 10 seconds');
  // Nothing found yet, nothing under way, and the status line in the page,
  // empty, for its words to come.
  assert.doesNotMatch(html, /data-insight-result|data-insight-searching|data-insight-cancel|data-why-workings/);
  assert.equal(statusLine(html), '');
});

test("a verdict a check failed is a button whose tooltip is the check's workings, shown under the line only when pressed, and a verdict that held is a plain word", () => {
  const html = panel(had), verdict = /<button type="button" class="verdict[^"]*"([^>]*)>([^<]*)<\/button>/.exec(whyLine(html)[1]);
  assert.ok(verdict, 'Partly is a button');
  assert.equal(verdict[2], 'Partly');
  assert.ok(why.check, 'the check that failed says why');
  assert.equal(attribute(verdict[1], 'title'), why.check);
  assert.equal(attribute(verdict[1], 'aria-expanded'), 'false');
  assert.doesNotMatch(html, /data-why-workings/, 'folded until pressed');
  // East 2 · 2 honba #14 is explained: its verdict is a word, with nothing
  // behind it.
  const plain = panel({ ...had, why: { state: 'done', result: plainWhy }, tables: 0, offered: false });
  assert.match(whyLine(plain)[1], /<strong class="verdict[^"]*">Why<\/strong>/);
  assert.equal(words(whyLine(plain)[1]), saidLines(plainWhy)[0]);
  assert.doesNotMatch(plain, /data-why-check|title=/);
});

test('until the Why line is had, neither it nor the button is drawn, and the status line waits in the page, empty', () => {
  for (const state of [{ ...had, why: null }, { ...had, why: { state: 'asking', result: null } }]) {
    const html = panel(state);
    assert.match(html, /<div class="insight[^"]*" data-insight="">/, 'not settled');
    assert.doesNotMatch(html, /data-why=|data-insight-search/);
    assert.equal(statusLine(html), '');
  }
  // A Why line that failed leaves the button to be pressed all the same.
  const failed = panel({ ...had, why: { state: 'failed', result: null } });
  assert.match(failed, /data-settled=""/);
  assert.doesNotMatch(failed, /data-why=/);
  assert.ok(button(failed), 'the button is offered');
});

test("no button is offered below the margin floor, where Trained AI does not answer the table as the review recorded, or while it cannot be reached, and none says how long without the review's own time a position", () => {
  for (const [what, html] of [['below the floor or unlike the review', panel({ ...had, tables: 0, offered: false })],
    ['out of reach', panel(had, { trainedAvailable: false })]]) {
    assert.doesNotMatch(html, /data-insight-search/, what);
    assert.ok(whyLine(html), `the Why line stays ${what}`);
  }
  const untimed = button(panel(had, { positionMs: null }));
  assert.equal(text(untimed[2]), 'What would change its mind?');
  assert.equal(attribute(untimed[1], 'aria-label'), null);
  // A slow device, half a second a position, and a long search: in minutes.
  const slow = button(panel({ ...had, tables: 400, most: 428 }, { positionMs: 500 }));
  assert.equal(text(slow[2]), 'What would change its mind? · about 3–4 min');
  assert.equal(attribute(slow[1], 'aria-label'), 'What would change its mind? About 3 to 4 minutes');
  // A fast one: one figure, a second said as one.
  const fast = button(panel(had, { positionMs: 20 }));
  assert.equal(text(fast[2]), 'What would change its mind? · about 1 s');
  assert.equal(attribute(fast[1], 'aria-label'), 'What would change its mind? About 1 second');
});

test('a search under way shows how far it has got, and Cancel, in place of the button', () => {
  const html = panel({ ...had, search: { state: 'searching', asked: 8, planned: 36 } });
  const progress = /<progress([^>]*)><\/progress>/.exec(html)?.[1];
  assert.deepEqual(['max', 'value', 'aria-label'].map(name => attribute(progress, name)), ['36', '8', 'Changes tried']);
  assert.equal(text(/<span class="count[^"]*">([^]*?)<\/span>/.exec(html)[1]), 'Trying changes · 8 of 36');
  assert.match(html, /<button type="button" class="app-control[^"]*" data-insight-cancel="">Cancel<\/button>/);
  assert.doesNotMatch(html, /data-insight-search=|data-insight-result/);
  assert.ok(whyLine(html), 'under the Why line');
});

test('what a search found is set out under the Why line, its lead in bold, a line for each kind of change and what moved nothing, with no button to ask again', () => {
  const html = panel({ ...had, search: { state: 'done', status: found.status, said: found.said } });
  const result = /<div class="found[^"]*" data-insight-result="">([^]*?)<\/div>/.exec(html)?.[1];
  assert.ok(result, 'what was found');
  const lead = /^<p class="lead[^"]*" tabindex="-1" data-insight-lead="">([^]*?)<\/p>/.exec(result)?.[1];
  assert.ok(lead, 'it leads with a line focus can go to');
  assert.match(lead, /<strong>One change and it throws your west wind:<\/strong>/);
  assert.equal(words(lead), 'One change and it throws your west wind:');
  const items = [...result.matchAll(/<li>([^]*?)<\/li>/g)].map(([, item]) => words(item));
  assert.deepEqual(items, found.said.items);
  assert.deepEqual(items.map(item => item.split(' (')[0]), ['A faster hand', 'A hand worth more', 'What is safe'], 'grouped by what each change did');
  assert.match(result, /<p[^>]*>Not moved by the score \(±16,000\) or the round\.<\/p>/);
  assert.doesNotMatch(html, /data-insight-search|data-insight-cancel|data-insight-searching/);
  assert.ok(html.indexOf('data-why=') < html.indexOf('data-insight-result'), 'under the Why line');
  // A pair found where no single change does: the lead, then the pair after
  // it in the same line.
  const pair = { lead: 'No single change does it. Two together:', detail: 'if North had not declared riichi, and your 7 characters had been a 2 circles: your west wind 80.0%, the 7 characters 9.0%.',
    items: [], still: '' };
  const paired = /<div class="found[^"]*" data-insight-result="">([^]*?)<\/div>/.exec(panel({ ...had, search: { state: 'done', status: 'searched', said: pair } }))[1];
  assert.equal(words(paired), `${pair.lead} ${pair.detail}`);
  assert.doesNotMatch(paired, /<ul/);
});

test('a search that failed says why under the button, which asks again', () => {
  const html = panel({ ...had, search: { state: 'failed', failure: 'The trained opponent did not answer in time' } });
  assert.ok(button(html), 'the button is back');
  assert.match(html, /<p class="failed[^"]*" data-insight-failure="">Search stopped: The trained opponent did not answer in time<\/p>/);
  assert.ok(html.indexOf('data-insight-search') < html.indexOf('data-insight-failure'));
});

test('a decision the reasoning says nothing of draws nothing, and the reasoning is not asked about it', () => {
  const own = decisionOf(entry);
  for (const [what, decision] of [['not reviewed yet', null], ['agreed', { ...own, advised_tile: own.played_tile, agreed: true }],
    ['an only move', { ...own, forced: true }], ['a call', { ...own, played_kind: 'pon', played_tile: null }],
    ['a riichi declared or not', { ...own, advised_kind: 'riichi', advised_tile: null }]]) {
    const insight = standIn(had), html = render(ReviewInsight, { props: { insight, hand: handWith(decision), step, positionMs: 157 } }).body;
    assert.doesNotMatch(html, /data-insight/, what);
    assert.deepEqual(insight.asked, [], what);
  }
  assert.doesNotMatch(render(ReviewInsight, { props: { insight: null, hand, step } }).body, /data-insight/, 'no reasoning');
});

test("Trained AI's view ends with the reasoning about its choice, and while Trained AI's move is kept back neither is drawn, nor anything that names its tile", () => {
  const insight = standIn({ ...had, search: { state: 'done', status: found.status, said: found.said } });
  const html = render(DecisionReplay, { props: { hand, step, hints: true, insight, positionMs: 157, trainedAvailable: true } }).body;
  const view = /<section class="view[^"]*"[^>]*data-view="">([^]*?)<\/section>/.exec(html)?.[1];
  assert.ok(view, "Trained AI's view");
  assert.ok(view.indexOf('role="list"') < view.indexOf('data-insight=""'), 'after the moves');
  assert.equal(words(whyLine(view)[1]), saidLines(why)[0]);
  assert.ok(view.includes('data-insight-result'));
  assert.deepEqual(insight.asked, [`${east3.index}:${step}`]);
  // Kept back, the decision is put to the player: no view, so no Why line,
  // no button and nothing found, and the reasoning is not even read.
  const quiet = standIn(had), hidden = render(DecisionReplay, { props: { hand, step, hints: true, moveHidden: true, insight: quiet } }).body;
  assert.doesNotMatch(hidden, /data-view|data-insight|data-why/);
  assert.doesNotMatch(text(hidden), /Trained AI puts|What would change its mind|throws your|furiten/);
  assert.deepEqual(quiet.asked, []);
  // Without the reasoning, as before it was built, the view ends with the moves.
  assert.doesNotMatch(render(DecisionReplay, { props: { hand, step, hints: true } }).body, /data-insight/);
});
