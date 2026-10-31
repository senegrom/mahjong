import test from 'node:test';
import assert from 'node:assert/strict';
import { render } from 'svelte/server';
import { stepSaid } from '../src/lib/game-review.js';
import { TREND_WINDOW } from '../src/lib/review-trend.js';
import { playedNetwork, trainedName } from '../src/lib/trained-network.js';
import { REVIEW_WIDE } from '../src/lib/ui.js';
import { componentLoader } from './fixtures/load-component.js';
import { text } from './fixtures/markup-text.js';

// Render the production components, substituting only the decision on show,
// whose own tests render it (review-replay-render.test.js). A server render
// has no screen to measure, so the review draws its list folded above the
// decision, as on a narrow screen.
const component = componentLoader({
  'DecisionReplay.svelte': 'data:text/javascript,export default function DecisionReplay() {}',
});
const GameReview = await component('GameReview.svelte');
const ReviewTrend = await component('ReviewTrend.svelte');

const seat = (wind, values = {}) => ({ seat: wind, turn: false, riichi: false, hand: [], drawn: null, melds: [], discards: [], ...values });
const view = { phase: 'act', wall: 45, pending_discard: null, pending_from: null, dora_types: [],
  seats: [seat('east', { turn: true, drawn: '3m' }), seat('south'), seat('west'), seat('north')] };
const choices = [{ kind: 'discard', tile: '7m', label: 'discards the 7 characters' }, { kind: 'discard', tile: '1m', label: 'discards the 1 characters' },
  { kind: 'discard', tile: '9p', label: 'discards the 9 circles' }];
const frame = { view, choices, lines: [], command: 0, played: { kind: 'discard', tile: '7m' } };
const decision = (values = {}) => ({
  hand: 0, index: 0, turn: 3, played: 'discards the 7 characters', played_kind: 'discard', played_tile: '7m',
  advised: 'discards the 1 characters', advised_kind: 'discard', advised_tile: '1m', agreed: false, forced: false,
  played_weight: 0.02, preferred_weight: 0.81, dora_types: [], weighted: [], frame,
  judgement: { stage: 'move', played: 0.02, preferred: 0.81, gap: 0.79, flagged: true }, ...values,
});
const hand = (index, values = {}) => ({ index, name: ['East 1', 'East 2', 'East 3'][index], counters: 0, seat: 'east',
  frames: [frame, frame], outcome: 'East wins by self-draw', change: 0, status: 'waiting', decisions: null, ...values });
/** A hand reviewed: its first decision flagged, its second agreed with. */
const done = (index, values = {}) => hand(index, { status: 'done',
  decisions: [decision({ hand: index }), decision({ hand: index, index: 1, agreed: true, judgement: null })], ...values });
// A finished match's review unless `over` says the match is under way.
const review = (values = {}) => ({
  id: '7:120', hands: [hand(0), hand(1), hand(2)], busy: false, failure: '', held: false, slow: false, estimate: null,
  complete: false, requested: true, positionMs: null, decisionMs: null, recorded: false, closed: false, over: true,
  onChange: () => {}, review: () => {}, resume: () => {}, stop: () => {}, ...values,
});
const page = (props = {}) => render(GameReview, { props: { trainedAvailable: true, ...props } }).body;
/** What a button marked with this attribute says. */
const buttonSaid = (html, attribute) => text(new RegExp(`<button[^>]*\\b${attribute}\\b[^>]*>([^]*?)</button>`).exec(html)?.[1] ?? '');
/** Where the step bar says the review is. */
const placeSaid = html => text(/<p class="place[^"]*" data-step="">([^]*?)<\/p>/.exec(html)?.[1] ?? '');
/** The step bar's Previous and Next: what each is named, whether it is
 * disabled, and where it waits, disabled to a screen reader but keeping its
 * focus, that it does. */
const steps = html => Object.fromEntries([...html.matchAll(/<button[^>]*\bdata-(previous|next)=""[^>]*aria-label="([^"]*)"([^>]*)>/g)]
  .map(([, which, label, rest]) => [which, { label, disabled: /(?:^|\s)disabled=""/.test(rest),
    ...(/\baria-disabled="true"/.test(rest) ? { waiting: true } : {}) }]));
/** Every hand of the list, as its header gives it. */
function heads(html) {
  return [...html.matchAll(/<li class="hand[^"]*" data-hand="(\d+)" data-status="(\w+)">\s*<div class="head[^"]*">([^]*?)<\/div>/g)]
    .map(([, index, status, head]) => ({
      hand: Number(index), status,
      name: text(/class="name[^"]*"[^>]*>([^<]*)</.exec(head)[1]), button: /<button[^>]*\bdata-hand-row=/.test(head),
      change: /<span class="change[^"]*">([^<]*)</.exec(head)?.[1] ?? null,
      outcome: /<span class="outcome[^"]*">([^<]*)</.exec(head)[1],
      state: /<span class="state[^"]*">([^<]*)</.exec(head)?.[1] ?? '',
      review: /<button[^>]*\bdata-review-hand="\d+"[^>]*aria-label="([^"]*)"/.exec(head)?.[1] ?? null }));
}
/** Every decision's line in the list: which decision, whether it is flagged,
 * on show or shown though its filter leaves it out, its name and what it
 * shows in words, the tiles being pictures. */
function lines(html) {
  return [...html.matchAll(/<button type="button" class="(line[^"]*)" data-step-to="([^"]+)"([^>]*)>([^]*?)<\/button>/g)]
    .map(([, classes, key, rest, inner]) => ({ key, flagged: /\bdata-flagged="/.test(rest), current: /aria-current="step"/.test(rest),
      outside: /\boutside\b/.test(classes), label: /aria-label="([^"]*)"/.exec(rest)[1], shown: text(inner) }));
}
/** What the trend's folded line says. */
const trendSaid = html => text(/<details class="trend[^"]*" data-trend=""[^>]*>\s*<summary[^>]*>([^]*?)<\/summary>/.exec(html)?.[1] ?? '');
/** What the folded list's summary says. */
const listSaid = html => text(/<details class="list[^"]*" data-review-index=""[^>]*>\s*<summary[^>]*>([^]*?)<\/summary>/.exec(html)?.[1] ?? '');
/** The filter's buttons: which, whether pressed, and what each says. */
const options = html => [...html.matchAll(/aria-pressed="(true|false)" data-filter-option="(\w+)">([^]*?)<\/button>/g)]
  .map(([, pressed, option, label]) => [option, pressed === 'true', text(label)]);
/** The review's two settings in its header: what each is called, and
 * whether it is on. */
const settings = html => Object.fromEntries([...html.matchAll(/<label[^>]*>(<input[^>]*\bdata-hide-(move|tiles)=""[^>]*>)([^]*?)<\/label>/g)]
  .map(([, input, which, said]) => [which, { said: text(said), checked: /\bchecked\b/.test(input) }]));
/** What the step bar's line of moves says, the tiles being pictures. */
const moveLine = html => text(/<p class="moves[^"]*" data-move-line="">([^]*?)<\/p>/.exec(html)?.[1] ?? '');

test('a review under way opens on the hand just played at its first decision, which Trained AI has not answered, and Next waits for its answers', () => {
  const html = page({ review: review({ over: false, busy: true, estimate: 95, hands: [done(0), done(1), hand(2, { status: 'reviewing', done: 1 })] }) });
  assert.match(html, /<h2[^>]*>Game so far<\/h2>/);
  assert.equal(buttonSaid(html, 'data-close-review'), 'Back to the result');
  assert.match(html, /role="status"[^>]*>Reviewing East 3…</);
  assert.match(html, />5 of 6 decisions · about 2 minutes left</);
  assert.match(html, /data-stop-review/);
  // The hand just played, at its first decision, with the player's move and
  // Trained AI's to come, said in one line from the start.
  const shown = { name: 'East 3', frames: [frame, frame] };
  assert.equal(placeSaid(html), 'East 3 · decision 1 of 2');
  assert.match(text(/<p class="moves[^"]*" data-move-line="">([^]*?)<\/p>/.exec(html)[1]), /^You · Trained AI …$/);
  assert.ok(html.includes(`role="status" data-step-said="">${stepSaid({ hand: shown, step: 0, decision: null })}</p>`));
  // Previous goes back to the flagged decision of the hand before; Next waits
  // for the hand's answers rather than passing over its decisions, keeping
  // its focus meanwhile.
  assert.deepEqual(steps(html), { previous: { label: 'Previous flagged decision', disabled: false },
    next: { label: 'Next flagged decision', disabled: false, waiting: true } });
  // The list, folded under the step bar, says its filter; the decision on
  // show is current in it, dim, as the flagged decisions do not hold it yet.
  assert.match(html, /<details class="list[^"]*" data-review-index="">/, 'folded');
  assert.equal(listSaid(html), 'Hands · Flagged (2)');
  assert.ok(html.indexOf('data-steps') < html.indexOf('data-review-index'), 'the step bar comes first');
  assert.deepEqual(lines(html).map(({ key, flagged, current, outside }) => [key, flagged, current, outside]),
    [['0:0', true, false, false], ['1:0', true, false, false], ['2:0', false, true, true]]);
  assert.equal(heads(html)[2].state, 'Reviewing, 1 of 2');
  // So far, and not yet in the trend. The trend opens the list, folded.
  assert.equal(trendSaid(html), `Matched ${trainedName('Trained AI')}: 50% so far`);
  // None of the list of cards it replaces, nor the ways in and out of it.
  assert.doesNotMatch(html, /Just played|data-latest-hand|data-replay|data-open=|data-every-flag|data-moved|Show all|Choose again|>Replay<|<h3/);
  assert.doesNotMatch(text(html), /looked at again/);
});

test('a finished review opens on the final hand at its first flagged decision, says it has reviewed every hand without showing it, and goes back to the standings', () => {
  const last = done(2, { decisions: [decision({ hand: 2, agreed: true, judgement: null }), decision({ hand: 2, index: 1 })] });
  const html = page({ review: review({ complete: true, hands: [done(0), done(1), last] }) });
  assert.match(html, /<h2[^>]*>Whole game<\/h2>/);
  assert.equal(buttonSaid(html, 'data-close-review'), 'Back to the standings');
  assert.match(html, /<p class="headline[^"]*visually-hidden[^"]*" role="status">All hands reviewed: 6 decisions\.<\/p>/);
  assert.doesNotMatch(html, /data-continue|data-stop-review|data-review-hand|<progress/);
  assert.equal(placeSaid(html), 'East 3 · decision 2 of 2');
  assert.deepEqual(steps(html), { previous: { label: 'Previous flagged decision', disabled: false }, next: { label: 'Next flagged decision', disabled: true } });
  // Every flagged decision, in the order played, the one on show current.
  assert.deepEqual(lines(html).map(({ key, current }) => [key, current]), [['0:0', false], ['1:0', false], ['2:1', true]]);
  // Whole, the trend says no "so far".
  assert.equal(trendSaid(html), `Matched ${trainedName('Trained AI')}: 50%`);
});

test('the list gives every hand in the order played, its name, what it changed and how it ended, and a line for each decision its filter keeps, both moves named in words', () => {
  const near = decision({ index: 1, advised_tile: '9p', advised: 'discards the 9 circles', played_weight: 0.2, preferred_weight: 0.7,
    judgement: { stage: 'move', played: 0.2, preferred: 0.7, gap: 0.5, flagged: false } });
  const agreed = decision({ index: 2, agreed: true, judgement: null });
  const forced = decision({ index: 3, agreed: true, forced: true, judgement: null });
  const first = hand(0, { status: 'done', change: 3900, frames: [frame, frame, frame, frame], decisions: [decision(), near, agreed, forced] });
  const second = hand(1, { change: -8000, outcome: "West wins on North's discard" });
  const third = hand(2, { status: 'done', frames: [], decisions: [], outcome: 'Exhaustive draw' });
  const reviewed = review({ hands: [first, second, third] });

  // Each hand: its name a button to it, what it changed for the player, how
  // it ended, and how far its review has got where it is not done.
  assert.deepEqual(heads(page({ review: reviewed })), [
    { hand: 0, status: 'done', name: 'East 1', button: true, change: '+3,900', outcome: 'East wins by self-draw', state: '', review: null },
    { hand: 1, status: 'waiting', name: 'East 2', button: true, change: '−8,000', outcome: "West wins on North's discard",
      state: '2 decisions, not reviewed', review: 'Review: East 2' },
    { hand: 2, status: 'done', name: 'East 3', button: false, change: null, outcome: 'Exhaustive draw', state: 'No decisions', review: null }]);

  // The flagged decisions: the one of the first hand, and the decision on
  // show, the first of the newest hand with a decision, not reviewed yet.
  const flagged = lines(page({ review: reviewed }));
  assert.deepEqual(flagged.map(({ key, flagged: mark, current, outside }) => [key, mark, current, outside]),
    [['0:0', true, false, false], ['1:0', false, true, true]]);
  assert.equal(flagged[0].label, 'Decision 1. You: discards the 7 characters. Trained AI: discards the 1 characters. Flagged.');
  assert.equal(flagged[0].shown, '1 →', 'its number, then both moves as tiles, Flagged said by its edge');
  assert.equal(flagged[1].label, 'Decision 1. You: discards the 7 characters. Not reviewed yet.');
  assert.equal(flagged[1].shown, '1 …');

  // Those where Trained AI differed: the flagged one says so in words.
  const different = lines(page({ review: reviewed, filter: 'differed' }));
  assert.deepEqual(different.map(({ key }) => key), ['0:0', '0:1', '1:0']);
  assert.equal(different[0].shown, '1 → Flagged');
  assert.equal(different[1].label, 'Decision 2. You: discards the 7 characters. Trained AI: discards the 9 circles.');
  assert.equal(different[1].shown, '2 →');

  // Every decision of every hand, reviewed or not, in the order played: a
  // tick where Trained AI agreed, the only move, and the dots of a hand not
  // reviewed yet. Here the filter keeps the decision on show.
  const every = lines(page({ review: reviewed, filter: 'every' }));
  assert.deepEqual(every.map(({ key, outside }) => [key, outside]), [['0:0', false], ['0:1', false], ['0:2', false], ['0:3', false],
    ['1:0', false], ['1:1', false]]);
  assert.deepEqual(every.map(({ shown }) => shown), ['1 → Flagged', '2 →', '3 ✓', '4 Only move', '1 …', '2 …']);
  assert.equal(every[2].label, 'Decision 3. You: discards the 7 characters. Trained AI agreed.');
  assert.equal(every[3].label, 'Decision 4. You: discards the 7 characters. Only move.');
});

test("the filter's buttons say in a word what each keeps, with how many, every decision of every hand counted, and the step bar is named for it", () => {
  const near = decision({ index: 1, judgement: { stage: 'move', played: 0.2, preferred: 0.7, gap: 0.5, flagged: false } });
  const first = hand(0, { status: 'done', frames: [frame, frame, frame], decisions: [decision(), near, decision({ index: 2, agreed: true, judgement: null })] });
  const reviewed = review({ hands: [first, hand(1), hand(2)] });
  const opening = page({ review: reviewed });
  assert.match(opening, /role="group" aria-label="Which decisions to step through"/);
  // All counts the decisions of the hands not reviewed yet too, as it lists
  // them: the table and the move are known before Trained AI answers.
  assert.deepEqual(options(opening), [['flagged', true, 'Flagged (1)'], ['differed', false, 'Differed (2)'], ['every', false, 'All (7)']]);
  // Where a word is not enough, the button's title says what keeps them.
  const titles = [...opening.matchAll(/<button[^>]*\btitle="([^"]*)"[^>]*\bdata-filter-option="(\w+)"/g)].map(([, title, option]) => [option, title]);
  assert.deepEqual(titles, [['flagged', "Your move under 5%, Trained AI's choice 60% or more"], ['differed', 'Trained AI chose another move']]);
  // What a choice of filter keeps is said from a line in the page from the start.
  assert.match(opening, /<p class="visually-hidden[^"]*" role="status" data-filter-said=""><\/p>/);
  for (const [filter, label, name] of [['differed', 'Differed (2)', 'decision where Trained AI differed'], ['every', 'All (7)', 'decision']]) {
    const html = page({ review: reviewed, filter });
    assert.equal(listSaid(html), `Hands · ${label}`);
    assert.equal(steps(html).previous.label, `Previous ${name}`);
    assert.equal(steps(html).next.label, `Next ${name}`);
  }
});

test('the decision on show is the place given, Previous and Next go across hands to the decisions the filter keeps, disabled only at the ends, and one it leaves out is drawn dim in its place', () => {
  const reviewed = review({ complete: true, hands: [done(0), done(1), done(2)] });
  // An agreed decision, which the flagged decisions leave out.
  const between = page({ review: reviewed, place: { hand: 1, step: 1 } });
  assert.equal(placeSaid(between), 'East 2 · decision 2 of 2');
  assert.deepEqual(steps(between), { previous: { label: 'Previous flagged decision', disabled: false },
    next: { label: 'Next flagged decision', disabled: false } });
  assert.deepEqual(lines(between).map(({ key, current, outside }) => [key, current, outside]),
    [['0:0', false, false], ['1:0', false, false], ['1:1', true, true], ['2:0', false, false]]);
  // At either end of the game the button that would leave it is disabled.
  assert.deepEqual(steps(page({ review: reviewed, place: { hand: 0, step: 0 } })),
    { previous: { label: 'Previous flagged decision', disabled: true }, next: { label: 'Next flagged decision', disabled: false } });
  assert.deepEqual(steps(page({ review: reviewed, place: { hand: 2, step: 1 }, filter: 'every' })),
    { previous: { label: 'Previous decision', disabled: false }, next: { label: 'Next decision', disabled: true } });
  // A place the review does not have is no place: it opens where it lands.
  assert.equal(placeSaid(page({ review: reviewed, place: { hand: 7, step: 0 } })), 'East 3 · decision 1 of 2');
});

test('a hand not reviewed yet says so, and every decision is stepped through it at once, while the flagged ones wait for its answers on a Next that keeps its focus', () => {
  const reviewing = review({ over: false, busy: true, hands: [done(0), hand(1, { status: 'reviewing', done: 0 }), done(2)] });
  // On the hand being reviewed, Next waits under the flagged decisions, said
  // to be disabled but not disabled outright, which would drop its focus;
  // under every decision it goes on.
  assert.deepEqual(steps(page({ review: reviewing, place: { hand: 1, step: 0 } })).next,
    { label: 'Next flagged decision', disabled: false, waiting: true });
  assert.deepEqual(steps(page({ review: reviewing, place: { hand: 1, step: 0 }, filter: 'every' })).next, { label: 'Next decision', disabled: false });
  // A hand waiting, not under review: Next goes on past it.
  const waiting = review({ hands: [done(0), hand(1), done(2)] });
  assert.deepEqual(steps(page({ review: waiting, place: { hand: 1, step: 0 } })).next, { label: 'Next flagged decision', disabled: false });
});

test("the header offers the review's two settings, off unless set, and with Trained AI's move kept back every line keeps it back, '?' in its place and said so, until the player reveals that decision", () => {
  const reviewed = review({ complete: true, hands: [done(0), done(1), done(2)] });
  const plain = page({ review: reviewed });
  assert.deepEqual(settings(plain), { move: { said: "Hide Trained AI's move", checked: false }, tiles: { said: "Hide other players' tiles", checked: false } });
  assert.ok(plain.indexOf('game-review-title') < plain.indexOf('data-hide-move') && plain.indexOf('data-hide-tiles') < plain.indexOf('data-close-review'),
    'between the heading and Back');
  assert.doesNotMatch(plain, /data-reveal/, 'nothing kept back, nothing to show');
  assert.doesNotMatch(plain, /<p class="moves[^"]*\bhiding\b/, "the step bar's line takes only its moves' room");
  // Every decision listed, each hand's first flagged and its second agreed:
  // a line gives the player's move and the flag, and keeps Trained AI's move
  // back, and that it agreed, to the eye and to a screen reader alike.
  const hidden = page({ review: reviewed, filter: 'every', hideMove: true });
  assert.equal(settings(hidden).move.checked, true);
  assert.deepEqual(lines(hidden).map(({ key, shown }) => [key, shown]),
    [['0:0', '1 → ? Flagged'], ['0:1', '2 → ?'], ['1:0', '1 → ? Flagged'], ['1:1', '2 → ?'], ['2:0', '1 → ? Flagged'], ['2:1', '2 → ?']]);
  assert.equal(lines(hidden)[0].label, "Decision 1. You: discards the 7 characters. Trained AI's move hidden. Flagged.");
  assert.equal(lines(hidden)[1].label, "Decision 2. You: discards the 7 characters. Trained AI's move hidden.");
  // The decision on show, the final hand's first, puts the question in the
  // step bar: Show answer in Trained AI's place.
  assert.equal(moveLine(hidden), 'You · Trained AI ? Show answer · Flagged');
  // A decision revealed, by Show answer or by a pick, shows its move again,
  // the pick as Now, whatever the settings; the rest stay kept back.
  const answered = page({ review: reviewed, filter: 'every', hideMove: true, hideTiles: true,
    revealed: { '0:0': { pick: null, answer: true }, '2:0': { pick: { kind: 'discard', tile: '9p' }, answer: true } } });
  assert.deepEqual(lines(answered).map(({ key, shown }) => [key, shown]),
    [['0:0', '1 → Flagged'], ['0:1', '2 → ?'], ['1:0', '1 → ? Flagged'], ['1:1', '2 → ?'], ['2:0', '1 → Flagged'], ['2:1', '2 → ?']]);
  assert.equal(moveLine(answered), 'You · Now · Trained AI · Flagged');
  // The step bar's line keeps the reveal's room, revealed or not, so that a
  // reveal leaves the bar, and the table under it, where they were.
  assert.match(hidden, /<p class="moves[^"]*\bhiding\b[^"]*" data-move-line="">/);
  assert.match(answered, /<p class="moves[^"]*\bhiding\b[^"]*" data-move-line="">/);
  // The other players' tiles alone kept back: Show tiles, both moves on show.
  assert.equal(moveLine(page({ review: reviewed, hideTiles: true })), 'You · Trained AI Show tiles · Flagged');
});

test('a repeated hand goes by its name with its honba wherever the review names it', () => {
  const repeat = done(1, { name: 'East 1 · 1 honba', counters: 1 });
  const html = page({ review: review({ complete: true, hands: [done(0), repeat] }) });
  assert.equal(placeSaid(html), 'East 1 · 1 honba · decision 1 of 2');
  assert.equal(heads(html)[1].name, 'East 1 · 1 honba', 'the list does not add the honba again');
});

test('a review of the game so far offers the game so far, the next hand or the rest, and keeps the game out of the trend until it is whole', () => {
  const earlier = { id: 'earlier', network: playedNetwork().version, at: Date.UTC(2026, 9, 1, 12), matched: 60, decisions: 100, suits: true };
  const reviewed = page({ trend: [earlier], review: review({ over: false, complete: true, hands: [done(0), done(1), done(2)] }) });
  assert.match(reviewed, /role="status"[^>]*>All hands reviewed: 6 decisions\.</);
  assert.doesNotMatch(reviewed, /data-continue|data-stop-review/);
  assert.equal(trendSaid(reviewed), `Matched ${trainedName('Trained AI')}: 50% so far`, 'the figure is the game so far');
  assert.match(reviewed, /Joins your trend once the game is over and fully reviewed\./);
  assert.doesNotMatch(reviewed, /points against/);

  // Not yet asked for, it offers the game so far, under the name the result's
  // button gives it; on a slow device that held after a hand, the next hand
  // or the rest.
  const unasked = page({ review: review({ over: false, requested: false, hands: [done(0), done(1), hand(2)] }) });
  assert.equal(buttonSaid(unasked, 'data-continue'), 'Review game so far');
  const held = page({ review: review({ over: false, held: true, slow: true, positionMs: 700, decisionMs: 800, estimate: 20, hands: [hand(0), hand(1), done(2)] }) });
  assert.equal(buttonSaid(held, 'data-next-hand'), 'Review next hand');
  assert.equal(buttonSaid(held, 'data-rest'), 'Review the rest');
  // With no decision made yet, there is nothing to step through, and the
  // screen says so.
  const none = page({ review: review({ over: false, complete: true, hands: [hand(0, { frames: [], status: 'done', decisions: [] })] }) });
  assert.doesNotMatch(none, /data-steps/);
  assert.match(none, /<p class="empty[^"]*">No decisions yet\.<\/p>/);
  const nothing = page({ review: review({ complete: true, hands: [hand(0, { frames: [], status: 'done', decisions: [] })] }) });
  assert.match(nothing, /<p class="empty[^"]*">You made no decisions\.<\/p>/);
});

test('a finished review whose filter keeps nothing says so in the list, in the words the filter says it in', () => {
  const agreed = decision({ agreed: true, judgement: null });
  const hands = [0, 1, 2].map(index => hand(index, { status: 'done', decisions: [agreed, { ...agreed, index: 1 }] }));
  assert.match(page({ review: review({ complete: true, hands }) }), /<p class="empty[^"]*">None flagged\.<\/p>/);
  assert.match(page({ review: review({ complete: true, hands }), filter: 'differed' }), /<p class="empty[^"]*">Trained AI agreed at every decision\.<\/p>/);
  // While the review goes on, the hands' own states say how far it has got.
  assert.doesNotMatch(page({ review: review({ hands }) }), /class="empty/);
});

test('a slow device that held after a hand offers the next hand or the rest, with how long the rest takes, and opens the list to choose a hand from', () => {
  const html = page({ review: review({ held: true, slow: true, positionMs: 700, decisionMs: 800, estimate: 200, hands: [done(0), hand(1), hand(2)] }) });
  assert.match(html, /role="status"[^>]*>Reviewed 1 of 3 hands; about 0\.8 seconds a decision on this device\.</);
  assert.match(html, /about 3 minutes left/);
  assert.equal(buttonSaid(html, 'data-next-hand'), 'Review next hand');
  assert.equal(buttonSaid(html, 'data-rest'), 'Review the rest');
  assert.match(html, /<details class="list[^"]*" data-review-index="" open="">/, 'the hands are open to choose from');
  assert.match(html, /data-review-hand="1"[^>]*aria-label="Review: East 2"/);
});

test('on a wide screen the list is a column of its own, after the decision in the page and headed for a screen reader, and nothing is folded', () => {
  // A screen that answers the review's media query, as the page reads it
  // while it draws.
  globalThis.matchMedia = query => ({ matches: query === REVIEW_WIDE, addEventListener() {}, removeEventListener() {} });
  try {
    const html = page({ review: review({ complete: true, hands: [done(0), done(1), done(2)] }) });
    assert.doesNotMatch(html, /<details class="list/);
    assert.match(html, /<aside class="rail[^"]*" data-review-index="" aria-labelledby="review-hands">\s*<h3 id="review-hands" class="visually-hidden[^"]*">Hands<\/h3>/);
    assert.match(html, /<div class="panes[^"]*\bwide\b/);
    // The decision first, so that Tab reaches Previous and Next before the
    // list; the grid draws the list on the left.
    assert.ok(html.indexOf('data-steps') < html.indexOf('data-review-index'));
    assert.deepEqual(lines(html).map(({ key, current }) => [key, current]), [['0:0', false], ['1:0', false], ['2:0', true]]);
    assert.match(html, /<details class="trend[^"]*" data-trend=""/, 'the trend opens the column');
  } finally { delete globalThis.matchMedia; }
  // Measuring nothing, a server render folds the list above the decision.
  assert.match(page({ review: review({ complete: true, hands: [done(0)] }) }), /<details class="list[^"]*" data-review-index=""/);
});

test('without the network the review waits and says so, and is not offered hand by hand', () => {
  // Not yet asked for, it starts by itself once Trained AI can be reached.
  const html = page({ trainedAvailable: false, review: review({ requested: false }) });
  assert.match(html, /role="status"[^>]*>Waiting until Trained AI can be reached\.</);
  assert.doesNotMatch(html, /data-review-hand|data-continue/);
  // Asked for already, it offers Continue again once Trained AI can be
  // reached, and meanwhile says only that it cannot be.
  const asked = page({ trainedAvailable: false, review: review({ requested: true }) });
  assert.match(asked, /role="status"[^>]*>Trained AI cannot be reached right now\.</);
  assert.doesNotMatch(asked, /data-review-hand|data-continue/);
});

test('the trend states the change against earlier games without judging it, with a table of the games', () => {
  const network = playedNetwork().version;
  const games = Array.from({ length: 4 }, (_, index) => ({ id: `g${index}`, network, at: Date.UTC(2026, 9, 10 + index, 12), matched: 60, decisions: 100, suits: true }));
  const figure = { matched: 65, decisions: 100, share: 0.65 };
  const html = render(ReviewTrend, { props: { figure, complete: true, current: 'g3',
    games: [...games.slice(0, 3), { ...games[3], matched: 65 }] } }).body;
  assert.equal(trendSaid(html), `Matched ${trainedName('Trained AI')}: 65%`, 'it names the version that judged, and gives the figure');
  assert.doesNotMatch(html, /Not counted/);
  // The decisions with a choice: a move with only one is left out.
  assert.match(html, />65 of 100 decisions with a choice</);
  assert.match(html, /\+5 points against your previous 3 games \(60%\)/);
  assert.match(html, /role="img" aria-label="Matched Trained AI in your last 4 reviewed games: from 60% to 65%"/);
  assert.equal((html.match(/<circle class="now/g) ?? []).length, 1, 'only this game is marked');
  assert.equal((html.match(/<tr/g) ?? []).length, 5, 'a row a game and the header');
  assert.match(html, /\(this game\)/);
  assert.doesNotMatch(html, /\b(up|down|good|bad|better|worse)\b/i, 'the change is not judged');
  // Under way: the figure so far, and no change until the game is whole.
  const partial = render(ReviewTrend, { props: { figure, complete: false, current: 'g9', games: games.slice(0, 2) } }).body;
  assert.match(partial, /so far/);
  assert.match(partial, /Joins your trend once fully reviewed\./);
  assert.doesNotMatch(partial, /points against|class="now/);
  // A game reviewed as it is played joins once it is over, too.
  const playing = render(ReviewTrend, { props: { figure, complete: false, over: false, current: 'g9', games: games.slice(0, 2) } }).body;
  assert.match(playing, /so far/);
  assert.match(playing, /Joins your trend once the game is over and fully reviewed\./);
  assert.doesNotMatch(playing, /points against|class="now/);
  // A first game has nothing to compare with and no line to draw.
  const first = render(ReviewTrend, { props: { figure, complete: true, current: 'g0', games: [{ ...games[0], matched: 65 }] } }).body;
  assert.doesNotMatch(first, /points against|<svg/);
  // No change is stated as plainly as any other.
  const level = render(ReviewTrend, { props: { figure: { matched: 60, decisions: 100, share: 0.6 }, complete: true, current: 'g3',
    games: [...games.slice(0, 3), games[3]] } }).body;
  assert.match(level, />Same as your previous 3 games \(60%\)</);
  assert.equal(TREND_WINDOW, 10);
});

test("folded to one line for the review's list, the trend gives the version and the figure, so far until the game is whole, and keeps the rest inside", () => {
  const network = playedNetwork().version;
  const games = Array.from({ length: 3 }, (_, index) => ({ id: `g${index}`, network, at: Date.UTC(2026, 9, 10 + index, 12), matched: 60, decisions: 100, suits: true }));
  const figure = { matched: 21, decisions: 100, share: 0.21 };
  const folded = render(ReviewTrend, { props: { figure, complete: false, over: false, current: 'g9', games } }).body;
  const summary = /<details class="trend[^"]*" data-trend="">\s*<summary[^>]*>([^]*?)<\/summary>/.exec(folded);
  assert.ok(summary, 'a list the player opens, folded');
  assert.equal(text(summary[1]), `Matched ${trainedName('Trained AI')}: 21% so far`);
  // Inside, the decisions the figure takes, what it is compared with and the
  // games.
  assert.match(folded, />21 of 100 decisions with a choice</);
  assert.match(folded, /Joins your trend once the game is over and fully reviewed\./);
  assert.match(folded, /<details class="games[^"]*"/);
  assert.doesNotMatch(folded, /class="label|class="value/, 'the figure is said once, in the summary');
  // The whole game: no "so far", and open where the review keeps it open.
  const whole = render(ReviewTrend, { props: { figure, complete: true, expanded: true, current: 'g2', games } }).body;
  assert.match(whole, /<details class="trend[^"]*" data-trend="" open="">/);
  assert.equal(text(/<summary[^>]*>([^]*?)<\/summary>/.exec(whole)[1]), `Matched ${trainedName('Trained AI')}: 21%`);
});

test('the trend compares and shows only the games judged by the version of Trained AI this page plays', () => {
  const network = playedNetwork().version, other = `${network}-other`;
  const at = index => Date.UTC(2026, 9, 10 + index, 12);
  const games = [
    { id: 'g0', network, at: at(0), matched: 60, decisions: 100, suits: true },
    // Another version judged these, one of them the same game again.
    { id: 'g1', network: other, at: at(1), matched: 20, decisions: 100, suits: true },
    { id: 'g0', network: other, at: at(2), matched: 25, decisions: 100, suits: true },
    // This version judged this one before the review averaged over the suits.
    { id: 'g2', network, at: at(2), matched: 30, decisions: 100 },
    { id: 'g3', network, at: at(3), matched: 70, decisions: 100, suits: true },
  ];
  const html = render(ReviewTrend, { props: { figure: { matched: 70, decisions: 100, share: 0.7 }, complete: true, current: 'g3', games } }).body;
  assert.match(html, /\+10 points against your previous game \(60%\)/, 'only the same version is compared');
  assert.match(html, /aria-label="Matched Trained AI in your last 2 reviewed games: from 60% to 70%"/);
  assert.equal((html.match(/<tr/g) ?? []).length, 3, 'a row for each of its games and the header');
  assert.match(html, /Not counted: 3 games judged otherwise\./);
  assert.doesNotMatch(html, /20%|25%|30%/);
  // The same version's game judged before the averaging is not counted alone
  // either, and the note says no more than that.
  const plain = render(ReviewTrend, { props: { figure: { matched: 70, decisions: 100, share: 0.7 }, complete: true, current: 'g3',
    games: [games[0], games[3], games[4]] } }).body;
  assert.match(plain, /\+10 points against your previous game \(60%\)/);
  assert.match(plain, />Not counted: 1 game judged otherwise\.</);
});
