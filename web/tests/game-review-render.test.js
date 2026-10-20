import test from 'node:test';
import assert from 'node:assert/strict';
import { render } from 'svelte/server';
import { TREND_WINDOW } from '../src/lib/review-trend.js';
import { playedNetwork, trainedName } from '../src/lib/trained-network.js';
import { componentLoader } from './fixtures/load-component.js';
import { text } from './fixtures/markup-text.js';

// Render the production components, substituting only the decision replay,
// which no test here opens.
const component = componentLoader({
  'DecisionReplay.svelte': 'data:text/javascript,export default function DecisionReplay() {}',
});
const GameReview = await component('GameReview.svelte');
const ReviewTrend = await component('ReviewTrend.svelte');

const frame = { view: {}, choices: [], lines: [], command: 0, played: { kind: 'discard', tile: '7m' } };
const decision = (values = {}) => ({
  hand: 0, index: 0, turn: 3, played: 'discards the 7 characters', played_kind: 'discard', played_tile: '7m',
  advised: 'discards the 1 characters', advised_kind: 'discard', advised_tile: '1m', agreed: false, forced: false,
  played_weight: 0.02, preferred_weight: 0.81, dora_types: [], weighted: [], frame,
  judgement: { stage: 'move', played: 0.02, preferred: 0.81, gap: 0.79, flagged: true }, ...values,
});
const hand = (index, values = {}) => ({ index, name: ['East 1', 'East 2', 'East 3'][index], counters: 0, seat: 'east',
  frames: [frame, frame], outcome: 'East wins by self-draw', status: 'waiting', decisions: null, ...values });
// A finished match's review unless `over` says the match is under way.
const review = (values = {}) => ({
  id: '7:120', hands: [hand(0), hand(1), hand(2)], busy: false, failure: '', held: false, slow: false, estimate: null,
  complete: false, requested: true, positionMs: null, recorded: false, closed: false, over: true,
  onChange: () => {}, review: () => {}, resume: () => {}, stop: () => {}, ...values,
});
const page = (props = {}) => render(GameReview, { props: { trainedAvailable: true, ...props } }).body;
/** What a button marked with this attribute says. */
const buttonSaid = (html, attribute) => text(new RegExp(`<button[^>]*\\b${attribute}\\b[^>]*>([^]*?)</button>`).exec(html)?.[1] ?? '');

test('a review under way leads with the flagged decisions found so far, and hides the answer', () => {
  const done = hand(0, { status: 'done', decisions: [decision(), decision({ index: 1, agreed: true, judgement: null })] });
  const html = page({ review: review({ busy: true, estimate: 95,
    hands: [done, hand(1, { status: 'reviewing', done: 1 }), hand(2)] }) });
  assert.match(html, /Your game, looked at again/);
  // The rule a decision is flagged by is the Flagged filter's title, out of
  // the way until asked for; no other filter has a title.
  const titles = [...html.matchAll(/<button[^>]*\btitle="([^"]*)"[^>]*\bdata-filter-option="(\w+)"/g)].map(([, title, option]) => [option, title]);
  assert.deepEqual(titles, [['flagged', "Your move under 5%, Trained AI's choice 60% or more"]]);
  assert.match(html, /role="status"[^>]*>Trained AI is reviewing East 2\./);
  assert.match(html, /3 of 6 decisions · the rest would take about 2 minutes/);
  assert.match(html, /data-stop-review/);
  assert.match(html, /East 1 · your decision 1/);
  assert.match(html, /You: discards the 7 characters/);
  assert.match(html, /Trained AI gave this move 2\.0%, and its own choice 81\.0%\./);
  assert.match(html, /aria-label="Replay and choose again: East 1, your decision 1"/, 'each button says which decision it opens');
  // The line that says where a decision went is there from the start, silent
  // and out of the layout, so a screen reader hears it when it speaks.
  assert.match(html, /<p class="moved[^"]*visually-hidden[^"]*" data-moved="" role="status">(<!---->)?<\/p>/);
  assert.doesNotMatch(html, /discards the 1 characters/, "Trained AI's own choice waits for a retry");
  assert.match(html, /1 of 2 matched/);
  assert.match(html, /Reviewing, 1 of 2/);
  assert.match(html, /2 decisions, not reviewed yet/);
  // The list of decisions and the list of hands are drawn without markers,
  // so each says it is a list in so many words: Safari does not tell
  // VoiceOver that a list whose markers are taken away is one.
  assert.match(html, /<ol class="flagged[^"]*"[^>]*\brole="list"/);
  assert.match(html, /<details class="hands[^"]*"[^>]*>[^]*?<ol[^>]*\brole="list"/);
  // So far, and not yet in the trend. The note is an element of its own, so
  // the space is looked for past any tags rather than by stripping them.
  assert.match(html, /50%(?:<[^>]*>)*\s+(?:<[^>]*>)*so far/, 'a space between the figure and its note');
});

test('a repeated hand goes by its name with its honba wherever the review names it', () => {
  const repeat = hand(1, { name: 'East 1 · 1 honba', counters: 1, status: 'done', decisions: [decision({ hand: 1 })] });
  const html = page({ review: review({ hands: [hand(0), repeat, hand(2)] }) });
  assert.match(html, /East 1 · 1 honba · your decision 1/);
  assert.match(html, /aria-label="Replay and choose again: East 1 · 1 honba, your decision 1"/);
  assert.match(html, /<span class="name[^"]*">East 1 · 1 honba<\/span>/, 'the list of hands does not add the honba again');
});

test('a flagged riichi tile says it is judged as the tile once riichi was declared', () => {
  const riichi = decision({ played_kind: 'riichi', advised_kind: 'riichi', played: 'declares riichi on the 7 characters',
    judgement: { stage: 'tile', played: 0.01, preferred: 0.9, gap: 0.89, flagged: true } });
  const html = page({ review: review({ hands: [hand(0, { status: 'done', decisions: [riichi] }), hand(1), hand(2)] }) });
  assert.match(html, /Trained AI gave this tile, once riichi was declared, 1\.0%, and its own choice 90\.0%\./);
});

test('a finished review with nothing flagged says so, and offers no more work', () => {
  const agreed = decision({ agreed: true, judgement: null });
  const html = page({ review: review({ complete: true,
    hands: [0, 1, 2].map(index => hand(index, { status: 'done', decisions: [agreed, { ...agreed, index: 1 }] })) }) });
  assert.match(html, /Every hand is reviewed: 6 decisions\./);
  assert.match(html, /No decision departed that far from Trained AI\./);
  assert.doesNotMatch(html, /data-continue|data-stop-review|data-review-hand/);
  // A finished game has no hand just played to lead with.
  assert.doesNotMatch(html, /data-latest-hand/);
  assert.equal(buttonSaid(html, 'data-close-review'), 'Back to the standings');
});

test('a review of the game so far leads with the hand just played, says so far, and goes back to the result', () => {
  const done = index => hand(index, { status: 'done', decisions: [decision({ hand: index }), decision({ hand: index, index: 1, agreed: true, judgement: null })] });
  /** The row of the hand just played: which hand, and what it says. */
  const lead = html => {
    const row = /<div class="latest[^"]*"[^>]*\bdata-latest-hand="(\d+)"[^>]*>([^]*?)<\/div>/.exec(html);
    return row && { hand: Number(row[1]), markup: row[2], said: text(row[2]) };
  };

  // Under way: the hand just played leads, and is asked about.
  const under = page({ review: review({ over: false, busy: true, hands: [done(0), done(1), hand(2, { status: 'reviewing', done: 1 })] }) });
  assert.match(under, /<h2[^>]*>Your game so far, looked at again<\/h2>/);
  assert.equal(buttonSaid(under, 'data-close-review'), 'Back to the result');
  assert.match(under, /role="status"[^>]*>Trained AI is reviewing East 3\./);
  const leading = lead(under);
  assert.equal(leading?.hand, 2, 'the hand just played leads');
  assert.equal(leading.said, 'The hand just played: East 3 East wins by self-draw Reviewing, 1 of 2');
  assert.doesNotMatch(under, /data-replay-latest/, 'nothing to replay before it is reviewed');
  const at = marker => under.indexOf(marker);
  assert.ok(at('data-close-review') >= 0 && at('data-close-review') < at('data-latest-hand') && at('data-latest-hand') < at('class="progress'),
    'directly after the header');
  // The list's heading is still the review's only heading after its title.
  assert.equal((under.match(/<h3/g) ?? []).length, 1);

  // Every hand so far reviewed: the hand just played is replayed from its own
  // row, and the game, not yet whole, waits to join the trend.
  const earlier = { id: 'earlier', network: playedNetwork().version, at: Date.UTC(2026, 9, 1, 12), matched: 60, decisions: 100 };
  const reviewed = page({ trend: [earlier], review: review({ over: false, complete: true, hands: [done(0), done(1), done(2)] }) });
  assert.match(lead(reviewed).markup, /<button[^>]*\bdata-replay-latest\b[^>]*aria-label="Replay this hand: East 3"[^>]*>Replay this hand<\/button>/);
  assert.match(reviewed, /role="status"[^>]*>Every hand so far is reviewed: 6 decisions\./);
  assert.doesNotMatch(reviewed, /data-continue|data-stop-review/);
  assert.match(reviewed, /50%(?:<[^>]*>)*\s+(?:<[^>]*>)*so far/, 'the figure is the game so far');
  assert.match(reviewed, /This game joins your trend once the game is over and every hand is reviewed\./);
  assert.doesNotMatch(reviewed, /points against/);

  // Not yet asked for, it offers the game so far; on a slow device that held
  // after a hand, every hand so far.
  const unasked = page({ review: review({ over: false, requested: false, hands: [done(0), done(1), hand(2)] }) });
  assert.equal(buttonSaid(unasked, 'data-continue'), 'Review the game so far');
  const held = page({ review: review({ over: false, held: true, slow: true, positionMs: 800, estimate: 20, hands: [hand(0), hand(1), done(2)] }) });
  assert.equal(buttonSaid(held, 'data-next-hand'), 'Review the next hand');
  assert.equal(buttonSaid(held, 'data-rest'), 'Review every hand so far');
  // With no decision made yet, the list of every decision says so.
  const none = page({ filter: 'every', review: review({ over: false, complete: true,
    hands: [hand(0, { frames: [], status: 'done', decisions: [] })] }) });
  assert.match(none, /You have made no decisions so far\./);
});


test('the list opens on the flagged decisions, and its filter lists those that differ or every one, saying how many each holds', () => {
  const near = decision({ index: 1, advised_tile: '9p', advised: 'discards the 9 circles', played_weight: 0.2, preferred_weight: 0.7,
    judgement: { stage: 'move', played: 0.2, preferred: 0.7, gap: 0.5, flagged: false } });
  const agreed = decision({ index: 2, agreed: true, judgement: null });
  const forced = decision({ index: 3, agreed: true, forced: true, judgement: null });
  const done = hand(0, { status: 'done', frames: [frame, frame, frame, frame], decisions: [decision(), near, agreed, forced] });
  const reviewed = review({ hands: [done, hand(1), hand(2)] });
  const items = html => [...html.matchAll(/<li[^>]*data-listed="([^"]+)"([^>]*)>([^]*?)<\/li>/g)]
    .map(([, key, attributes, inner]) => ({ key, flagged: /data-flagged=/.test(attributes), said: text(inner) }));
  const options = html => [...html.matchAll(/aria-pressed="(true|false)" data-filter-option="(\w+)">([^]*?)<\/button>/g)]
    .map(([, pressed, option, label]) => [option, pressed === 'true', text(label)]);

  const opening = page({ review: reviewed });
  assert.match(opening, /role="group" aria-label="Which decisions to list"/);
  assert.deepEqual(options(opening), [['flagged', true, 'Flagged (1)'], ['differed', false, 'Trained AI chose differently (2)'],
    ['every', false, 'Every decision (4)']]);
  // What a choice of filter shows is said from a line in the page from the start.
  assert.match(opening, /<p class="visually-hidden[^"]*" role="status" data-filter-said=""><\/p>/);
  assert.match(opening, /<h3[^>]*>Where you departed most from Trained AI<\/h3>/);
  assert.deepEqual(items(opening).map(item => [item.key, item.flagged]), [['0:0', true]]);

  // Those that differ: the flagged one marked as such, and replayed to choose
  // again; the other names Trained AI's choice and is replayed to look at.
  const different = page({ review: reviewed, filter: 'differed' });
  assert.match(different, /<h3[^>]*>Where Trained AI chose differently<\/h3>/);
  assert.deepEqual(options(different).map(([option, pressed]) => [option, pressed]), [['flagged', false], ['differed', true], ['every', false]]);
  const [flaggedItem, nearItem] = items(different);
  assert.deepEqual([flaggedItem.key, nearItem.key], ['0:0', '0:1'], 'the widest gap first');
  assert.match(flaggedItem.said, /^East 1 · your decision 1 · Flagged You: discards the 7 characters Trained AI gave this move 2\.0%, and its own choice 81\.0%\. Replay and choose again$/);
  assert.doesNotMatch(flaggedItem.said, /discards the 1 characters/, "a flagged decision's answer waits for its retry");
  assert.equal(nearItem.said, 'East 1 · your decision 2 You: discards the 7 characters Trained AI: discards the 9 circles '
    + 'Trained AI gave this move 20.0%, and its own choice 70.0%. Replay');
  assert.match(different, /data-open="0:1" aria-label="Replay: East 1, your decision 2"/);

  // Every decision, in the order played, each saying how it went.
  const every = page({ review: reviewed, filter: 'every' });
  assert.match(every, /<h3[^>]*>Every decision you made<\/h3>/);
  assert.deepEqual(items(every).map(item => item.key), ['0:0', '0:1', '0:2', '0:3']);
  assert.match(items(every)[2].said, /You: discards the 7 characters Trained AI chose the same\. Replay$/);
  assert.match(items(every)[3].said, /It was the only move\. Replay$/);

  // A reviewed hand can be replayed whole; one not yet reviewed, reviewed.
  assert.match(opening, /data-replay-hand="0" aria-label="Replay this hand: East 1"/);
  assert.match(opening, /data-review-hand="1" aria-label="Review this hand: East 2"/);
});

test('a slow device that held after a hand offers the next hand or the rest, with how long the rest takes', () => {
  const html = page({ review: review({ held: true, slow: true, positionMs: 800, estimate: 200,
    hands: [hand(0, { status: 'done', decisions: [decision()] }), hand(1), hand(2)] }) });
  assert.match(html, /Reviewed 1 of 3 hands\. A decision takes about 0\.8 seconds on this device\./);
  assert.match(html, /the rest would take about 3 minutes/);
  assert.match(html, /data-next-hand/);
  assert.match(html, /Review the rest of the game/);
  assert.match(html, /<details class="hands[^"]*" open/, 'the hands are open to choose from');
  assert.match(html, /data-review-hand="1"[^>]*aria-label="Review this hand: East 2"/);
});

test('without the network the review waits and says so, and is not offered hand by hand', () => {
  const html = page({ trainedAvailable: false, review: review({ requested: false }) });
  assert.match(html, /The review needs Trained AI, which cannot be reached right now\. It starts as soon as Trained AI can be reached\./);
  assert.doesNotMatch(html, /data-review-hand|data-continue/);
});

test('the trend tile states the change against earlier games without judging it, with a table of the games', () => {
  const network = playedNetwork().version;
  const games = Array.from({ length: 4 }, (_, index) => ({ id: `g${index}`, network, at: Date.UTC(2026, 9, 10 + index, 12), matched: 60, decisions: 100 }));
  const figure = { matched: 65, decisions: 100, share: 0.65 };
  const html = render(ReviewTrend, { props: { figure, complete: true, current: 'g3',
    games: [...games.slice(0, 3), { ...games[3], matched: 65 }] } }).body;
  assert.match(html, new RegExp(`>Matched ${trainedName('Trained AI').replaceAll('.', '\\.')}<`), 'it names the version that judged');
  assert.doesNotMatch(html, /Not counted here/);
  assert.match(html, />65%</);
  assert.match(html, /65 of 100 decisions\. Moves with only one choice are left out\./);
  assert.match(html, /\+5 points against your previous 3 games \(60%\)/);
  assert.match(html, /role="img" aria-label="Matched Trained AI in your last 4 reviewed games, on a scale of 0 to 100%: from 60% to 65%"/);
  assert.equal((html.match(/<circle class="now/g) ?? []).length, 1, 'only this game is marked');
  assert.equal((html.match(/<tr/g) ?? []).length, 5, 'a row a game and the header');
  assert.match(html, /\(this game\)/);
  assert.doesNotMatch(html, /\b(up|down|good|bad|better|worse)\b/i, 'the change is not judged');
  // Under way: the figure so far, and no change until the game is whole.
  const partial = render(ReviewTrend, { props: { figure, complete: false, current: 'g9', games: games.slice(0, 2) } }).body;
  assert.match(partial, /so far/);
  assert.match(partial, /This game joins your trend once every hand is reviewed\./);
  assert.doesNotMatch(partial, /points against|class="now/);
  // A game reviewed as it is played joins once it is over, too.
  const playing = render(ReviewTrend, { props: { figure, complete: false, over: false, current: 'g9', games: games.slice(0, 2) } }).body;
  assert.match(playing, /so far/);
  assert.match(playing, /This game joins your trend once the game is over and every hand is reviewed\./);
  assert.doesNotMatch(playing, /points against|class="now/);
  // A first game has nothing to compare with and no line to draw.
  const first = render(ReviewTrend, { props: { figure, complete: true, current: 'g0', games: [{ ...games[0], matched: 65 }] } }).body;
  assert.doesNotMatch(first, /points against|<svg/);
  assert.equal(TREND_WINDOW, 10);
});

test('the trend compares and shows only the games judged by the version of Trained AI this page plays', () => {
  const network = playedNetwork().version, other = `${network}-other`;
  const at = index => Date.UTC(2026, 9, 10 + index, 12);
  const games = [
    { id: 'g0', network, at: at(0), matched: 60, decisions: 100 },
    // Another version judged these, one of them the same game again.
    { id: 'g1', network: other, at: at(1), matched: 20, decisions: 100 },
    { id: 'g0', network: other, at: at(2), matched: 25, decisions: 100 },
    { id: 'g3', network, at: at(3), matched: 70, decisions: 100 },
  ];
  const html = render(ReviewTrend, { props: { figure: { matched: 70, decisions: 100, share: 0.7 }, complete: true, current: 'g3', games } }).body;
  assert.match(html, /\+10 points against your previous game \(60%\)/, 'only the same version is compared');
  assert.match(html, /aria-label="Matched Trained AI in your last 2 reviewed games, on a scale of 0 to 100%: from 60% to 70%"/);
  assert.equal((html.match(/<tr/g) ?? []).length, 3, 'a row for each of its games and the header');
  assert.match(html, /Not counted here: 2 games judged by another version of Trained AI\./);
  assert.doesNotMatch(html, /20%|25%/);
});
