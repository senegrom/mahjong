import test from 'node:test';
import assert from 'node:assert/strict';
import { render } from 'svelte/server';
import { stepSaid } from '../src/lib/game-review.js';
import { componentLoader } from './fixtures/load-component.js';
import { text } from './fixtures/markup-text.js';

const ReviewStepBar = await componentLoader()('ReviewStepBar.svelte');

// A hand of East 4 as the review keeps it: at each of the player's decisions
// the table and the moves open there, named as the engine names them.
const seat = (wind, values = {}) => ({ seat: wind, turn: false, riichi: false, hand: [], drawn: null, melds: [], ...values });
const turn = { phase: 'act', wall: 36, pending_discard: null, pending_from: null,
  seats: [seat('north', { turn: true, drawn: '7z' }), seat('east'), seat('south'), seat('west')] };
const offer = { ...turn, phase: 'call', wall: 64, pending_discard: '7p', pending_from: 'south',
  seats: [seat('north'), seat('east'), seat('south', { turn: true }), seat('west')] };
const quad = { kind: 'extended-kan', tiles: ['7p', '7p', '7p', '7p'], from: 'left', claimed_tile: '7p' };
const robbed = { ...offer, seats: [seat('north'), seat('east'), seat('south', { turn: true, melds: [quad] }), seat('west')] };
const choice = (kind, tile, label, weight, conditionalWeight = null) => ({ kind, tile, label, weight, conditionalWeight });
const discards = [choice('discard', '7z', 'discards the red dragon', 0.02), choice('discard', '8s', 'discards the 8 bamboo', 0.955)];
const reaches = [choice('riichi', '3s', 'declares riichi on the 3 bamboo', 0.9, 0.02), choice('riichi', '6m', 'declares riichi on the 6 characters', 0.9, 0.9)];
const calls = [choice('pon', null, 'calls a triplet', 0.99), choice('pass', null, 'passes', 0.003), choice('ron', null, 'wins on the discard', 0.007)];
const frame = (view, choices, kind, tile = null) => ({ view, choices, played: { kind, tile }, lines: [] });
const hand = { index: 3, name: 'East 4', frames: [frame(offer, calls, 'pass'), frame(turn, discards, 'discard', '7z'),
  frame(turn, reaches, 'riichi', '3s'), frame(robbed, calls, 'pass'), frame(turn, discards, 'discard', '8s')] };
const decision = (step, advised, values = {}) => ({ played_kind: hand.frames[step].played.kind, played_tile: hand.frames[step].played.tile,
  advised_kind: advised.kind, advised_tile: advised.tile, advised: advised.label, agreed: false, forced: false,
  judgement: { flagged: false }, weighted: hand.frames[step].choices, ...values });

const bar = props => render(ReviewStepBar, { props: { hand, step: 1, decision: null, previous: null, next: null, onstep() {}, ...props } }).body;
/** The bar's Previous and Next: what each is named and shows, whether it is
 * disabled, and whether it waits, disabled to a screen reader and the eye
 * but focusable all the same. */
const buttons = markup => [...markup.matchAll(/<button[^>]*\bdata-(previous|next)=""[^>]*aria-label="([^"]*)"([^>]*)>([^<]*)<\/button>/g)]
  .map(([, which, label, rest, shown]) => ({ which, label, disabled: /(?:^|\s)disabled=""/.test(rest), waiting: /\baria-disabled="true"/.test(rest), shown }));
/** The bar's line of moves: its words, the tiles being pictures, and the
 * tiles it shows, by name. */
function line(html) {
  const markup = /<p class="moves[^"]*" data-move-line="">([^]*?)<\/p>/.exec(html)[1];
  return { said: text(markup), tiles: [...markup.matchAll(/class="tile [^"]*"[^>]*aria-label="([^"]*)"/g)].map(match => match[1]) };
}

test("the step bar gives the player's move beside Trained AI's with their weights, and the flag, as each kind of decision has them", () => {
  const flagged = decision(1, discards[1], { judgement: { flagged: true } });
  assert.deepEqual(line(bar({ decision: flagged })), { said: 'You 2.0% · Trained AI 95.5% · Flagged', tiles: ['red dragon', '8 bamboo'] });
  assert.match(bar({ decision: flagged }), /<strong class="flag[^"]*">Flagged<\/strong>/);
  assert.deepEqual(line(bar({ decision: decision(1, discards[1]) })), { said: 'You 2.0% · Trained AI 95.5%', tiles: ['red dragon', '8 bamboo'] });
  // Agreed, or the only move: nothing more to give.
  assert.deepEqual(line(bar({ step: 4, decision: decision(4, discards[1], { agreed: true, judgement: null }) })),
    { said: 'You 95.5% · Trained AI agreed', tiles: ['8 bamboo'] });
  assert.deepEqual(line(bar({ step: 4, decision: decision(4, discards[1], { agreed: true, forced: true, judgement: null }) })),
    { said: 'You · Only move', tiles: ['8 bamboo'] });
  // An answer to another's discard says whose tile it was, and a robbed
  // quad that it was a kan; a move without a tile is its word.
  assert.deepEqual(line(bar({ step: 0, decision: decision(0, calls[0]) })),
    { said: "On South's : You Pass 0.3% · Trained AI Pon 99.0%", tiles: ['7 circles'] });
  assert.deepEqual(line(bar({ step: 3, decision: decision(3, calls[2]) })),
    { said: "On South's kan : You Pass 0.3% · Trained AI Ron 0.7%", tiles: ['7 circles'] });
  // Both declared riichi: the tiles are weighed given the declaration.
  assert.deepEqual(line(bar({ step: 2, decision: decision(2, reaches[1], { judgement: { flagged: true } }) })),
    { said: 'You Riichi 2.0% · Trained AI Riichi 90.0%, given riichi · Flagged', tiles: ['3 bamboo', '6 characters'] });
  // A hand not reviewed yet: the player's move, and Trained AI's to come.
  assert.deepEqual(line(bar({ decision: null })), { said: 'You · Trained AI …', tiles: ['red dragon'] });
  // "Trained" goes on the narrowest screens, where the line says "AI".
  assert.match(bar({ decision: flagged }), /<span class="long[^"]*">Trained <\/span>AI/);
});

test('the step bar says where it is, steps by the filter its buttons are named for, and says the step in one line from a status in the page', () => {
  const flagged = decision(1, discards[1], { judgement: { flagged: true } });
  const html = bar({ decision: flagged, previous: { hand: 3, step: 0 }, next: { hand: 3, step: 4 }, filter: 'flagged' });
  assert.match(html, /<p class="place[^"]*" data-step="">East 4 · <span class="long[^"]*">decision <\/span>2 of 5<\/p>/);
  assert.ok(html.includes(`role="status" data-step-said="">${stepSaid({ hand, step: 1, decision: flagged })}</p>`));
  assert.match(html, /<div class="steps[^"]*" role="group" aria-label="Steps" data-steps="">/);
  // Each button shows its word and is named for the decisions it steps to.
  assert.deepEqual(buttons(html), [{ which: 'previous', label: 'Previous flagged decision', disabled: false, waiting: false, shown: 'Previous' },
    { which: 'next', label: 'Next flagged decision', disabled: false, waiting: false, shown: 'Next' }]);
  assert.deepEqual(buttons(bar({ filter: 'differed', next: { hand: 3, step: 4 } })).map(({ label, disabled }) => [label, disabled]),
    [['Previous decision where Trained AI differed', true], ['Next decision where Trained AI differed', false]]);
  // Only at the ends is a button disabled.
  assert.deepEqual(buttons(bar({ previous: { hand: 3, step: 0 } })).map(({ label, disabled }) => [label, disabled]),
    [['Previous decision', false], ['Next decision', true]]);
});

test("with Trained AI's move kept back, the step bar gives the player's move without its weight, Show answer in Trained AI's place and the flag, and says the move is hidden", () => {
  const flagged = decision(1, discards[1], { judgement: { flagged: true } });
  const html = bar({ decision: flagged, moveHidden: true, previous: { hand: 3, step: 0 }, next: { hand: 3, step: 4 } });
  assert.deepEqual(line(html), { said: 'You · Trained AI ? Show answer · Flagged', tiles: ['red dragon'] });
  assert.match(html, /<button type="button" class="app-control reveal[^"]*" data-reveal="">Show answer<\/button>/);
  assert.match(html, /<span class="long[^"]*">Trained <\/span>AI \?/, '"AI ?" on the narrowest screens');
  assert.ok(html.includes(`role="status" data-step-said="">${stepSaid({ hand, step: 1, decision: flagged, hidden: true })}</p>`));
  // Nothing holds the player at a hidden decision: both ways on stay open.
  assert.deepEqual(buttons(html).map(({ disabled, waiting }) => [disabled, waiting]), [[false, false], [false, false]]);
  // Agreed, nothing says so; an answer to another's discard still says
  // whose tile it was, which the table shows.
  assert.deepEqual(line(bar({ step: 4, decision: decision(4, discards[1], { agreed: true, judgement: null }), moveHidden: true })),
    { said: 'You · Trained AI ? Show answer', tiles: ['8 bamboo'] });
  assert.deepEqual(line(bar({ step: 0, decision: decision(0, calls[0]), moveHidden: true })),
    { said: "On South's : You Pass · Trained AI ? Show answer", tiles: ['7 circles'] });
  // Both kept back: the one button shows the answer and the tiles with it.
  assert.equal(line(bar({ decision: flagged, moveHidden: true, tilesHidden: true })).said, 'You · Trained AI ? Show answer · Flagged');
});

test("with only the other players' tiles kept back, the step bar gives both moves and offers Show tiles, at an only move and a hand not reviewed too", () => {
  const flagged = decision(1, discards[1], { judgement: { flagged: true } });
  assert.deepEqual(line(bar({ decision: flagged, tilesHidden: true })),
    { said: 'You 2.0% · Trained AI 95.5% Show tiles · Flagged', tiles: ['red dragon', '8 bamboo'] });
  assert.match(bar({ decision: flagged, tilesHidden: true }), /<button type="button" class="app-control reveal[^"]*" data-reveal="">Show tiles<\/button>/);
  assert.equal(line(bar({ step: 4, decision: decision(4, discards[1], { agreed: true, forced: true, judgement: null }), tilesHidden: true })).said,
    'You · Only move Show tiles');
  assert.equal(line(bar({ decision: null, tilesHidden: true })).said, 'You · Trained AI … Show tiles');
  // Nothing kept back, nothing to show.
  assert.doesNotMatch(bar({ decision: flagged }), /data-reveal/);
});

test("while the settings keep anything back, the line of moves keeps the reveal's room with the button or without, so that a reveal leaves the bar as tall, and otherwise takes only its moves' room", () => {
  const flagged = decision(1, discards[1], { judgement: { flagged: true } });
  /** Whether the line keeps the room of the reveal's button. */
  const room = html => /<p class="moves[^"]*\bhiding\b[^"]*" data-move-line="">/.test(html);
  assert.equal(room(bar({ decision: flagged, moveHidden: true, tilesHidden: true, hiding: true })), true, 'kept back, the button there');
  assert.equal(room(bar({ decision: flagged, pick: { kind: 'discard', tile: '8s' }, hiding: true })), true, 'revealed, the button gone');
  assert.equal(room(bar({ decision: flagged })), false, 'nothing kept back at any decision');
});

test("a move the player picked before the answer was shown is given as Now, with its weight, between their move and Trained AI's, and said in the step's line", () => {
  const flagged = decision(1, discards[1], { judgement: { flagged: true } }), pick = { kind: 'discard', tile: '8s' };
  const html = bar({ decision: flagged, pick });
  assert.deepEqual(line(html), { said: 'You 2.0% · Now 95.5% · Trained AI 95.5% · Flagged', tiles: ['red dragon', '8 bamboo', '8 bamboo'] });
  assert.ok(html.includes(`role="status" data-step-said="">${stepSaid({ hand, step: 1, decision: flagged, pick })}</p>`));
  assert.match(stepSaid({ hand, step: 1, decision: flagged, pick }), / Now: discards the 8 bamboo, 95\.5%\. Flagged\.$/);
  // A move without a tile is its word, an answer to a discard too.
  assert.equal(line(bar({ step: 0, decision: decision(0, calls[0]), pick: { kind: 'pon', tile: null } })).said,
    "On South's : You Pass 0.3% · Now Pon 99.0% · Trained AI Pon 99.0%");
  assert.doesNotMatch(html, /data-reveal/, 'picking showed the answer');
});

test('Next, waiting for the answers of the hand on show, says it is disabled but is not disabled outright, so that it keeps its focus', () => {
  const html = bar({ previous: { hand: 3, step: 0 }, waiting: true, filter: 'flagged' });
  assert.deepEqual(buttons(html).map(({ which, label, disabled, waiting }) => ({ which, label, disabled, waiting })), [
    { which: 'previous', label: 'Previous flagged decision', disabled: false, waiting: false },
    { which: 'next', label: 'Next flagged decision', disabled: false, waiting: true }]);
});
