import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';
import { TREND_WINDOW } from '../src/lib/review-trend.js';

// Render the production components, substituting only the artwork URL and,
// where a test does not open one, the decision replay.
async function component(name, imports = {}) {
  const file = new URL(`../src/lib/${name}.svelte`, import.meta.url);
  const source = await readFile(file, 'utf8');
  const { js, warnings } = compile(source, { filename: file.pathname, generate: 'server' });
  assert.deepEqual(warnings, []);
  const code = js.code
    .replace(/import dragonUrl from '[^']+';/, "const dragonUrl = 'white-dragon.webp';")
    .replace(/from (['"])([^'"]+)\1/g, (_, _quote, path) =>
      `from ${JSON.stringify(imports[path] ?? (path.startsWith('.') ? new URL(path, file).href : import.meta.resolve(path)))}`);
  return `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`;
}
const tile = await component('Tile');
const trend = await component('ReviewTrend');
const replay = 'data:text/javascript,export default function DecisionReplay() {}';
const { default: GameReview } = await import(await component('GameReview', {
  './Tile.svelte': tile, './ReviewTrend.svelte': trend, './DecisionReplay.svelte': replay,
}));
const { default: ReviewTrend } = await import(trend);

const frame = { view: {}, choices: [], lines: [], command: 0, played: { kind: 'discard', tile: '7m' } };
const decision = (values = {}) => ({
  hand: 0, index: 0, turn: 3, played: 'discards the 7 characters', played_kind: 'discard', played_tile: '7m',
  advised: 'discards the 1 characters', advised_kind: 'discard', advised_tile: '1m', agreed: false, forced: false,
  played_weight: 0.02, preferred_weight: 0.81, dora_types: [], weighted: [], frame,
  judgement: { stage: 'move', played: 0.02, preferred: 0.81, gap: 0.79, flagged: true }, ...values,
});
const hand = (index, values = {}) => ({ index, name: ['East 1', 'East 2', 'East 3'][index], counters: 0, seat: 'east',
  frames: [frame, frame], outcome: 'East wins by self-draw', status: 'waiting', decisions: null, ...values });
const review = (values = {}) => ({
  id: '7:120', hands: [hand(0), hand(1), hand(2)], busy: false, failure: '', held: false, slow: false, estimate: null,
  complete: false, requested: true, positionMs: null, recorded: false, closed: false,
  onChange: () => {}, review: () => {}, resume: () => {}, stop: () => {}, ...values,
});
const page = (props = {}) => render(GameReview, { props: { trainedAvailable: true, ...props } }).body;

test('a review under way leads with the flagged decisions found so far, and hides the answer', () => {
  const done = hand(0, { status: 'done', decisions: [decision(), decision({ index: 1, agreed: true, judgement: null })] });
  const html = page({ review: review({ busy: true, estimate: 95,
    hands: [done, hand(1, { status: 'reviewing', done: 1 }), hand(2)] }) });
  assert.match(html, /Your game, looked at again/);
  assert.match(html, /under 5% of Trained AI's weight while its own choice had 60% or more/);
  assert.match(html, /role="status"[^>]*>Trained AI is reviewing East 2\./);
  assert.match(html, /3 of 6 decisions · the rest would take about 2 minutes/);
  assert.match(html, /data-stop-review/);
  assert.match(html, /East 1 · your decision 1/);
  assert.match(html, /You: discards the 7 characters/);
  assert.match(html, /Trained AI gave this move 2\.0%/);
  assert.match(html, /Replay and choose again/);
  assert.doesNotMatch(html, /discards the 1 characters/, "Trained AI's own choice waits for a retry");
  assert.match(html, /1 of 2 matched/);
  assert.match(html, /Reviewing, 1 of 2/);
  assert.match(html, /2 decisions, not reviewed yet/);
  // So far, and not yet in the trend.
  assert.match(html.replace(/<[^>]*>/g, ''), /50% so far/, 'a space between the figure and its note');
});

test('a flagged riichi tile says it is judged as the tile once riichi was declared', () => {
  const riichi = decision({ played_kind: 'riichi', advised_kind: 'riichi', played: 'declares riichi on the 7 characters',
    judgement: { stage: 'tile', played: 0.01, preferred: 0.9, gap: 0.89, flagged: true } });
  const html = page({ review: review({ hands: [hand(0, { status: 'done', decisions: [riichi] }), hand(1), hand(2)] }) });
  assert.match(html, /Trained AI gave this tile, once riichi was declared, 1\.0%/);
});

test('a finished review with nothing flagged says so, and offers no more work', () => {
  const agreed = decision({ agreed: true, judgement: null });
  const html = page({ review: review({ complete: true,
    hands: [0, 1, 2].map(index => hand(index, { status: 'done', decisions: [agreed, { ...agreed, index: 1 }] })) }) });
  assert.match(html, /Every hand is reviewed: 6 decisions\./);
  assert.match(html, /No decision departed that far from Trained AI\./);
  assert.doesNotMatch(html, /data-continue|data-stop-review|data-review-hand/);
});

test('a slow device that held after a hand offers the next hand or the rest, with how long the rest takes', () => {
  const html = page({ review: review({ held: true, slow: true, positionMs: 800, estimate: 200,
    hands: [hand(0, { status: 'done', decisions: [decision()] }), hand(1), hand(2)] }) });
  assert.match(html, /Reviewed 1 of 3 hands\. A decision takes about 0\.8 seconds on this device\./);
  assert.match(html, /the rest would take about 3 minutes/);
  assert.match(html, /data-next-hand/);
  assert.match(html, /Review the rest of the game/);
  assert.match(html, /<details class="hands[^"]*" open/, 'the hands are open to choose from');
  assert.match(html, /data-review-hand="1"/);
});

test('without the network the review waits and says so, and is not offered hand by hand', () => {
  const html = page({ trainedAvailable: false, review: review({ requested: false }) });
  assert.match(html, /The review needs Trained AI, which cannot be reached right now\. It starts as soon as Trained AI can be reached\./);
  assert.doesNotMatch(html, /data-review-hand|data-continue/);
});

test('the trend tile states the change against earlier games without judging it, with a table of the games', () => {
  const games = Array.from({ length: 4 }, (_, index) => ({ id: `g${index}`, at: Date.UTC(2026, 9, 10 + index, 12), matched: 60, decisions: 100 }));
  const figure = { matched: 65, decisions: 100, share: 0.65 };
  const html = render(ReviewTrend, { props: { figure, complete: true, current: 'g3',
    games: [...games.slice(0, 3), { ...games[3], matched: 65 }] } }).body;
  assert.match(html, /Matched Trained AI/);
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
  // A first game has nothing to compare with and no line to draw.
  const first = render(ReviewTrend, { props: { figure, complete: true, current: 'g0', games: [{ ...games[0], matched: 65 }] } }).body;
  assert.doesNotMatch(first, /points against|<svg/);
  assert.equal(TREND_WINDOW, 10);
});
