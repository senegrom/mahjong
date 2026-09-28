import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';
import { readBeliefs } from '../src/lib/beliefs.js';

// Render the production components, substituting only the artwork URL and the
// browser-only policy worker (SSR never executes the analysis effect).
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
const preference = await component('ReviewPreference');
const policy = 'data:text/javascript,export const analyzePolicy = () => { throw new Error("unexpected SSR inference"); };';
const { default: Review } = await import(await component('Review', {
  './Tile.svelte': tile, './ReviewPreference.svelte': preference, './policy.js': policy,
}));
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

test('the belief panel names honours as the page names tiles, and marks their dora', async () => {
  const agents = 'data:text/javascript,export const AGENTS = { full: "Trained" };';
  const { default: AgentWeights } = await import(await component('AgentWeights', { './Tile.svelte': tile, './agents.js': agents }));
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
