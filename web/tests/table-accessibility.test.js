import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';
import { meldWords } from '../src/lib/ui.js';

// Server-render the production components, compiling the components they
// nest in the same way. Only artwork URLs are substituted.
const compiled = new Map();
function load(file) {
  if (!compiled.has(file.href)) compiled.set(file.href, (async () => {
    const source = await readFile(file, 'utf8');
    const { js, warnings } = compile(source, { filename: fileURLToPath(file), generate: 'server' });
    assert.deepEqual(warnings.map(({ code, message }) => ({ code, message })), []);
    let code = js.code.replace(/import (\w+) from '[^']+\.webp';/g, "const $1 = 'artwork.webp';");
    for (const specifier of new Set([...code.matchAll(/from (['"])([^'"]+)\1/g)].map(match => match[2]))) {
      const target = !specifier.startsWith('.') ? import.meta.resolve(specifier)
        : specifier.endsWith('.svelte') ? await load(new URL(specifier, file)) : new URL(specifier, file).href;
      code = code.replaceAll(`'${specifier}'`, JSON.stringify(target)).replaceAll(`"${specifier}"`, JSON.stringify(target));
    }
    return `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`;
  })());
  return compiled.get(file.href);
}
const component = async path => (await import(await load(new URL(`../src/lib/${path}`, import.meta.url)))).default;
const html = async (path, props) => render(await component(path), { props }).body;
const seat = (wind, player, extra = {}) => ({ seat: wind, player, hand_size: 13, melds: [], discards: [], score: 25000,
  riichi: false, turn: false, controller: player ? 'club' : undefined, ...extra });

test('only a real toggle announces a pressed state', async () => {
  assert.doesNotMatch(await html('Tile.svelte', { tile: '1m', onclick() {} }), /aria-pressed/);
  assert.doesNotMatch(await html('Tile.svelte', { tile: '1m', onclick() {}, selected: true }), /aria-pressed/,
    'a suggested tile that plays a move is not a pressed toggle');
  assert.match(await html('HandTile.svelte', { tile: '1m', onclick() {}, toggle: true, selected: true }), /aria-pressed="true"/);
  assert.match(await html('HandTile.svelte', { tile: '1m', onclick() {}, toggle: true }), /aria-pressed="false"/);
});

test('tile entry names tiles in words and its buttons are plain buttons', async () => {
  const body = await html('TileEntry.svelte', { label: 'Your starting hand', tiles: ['1m'], onchange() {}, expanded: true });
  assert.match(body, /aria-label="Remove 1 characters"/);
  assert.match(body, /aria-label="Add 9 bamboo to Your starting hand"/);
  assert.match(body, /aria-label="Add red dragon to Your starting hand"/);
  assert.doesNotMatch(body, /Remove 1m|Add 9s /);
  assert.doesNotMatch(body, /aria-pressed/);
  assert.match(body, /class="palette[^"]*" role="group" aria-label="Your starting hand"/);
});

test('labelled containers carry a role, so screen readers read their names', async () => {
  assert.match(await html('Discards.svelte', { discards: [{ tile: '3p', riichi: false, claimed: false, drawn: false }] }),
    /role="group" aria-label="discards"/);
  const melds = await html('Melds.svelte', { melds: [
    { kind: 'pon', tiles: ['5s', '5s', '5s'], from: 'left', claimed_tile: '5s' },
    { kind: 'chii', tiles: ['4m', '5m', '6m'], from: 'left', claimed_tile: '5m' },
  ] });
  assert.match(melds, /role="group" aria-label="Pon of 5 bamboo"/);
  assert.match(melds, /role="group" aria-label="Chii of 4–5–6 characters"/);
  const dealer = await html('Seat.svelte', { seat: seat('south', 1), dealer: true });
  assert.match(dealer, /<div class="held[^"]*" role="img" aria-label="13 tiles in hand"/);
  // The badge's own text is read; a label on a plain span would be ignored.
  assert.match(dealer, /<span class="dealer-badge[^"]*">Dealer<\/span>/);
  assert.equal(meldWords({ kind: 'concealed-kan', tiles: ['7z', '7z', '7z', '7z'] }), 'Concealed kan of red dragon');
});

test('the table inspection names each opponent once: position, wind, score', async () => {
  const view = { round: 'east', kyoku: 1, wall: 70, counters: 0, riichi_sticks: 0, dora_indicators: ['1z'], dora_types: [], outcome: null,
    seats: [seat('east', 0), seat('south', 1), seat('west', 2), seat('north', 3)] };
  const body = await html('app/MatchTable.svelte', { view, hints: true, thinking: false, pendingOpponent: null });
  const headings = [...body.matchAll(/<h3[^>]*>([^<]*)<\/h3>/g)].map(match => match[1]);
  assert.deepEqual(headings, ['You · East · 25,000', 'Right · South · 25,000', 'Opposite · West · 25,000', 'Left · North · 25,000']);
  assert.match(body, /class="centre[^"]*" role="group" aria-label="the table"/);
  assert.match(body, /class="indicators[^"]*" role="group" aria-label="dora indicators"/);
});
