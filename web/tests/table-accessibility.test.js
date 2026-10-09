import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { compile, parse } from 'svelte/compiler';
import { render } from 'svelte/server';
import { meldWords } from '../src/lib/ui.js';
import { TILE_TYPES, tileShorthand } from '../src/lib/tiles.js';

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
  assert.match(dealer, /role="img" aria-label="Club opponent"/);
  assert.match(await html('HandTile.svelte', { tile: '1m', remaining: 2, showRemaining: true }),
    /class="copy-count[^"]*" role="img"[^>]*aria-label="2 unseen 1 characters remain"/);
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

// Every displayed tile in some markup: its classes, its accessible name and its hover text.
const shownTiles = markup => [...markup.matchAll(/<span class="(tile [^"]*)"[^>]*aria-label="([^"]*)" title="([^"]*)"/g)]
  .map(([, classes, label, title]) => ({ classes: classes.split(/\s+/), label, title }));
const marks = tile => ['from-draw', 'claimed'].filter(name => tile.classes.includes(name));

test('discard rows mark tiles thrown from the draw and claimed tiles, to the eye and by name', async () => {
  const row = await html('Discards.svelte', { dora: ['5p'], discards: [
    { tile: '3p', drawn: true, riichi: false, claimed: false },
    { tile: '4p', drawn: false, riichi: false, claimed: false },
    { tile: '5p', drawn: true, riichi: true, claimed: true },
    { tile: '6p', riichi: false, claimed: false },
    { tile: '7p', drawn: false, riichi: false, claimed: true },
  ] });
  const tiles = shownTiles(row);
  assert.deepEqual(tiles.map(tile => tile.label), ['3 circles, discarded from the draw', '4 circles',
    '5 circles, claimed, riichi declaration, discarded from the draw, dora', '6 circles', '7 circles, claimed']);
  // The pointer's hover text and the screen reader's name are the same words.
  for (const tile of tiles) assert.equal(tile.title, tile.label);
  assert.deepEqual(tiles.map(marks), [['from-draw'], [], ['from-draw', 'claimed'], [], ['claimed']]);
  for (const name of ['rotated', 'ringed']) assert.ok(tiles[2].classes.includes(name), name);
});

test('the table, the inspection of all discards and your own row all carry both marks', async () => {
  const thrown = { tile: '2s', drawn: true, riichi: false, claimed: false };
  const taken = { tile: '8m', drawn: false, riichi: false, claimed: true };
  const kept = { tile: '9p', drawn: false, riichi: false, claimed: false };
  const own = { tile: '4z', drawn: true, riichi: false, claimed: true };
  const view = { phase: 'call', round: 'east', kyoku: 1, wall: 60, counters: 0, riichi_sticks: 0, dora_indicators: ['1z'], dora_types: [],
    dora: [], safe: [], shanten: 1, waits: [], waits_left: [], furiten: false, pending_discard: null, pending_from: null, outcome: null,
    seats: [seat('east', 0, { hand: ['1m'], drawn: null, discards: [own] }), seat('south', 1, { discards: [thrown, taken, kept] }),
      seat('west', 2), seat('north', 3)] };
  // The opponent's row at the table, then the inspection's rows for you and for them.
  const table = shownTiles(await html('app/MatchTable.svelte', { view, hints: true, thinking: false, pendingOpponent: null }))
    .filter(tile => !tile.label.startsWith('east wind'));
  assert.deepEqual(table.map(tile => [tile.label, marks(tile)]), [
    ['2 bamboo, discarded from the draw', ['from-draw']], ['8 characters, claimed', ['claimed']], ['9 circles', []],
    ['north wind, claimed, discarded from the draw', ['from-draw', 'claimed']],
    ['2 bamboo, discarded from the draw', ['from-draw']], ['8 characters, claimed', ['claimed']], ['9 circles', []],
  ]);
  const hand = await html('app/PlayerHand.svelte', { view, engine: null, closed: false, hints: true, busy: false, blocked: false,
    discardChoices: [], picked: null, selected: null, canDiscard: () => false, selectTile() {}, syncHandFocus() {} });
  const discards = shownTiles(hand.slice(hand.indexOf('class="own-discards')));
  assert.deepEqual(discards.map(tile => [tile.label, marks(tile)]), [['north wind, claimed, discarded from the draw', ['from-draw', 'claimed']]]);
});

test('the tile guide explains both discard marks, each with its swatch', async () => {
  const settings = await html('app/AppSettings.svelte', { tileFace: 'classic', difficulty: 'club', opponents: ['club', 'club', 'club'],
    ready: true, busy: false, saveConflict: '', trainedAvailable: false,
    offline: { coreReady: true, aiReady: false, hasModel: false, phase: 'ready', progress: 0 },
    changeOpponents() {}, startFresh: () => true, configureTable() {}, downloadAi() {}, onfacechange() {},
    onconfirmationchange() {}, onshortcutschange() {} });
  const guide = settings.slice(settings.indexOf('class="guide'));
  for (const [swatch, words] of [['from-draw', /darker face[\s\S]*thrown straight from the draw \(tsumogiri\)/],
    ['claimed', /dark green border[\s\S]*claimed the tile for a call/]]) {
    assert.match(guide, new RegExp(`<dt[^>]*><span class="swatch ${swatch}[ "][^>]*></span>`), swatch);
    assert.match(guide, words);
  }
  // A claimed tile is bordered now, no longer faded.
  assert.doesNotMatch(guide, /see-through/);
  // A tile written out in place of its picture, with the letters an honour
  // is given on the smallest faces, as Tile shows them.
  const [winds, dragons] = [TILE_TYPES.slice(27, 31), TILE_TYPES.slice(31)].map(tiles => tiles.map(tileShorthand));
  const list = letters => `${letters.slice(0, -1).join(', ')} and ${letters.at(-1)}`;
  assert.match(guide, new RegExp(`no picture for this tile yet[\\s\\S]*${list(winds)} for the winds, ${list(dragons)} for the dragons`));
});

test('no generic element carries a name a screen reader would ignore', async () => {
  // ARIA gives a span, div or paragraph no name: its aria-label is dropped
  // unless the element also has a role. Check every component's markup.
  const generic = new Set(['div', 'span', 'p', 'b', 'strong', 'em', 'i', 'small', 'code']);
  const root = new URL('../src/', import.meta.url);
  const files = (await readdir(root, { recursive: true })).filter(name => name.endsWith('.svelte'));
  assert.ok(files.length > 20);
  const unnamed = [];
  const visit = (node, file) => {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'RegularElement' && generic.has(node.name)) {
      const names = node.attributes.filter(attribute => attribute.type === 'Attribute').map(attribute => attribute.name);
      if (names.includes('aria-label') && !names.includes('role')) unnamed.push(`${file}: <${node.name}> at ${node.start}`);
    }
    for (const [key, value] of Object.entries(node)) if (key !== 'parent') visit(value, file);
  };
  // A file URL reads either separator, so readdir's own paths resolve as they are.
  for (const file of files) visit(parse(await readFile(new URL(file, root), 'utf8'), { modern: true }).fragment, file);
  assert.deepEqual(unnamed, []);
});
