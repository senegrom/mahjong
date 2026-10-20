import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { parse } from 'svelte/compiler';
import { render } from 'svelte/server';
import { meldWords } from '../src/lib/ui.js';
import { TILE_TYPES, tileShorthand, tileWords } from '../src/lib/tiles.js';
import { componentLoader } from './fixtures/load-component.js';

// Server-render the production components and the components they nest.
const component = componentLoader();
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

test('a hand tile is a toggle only when a press selects it for confirmation', async () => {
  const view = { phase: 'act', round: 'east', kyoku: 1, wall: 60, counters: 0, riichi_sticks: 0, dora_indicators: ['1z'], dora_types: [],
    dora: [], safe: [], shanten: 1, waits: [], waits_left: [], furiten: false, pending_discard: null, pending_from: null, outcome: null,
    seats: [seat('east', 0, { hand: ['1m', '2m'], drawn: '3m', turn: true }), seat('south', 1), seat('west', 2), seat('north', 3)] };
  const hand = props => html('app/PlayerHand.svelte', { view, engine: null, closed: false, hints: false, busy: false, blocked: false,
    discardChoices: ['1m', '2m', '3m'].map(tile => ({ kind: 'discard', tile })), picked: null, selected: null,
    canDiscard: () => true, selectTile() {}, syncHandFocus() {}, ...props });
  const pressed = markup => [...markup.matchAll(/<button class="tile [^>]*>/g)].map(([button]) => button.match(/aria-pressed="(\w+)"/)?.[1] ?? null);
  // Without confirmation a press discards: a plain button, even under the
  // arrow keys' marker.
  assert.deepEqual(pressed(await hand({ confirmDiscards: false, picked: 0 })), [null, null, null]);
  // With it, a press selects the tile, and the selection is its pressed state.
  assert.deepEqual(pressed(await hand({ confirmDiscards: true, selected: 1 })), ['false', 'true', 'false']);
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

test('an open kan is named as one, whichever table it is on', async () => {
  // The game's view calls it 'claimed-kan'; the physical and guided tables call it 'kan'.
  const melds = await html('Melds.svelte', { melds: [{ kind: 'claimed-kan', tiles: ['5s', '5s', '5s', '5s'], from: 'across', claimed_tile: '5s' }] });
  assert.match(melds, /role="group" aria-label="Open kan of 5 bamboo"/);
  assert.equal(meldWords({ kind: 'kan', tiles: ['5s', '5s', '5s', '5s'] }), 'Open kan of 5 bamboo');
  assert.equal(meldWords({ kind: 'extended-kan', tiles: ['5s', '5s', '5s', '5s'] }), 'Added kan of 5 bamboo');
});

test('riichi, calls, an offered discard and a selection are said by regions already in the page', async () => {
  // A screen reader listens to a live region only once it exists, and JAWS
  // and NVDA often miss one inserted along with its words. So each region is
  // rendered, empty, before there is anything to say.
  const said = markup => [...markup.matchAll(/<(?:span|p) class="visually-hidden[^"]*" role="status">([^<]*)<\/(?:span|p)>/g)]
    .map(match => match[1]);
  assert.deepEqual(said(await html('Seat.svelte', { seat: seat('south', 1) })), ['']);
  const riichi = await html('Seat.svelte', { seat: seat('south', 1, { riichi: true }) });
  assert.deepEqual(said(riichi), ['South declared riichi']);
  // The badge is for the eye; the region says the same and names the seat.
  assert.match(riichi, /<span class="called[^"]*" aria-hidden="true">Riichi<\/span>/);

  const view = extra => ({ phase: 'act', round: 'east', kyoku: 1, wall: 60, counters: 0, riichi_sticks: 0, dora_indicators: ['1z'],
    dora_types: [], dora: [], safe: [], shanten: 1, waits: [], waits_left: [], furiten: false, pending_discard: null, pending_from: null,
    outcome: null, seats: [seat('east', 0, { hand: ['5s'], drawn: null }), seat('south', 1), seat('west', 2), seat('north', 3)], ...extra });
  const choices = props => html('app/TurnChoices.svelte', { view: view(), shownDora: [], busy: false, thinking: false, failure: '',
    saveConflict: '', loadNote: '', pendingOpponent: null, myTurn: true, confirmDiscards: true, shortcuts: false, touch: false,
    selectedTile: null, callChoices: [], choose() {}, discard() {}, oncancel() {}, ...props });
  assert.deepEqual(said(await choices({})), ['']);
  const call = await choices({ view: view({ phase: 'call', pending_discard: '5s', pending_from: 'south' }), myTurn: false,
    callChoices: [{ kind: 'pon', tile: null }, { kind: 'pass', tile: null }] });
  assert.deepEqual(said(call), ['South discarded the 5 bamboo']);
  const selection = await choices({ selectedTile: '5s' });
  assert.deepEqual(said(selection), ['Selected: 5 bamboo']);
  // The panels still come and go for the eye, but not as regions of their own.
  assert.match(call, /class="call-stage/);
  assert.match(selection, /class="confirm-discard/);
  for (const markup of [call, selection]) assert.doesNotMatch(markup, /aria-live/);
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

test("a review's answer to a discard outlines the tile on offer in the discarder's row, and names it so", async () => {
  const thrown = (tile, extra = {}) => ({ tile, drawn: false, riichi: false, claimed: false, ...extra });
  // The tiles a row outlines, by name, past the markers a server render
  // leaves around a snippet and a tile.
  const outlined = markup => [...markup.matchAll(/<span class="offered[^"]*">(?:<!--[^>]*-->)*<span class="tile [^"]*"[^>]*aria-label="([^"]*)"/g)]
    .map(match => match[1]);
  // The row outlines the tile it is given, and its name and hover text say why.
  const row = await html('Discards.svelte', { marked: 1, dora: ['7p'], discards: [thrown('3p'), thrown('7p', { drawn: true })] });
  assert.deepEqual(outlined(row), ['7 circles, discarded from the draw, dora, the tile on offer']);
  assert.deepEqual(shownTiles(row).map(tile => tile.title), ['3 circles', '7 circles, discarded from the draw, dora, the tile on offer']);
  assert.deepEqual(outlined(await html('Discards.svelte', { discards: [thrown('3p'), thrown('7p')] })), []);

  // South let the 7 circles go and the player answers it. West threw one
  // earlier, which is not on offer.
  const view = seats => ({ phase: 'call', round: 'east', kyoku: 1, wall: 50, counters: 0, riichi_sticks: 0, dora_indicators: ['1z'],
    dora_types: [], dora: [], safe: [], shanten: 1, waits: [], waits_left: [], furiten: false, pending_discard: '7p', pending_from: 'south',
    outcome: null, seats });
  const offered = view([seat('east', 0, { hand: ['1m'], drawn: null }), seat('south', 1, { discards: [thrown('3p'), thrown('7p')] }),
    seat('west', 2, { discards: [thrown('7p')] }), seat('north', 3)]);
  const table = props => html('app/MatchTable.svelte', { hints: true, thinking: false, pendingOpponent: null, ...props });
  // A review's table outlines it in South's row alone, not in the inspection
  // of all discards, which lists every row as it lies.
  assert.deepEqual(outlined(await table({ view: offered, review: true })), ['7 circles, the tile on offer']);
  // Play's table names the tile on offer in the call itself, and outlines nothing.
  assert.deepEqual(outlined(await table({ view: offered })), []);
  // A quad offered to be robbed holds every copy of its tile, so no row ends
  // with it, and nothing is outlined.
  const quad = { kind: 'extended-kan', tiles: ['7p', '7p', '7p', '7p'], from: 'left', claimed_tile: '7p' };
  const robbing = view([seat('east', 0, { hand: ['5p', '6p'], drawn: null }), seat('south', 1, { melds: [quad], discards: [thrown('3p')] }),
    seat('west', 2), seat('north', 3)]);
  assert.deepEqual(outlined(await table({ view: robbing, review: true })), []);
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
  // A tile written out in place of its picture: on smaller faces each suit
  // by the letter it is written with under its number, and each honour by
  // its letters, as Tile shows them, so the guide never leaves a suit to
  // its colour alone.
  const list = letters => `${letters.slice(0, -1).join(', ')} and ${letters.at(-1)}`;
  const suits = ['1m', '1p', '1s'].map(tile => `${tileShorthand(tile).slice(1)} for ${tileWords(tile).split(' ')[1]}`);
  const [winds, dragons] = [TILE_TYPES.slice(27, 31), TILE_TYPES.slice(31)].map(tiles => tiles.map(tileShorthand));
  assert.match(guide, new RegExp(`no picture for this tile yet[\\s\\S]*${list(suits)}[\\s\\S]*${list(winds)} for the winds, ${list(dragons)} for the dragons`));
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

// A win as the engine describes it: its words, and the same as numbers.
const won = (seat, by, payments, extra = {}) => ({ seat, by, from: by === 'discard' ? 'west' : null, hand: [], melds: [],
  winning_tile: '5p', yaku: [], han: 5, dora: 0, dora_types: [], dora_indicators: [], ura_indicators: [], ura_dora: 0, fu: 30,
  limit: 'mangan', bets: 0, payments: { from_dealer: 0, from_each_other: 0, ...payments }, ...extra });
const ended = (wins, line) => ({ kind: wins.length ? 'win' : 'draw', line, wins, tenpai: [], changes: [0, 0, 0, 0] });
const sheet = (outcome, props = {}) => html('ScoreScreen.svelte', { outcome, onnext() {},
  seats: ['east', 'south', 'west', 'north'].map((wind, player) => seat(wind, player)), ...props });
const hero = markup => ({ label: markup.match(/class="hero-score[^"]*" role="img" aria-label="([^"]*)"/)?.[1],
  value: markup.match(/class="hero-score[\s\S]*?<b[^>]*>([^<]*)<\/b>/)?.[1] });

test('the result headline says what the hand is worth, to the eye and by name', async () => {
  for (const [win, value] of [
    // A self-draw is paid by all three, so its first payment is only part of it.
    [won('south', 'self-draw', { total: 8000, from_dealer: 4000, from_each_other: 2000 },
      { payment: '4000 from the dealer and 2000 from the others' }), '2,000 / 4,000'],
    [won('east', 'self-draw', { total: 12000, from_each_other: 4000 }, { payment: '4000 from each' }), '4,000 all'],
    [won('south', 'discard', { total: 8000 }, { payment: '8000 from the discarder' }), '8,000'],
  ]) {
    const shown = hero(await sheet(ended([win], 'A win')));
    assert.equal(shown.value, value, win.payment);
    assert.equal(shown.label, `${win.seat === 'east' ? 'East' : 'South'} ${win.by === 'self-draw' ? 'Tsumo' : 'Ron'}, mangan, ${value}`);
  }
});

test('Watch calls its first seat the followed agent, and its draw keeps the riichi bets note', async () => {
  // AgentWatch hands both result screens these: the first seat is the agent
  // being followed, and the bets are the table's.
  const draw = await sheet(ended([], 'Exhaustive draw'), { playerLabel: 'Followed agent', bets: 2 });
  assert.match(draw, /<th scope="row"[^>]*>Followed agent<\/th>/);
  assert.match(draw, /2 riichi bets stay on the table for the next winner/);
  const standings = await html('Standings.svelte', { playerLabel: 'Followed agent', onagain() {}, standings: [
    { player: 1, tied: false, place: 1, seat: 'south', score: 40000, uma: 15000, total: 25000, you: false },
    { player: 0, tied: false, place: 2, seat: 'east', score: 30000, uma: 5000, total: 5000, you: true }] });
  assert.match(standings, /Followed agent finished 2nd/);
  assert.match(standings, /<td[^>]*>Followed agent<\/td>/);
  for (const markup of [draw, standings]) assert.doesNotMatch(markup, /\bYou\b/);
  // Play keeps calling it you.
  assert.match(await sheet(ended([], 'Exhaustive draw')), /<th scope="row"[^>]*>You<\/th>/);
});

test("every result offers Trained AI's review of the game: of the game so far before the last hand, at every size, and of the whole game at the final hand, where the standings it covers on a phone offer it too", async () => {
  const draw = ended([], 'Exhaustive draw');
  const final = await sheet(draw, { finalHand: true, onreview() {}, onlog() {}, ongame() {}, ongamereview: () => true });
  // Shown on a phone only, beside the game's file, and "Review game" where
  // the labels are short.
  assert.match(final, /<button class="quiet whole-game[^"]*" data-review-game[^>]*>Review <span class="whole[^"]*">whole<\/span> game<\/button>/);
  // The hand's review stays first, "Review hand" where the labels are short.
  assert.match(final, /<button class="quiet[^"]*">(<!--[^>]*-->)?Review <span class="qualifier[^"]*">final<\/span> hand(<!--[^>]*-->)?<\/button>/);
  assert.ok(final.indexOf('>Review <span class="qualifier') < final.indexOf('data-review-game'), "the hand's review stays first");

  // Before the last hand nothing else offers the review of the game so far,
  // so the result offers it at every size, a plain button and not one of the
  // final hand's, which only a phone shows; "Review game" where the labels
  // are short. It follows the hand's own buttons, the hand's review still
  // first, which reads "View table" where the labels are short.
  const earlier = await sheet(draw, { onreview() {}, onlog() {}, ongame() {}, ongamereview: () => true });
  const reviews = [...earlier.matchAll(/<button[^>]*\bdata-review-game\b[^>]*>[^]*?<\/button>/g)].map(([button]) => button);
  assert.equal(reviews.length, 1);
  assert.match(reviews[0], /^<button class="quiet[^"]*"[^>]*\btitle="[^"]+"[^>]*>(<!--[^>]*-->)?Review game <span class="so-far[^"]*">so far<\/span>(<!--[^>]*-->)?<\/button>$/);
  assert.doesNotMatch(reviews[0], /whole-game/);
  assert.match(earlier, /<button class="quiet[^"]*">(<!--[^>]*-->)?View table <span class="qualifier[^"]*">\/ my hand<\/span>(<!--[^>]*-->)?<\/button>/);
  const at = marker => earlier.indexOf(marker);
  assert.ok(at('>View table') >= 0 && at('>View table') < at('data-review-game') && at('data-review-game') < at('data-save-game'),
    "the hand's review, then the game's, then the game's file");

  // Not in Watch, which reviews no game, at any hand.
  assert.doesNotMatch(await sheet(draw, { finalHand: true, onreview() {}, ongame() {} }), /data-review-game/);
  assert.doesNotMatch(await sheet(draw, { onreview() {}, ongame() {} }), /data-review-game/);
});

test('a quad offered to be robbed is announced as a kan, not as a discard', async () => {
  const quad = { kind: 'extended-kan', tiles: ['1p', '1p', '1p', '1p'], from: 'left', claimed_tile: '1p' };
  const view = { phase: 'call', round: 'east', kyoku: 1, wall: 40, counters: 0, riichi_sticks: 0, dora_indicators: ['1z'], dora_types: [],
    dora: [], safe: [], shanten: 0, waits: ['1p'], waits_left: [0], furiten: false, pending_discard: '1p', pending_from: 'south', outcome: null,
    seats: [seat('east', 0, { hand: ['2p', '3p'], drawn: null }), seat('south', 1, { melds: [quad] }), seat('west', 2), seat('north', 3)] };
  const choices = (offered, props = {}) => html('app/TurnChoices.svelte', { view: offered, shownDora: [], busy: false, thinking: false,
    failure: '', saveConflict: '', loadNote: '', pendingOpponent: null, myTurn: false, confirmDiscards: false, shortcuts: true,
    touch: false, selectedTile: null, callChoices: [{ kind: 'ron', tile: null }, { kind: 'pass', tile: null }],
    choose() {}, discard() {}, oncancel() {}, ...props });
  const robbing = await choices(view);
  assert.match(robbing, /<span class="call-kicker[^"]*">Kan<\/span>/);
  assert.match(robbing, /<strong[^>]*>South<\/strong><span[^>]*>declared a kan of the 1 circles<\/span>/);
  assert.match(robbing, /aria-label="Ron — rob the quad"/);
  assert.doesNotMatch(robbing, /discarded|on discard/);
  // An ordinary discard of the same tile, by a seat with no quad of it.
  const discarded = await choices({ ...view, seats: [view.seats[0], seat('south', 1), view.seats[2], view.seats[3]] });
  assert.match(discarded, /<span class="call-kicker[^"]*">Discard<\/span>/);
  assert.match(discarded, /<strong[^>]*>South<\/strong><span[^>]*>discarded the 1 circles<\/span>/);
  assert.match(discarded, /aria-label="Ron — win on discard"/);
});
