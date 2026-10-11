import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';

// Render the actual component. Node cannot import a .webp, so substitute only
// its URL; presentation and reactive prop logic remain the production code.
const file = new URL('../src/lib/Tile.svelte', import.meta.url);
const source = await readFile(file, 'utf8');
const compiled = compile(source, { filename: file.pathname, generate: 'server' });
const js = compiled.js.code
  .replace(/import dragonUrl from '[^']+';/, "const dragonUrl = 'white-dragon.webp';")
  .replace(/from (['"])([^'"]+)\1/g, (_, _quote, name) =>
    `from ${JSON.stringify(name.startsWith('.') ? new URL(name, file).href : import.meta.resolve(name))}`);
const { default: Tile } = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
const tile = (props) => render(Tile, { props }).body;

test('hidden dora cannot leak its identity through the ring or foil', () => {
  for (const onclick of [null, () => {}]) {
    const html = tile({ tile: '5z', dora: true, facedown: true, onclick });
    assert.doesNotMatch(html, /\bringed\b|class="foil\b|class="haku-dragon-reveal\b/);
    assert.match(html, /aria-label="face-down tile"/);
  }
});

test('unknown face-down tiles also suppress a stray dora prop', () => {
  assert.doesNotMatch(tile({ tile: null, dora: true }), /\bringed\b|class="foil\b|class="haku-dragon-reveal\b/);
});

test('face-up white-dragon dora still has its ring, artwork and shine', () => {
  const html = tile({ tile: '5z', dora: true });
  assert.match(html, /\bringed\b/);
  assert.match(html, /class="haku-dragon-reveal\b/);
  assert.match(html, /class="foil\b/);
  assert.match(html, /aria-label="white dragon, dora"/);
});

test('ordinary white dragons remain blank and other dora keep their own artwork', () => {
  assert.doesNotMatch(tile({ tile: '5z' }), /class="foil\b|class="haku-dragon-reveal\b/);
  const html = tile({ tile: '6z', dora: true });
  assert.match(html, /class="foil\b/);
  assert.doesNotMatch(html, /class="haku-dragon-reveal\b/);
});

test('newly drawn alone is a label, not a border', () => {
  const html = tile({ tile: '9s', drawn: true, onclick() {} });
  assert.match(html, /just drawn/);
  assert.doesNotMatch(html, /\bringed\b|data-readiness=/);
});

test('only legal actionable one-shanten and tenpai discards get readiness rings', () => {
  const gold = tile({ tile: '2z', discardShanten: 0, onclick() {} });
  assert.match(gold, /data-readiness="ready"/);
  assert.match(gold, /discard leaves a ready hand \(tenpai\)/);
  assert.match(gold, /--ring:\s*var\(--gold, #d8a12a\)/);
  const silver = tile({ tile: '7m', discardShanten: 1, onclick() {} });
  assert.match(silver, /data-readiness="one-away"/);
  assert.match(silver, /--ring:\s*#c5cbd3/);
  assert.match(silver, /discard leaves one tile from ready/);
  for (const discardShanten of [-1, 2, 3, null, undefined]) {
    assert.doesNotMatch(tile({ tile: '7m', discardShanten, onclick() {} }), /\bringed\b|data-readiness=/);
  }
});

test('disabled, unknown, face-down and display-only tiles cannot retain readiness hints', () => {
  for (const props of [{ disabled: true }, { facedown: true }, { tile: null }, { onclick: null }]) {
    for (const discardShanten of [0, 1]) {
      const html = tile({ tile: '5z', onclick() {}, discardShanten, ...props });
      assert.doesNotMatch(html, /\bringed\b|data-readiness=|discard leaves/);
    }
  }
});

test('readiness rings coexist with unchanged dora foil, dragon artwork, safety and selection', () => {
  for (const discardShanten of [0, 1]) {
    const html = tile({ tile: '5z', dora: true, safe: true, selected: true, discardShanten, onclick() {} });
    assert.match(html, /repeating-linear-gradient/);
    for (const colour of ['#e2453d', '#7fd1a0', '#4ea3ff', discardShanten ? '#c5cbd3' : 'var(--gold, #d8a12a)']) {
      assert.ok(html.includes(colour), colour);
    }
    assert.match(html, /class="foil\b/);
    assert.match(html, /class="haku-dragon-reveal\b/);
  }
});


test('explicit tile descriptions are exposed to assistive technology', () => {
  const description = '5 circles, claimed, riichi declaration, discarded from the draw, dora';
  const html = tile({ tile: '5p', title: description, dora: true });
  assert.match(html, new RegExp(`aria-label="${description}"`));
  assert.match(html, new RegExp(`title="${description}"`));
});

const classes = html => html.match(/class="(tile [^"]*)"/)[1].split(/\s+/);

test('a discard thrown from the draw is shaded and says so; one from the hand does neither', () => {
  for (const onclick of [null, () => {}]) {
    const thrown = tile({ tile: '3p', fromDraw: true, onclick });
    assert.ok(classes(thrown).includes('from-draw'));
    assert.match(thrown, /aria-label="3 circles, discarded from the draw"/);
    assert.match(thrown, /title="3 circles, discarded from the draw"/);
    const kept = tile({ tile: '3p', onclick });
    assert.doesNotMatch(kept, /\bfrom-draw\b|from the draw/);
    assert.match(kept, /aria-label="3 circles"/);
  }
});

test('a claimed discard is bordered and says so; an unclaimed one does neither', () => {
  for (const onclick of [null, () => {}]) {
    const taken = tile({ tile: '6z', claimed: true, onclick });
    assert.ok(classes(taken).includes('claimed'));
    assert.match(taken, /aria-label="green dragon, claimed"/);
    assert.match(taken, /title="green dragon, claimed"/);
    const left = tile({ tile: '6z', onclick });
    assert.doesNotMatch(left, /\bclaimed\b/);
    assert.match(left, /aria-label="green dragon"/);
  }
});

test('a hidden face carries neither discard mark, in its look or its name', () => {
  // As with dora, a face-down tile is only ever named as face down.
  for (const onclick of [null, () => {}]) {
    const hidden = tile({ tile: '3p', facedown: true, fromDraw: true, claimed: true, onclick });
    assert.doesNotMatch(hidden, /\bfrom-draw\b|from the draw|\bclaimed\b/);
    assert.match(hidden, /aria-label="face-down tile"/);
  }
  assert.doesNotMatch(tile({ tile: null, fromDraw: true, claimed: true }), /\bfrom-draw\b|\bclaimed\b/);
});

test('both discard marks combine with the riichi turn and the dora ring and foil', () => {
  const html = tile({ tile: '7z', fromDraw: true, rotated: true, claimed: true, dora: true, size: 'small' });
  for (const name of ['rotated', 'claimed', 'from-draw', 'ringed']) assert.ok(classes(html).includes(name), name);
  assert.match(html, /--ring:\s*#e2453d/);
  assert.match(html, /class="foil\b/);
  assert.match(html, /aria-label="red dragon, claimed, discarded from the draw, dora"/);
  // Each mark alone leaves the other off.
  assert.ok(!classes(tile({ tile: '7z', fromDraw: true })).includes('claimed'));
  assert.ok(!classes(tile({ tile: '7z', claimed: true })).includes('from-draw'));
});

test('the discard marks in Tile fall back to the values the theme gives them', async () => {
  // The tile-effects check renders Tile without app.css, so it sees these
  // fallbacks: they must be the theme's own values, or it checks another look.
  const theme = await readFile(new URL('../src/app.css', import.meta.url), 'utf8');
  const token = name => theme.match(new RegExp(`--${name}:\\s*([^;]+);`))[1].trim();
  const literal = text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  assert.match(source, new RegExp(`var\\(--from-draw-shade,\\s*${literal(token('from-draw-shade'))}\\)`));
  assert.match(source, new RegExp(`var\\(--claimed-border,\\s*${literal(token('claimed-border'))}\\)`));
  // The border is the claimed mark: nothing fades a claimed tile any more.
  assert.doesNotMatch(source + theme, /claimed-opacity|opacity:\s*var\(--claimed/);
});
