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
  .replace("'./tiles.js'", JSON.stringify(new URL('../src/lib/tiles.js', import.meta.url).href))
  .replace(/(['"])(svelte\/[^'"]+)\1/g, (_, quote, name) => JSON.stringify(import.meta.resolve(name)));
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
