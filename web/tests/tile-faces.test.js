import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';
import { TILE_TYPES, tileWords } from '../src/lib/tiles.js';
import { TILE_FACE_CONTEXT, TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';
import { readSettings } from '../src/lib/session.js';

const publicRoot = new URL('../public/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('tiles/matisse/manifest.json', publicRoot), 'utf8'));

test('all 34 Matisse faces resolve to approved art or a black text placeholder', () => {
  const approved = ['1p', '3p', '5p', '2s', '5s', '6s', '7s', '8s', '9s', '1z', '7z', '7m', '1s', '6z', '5z'];
  assert.equal(TILE_TYPES.length, 34);
  assert.deepEqual(manifest.tiles.map(tile => tile.tile).sort(), [...approved].sort());
  assert.equal(manifest.placeholders.length, 19);
  for (const tile of TILE_TYPES) {
    const url = tileImage(tile, 'matisse');
    const svg = readFileSync(new URL(url, publicRoot), 'utf8');
    assert.match(svg, /viewBox="0 0 300 400"/);
    if (approved.includes(tile)) {
      assert.match(url, /\/approved\//);
      assert.match(svg, /data:image\/png;base64,/);
    } else {
      assert.match(url, /\/placeholders\//);
      assert.ok(svg.includes(`<title id="title">${tileWords(tile)}</title>`));
      assert.match(svg, /<text[^>]*fill="#000"/);
      assert.doesNotMatch(svg, /<image|<path/);
      assert.equal([...svg.matchAll(/<tspan[^>]*>(.*?)<\/tspan>/g)].map(match => match[1]).join(' '), tileWords(tile));
    }
  }
  assert.equal(tileImage('7m', 'matisse'), 'tiles/matisse/approved/Man7.svg');
});

test('hidden tiles cannot reveal their identity through either face set', () => {
  for (const face of ['classic', 'matisse']) {
    for (const tile of TILE_TYPES) assert.equal(tileImage(tile, face, true), 'tiles/Back.svg');
    assert.equal(tileImage(null, face), 'tiles/Back.svg');
  }
  assert.equal(tileImage('5z'), 'tiles/Haku.svg');
  assert.equal(tileImage('5z', 'matisse'), 'tiles/matisse/approved/Haku.svg');
  assert.equal(tileImage('7m', 'unrecognized'), 'tiles/Man7.svg');
});

test('both complete face sets are in the preload inventory with valid files', () => {
  assert.equal(TILE_IMAGE_URLS.length, 71);
  assert.ok(TILE_IMAGE_URLS.includes('tiles/matisse/approved/Haku-foil.svg'));
  for (const face of ['classic', 'matisse']) {
    for (const tile of TILE_TYPES) assert.ok(TILE_IMAGE_URLS.includes(tileImage(tile, face)));
  }
  for (const url of TILE_IMAGE_URLS) assert.match(readFileSync(new URL(url, publicRoot), 'utf8'), /<svg/);
});

test('tile face survives preference restoration and old or invalid settings stay Classic', () => {
  const read = value => readSettings({ getItem: () => JSON.stringify(value) });
  assert.equal(read({ version: 1, tileFace: 'matisse' }).tileFace, 'matisse');
  for (const tileFace of [undefined, null, '', false, {}, 'other', '../other']) {
    assert.equal(read({ version: 1, tileFace }).tileFace, 'classic');
  }
  assert.equal(read({ version: 2, tileFace: 'matisse' }).tileFace, 'classic');
});

test('the real Tile component respects the selected face, foil and hidden state', async t => {
  const source = new URL('../src/lib/Tile.svelte', import.meta.url);
  const directory = mkdtempSync(fileURLToPath(new URL('../.tile-effects-unit-', import.meta.url)));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const compiled = compile(readFileSync(source, 'utf8'), { filename: fileURLToPath(source), generate: 'server' });
  // Node needs a URL module for the image import; all component logic and
  // shared modules are compiled from the production sources without changes.
  const code = compiled.js.code.replace(/from (['"])(\.[^'"]+)\1/g, (_match, _quote, specifier) => {
    const url = new URL(specifier, source).href;
    return `from ${JSON.stringify(specifier.endsWith('.webp')
      ? `data:text/javascript,${encodeURIComponent(`export default ${JSON.stringify(url)}`)}` : url)}`;
  });
  const output = `${directory}/Tile.mjs`;
  writeFileSync(output, code);
  const { default: Tile } = await import(pathToFileURL(output).href);
  for (const interactive of [false, true]) {
    const show = (tile, face, props = {}) => render(Tile, {
      props: { tile, ...(interactive ? { onclick() {} } : {}), ...props },
      context: new Map([[TILE_FACE_CONTEXT, () => face]]),
    }).body;
    for (const tile of TILE_TYPES) {
      assert.ok(show(tile, 'matisse').includes(`src="${tileImage(tile, 'matisse')}"`));
      const hidden = show(tile, 'matisse', { facedown: true, dora: true });
      assert.match(hidden, /src="tiles\/Back.svg"/);
      assert.doesNotMatch(hidden, /matisse\/|class="foil|haku-dragon-reveal/);
    }
    const white = show('5z', 'matisse', { dora: true });
    assert.match(white, /approved\/Haku.svg/);
    assert.match(white, /class="foil/);
    assert.match(white, /haku-dragon-reveal/);
    assert.match(white, /approved\/Haku-foil.svg/);
    assert.doesNotMatch(show('5z', 'matisse'), /haku-dragon-reveal/);
    assert.match(show('5z', 'classic', { dora: true }), /haku-dragon-reveal/);
    assert.match(show('7m', 'classic'), /src="tiles\/Man7.svg"/);
  }
});
