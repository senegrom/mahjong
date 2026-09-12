import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, mkdtempSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { TILE_FACE_CONTEXT, TILE_FACE_OPTIONS, TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { readSettings } from '../src/lib/session.js';

const publicRoot = new URL('../public/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('tiles/matisse/manifest.json', publicRoot), 'utf8'));
const faces = TILE_FACE_OPTIONS.map(face => face.value);

test('Van Gogh preserves the selected artwork, excludes K and uses L for white dragon', () => {
  const approved = ['1p', '5p', '3s', '3m', '7z', '1s', '2p', '9p', '6s', '5z'];
  assert.deepEqual(VAN_GOGH_APPROVED, approved);
  const set = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
  assert.deepEqual(set.tiles.map(tile => tile.tile), approved);
  assert.deepEqual(set.tiles.map(tile => tile.candidate), ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'J', 'L']);
  assert.deepEqual(set.rejected.map(tile => tile.candidate), ['K']);
  assert.equal(existsSync(new URL('tiles/van-gogh/approved/Ton.svg', publicRoot)), false);
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  for (const source of set.sources) {
    assert.equal(hash(readFileSync(new URL(`../../${source.source}`, import.meta.url))), source.sha256);
  }
  for (const tile of TILE_TYPES) {
    const entry = set.tiles.find(entry => entry.tile === tile);
    const url = tileImage(tile, 'van-gogh');
    if (!entry) {
      assert.equal(url, tileImage(tile, 'classic'));
      continue;
    }
    assert.equal(url, `tiles/van-gogh/${entry.svg}`);
    const png = readFileSync(new URL(`tiles/van-gogh/${entry.png}`, publicRoot));
    const svg = readFileSync(new URL(url, publicRoot), 'utf8');
    assert.equal(hash(png), entry.pngSha256);
    assert.match(svg, /viewBox="0 0 300 400"/);
    assert.match(svg, /rx="26"/);
    assert.ok(svg.includes(`data:image/png;base64,${png.toString('base64')}`));
  }
  assert.equal(tileImage('not-a-tile', 'van-gogh'), 'tiles/Front.svg');
});

test('all 34 Matisse faces resolve to approved art with no placeholders', () => {
  assert.equal(TILE_TYPES.length, 34);
  assert.deepEqual(manifest.tiles.map(tile => tile.tile).sort(), [...TILE_TYPES].sort());
  assert.equal(manifest.placeholders.length, 0);
  for (const tile of TILE_TYPES) {
    const url = tileImage(tile, 'matisse');
    const svg = readFileSync(new URL(url, publicRoot), 'utf8');
    assert.match(svg, /viewBox="0 0 300 400"/);
    assert.match(url, /\/approved\//);
    assert.match(svg, /data:image\/png;base64,/);
  }
  assert.equal(tileImage('8m', 'matisse'), 'tiles/matisse/approved/Man8.svg');
});

test('Dali resolves seven approved images and placeholders for the rest', () => {
  const approved = new Set(['1p', '3p', '5p', '1s', '2s', '8m', '7z']);
  for (const tile of TILE_TYPES) {
    const url = tileImage(tile, 'dali');
    const svg = readFileSync(new URL(url, publicRoot), 'utf8');
    assert.match(svg, /viewBox="0 0 300 400"/);
    if (approved.has(tile)) {
      assert.match(url, /\/dali\/approved\//);
      assert.match(svg, /data:image\/png;base64,/);
    }
    else assert.equal(url, 'tiles/dali/placeholders/placeholder.svg');
  }
  assert.equal(tileImage('1p', 'dali'), 'tiles/dali/approved/Pin1.svg');
  assert.equal(tileImage('3p', 'dali'), 'tiles/dali/approved/Pin3.svg');
  assert.equal(tileImage('7z', 'dali'), 'tiles/dali/approved/Chun.svg');
});

test('hidden tiles cannot reveal their identity through any face set', () => {
  for (const face of faces) {
    for (const tile of TILE_TYPES) assert.equal(tileImage(tile, face, true), 'tiles/Back.svg');
    assert.equal(tileImage(null, face), 'tiles/Back.svg');
  }
  assert.equal(tileImage('5z'), 'tiles/Haku.svg');
  assert.equal(tileImage('5z', 'matisse'), 'tiles/matisse/approved/Haku.svg');
  assert.equal(tileImage('7m', 'classic'), 'tiles/Man7.svg');
});

test('all selectable face sets are in the preload inventory with valid files', () => {
  assert.equal(new Set(TILE_IMAGE_URLS).size, TILE_IMAGE_URLS.length);
  assert.ok(TILE_IMAGE_URLS.includes('tiles/matisse/approved/Haku-foil.svg'));
  assert.ok(TILE_IMAGE_URLS.includes('tiles/dali/approved/Pin1.svg'));
  assert.ok(TILE_IMAGE_URLS.includes('tiles/dali/placeholders/placeholder.svg'));
  assert.equal(TILE_IMAGE_URLS.some(url => url.startsWith('tiles/cubist/')), false);
  assert.equal(TILE_IMAGE_URLS.filter(url => url.startsWith('tiles/van-gogh/')).length, 10);
  for (const face of faces) {
    for (const tile of TILE_TYPES) assert.ok(TILE_IMAGE_URLS.includes(tileImage(tile, face)));
  }
  for (const url of TILE_IMAGE_URLS) assert.match(readFileSync(new URL(url, publicRoot), 'utf8'), /<svg/);
});

test('tile face survives preference restoration and retired or invalid settings use Classic', () => {
  const read = value => readSettings({ getItem: () => JSON.stringify(value) });
  assert.equal(read({ version: 1, tileFace: 'matisse' }).tileFace, 'matisse');
  assert.equal(read({ version: 1, tileFace: 'dali' }).tileFace, 'dali');
  assert.equal(read({ version: 1, tileFace: 'cubist' }).tileFace, 'classic');
  assert.equal(read({ version: 1, tileFace: 'van-gogh' }).tileFace, 'van-gogh');
  for (const tileFace of [undefined, null, '', false, {}, 'other', '../other']) {
    assert.equal(read({ version: 1, tileFace }).tileFace, 'classic');
  }
  assert.equal(read({ version: 2, tileFace: 'dali' }).tileFace, 'classic');
});

test('the real Tile component respects the selected face and hidden state', async t => {
  const source = new URL('../src/lib/Tile.svelte', import.meta.url);
  const directory = mkdtempSync(fileURLToPath(new URL('../.tile-effects-unit-', import.meta.url)));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const compiled = compile(readFileSync(source, 'utf8'), { filename: fileURLToPath(source), generate: 'server' });
  const code = compiled.js.code.replace(/from (["'])(\.[^'"]+)\1/g, (_match, _quote, specifier) => {
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
      assert.ok(show(tile, 'dali').includes(`src="${tileImage(tile, 'dali')}"`));
      assert.ok(show(tile, 'van-gogh').includes(`src="${tileImage(tile, 'van-gogh')}"`));
      for (const face of faces) {
        const hidden = show(tile, face, { facedown: true, dora: true });
        assert.match(hidden, /src="tiles\/Back.svg"/);
        assert.doesNotMatch(hidden, /dali\/|matisse\/|van-gogh\/|class="foil|haku-dragon-reveal/);
      }
    }
    const white = show('5z', 'matisse', { dora: true });
    assert.match(white, /approved\/Haku.svg/);
    assert.match(white, /class="foil/);
    assert.match(white, /haku-dragon-reveal/);
    assert.match(white, /approved\/Haku-foil.svg/);
    assert.doesNotMatch(show('5z', 'matisse'), /haku-dragon-reveal/);
    assert.match(show('5z', 'classic', { dora: true }), /haku-dragon-reveal/);
    assert.match(show('7m', 'classic'), /src="tiles\/Man7.svg"/);
    const vanGoghWhite = show('5z', 'van-gogh', { dora: true, rotated: true, size: 'small' });
    assert.match(vanGoghWhite, /tiles\/van-gogh\/approved\/Haku.svg/);
    assert.match(vanGoghWhite, /\bvan-gogh\b/);
    assert.match(vanGoghWhite, /\bringed\b/);
    assert.match(vanGoghWhite, /class="foil/);
    assert.doesNotMatch(vanGoghWhite, /haku-dragon-reveal|Haku-foil/);
    assert.doesNotMatch(show('1z', 'van-gogh'), /\bvan-gogh\b/);
  }
});
