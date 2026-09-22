import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
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

test('Van Gogh preserves selected Almond Branches, East A and North B, excludes K and uses L for white dragon', () => {
  const approved = ['1p', '5p', '3s', '3m', '7z', '1s', '2p', '9p', '6s', '5z', '1z', '4z', '2m', '4m', '2s'];
  assert.deepEqual(VAN_GOGH_APPROVED, approved);
  const set = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
  assert.deepEqual(set.tiles.map(tile => tile.tile), approved);
  assert.deepEqual(set.tiles.map(tile => tile.candidate), ['A', 'B', 'C', 'Characters A', 'E', 'G', 'H', 'I', 'J', 'L', 'East A', 'North B', 'Characters B', 'Characters C', 'Bamboo B (green)']);
  assert.deepEqual(set.rejected.map(tile => tile.candidate), ['K']);
  assert.equal(tileImage('1z', 'van-gogh'), 'tiles/van-gogh/approved/Ton.svg');
  assert.equal(set.tiles.find(tile => tile.tile === '1z').source,
    'docs/design/van-gogh/studies/03-east-wind-alternatives.png');
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const three = set.tiles.find(tile => tile.tile === '3m');
  assert.equal(three.source, 'docs/design/van-gogh/studies/06-three-characters-new-directions.png');
  assert.deepEqual(three.crop, { x: 25, y: 93, width: 523, height: 782 });
  assert.equal(hash(readFileSync(new URL(`../../${three.source}`, import.meta.url))),
    'a0d0c325a5f63f6121555e099afcbba65cfaeb2941aee3962572b371282bfdd7');
  assert.equal(set.superseded.find(tile => tile.candidate === 'D').activeCandidate, 'Characters A');
  const north = set.tiles.find(tile => tile.tile === '4z');
  assert.equal(tileImage('4z', 'van-gogh'), 'tiles/van-gogh/approved/Pei.svg');
  assert.equal(north.source, 'docs/design/van-gogh/studies/04-north-wind-ribbons.png');
  assert.deepEqual(north.crop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.equal(hash(readFileSync(new URL(`../../${north.source}`, import.meta.url))),
    '3a51fc2f26cbd476cce085d93b52a5a81c5a17d14d4bcf00a0597a402435588e');
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

test('Dali resolves thirteen approved images and placeholders for the rest', () => {
  const approved = new Set(['1p', '3p', '5p', '1s', '2s', '5s', '7s', '5m', '6m', '7m', '8m', '9m', '7z']);
  for (const tile of TILE_TYPES) {
    const url = tileImage(tile, 'dali');
    const svg = readFileSync(new URL(url, publicRoot), 'utf8');
    assert.match(svg, /viewBox="0 0 300 400"/);
    if (approved.has(tile)) {
      assert.match(url, /\/dali\/approved\//);
      assert.match(svg, tile === '7s' ? /data:image\/webp;base64,/ : /data:image\/png;base64,/);
    }
    else assert.equal(url, 'tiles/dali/placeholders/placeholder.svg');
  }
  assert.equal(tileImage('1p', 'dali'), 'tiles/dali/approved/Pin1.svg');
  assert.equal(tileImage('3p', 'dali'), 'tiles/dali/approved/Pin3.svg');
  assert.equal(tileImage('5s', 'dali'), 'tiles/dali/approved/Sou5.svg');
  assert.equal(tileImage('7s', 'dali'), 'tiles/dali/approved/Sou7.svg');
  assert.equal(tileImage('5m', 'dali'), 'tiles/dali/approved/Man5.svg');
  assert.equal(tileImage('6m', 'dali'), 'tiles/dali/approved/Man6.svg');
  assert.equal(tileImage('7m', 'dali'), 'tiles/dali/approved/Man7.svg');
  assert.equal(tileImage('8m', 'dali'), 'tiles/dali/approved/Man8.svg');
  assert.equal(tileImage('9m', 'dali'), 'tiles/dali/approved/Man9.svg');
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
  assert.equal(TILE_IMAGE_URLS.filter(url => url.startsWith('tiles/van-gogh/')).length, 15);
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
    for (const tile of ['1z', '4z', '2m', '4m', '2s']) {
      const wind = show(tile, 'van-gogh', { dora: true, size: 'small' });
      assert.match(wind, /\bvan-gogh\b/);
      assert.match(wind, /\bringed\b/);
    }
    assert.doesNotMatch(show('2z', 'van-gogh'), /\bvan-gogh\b/);
  }
});

test('Van Gogh 2 and 4 characters preserve their exact approved source images', () => {
  const set = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
  assert.equal(set.remaining.length, 19);
  for (const [tile, filename, expectedBlob] of [
    ['2m', '07-two-characters-night-cafe.png', '5ce8e6822ece3d11b0e33b21a666b6272717adc5'],
    ['4m', '08-four-characters-cypress-fields.png', 'e3410ae633e835c6300a24bbce86e59e8999eab0'],
  ]) {
    const entry = set.tiles.find(entry => entry.tile === tile);
    assert.ok(entry);
    assert.equal(entry.source, `docs/design/van-gogh/studies/${filename}`);
    const source = readFileSync(new URL(`../../${entry.source}`, import.meta.url));
    const header = Buffer.from(`blob ${source.length}\0`);
    assert.equal(createHash('sha1').update(header).update(source).digest('hex'), expectedBlob);
    assert.deepEqual(readFileSync(new URL(`tiles/van-gogh/${entry.png}`, publicRoot)), source);
    assert.deepEqual(entry.crop, { x: 0, y: 0, width: source.readUInt32BE(16), height: source.readUInt32BE(20) });
    assert.ok(!set.remaining.includes(tile));
    assert.ok(TILE_IMAGE_URLS.includes(tileImage(tile, 'van-gogh')));
  }
});


test('Van Gogh green Garden Rhythm B is the approved two bamboo', () => {
  const set = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
  const entry = set.tiles.find(tile => tile.tile === '2s');
  assert.ok(entry);
  assert.equal(entry.candidate, 'Bamboo B (green)');
  assert.equal(entry.name, 'Sou2');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/11-two-bamboo-garden-rhythm-green.webp');
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 });
  const source = readFileSync(new URL(`../../${entry.source}`, import.meta.url));
  assert.equal(createHash('sha256').update(source).digest('hex'),
    '528acc47e037fdce2fc9f48f827bbe554da35a1cc0186a4b74f1ac9fea0ad7ec');
  assert.equal(tileImage('2s', 'van-gogh'), 'tiles/van-gogh/approved/Sou2.svg');
  assert.ok(TILE_IMAGE_URLS.includes(tileImage('2s', 'van-gogh')));
  assert.ok(!set.remaining.includes('2s'));
  const provenance = JSON.parse(readFileSync(new URL('../../docs/design/van-gogh/two-bamboo-green.json', import.meta.url), 'utf8'));
  assert.equal(provenance.sourceSha256, createHash('sha256').update(source).digest('hex'));
  assert.equal(provenance.originalSha256, 'ba6cb69a60cc24173125750b412e3899a4406f28557256995ce81d4b34893748');
  assert.deepEqual(provenance.originalCrop, { x: 508, y: 143, width: 433, height: 667 });
});
