import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { DALI_APPROVED } from '../src/lib/dali-faces.js';
import { TILE_FACE_OPTIONS, TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = new URL('../../', import.meta.url);
const publicRoot = new URL('../public/', import.meta.url);
const set = JSON.parse(readFileSync(new URL('tiles/dali/manifest.json', publicRoot), 'utf8'));
// Other sets can gain approved art independently; test against their own manifest.
const vanGogh = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

// The approved bamboo faces whose artwork, original and embedded raster are
// pinned byte for byte.
const FACES = [
  {
    tile: '3s', name: 'Sou3', words: 'three bamboo', direction: 'The Bamboo That Tied Itself — A',
    source: 'docs/design/dali/studies/three-bamboo-a-approved.svg',
    artwork: '43741c2f0f26ef70f89044f9a822ecfcffdb95fbdc2f130b7152eb3302363c95',
    original: '67d151a89d59f2f3907b6e9fa0b0968ca1d5c95faa2e6e6003a53fd567e122a0',
    raster: '45efaeaf1bef0f270dea09d039a93070ea2ae218eef02454ffa474429c3f8eb7',
    originalFilename: 'wide_triptych_art_image_with_three_vertical_panels.png',
    originalDimensions: [1536, 1024], rasterDimensions: [300, 400],
    processing: /Selected panel A cropped from the original triptych; border and caption removed/,
  },
  {
    tile: '8s', name: 'Sou8', words: 'eight bamboo', direction: 'Emerald Moonlit Seascape',
    source: 'docs/design/dali/studies/eight-bamboo-emerald-moonlit-approved.svg',
    artwork: '05e1d03a51c52ffb88a9df62985cd4d3b007364bd400a95a260d47d3d8832767',
    original: 'b57c7c4d29433d4c07fe78a57b6962f8b0184c5d760284e935fb31f7e102dd67',
    raster: '8308e85ce25d05b21d790eb6b06579e5a55ee2b225d8ce65eb3ff2e6c47e9974',
    originalFilename: 'emerald_moonlit_bamboo_seascape.png', rasterDimensions: [300, 400],
    processing: /Complete approved greener portrait resized with Lanczos/,
  },
  {
    tile: '9s', name: 'Sou9', words: 'nine bamboo', direction: 'The Surreal Grove',
    source: 'docs/design/dali/studies/nine-bamboo-surreal-grove-approved.svg',
    artwork: 'bcab49c117abc90e41582bcf83fdc40cd0038941413171546dbb00dbd22e9573',
    original: 'b0775f5a4990ac3cdb5c866ed9663f60e07cdbd380b716faf25ae7bffb52abb8',
    raster: 'cbb3100e53f0fefbdce3da96032f4bdd43bf17c3a308b1d253da76e49dcd21a3',
    originalFilename: 'surreal_bamboo_grid_in_a_dreamy_landscape.png', rasterDimensions: [300, 400],
    processing: /no redraw, added border, tile texture or color changes/,
  },
  {
    tile: '7s', name: 'Sou7', words: 'seven bamboo', direction: 'The Dream Cabinet',
    source: 'docs/design/dali/studies/seven-bamboo-dream-cabinets-approved.svg',
    artwork: '80752661ba392a5c7b4c686e7f80612a2b768173b5fa8cde830025b82a3701eb',
    original: 'f788e4ea0232f5d6ed23d7acf453936f2f55383de5a2695e917e4178d77168f7',
    raster: '4187644e14baf56196933646d3f4e24d4e3e77d95fb2a94e32d60ec9577202c0',
    originalFilename: 'seven_bamboo_dream_cabinets.png', rasterDimensions: [432, 576],
    processing: /Complete portrait resized with Lanczos/,
  },
];

test('eighteen approved identities and sixteen placeholders cover the thirty-four tiles once', () => {
  assert.deepEqual([...DALI_APPROVED], ['1p', '3p', '5p', '1s', '2s', '3s', '5s', '7s', '8s', '9s', '5m', '6m', '7m', '8m', '9m', '1z', '4z', '7z']);
  assert.equal(set.tiles.length, 18);
  assert.equal(set.placeholders.length, 16);
  assert.deepEqual(set.tiles.map(entry => entry.tile), [...DALI_APPROVED]);
  const identities = [...set.tiles, ...set.placeholders].map(entry => entry.tile);
  assert.equal(new Set(identities).size, 34);
  assert.deepEqual(identities.sort(), [...TILE_TYPES].sort());
  assert.equal(DALI_APPROVED.includes('3s'), true);
  assert.equal(tileImage('3s', 'dali'), 'tiles/dali/approved/Sou3.svg');
  for (const tile of ['4s', '6s']) {
    assert.equal(tileImage(tile, 'dali'), 'tiles/dali/placeholders/placeholder.svg');
  }
  assert.equal(tileImage('5s', 'dali'), 'tiles/dali/approved/Sou5.svg');
});

for (const face of FACES) {
  const selected = set.tiles.find(entry => entry.tile === face.tile);

  test(`${face.direction} is the approved ${face.words}`, () => {
    assert.ok(selected);
    assert.equal(selected.name, face.name);
    assert.equal(selected.direction, face.direction);
    assert.equal(selected.status, 'approved');
    assert.equal(tileImage(face.tile, 'dali'), `tiles/dali/approved/${face.name}.svg`);
  });

  test(`${face.words}: source and runtime SVG are byte-identical and hash-pinned`, () => {
    assert.equal(selected.source, face.source);
    const source = readFileSync(new URL(selected.source, root));
    const artwork = readFileSync(new URL(`tiles/dali/${selected.svg}`, publicRoot));
    assert.deepEqual(artwork, source);
    assert.equal(sha256(artwork), face.artwork);
    assert.equal(selected.sourceSha256, face.artwork);
    assert.equal(selected.svgSha256, face.artwork);
    assert.equal(Object.hasOwn(selected, 'png'), false);
    assert.deepEqual(selected.crop, { x: 0, y: 0, width: face.rasterDimensions[0], height: face.rasterDimensions[1] });
  });

  test(`${face.words}: the embedded WebP records its original and makes no external requests`, () => {
    const svg = readFileSync(new URL(`tiles/dali/${selected.svg}`, publicRoot), 'utf8');
    const metadata = JSON.parse(svg.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
    assert.equal(metadata.originalFilename, face.originalFilename);
    assert.equal(metadata.originalSha256, face.original);
    assert.deepEqual(metadata.originalDimensions, face.originalDimensions ?? [1086, 1448]);
    assert.deepEqual(metadata.rasterDimensions, face.rasterDimensions);
    assert.equal(metadata.rasterMimeType, 'image/webp');
    assert.equal(metadata.rasterSha256, face.raster);
    assert.match(metadata.processing, face.processing);
    const raster = Buffer.from(svg.match(/data:image\/webp;base64,([^"\s]+)/)[1], 'base64');
    assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
    assert.equal(raster.toString('ascii', 8, 12), 'WEBP');
    assert.equal(sha256(raster), face.raster);
    assert.match(svg, /viewBox="0 0 300 400"/);
    assert.match(svg, /rx="26"/);
    assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
    assert.doesNotMatch(svg, /(?:href|src)=["']https?:|<script|<foreignObject/);
  });

  test(`${face.words} preloads once, keeps hidden tiles private and leaves other sets alone`, () => {
    const image = tileImage(face.tile, 'dali');
    assert.equal(TILE_IMAGE_URLS.filter(url => url === image).length, 1);
    for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage(face.tile, value, true), 'tiles/Back.svg');
    assert.equal(tileImage(face.tile, 'classic'), `tiles/${face.name}.svg`);
    assert.equal(tileImage(face.tile, 'matisse'), `tiles/matisse/approved/${face.name}.svg`);
    const other = vanGogh.tiles.find(entry => entry.tile === face.tile);
    assert.equal(tileImage(face.tile, 'van-gogh'), other ? `tiles/van-gogh/${other.svg}` : `tiles/${face.name}.svg`);
  });
}

test('nine bamboo is represented in both reproducible export definitions and the preview', () => {
  const exporter = readFileSync(new URL('../scripts/export-dali-tiles.mjs', import.meta.url), 'utf8');
  assert.match(exporter, /\['Sou9', '9s', '9 bamboo', 'The Surreal Grove', 'nine-bamboo-surreal-grove-approved.svg', \[0, 0, 300, 400\]\]/);
  const preview = readFileSync(new URL('tiles/dali/preview.html', publicRoot), 'utf8');
  assert.ok(preview.includes('approved/Sou9.svg'));
  assert.ok(preview.includes('18 approved faces'));
  assert.ok(preview.includes('remaining 16 tiles'));
});

test('all existing PNG-backed Dali exports retain their recorded source and raster hashes', () => {
  for (const entry of set.tiles.filter(entry => entry.png)) {
    const source = readFileSync(new URL(entry.source, root));
    const raster = readFileSync(new URL(`tiles/dali/${entry.png}`, publicRoot));
    const svg = readFileSync(new URL(`tiles/dali/${entry.svg}`, publicRoot), 'utf8');
    assert.equal(sha256(source), entry.sourceSha256);
    assert.equal(sha256(raster), entry.pngSha256);
    assert.ok(svg.includes(`data:image/png;base64,${raster.toString('base64')}`));
  }
});

test('eight bamboo is the greener revision in the exporter, preview and provenance', () => {
  const exporter = readFileSync(new URL('../scripts/export-dali-tiles.mjs', import.meta.url), 'utf8');
  assert.match(exporter, /\['Sou8', '8s', '8 bamboo', 'Emerald Moonlit Seascape', 'eight-bamboo-emerald-moonlit-approved.svg', \[0, 0, 300, 400\]\]/);
  const preview = readFileSync(new URL('tiles/dali/preview.html', publicRoot), 'utf8');
  assert.ok(preview.includes('approved/Sou8.svg'));
  assert.ok(preview.includes('18 approved faces'));
  assert.ok(preview.includes('remaining 16 tiles'));
  const provenance = JSON.parse(readFileSync(new URL('docs/design/dali/eight-bamboo.json', root), 'utf8'));
  const face = FACES.find(entry => entry.tile === '8s');
  assert.equal(provenance.originalSha256, face.original);
  assert.equal(provenance.rasterSha256, face.raster);
  assert.equal(provenance.svgSha256, face.artwork);
});

test('three bamboo preserves selected panel A and is reproducible in the exporter and preview', () => {
  const face = FACES.find(entry => entry.tile === '3s');
  const provenance = JSON.parse(readFileSync(new URL('docs/design/dali/three-bamboo.json', root), 'utf8'));
  assert.equal(provenance.tile, '3s');
  assert.equal(provenance.selectedPanel, 'A');
  assert.equal(provenance.originalSha256, face.original);
  assert.equal(provenance.rasterSha256, face.raster);
  assert.equal(provenance.svgSha256, face.artwork);
  assert.deepEqual(provenance.originalCrop, { x: 11, y: 12, width: 482, height: 877 });
  const svg = readFileSync(new URL(face.source, root), 'utf8');
  const metadata = JSON.parse(svg.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
  assert.equal(metadata.selectedPanel, 'A');
  assert.deepEqual(metadata.originalCrop, provenance.originalCrop);
  const exporter = readFileSync(new URL('../scripts/export-dali-tiles.mjs', import.meta.url), 'utf8');
  assert.match(exporter, /\['Sou3', '3s', '3 bamboo', 'The Bamboo That Tied Itself — A', 'three-bamboo-a-approved.svg', \[0, 0, 300, 400\]\]/);
  const preview = readFileSync(new URL('tiles/dali/preview.html', publicRoot), 'utf8');
  assert.ok(preview.includes('approved/Sou3.svg'));
  assert.ok(preview.includes('The Bamboo That Tied Itself'));
  assert.ok(preview.includes('18 approved faces'));
  assert.ok(preview.includes('remaining 16 tiles'));
});
