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
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

// The approved bamboo faces whose artwork, original and embedded raster are
// pinned byte for byte.
const FACES = [
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

test('fourteen approved identities and twenty placeholders cover the thirty-four tiles once', () => {
  assert.deepEqual([...DALI_APPROVED], ['1p', '3p', '5p', '1s', '2s', '5s', '7s', '9s', '5m', '6m', '7m', '8m', '9m', '7z']);
  assert.equal(set.tiles.length, 14);
  assert.equal(set.placeholders.length, 20);
  assert.deepEqual(set.tiles.map(entry => entry.tile), [...DALI_APPROVED]);
  const identities = [...set.tiles, ...set.placeholders].map(entry => entry.tile);
  assert.equal(new Set(identities).size, 34);
  assert.deepEqual(identities.sort(), [...TILE_TYPES].sort());
  // Three bamboo stays unselected beside its approved neighbours.
  assert.equal(DALI_APPROVED.includes('3s'), false);
  assert.equal(tileImage('3s', 'dali'), 'tiles/dali/placeholders/placeholder.svg');
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
    assert.deepEqual(metadata.originalDimensions, [1086, 1448]);
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
    assert.equal(tileImage(face.tile, 'van-gogh'), `tiles/${face.name}.svg`);
  });
}

test('nine bamboo is represented in both reproducible export definitions and the preview', () => {
  const exporter = readFileSync(new URL('../scripts/export-dali-tiles.mjs', import.meta.url), 'utf8');
  assert.match(exporter, /\['Sou9', '9s', '9 bamboo', 'The Surreal Grove', 'nine-bamboo-surreal-grove-approved.svg', \[0, 0, 300, 400\]\]/);
  const preview = readFileSync(new URL('tiles/dali/preview.html', publicRoot), 'utf8');
  assert.ok(preview.includes('approved/Sou9.svg'));
  assert.ok(preview.includes('14 approved faces'));
  assert.ok(preview.includes('remaining 20 tiles'));
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
