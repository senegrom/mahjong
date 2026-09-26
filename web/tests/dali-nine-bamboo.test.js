import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { DALI_APPROVED, TILE_FACE_OPTIONS, TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = new URL('../../', import.meta.url);
const publicRoot = new URL('../public/', import.meta.url);
const set = JSON.parse(readFileSync(new URL('tiles/dali/manifest.json', publicRoot), 'utf8'));
const selected = set.tiles.find(entry => entry.tile === '9s');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const artworkHash = 'bcab49c117abc90e41582bcf83fdc40cd0038941413171546dbb00dbd22e9573';
const originalHash = 'b0775f5a4990ac3cdb5c866ed9663f60e07cdbd380b716faf25ae7bffb52abb8';
const rasterHash = 'cbb3100e53f0fefbdce3da96032f4bdd43bf17c3a308b1d253da76e49dcd21a3';

test('approved Surreal Grove is nine bamboo, with fourteen approved identities and twenty placeholders', () => {
  assert.ok(selected);
  assert.equal(selected.name, 'Sou9');
  assert.equal(selected.direction, 'The Surreal Grove');
  assert.equal(selected.status, 'approved');
  assert.equal(tileImage('9s', 'dali'), 'tiles/dali/approved/Sou9.svg');
  assert.deepEqual([...DALI_APPROVED], ['1p', '3p', '5p', '1s', '2s', '5s', '7s', '9s', '5m', '6m', '7m', '8m', '9m', '7z']);
  assert.equal(set.tiles.length, 14);
  assert.equal(set.placeholders.length, 20);
  assert.deepEqual(set.tiles.map(entry => entry.tile), [...DALI_APPROVED]);
  const identities = [...set.tiles, ...set.placeholders].map(entry => entry.tile);
  assert.equal(new Set(identities).size, 34);
  assert.deepEqual(identities.sort(), [...TILE_TYPES].sort());
});

test('nine-bamboo source and runtime artwork match the approved game rendering exactly', () => {
  assert.equal(selected.source, 'docs/design/dali/studies/nine-bamboo-surreal-grove-approved.svg');
  const source = readFileSync(new URL(selected.source, root));
  const runtime = readFileSync(new URL(`tiles/dali/${selected.svg}`, publicRoot));
  assert.deepEqual(runtime, source);
  assert.equal(hash(source), artworkHash);
  assert.equal(selected.sourceSha256, artworkHash);
  assert.equal(selected.svgSha256, artworkHash);
  assert.deepEqual(selected.crop, { x: 0, y: 0, width: 300, height: 400 });
});

test('nine-bamboo embedded WebP records the final red-accent original and has no external image requests', () => {
  const svg = readFileSync(new URL(`tiles/dali/${selected.svg}`, publicRoot), 'utf8');
  const meta = JSON.parse(svg.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
  assert.equal(meta.originalFilename, 'surreal_bamboo_grid_in_a_dreamy_landscape.png');
  assert.equal(meta.originalSha256, originalHash);
  assert.deepEqual(meta.originalDimensions, [1086, 1448]);
  assert.deepEqual(meta.rasterDimensions, [300, 400]);
  assert.equal(meta.rasterMimeType, 'image/webp');
  assert.equal(meta.rasterSha256, rasterHash);
  assert.match(meta.processing, /no redraw, added border, tile texture or color changes/);
  const raster = Buffer.from(svg.match(/data:image\/webp;base64,([^"\s]+)/)[1], 'base64');
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 12), 'WEBP');
  assert.equal(hash(raster), rasterHash);
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(svg, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('nine bamboo preloads once, keeps hidden tiles private, and leaves other sets and three bamboo alone', () => {
  const url = tileImage('9s', 'dali');
  assert.equal(TILE_IMAGE_URLS.filter(image => image === url).length, 1);
  for (const { value: face } of TILE_FACE_OPTIONS) {
    assert.equal(tileImage('9s', face, true), 'tiles/Back.svg');
  }
  assert.equal(tileImage('9s', 'classic'), 'tiles/Sou9.svg');
  assert.equal(tileImage('9s', 'matisse'), 'tiles/matisse/approved/Sou9.svg');
  assert.equal(tileImage('9s', 'van-gogh'), 'tiles/Sou9.svg');
  assert.equal(tileImage('3s', 'dali'), 'tiles/dali/placeholders/placeholder.svg');
  assert.equal(set.tiles.find(entry => entry.tile === '7s').svgSha256,
    '80752661ba392a5c7b4c686e7f80612a2b768173b5fa8cde830025b82a3701eb');
});

test('nine bamboo is represented in both reproducible export definitions and the preview', () => {
  const exporter = readFileSync(new URL('../scripts/export-dali-tiles.mjs', import.meta.url), 'utf8');
  assert.match(exporter, /\['Sou9', '9s', '9 bamboo', 'The Surreal Grove', 'nine-bamboo-surreal-grove-approved.svg', \[0, 0, 300, 400\]\]/);
  const preview = readFileSync(new URL('tiles/dali/preview.html', publicRoot), 'utf8');
  assert.ok(preview.includes('approved/Sou9.svg'));
  assert.ok(preview.includes('14 approved faces'));
  assert.ok(preview.includes('remaining 20 tiles'));
});
