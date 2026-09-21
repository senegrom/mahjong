import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { DALI_APPROVED, TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = new URL('../../', import.meta.url);
const publicRoot = new URL('../public/', import.meta.url);
const set = JSON.parse(readFileSync(new URL('tiles/dali/manifest.json', publicRoot), 'utf8'));
const selected = set.tiles.find(entry => entry.tile === '7s');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const artworkHash = '80752661ba392a5c7b4c686e7f80612a2b768173b5fa8cde830025b82a3701eb';
const originalHash = 'f788e4ea0232f5d6ed23d7acf453936f2f55383de5a2695e917e4178d77168f7';
const rasterHash = '4187644e14baf56196933646d3f4e24d4e3e77d95fb2a94e32d60ec9577202c0';

test('Dream Cabinet is assigned to seven bamboo and three bamboo stays unselected', () => {
  assert.ok(selected);
  assert.equal(selected.name, 'Sou7');
  assert.equal(selected.direction, 'The Dream Cabinet');
  assert.equal(selected.status, 'approved');
  assert.equal(tileImage('7s', 'dali'), 'tiles/dali/approved/Sou7.svg');
  assert.ok(DALI_APPROVED.includes('7s'));
  assert.equal(DALI_APPROVED.includes('3s'), false);
  assert.equal(tileImage('3s', 'dali'), 'tiles/dali/placeholders/placeholder.svg');
  assert.equal(tileImage('5s', 'dali'), 'tiles/dali/approved/Sou5.svg');
  assert.equal(set.tiles.length, 13);
  assert.equal(set.placeholders.length, 21);
  assert.deepEqual(set.tiles.map(entry => entry.tile), [...DALI_APPROVED]);
  const identities = [...set.tiles, ...set.placeholders].map(entry => entry.tile);
  assert.equal(new Set(identities).size, 34);
  assert.deepEqual(identities.sort(), [...TILE_TYPES].sort());
});

test('Dream Cabinet source and runtime SVG are byte-identical and hash-pinned', () => {
  assert.equal(selected.source, 'docs/design/dali/studies/seven-bamboo-dream-cabinets-approved.svg');
  const source = readFileSync(new URL(selected.source, root));
  const artwork = readFileSync(new URL(`tiles/dali/${selected.svg}`, publicRoot));
  assert.deepEqual(artwork, source);
  assert.equal(sha256(artwork), artworkHash);
  assert.equal(selected.sourceSha256, artworkHash);
  assert.equal(selected.svgSha256, artworkHash);
  assert.equal(Object.hasOwn(selected, 'png'), false);
  assert.deepEqual(selected.crop, { x: 0, y: 0, width: 432, height: 576 });
});

test('embedded optimized image records its original provenance and complete portrait', () => {
  const svg = readFileSync(new URL(`tiles/dali/${selected.svg}`, publicRoot), 'utf8');
  const metadata = JSON.parse(svg.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
  assert.equal(metadata.originalFilename, 'seven_bamboo_dream_cabinets.png');
  assert.equal(metadata.originalSha256, originalHash);
  assert.deepEqual(metadata.originalDimensions, [1086, 1448]);
  assert.deepEqual(metadata.rasterDimensions, [432, 576]);
  assert.equal(metadata.rasterMimeType, 'image/webp');
  assert.equal(metadata.rasterSha256, rasterHash);
  assert.match(metadata.processing, /Complete portrait resized with Lanczos/);
  const raster = Buffer.from(svg.match(/data:image\/webp;base64,([^"\s]+)/)[1], 'base64');
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 12), 'WEBP');
  assert.equal(sha256(raster), rasterHash);
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(svg, /(?:href|src)=["']https?:/);
});

test('seven bamboo preloads once without exposing hidden tiles or changing other sets', () => {
  const image = tileImage('7s', 'dali');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === image).length, 1);
  assert.equal(tileImage('7s', 'dali', true), 'tiles/Back.svg');
  assert.equal(tileImage('7s', 'classic'), 'tiles/Sou7.svg');
  assert.equal(tileImage('7s', 'matisse'), 'tiles/matisse/approved/Sou7.svg');
  assert.equal(tileImage('7s', 'van-gogh'), 'tiles/Sou7.svg');
});

test('all existing PNG-backed Dali exports retain their recorded source and raster hashes', () => {
  for (const entry of set.tiles.filter(entry => entry.tile !== '7s')) {
    const source = readFileSync(new URL(entry.source, root));
    const raster = readFileSync(new URL(`tiles/dali/${entry.png}`, publicRoot));
    const svg = readFileSync(new URL(`tiles/dali/${entry.svg}`, publicRoot), 'utf8');
    assert.equal(sha256(source), entry.sourceSha256);
    assert.equal(sha256(raster), entry.pngSha256);
    assert.ok(svg.includes(`data:image/png;base64,${raster.toString('base64')}`));
  }
});
