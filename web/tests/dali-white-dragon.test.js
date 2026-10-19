import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { DALI_APPROVED } from '../src/lib/dali-faces.js';
import { TILE_FACE_OPTIONS, TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = new URL('../../', import.meta.url);
const read = path => readFileSync(new URL(path, root));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/dali/manifest.json'));
const provenance = JSON.parse(read('docs/design/dali/white-dragon.json'));
const selected = set.tiles.find(entry => entry.tile === '5z');
const svg = read('web/public/tiles/dali/approved/Haku.svg');
const markup = svg.toString();
const metadata = JSON.parse(markup.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
const originalHash = 'f1c12e186b3def7e5277c906c5cfb39f50cc9349bfcf9b66fa14f2d24578f193';
const rasterHash = '7576d2f0519c65659d0cef8ffc39545da55471341e274592bbb31de97449ac15';
const svgHash = '019442ea211bf281e98271a692e115dd529ec68a288bfb9303fe7a43ee4a74cb';

test('White Dragon uses the selected first silk painting, not the shadow or skeleton alternatives', () => {
  assert.equal(selected.name, 'Haku');
  assert.equal(selected.status, 'approved');
  assert.equal(selected.direction, 'The Fold That Woke Up — A');
  assert.equal(metadata.selectedOption, 'A (first), silken white dragon');
  assert.equal(provenance.selectedOption, metadata.selectedOption);
  assert.equal(metadata.originalFilename, 'silken_dragon_over_mirror_waters.png');
  assert.equal(metadata.originalSha256, originalHash);
  assert.equal(provenance.originalSha256, originalHash);
  assert.equal(provenance.originalStoredInRepository, false);
});

test('White Dragon preserves the complete 3:4 portrait in hash-pinned source and runtime SVGs', () => {
  assert.equal(selected.source, 'docs/design/dali/studies/white-dragon-a-approved.svg');
  assert.deepEqual(svg, read(selected.source));
  for (const value of [hash(svg), selected.sourceSha256, selected.svgSha256, provenance.svgSha256]) assert.equal(value, svgHash);
  assert.deepEqual(metadata.originalDimensions, [1086, 1448]);
  assert.deepEqual(metadata.rasterDimensions, metadata.originalDimensions);
  assert.deepEqual(metadata.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.deepEqual(selected.crop, metadata.originalCrop);
  assert.equal(metadata.rasterMimeType, 'image/avif');
  const raster = Buffer.from(markup.match(/data:image\/avif;base64,([^"\s]+)/)[1], 'base64');
  assert.equal(raster.length, 105717);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [1086, 1448]);
  assert.deepEqual(raster, read(metadata.rasterSource));
  for (const value of [hash(raster), metadata.rasterSha256, provenance.rasterSha256]) assert.equal(value, rasterHash);
  assert.deepEqual(metadata.encoding, { encoder: 'Pillow 12.3.0 AVIF', quality: 50, speed: 0, lossless: false });
  assert.match(metadata.processing, /no resizing, crop, redraw, colour edit, added border or tile texture/);
});

test('White Dragon preloads once while hidden tiles, geometry and other face sets are unchanged', () => {
  assert.equal(tileImage('5z', 'dali'), 'tiles/dali/approved/Haku.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === 'tiles/dali/approved/Haku.svg').length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('5z', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('5z', 'classic'), 'tiles/Haku.svg');
  assert.equal(tileImage('5z', 'matisse'), 'tiles/matisse/approved/Haku.svg');
  assert.equal(tileImage('5z', 'van-gogh'), 'tiles/van-gogh/approved/Haku.svg');
  assert.match(markup, /viewBox="0 0 300 400"/);
  assert.match(markup, /rx="26"/);
  assert.match(markup, /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(markup, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('White Dragon is reproducibly exported and previewed with exact coverage of all 34 identities', () => {
  assert.deepEqual(set.tiles.map(entry => entry.tile), [...DALI_APPROVED]);
  assert.equal(DALI_APPROVED.filter(tile => tile === '5z').length, 1);
  assert.deepEqual([...set.tiles, ...set.placeholders].map(entry => entry.tile).sort(), [...TILE_TYPES].sort());
  assert.equal(set.placeholders.some(entry => entry.tile === '5z'), false);
  assert.equal(set.fallback, 'text');
  for (const { tile } of set.placeholders) assert.equal(tileImage(tile, 'dali'), null);
  const exporter = read('web/scripts/export-dali-tiles.mjs').toString();
  assert.ok(exporter.includes("['Haku', '5z', 'White dragon', 'The Fold That Woke Up — A', 'white-dragon-a-approved.svg', [0, 0, 1086, 1448]]"));
  const preview = read('web/public/tiles/dali/preview.html').toString();
  assert.ok(preview.includes('approved/Haku.svg'));
  assert.ok(preview.includes('The Fold That Woke Up'));
  assert.ok(preview.includes(`${set.tiles.length} approved faces`));
  assert.ok(preview.includes(`remaining ${set.placeholders.length} tiles show their names`));
});
