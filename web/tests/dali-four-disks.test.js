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
const provenance = JSON.parse(read('docs/design/dali/four-disks.json'));
const selected = set.tiles.find(entry => entry.tile === '4p');
const svg = read('web/public/tiles/dali/approved/Pin4.svg');
const markup = svg.toString();
const metadata = JSON.parse(markup.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
const originalHash = '09488a560c3d2299784740f85737f9430279a165f9bdf8c005835d2aa99e67d8';
const rasterHash = '76e9743f0478a55bf14fba62c543ec013580eaddaeac6b23b4ecd130baf70a20';
const svgHash = '8a69e8bc8b91e96808b56c801099d1dc857778e2b51fd2e66d3e00b604b11ba8';

test('four disks uses selected middle staircase B, not the portals or shadows', () => {
  assert.equal(selected.name, 'Pin4');
  assert.equal(selected.direction, 'The Soft Staircase — B');
  assert.equal(selected.status, 'approved');
  assert.equal(metadata.selectedOption, 'B (middle), staircase');
  assert.equal(provenance.selectedOption, metadata.selectedOption);
  assert.equal(metadata.originalFilename, 'surreal_jade_medallions_amid_impossible_stairways.png');
  assert.equal(metadata.originalSha256, originalHash);
  assert.equal(provenance.originalSha256, originalHash);
  assert.equal(provenance.originalStoredInRepository, false);
  assert.equal(DALI_APPROVED.filter(tile => tile === '4p').length, 1);
  assert.equal(set.placeholders.some(entry => entry.tile === '4p'), false);
});

test('the full-resolution staircase is self-contained and source/runtime bytes are pinned', () => {
  assert.equal(selected.source, 'docs/design/dali/studies/four-disks-b-approved.svg');
  assert.deepEqual(svg, read(selected.source));
  for (const value of [hash(svg), selected.sourceSha256, selected.svgSha256, provenance.svgSha256]) assert.equal(value, svgHash);
  assert.deepEqual(metadata.originalDimensions, [1086, 1448]);
  assert.deepEqual(metadata.rasterDimensions, [1086, 1448]);
  assert.equal(metadata.rasterMimeType, 'image/avif');
  const raster = Buffer.from(markup.match(/data:image\/avif;base64,([^"\s]+)/)[1], 'base64');
  assert.equal(raster.length, 66841);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [1086, 1448]);
  assert.deepEqual(raster, read(metadata.rasterSource));
  for (const value of [hash(raster), metadata.rasterSha256, provenance.rasterSha256]) assert.equal(value, rasterHash);
  assert.match(metadata.processing, /no resizing, crop, redraw, colour edit, added border or tile texture/);
  assert.equal(metadata.encoding.lossless, false);
  assert.match(markup, /viewBox="0 0 300 400"/);
  assert.match(markup, /rx="26"/);
  assert.match(markup, /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(markup, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('four disks preloads once and does not change hidden tiles or other sets', () => {
  assert.equal(tileImage('4p', 'dali'), 'tiles/dali/approved/Pin4.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('4p', 'dali')).length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('4p', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('4p', 'classic'), 'tiles/Pin4.svg');
  assert.equal(tileImage('4p', 'matisse'), 'tiles/matisse/approved/Pin4.svg');
  const vanGogh = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
  const other = vanGogh.tiles.find(entry => entry.tile === '4p');
  assert.equal(tileImage('4p', 'van-gogh'), other ? `tiles/van-gogh/${other.svg}` : 'tiles/Pin4.svg');
});

test('the staircase is registered for reproducible export, preview and complete tile coverage', () => {
  assert.equal(set.tiles.length, 21);
  assert.equal(set.placeholders.length, 13);
  assert.deepEqual(set.tiles.map(entry => entry.tile), [...DALI_APPROVED]);
  assert.deepEqual([...set.tiles, ...set.placeholders].map(entry => entry.tile).sort(), [...TILE_TYPES].sort());
  assert.deepEqual(selected.crop, { x: 0, y: 0, width: 1086, height: 1448 });
  const exporter = read('web/scripts/export-dali-tiles.mjs').toString();
  assert.ok(exporter.includes("['Pin4', '4p', '4 disks', 'The Soft Staircase — B', 'four-disks-b-approved.svg', [0, 0, 1086, 1448]]"));
  const preview = read('web/public/tiles/dali/preview.html').toString();
  for (const text of ['approved/Pin4.svg', 'The Soft Staircase', '21 approved faces', 'remaining 13 tiles']) assert.ok(preview.includes(text));
});
