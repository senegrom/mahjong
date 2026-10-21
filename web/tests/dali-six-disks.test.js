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
const provenance = JSON.parse(read('docs/design/dali/six-disks.json'));
const selected = set.tiles.find(entry => entry.tile === '6p');
const svg = read('web/public/tiles/dali/approved/Pin6.svg');
const markup = svg.toString();
const metadata = JSON.parse(markup.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
const expectedSvg = 'bfedb68eb1948f9fb2b0f5f193cebc27bf2379933f13448558c6021cc1b0dbdc';
const expectedRaster = 'd0aa2a0d99ec502db5995bb52e945d7facaa8cf2a8ebeba0918a49af699b4d7f';
const expectedOriginal = 'bbbdab7d9d3ed1435917f2e1901a9a018ea84bb69220fb8be2090f0cbf258dd3';

test('six disks is the approved 3:4 spiral staircase C, not a seashell alternative', () => {
  assert.equal(selected.name, 'Pin6');
  assert.equal(selected.status, 'approved');
  assert.equal(selected.direction, 'The Spiral Staircase — C');
  assert.equal(metadata.selectedOption, 'C (third), spiral staircase; approved 3:4 recreation');
  assert.equal(provenance.selectedOption, metadata.selectedOption);
  assert.equal(metadata.originalFilename, 'spiral_tower_of_celestial_orbs.png');
  assert.equal(metadata.originalSha256, expectedOriginal);
  assert.equal(provenance.originalSha256, expectedOriginal);
  assert.equal(provenance.originalStoredInRepository, false);
});

test('the complete full-resolution portrait is pinned in identical source and runtime SVGs', () => {
  assert.equal(selected.source, 'docs/design/dali/studies/six-disks-c-spiral-approved.svg');
  assert.deepEqual(svg, read(selected.source));
  for (const value of [hash(svg), selected.sourceSha256, selected.svgSha256, provenance.svgSha256]) assert.equal(value, expectedSvg);
  assert.deepEqual(metadata.originalDimensions, [1086, 1448]);
  assert.deepEqual(metadata.rasterDimensions, metadata.originalDimensions);
  assert.deepEqual(metadata.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.deepEqual(selected.crop, metadata.originalCrop);
  assert.equal(metadata.rasterMimeType, 'image/avif');
  assert.equal(metadata.encoding.lossless, false);
  assert.equal(metadata.encoding.quality, 50);
  assert.equal(metadata.encoding.speed, 6);
  const raster = Buffer.from(markup.match(/data:image\/avif;base64,([^"\s]+)/)[1], 'base64');
  assert.equal(raster.length, 116915);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [1086, 1448]);
  assert.deepEqual(raster, read(metadata.rasterSource));
  for (const value of [hash(raster), metadata.rasterSha256, provenance.rasterSha256]) assert.equal(value, expectedRaster);
  assert.match(metadata.processing, /no resizing, crop, redraw, colour edit, added border or tile texture/);
});

test('six disks completes ranks 1 through 6 with one entry per tile', () => {
  assert.equal(set.tiles.length, 27);
  assert.equal(set.placeholders.length, 7);
  assert.deepEqual(set.tiles.map(entry => entry.tile), [...DALI_APPROVED]);
  assert.deepEqual([...set.tiles, ...set.placeholders].map(entry => entry.tile).sort(), [...TILE_TYPES].sort());
  for (let n = 1; n <= 6; n++) {
    assert.equal(DALI_APPROVED.filter(tile => tile === `${n}p`).length, 1);
    assert.equal(tileImage(`${n}p`, 'dali'), `tiles/dali/approved/Pin${n}.svg`);
  }
  for (const entry of set.placeholders) assert.equal(tileImage(entry.tile, 'dali'), null);
});

test('six disks preloads once with unchanged face geometry and hidden-tile privacy', () => {
  assert.equal(TILE_IMAGE_URLS.filter(url => url === 'tiles/dali/approved/Pin6.svg').length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('6p', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('6p', 'classic'), 'tiles/Pin6.svg');
  assert.match(markup, /viewBox="0 0 300 400"/);
  assert.match(markup, /rx="26"/);
  assert.match(markup, /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(markup, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('the approved spiral is registered in the exporter and shown in the preview', () => {
  const exporter = read('web/scripts/export-dali-tiles.mjs').toString();
  assert.ok(exporter.includes("['Pin6', '6p', '6 disks', 'The Spiral Staircase — C', 'six-disks-c-spiral-approved.svg', [0, 0, 1086, 1448]]"));
  const preview = read('web/public/tiles/dali/preview.html').toString();
  assert.ok(preview.includes('approved/Pin6.svg'));
  assert.ok(preview.includes('The Spiral Staircase'));
  assert.ok(preview.includes('27 approved faces'));
  assert.ok(preview.includes('remaining 7 tiles show their names'));
});
