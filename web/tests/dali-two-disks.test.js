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
const provenance = JSON.parse(read('docs/design/dali/two-disks.json'));
const selected = set.tiles.find(entry => entry.tile === '2p');
const svg = read('web/public/tiles/dali/approved/Pin2.svg');
const markup = svg.toString();
const metadata = JSON.parse(markup.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
const originalHash = '533ee58b56f9a8e8d78de006c59932315c5e21ffe976c9c656337bdcc0af138b';
const rasterHash = '88bc262e1eaed0e05aeb6c0e52241639770ad12623c51851bc48d865b8bfe40d';
const svgHash = '309cf800d3679cdad4943bfdb6eb580320927d26bfd4ef1f74b69ab9de394601';

test('two disks uses the selected middle balance painting and completes ranks one through five', () => {
  assert.equal(selected.name, 'Pin2');
  assert.equal(selected.direction, 'The Impossible Balance — B');
  assert.equal(selected.status, 'approved');
  assert.equal(metadata.selectedOption, 'B (middle), impossible balance');
  assert.equal(provenance.selectedOption, metadata.selectedOption);
  assert.equal(metadata.originalFilename, 'melting_medallions_over_a_surreal_salt_flat.png');
  assert.equal(metadata.originalSha256, originalHash);
  assert.equal(provenance.originalSha256, originalHash);
  assert.equal(provenance.originalStoredInRepository, false);
  assert.equal(DALI_APPROVED.filter(tile => tile === '2p').length, 1);
  assert.equal(set.placeholders.some(entry => entry.tile === '2p'), false);
  for (let n = 1; n <= 5; n++) assert.equal(tileImage(`${n}p`, 'dali'), `tiles/dali/approved/Pin${n}.svg`);
});

test('the approved balance retains the entire full-resolution raster and hash-pinned source/runtime', () => {
  assert.equal(selected.source, 'docs/design/dali/studies/two-disks-b-approved.svg');
  assert.deepEqual(svg, read(selected.source));
  for (const value of [hash(svg), selected.sourceSha256, selected.svgSha256, provenance.svgSha256]) assert.equal(value, svgHash);
  assert.deepEqual(metadata.originalDimensions, [1086, 1448]);
  assert.deepEqual(metadata.rasterDimensions, [1086, 1448]);
  assert.deepEqual(metadata.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.equal(metadata.rasterMimeType, 'image/avif');
  const raster = Buffer.from(markup.match(/data:image\/avif;base64,([^"\s]+)/)[1], 'base64');
  assert.equal(raster.length, 96680);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [1086, 1448]);
  assert.deepEqual(raster, read(metadata.rasterSource));
  for (const value of [hash(raster), metadata.rasterSha256, provenance.rasterSha256]) assert.equal(value, rasterHash);
  assert.equal(provenance.encoding.lossless, false);
  assert.match(metadata.processing, /no resizing, crop, redraw, colour edit, added border or tile texture/);
  assert.match(markup, /viewBox="0 0 300 400"/);
  assert.match(markup, /rx="26"/);
  assert.match(markup, /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(markup, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('two disks preloads once while hidden tiles, other sets and named fallbacks remain intact', () => {
  const image = 'tiles/dali/approved/Pin2.svg';
  assert.equal(tileImage('2p', 'dali'), image);
  assert.equal(TILE_IMAGE_URLS.filter(url => url === image).length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('2p', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('2p', 'classic'), 'tiles/Pin2.svg');
  assert.equal(tileImage('2p', 'matisse'), 'tiles/matisse/approved/Pin2.svg');
  const vanGogh = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
  const other = vanGogh.tiles.find(entry => entry.tile === '2p');
  assert.equal(tileImage('2p', 'van-gogh'), other ? `tiles/van-gogh/${other.svg}` : null);
  assert.equal(set.fallback, 'text');
  for (const { tile } of set.placeholders) assert.equal(tileImage(tile, 'dali'), null);
});

test('two disks is registered for reproducible export and the twenty-nine-face preview', () => {
  assert.equal(set.tiles.length, 29);
  assert.equal(set.placeholders.length, 5);
  assert.deepEqual(set.tiles.map(entry => entry.tile), [...DALI_APPROVED]);
  assert.deepEqual([...set.tiles, ...set.placeholders].map(entry => entry.tile).sort(), [...TILE_TYPES].sort());
  assert.deepEqual(selected.crop, { x: 0, y: 0, width: 1086, height: 1448 });
  const exporter = read('web/scripts/export-dali-tiles.mjs').toString();
  assert.ok(exporter.includes("['Pin2', '2p', '2 disks', 'The Impossible Balance — B', 'two-disks-b-approved.svg', [0, 0, 1086, 1448]]"));
  const preview = read('web/public/tiles/dali/preview.html').toString();
  assert.ok(preview.includes('approved/Pin2.svg'));
  assert.ok(preview.includes('The Impossible Balance'));
  assert.ok(preview.includes('29 approved faces'));
  assert.ok(preview.includes('remaining 5 tiles show their names'));
});
