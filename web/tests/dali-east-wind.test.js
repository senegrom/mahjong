import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { DALI_APPROVED } from '../src/lib/dali-faces.js';
import { TILE_FACE_OPTIONS, TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = new URL('../../', import.meta.url);
const read = path => readFileSync(new URL(path, root));
const set = JSON.parse(read('web/public/tiles/dali/manifest.json'));
const provenance = JSON.parse(read('docs/design/dali/east-wind.json'));
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const selected = set.tiles.find(entry => entry.tile === '1z');
const svg = read('web/public/tiles/dali/approved/Ton.svg');
const metadata = JSON.parse(svg.toString().match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);

test('the approved first wind painting is East, distinct from the selected North adaptation', () => {
  assert.equal(selected.name, 'Ton');
  assert.equal(selected.direction, 'The Dreaming East');
  assert.equal(selected.status, 'approved');
  assert.equal(tileImage('1z', 'dali'), 'tiles/dali/approved/Ton.svg');
  assert.equal(DALI_APPROVED.includes('4z'), true);
  assert.equal(tileImage('4z', 'dali'), 'tiles/dali/approved/Pei.svg');
  assert.notDeepEqual(svg, read('web/public/tiles/dali/approved/Pei.svg'));
  const north = JSON.parse(read('docs/design/dali/north-wind-selection.json'));
  assert.equal(north.tile, '4z');
  assert.equal(north.selectedPanel, 'second (middle)');
  assert.equal(north.status, 'approved');
});

test('East source, runtime and embedded approved image are hash-pinned', () => {
  assert.deepEqual(svg, read(selected.source));
  assert.equal(sha256(svg), '8e60789814934608ce33f7dfae8959d14b30d045fb97b3529e5cf9dbb5e6dd48');
  assert.equal(selected.sourceSha256, provenance.svgSha256);
  assert.equal(selected.svgSha256, provenance.svgSha256);
  assert.equal(metadata.originalSha256, '343c96e9ebcd5c84f0c86923327c2ea6ea0952feb4e18587e6888e88f4f2a622');
  assert.deepEqual(metadata.crop, [0, 85, 558, 864]);
  assert.deepEqual(metadata.paddedPortraitDimensions, [648, 864]);
  assert.deepEqual(metadata.padding, { left: 45, right: 45 });
  const raster = Buffer.from(svg.toString().match(/data:image\/webp;base64,([^"\s]+)/)[1], 'base64');
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 12), 'WEBP');
  assert.equal(sha256(raster), '5345a14add4224499edc9e94338c9dc45a8a205caa227789fa153fc1369b9686');
  assert.equal(metadata.rasterSha256, provenance.rasterSha256);
});

test('East uses the existing clipping and preload without exposing hidden tiles', () => {
  assert.deepEqual(metadata.rasterDimensions, [300, 400]);
  assert.match(svg.toString(), /viewBox="0 0 300 400"/);
  assert.match(svg.toString(), /rx="26"/);
  assert.match(svg.toString(), /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(svg.toString(), /(?:href|src)=["']https?:|<script|<foreignObject/);
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('1z', 'dali')).length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('1z', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('1z', 'classic'), 'tiles/Ton.svg');
  assert.equal(tileImage('1z', 'matisse'), 'tiles/matisse/approved/Ton.svg');
  assert.equal(tileImage('1z', 'van-gogh'), 'tiles/van-gogh/approved/Ton.svg');
});

test('the registry covers all tiles once and East is reproducibly exported and previewed', () => {
  assert.equal(set.tiles.length, 20);
  assert.equal(set.placeholders.length, 14);
  assert.deepEqual(set.tiles.map(entry => entry.tile), [...DALI_APPROVED]);
  assert.deepEqual([...set.tiles, ...set.placeholders].map(entry => entry.tile).sort(), [...TILE_TYPES].sort());
  const exporter = read('web/scripts/export-dali-tiles.mjs').toString();
  assert.match(exporter, /\['Ton', '1z', 'East wind', 'The Dreaming East', 'east-wind-first-approved.svg', \[0, 0, 300, 400\]\]/);
  const preview = read('web/public/tiles/dali/preview.html').toString();
  assert.ok(preview.includes('approved/Ton.svg'));
  assert.ok(preview.includes('20 approved faces'));
  assert.ok(preview.includes('remaining 14 tiles'));
});
