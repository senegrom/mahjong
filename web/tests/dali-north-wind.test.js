import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { DALI_APPROVED } from '../src/lib/dali-faces.js';
import { TILE_FACE_OPTIONS, TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = new URL('../../', import.meta.url);
const read = path => readFileSync(new URL(path, root));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/dali/manifest.json'));
const provenance = JSON.parse(read('docs/design/dali/north-wind.json'));
const selection = JSON.parse(read('docs/design/dali/north-wind-selection.json'));
const selected = set.tiles.find(entry => entry.tile === '4z');
const svg = read('web/public/tiles/dali/approved/Pei.svg');
const markup = svg.toString();
const metadata = JSON.parse(markup.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);

test('North uses the approved readable 北 recreation, not a renamed East image', () => {
  assert.equal(selected.name, 'Pei');
  assert.equal(selected.direction, 'The Wind-Carved Arch');
  assert.equal(selected.status, 'approved');
  assert.equal(selection.status, 'approved');
  assert.equal(selection.selectedPanel, 'second (middle)');
  assert.equal(metadata.glyph, '北');
  assert.equal(metadata.selectedPanel, selection.selectedPanel);
  assert.equal(metadata.selectedBoardSha256, selection.originalSha256);
  assert.equal(metadata.originalSha256, selection.adaptationSha256);
  assert.equal(selection.adaptationFilename, 'stone_north_in_a_dreamscape.png');
  assert.equal(metadata.originalFilename, selection.adaptationFilename);
  assert.equal(metadata.revision, 'Recreated stone North with clearly separated strokes');
  assert.notDeepEqual(svg, read('web/public/tiles/dali/approved/Ton.svg'));
  assert.equal(DALI_APPROVED.filter(tile => tile === '4z').length, 1);
  assert.equal(set.placeholders.some(entry => entry.tile === '4z'), false);
});

test('North preserves the complete approved recreation and pins its source, runtime and raster', () => {
  assert.equal(selected.source, 'docs/design/dali/studies/north-wind-second-approved.svg');
  assert.deepEqual(svg, read(selected.source));
  const svgHash = '14f156505fae005791c8f6d2aea9a8c0d9f8dcce60f915055e17729a9ab51ec8';
  assert.equal(hash(svg), svgHash);
  assert.equal(selected.sourceSha256, svgHash);
  assert.equal(selected.svgSha256, svgHash);
  assert.equal(provenance.svgSha256, svgHash);
  assert.equal(metadata.originalSha256, '0c514cd4e8c83ba7f0e57889b953d97d1a2a3918fa71ae721e12a2ab04430c65');
  assert.equal(metadata.originalSha256, provenance.originalSha256);
  assert.deepEqual(metadata.originalDimensions, [1086, 1448]);
  assert.deepEqual(metadata.originalDimensions, provenance.originalDimensions);
  assert.deepEqual(metadata.rasterDimensions, [450, 600]);
  assert.deepEqual(metadata.rasterDimensions, provenance.rasterDimensions);
  const raster = Buffer.from(markup.match(/data:image\/webp;base64,([^"\s]+)/)[1], 'base64');
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 12), 'WEBP');
  assert.equal(raster.toString('ascii', 12, 16), 'VP8 ');
  assert.equal(raster.readUInt16LE(26) & 0x3fff, 450);
  assert.equal(raster.readUInt16LE(28) & 0x3fff, 600);
  assert.equal(hash(raster), '4baea275a85b5adeed19321ddf728f3c7cdff8d1f9a013afa66b10436b23d65f');
  assert.equal(hash(raster), metadata.rasterSha256);
  assert.equal(hash(raster), provenance.rasterSha256);
  assert.equal(metadata.processing, provenance.processing);
  assert.match(metadata.processing, /complete portrait resized with Lanczos/);
  assert.match(metadata.processing, /No new redraw, tile background, border or colour changes/);
  assert.equal(provenance.supersededOriginalSha256, '6422a05ce9bbb980eb28289fa03080bd70c1eeb7fae54018c8612e5e62def9a5');
  assert.notEqual(metadata.originalSha256, provenance.supersededOriginalSha256);
});

test('North preloads once, stays self-contained and never exposes a hidden tile', () => {
  assert.equal(tileImage('4z', 'dali'), 'tiles/dali/approved/Pei.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('4z', 'dali')).length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('4z', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('4z', 'classic'), 'tiles/Pei.svg');
  assert.equal(tileImage('4z', 'matisse'), 'tiles/matisse/approved/Pei.svg');
  assert.equal(tileImage('4z', 'van-gogh'), 'tiles/van-gogh/approved/Pei.svg');
  assert.match(markup, /viewBox="0 0 300 400"/);
  assert.match(markup, /rx="26"/);
  assert.match(markup, /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(markup, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('North is registered in the reproducible exporter and appears in the preview', () => {
  assert.equal(set.tiles.length, 19);
  assert.equal(set.placeholders.length, 15);
  assert.deepEqual(set.tiles.map(entry => entry.tile), [...DALI_APPROVED]);
  // The crop is the SVG's logical canvas, not its higher-density embedded raster.
  assert.deepEqual(selected.crop, { x: 0, y: 0, width: 300, height: 400 });
  const exporter = read('web/scripts/export-dali-tiles.mjs').toString();
  assert.match(exporter, /\['Pei', '4z', 'North wind', 'The Wind-Carved Arch', 'north-wind-second-approved.svg', \[0, 0, 300, 400\]\]/);
  const preview = read('web/public/tiles/dali/preview.html').toString();
  assert.ok(preview.includes('approved/Pei.svg'));
  assert.ok(preview.includes('The Wind-Carved Arch'));
  assert.ok(preview.includes('19 approved faces'));
  assert.ok(preview.includes('remaining 15 tiles'));
});
