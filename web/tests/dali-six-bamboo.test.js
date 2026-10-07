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
const provenance = JSON.parse(read('docs/design/dali/six-bamboo.json'));
const selected = set.tiles.find(entry => entry.tile === '6s');
const svg = read('web/public/tiles/dali/approved/Sou6.svg');
const markup = svg.toString();
const metadata = JSON.parse(markup.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
const expectedSvg = '174285598c4569561efa7f0832f3b6c34fd52e653ffca6bba5dc3e941c237762';
const expectedRaster = 'b71742d237ae7bce1595ec0baecac878c666f27949a2ccceb1b696e4ee6676ea';
const expectedOriginal = '233988575615df0c4a4247b0f3ca05b67aad37a0135274b21e281fc79c280da6';

test('six bamboo is selected green C and completes the bamboo suit', () => {
  assert.equal(selected.name, 'Sou6');
  assert.equal(selected.direction, 'The Impossible Reflection — Green C');
  assert.equal(selected.status, 'approved');
  assert.equal(metadata.selectedOption, 'C (third), green-hued revision');
  assert.equal(provenance.selectedOption, metadata.selectedOption);
  for (let n = 1; n <= 9; n++) {
    assert.equal(DALI_APPROVED.filter(tile => tile === `${n}s`).length, 1);
    assert.equal(tileImage(`${n}s`, 'dali'), `tiles/dali/approved/Sou${n}.svg`);
  }
  assert.equal(set.tiles.length, 21);
  assert.equal(set.placeholders.length, 13);
  assert.deepEqual(set.tiles.map(entry => entry.tile), [...DALI_APPROVED]);
  assert.deepEqual([...set.tiles, ...set.placeholders].map(entry => entry.tile).sort(), [...TILE_TYPES].sort());
});

test('green C preserves the complete approved portrait and pins all image bytes', () => {
  assert.equal(selected.source, 'docs/design/dali/studies/six-bamboo-c-green-approved.svg');
  assert.deepEqual(svg, read(selected.source));
  for (const value of [hash(svg), selected.sourceSha256, selected.svgSha256, provenance.svgSha256]) assert.equal(value, expectedSvg);
  assert.equal(metadata.originalFilename, 'emerald_oasis_in_a_surreal_landscape.png');
  assert.equal(metadata.originalSha256, expectedOriginal);
  assert.equal(provenance.originalSha256, expectedOriginal);
  assert.equal(provenance.originalStoredInRepository, false);
  assert.deepEqual(metadata.originalDimensions, [1086, 1448]);
  assert.deepEqual(metadata.rasterDimensions, [450, 600]);
  assert.equal(metadata.rasterMimeType, 'image/webp');
  const raster = Buffer.from(markup.match(/data:image\/webp;base64,([^"\s]+)/)[1], 'base64');
  assert.equal(raster.length, 42892);
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 12), 'WEBP');
  for (const value of [hash(raster), metadata.rasterSha256, provenance.rasterSha256]) assert.equal(value, expectedRaster);
  assert.match(metadata.processing, /No crop, redraw, added tile background, border or further colour changes/);
});

test('six bamboo preloads once and keeps hidden tiles and other sets intact', () => {
  assert.equal(TILE_IMAGE_URLS.filter(url => url === 'tiles/dali/approved/Sou6.svg').length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('6s', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('6s', 'classic'), 'tiles/Sou6.svg');
  assert.equal(tileImage('6s', 'matisse'), 'tiles/matisse/approved/Sou6.svg');
  assert.equal(tileImage('6s', 'van-gogh'), 'tiles/van-gogh/approved/Sou6.svg');
  assert.match(markup, /viewBox="0 0 300 400"/);
  assert.match(markup, /rx="26"/);
  assert.match(markup, /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(markup, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('green C is registered for export and shown in the preview', () => {
  assert.deepEqual(selected.crop, { x: 0, y: 0, width: 450, height: 600 });
  const exporter = read('web/scripts/export-dali-tiles.mjs').toString();
  assert.ok(exporter.includes("['Sou6', '6s', '6 bamboo', 'The Impossible Reflection — Green C', 'six-bamboo-c-green-approved.svg', [0, 0, 450, 600]]"));
  const preview = read('web/public/tiles/dali/preview.html').toString();
  assert.ok(preview.includes('approved/Sou6.svg'));
  assert.ok(preview.includes('The Impossible Reflection'));
  assert.ok(preview.includes('21 approved faces'));
  assert.ok(preview.includes('remaining 13 tiles'));
});
