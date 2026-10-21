import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { DALI_APPROVED } from '../src/lib/dali-faces.js';
import { TILE_FACE_OPTIONS, TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';
import { faceImage, placeStudyImage } from '../scripts/face-fit.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/dali/manifest.json'));
const record = JSON.parse(read('docs/design/dali/one-character.json'));
const entry = set.tiles.find(tile => tile.tile === '1m');
const SOURCE_HASH = 'a33eb1b7f79fe603f59d363b93650428f1b10b0ff159bbbabdf89367261c03a3';
const RUNTIME_HASH = 'be55c915b9e280937e947967f3047ea3acca5de76a2832c48044a6a1a47c0f34';
const RASTER_HASH = '7aba9e73d7754bed3274235ebc262d0b73579aa21c9a0acdabef40dfca82077f';

test('one character uses selected aqueduct A, not the zipper B or brush C', () => {
  assert.ok(entry);
  assert.equal(entry.name, 'Man1');
  assert.equal(entry.direction, 'The Aqueduct That Flows Upward — A');
  assert.equal(record.selectedOption, 'A (left), the aqueduct');
  assert.equal(entry.source, 'docs/design/dali/studies/one-character-a-aqueduct-approved.svg');
  assert.equal(record.source, entry.source);
  assert.equal(record.runtime, 'web/public/tiles/dali/approved/Man1.svg');
  assert.equal(record.originalFilename, 'dalí_inspired_mahjong_three_surreal_horizons.png');
  assert.equal(record.originalSha256, '1047241d3c1451d00d92c3ebbff04eeca8ff4afabf153390dd1819c355218287');
  assert.equal(record.cropSha256, 'f08dc9d99b8574679908ea4821bd9c48174eeb9a15122b4df08e0db9c94cd7d6');
  for (const value of [hash(read(record.source)), record.sourceSvgSha256, entry.sourceSha256]) assert.equal(value, SOURCE_HASH);
  for (const value of [hash(read(record.runtime)), record.runtimeSvgSha256, entry.svgSha256]) assert.equal(value, RUNTIME_HASH);
});

test('the full artwork panel remains at 454 by 804 pixels in the self-contained raster', () => {
  const svg = read(record.source).toString();
  const raster = Buffer.from(svg.match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/)[1], 'base64');
  assert.equal(hash(raster), RASTER_HASH);
  assert.equal(record.rasterSha256, RASTER_HASH);
  assert.deepEqual(raster, read(record.rasterSource));
  assert.equal(raster.length, 41959);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [454, 804]);
  assert.deepEqual(record.originalDimensions, [1448, 1086]);
  assert.deepEqual(record.originalCrop, { x: 24, y: 190, width: 454, height: 804 });
  assert.deepEqual(record.rasterDimensions, [454, 804]);
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 454, height: 804 });
  assert.equal(record.originalStoredInRepository, false);
  assert.equal(record.encoding.lossless, false);
  const metadata = { ...record };
  delete metadata.sourceSvgSha256;
  delete metadata.runtimeSvgSha256;
  assert.deepEqual(JSON.parse(svg.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]), metadata);
});

test('the face changes only placement, using the documented two-percent fit', () => {
  const expected = { x: 0, y: -80.57, width: 300, height: 520.86 };
  assert.deepEqual(record.fit, { painting: [454, 804], stretch: 0.02, anchor: 2 / 3, bleedPixels: 0 });
  assert.deepEqual(faceImage(record.fit, [454, 804]), expected);
  assert.deepEqual(entry.fit.painting, { width: 454, height: 804 });
  assert.equal(entry.fit.stretch, 0.02);
  assert.equal(entry.fit.anchor, 2 / 3);
  assert.equal(entry.fit.bleedPixels, 0);
  assert.deepEqual(entry.fit.image, expected);
  assert.deepEqual(read(record.runtime), placeStudyImage(read(record.source), expected));
  const svg = read(record.runtime).toString();
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.equal((svg.match(/<image /g) ?? []).length, 1);
  assert.doesNotMatch(svg, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('one character preloads once and preserves complete suits, dragons and hidden faces', () => {
  assert.equal(set.tiles.length, 29);
  assert.equal(set.placeholders.length, 5);
  assert.deepEqual(set.tiles.map(tile => tile.tile), [...DALI_APPROVED]);
  assert.deepEqual([...set.tiles, ...set.placeholders].map(tile => tile.tile).sort(), [...TILE_TYPES].sort());
  assert.deepEqual(set.placeholders.map(tile => tile.tile).sort(), ['2m', '2z', '3m', '3z', '4m']);
  const url = 'tiles/dali/approved/Man1.svg';
  assert.equal(tileImage('1m', 'dali'), url);
  assert.equal(DALI_APPROVED.filter(tile => tile === '1m').length, 1);
  assert.equal(TILE_IMAGE_URLS.filter(value => value === url).length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('1m', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('1m', 'classic'), 'tiles/Man1.svg');
  for (const face of ['matisse', 'van-gogh']) assert.equal(tileImage('1m', face), `tiles/${face}/approved/Man1.svg`);
  for (const suit of ['p', 's']) for (let n = 1; n <= 9; n++) assert.ok(DALI_APPROVED.includes(`${n}${suit}`));
  for (const tile of ['5z', '6z', '7z']) assert.ok(DALI_APPROVED.includes(tile));
});

test('the aqueduct is in the gallery without changing the fourteen-tile example hand', () => {
  const preview = read('web/public/tiles/dali/preview.html').toString();
  for (const text of ['29 approved faces', 'The Aqueduct That Flows Upward', 'remaining 5 tiles']) assert.ok(preview.includes(text), text);
  assert.match(preview.match(/<div class="gallery">([\s\S]*?)<\/div>/)[1], /approved\/Man1\.svg/);
  const hand = preview.match(/<div class="hand">([\s\S]*?)<\/div>/)[1];
  assert.deepEqual([...hand.matchAll(/src="approved\/([^"/]+)\.svg"/g)].map(match => match[1]),
    ['Sou3', 'Sou4', 'Ton', 'Pei', 'Pin6', 'Sou9', 'Pin4', 'Pin2', 'Man5', 'Man6', 'Man7', 'Man8', 'Man9', 'Chun']);
});

test('--only=1m repeats in isolation and never rewrites another face', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'dali-aqueduct-'));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const put = (relative, bytes) => {
    const destination = path.join(temporary, relative);
    mkdirSync(path.dirname(destination), { recursive: true });
    writeFileSync(destination, bytes);
  };
  put('package.json', '{"type":"module"}');
  for (const relative of ['web/scripts/export-dali-tiles.mjs', 'web/scripts/face-fit.mjs', 'web/src/lib/tiles.js']) put(relative, read(relative));
  const protectedFiles = [];
  for (const tile of set.tiles) {
    put(tile.source, read(tile.source));
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/dali/${file}`;
      const bytes = read(relative);
      put(relative, bytes);
      if (tile.tile !== '1m') protectedFiles.push([relative, bytes]);
    }
  }
  const exporter = path.join(temporary, 'web/scripts/export-dali-tiles.mjs');
  const run = selection => execFileSync(process.execPath, [exporter, selection], { cwd: temporary, stdio: 'pipe' });
  const paths = [record.runtime, 'web/public/tiles/dali/manifest.json', 'web/public/tiles/dali/preview.html', 'web/src/lib/dali-faces.js'];
  for (let attempt = 0; attempt < 2; attempt++) {
    run('--only=1m');
    for (const relative of paths) assert.deepEqual(readFileSync(path.join(temporary, relative)), read(relative), relative);
    for (const [relative, bytes] of protectedFiles) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes, relative);
  }
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
  for (const relative of paths) assert.deepEqual(readFileSync(path.join(temporary, relative)), read(relative), relative);
});
