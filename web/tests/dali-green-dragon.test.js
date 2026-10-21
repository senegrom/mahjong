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

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/dali/manifest.json'));
const record = JSON.parse(read('docs/design/dali/green-dragon.json'));
const entry = set.tiles.find(tile => tile.tile === '6z');
const SVG_HASH = 'd25dd23f919dff9f8fc700c4c45527f31d944a858c8c2479bebd9a7684500205';
const RASTER_HASH = '928ca85d303521af0996374e4e4380d030cc7d20239ad9164c7f48d97658fea9';

test('Green Dragon is the selected latest greener triptych middle B, not an earlier standalone or A/C', () => {
  assert.ok(entry);
  assert.equal(entry.name, 'Hatsu');
  assert.equal(entry.direction, 'The River That Escaped — Green B');
  assert.equal(record.selectedOption, 'B (middle), greener river dragon from the latest triptych');
  assert.equal(record.originalFilename, 'a_wide_triptych_fantasy_surreal_digital_painting.png');
  assert.equal(record.originalSha256, 'c58942a46c6833a55b2e15e6bda3d51e8ae7a429c3745884ae3a69d7eb72e768');
  assert.equal(record.cropSha256, '91cf1d2c3f93d7bdeb39a734fd18d8e24bdf7f6b090b48d0e63ba050c54f71da');
  assert.equal(entry.source, 'docs/design/dali/studies/green-dragon-b-river-approved.svg');
  assert.equal(record.source, entry.source);
  assert.equal(record.runtime, 'web/public/tiles/dali/approved/Hatsu.svg');
  assert.deepEqual(read(record.source), read(record.runtime));
  for (const value of [hash(read(record.source)), record.svgSha256, entry.sourceSha256, entry.svgSha256]) assert.equal(value, SVG_HASH);
});

test('the 3:4 crop removes separators and option letters with no resizing or further colour edit', () => {
  const svg = read(record.runtime).toString();
  const raster = Buffer.from(svg.match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/)[1], 'base64');
  assert.equal(hash(raster), RASTER_HASH);
  assert.equal(record.rasterSha256, RASTER_HASH);
  assert.deepEqual(raster, read(record.rasterSource));
  assert.equal(raster.length, 60307);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [606, 808]);
  assert.deepEqual(record.originalDimensions, [1881, 836]);
  assert.deepEqual(record.originalCrop, { x: 639, y: 0, width: 606, height: 808 });
  assert.deepEqual(record.rasterDimensions, [606, 808]);
  assert.equal(606 / 808, 3 / 4);
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 606, height: 808 });
  assert.equal(record.originalStoredInRepository, false);
  assert.deepEqual(record.encoding, { encoder: 'Pillow 12.3.0 AVIF', quality: 50, speed: 0, lossless: false });
  const expected = { ...record };
  delete expected.svgSha256;
  assert.deepEqual(JSON.parse(svg.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]), expected);
  assert.match(record.processing, /No resizing, stretching, repainting, further recolouring/);
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.equal((svg.match(/<image /g) ?? []).length, 1);
  assert.doesNotMatch(svg, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('Green Dragon completes the trio, preloads once and preserves hidden faces and other artists', () => {
  assert.equal(set.tiles.length, 29);
  assert.equal(set.placeholders.length, 5);
  assert.deepEqual(set.tiles.map(tile => tile.tile), [...DALI_APPROVED]);
  assert.deepEqual([...set.tiles, ...set.placeholders].map(tile => tile.tile).sort(), [...TILE_TYPES].sort());
  for (const [tile, name] of [['5z', 'Haku'], ['6z', 'Hatsu'], ['7z', 'Chun']]) {
    assert.equal(DALI_APPROVED.filter(value => value === tile).length, 1);
    assert.equal(tileImage(tile, 'dali'), `tiles/dali/approved/${name}.svg`);
  }
  assert.equal(set.tiles.find(tile => tile.tile === '5z').direction, 'The Fold That Woke Up — A');
  assert.equal(set.tiles.find(tile => tile.tile === '7z').direction, 'Molten Ruby');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === 'tiles/dali/approved/Hatsu.svg').length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('6z', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('6z', 'classic'), 'tiles/Hatsu.svg');
  for (const face of ['matisse', 'van-gogh']) assert.equal(tileImage('6z', face), `tiles/${face}/approved/Hatsu.svg`);
  for (const { tile } of set.placeholders) assert.equal(tileImage(tile, 'dali'), null);
});

test('Green Dragon is in the gallery without changing the existing fourteen-tile example hand', () => {
  const preview = read('web/public/tiles/dali/preview.html').toString();
  for (const text of ['29 approved faces', 'The River That Escaped', 'remaining 5 tiles']) assert.ok(preview.includes(text), text);
  assert.match(preview.match(/<div class="gallery">([\s\S]*?)<\/div>/)[1], /approved\/Hatsu\.svg/);
  const hand = preview.match(/<div class="hand">([\s\S]*?)<\/div>/)[1];
  assert.deepEqual([...hand.matchAll(/src="approved\/([^"/]+)\.svg"/g)].map(match => match[1]),
    ['Sou3', 'Sou4', 'Ton', 'Pei', 'Pin6', 'Sou9', 'Pin4', 'Pin2', 'Man5', 'Man6', 'Man7', 'Man8', 'Man9', 'Chun']);
});

test('--only=6z repeats exactly in isolation without rewriting any other face', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'dali-green-river-'));
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
      if (tile.tile !== '6z') protectedFiles.push([relative, bytes]);
    }
  }
  const exporter = path.join(temporary, 'web/scripts/export-dali-tiles.mjs');
  const run = selection => execFileSync(process.execPath, [exporter, selection], { cwd: temporary, stdio: 'pipe' });
  const paths = [record.runtime, 'web/public/tiles/dali/manifest.json', 'web/public/tiles/dali/preview.html', 'web/src/lib/dali-faces.js'];
  for (let attempt = 0; attempt < 2; attempt++) {
    run('--only=6z');
    for (const relative of paths) assert.deepEqual(readFileSync(path.join(temporary, relative)), read(relative), relative);
    for (const [relative, bytes] of protectedFiles) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes, relative);
  }
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
  for (const relative of paths) assert.deepEqual(readFileSync(path.join(temporary, relative)), read(relative), relative);
});
