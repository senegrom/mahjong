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
const record = JSON.parse(read('docs/design/dali/eight-disks.json'));
const entry = set.tiles.find(tile => tile.tile === '8p');
const SVG_HASH = 'c25c605794ae06fce15a758ce34840ef555d33af1558d58f43e017d5a448c724';
const RASTER_HASH = 'e1c99935fa07713f8a2c37175090036dde471997573f78ee7dbcf713607b55e0';
const ORIGINAL_HASH = '6ec8fa78144f1e2089534b241533d861a80a299ab97cafedd84b28a6a6a4e863';
const CROP_HASH = 'c44b62605efe0c7b4c2f45611e8ccb1671babd99c4ee08dcecf683bb1f2a38a2';

test('eight disks uses approved cupboard C, not the curtain or butterfly', () => {
  assert.ok(entry);
  assert.equal(entry.name, 'Pin8');
  assert.equal(entry.direction, 'The Cabinet of Captured Moons — C');
  assert.equal(record.selectedOption, 'C (right), the cupboard with eight disks');
  assert.equal(entry.source, 'docs/design/dali/studies/eight-disks-c-cabinet-approved.svg');
  assert.equal(record.source, entry.source);
  assert.equal(record.runtime, 'web/public/tiles/dali/approved/Pin8.svg');
  assert.equal(record.originalFilename, 'a_wide_triptych_surrealist_dreamlike_illustratio.png');
  assert.equal(record.originalSha256, ORIGINAL_HASH);
  assert.equal(record.cropSha256, CROP_HASH);
  assert.deepEqual(read(record.source), read(record.runtime));
  for (const value of [hash(read(record.source)), record.svgSha256, entry.sourceSha256, entry.svgSha256]) assert.equal(value, SVG_HASH);
  assert.deepEqual(record.countedObjects, { count: 8, arrangement: [2, 2, 2, 2], subject: 'moon-disks in two columns of four cupboard compartments' });
});

test('the cupboard crop retains its 609 by 812 pixels and exact 3:4 proportions', () => {
  const svg = read(record.runtime).toString();
  const metadata = JSON.parse(svg.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
  const raster = Buffer.from(svg.match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/)[1], 'base64');
  assert.equal(hash(raster), RASTER_HASH);
  assert.equal(record.rasterSha256, RASTER_HASH);
  assert.deepEqual(raster, read(record.rasterSource));
  assert.equal(raster.length, 42787);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [609, 812]);
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 609, height: 812 });
  assert.deepEqual(record.originalDimensions, [1881, 836]);
  assert.deepEqual(record.selectedPanel, { x: 1271, y: 0, width: 610, height: 836 });
  assert.deepEqual(record.originalCrop, { x: 1271, y: 0, width: 609, height: 812 });
  assert.deepEqual(record.rasterDimensions, [609, 812]);
  assert.equal(entry.crop.width / entry.crop.height, 3 / 4);
  assert.equal(entry.fit, undefined);
  assert.equal(record.originalStoredInRepository, false);
  assert.equal(record.encoding.lossless, false);
  assert.equal(record.encoding.quality, 50);
  const embeddedRecord = { ...record };
  delete embeddedRecord.svgSha256;
  assert.deepEqual(metadata, embeddedRecord);
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.equal((svg.match(/<image /g) ?? []).length, 1);
  assert.doesNotMatch(svg, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('eight disks is registered and preloaded once without changing neighboring selections or hidden tiles', () => {
  assert.equal(set.tiles.length, 27);
  assert.equal(set.placeholders.length, 7);
  assert.deepEqual(set.tiles.map(tile => tile.tile), [...DALI_APPROVED]);
  assert.deepEqual([...set.tiles, ...set.placeholders].map(tile => tile.tile).sort(), [...TILE_TYPES].sort());
  assert.equal(set.tiles.filter(tile => tile.tile === '8p').length, 1);
  for (let rank = 1; rank <= 8; rank++) assert.ok(DALI_APPROVED.includes(`${rank}p`));
  assert.equal(set.tiles.find(tile => tile.tile === '7p').direction, 'The Shell That Holds the Sea — A');
  assert.equal(set.tiles.find(tile => tile.tile === '5z').direction, 'The Fold That Woke Up — A');
  const image = tileImage('8p', 'dali');
  assert.equal(image, 'tiles/dali/approved/Pin8.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === image).length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('8p', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('8p', 'classic'), 'tiles/Pin8.svg');
  assert.equal(tileImage('8p', 'matisse'), 'tiles/matisse/approved/Pin8.svg');
  assert.equal(tileImage('8p', 'van-gogh'), 'tiles/van-gogh/approved/Pin8.svg');
});

test('the cupboard appears in the gallery and preserves the existing fourteen-tile example hand', () => {
  const preview = read('web/public/tiles/dali/preview.html').toString();
  assert.match(preview, /27 approved faces/);
  assert.match(preview, /The Cabinet of Captured Moons — C/);
  assert.match(preview, /Ranks 1–9 of disks are complete/);
  assert.match(preview, /remaining 7 tiles/);
  const gallery = preview.match(/<div class="gallery">([\s\S]*?)<\/div>/)[1];
  assert.match(gallery, /approved\/Pin8\.svg/);
  const hand = preview.match(/<div class="hand">([\s\S]*?)<\/div>/)[1];
  const names = [...hand.matchAll(/src="approved\/([^"/]+)\.svg"/g)].map(match => match[1]);
  assert.deepEqual(names, ['Sou3', 'Sou4', 'Ton', 'Pei', 'Pin6', 'Sou9', 'Pin4', 'Pin2', 'Man5', 'Man6', 'Man7', 'Man8', 'Man9', 'Chun']);
});

test('--only=8p repeats exactly, rejects unknown identities and preserves every other face', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'dali-eight-cabinet-'));
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
      if (tile.tile !== '8p') protectedFiles.push([relative, bytes]);
    }
  }
  const exporter = path.join(temporary, 'web/scripts/export-dali-tiles.mjs');
  const run = selection => execFileSync(process.execPath, [exporter, selection], { cwd: temporary, stdio: 'pipe' });
  run('--only=8p');
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.source));
  for (const [relative, bytes] of protectedFiles) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes, relative);
  const manifestPath = path.join(temporary, 'web/public/tiles/dali/manifest.json');
  assert.deepEqual(JSON.parse(readFileSync(manifestPath)), set);
  const generatedFiles = [manifestPath, path.join(temporary, record.runtime), path.join(temporary, 'web/public/tiles/dali/preview.html'), path.join(temporary, 'web/src/lib/dali-faces.js')];
  const first = generatedFiles.map(file => readFileSync(file));
  run('--only=8p');
  generatedFiles.forEach((file, index) => assert.deepEqual(readFileSync(file), first[index]));
  for (const [relative, bytes] of protectedFiles) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes, relative);
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
  generatedFiles.forEach((file, index) => assert.deepEqual(readFileSync(file), first[index]));
});
