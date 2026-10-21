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
import { faceImage, fitRecord, placeStudyImage } from '../scripts/face-fit.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/dali/manifest.json'));
const record = JSON.parse(read('docs/design/dali/nine-disks.json'));
const entry = set.tiles.find(tile => tile.tile === '9p');
const SOURCE_HASH = '540a965b379aca2f7d3c2f423fac12bc0ae211f8b547629565f1422f19a47b4f';
const SVG_HASH = '2914a76cc773285991932322efb90ef0003a2d4617d91e72fe436a63558a002d';
const RASTER_HASH = '4e0cc2a4398d4132936dd9d8371c4114403a8c2ee7ae343e1f7fbcd5479aec33';

test('nine disks is the selected peacock C, not the carpet or orchard', () => {
  assert.ok(entry);
  assert.equal(entry.name, 'Pin9');
  assert.equal(entry.direction, 'The Peacock of Small Universes — C');
  assert.equal(record.selectedOption, 'C (right), the peacock with nine circular tail windows');
  assert.equal(entry.source, 'docs/design/dali/studies/nine-disks-c-peacock-approved.svg');
  assert.equal(record.source, entry.source);
  assert.equal(record.runtime, 'web/public/tiles/dali/approved/Pin9.svg');
  assert.equal(record.originalSha256, '77544c267c50835624b6393bba298734a1f2b78043292839051eba0f49562406');
  assert.equal(record.cropSha256, 'dcdde04eefbea72173ac4a00ba7ccd0bfe8fdfcabfe3a19ade93337162baa45b');
  for (const value of [hash(read(record.source)), record.sourceSha256, entry.sourceSha256]) assert.equal(value, SOURCE_HASH);
  for (const value of [hash(read(record.runtime)), record.svgSha256, entry.svgSha256]) assert.equal(value, SVG_HASH);
  assert.deepEqual(record.countedObjects, { count: 9, arrangement: [1, 2, 2, 2, 2], subject: 'circular landscape windows in the peacock tail' });
});

test('the full panel and reflection remain encoded, with a zero-stretch playable fit', () => {
  const svg = read(record.runtime).toString();
  const raster = Buffer.from(svg.match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/)[1], 'base64');
  assert.equal(hash(raster), RASTER_HASH);
  assert.equal(record.rasterSha256, RASTER_HASH);
  assert.deepEqual(raster, read(record.rasterSource));
  assert.equal(raster.length, 40634);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [462, 847]);
  assert.deepEqual(record.originalDimensions, [1448, 1086]);
  assert.deepEqual(record.originalCrop, { x: 973, y: 116, width: 462, height: 847 });
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 462, height: 847 });
  assert.deepEqual(record.fit, { painting: [462, 847], stretch: 0, anchor: 1 / 11, bleedPixels: 0 });
  const box = faceImage(record.fit, record.rasterDimensions);
  assert.deepEqual(box, { x: 0, y: -13.64, width: 300, height: 550 });
  assert.deepEqual(entry.fit, fitRecord(record.fit, record.rasterDimensions));
  assert.deepEqual(read(record.runtime), placeStudyImage(read(record.source), box));
  assert.ok(Math.abs(box.width / box.height / (462 / 847) - 1) < 0.00001);
  assert.deepEqual(record.visiblePanelRegion, { x: 0, y: 21, width: 462, height: 616 });
  assert.equal(record.originalStoredInRepository, false);
  assert.equal(record.encoding.lossless, false);
  const metadata = JSON.parse(svg.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
  const expected = { ...record };
  delete expected.sourceSha256;
  delete expected.svgSha256;
  assert.deepEqual(metadata, expected);
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.equal((svg.match(/<image /g) ?? []).length, 1);
  assert.doesNotMatch(svg, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('the peacock completes all nine disks without changing neighboring selections or hidden faces', () => {
  assert.equal(set.tiles.length, 27);
  assert.equal(set.placeholders.length, 7);
  assert.deepEqual(set.tiles.map(tile => tile.tile), [...DALI_APPROVED]);
  assert.deepEqual([...set.tiles, ...set.placeholders].map(tile => tile.tile).sort(), [...TILE_TYPES].sort());
  for (let rank = 1; rank <= 9; rank++) {
    assert.equal(DALI_APPROVED.filter(tile => tile === `${rank}p`).length, 1);
    assert.equal(tileImage(`${rank}p`, 'dali'), `tiles/dali/approved/Pin${rank}.svg`);
  }
  assert.equal(set.tiles.find(tile => tile.tile === '7p').direction, 'The Shell That Holds the Sea — A');
  assert.equal(set.tiles.find(tile => tile.tile === '8p').direction, 'The Cabinet of Captured Moons — C');
  assert.equal(set.tiles.find(tile => tile.tile === '5z').direction, 'The Fold That Woke Up — A');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === 'tiles/dali/approved/Pin9.svg').length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('9p', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('9p', 'classic'), 'tiles/Pin9.svg');
  assert.equal(tileImage('9p', 'matisse'), 'tiles/matisse/approved/Pin9.svg');
  assert.equal(tileImage('9p', 'van-gogh'), 'tiles/van-gogh/approved/Pin9.svg');
});

test('the peacock appears in the gallery without changing the fourteen-tile hand', () => {
  const preview = read('web/public/tiles/dali/preview.html').toString();
  for (const text of ['27 approved faces', 'The Peacock of Small Universes — C', 'Ranks 1–9 of disks are complete', 'remaining 7 tiles']) assert.ok(preview.includes(text), text);
  assert.match(preview.match(/<div class="gallery">([\s\S]*?)<\/div>/)[1], /approved\/Pin9\.svg/);
  const hand = preview.match(/<div class="hand">([\s\S]*?)<\/div>/)[1];
  assert.deepEqual([...hand.matchAll(/src="approved\/([^"/]+)\.svg"/g)].map(match => match[1]),
    ['Sou3', 'Sou4', 'Ton', 'Pei', 'Pin6', 'Sou9', 'Pin4', 'Pin2', 'Man5', 'Man6', 'Man7', 'Man8', 'Man9', 'Chun']);
});

test('--only=9p repeats exactly in isolation and preserves every previous face', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'dali-nine-peacock-'));
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
      if (tile.tile !== '9p') protectedFiles.push([relative, bytes]);
    }
  }
  const exporter = path.join(temporary, 'web/scripts/export-dali-tiles.mjs');
  const run = selection => execFileSync(process.execPath, [exporter, selection], { cwd: temporary, stdio: 'pipe' });
  const paths = [record.runtime, 'web/public/tiles/dali/manifest.json', 'web/public/tiles/dali/preview.html', 'web/src/lib/dali-faces.js'];
  for (let attempt = 0; attempt < 2; attempt++) {
    run('--only=9p');
    for (const relative of paths) assert.deepEqual(readFileSync(path.join(temporary, relative)), read(relative), relative);
    for (const [relative, bytes] of protectedFiles) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes, relative);
  }
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
  for (const relative of paths) assert.deepEqual(readFileSync(path.join(temporary, relative)), read(relative), relative);
});
