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
const record = JSON.parse(read('docs/design/dali/seven-disks.json'));
const entry = set.tiles.find(tile => tile.tile === '7p');
const SVG_HASH = '1d6bb184b46ad23b34c2454904287a56e11f991b1866645a194238848ed00fa2';
const RASTER_HASH = '021e5788c3e11ed089ed79981498c97d575a60a04eb439427143e5b0b2fe3664';
const ORIGINAL_HASH = '95a9b7185cb45732aa6f60cf2de81580482fd1e25b6b0bceb47f8b85329a6d23';

test('seven disks uses approved seashell A, not the buttoned sky or ivory instrument', () => {
  assert.ok(entry);
  assert.equal(entry.name, 'Pin7');
  assert.equal(entry.direction, 'The Shell That Holds the Sea — A');
  assert.equal(record.selectedOption, 'A (first), seven pearls in a seashell');
  assert.equal(entry.source, 'docs/design/dali/studies/seven-disks-a-shell-approved.svg');
  assert.equal(record.source, entry.source);
  assert.equal(record.runtime, 'web/public/tiles/dali/approved/Pin7.svg');
  assert.equal(record.originalFilename, 'seven_pearls_in_a_dreaming_seashell.png');
  assert.equal(record.originalSha256, ORIGINAL_HASH);
  assert.deepEqual(read(record.source), read(record.runtime));
  assert.equal(hash(read(record.source)), SVG_HASH);
  assert.equal(record.svgSha256, SVG_HASH);
  assert.equal(entry.sourceSha256, SVG_HASH);
  assert.equal(entry.svgSha256, SVG_HASH);
  assert.deepEqual(record.countedObjects, { count: 7, arrangement: [3, 1, 3], subject: 'pearls inside the seashell' });
});

test('the entire 1086 by 1448 seashell portrait is retained in the full-dimension AVIF', () => {
  const svg = read(record.runtime).toString();
  const metadata = JSON.parse(svg.match(/<metadata>([\s\S]*?)<\/metadata>/)[1]);
  const raster = Buffer.from(svg.match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/)[1], 'base64');
  assert.equal(hash(raster), RASTER_HASH);
  assert.equal(record.rasterSha256, RASTER_HASH);
  assert.deepEqual(raster, read(record.rasterSource));
  assert.equal(raster.length, 112979);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [1086, 1448]);
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.deepEqual(record.originalDimensions, [1086, 1448]);
  assert.deepEqual(record.rasterDimensions, [1086, 1448]);
  assert.deepEqual(record.originalCrop, entry.crop);
  assert.equal(record.originalStoredInRepository, false);
  assert.equal(record.encoding.lossless, false);
  assert.equal(record.encoding.quality, 50);
  assert.equal(record.encoding.speed, 0);
  const embeddedRecord = { ...record };
  delete embeddedRecord.svgSha256;
  assert.deepEqual(metadata, embeddedRecord);
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.equal((svg.match(/<image /g) ?? []).length, 1);
  assert.doesNotMatch(svg, /(?:href|src)=["']https?:|<script|<foreignObject/);
});

test('seven disks is registered and preloaded once while hidden tiles and other sets remain intact', () => {
  assert.equal(set.tiles.length, 26);
  assert.equal(set.placeholders.length, 8);
  assert.deepEqual(set.tiles.map(tile => tile.tile), [...DALI_APPROVED]);
  assert.deepEqual([...set.tiles, ...set.placeholders].map(tile => tile.tile).sort(), [...TILE_TYPES].sort());
  assert.equal(set.tiles.filter(tile => tile.tile === '7p').length, 1);
  for (let rank = 1; rank <= 7; rank++) assert.ok(DALI_APPROVED.includes(`${rank}p`));
  const image = tileImage('7p', 'dali');
  assert.equal(image, 'tiles/dali/approved/Pin7.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === image).length, 1);
  for (const { value } of TILE_FACE_OPTIONS) assert.equal(tileImage('7p', value, true), 'tiles/Back.svg');
  assert.equal(tileImage('7p', 'classic'), 'tiles/Pin7.svg');
  assert.equal(tileImage('7p', 'matisse'), 'tiles/matisse/approved/Pin7.svg');
  assert.equal(tileImage('7p', 'van-gogh'), 'tiles/van-gogh/approved/Pin7.svg');
});

test('the shell is in the gallery without replacing a tile in the existing fourteen-tile example hand', () => {
  const preview = read('web/public/tiles/dali/preview.html').toString();
  assert.match(preview, /26 approved faces/);
  assert.match(preview, /The Shell That Holds the Sea — A/);
  assert.match(preview, /Ranks 1–8 of disks are complete/);
  assert.match(preview, /remaining 8 tiles/);
  const gallery = preview.match(/<div class="gallery">([\s\S]*?)<\/div>/)[1];
  assert.match(gallery, /approved\/Pin7\.svg/);
  const hand = preview.match(/<div class="hand">([\s\S]*?)<\/div>/)[1];
  const names = [...hand.matchAll(/src="approved\/([^"/]+)\.svg"/g)].map(match => match[1]);
  assert.deepEqual(names, ['Sou3', 'Sou4', 'Ton', 'Pei', 'Pin6', 'Sou9', 'Pin4', 'Pin2', 'Man5', 'Man6', 'Man7', 'Man8', 'Man9', 'Chun']);
});

test('--only=7p is deterministic, rejects unknown identities and never rewrites another face', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'dali-seven-shell-'));
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
      if (tile.tile !== '7p') protectedFiles.push([relative, bytes]);
    }
  }
  const exporter = path.join(temporary, 'web/scripts/export-dali-tiles.mjs');
  const run = selection => execFileSync(process.execPath, [exporter, selection], { cwd: temporary, stdio: 'pipe' });
  run('--only=7p');
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.source));
  for (const [relative, bytes] of protectedFiles) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes, relative);
  const manifestPath = path.join(temporary, 'web/public/tiles/dali/manifest.json');
  assert.deepEqual(JSON.parse(readFileSync(manifestPath)), set);
  const generatedFiles = [manifestPath, path.join(temporary, record.runtime), path.join(temporary, 'web/public/tiles/dali/preview.html')];
  const first = generatedFiles.map(file => readFileSync(file));
  run('--only=7p');
  generatedFiles.forEach((file, index) => assert.deepEqual(readFileSync(file), first[index]));
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
  generatedFiles.forEach((file, index) => assert.deepEqual(readFileSync(file), first[index]));
});
