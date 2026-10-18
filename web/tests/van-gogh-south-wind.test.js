import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { faceImage, placeStudyImage } from '../scripts/face-fit.mjs';
import { TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const record = JSON.parse(read('docs/design/van-gogh/south-wind-billowing-courtyard-a.json'));
const entry = set.tiles.find(tile => tile.tile === '2z');
const settings = { painting: [393, 640], stretch: 0.02, anchor: 0.51, bleedPixels: 0 };

test('South wind is approved Billowing Courtyard A, not the boat B or gate C', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'South A');
  assert.equal(entry.name, 'Nan');
  assert.equal(record.title, 'The Billowing Courtyard');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/27-south-wind-billowing-courtyard-a-approved.svg');
  assert.equal(entry.source, record.source);
  assert.equal(record.originalSha256, 'b70f75eb381cf9c93f6b236c9e55d88eca27761bd63ccfbc0b0cf0ecb5117abc');
  assert.equal(record.selectedCropSha256, '2757c24eddbe96f30e023309168cd7d98401aaf64060b0b2e48022206701139c');
  assert.deepEqual(record.originalDimensions, [1448, 1086]);
  assert.deepEqual(record.originalCrop, { x: 95, y: 208, width: 393, height: 640 });
  assert.equal(hash(read(record.source)), '490f86658f1afc854175d81d9fe0e89c90d86e5b18b212cdb79b8fb752bc0764');
  assert.equal(hash(read(record.runtime)), '0c6f8facecbe662a034af16fe39b0aaa14bd7916bd74882cbb75292d65e74163');
  assert.equal(entry.svgSha256, record.runtimeSvgSha256);
});

test('South retains the exact encoded A painting and differs from its study only in fit', () => {
  const source = read(record.source), runtime = read(record.runtime);
  assert.deepEqual(runtime, placeStudyImage(source, faceImage(settings, [300, 400])));
  assert.deepEqual(entry.fit, record.fit);
  assert.deepEqual(entry.fit.painting, { width: 393, height: 640 });
  assert.deepEqual(entry.fit.image, { x: 0, y: -40.27, width: 300, height: 478.97 });
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 });
  const svg = runtime.toString('utf8');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  const embedded = svg.match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(raster.length, 49477);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  assert.equal(hash(raster), '44801ce1a0171dbd5acb9bab04392d331a61ebcf474792348538a669ae5fcf56');
  assert.equal(hash(raster), record.rasterSha256);
  assert.equal(hash(raster), entry.rasterSha256);
  assert.equal(entry.rasterMimeType, 'image/avif');
  assert.deepEqual(record.rasterDimensions, [300, 400]);
  assert.doesNotMatch(svg, /(?:href|src)="https?:/);
});

test('South is registered and preloaded once while prior numbered selections remain intact', () => {
  assert.equal(set.tiles.filter(tile => tile.tile === '2z').length, 1);
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '2z').length, 1);
  assert.equal(tileImage('2z', 'van-gogh'), 'tiles/van-gogh/approved/Nan.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('2z', 'van-gogh')).length, 1);
  assert.ok(!set.remaining.includes('2z'));
  assert.deepEqual(set.tiles.map(tile => tile.tile), [...VAN_GOGH_APPROVED]);
  assert.equal(set.remaining.length, 34 - VAN_GOGH_APPROVED.length);
  for (const suit of ['m', 'p', 's']) {
    for (let rank = 1; rank <= 9; rank++) assert.ok(VAN_GOGH_APPROVED.includes(`${rank}${suit}`));
  }
  for (const [tile, candidate] of [['7p', 'Seven Apples (corrected)'], ['8p', 'Eight Lemons'], ['9p', 'Disks C'], ['1m', 'Red Sail C']]) {
    assert.equal(set.tiles.find(item => item.tile === tile).candidate, candidate);
  }
});

test('the courtyard appears in the gallery without changing the fourteen-tile sample hand', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  assert.match(preview, /South A · South wind/);
  assert.match(preview, /The Billowing Courtyard A for South wind/);
  const rack = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(rack);
  assert.equal((rack.match(/<img /g) ?? []).length, 14);
  for (const name of ['Man9', 'Man8', 'Pin4', 'Man7', 'Man6', 'Man5', 'Man2', 'Man3', 'Man4', 'Pin9', 'Pin3', 'Sou9', 'Sou6', 'Sou8']) {
    assert.ok(rack.includes(`approved/${name}.svg`), name);
  }
});

test('--only=2z reproduces South A and preserves every other playable image', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-south-a-'));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const put = (relative, bytes) => {
    const target = path.join(temporary, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, bytes);
  };
  put('package.json', '{"type":"module"}');
  for (const file of ['web/scripts/export-van-gogh-tiles.mjs', 'web/scripts/face-fit.mjs', 'web/src/lib/tiles.js']) put(file, read(file));
  for (const source of set.sources) {
    put(source.source, source.source.endsWith('.svg') ? read(source.source) : Buffer.from(`source fixture: ${source.id}`));
  }
  const unchanged = [];
  for (const tile of set.tiles.filter(tile => tile.tile !== '2z')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  assert.equal(run('--only=2z').trim(), `Exported ${set.tiles.length} approved Van Gogh faces; ${set.remaining.length} identities show their names until painted.`);
  for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.runtime));
  const manifestPath = path.join(temporary, 'web/public/tiles/van-gogh/manifest.json');
  const first = readFileSync(manifestPath);
  const generated = JSON.parse(first);
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '2z'), entry);
  assert.deepEqual(generated.remaining, set.remaining);
  run('--only=2z');
  assert.deepEqual(readFileSync(manifestPath), first);
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.runtime));
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
