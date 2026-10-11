import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = new URL('../../', import.meta.url);
const read = relative => readFileSync(new URL(relative, root));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const record = JSON.parse(read('docs/design/van-gogh/seven-disks-apples.json'));
const entry = set.tiles.find(tile => tile.tile === '7p');
const source = 'docs/design/van-gogh/studies/23-seven-disks-apples-corrected-approved.svg';
const runtime = 'web/public/tiles/van-gogh/approved/Pin7.svg';
const rasterHash = '08d3c718a05d824e63f1d9debbd3cef8edb05f00554c44e064224b59debe5668';
const svgHash = '20ee711b62a73dd50b54f6fd635a4fb6eb26a7c58b47501d3b38d00e4a3ce4e7';

test('corrected Seven Apples is registered exactly once as seven disks', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Seven Apples (corrected)');
  assert.equal(entry.name, 'Pin7');
  assert.equal(entry.source, source);
  assert.equal(entry.svg, 'approved/Pin7.svg');
  assert.equal(set.tiles.filter(tile => tile.tile === '7p').length, 1);
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '7p').length, 1);
  assert.ok(!set.remaining.includes('7p'));
  assert.equal(set.fallback, 'text');
  assert.deepEqual(VAN_GOGH_APPROVED, set.tiles.map(tile => tile.tile));
  assert.deepEqual([...VAN_GOGH_APPROVED, ...set.remaining].sort(), [...TILE_TYPES].sort());
});

test('seven apples source and runtime preserve the exact approved crop export', () => {
  assert.deepEqual(read(runtime), read(source));
  assert.equal(hash(read(source)), svgHash);
  assert.equal(entry.svgSha256, svgHash);
  assert.equal(record.svgSha256, svgHash);
  assert.equal(set.sources.find(item => item.source === source).sha256, svgHash);
  const svg = read(runtime).toString('utf8');
  const match = svg.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(match);
  const raster = Buffer.from(match[1], 'base64');
  assert.equal(raster.length, 56596);
  assert.equal(hash(raster), rasterHash);
  assert.equal(entry.rasterSha256, rasterHash);
  assert.equal(record.rasterSha256, rasterHash);
  assert.equal(entry.rasterMimeType, 'image/webp');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(svg, /<script|<foreignObject|(?:href|src)="https?:/);
});

test('the selected four-plus-three painting is framed at exact 3:4 without stretching', () => {
  assert.equal(record.originalSha256, '0767ce0cf664fd0ef23e318417b9fca5ed45a0c29a6e304a377483ea475d9c27');
  assert.equal(record.cropSha256, '6f08b65a6db85d8ac159dbf30611ced4d0b6f208177c03d202f439c194eeb11a');
  assert.deepEqual(record.originalSize, [1024, 1536]);
  assert.deepEqual(record.originalCrop, { x: 62, y: 180, width: 900, height: 1200 });
  assert.deepEqual(record.visibleApples, { inBowl: 4, onCloth: 3, total: 7 });
  assert.equal(record.appleCentersOriginal.length, 7);
  assert.equal(record.aspectRatio, '3:4');
  assert.equal(record.stretch, 0);
  assert.equal(record.originalCrop.width / record.originalCrop.height, 3 / 4);
  assert.equal(record.canvas.width / record.canvas.height, 3 / 4);
  assert.equal(306 / 408, 3 / 4);
  assert.equal(entry.fit, undefined);
  assert.equal(record.webpQuality, 90);
  assert.equal(record.source, source);
  assert.equal(record.runtime, runtime);
});

test('seven apples preloads once and keeps other sets and hidden tiles unchanged', () => {
  assert.equal(tileImage('7p', 'van-gogh'), 'tiles/van-gogh/approved/Pin7.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('7p', 'van-gogh')).length, 1);
  assert.equal(tileImage('7p', 'classic'), 'tiles/Pin7.svg');
  assert.equal(tileImage('7p', 'matisse'), 'tiles/matisse/approved/Pin7.svg');
  assert.equal(tileImage('7p', 'van-gogh', true), 'tiles/Back.svg');
});

test('the seven-apple gallery entry leaves the fourteen-tile example hand intact', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  assert.match(preview, /Seven Apples at the Window/);
  assert.match(preview, /<figure><img src="approved\/Pin7.svg"/);
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
  for (const name of ['Man8', 'Pin4', 'Man7', 'Man6', 'Man5', 'Pin9', 'Pin3', 'Sou9', 'Sou6', 'Sou8']) {
    assert.ok(hand.includes(`approved/${name}.svg`));
  }
});

test('--only=7p repeats exactly without changing earlier approved artwork or fits', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-seven-disks-'));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const put = (relative, bytes) => {
    const target = path.join(temporary, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, bytes);
  };
  for (const relative of ['web/package.json', 'web/src/lib/tiles.js', 'web/scripts/export-van-gogh-tiles.mjs', 'web/scripts/face-fit.mjs']) put(relative, read(relative));
  for (const item of set.sources) put(item.source, read(item.source));
  const protectedFiles = new Map();
  for (const tile of set.tiles) {
    for (const relative of [tile.svg, tile.png].filter(Boolean)) {
      const target = `web/public/tiles/van-gogh/${relative}`;
      const bytes = read(target);
      put(target, bytes);
      protectedFiles.set(target, bytes);
    }
  }
  const generated = ['web/public/tiles/van-gogh/manifest.json', 'web/src/lib/van-gogh-faces.js', 'web/public/tiles/van-gogh/preview.html'];
  for (let run = 0; run < 2; run++) {
    execFileSync(process.execPath, ['web/scripts/export-van-gogh-tiles.mjs', '--only=7p'], { cwd: temporary, stdio: 'pipe' });
    for (const [relative, bytes] of protectedFiles) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
    for (const relative of generated) assert.deepEqual(readFileSync(path.join(temporary, relative)), read(relative));
  }
});
