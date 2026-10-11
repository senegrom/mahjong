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
const record = JSON.parse(read('docs/design/van-gogh/six-disks-kitchen.json'));
const entry = set.tiles.find(tile => tile.tile === '6p');
const source = 'docs/design/van-gogh/studies/22-six-disks-provencal-kitchen-b-approved.svg';
const runtime = 'web/public/tiles/van-gogh/approved/Pin6.svg';
const rasterHash = '95d81654ea7db8776840592f0f28e79a33c936fc201b58c2a154bb4fbe9d9e46';
const svgHash = '9c0e3ae5048973c0e1e825346d5cabed550411aaef46c6ddf600cd7e488e1303';
// The face is the approved study with only its picture's box moved.
const withoutBox = svg => svg.toString().replace(/ x="[^"]*" y="[^"]*" width="[^"]*" height="[^"]*"/, '');

test('Kitchen B is the approved six disks, registered exactly once with all other identities retained', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Kitchen B');
  assert.equal(entry.name, 'Pin6');
  assert.equal(entry.source, source);
  assert.equal(entry.svg, 'approved/Pin6.svg');
  assert.equal(set.tiles.filter(tile => tile.tile === '6p').length, 1);
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '6p').length, 1);
  assert.ok(!set.remaining.includes('6p'));
  assert.deepEqual(VAN_GOGH_APPROVED, set.tiles.map(tile => tile.tile));
  assert.deepEqual([...VAN_GOGH_APPROVED, ...set.remaining].sort(), [...TILE_TYPES].sort());
});

test('the kitchen face keeps the checksum-pinned approved export, differing only in its fit', () => {
  assert.equal(withoutBox(read(runtime)), withoutBox(read(source)));
  assert.equal(hash(read(source)), svgHash);
  assert.equal(entry.svgSha256, hash(read(runtime)));
  assert.equal(record.svgSha256, svgHash);
  assert.equal(set.sources.find(item => item.source === source).sha256, svgHash);
  const svg = read(runtime).toString('utf8');
  const match = svg.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(match);
  const raster = Buffer.from(match[1], 'base64');
  assert.equal(raster.length, 57930);
  assert.equal(hash(raster), rasterHash);
  assert.equal(entry.rasterSha256, rasterHash);
  assert.equal(record.rasterSha256, rasterHash);
  assert.equal(entry.rasterMimeType, 'image/webp');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  // The 300 × 400 export squeezed the 464 × 873 panel; the face's box restores its
  // shape and crops the jug, lemons and dresser below the plates, not the plates.
  const { x, y, width, height } = entry.fit.image;
  assert.ok(svg.includes(`x="${x}" y="${y}" width="${width}" height="${height}" preserveAspectRatio="none"`));
  assert.deepEqual(entry.fit.painting, { width: record.originalCrop.width, height: record.originalCrop.height });
  assert.ok(Math.abs(width / height / (464 / 873) - 1 - entry.fit.stretch) < 0.001);
  assert.doesNotMatch(svg, /<script|<foreignObject|(?:href|src)="https?:/);
});

test('kitchen provenance identifies the middle-panel selection and documents the aspect-ratio conversion', () => {
  assert.equal(record.candidate, 'Kitchen B');
  assert.equal(record.title, 'The Provençal Kitchen');
  assert.equal(record.source, source);
  assert.equal(record.runtime, runtime);
  assert.equal(record.originalBoardSha256, '106cdbb426b79422ab7e8672928010493bec1565dd1ff4b38473ed339842a68d');
  assert.equal(record.fullResolutionCropSha256, '0846ee8b5242c7ff5775d6098f7a31c676cda0750767476de15825593fed6f26');
  assert.deepEqual(record.originalBoardSize, { width: 1491, height: 1055 });
  assert.deepEqual(record.originalCrop, { x: 514, y: 147, width: 464, height: 873 });
  assert.deepEqual(record.productionSize, { width: 300, height: 400 });
  assert.equal(record.webpQuality, 90);
  assert.match(record.processing, /aspect ratio changes/);
});

test('six disks preloads once and does not alter Classic, Matisse or hidden faces', () => {
  assert.equal(tileImage('6p', 'van-gogh'), 'tiles/van-gogh/approved/Pin6.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('6p', 'van-gogh')).length, 1);
  assert.equal(tileImage('6p', 'classic'), 'tiles/Pin6.svg');
  assert.equal(tileImage('6p', 'matisse'), 'tiles/matisse/approved/Pin6.svg');
  assert.equal(tileImage('6p', 'van-gogh', true), 'tiles/Back.svg');
});

test('the kitchen appears in the gallery while the existing example hand stays at fourteen tiles', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  assert.match(preview, /The Provençal Kitchen/);
  assert.match(preview, /<figure><img src="approved\/Pin6.svg"/);
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
  for (const name of ['Man8', 'Pin4', 'Man7', 'Man6', 'Man5', 'Pin9', 'Pin3', 'Sou9', 'Sou6', 'Sou8']) {
    assert.ok(hand.includes(`approved/${name}.svg`));
  }
});

test('--only=6p is reproducible and never changes another approved face', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-six-disks-'));
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
    execFileSync(process.execPath, ['web/scripts/export-van-gogh-tiles.mjs', '--only=6p'], { cwd: temporary, stdio: 'pipe' });
    for (const [relative, bytes] of protectedFiles) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
    for (const relative of generated) assert.deepEqual(readFileSync(path.join(temporary, relative)), read(relative));
  }
});
