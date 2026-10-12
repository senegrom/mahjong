import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, cpSync, mkdirSync, mkdtempSync, rmSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const manifest = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const provenance = JSON.parse(read('docs/design/van-gogh/green-dragon-vineyard.json'));
const entry = manifest.tiles.find(tile => tile.tile === '6z');
const sourceHash = 'd4807a9a2d6c712b11310a9f77ee22ef5fe7c65d86e6a63141eaf2f247154994';
const rasterHash = '3ee4e8913303c7d01b6d56bbbdf82a5518ad463d4e7fd66687a02655aa8422df';

test('green dragon uses first-board Vineyard C, not the mountain, vase or other alternatives', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Vineyard Dragon C');
  assert.equal(entry.name, 'Hatsu');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/25-green-dragon-vineyard-c-approved.svg');
  assert.equal(provenance.originalFile, 'van_gogh_dragons_a_b_c.png');
  assert.equal(provenance.originalSha256, '98b856ce6fa6bd6b6db62ed0ed64fd426284ce5cfa2b60d88c9e40d71533d493');
  assert.equal(provenance.candidate, entry.candidate);
  assert.match(provenance.palette, /blue sky and golden vineyard/);
});

test('Vineyard C source, runtime and AVIF bytes retain their approved hashes', () => {
  const source = read(entry.source);
  const runtime = read(`web/public/tiles/van-gogh/${entry.svg}`);
  assert.deepEqual(source, runtime);
  assert.equal(hash(source), sourceHash);
  assert.equal(entry.svgSha256, sourceHash);
  assert.equal(provenance.svgSha256, sourceHash);
  assert.equal(entry.rasterMimeType, 'image/avif');
  const match = source.toString().match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(match);
  const raster = Buffer.from(match[1], 'base64');
  assert.equal(hash(raster), rasterHash);
  assert.equal(entry.rasterSha256, rasterHash);
  assert.equal(raster.length, provenance.rasterBytes);
  const size = raster.indexOf(Buffer.from('ispe'));
  assert.ok(size > 0);
  assert.equal(raster.readUInt32BE(size + 8), 300);
  assert.equal(raster.readUInt32BE(size + 12), 400);
});

test('Vineyard C keeps exact 3:4 proportions and the shared safe SVG presentation', () => {
  assert.deepEqual(provenance.originalCrop, { x: 979, y: 122, width: 450, height: 600 });
  assert.equal(provenance.cropSha256, 'cfb0e1bf1eb4016157566cb81adef74d4a56c916e9355bf03660903708e446a0');
  assert.equal(provenance.originalCrop.width * 4, provenance.originalCrop.height * 3);
  assert.equal(provenance.stretch, 0);
  assert.equal(provenance.aspectRatio, '3:4');
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 });
  assert.equal(entry.fit, undefined);
  const svg = read(entry.source).toString();
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(svg, /<script\b|<foreignObject\b|<text\b|(?:href|src)="https?:/);
});

test('green dragon is approved and preloaded once; hidden and other sets remain independent', () => {
  assert.equal(manifest.tiles.filter(tile => tile.tile === '6z').length, 1);
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '6z').length, 1);
  assert.ok(!manifest.remaining.includes('6z'));
  assert.deepEqual(VAN_GOGH_APPROVED, manifest.tiles.map(tile => tile.tile));
  assert.deepEqual([...VAN_GOGH_APPROVED, ...manifest.remaining].sort(), [...TILE_TYPES].sort());
  assert.equal(tileImage('6z', 'van-gogh'), 'tiles/van-gogh/approved/Hatsu.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('6z', 'van-gogh')).length, 1);
  assert.equal(tileImage('6z', 'van-gogh', true), 'tiles/Back.svg');
  assert.equal(tileImage('6z', 'classic'), 'tiles/Hatsu.svg');
  assert.equal(tileImage('6z', 'matisse'), 'tiles/matisse/approved/Hatsu.svg');
});

test('Vineyard C appears in the gallery without extending the fourteen-tile sample hand', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString();
  assert.match(preview, /Green Dragon at the Vineyard C/);
  assert.match(preview, /src="approved\/Hatsu.svg" alt="Green dragon"/);
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
  assert.match(read('web/scripts/export-van-gogh-tiles.mjs').toString(), /\['Vineyard Dragon C', '6z', 'Green dragon', 'greenDragonVineyard'/);
});

test('--only=6z is repeatable and never rewrites another approved face or fit', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-vineyard-dragon-'));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const copy = relative => {
    const target = path.join(temporary, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    cpSync(path.join(root, relative), target, { recursive: true });
  };
  for (const relative of ['web/scripts/export-van-gogh-tiles.mjs', 'web/scripts/face-fit.mjs', 'web/src/lib/tiles.js', 'web/public/tiles/van-gogh/approved']) copy(relative);
  for (const item of manifest.sources) copy(item.source);
  const approvedDir = path.join(temporary, 'web/public/tiles/van-gogh/approved');
  const before = Object.fromEntries(readdirSync(approvedDir).map(name => [name, hash(readFileSync(path.join(approvedDir, name)))]));
  const outputs = ['web/public/tiles/van-gogh/manifest.json', 'web/public/tiles/van-gogh/preview.html', 'web/src/lib/van-gogh-faces.js'];
  let first;
  for (let run = 0; run < 2; run++) {
    execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), '--only=6z'], { cwd: temporary });
    const actual = Object.fromEntries(readdirSync(approvedDir).map(name => [name, hash(readFileSync(path.join(approvedDir, name)))]));
    assert.deepEqual(actual, before);
    const exported = JSON.parse(readFileSync(path.join(temporary, outputs[0])));
    assert.deepEqual(exported.tiles, manifest.tiles);
    const current = outputs.map(relative => hash(readFileSync(path.join(temporary, relative))));
    if (first) assert.deepEqual(current, first);
    first = current;
  }
});
