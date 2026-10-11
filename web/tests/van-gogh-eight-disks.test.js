import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const record = JSON.parse(read('docs/design/van-gogh/eight-disks-lemons.json'));
const entry = set.tiles.find(tile => tile.tile === '8p');

test('eight disks uses the exact approved eight-lemon painting', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Eight Lemons');
  assert.equal(entry.name, 'Pin8');
  assert.equal(record.title, 'Lemons by the Sea');
  assert.equal(entry.source, record.source);
  const source = read(record.source), runtime = read(record.runtime);
  assert.deepEqual(runtime, source);
  assert.equal(hash(source), 'f0d9ed90074049d1db9ebee3731351bf025cd55d9bc296f279941b4ef8618864');
  assert.equal(hash(runtime), entry.svgSha256);
  assert.equal(hash(source), record.svgSha256);
  const xml = runtime.toString('utf8');
  assert.match(xml, /viewBox="0 0 300 400"/);
  assert.match(xml, /rx="26"/);
  assert.match(xml, /x="-3" y="-4" width="306" height="408"/);
  const match = xml.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(match);
  const raster = Buffer.from(match[1], 'base64');
  assert.equal(hash(raster), 'b97d16c560e0d88b8155e42ed0bc90b35bc48e0357df88d4569bec3986afb757');
  assert.equal(hash(raster), record.rasterSha256);
  assert.equal(hash(raster), entry.rasterSha256);
  assert.equal(raster.length, 69178);
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 16), 'WEBPVP8 ');
  assert.equal(raster.readUInt16LE(26) & 0x3fff, 300);
  assert.equal(raster.readUInt16LE(28) & 0x3fff, 400);
});

test('the full approved lemon composition is 3:4 without a corrective fit', () => {
  assert.equal(record.originalSha256, '1989d02649ee8029e3386e79c729affc2f381a8b9f7eca0cb535a7c76f131bb7');
  assert.deepEqual(record.originalDimensions, [1086, 1448]);
  assert.deepEqual(record.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.deepEqual(record.rasterDimensions, [300, 400]);
  assert.equal(record.originalDimensions[0] / record.originalDimensions[1], 3 / 4);
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 });
  assert.equal(entry.fit, undefined);
});

test('all disks and bamboo are painted while Seven Apples and the Potter’s Table remain selected', () => {
  for (let n = 1; n <= 9; n++) {
    for (const suit of ['p', 's']) assert.ok(VAN_GOGH_APPROVED.includes(`${n}${suit}`));
  }
  assert.equal(set.tiles.filter(tile => tile.tile === '8p').length, 1);
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '8p').length, 1);
  assert.equal(set.tiles.find(tile => tile.tile === '7p').candidate, 'Seven Apples (corrected)');
  assert.equal(set.tiles.find(tile => tile.tile === '9p').candidate, 'Disks C');
  assert.equal(tileImage('8p', 'van-gogh'), 'tiles/van-gogh/approved/Pin8.svg');
  assert.ok(TILE_IMAGE_URLS.includes(tileImage('8p', 'van-gogh')));
  assert.ok(!set.remaining.includes('8p'));
  assert.equal(set.tiles.length + set.remaining.length, 34);
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  assert.match(preview, /Lemons by the Sea for 8 disks/);
  const rack = preview.match(/<div class="rack" id="rack">(.*?)<\/div>/)[1];
  assert.equal((rack.match(/<img /g) ?? []).length, 14);
  assert.match(rack, /approved\/Pin8\.svg/);
});

test('--only=8p is deterministic and preserves every other playable image', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-eight-lemons-'));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const put = (relative, bytes) => {
    const target = path.join(temporary, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, bytes);
  };
  put('package.json', '{"type":"module"}');
  for (const file of ['web/scripts/export-van-gogh-tiles.mjs', 'web/scripts/face-fit.mjs', 'web/src/lib/tiles.js']) put(file, read(file));
  for (const source of set.sources) put(source.source, source.source.endsWith('.svg') ? read(source.source) : Buffer.from(`source fixture: ${source.id}`));
  const unchanged = [];
  for (const tile of set.tiles.filter(tile => tile.tile !== '8p')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = () => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), '--only=8p'], { encoding: 'utf8', stdio: 'pipe' });
  assert.equal(run().trim(), `Exported ${set.tiles.length} approved Van Gogh faces; ${set.remaining.length} identities show their names until painted.`);
  for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.source));
  const manifestPath = path.join(temporary, 'web/public/tiles/van-gogh/manifest.json');
  const generated = JSON.parse(readFileSync(manifestPath));
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '8p'), entry);
  assert.deepEqual(generated.remaining, set.remaining);
  const first = readFileSync(manifestPath);
  run();
  assert.deepEqual(readFileSync(manifestPath), first);
});
