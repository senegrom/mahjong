import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const record = JSON.parse(read('docs/design/van-gogh/eight-characters-olive-grove.json'));
const entry = set.tiles.find(tile => tile.tile === '8m');

test('eight characters uses the approved Starry Olive Grove, not the earlier seven study', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Starry Olive Grove');
  assert.equal(entry.name, 'Man8');
  assert.equal(entry.source, record.source);
  assert.deepEqual(read(record.runtime), read(record.source));
  assert.equal(hash(read(record.runtime)), 'bffea32bd7cdf0b88bb735164f1a5ca94c9e2ec42d5a323623e4e7975620cf95');
  assert.equal(hash(read(record.runtime)), entry.svgSha256);
  assert.equal(entry.svgSha256, record.svgSha256);
  assert.equal(record.originalFilename, 'starry_olive_grove_八萬.png');
  assert.equal(record.originalSha256, '66a5db1f2b4be09595809db4a7b46a2267618d24a09ab45645cc1b66cc10f152');
  assert.deepEqual(record.originalDimensions, [1086, 1448]);
  assert.deepEqual(record.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
});

test('olive SVG embeds the exact game raster with standard rounded face geometry', () => {
  const svg = read(record.runtime).toString('utf8');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.doesNotMatch(svg, /<script|href="https?:/);
  const embedded = svg.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(hash(raster), '53d1e0224d3b032e75d71e631d4b9e05e2cf7035f6dfa24158ec1b1fc50d9b06');
  assert.equal(hash(raster), entry.rasterSha256);
  assert.equal(hash(raster), record.rasterSha256);
  assert.equal(raster.length, 57764);
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 16), 'WEBPVP8 ');
  assert.equal(raster.readUInt16LE(26) & 0x3fff, 300);
  assert.equal(raster.readUInt16LE(28) & 0x3fff, 400);
});

test('olive eight is registered and preloaded once, completing characters two through eight', () => {
  assert.equal(set.tiles.filter(tile => tile.tile === '8m').length, 1);
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '8m').length, 1);
  assert.deepEqual(VAN_GOGH_APPROVED, set.tiles.map(tile => tile.tile));
  assert.equal(set.remaining.length, 34 - VAN_GOGH_APPROVED.length);
  assert.equal(new Set([...VAN_GOGH_APPROVED, ...set.remaining]).size, 34);
  for (let rank = 2; rank <= 8; rank++) assert.ok(VAN_GOGH_APPROVED.includes(`${rank}m`));
  assert.ok(!set.remaining.includes('8m'));
  assert.equal(tileImage('8m', 'van-gogh'), 'tiles/van-gogh/approved/Man8.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('8m', 'van-gogh')).length, 1);
  assert.equal(tileImage('8m', 'classic'), 'tiles/Man8.svg');
  assert.equal(tileImage('8m', 'van-gogh', true), 'tiles/Back.svg');
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, entry.svgSha256);
});

test('olive eight appears in the preview without extending the fourteen-tile hand', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.match(hand, /approved\/Man8\.svg/);
  assert.match(hand, /approved\/Man7\.svg/);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
});

test('--only=8m copies the approved olive painting exactly and preserves every other face', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-olive-'));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const put = (relative, bytes) => {
    const target = path.join(temporary, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, bytes);
  };
  put('package.json', '{"type":"module"}');
  put('web/scripts/export-van-gogh-tiles.mjs', read('web/scripts/export-van-gogh-tiles.mjs'));
  put('web/scripts/face-fit.mjs', read('web/scripts/face-fit.mjs'));
  put('web/src/lib/tiles.js', read('web/src/lib/tiles.js'));
  for (const source of set.sources) {
    put(source.source, source.source.endsWith('.svg') ? read(source.source) : Buffer.from(`source fixture: ${source.id}`));
  }
  const unchanged = [];
  for (const tile of set.tiles.filter(tile => tile.tile !== '8m')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  assert.equal(run('--only=8m').trim(), `Exported ${set.tiles.length} approved Van Gogh faces; ${set.remaining.length} identities show their names until painted.`);
  for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.source));
  const manifestPath = path.join(temporary, 'web/public/tiles/van-gogh/manifest.json');
  const first = readFileSync(manifestPath);
  const generated = JSON.parse(first);
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '8m'), entry);
  assert.deepEqual(generated.remaining, set.remaining);
  assert.deepEqual(generated.superseded, set.superseded);
  run('--only=8m');
  assert.deepEqual(readFileSync(manifestPath), first);
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
