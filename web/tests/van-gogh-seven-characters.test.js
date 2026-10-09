import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { tileImage, TILE_IMAGE_URLS } from '../src/lib/tile-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const record = JSON.parse(read('docs/design/van-gogh/seven-characters-irises.json'));
const entry = set.tiles.find(tile => tile.tile === '7m');

test('seven characters uses the approved last blue-iris painting, not olive or wheat', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Irises C');
  assert.equal(entry.name, 'Man7');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/20-seven-characters-irises-c-approved.svg');
  assert.equal(entry.source, record.source);
  assert.equal(record.title, 'Irises at Dusk');
  assert.equal(record.tile, '7m');
  assert.equal(record.originalFilename, 'van_gogh_inspired_irises_and_golden_characters.png');
  assert.equal(record.originalSha256, '9387a369fe64f455a2f6c0c4a86d5e1aea83b2026e117077aa5fd42da03fe403');
  assert.deepEqual(record.originalDimensions, [1086, 1448]);
  assert.deepEqual(record.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.deepEqual(record.rasterDimensions, [300, 400]);
  assert.equal(record.originalDimensions[0] * 400, record.originalDimensions[1] * 300);
  const source = read(record.source), runtime = read(record.runtime);
  assert.deepEqual(runtime, source);
  assert.equal(hash(runtime), '1b688f9e70ae70b12af8cf3ebc1acaeb38b01c90dd0ad2e7a46ce4888642650b');
  assert.equal(hash(runtime), record.svgSha256);
  assert.equal(hash(runtime), entry.svgSha256);
});

test('iris SVG keeps the shared geometry and the exact self-contained approved raster', () => {
  const svg = read(record.source).toString('utf8');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.match(svg, /<title id="title">Seven characters — Van Gogh<\/title>/);
  assert.doesNotMatch(svg, /<script\b|<foreignObject\b|<text\b|href="https?:/);
  const match = svg.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(match);
  const raster = Buffer.from(match[1], 'base64');
  assert.equal(raster.length, 57100);
  assert.equal(hash(raster), '111ae0918dc510db91441a50b46f6130b1199a2ab974f89d9059f469b90c4b60');
  assert.equal(hash(raster), record.rasterSha256);
  assert.equal(hash(raster), entry.rasterSha256);
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 16), 'WEBPVP8 ');
  assert.equal(raster.readUInt16LE(26) & 0x3fff, 300);
  assert.equal(raster.readUInt16LE(28) & 0x3fff, 400);
});

test('7m is registered and preloaded once without changing other sets or hidden tiles', () => {
  assert.equal(set.tiles.filter(tile => tile.tile === '7m').length, 1);
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '7m').length, 1);
  assert.ok(!set.remaining.includes('7m'));
  assert.deepEqual(VAN_GOGH_APPROVED, set.tiles.map(tile => tile.tile));
  assert.deepEqual([...VAN_GOGH_APPROVED, ...set.remaining].sort(), [...TILE_TYPES].sort());
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, entry.svgSha256);
  assert.equal(tileImage('7m', 'van-gogh'), 'tiles/van-gogh/approved/Man7.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('7m', 'van-gogh')).length, 1);
  assert.equal(tileImage('7m', 'classic'), 'tiles/Man7.svg');
  assert.equal(tileImage('7m', 'matisse'), 'tiles/matisse/approved/Man7.svg');
  assert.equal(tileImage('7m', 'van-gogh', true), 'tiles/Back.svg');
});

test('Irises at Dusk appears in the preview and its hand still has fourteen tiles', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  assert.match(preview, /Irises at Dusk/);
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.match(hand, /approved\/Man7\.svg/);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
});

test('--only=7m copies C exactly and preserves every other face across repeated exports', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-seven-characters-'));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const put = (relative, bytes) => {
    const target = path.join(temporary, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, bytes);
  };
  put('package.json', '{"type":"module"}');
  put('web/scripts/export-van-gogh-tiles.mjs', read('web/scripts/export-van-gogh-tiles.mjs'));
  put('web/src/lib/tiles.js', read('web/src/lib/tiles.js'));
  for (const source of set.sources) {
    put(source.source, source.source.endsWith('.svg') ? read(source.source) : Buffer.from(`source fixture: ${source.id}`));
  }
  const unchanged = [];
  for (const tile of set.tiles.filter(tile => tile.tile !== '7m')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  for (let attempt = 0; attempt < 2; attempt++) {
    assert.equal(run('--only=7m').trim(), `Exported ${set.tiles.length} approved Van Gogh faces; ${set.remaining.length} identities show their names until painted.`);
    for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
    assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.source));
    const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
    assert.deepEqual(generated.tiles.find(tile => tile.tile === '7m'), entry);
    assert.deepEqual(generated.remaining, set.remaining);
    assert.deepEqual(generated.superseded, set.superseded);
  }
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
