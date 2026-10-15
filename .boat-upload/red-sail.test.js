import assert from 'node:assert/strict';
import test from 'node:test';
import { readdirSync, readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { tileImage, TILE_IMAGE_URLS } from '../src/lib/tile-faces.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';

const root = new URL('../../', import.meta.url);
const read = path => readFileSync(new URL(path, root));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const source = 'docs/design/van-gogh/studies/26-one-characters-red-sail-c-approved.svg';
const runtime = 'web/public/tiles/van-gogh/approved/Man1.svg';
const manifestPath = 'web/public/tiles/van-gogh/manifest.json';
const manifest = JSON.parse(read(manifestPath));
const record = JSON.parse(read('docs/design/van-gogh/one-characters-red-sail.json'));
const entry = manifest.tiles.find(item => item.tile === '1m');
const SOURCE_HASH = '7f82fc7cc6ca21f37cba9ccfe5f0292e836eb1f584d5aa7097feaf14ca7ddcf9';
const SVG_HASH = '85a214007ce059ed61aa47f2257db9ce2c6499ed58c95954d926d88fd2dbcfea';
const RASTER_HASH = '0a6b2b96aaebe241576e94d6cb1d80acf5fc862a6b7a33f8a5323b2c4bbbcc70';

test('one character uses the selected red-sail boat, not the sower or window alternatives', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Red Sail C');
  assert.equal(entry.source, source);
  assert.equal(entry.svg, 'approved/Man1.svg');
  assert.equal(hash(read(source)), SOURCE_HASH);
  assert.equal(hash(read(runtime)), SVG_HASH);
  assert.equal(read(source).toString().replace('x="-3" y="-4" width="306" height="408"', 'x="0" y="0" width="300" height="400"'), read(runtime).toString());
  assert.equal(record.originalFilename, 'crimson_sail_in_a_mediterranean_harbour.png');
  assert.equal(record.originalSha256, '81570fc132f12f48fceb262ecc0f7e6a4f1e9109198d38281a149dea0a8e4d2a');
  assert.deepEqual(record.originalDimensions, [1086, 1448]);
  assert.equal(record.sourceSha256, SOURCE_HASH);
  assert.equal(record.runtimeSvgSha256, SVG_HASH);
});

test('the boat keeps the complete 3:4 painting with no stretch or bleed and a self-contained raster', () => {
  const svg = read(runtime).toString();
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.equal((svg.match(/<image\b/g) ?? []).length, 1);
  assert.match(svg, /x="0" y="0" width="300" height="400" preserveAspectRatio="none"/);
  assert.deepEqual(entry.fit.painting, { width: 1086, height: 1448 });
  assert.equal(entry.fit.stretch, 0);
  assert.equal(entry.fit.bleedPixels, 0);
  assert.deepEqual(entry.fit.image, { x: 0, y: 0, width: 300, height: 400 });
  const match = svg.match(/href="data:(image\/avif);base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(match);
  const bytes = Buffer.from(match[2], 'base64');
  assert.equal(bytes.length, 36049);
  assert.equal(bytes.subarray(4, 12).toString(), 'ftypavif');
  assert.equal(hash(bytes), RASTER_HASH);
  assert.equal(entry.rasterSha256, RASTER_HASH);
  assert.equal(record.rasterSha256, RASTER_HASH);
  assert.equal(record.rasterMimeType, 'image/avif');
  assert.doesNotMatch(svg, /(?:href|src)="(?:https?:|\/\/)/);
});

test('the boat is registered and preloaded once, completing the characters suit without revealing hidden tiles', () => {
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '1m').length, 1);
  assert.equal(manifest.tiles.filter(item => item.tile === '1m').length, 1);
  assert.equal(TILE_IMAGE_URLS.filter(url => url === 'tiles/van-gogh/approved/Man1.svg').length, 1);
  assert.equal(tileImage('1m', 'van-gogh'), 'tiles/van-gogh/approved/Man1.svg');
  assert.equal(tileImage('1m', 'van-gogh', true), 'tiles/Back.svg');
  assert.equal(tileImage(null, 'van-gogh'), 'tiles/Back.svg');
  assert.equal(tileImage('1m', 'classic'), 'tiles/Man1.svg');
  assert.equal(tileImage('1m', 'matisse'), 'tiles/matisse/approved/Man1.svg');
  for (const suit of ['m', 'p', 's']) {
    for (let rank = 1; rank <= 9; rank += 1) assert.ok(VAN_GOGH_APPROVED.includes(`${rank}${suit}`));
  }
  assert.ok(!manifest.remaining.includes('1m'));
  assert.deepEqual([...VAN_GOGH_APPROVED, ...manifest.remaining].sort(), [...TILE_TYPES].sort());
});

test('the red sail appears in the gallery and in the fourteen-tile example hand', () => {
  const html = read('web/public/tiles/van-gogh/preview.html').toString();
  assert.match(html, /Red Sail C · One character/);
  const rack = html.match(/<div class="rack"[^>]*>(.*?)<\/div>/s)?.[1];
  assert.ok(rack);
  assert.equal((rack.match(/<img\b/g) ?? []).length, 14);
  assert.equal((rack.match(/approved\/Man1\.svg/g) ?? []).length, 1);
});

test('--only=1m reproduces the approved boat exactly without rewriting other artwork', () => {
  const paths = ['matisse', 'dali', 'van-gogh'].flatMap(set => {
    const folder = `web/public/tiles/${set}/approved/`;
    return readdirSync(new URL(folder, root)).filter(name => /\.(?:svg|png|webp|avif)$/.test(name)).map(name => folder + name);
  });
  const before = new Map(paths.map(path => [path, hash(read(path))]));
  const metadata = [manifestPath, 'web/src/lib/van-gogh-faces.js', 'web/public/tiles/van-gogh/preview.html'];
  const beforeMetadata = new Map(metadata.map(path => [path, hash(read(path))]));
  for (let attempt = 0; attempt < 2; attempt += 1) {
    execFileSync(process.execPath, ['web/scripts/export-van-gogh-tiles.mjs', '--only=1m'], { cwd: fileURLToPath(root) });
    for (const [path, expected] of before) assert.equal(hash(read(path)), expected, path);
    for (const [path, expected] of beforeMetadata) assert.equal(hash(read(path)), expected, path);
    assert.equal(hash(read(runtime)), SVG_HASH);
  }
});
