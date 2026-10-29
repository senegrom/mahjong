import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
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
const record = JSON.parse(read('docs/design/van-gogh/four-bamboo-sunlit-garden-b.json'));
const entry = set.tiles.find(tile => tile.tile === '4s');
const sourceHash = 'dad4f4cdf389ef5dbadbe82261e036054b5325e98b4e5eb1d562c13cc4749a59';
const runtimeHash = '574e208e0e6dbb4a247b1d58fc09a2e9a610bf43d64d9d7a2948d1f5c2efe841';
const rasterHash = 'bee554c0d08868e23f20ad7b82dd57df88dbd25b19fc87ea297c2ddb6dfc2b75';

test('four bamboo uses the selected middle Sunlit Garden B, not either moonlit alternative', () => {
  assert.equal(entry.candidate, 'Sunlit Garden B');
  assert.equal(entry.name, 'Sou4');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/35-four-bamboo-sunlit-garden-b-approved.svg');
  assert.equal(record.source, entry.source);
  assert.equal(record.tile, '4s');
  assert.equal(record.originalBoardSha256, '030abbb0fa849e6b87fa9ba77f29576ce88e3890453c260c5e23cbb529b9b4b2');
  assert.deepEqual(record.originalBoardDimensions, [1882, 836]);
  assert.deepEqual(record.originalCrop, { x: 640, y: 16, width: 600, height: 800 });
  assert.equal(record.fullResolutionCropSha256, 'c416e891d268623b2db453e4e11c8eeb5a8080541c90e278d1e0f7b564c311b5');
  assert.equal(hash(read(record.source)), sourceHash);
  assert.equal(record.svgSha256, sourceHash);
  assert.equal(hash(read(record.runtime)), runtimeHash);
  assert.equal(record.runtimeSvgSha256, runtimeHash);
  assert.equal(entry.svgSha256, runtimeHash);
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, sourceHash);
});

test('Sunlit Garden retains its exact 3:4 crop with zero stretch and no extra bleed', () => {
  assert.deepEqual(entry.fit, {
    painting: { width: 600, height: 800 }, stretch: 0, anchor: 0.5, bleedPixels: 0,
    image: { x: 0, y: 0, width: 300, height: 400 },
  });
  const svg = read(record.runtime).toString();
  assert.equal(read(record.source).toString().replace('x="-3" y="-4" width="306" height="408"', 'x="0" y="0" width="300" height="400"'), svg);
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.doesNotMatch(svg, /<script|<foreignObject|href="https?:/i);
  const embedded = svg.match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(raster.length, 48483);
  assert.equal(hash(raster), rasterHash);
  assert.equal(record.rasterSha256, rasterHash);
  assert.equal(entry.rasterSha256, rasterHash);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.equal(raster.readUInt32BE(ispe + 8), 300);
  assert.equal(raster.readUInt32BE(ispe + 12), 400);
});

test('Moonlit Four remains archived without leaving an obsolete playable PNG', () => {
  const previous = set.superseded.find(tile => tile.tile === '4s' && tile.candidate === 'Bamboo C (green)');
  assert.equal(previous.activeCandidate, 'Sunlit Garden B');
  assert.equal(previous.archivedSvg, record.archivedSvg);
  assert.equal(previous.archivedPng, record.archivedPng);
  assert.equal(hash(read(record.archivedSvg)), record.archivedSvgSha256);
  assert.equal(hash(read(record.archivedPng)), record.archivedPngSha256);
  assert.equal(hash(read(previous.source)), '4d1e76c1716b6086524dfcf94ec8141fa9196a78fce5ecffa517e904204304fb');
  assert.equal(existsSync(path.join(root, 'web/public/tiles/van-gogh/approved/Sou4.png')), false);
});

test('Sunlit Garden is registered and preloaded once, preserving the other selected replacements', () => {
  assert.equal(set.tiles.length, 34);
  assert.equal(set.remaining.length, 0);
  assert.deepEqual(set.tiles.map(tile => tile.tile), VAN_GOGH_APPROVED);
  assert.deepEqual([...VAN_GOGH_APPROVED].sort(), [...TILE_TYPES].sort());
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '4s').length, 1);
  assert.equal(tileImage('4s', 'van-gogh'), 'tiles/van-gogh/approved/Sou4.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('4s', 'van-gogh')).length, 1);
  assert.equal(tileImage('4s', 'classic'), 'tiles/Sou4.svg');
  assert.equal(tileImage('4s', 'van-gogh', true), 'tiles/Back.svg');
  for (const [tile, candidate] of [['2s', 'Flutes B'], ['3s', 'Vase A'], ['6m', 'Scarlet Loom C'], ['5z', 'Almond Blossom C'], ['7z', 'Painter C']]) {
    assert.equal(set.tiles.find(entry => entry.tile === tile).candidate, candidate);
  }
});

test('Sunlit Garden appears in the gallery without changing the fourteen-tile sample hand', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString();
  assert.match(preview, /Sunlit Garden B for 4 bamboo/);
  assert.match(preview, /Sunlit Garden B · Four bamboo/);
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
});

test('--only=4s reproduces Sunlit Garden exactly and leaves every other face unchanged', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-four-garden-'));
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
  for (const tile of set.tiles.filter(tile => tile.tile !== '4s')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  run('--only=4s');
  const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '4s'), entry);
  assert.deepEqual(generated.superseded, set.superseded);
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.runtime));
  run('--only=4s');
  assert.equal(hash(readFileSync(path.join(temporary, record.runtime))), runtimeHash);
  for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
