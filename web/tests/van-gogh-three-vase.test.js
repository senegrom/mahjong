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
const record = JSON.parse(read('docs/design/van-gogh/three-bamboo-vase-a.json'));
const entry = set.tiles.find(tile => tile.tile === '3s');
const sourceHash = 'f60e7639f7b7422f48f74a1996ef20e0f76e22a4c322ff5b833c00c11cdc7f38';
const runtimeHash = '106e53ecf2d0374e7aa861d4cdca7ae352ad0c21c9dff5cd1e0ae78710e32535';
const rasterHash = '9e004fae4f28c634f74f55c2ad1b5bad84c51941994772ee1106d8c52f18f174';

test('three bamboo uses the selected left-hand vase A rather than shoots or pond', () => {
  assert.equal(entry.candidate, 'Vase A');
  assert.equal(entry.name, 'Sou3');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/33-three-bamboo-vase-a-approved.svg');
  assert.equal(record.source, entry.source);
  assert.equal(record.tile, '3s');
  assert.equal(record.originalBoardSha256, '8dd5305e3eb699112be9066facf28f4b126a1170ce8ebdb837903ca037b9fdb9');
  assert.deepEqual(record.originalBoardDimensions, [1881, 836]);
  assert.deepEqual(record.originalCrop, { x: 22, y: 23, width: 576, height: 768 });
  assert.equal(record.fullResolutionCropSha256, '86de33ef368bd44dc45835d7cdfdaa63abe5074a25483170bbb9291219074879');
  assert.equal(hash(read(record.source)), sourceHash);
  assert.equal(record.svgSha256, sourceHash);
  assert.equal(hash(read(record.runtime)), runtimeHash);
  assert.equal(record.runtimeSvgSha256, runtimeHash);
  assert.equal(entry.svgSha256, runtimeHash);
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, sourceHash);
});

test('the vase retains its exact portrait crop with zero stretch and no extra bleed', () => {
  assert.deepEqual(entry.fit, {
    painting: { width: 576, height: 768 }, stretch: 0, anchor: 0.5, bleedPixels: 0,
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
  assert.equal(raster.length, 42493);
  assert.equal(hash(raster), rasterHash);
  assert.equal(record.rasterSha256, rasterHash);
  assert.equal(entry.rasterSha256, rasterHash);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.equal(raster.readUInt32BE(ispe + 8), 300);
  assert.equal(raster.readUInt32BE(ispe + 12), 400);
});

test('Triple Shoots remains archived without leaving an obsolete playable PNG', () => {
  const previous = set.superseded.find(tile => tile.tile === '3s' && tile.candidate === 'Bamboo A (green)');
  assert.equal(previous.activeCandidate, 'Vase A');
  assert.equal(previous.archivedSvg, record.archivedSvg);
  assert.equal(previous.archivedPng, record.archivedPng);
  assert.equal(hash(read(record.archivedSvg)), record.archivedSvgSha256);
  assert.equal(hash(read(record.archivedPng)), record.archivedPngSha256);
  assert.equal(record.archivedPngSha256, '71b40f6a195e24ab4b785cf9680a2b42e895d9a63fad17ef44509c7a785d3ae3');
  assert.equal(hash(read(previous.source)), 'f2e1617a2e894622a28922f309747a8c2d0bb681fedd4ca37c1ba4b810135c7c');
  assert.equal(existsSync(path.join(root, 'web/public/tiles/van-gogh/approved/Sou3.png')), false);
});

test('Vase A is registered and preloaded once while other face sets and hidden tiles stay independent', () => {
  assert.equal(set.tiles.length, 34);
  assert.equal(set.remaining.length, 0);
  assert.deepEqual(set.tiles.map(tile => tile.tile), VAN_GOGH_APPROVED);
  assert.deepEqual([...VAN_GOGH_APPROVED].sort(), [...TILE_TYPES].sort());
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '3s').length, 1);
  assert.equal(tileImage('3s', 'van-gogh'), 'tiles/van-gogh/approved/Sou3.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('3s', 'van-gogh')).length, 1);
  assert.equal(tileImage('3s', 'classic'), 'tiles/Sou3.svg');
  assert.equal(tileImage('3s', 'van-gogh', true), 'tiles/Back.svg');
  assert.equal(set.tiles.find(tile => tile.tile === '6m').candidate, 'Scarlet Loom C');
  assert.equal(set.tiles.find(tile => tile.tile === '5z').candidate, 'Almond Blossom C');
  assert.equal(set.tiles.find(tile => tile.tile === '7z').candidate, 'Painter C');
});

test('the vase is shown in the gallery without changing the fourteen-tile sample hand', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString();
  assert.match(preview, /Three Bamboo in a Vase A for 3 bamboo/);
  assert.match(preview, /Vase A · Three bamboo/);
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
});

test('--only=3s reproduces the exact vase, preserves all other artwork and is repeatable', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-three-vase-'));
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
  for (const tile of set.tiles.filter(tile => tile.tile !== '3s')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  run('--only=3s');
  const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '3s'), entry);
  assert.deepEqual(generated.superseded, set.superseded);
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.runtime));
  run('--only=3s');
  assert.equal(hash(readFileSync(path.join(temporary, record.runtime))), runtimeHash);
  for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
