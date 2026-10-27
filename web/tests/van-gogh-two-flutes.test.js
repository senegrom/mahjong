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
const record = JSON.parse(read('docs/design/van-gogh/two-bamboo-flutes-b.json'));
const entry = set.tiles.find(tile => tile.tile === '2s');
const sourceHash = 'b96e075418830bb5f65c401a01efdfba1d492c8be5e55f7795678f41c1b45435';
const runtimeHash = '364fccf7e6fac3ab09bc7f4f1c0f7d913a25e799a304d707450ea5426a15fca2';
const rasterHash = 'a7ea6872aa22a34527eb4de3d2c9f5cfdc2efb8d0eea3d2dfd030c4d7b2e2636';

test('two bamboo uses selected flutes B rather than the bridge or terracotta-wall alternatives', () => {
  assert.equal(entry.candidate, 'Flutes B');
  assert.equal(entry.name, 'Sou2');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/34-two-bamboo-flutes-b-approved.svg');
  assert.equal(record.source, entry.source);
  assert.equal(record.tile, '2s');
  assert.equal(record.originalSha256, '673d03362f020ade5f7b1e256997e641ef58946d6a300def8e0a6a2c2061a785');
  assert.deepEqual(record.originalDimensions, [1086, 1448]);
  assert.deepEqual(record.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.equal(hash(read(record.source)), sourceHash);
  assert.equal(record.svgSha256, sourceHash);
  assert.equal(hash(read(record.runtime)), runtimeHash);
  assert.equal(record.runtimeSvgSha256, runtimeHash);
  assert.equal(entry.svgSha256, runtimeHash);
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, sourceHash);
});

test('the flute painting retains its exact portrait crop with zero stretch and no extra bleed', () => {
  assert.deepEqual(entry.fit, {
    painting: { width: 1086, height: 1448 }, stretch: 0, anchor: 0.5, bleedPixels: 0,
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
  assert.equal(raster.length, 43105);
  assert.equal(hash(raster), rasterHash);
  assert.equal(record.rasterSha256, rasterHash);
  assert.equal(entry.rasterSha256, rasterHash);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.equal(raster.readUInt32BE(ispe + 8), 300);
  assert.equal(raster.readUInt32BE(ispe + 12), 400);
});

test('Garden Rhythm remains archived without leaving an obsolete playable PNG', () => {
  const previous = set.superseded.find(tile => tile.tile === '2s' && tile.candidate === 'Bamboo B (green)');
  assert.equal(previous.activeCandidate, 'Flutes B');
  assert.equal(previous.archivedSvg, record.archivedSvg);
  assert.equal(previous.archivedPng, record.archivedPng);
  assert.equal(hash(read(record.archivedSvg)), record.archivedSvgSha256);
  assert.equal(hash(read(record.archivedPng)), record.archivedPngSha256);
  assert.equal(record.archivedPngSha256, '581df3d6e7992d547af374a676c5a051314cc698f16fe918fde1decb2099e14d');
  assert.equal(hash(read(previous.source)), '528acc47e037fdce2fc9f48f827bbe554da35a1cc0186a4b74f1ac9fea0ad7ec');
  assert.equal(existsSync(path.join(root, 'web/public/tiles/van-gogh/approved/Sou2.png')), false);
});

test('Flutes B is registered and preloaded once while other face sets and hidden tiles stay independent', () => {
  assert.equal(set.tiles.length, 34);
  assert.equal(set.remaining.length, 0);
  assert.deepEqual(set.tiles.map(tile => tile.tile), VAN_GOGH_APPROVED);
  assert.deepEqual([...VAN_GOGH_APPROVED].sort(), [...TILE_TYPES].sort());
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '2s').length, 1);
  assert.equal(tileImage('2s', 'van-gogh'), 'tiles/van-gogh/approved/Sou2.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('2s', 'van-gogh')).length, 1);
  assert.equal(tileImage('2s', 'classic'), 'tiles/Sou2.svg');
  assert.equal(tileImage('2s', 'van-gogh', true), 'tiles/Back.svg');
  assert.equal(set.tiles.find(tile => tile.tile === '3s').candidate, 'Vase A');
  assert.equal(set.tiles.find(tile => tile.tile === '6m').candidate, 'Scarlet Loom C');
  assert.equal(set.tiles.find(tile => tile.tile === '5z').candidate, 'Almond Blossom C');
  assert.equal(set.tiles.find(tile => tile.tile === '7z').candidate, 'Painter C');
});

test('the flute painting is shown in the gallery without changing the fourteen-tile sample hand', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString();
  assert.match(preview, /The Bamboo Duet B for 2 bamboo/);
  assert.match(preview, /Flutes B · Two bamboo/);
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
});

test('--only=2s reproduces the exact flute painting, preserves all other artwork and is repeatable', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-two-flutes-'));
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
  for (const tile of set.tiles.filter(tile => tile.tile !== '2s')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  run('--only=2s');
  const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '2s'), entry);
  assert.deepEqual(generated.superseded, set.superseded);
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.runtime));
  run('--only=2s');
  assert.equal(hash(readFileSync(path.join(temporary, record.runtime))), runtimeHash);
  for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
