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
import { TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = data => createHash('sha256').update(data).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const record = JSON.parse(read('docs/design/van-gogh/two-characters-roof-garden-c.json'));
const entry = set.tiles.find(tile => tile.tile === '2m');
const sourceHash = '3a1a10ccba933fc27b5b267d67b9c3f394ef0f6c1e83fd648be38ccc4420059e';
const runtimeHash = 'd3aa6d977455047b42716dec3069174e86b99171d04517de05544f3aa4cf72c6';
const rasterHash = '097c1b200cf2603502818fd04249308cb5f8c2317b82ef20664f29f8bc71df64';

test('two characters uses the exact right-hand roof-and-flower-pots C selection', () => {
  assert.equal(entry.candidate, 'Roof Garden C');
  assert.equal(entry.name, 'Man2');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/37-two-characters-roof-garden-c-approved.svg');
  assert.equal(record.originalFilename, 'impasto_lanternlit_temple_gate.png');
  assert.equal(record.originalSha256, '41985fa1d76b04bbdface9459132c3b1f8379e7cf857854edc3334fec68237f7');
  assert.deepEqual(record.originalDimensions, [1086, 1448]);
  assert.deepEqual(record.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.equal(hash(read(record.source)), sourceHash);
  assert.equal(hash(read(record.runtime)), runtimeHash);
  assert.equal(record.sourceSha256, sourceHash);
  assert.equal(record.runtimeSvgSha256, runtimeHash);
  assert.equal(entry.svgSha256, runtimeHash);
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, sourceHash);
});

test('the two roofs and flowers retain their entire native portrait with zero stretch or bleed', () => {
  assert.deepEqual(entry.fit, {
    painting: { width: 1086, height: 1448 }, stretch: 0, anchor: 0.5, bleedPixels: 0,
    image: { x: 0, y: 0, width: 300, height: 400 },
  });
  const svg = read(record.runtime).toString();
  assert.equal(read(record.source).toString().replace('x="-3" y="-4" width="306" height="408"', 'x="0" y="0" width="300" height="400"'), svg);
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.doesNotMatch(svg, /<script|<foreignObject|<text|href="https?:/i);
  const embedded = svg.match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(raster.length, 40984);
  assert.equal(hash(raster), rasterHash);
  assert.equal(entry.rasterSha256, rasterHash);
  assert.equal(record.rasterSha256, rasterHash);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [300, 400]);
});

test('original Night Cafe stays recoverable byte-for-byte without an obsolete active PNG', () => {
  const previous = set.superseded.find(tile => tile.tile === '2m' && tile.candidate === 'Characters B');
  assert.equal(previous.activeCandidate, 'Roof Garden C');
  assert.equal(previous.source, 'docs/design/van-gogh/studies/07-two-characters-night-cafe.png');
  assert.equal(previous.archivedSvg, record.archivedSvg);
  assert.equal(previous.archivedPng, record.archivedPng);
  assert.equal(hash(read(record.archivedSvg)), record.archivedSvgSha256);
  assert.equal(hash(read(record.archivedPng)), record.archivedPngSha256);
  const original = read(previous.source);
  assert.equal(createHash('sha1').update(Buffer.from(`blob ${original.length}\0`)).update(original).digest('hex'), '5ce8e6822ece3d11b0e33b21a666b6272717adc5');
  assert.deepEqual(read(record.archivedPng), original);
  assert.equal(existsSync(path.join(root, 'web/public/tiles/van-gogh/approved/Man2.png')), false);
});

test('Roof Garden C preloads once and keeps all 34 identities, other sets and hidden tiles intact', () => {
  assert.equal(set.tiles.length, 34);
  assert.equal(set.remaining.length, 0);
  assert.deepEqual(set.tiles.map(tile => tile.tile), VAN_GOGH_APPROVED);
  assert.deepEqual([...VAN_GOGH_APPROVED].sort(), [...TILE_TYPES].sort());
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '2m').length, 1);
  assert.equal(tileImage('2m', 'van-gogh'), 'tiles/van-gogh/approved/Man2.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('2m', 'van-gogh')).length, 1);
  assert.equal(tileImage('2m', 'classic'), 'tiles/Man2.svg');
  assert.equal(tileImage('2m', 'van-gogh', true), 'tiles/Back.svg');
  for (const [tile, candidate] of [['2s', 'Flutes B'], ['3s', 'Vase A'], ['4s', 'Sunlit Garden B'], ['6m', 'Scarlet Loom C'], ['9m', 'Poppy Hillside B'], ['5z', 'Almond Blossom C'], ['7z', 'Painter C']]) {
    assert.equal(set.tiles.find(entry => entry.tile === tile).candidate, candidate);
  }
});

test('Roof Garden C appears in the gallery and the unchanged fourteen-tile sample hand', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString();
  assert.match(preview, /Roof Garden C · Two characters/);
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
  assert.ok(hand.includes('approved/Man2.svg'));
});

test('--only=2m is repeatable and preserves every other playable image', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-two-roofs-'));
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
  for (const tile of set.tiles.filter(tile => tile.tile !== '2m')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  for (let i = 0; i < 2; i++) {
    run('--only=2m');
    const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
    assert.deepEqual(generated.tiles.find(tile => tile.tile === '2m'), entry);
    assert.deepEqual(generated.superseded, set.superseded);
    assert.equal(hash(readFileSync(path.join(temporary, record.runtime))), runtimeHash);
    for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  }
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
