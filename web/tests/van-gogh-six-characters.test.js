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
const record = JSON.parse(read('docs/design/van-gogh/six-characters-scarlet-loom-c.json'));
const entry = set.tiles.find(tile => tile.tile === '6m');
// The face is the approved study with only its picture's box moved.
const withoutBox = svg => svg.toString('utf8').replace(/ x="[^"]*" y="[^"]*" width="[^"]*" height="[^"]*"/, '');

test('six characters uses the exact selected Scarlet Loom C, not Lemon Terrace or the other proposals', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Scarlet Loom C');
  assert.equal(entry.name, 'Man6');
  assert.equal(entry.label, 'Six characters');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/32-six-characters-scarlet-loom-c-approved.svg');
  assert.equal(entry.source, record.source);
  assert.equal(record.tile, '6m');
  const source = read(entry.source), runtime = read(record.runtime);
  assert.equal(withoutBox(runtime), withoutBox(source));
  assert.equal(hash(source), '1530fc9cfa0b9a4dcbb6bb7129d05300467a6c95c21ed73647285389dd04dbed');
  assert.equal(hash(runtime), '952f5ba7b31e9ff6253fc2f8660fe23088b1e3a6826e30df59f60ff0c3d47633');
  assert.equal(hash(runtime), entry.svgSha256);
  assert.equal(hash(runtime), record.runtimeSvgSha256);
  assert.equal(hash(source), record.svgSha256);
  assert.equal(record.originalSha256, 'de7bf8b8be8d6a1a445f045e3bb73e152a2ea2bc767c8b37ca6beb010cac2f73');
  assert.deepEqual(record.originalDimensions, [1086, 1448]);
  assert.deepEqual(record.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
});

test('Scarlet Loom preserves the full portrait with zero stretch and zero extra bleed', () => {
  const svg = read(record.runtime).toString('utf8');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.ok(svg.includes('x="0" y="0" width="300" height="400" preserveAspectRatio="none"'));
  assert.deepEqual(entry.fit, { painting: { width: 1086, height: 1448 }, stretch: 0, anchor: 0.5, bleedPixels: 0, image: { x: 0, y: 0, width: 300, height: 400 } });
  assert.doesNotMatch(svg, /<script|<foreignObject|href="https?:/i);
  const embedded = svg.match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(raster.length, 43761);
  assert.equal(hash(raster), '6b1754cbada078e40e15ef3e9791355e574d11a8c595d3a3fcbb7470c828d806');
  assert.equal(hash(raster), entry.rasterSha256);
  assert.equal(hash(raster), record.rasterSha256);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.equal(raster.readUInt32BE(ispe + 8), 300);
  assert.equal(raster.readUInt32BE(ispe + 12), 400);
});

test('Lemon Terrace remains archived exactly and is superseded only for six characters', () => {
  assert.equal(hash(read(record.archivedSvg)), 'b448abf38e3f768de796fb65d9b1265c7d146ae3792264796bb2dc67407b319d');
  assert.equal(hash(read(record.archivedSvg)), record.archivedSvgSha256);
  assert.equal(hash(read(record.supersededSource)), 'dcd95ef26e4a2dec009d4d5c12aa8b87cd00a57be03f395349707b5fb3808418');
  const previous = set.superseded.filter(tile => tile.tile === '6m');
  assert.equal(previous.length, 1);
  assert.equal(previous[0].candidate, 'Lemon Terrace');
  assert.equal(previous[0].activeCandidate, 'Scarlet Loom C');
  assert.equal(previous[0].archivedSvg, record.archivedSvg);
  assert.equal(set.tiles.length, 34);
});

test('six characters is registered and preloaded once without changing Classic or hidden tiles', () => {
  assert.equal(set.tiles.filter(tile => tile.tile === '6m').length, 1);
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '6m').length, 1);
  assert.ok(!set.remaining.includes('6m'));
  assert.deepEqual(set.tiles.map(tile => tile.tile), VAN_GOGH_APPROVED);
  assert.deepEqual([...VAN_GOGH_APPROVED, ...set.remaining].sort(), [...TILE_TYPES].sort());
  assert.equal(tileImage('6m', 'van-gogh'), 'tiles/van-gogh/approved/Man6.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('6m', 'van-gogh')).length, 1);
  assert.equal(tileImage('6m', 'classic'), 'tiles/Man6.svg');
  assert.equal(tileImage('6m', 'van-gogh', true), 'tiles/Back.svg');
  assert.notEqual(tileImage('6m', 'van-gogh'), tileImage('6s', 'van-gogh'));
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, record.svgSha256);
});

test('the preview includes Scarlet Loom C and keeps a fourteen-tile hand', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
  for (const name of ['Man6', 'Man5', 'Pin9', 'Pin3', 'Sou9', 'Sou6', 'Sou8']) {
    assert.ok(hand.includes(`approved/${name}.svg`));
  }
  assert.match(preview, /The Scarlet Loom C for 6 characters/);
});

test('--only=6m reproduces the fitted image, preserves other artwork and is repeatable', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-six-characters-'));
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
  for (const tile of set.tiles.filter(tile => tile.tile !== '6m')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  assert.equal(run('--only=6m').trim(), `Exported ${set.tiles.length} approved Van Gogh faces; ${set.remaining.length} identities show their names until painted.`);
  const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '6m'), entry);
  assert.deepEqual(generated.remaining, set.remaining);
  assert.deepEqual(generated.superseded, set.superseded);
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.runtime));
  const first = readFileSync(path.join(temporary, record.runtime));
  run('--only=6m');
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), first);
  for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
