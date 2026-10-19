import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { tileImage, TILE_IMAGE_URLS } from '../src/lib/tile-faces.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';

const root = new URL('../../', import.meta.url);
const read = relative => readFileSync(new URL(relative, root));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const source = 'docs/design/van-gogh/studies/28-north-wind-winter-b-approved.svg';
const runtime = 'web/public/tiles/van-gogh/approved/Pei.svg';
const manifestPath = 'web/public/tiles/van-gogh/manifest.json';
const manifest = JSON.parse(read(manifestPath));
const record = JSON.parse(read('docs/design/van-gogh/north-wind-winter.json'));
const entry = manifest.tiles.find(item => item.tile === '4z');
const SOURCE_HASH = '731701567e7f7ee83311e8d19953e339f87b4997d9932f24f0c92aa6c3cb142f';
const SVG_HASH = 'c4f2aecb4f54d8cfc7d207cc7a7545d1a7ad99270d5c7514b6b13719ad6e8d81';
const RASTER_HASH = 'fc5df8e9e4b9666885e5c2b20e45071574ee580bc57f6e9207b6d81e256e0b24';

test('North wind uses the selected winter B, not the cypress or harbour alternatives', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Winter B');
  assert.equal(entry.source, source);
  assert.equal(entry.svg, 'approved/Pei.svg');
  assert.equal(hash(read(source)), SOURCE_HASH);
  assert.equal(hash(read(runtime)), SVG_HASH);
  assert.equal(read(source).toString().replace('x="-3" y="-4" width="306" height="408"', 'x="0" y="0" width="300" height="400"'), read(runtime).toString());
  assert.equal(record.originalFilename, 'windy_northern_landscapes_triptych.png');
  assert.equal(record.originalSha256, '0e5e57ab728bbe24f6f7856fd13714e2d2d005cf448a6b5d3a2571714f1d9622');
  assert.deepEqual(record.originalDimensions, [1448, 1086]);
  assert.equal(record.sourceSha256, SOURCE_HASH);
  assert.equal(record.runtimeSvgSha256, SVG_HASH);
});

test('winter North keeps the documented 3:4 crop with no stretch or bleed and a self-contained raster', () => {
  const svg = read(runtime).toString();
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.equal((svg.match(/<image\b/g) ?? []).length, 1);
  assert.match(svg, /x="0" y="0" width="300" height="400" preserveAspectRatio="none"/);
  assert.deepEqual(entry.fit.painting, { width: 456, height: 608 });
  assert.equal(entry.fit.stretch, 0);
  assert.equal(entry.fit.bleedPixels, 0);
  assert.deepEqual(entry.fit.image, { x: 0, y: 0, width: 300, height: 400 });
  const match = svg.match(/href="data:(image\/avif);base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(match);
  const bytes = Buffer.from(match[2], 'base64');
  assert.equal(bytes.length, 37117);
  assert.equal(bytes.subarray(4, 12).toString(), 'ftypavif');
  assert.equal(hash(bytes), RASTER_HASH);
  assert.equal(entry.rasterSha256, RASTER_HASH);
  assert.equal(record.rasterSha256, RASTER_HASH);
  assert.equal(record.rasterMimeType, 'image/avif');
  assert.doesNotMatch(svg, /(?:href|src)="(?:https?:|\/\/)/);
});

test('winter North is registered and preloaded once without revealing hidden tiles', () => {
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '4z').length, 1);
  assert.equal(manifest.tiles.filter(item => item.tile === '4z').length, 1);
  assert.equal(TILE_IMAGE_URLS.filter(url => url === 'tiles/van-gogh/approved/Pei.svg').length, 1);
  assert.equal(tileImage('4z', 'van-gogh'), 'tiles/van-gogh/approved/Pei.svg');
  assert.equal(tileImage('4z', 'van-gogh', true), 'tiles/Back.svg');
  assert.equal(tileImage(null, 'van-gogh'), 'tiles/Back.svg');
  assert.equal(tileImage('4z', 'classic'), 'tiles/Pei.svg');
  assert.equal(tileImage('4z', 'matisse'), 'tiles/matisse/approved/Pei.svg');
  for (const suit of ['m', 'p', 's']) {
    for (let rank = 1; rank <= 9; rank += 1) assert.ok(VAN_GOGH_APPROVED.includes(`${rank}${suit}`));
  }
  assert.ok(!manifest.remaining.includes('4z'));
  assert.deepEqual([...VAN_GOGH_APPROVED, ...manifest.remaining].sort(), [...TILE_TYPES].sort());
});

test('winter North appears in the gallery without extending the fourteen-tile example hand', () => {
  const html = read('web/public/tiles/van-gogh/preview.html').toString();
  assert.match(html, /Winter B · North wind/);
  const rack = html.match(/<div class="rack"[^>]*>(.*?)<\/div>/s)?.[1];
  assert.ok(rack);
  assert.equal((rack.match(/<img\b/g) ?? []).length, 14);
  assert.match(html, /<img src="approved\/Pei\.svg" alt="North wind"/);
});

test('--only=4z reproduces the approved winter North exactly without rewriting other artwork', t => {
  // In a copy of everything the export reads and writes, never in the
  // repository: other test files read the manifest while this one runs.
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-north-winter-'));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const folder = 'web/public/tiles/van-gogh/';
  // What the export writes, as committed: a Windows checkout gives these CRLF.
  const generated = ['web/src/lib/van-gogh-faces.js', manifestPath, `${folder}preview.html`];
  const copied = [
    'web/scripts/export-van-gogh-tiles.mjs', 'web/scripts/face-fit.mjs', 'web/src/lib/tiles.js', ...generated,
    ...manifest.sources.map(item => item.source),
    ...readdirSync(new URL(`${folder}approved/`, root)).map(name => `${folder}approved/${name}`),
  ];
  for (const relative of copied) {
    const target = path.join(temporary, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, generated.includes(relative) ? read(relative).toString('utf8').replaceAll('\r\n', '\n') : read(relative));
  }
  writeFileSync(path.join(temporary, 'package.json'), '{"type":"module"}');
  // Every file under the copy, so a write anywhere, a new file included, is seen.
  const tree = () => new Map(readdirSync(temporary, { recursive: true, withFileTypes: true })
    .filter(item => item.isFile())
    .map(item => [path.relative(temporary, path.join(item.parentPath, item.name)),
      hash(readFileSync(path.join(item.parentPath, item.name)))]));
  const before = tree();
  for (let attempt = 0; attempt < 2; attempt += 1) {
    execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), '--only=4z'], { stdio: 'pipe' });
    assert.deepEqual(tree(), before);
    assert.equal(hash(readFileSync(path.join(temporary, runtime))), SVG_HASH);
  }
});


test('the prior North ribbons and their original source remain recoverable unchanged', () => {
  const archived = manifest.superseded.find(item => item.tile === '4z' && item.candidate === 'North B');
  assert.ok(archived);
  assert.equal(archived.activeCandidate, 'Winter B');
  assert.equal(hash(read(record.archivedSvg)), record.archivedSvgSha256);
  assert.equal(hash(read(record.archivedPng)), record.archivedPngSha256);
  assert.equal(hash(read(record.supersededSource)), '3a51fc2f26cbd476cce085d93b52a5a81c5a17d14d4bcf00a0597a402435588e');
  assert.equal(record.previousNorthEntry.candidate, 'North B');
  assert.deepEqual(record.originalCrop, { x: 497, y: 126, width: 456, height: 608 });
  assert.deepEqual(record.fullPanelCrop, { x: 497, y: 108, width: 456, height: 856 });
});
