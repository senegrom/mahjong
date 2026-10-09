import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = new URL('../../', import.meta.url);
const read = path => readFileSync(new URL(path, root));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const provenance = JSON.parse(read('docs/design/van-gogh/three-disks-lanterns.json'));
const source = 'docs/design/van-gogh/studies/17-three-disks-cafe-lanterns-b-approved.svg';
const runtime = 'web/public/tiles/van-gogh/approved/Pin3.svg';
const entry = set.tiles.find(tile => tile.tile === '3p');
const expectedSvg = 'a5fe322fffbe4717b57897209fca1cbaa737989bb318884a510330c5fb6d8713';
const expectedRaster = '74d9a2bf1807c3f58a4f5728794d83dfacfdeac1b7fd3400051c35077d57e105';
// The face is the approved study with only its picture's box moved.
const withoutBox = svg => svg.toString('utf8').replace(/ x="[^"]*" y="[^"]*" width="[^"]*" height="[^"]*"/, '');

test('Three Cafe Lanterns B is the approved Van Gogh three disks, registered once', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Lanterns B');
  assert.equal(entry.name, 'Pin3');
  assert.equal(entry.source, source);
  assert.equal(entry.svg, 'approved/Pin3.svg');
  assert.equal(entry.status, 'approved');
  assert.equal(entry.png, undefined);
  assert.equal(set.tiles.filter(tile => tile.tile === '3p').length, 1);
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '3p').length, 1);
  assert.ok(!set.remaining.includes('3p'));
  assert.deepEqual(set.tiles.map(tile => tile.tile), VAN_GOGH_APPROVED);
  assert.deepEqual([...set.tiles.map(tile => tile.tile), ...set.remaining].sort(), [...TILE_TYPES].sort());
});

test('the lantern face preserves the hash-pinned approved crop export, differing only in its fit', () => {
  assert.equal(hash(read(source)), expectedSvg);
  assert.equal(withoutBox(read(runtime)), withoutBox(read(source)));
  assert.equal(entry.svgSha256, hash(read(runtime)));
  assert.equal(set.sources.find(item => item.source === source).sha256, expectedSvg);
  assert.equal(provenance.sourceSha256, expectedSvg);
  assert.equal(provenance.source, source);
  assert.equal(provenance.runtime, runtime);
  assert.equal(provenance.candidate, entry.candidate);
  assert.equal(provenance.title, 'Three Café Lanterns');
});

test('lantern SVG is self-contained and shows the lanterns at their painted proportions', () => {
  const svg = read(runtime).toString('utf8');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  // The 300 × 400 export squeezed the 442 × 860 panel; the face's box restores its
  // shape and crops it to the tile, below the lanterns more than above them.
  const { x, y, width, height } = entry.fit.image;
  assert.ok(svg.includes(`x="${x}" y="${y}" width="${width}" height="${height}" preserveAspectRatio="none"`));
  assert.deepEqual(entry.fit.painting, { width: provenance.originalCrop.width, height: provenance.originalCrop.height });
  assert.ok(Math.abs(width / height / (442 / 860) - 1 - entry.fit.stretch) < 0.001);
  const above = -y, below = height - 400 - above;
  assert.ok(above > 0 && above < below);
  assert.equal((svg.match(/<image\b/g) ?? []).length, 1);
  assert.doesNotMatch(svg, /<(?:script|foreignObject|iframe)\b/);
  assert.doesNotMatch(svg, /(?:href|src)="(?!data:)/);
  const embedded = svg.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(raster.length, 68688);
  assert.equal(hash(raster), expectedRaster);
  assert.equal(entry.rasterSha256, expectedRaster);
  assert.equal(entry.rasterMimeType, 'image/webp');
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 16), 'WEBPVP8 ');
  assert.equal(raster.readUInt32LE(4) + 8, raster.length);
  assert.equal(raster.readUInt16LE(26) & 0x3fff, 300);
  assert.equal(raster.readUInt16LE(28) & 0x3fff, 400);
});

test('lantern provenance distinguishes the full-resolution original from the game export', () => {
  assert.equal(provenance.originalBoardSha256, '878e3350a936d15eec6658fd4f8d698e9ee8fc9f8c294565124776bab30d3591');
  assert.deepEqual(provenance.originalSize, { width: 1448, height: 1086 });
  assert.deepEqual(provenance.originalCrop, { x: 503, y: 127, width: 442, height: 860 });
  assert.equal(provenance.fullResolutionCropSha256, '27628def3f7ff38810daf9fcac5b6ae0efa48b2fe76417eb5c16f0f4999447b6');
  assert.equal(provenance.raster.sha256, expectedRaster);
  assert.equal(provenance.raster.quality, 90);
  assert.equal(provenance.raster.lossless, false);
  assert.equal(provenance.approvalArchive, 'van-gogh-3-disks-lanterns-prepared.zip');
});

test('lantern preloading is unique and hidden tiles never reveal their identity', () => {
  const url = 'tiles/van-gogh/approved/Pin3.svg';
  assert.equal(tileImage('3p', 'van-gogh'), url);
  assert.equal(TILE_IMAGE_URLS.filter(item => item === url).length, 1);
  for (const face of ['classic', 'van-gogh', 'matisse', 'dali']) {
    assert.equal(tileImage('3p', face, true), 'tiles/Back.svg');
    assert.equal(tileImage(null, face), 'tiles/Back.svg');
  }
  assert.equal(tileImage('3p', 'classic'), 'tiles/Pin3.svg');
  assert.equal(tileImage('3p', 'matisse'), 'tiles/matisse/approved/Pin3.svg');
  assert.equal(tileImage('3p', 'dali'), 'tiles/dali/approved/Pin3.svg');
});

test('the selective exporter and preview include the lanterns without losing nine bamboo', () => {
  const exporter = read('web/scripts/export-van-gogh-tiles.mjs').toString('utf8');
  assert.ok(exporter.includes(source));
  assert.ok(exporter.includes("['Lanterns B', '3p', 'Three disks', 'threeLanterns', [0, 0, 300, 400]]"));
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  assert.ok(preview.includes('approved/Pin3.svg'));
  assert.ok(preview.includes('Three Café Lanterns'));
  assert.ok(preview.includes('approved/Sou9.svg'));
  for (let number = 1; number <= 9; number += 1) assert.ok(VAN_GOGH_APPROVED.includes(`${number}s`));
});
