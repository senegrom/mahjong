import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { TILE_TYPES, tileFile } from '../src/lib/tiles.js';
import { faceImage, fitRecord, placeStudyImage } from '../scripts/face-fit.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const record = JSON.parse(read('docs/design/van-gogh/five-bamboo-standalone-b.json'));
const entry = set.tiles.find(tile => tile.tile === '5s');
const sourceHash = 'd4e9942f0d1b70ca6e56e1c9bd4aab525062dc9a5fb102d2d397bdb681ca2d64';
const runtimeHash = 'b1cac0b498e2ac98717b8ba6a770eff207f3a0fa48b700a5e660466381747054';
const rasterHash = 'ab4b1ec91ef46b24773d53bf03bb5971c8054aa3900902ab2964476dc7038553';

test('five bamboo uses the confirmed standalone middle painting, never the later triptych', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Copper Sunset Standalone B');
  assert.equal(entry.name, 'Sou5');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/38-five-bamboo-standalone-b-approved.svg');
  assert.equal(entry.source, record.source);
  assert.equal(record.originalFilename, 'impasto_bamboo_at_sunset.png');
  assert.equal(record.originalSha256, '41e71c3517d1ba0418625b7830e2c497b2c5d114e7bdf6d5218c6f01a230793b');
  assert.match(record.approval, /lower-left rock/);
  assert.match(record.approval, /rejecting the later triptych/);
  assert.equal(hash(read(record.source)), sourceHash);
  assert.equal(hash(read(record.runtime)), runtimeHash);
  assert.equal(record.sourceSha256, sourceHash);
  assert.equal(record.svgSha256, runtimeHash);
  assert.equal(entry.svgSha256, runtimeHash);
});

test('the AVIF is the pinned 300 by 400 export of the approved painting', () => {
  const embedded = read(record.source).toString().match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(hash(raster), rasterHash);
  assert.equal(record.rasterSha256, rasterHash);
  assert.equal(entry.rasterSha256, rasterHash);
  assert.equal(entry.rasterMimeType, 'image/avif');
  assert.equal(raster.length, 40596);
  assert.equal(raster.toString('ascii', 4, 8), 'ftyp');
  assert.equal(raster.toString('ascii', 8, 12), 'avif');
  const ispe = raster.indexOf('ispe');
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [300, 400]);
});

test('all five stalks and the rock retain the complete native portrait with no extra bleed', () => {
  assert.deepEqual(record.originalDimensions, [1086, 1448]);
  assert.deepEqual(record.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.deepEqual(record.fit, { painting: [1086, 1448], stretch: 0, anchor: 0.5, bleedPixels: 0 });
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 });
  assert.deepEqual(entry.fit, fitRecord(record.fit, [300, 400]));
  assert.deepEqual(entry.fit.image, { x: 0, y: 0, width: 300, height: 400 });
  assert.equal(entry.png, undefined);
  const face = read(record.runtime);
  assert.match(face.toString(), /viewBox="0 0 300 400"/);
  assert.match(face.toString(), /rx="26"/);
  assert.doesNotMatch(face.toString(), /<script|<text|href="https?:/);
  assert.deepEqual(face, placeStudyImage(read(record.source), faceImage(record.fit, [300, 400])));
});

test('the original Copper Sunset remains archived byte-for-byte without an obsolete active PNG', () => {
  const previous = set.superseded.find(tile => tile.tile === '5s' && tile.candidate === 'Copper Sunset');
  assert.ok(previous);
  assert.equal(previous.activeCandidate, entry.candidate);
  assert.equal(previous.source, 'docs/design/van-gogh/studies/15-five-bamboo-copper-sunset.webp');
  assert.equal(previous.archivedSvg, record.archivedSvg);
  assert.equal(previous.archivedPng, record.archivedPng);
  assert.equal(hash(read(record.archivedSvg)), 'b698dc152fa7ecb8ec5df5e71a1efb4912d79669e4e4e3e614712366a830d895');
  assert.equal(hash(read(record.archivedPng)), '84a457f7666ec7b2099cb2b65498e3791298fd1e1c65df87f54a86055b5840db');
  const oldEmbedded = read(record.archivedSvg).toString().match(/href="data:image\/png;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(oldEmbedded);
  assert.deepEqual(Buffer.from(oldEmbedded[1], 'base64'), read(record.archivedPng));
  assert.equal(existsSync(path.join(root, 'web/public/tiles/van-gogh/approved/Sou5.png')), false);
});

test('the replacement keeps all 34 identities, the gallery and the existing fourteen-tile hand', () => {
  assert.deepEqual(VAN_GOGH_APPROVED, set.tiles.map(tile => tile.tile));
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '5s').length, 1);
  assert.deepEqual([...VAN_GOGH_APPROVED].sort(), [...TILE_TYPES].sort());
  assert.deepEqual(set.remaining, []);
  assert.equal(entry.svg, `approved/${tileFile('5s')}.svg`);
  assert.equal(set.sources.find(source => source.source === record.source).sha256, sourceHash);
  const preview = read('web/public/tiles/van-gogh/preview.html').toString();
  assert.match(preview, /Copper Sunset Standalone B · Five bamboo/);
  const rack = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(rack);
  assert.equal((rack.match(/<img /g) ?? []).length, 14);
});

test('--only=5s repeats exactly and never rewrites the other faces', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-five-standalone-'));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const put = (relative, bytes) => {
    const target = path.join(temporary, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, bytes);
  };
  put('package.json', '{"type":"module"}');
  for (const relative of ['web/scripts/export-van-gogh-tiles.mjs', 'web/scripts/face-fit.mjs', 'web/src/lib/tiles.js']) put(relative, read(relative));
  for (const source of set.sources) put(source.source, source.source.endsWith('.svg') ? read(source.source) : Buffer.from(`source fixture ${source.id}`));
  const unchanged = [];
  for (const tile of set.tiles.filter(tile => tile.tile !== '5s')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`, bytes = Buffer.from(`unchanged fixture ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  for (let run = 0; run < 2; run++) {
    execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), '--only=5s'], { stdio: 'pipe' });
    assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.runtime));
    for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
    const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
    assert.deepEqual(generated.tiles.find(tile => tile.tile === '5s'), entry);
    assert.deepEqual(generated.remaining, set.remaining);
  }
});
