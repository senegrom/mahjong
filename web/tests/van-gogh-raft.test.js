import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const provenance = JSON.parse(read('docs/design/van-gogh/eight-bamboo-raft.json'));
const entry = set.tiles.find(tile => tile.tile === '8s');

test('Bamboo Raft preserves the approved SVG and embedded eight-bamboo painting', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Bamboo Raft');
  assert.equal(entry.name, 'Sou8');
  assert.equal(entry.source, provenance.source);
  const source = read(entry.source), runtime = read(provenance.runtime);
  assert.deepEqual(runtime, source);
  assert.equal(hash(runtime), '94f1651a4aff20b81f82467e7183c886c189c0e5f2a69824026313798a35a856');
  assert.equal(hash(runtime), entry.svgSha256);
  assert.equal(hash(runtime), provenance.svgSha256);
  assert.equal(provenance.originalSha256, '19a9ca0948f628609844b640a94b94ba29ceb33108c751fcae9e20ab36aaa0fa');
  assert.deepEqual(provenance.originalDimensions, [1024, 1536]);
  const svg = source.toString('utf8');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  const embedded = svg.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(hash(raster), 'c0dbf2f2b2793681a2b31d28282b80113b4cd33f33b7bb50d52a6ffb69580aa3');
  assert.equal(hash(raster), entry.rasterSha256);
  assert.equal(hash(raster), provenance.rasterSha256);
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 16), 'WEBPVP8 ');
  assert.equal(raster.readUInt16LE(26) & 0x3fff, 300);
  assert.equal(raster.readUInt16LE(28) & 0x3fff, 400);
});

test('eight bamboo is approved exactly once and appears in the preview', () => {
  assert.equal(set.tiles.length, VAN_GOGH_APPROVED.length);
  assert.equal(set.remaining.length, 34 - VAN_GOGH_APPROVED.length);
  assert.equal(set.tiles.filter(tile => tile.tile === '8s').length, 1);
  assert.ok(!set.remaining.includes('8s'));
  assert.deepEqual(VAN_GOGH_APPROVED, set.tiles.map(tile => tile.tile));
  assert.equal(new Set([...VAN_GOGH_APPROVED, ...set.remaining]).size, 34);
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, entry.svgSha256);
  assert.match(read('web/public/tiles/van-gogh/preview.html').toString('utf8'), /approved\/Sou8\.svg/);
});

test('--only=8s preserves all other artwork and copies the raft byte-for-byte', t => {
  // Isolated fixtures exercise the real exporter without rewriting repository files.
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-raft-'));
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
  for (const tile of set.tiles.filter(tile => tile.tile !== '8s')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  assert.equal(run('--only=8s').trim(), `Exported ${set.tiles.length} approved Van Gogh faces; ${set.remaining.length} identities use Classic artwork.`);
  for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  assert.deepEqual(readFileSync(path.join(temporary, provenance.runtime)), read(provenance.source));
  const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '8s'), entry);
  assert.deepEqual(generated.remaining, set.remaining);
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
