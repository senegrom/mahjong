import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { tileImage, TILE_IMAGE_URLS } from '../src/lib/tile-faces.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const provenance = JSON.parse(read('docs/design/van-gogh/five-characters-vineyard.json'));
const entry = set.tiles.find(tile => tile.tile === '5m');
const expectedSvg = '167d067218dcb1a9ae71ba43e0f4d4552ef39057c84c6f0d7f52ec566a3a9280';
const expectedRaster = 'c7b5707a3a4ac45425326ba9f38d29b47d5cb52b0a3f2a9d80d1ee5e561e7922';

test('five characters uses the approved first vineyard painting, not wheat or irises', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Vineyard A');
  assert.equal(entry.name, 'Man5');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/18-five-characters-vineyard-a-approved.svg');
  assert.equal(provenance.source, entry.source);
  assert.equal(provenance.runtime, 'web/public/tiles/van-gogh/approved/Man5.svg');
  assert.equal(provenance.originalFilename, 'vineyard_characters_at_golden_sunset.png');
  assert.equal(provenance.originalSha256, '770fcd714d8d8e99990e73a78389989bcb7d1b6ebc8c1229e62062c288c9503a');
  assert.deepEqual(provenance.originalDimensions, [1086, 1448]);
  assert.deepEqual(provenance.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  const source = read(entry.source), runtime = read(provenance.runtime);
  assert.deepEqual(runtime, source);
  for (const actual of [hash(source), entry.svgSha256, provenance.svgSha256]) assert.equal(actual, expectedSvg);
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, expectedSvg);
});

test('vineyard export keeps the standard face geometry and exact embedded raster', () => {
  const svg = read(provenance.runtime).toString('utf8');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.match(svg, /<title id="title">Five characters — Van Gogh<\/title>/);
  const embedded = svg.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  for (const actual of [hash(raster), entry.rasterSha256, provenance.rasterSha256]) assert.equal(actual, expectedRaster);
  assert.equal(raster.length, 55626);
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 16), 'WEBPVP8 ');
  assert.equal(raster.readUInt16LE(26) & 0x3fff, 300);
  assert.equal(raster.readUInt16LE(28) & 0x3fff, 400);
  assert.doesNotMatch(svg, /<script|<foreignObject|href="https?:/i);
});

test('vineyard is registered once and preloaded without changing Classic or hidden faces', () => {
  assert.equal(set.tiles.filter(tile => tile.tile === '5m').length, 1);
  assert.ok(!set.remaining.includes('5m'));
  assert.deepEqual(VAN_GOGH_APPROVED, set.tiles.map(tile => tile.tile));
  assert.deepEqual([...VAN_GOGH_APPROVED, ...set.remaining].sort(), [...TILE_TYPES].sort());
  assert.equal(tileImage('5m', 'van-gogh'), 'tiles/van-gogh/approved/Man5.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('5m', 'van-gogh')).length, 1);
  assert.equal(tileImage('5m', 'classic'), 'tiles/Man5.svg');
  assert.equal(tileImage('5m', 'van-gogh', true), 'tiles/Back.svg');
  for (const tile of ['2m', '3m', '4m', '3p', ...Array.from({ length: 9 }, (_, i) => `${i + 1}s`)]) {
    assert.ok(VAN_GOGH_APPROVED.includes(tile), `Previous selection ${tile} is retained`);
  }
});

test('the vineyard appears in the preview without exceeding fourteen hand tiles', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
  for (const name of ['Man5', 'Man2', 'Man3', 'Man4', 'Pin3', 'Sou6', 'Sou8', 'Sou9']) {
    assert.ok(hand.includes(`approved/${name}.svg`));
  }
  assert.match(preview, /Vineyard A · Five characters/);
});

test('--only=5m is repeatable and never rewrites other artwork', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-vineyard-'));
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
  for (const tile of set.tiles.filter(tile => tile.tile !== '5m')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  let firstManifest;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    assert.equal(run('--only=5m').trim(), `Exported ${set.tiles.length} approved Van Gogh faces; ${set.remaining.length} identities show their names until painted.`);
    for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
    assert.deepEqual(readFileSync(path.join(temporary, provenance.runtime)), read(provenance.source));
    const bytes = readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json'));
    const generated = JSON.parse(bytes);
    assert.deepEqual(generated.tiles.find(tile => tile.tile === '5m'), entry);
    assert.deepEqual(generated.remaining, set.remaining);
    assert.deepEqual(generated.superseded, set.superseded);
    if (firstManifest) assert.deepEqual(bytes, firstManifest);
    firstManifest = bytes;
  }
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
