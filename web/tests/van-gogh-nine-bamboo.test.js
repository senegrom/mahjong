import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { tileImage, TILE_IMAGE_URLS } from '../src/lib/tile-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const provenance = JSON.parse(read('docs/design/van-gogh/nine-bamboo-wind-chime.json'));
const entry = set.tiles.find(tile => tile.tile === '9s');

test('nine bamboo preserves the approved first wind-chime painting and its provenance', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Wind Chime A');
  assert.equal(entry.name, 'Sou9');
  assert.equal(entry.source, provenance.source);
  const source = read(entry.source), runtime = read(provenance.runtime);
  assert.deepEqual(runtime, source);
  assert.equal(hash(runtime), '8085bc2a65d372c8b28fc673063a4fd3de090849e4de6749a96de6c719da74ac');
  assert.equal(hash(runtime), entry.svgSha256);
  assert.equal(hash(runtime), provenance.svgSha256);
  assert.equal(provenance.originalFilename, 'nine_tube_bamboo_chime_under_moonlight.png');
  assert.equal(provenance.originalSha256, '0aa8568cf2f329f1fffa41eabef13135fcd1e265f82586cd2a4c6fccff8ceda1');
  assert.deepEqual(provenance.originalDimensions, [1086, 1448]);
  assert.deepEqual(provenance.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  const svg = source.toString('utf8');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  const embedded = svg.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(hash(raster), 'ab954a7208567a80d87d3be0f0ba15373dd5cc644feb8c766880f404435dc2f3');
  assert.equal(hash(raster), entry.rasterSha256);
  assert.equal(hash(raster), provenance.rasterSha256);
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 16), 'WEBPVP8 ');
  assert.equal(raster.readUInt16LE(26) & 0x3fff, 300);
  assert.equal(raster.readUInt16LE(28) & 0x3fff, 400);
});

test('all nine bamboo are painted; the chime is preloaded and appears in the 14-tile hand', () => {
  assert.deepEqual(VAN_GOGH_APPROVED, set.tiles.map(tile => tile.tile));
  for (let rank = 1; rank <= 9; rank++) {
    const tile = `${rank}s`;
    assert.equal(set.tiles.filter(entry => entry.tile === tile).length, 1);
    assert.ok(!set.remaining.includes(tile));
    assert.equal(tileImage(tile, 'van-gogh'), `tiles/van-gogh/approved/Sou${rank}.svg`);
    assert.ok(TILE_IMAGE_URLS.includes(tileImage(tile, 'van-gogh')));
  }
  assert.equal(new Set([...VAN_GOGH_APPROVED, ...set.remaining]).size, 34);
  assert.equal(set.remaining.length, 34 - VAN_GOGH_APPROVED.length);
  assert.equal(tileImage('9s', 'van-gogh', true), 'tiles/Back.svg');
  assert.equal(tileImage('9s', 'classic'), 'tiles/Sou9.svg');
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, entry.svgSha256);
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.match(hand, /approved\/Sou9\.svg/);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
});

test('--only=9s copies the chime exactly, preserves other artwork and is repeatable', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-nine-'));
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
  for (const tile of set.tiles.filter(tile => tile.tile !== '9s')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  for (let attempt = 0; attempt < 2; attempt++) {
    assert.equal(run('--only=9s').trim(), `Exported ${set.tiles.length} approved Van Gogh faces; ${set.remaining.length} identities show their names until painted.`);
    for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
    assert.deepEqual(readFileSync(path.join(temporary, provenance.runtime)), read(provenance.source));
    const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
    assert.deepEqual(generated.tiles.find(tile => tile.tile === '9s'), entry);
    assert.deepEqual(generated.remaining, set.remaining);
    assert.deepEqual(generated.superseded, set.superseded);
  }
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
