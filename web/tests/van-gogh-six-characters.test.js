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
const record = JSON.parse(read('docs/design/van-gogh/six-characters-lemon-terrace.json'));
const entry = set.tiles.find(tile => tile.tile === '6m');

test('six characters uses the approved red calligraphy and lemon landscape, not the sunflower study', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Lemon Terrace');
  assert.equal(entry.name, 'Man6');
  assert.equal(entry.label, 'Six characters');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/19-six-characters-lemon-terrace-approved.svg');
  assert.equal(entry.source, record.source);
  assert.equal(record.tile, '6m');
  const source = read(entry.source), runtime = read(record.runtime);
  assert.deepEqual(runtime, source);
  assert.equal(hash(source), 'dcd95ef26e4a2dec009d4d5c12aa8b87cd00a57be03f395349707b5fb3808418');
  assert.equal(hash(source), entry.svgSha256);
  assert.equal(hash(source), record.svgSha256);
  assert.equal(record.originalSha256, '5d112801bf30a7e9f16b145c8e84deb71e8e13474b5cfe165189b23c3b4a55f9');
  assert.deepEqual(record.originalDimensions, [1295, 1214]);
  assert.deepEqual(record.originalCrop, { x: 0, y: 0, width: 1295, height: 1214 });
  assert.match(record.processing, /not a proportional resize/);
});

test('Lemon Terrace preserves the approved embedded raster and standard face geometry', () => {
  const svg = read(record.runtime).toString('utf8');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.match(svg, /preserveAspectRatio="none"/);
  assert.doesNotMatch(svg, /<script|<foreignObject|href="https?:/i);
  const embedded = svg.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(raster.length, 53298);
  assert.equal(hash(raster), '1fb6ccae9c621b881d676d89d79e452ce6b79e44f365dfc89df404de2ea80cf0');
  assert.equal(hash(raster), entry.rasterSha256);
  assert.equal(hash(raster), record.rasterSha256);
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 16), 'WEBPVP8 ');
  assert.equal(raster.readUInt16LE(26) & 0x3fff, 300);
  assert.equal(raster.readUInt16LE(28) & 0x3fff, 400);
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
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, entry.svgSha256);
});

test('the preview includes Lemon Terrace and keeps a fourteen-tile hand', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
  for (const name of ['Man6', 'Man5', 'Pin9', 'Pin3', 'Sou9', 'Sou6', 'Sou8']) {
    assert.ok(hand.includes(`approved/${name}.svg`));
  }
  assert.match(preview, /Lemon Terrace for 6 characters/);
});

test('--only=6m copies the approved image exactly, preserves other artwork and is repeatable', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-six-characters-'));
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
  for (const tile of set.tiles.filter(tile => tile.tile !== '6m')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  assert.equal(run('--only=6m').trim(), `Exported ${set.tiles.length} approved Van Gogh faces; ${set.remaining.length} identities use Classic artwork.`);
  const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '6m'), entry);
  assert.deepEqual(generated.remaining, set.remaining);
  assert.deepEqual(generated.superseded, set.superseded);
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.source));
  const first = readFileSync(path.join(temporary, record.runtime));
  run('--only=6m');
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), first);
  for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
