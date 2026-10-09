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
const provenance = JSON.parse(read('docs/design/van-gogh/six-bamboo-green-still-life.json'));
const entry = set.tiles.find(tile => tile.tile === '6s');

test('six bamboo uses the exact approved green B still life, not a five-stalk study', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Six B (green)');
  assert.equal(entry.name, 'Sou6');
  assert.equal(entry.source, provenance.source);
  const source = read(entry.source), runtime = read(provenance.runtime);
  assert.deepEqual(runtime, source);
  assert.equal(hash(runtime), '8cf0cb5071682f70ae26ac336b3f8ce22e994444a73c7cbbf1ef1979c6e36561');
  assert.equal(hash(runtime), entry.svgSha256);
  assert.equal(hash(runtime), provenance.svgSha256);
  assert.equal(provenance.originalFilename, 'six_stalk_bamboo_still_life.png');
  assert.equal(provenance.originalSha256, '40bd5b533cd3545a40fbc7019fde14f0134a38cb5f0d7a794c2d241bea2d3936');
  assert.deepEqual(provenance.originalDimensions, [1086, 1448]);
  assert.deepEqual(provenance.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  const svg = source.toString('utf8');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  const embedded = svg.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(hash(raster), '91cadccb181114f3e51e923aadedd4d04fc7c6d95ac55a154de07dd6e89bd0db');
  assert.equal(hash(raster), entry.rasterSha256);
  assert.equal(hash(raster), provenance.rasterSha256);
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 16), 'WEBPVP8 ');
  assert.equal(raster.readUInt16LE(26) & 0x3fff, 300);
  assert.equal(raster.readUInt16LE(28) & 0x3fff, 400);
});

test('green still life replaces J without adding a tile or changing the 14-tile preview hand', () => {
  assert.equal(set.tiles.length, VAN_GOGH_APPROVED.length);
  assert.equal(set.remaining.length, 34 - VAN_GOGH_APPROVED.length);
  assert.equal(set.tiles.filter(tile => tile.tile === '6s').length, 1);
  assert.ok(!set.remaining.includes('6s'));
  assert.deepEqual(VAN_GOGH_APPROVED, set.tiles.map(tile => tile.tile));
  assert.equal(new Set([...VAN_GOGH_APPROVED, ...set.remaining]).size, 34);
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, entry.svgSha256);
  assert.deepEqual(set.superseded.find(tile => tile.tile === '6s'), {
    candidate: 'J', tile: '6s', activeCandidate: 'Six B (green)',
    source: 'docs/design/van-gogh/studies/02-van-gogh-concepts.png',
  });
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.match(hand, /approved\/Sou6\.svg/);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
});

test('--only=6s preserves every other face, including the raft SVG, and copies B exactly', t => {
  // Sentinel fixtures prove that unselected production images are not rewritten.
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-six-'));
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
  for (const tile of set.tiles.filter(tile => tile.tile !== '6s')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  assert.equal(run('--only=6s').trim(), `Exported ${set.tiles.length} approved Van Gogh faces; ${set.remaining.length} identities use Classic artwork.`);
  for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  assert.deepEqual(readFileSync(path.join(temporary, provenance.runtime)), read(provenance.source));
  const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '6s'), entry);
  assert.deepEqual(generated.remaining, set.remaining);
  assert.deepEqual(generated.superseded, set.superseded);
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
