import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, mkdtempSync, mkdirSync, cpSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = p => readFileSync(path.join(root, p));
const hash = b => createHash('sha256').update(b).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const provenance = JSON.parse(read('docs/design/van-gogh/west-wind-clouds.json'));
const entry = set.tiles.find(t => t.tile === '3z');

test('Cloudborne West completes exactly all 34 Van Gogh identities', () => {
  assert.equal(set.tiles.length, 34);
  assert.equal(new Set(VAN_GOGH_APPROVED).size, 34);
  assert.deepEqual([...VAN_GOGH_APPROVED].sort(), [...TILE_TYPES].sort());
  assert.deepEqual(set.tiles.map(t => t.tile), VAN_GOGH_APPROVED);
  assert.deepEqual(set.remaining, []);
  assert.equal(entry.candidate, 'West B (cloud strokes)');
  assert.equal(entry.name, 'Shaa');
  assert.equal(entry.source, provenance.source);
});

test('West preserves the latest approved lighthouse export and pinned hashes', () => {
  assert.equal(provenance.originalFile, 'stormy_sunset_lighthouse_seascape.png');
  assert.equal(provenance.originalSha256, 'f6bada873d43c8e48b8a0683566e98b200da9cf973859f4e68962ea5db34b10a');
  const svg = read(provenance.runtime);
  assert.deepEqual(svg, read(provenance.source));
  assert.equal(hash(svg), 'c756af363fcc16806aaff468e5855947b7e72add6b0feb290eab051ecaa585cd');
  assert.equal(entry.svgSha256, hash(svg));
  assert.equal(provenance.svgSha256, hash(svg));
  const match = svg.toString().match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(match);
  const raster = Buffer.from(match[1], 'base64');
  assert.equal(hash(raster), 'd75dd1b0cb95a8d8759355506e62ff49d11fca885a77b7221aff26e2e7b595ba');
  assert.equal(entry.rasterSha256, hash(raster));
  assert.equal(provenance.rasterSha256, hash(raster));
  assert.equal(entry.rasterMimeType, 'image/avif');
});

test('West retains the entire exact 3:4 frame without stretching or bleed', () => {
  assert.deepEqual(provenance.originalSize, [1086, 1448]);
  assert.deepEqual(provenance.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.equal(provenance.originalSize[0] * 4, provenance.originalSize[1] * 3);
  assert.equal(provenance.stretch, 0);
  assert.equal(provenance.bleed, 0);
  assert.equal(entry.fit, undefined);
  const svg = read(provenance.runtime).toString();
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="0" y="0" width="300" height="400" preserveAspectRatio="xMidYMid meet"/);
  assert.doesNotMatch(svg, /<script|<foreignObject|<text|href="https?:/);
});

test('West preloads once, keeps hidden tiles private and leaves other face sets alone', () => {
  assert.equal(tileImage('3z', 'van-gogh'), 'tiles/van-gogh/approved/Shaa.svg');
  assert.equal(TILE_IMAGE_URLS.filter(u => u === tileImage('3z', 'van-gogh')).length, 1);
  assert.equal(tileImage('3z', 'van-gogh', true), 'tiles/Back.svg');
  assert.equal(tileImage('3z', 'classic'), 'tiles/Shaa.svg');
  assert.equal(tileImage('3z', 'matisse'), 'tiles/matisse/approved/Shaa.svg');
  assert.equal(TILE_IMAGE_URLS.filter(u => u.startsWith('tiles/van-gogh/approved/')).length, 34);
});

test('the complete gallery shows West and retains a fourteen-tile hand', () => {
  const page = read('web/public/tiles/van-gogh/preview.html').toString();
  const gallery = page.match(/<div class="gallery">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(gallery);
  assert.equal((gallery.match(/<figure>/g) ?? []).length, 34);
  assert.match(gallery, /approved\/Shaa\.svg/);
  assert.match(page, /Every tile type now has approved artwork\./);
  assert.doesNotMatch(page, /remaining 0 tiles/);
  const hand = page.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
});

test('--only=3z re-exports exactly and never changes other existing paintings', t => {
  const temp = mkdtempSync(path.join(tmpdir(), 'van-gogh-west-'));
  t.after(() => rmSync(temp, { recursive: true, force: true }));
  const copy = p => {
    mkdirSync(path.dirname(path.join(temp, p)), { recursive: true });
    cpSync(path.join(root, p), path.join(temp, p), { recursive: true });
  };
  for (const p of ['web/package.json', 'web/scripts/export-van-gogh-tiles.mjs', 'web/scripts/face-fit.mjs', 'web/src/lib/tiles.js', 'web/src/lib/van-gogh-faces.js', 'web/public/tiles/van-gogh']) copy(p);
  for (const item of set.sources) copy(item.source);
  const expected = new Map(set.tiles.flatMap(item => [item.svg, ...(item.png ? [item.png] : [])]).map(p => {
    const full = `web/public/tiles/van-gogh/${p}`;
    return [full, hash(read(full))];
  }));
  for (let i = 0; i < 2; i++) {
    execFileSync(process.execPath, [path.join(temp, 'web/scripts/export-van-gogh-tiles.mjs'), '--only=3z']);
    for (const [p, digest] of expected) assert.equal(hash(readFileSync(path.join(temp, p))), digest, p);
    assert.deepEqual(JSON.parse(readFileSync(path.join(temp, 'web/public/tiles/van-gogh/manifest.json'))), set);
    assert.deepEqual(readFileSync(path.join(temp, 'web/src/lib/van-gogh-faces.js')), read('web/src/lib/van-gogh-faces.js'));
  }
});
