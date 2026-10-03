import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const record = JSON.parse(read('docs/design/van-gogh/nine-disks-potters-table-c.json'));
const entry = set.tiles.find(tile => tile.tile === '9p');

test('Potter’s Table C is the selected nine-disks painting, not an infographic or another candidate', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Disks C');
  assert.equal(entry.name, 'Pin9');
  assert.equal(record.title, 'The Potter’s Table');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/17-nine-disks-potters-table-c-approved.svg');
  assert.equal(entry.source, record.source);
  const source = read(entry.source), runtime = read(record.runtime);
  assert.deepEqual(runtime, source);
  assert.equal(hash(source), '7bfed3b794e8a09dc61f2020662d9c70572d883f05194267df07f535778591b1');
  assert.equal(hash(runtime), entry.svgSha256);
  assert.equal(hash(source), record.svgSha256);
  assert.equal(record.originalSha256, 'dcd5614654e482e650982f9fe7e98a9b22e834219fea0fec4effc3deb268e209');
  assert.equal(record.fullResolutionCropSha256, 'f81be31b80ba4d9eb0f80e3bf3efb565a824dce9fc2ad3741ca9a21df4276c07');
  assert.deepEqual(record.originalDimensions, [1295, 1214]);
  assert.deepEqual(record.originalCrop, { x: 551, y: 56, width: 729, height: 1093 });
  const svg = source.toString('utf8');
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  const embedded = svg.match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(hash(raster), '847c979c08614d41bbcaab6c802e100050b4d684de46c6474eff7aa348d2d670');
  assert.equal(hash(raster), entry.rasterSha256);
  assert.equal(hash(raster), record.rasterSha256);
  assert.equal(raster.length, 77626);
  assert.equal(entry.rasterMimeType, 'image/webp');
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 16), 'WEBPVP8 ');
  assert.equal(raster.readUInt16LE(26) & 0x3fff, 300);
  assert.equal(raster.readUInt16LE(28) & 0x3fff, 400);
});

test('Nine Stars remains archived while the replacement is registered exactly once', () => {
  const previous = set.superseded.find(tile => tile.tile === '9p');
  assert.equal(previous.candidate, 'I');
  assert.equal(previous.activeCandidate, 'Disks C');
  assert.equal(previous.source, record.supersedes.source);
  assert.equal(previous.archivedPng, record.supersedes.archivedPng);
  assert.equal(previous.archivedSvg, record.supersedes.archivedSvg);
  const png = read(previous.archivedPng), svg = read(previous.archivedSvg);
  assert.equal(hash(png), '61eb9c2563a8039eb755315b1f5ef670a472fcff2e1facaef4b0e05e1af0267d');
  assert.equal(hash(png), record.supersedes.pngSha256);
  assert.equal(hash(svg), record.supersedes.svgSha256);
  assert.ok(svg.toString('utf8').includes(`data:image/png;base64,${png.toString('base64')}`));
  assert.equal(existsSync(path.join(root, 'web/public/tiles/van-gogh/approved/Pin9.png')), false);
  assert.equal(set.tiles.filter(tile => tile.tile === '9p').length, 1);
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '9p').length, 1);
  assert.equal(tileImage('9p', 'van-gogh'), 'tiles/van-gogh/approved/Pin9.svg');
  assert.ok(TILE_IMAGE_URLS.includes(tileImage('9p', 'van-gogh')));
  assert.ok(!set.remaining.includes('9p'));
  assert.equal(set.tiles.length, VAN_GOGH_APPROVED.length);
  assert.equal(set.remaining.length, 34 - VAN_GOGH_APPROVED.length);
  assert.match(read('web/public/tiles/van-gogh/preview.html').toString('utf8'), /The Potter’s Table C/);
});

test('--only=9p copies C exactly and preserves every other playable image', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-potters-table-'));
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
  for (const tile of set.tiles.filter(tile => tile.tile !== '9p')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = () => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), '--only=9p'], { encoding: 'utf8', stdio: 'pipe' });
  assert.equal(run().trim(), `Exported ${set.tiles.length} approved Van Gogh faces; ${set.remaining.length} identities use Classic artwork.`);
  for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.source));
  const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '9p'), entry);
  assert.deepEqual(generated.remaining, set.remaining);
  const first = readFileSync(path.join(temporary, record.runtime));
  run();
  assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), first);
});
