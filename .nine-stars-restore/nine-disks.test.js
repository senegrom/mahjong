import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
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
const record = JSON.parse(read('docs/design/van-gogh/nine-stars-restored.json'));
const potters = JSON.parse(read('docs/design/van-gogh/nine-disks-potters-table-c.json'));
const entry = set.tiles.find(tile => tile.tile === '9p');

 test('Nine Stars I restores the exact original nine disks, not the rejected eight-star adaptation', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'I');
  assert.equal(entry.name, 'Pin9');
  assert.equal(record.title, 'Nine Stars');
  assert.equal(entry.source, record.source);
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 380, height: 471 });
  const original = read(record.source);
  const png = read(record.runtimePng);
  const svg = read(record.runtimeSvg);
  assert.deepEqual(png, original);
  assert.deepEqual(svg, read('docs/design/van-gogh/superseded/nine-stars-Pin9.svg'));
  assert.equal(hash(png), '61eb9c2563a8039eb755315b1f5ef670a472fcff2e1facaef4b0e05e1af0267d');
  assert.equal(hash(svg), '34f0fd31056df47c064383ca79b67d17aa9bd5a62a7aae1d3f9188701e98a89a');
  assert.equal(hash(png), entry.pngSha256);
  assert.equal(hash(png), record.pngSha256);
  assert.equal(hash(svg), record.svgSha256);
  assert.deepEqual([...png.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  assert.deepEqual([png.readUInt32BE(16), png.readUInt32BE(20)], [380, 471]);
  assert.deepEqual(record.originalStudyCrop, { x: 838, y: 100, width: 380, height: 471 });
  assert.equal(potters.supersedes.pngSha256, hash(png));
  assert.equal(potters.supersedes.svgSha256, hash(svg));
  const xml = svg.toString('utf8');
  assert.match(xml, /viewBox="0 0 300 400"/);
  assert.match(xml, /rx="26"/);
  assert.match(xml, /x="-3" y="-4" width="306" height="408"/);
  assert.ok(xml.includes(`data:image/png;base64,${png.toString('base64')}`));
});

test('The Potter’s Table C remains preserved with its original source and provenance', () => {
  const previous = set.superseded.find(tile => tile.tile === '9p');
  assert.equal(previous.candidate, 'Disks C');
  assert.equal(previous.activeCandidate, 'I');
  assert.equal(previous.source, potters.source);
  assert.equal(previous.archivedSvg, record.supersedes.archivedSvg);
  const source = read(potters.source);
  assert.deepEqual(read(previous.archivedSvg), source);
  assert.equal(hash(source), '7bfed3b794e8a09dc61f2020662d9c70572d883f05194267df07f535778591b1');
  assert.equal(hash(source), potters.svgSha256);
  assert.equal(hash(source), record.supersedes.svgSha256);
  assert.equal(potters.originalSha256, 'dcd5614654e482e650982f9fe7e98a9b22e834219fea0fec4effc3deb268e209');
  assert.equal(potters.fullResolutionCropSha256, 'f81be31b80ba4d9eb0f80e3bf3efb565a824dce9fc2ad3741ca9a21df4276c07');
  assert.deepEqual(potters.originalCrop, { x: 551, y: 56, width: 729, height: 1093 });
  const embedded = source.toString('utf8').match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  assert.equal(hash(Buffer.from(embedded[1], 'base64')), '847c979c08614d41bbcaab6c802e100050b4d684de46c6474eff7aa348d2d670');
});

test('restored nine disks is registered exactly once, preloaded and shown in the preview', () => {
  assert.equal(set.tiles.filter(tile => tile.tile === '9p').length, 1);
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '9p').length, 1);
  assert.equal(tileImage('9p', 'van-gogh'), 'tiles/van-gogh/approved/Pin9.svg');
  assert.ok(TILE_IMAGE_URLS.includes(tileImage('9p', 'van-gogh')));
  assert.ok(!set.remaining.includes('9p'));
  assert.deepEqual(set.tiles.map(tile => tile.tile), [...VAN_GOGH_APPROVED]);
  assert.equal(set.remaining.length, 34 - VAN_GOGH_APPROVED.length);
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, record.pngSha256);
  const preview = read('web/public/tiles/van-gogh/preview.html').toString('utf8');
  assert.match(preview, /Nine Stars I for 9 disks/);
  assert.match(preview, /approved\/Pin9\.svg/);
});

test('--only=9p restores both original exports and preserves every other playable image', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-nine-stars-'));
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
    put(source.source, source.source.endsWith('.svg') || source.source === record.source
      ? read(source.source) : Buffer.from(`source fixture: ${source.id}`));
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
  assert.deepEqual(readFileSync(path.join(temporary, record.runtimePng)), read(record.runtimePng));
  assert.deepEqual(readFileSync(path.join(temporary, record.runtimeSvg)), read(record.runtimeSvg));
  const manifestPath = path.join(temporary, 'web/public/tiles/van-gogh/manifest.json');
  const generated = JSON.parse(readFileSync(manifestPath));
  assert.deepEqual(generated.tiles.find(tile => tile.tile === '9p'), entry);
  assert.deepEqual(generated.remaining, set.remaining);
  const first = readFileSync(manifestPath);
  run();
  assert.deepEqual(readFileSync(manifestPath), first);
  assert.deepEqual(readFileSync(path.join(temporary, record.runtimeSvg)), read(record.runtimeSvg));
});
