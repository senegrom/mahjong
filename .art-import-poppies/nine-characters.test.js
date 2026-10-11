import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';
import { faceImage, fitRecord, placeStudyImage } from '../scripts/face-fit.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = data => createHash('sha256').update(data).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const record = JSON.parse(read('docs/design/van-gogh/nine-characters-poppy-field.json'));
const entry = set.tiles.find(tile => tile.tile === '9m');

test('nine characters uses the approved revised poppy field with smaller Wan', () => {
  assert.ok(entry);
  assert.equal(entry.candidate, 'Poppies B (smaller Wan)');
  assert.equal(entry.name, 'Man9');
  assert.equal(entry.source, record.source);
  assert.equal(record.originalFilename, 'van_gogh_style_poppy_field_characters.png');
  assert.equal(record.originalSha256, 'ff3137d35a61f8569d3f6c7ff4c98dc2cc2809fe2b8d0b49be109d20a3ff4ac6');
  assert.deepEqual(record.originalDimensions, [1122, 1402]);
  assert.equal(hash(read(record.source)), '99f6155e1a7972f082fac869aca5e24559d65508be1ad175c2457b578372825a');
  assert.equal(hash(read(record.runtime)), 'ed88ef4bd1f0531ba649490137076547633e5fc99b8afef9bde9af2beda4b140');
  assert.equal(entry.svgSha256, record.svgSha256);
});

test('the source embeds the approved AVIF and the face changes placement only, with zero stretch', () => {
  const source = read(record.source), face = read(record.runtime);
  const embedded = source.toString().match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(hash(raster), '5ddc2af7ede1d69f7a51e71ab308afcca157fbaf9b5a2cdfd8b09e56cc539f03');
  assert.equal(entry.rasterMimeType, 'image/avif');
  assert.equal(entry.rasterSha256, hash(raster));
  assert.equal(raster.toString('ascii', 4, 8), 'ftyp');
  assert.equal(raster.toString('ascii', 8, 12), 'avif');
  const ispe = raster.indexOf('ispe');
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [300, 400]);
  assert.deepEqual(record.fit, { painting: [1122, 1402], stretch: 0, anchor: 0.5, bleedPixels: 0 });
  assert.deepEqual(entry.fit, fitRecord(record.fit, record.rasterDimensions));
  assert.deepEqual(face, placeStudyImage(source, faceImage(record.fit, record.rasterDimensions)));
  assert.match(face.toString(), /viewBox="0 0 300 400"/);
  assert.match(face.toString(), /rx="26"/);
  assert.doesNotMatch(face.toString(), /<script|<text|href="https?:/);
});

test('nine characters is registered and preloaded exactly once, preserving the other suits and hidden tiles', () => {
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '9m').length, 1);
  assert.deepEqual(VAN_GOGH_APPROVED, set.tiles.map(tile => tile.tile));
  assert.equal(new Set([...VAN_GOGH_APPROVED, ...set.remaining]).size, 34);
  for (let n = 2; n <= 9; n++) assert.ok(VAN_GOGH_APPROVED.includes(`${n}m`));
  for (let n = 1; n <= 9; n++) assert.ok(VAN_GOGH_APPROVED.includes(`${n}s`));
  assert.equal(tileImage('9m', 'van-gogh'), 'tiles/van-gogh/approved/Man9.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('9m', 'van-gogh')).length, 1);
  assert.equal(tileImage('9m', 'van-gogh', true), 'tiles/Back.svg');
  assert.equal(tileImage('9m', 'classic'), 'tiles/Man9.svg');
  assert.equal(set.sources.find(source => source.source === record.source).sha256, record.sourceSha256);
});

test('the revised poppies appear in the gallery and the fourteen-tile hand', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString();
  assert.match(preview, /Poppies B \(smaller Wan\) · Nine characters/);
  const rack = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(rack);
  assert.equal((rack.match(/<img /g) ?? []).length, 14);
  for (const name of ['Man9', 'Man8', 'Man7', 'Man6', 'Man5', 'Man2', 'Man3', 'Man4', 'Pin4', 'Pin9', 'Pin3', 'Sou9', 'Sou6', 'Sou8']) {
    assert.ok(rack.includes(`approved/${name}.svg`), name);
  }
});

test('--only=9m reproduces the fitted poppy face and leaves all other artwork untouched', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-poppies-'));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const put = (relative, data) => {
    const target = path.join(temporary, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, data);
  };
  put('package.json', '{"type":"module"}');
  for (const file of ['web/scripts/export-van-gogh-tiles.mjs', 'web/scripts/face-fit.mjs', 'web/src/lib/tiles.js']) put(file, read(file));
  for (const source of set.sources) put(source.source, source.source.endsWith('.svg') ? read(source.source) : Buffer.from(`source fixture ${source.id}`));
  const unchanged = [];
  for (const tile of set.tiles.filter(tile => tile.tile !== '9m')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`, data = Buffer.from(`unchanged fixture ${file}`);
      put(relative, data);
      unchanged.push([relative, data]);
    }
  }
  for (let run = 0; run < 2; run++) {
    execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), '--only=9m'], { stdio: 'pipe' });
    assert.deepEqual(readFileSync(path.join(temporary, record.runtime)), read(record.runtime));
    for (const [relative, data] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), data);
    const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
    assert.deepEqual(generated.tiles.find(tile => tile.tile === '9m'), entry);
    assert.deepEqual(generated.remaining, set.remaining);
  }
});
