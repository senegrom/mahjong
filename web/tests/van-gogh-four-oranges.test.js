import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { TILE_FACE_OPTIONS, TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = new URL('../../', import.meta.url);
const read = p => readFileSync(new URL(p, root));
const hash = data => createHash('sha256').update(data).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const provenance = JSON.parse(read('docs/design/van-gogh/four-disks-oranges.json'));
const source = 'docs/design/van-gogh/studies/20-four-disks-oranges-a-approved.svg';
const runtime = 'web/public/tiles/van-gogh/approved/Pin4.svg';
const entry = set.tiles.find(t => t.tile === '4p');

test('Four Oranges A is approved exactly once as four disks, without replacing another face', () => {
  assert.equal(VAN_GOGH_APPROVED.filter(t => t === '4p').length, 1);
  assert.equal(set.tiles.filter(t => t.tile === '4p').length, 1);
  assert.equal(entry.candidate, 'Oranges A');
  assert.equal(entry.name, 'Pin4');
  assert.equal(entry.label, 'Four disks');
  assert.equal(entry.source, source);
  assert.equal(entry.status, 'approved');
  assert.equal(set.remaining.includes('4p'), false);
  assert.deepEqual(set.tiles.map(t => t.tile), VAN_GOGH_APPROVED);
  assert.deepEqual([...VAN_GOGH_APPROVED, ...set.remaining].sort(), [...TILE_TYPES].sort());
});

test('orange source and runtime contain the identical checksum-pinned approved crop export', () => {
  const svg = read(source);
  assert.deepEqual(read(runtime), svg);
  assert.equal(hash(svg), '52d7428647df0da97da13594c0913a4849e723ed9ff5d921b5fe2b3b6944a4d1');
  assert.equal(entry.svgSha256, hash(svg));
  assert.equal(provenance.sourceSha256, hash(svg));
  const embedded = svg.toString().match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(raster.length, 66718);
  assert.equal(hash(raster), 'e2dcad697285d2baa13e0b4f227b1e2f7016279f15252cd4aa5f53fe6922a9ba');
  assert.equal(entry.rasterSha256, hash(raster));
  assert.equal(provenance.rasterSha256, hash(raster));
  assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
  assert.equal(raster.toString('ascii', 8, 12), 'WEBP');
});

test('orange SVG is self-contained and uses the established tile presentation', () => {
  const svg = read(runtime).toString();
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.match(svg, /x="-3" y="-4" width="306" height="408"/);
  assert.match(svg, /preserveAspectRatio="none"/);
  assert.doesNotMatch(svg, /<script|<foreignObject|(?:href|src)="https?:/i);
  assert.equal((svg.match(/<image\b/g) || []).length, 1);
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 });
});

test('orange provenance identifies left-panel A and distinguishes original resolution from game export', () => {
  assert.equal(provenance.tile, '4p');
  assert.equal(provenance.candidate, 'Oranges A');
  assert.equal(provenance.source, source);
  assert.equal(provenance.runtime, runtime);
  assert.deepEqual(provenance.originalBoardDimensions, { width: 1448, height: 1086 });
  assert.deepEqual(provenance.originalCrop, { x: 25, y: 138, width: 442, height: 796 });
  assert.equal(provenance.originalBoardSha256, '70b6502946fd751f9e96467dad7da602734c850a63801864f03304ec220823ae');
  assert.equal(provenance.fullResolutionCropSha256, 'f6a436efce67eb717296820ce086a2ac5a6ee1fbf7f6580c9d84bc509f623cc5');
  assert.deepEqual(provenance.rasterDimensions, { width: 300, height: 400 });
  assert.match(provenance.approval, /left panel/);
  assert.match(provenance.processing, /not a full-resolution original/);
});

test('four oranges preloads once and never reveals a hidden tile', () => {
  const url = 'tiles/van-gogh/approved/Pin4.svg';
  assert.equal(tileImage('4p', 'van-gogh'), url);
  assert.equal(TILE_IMAGE_URLS.filter(p => p === url).length, 1);
  assert.equal(tileImage('4p', 'classic'), 'tiles/Pin4.svg');
  for (const { value } of TILE_FACE_OPTIONS) {
    assert.equal(tileImage('4p', value, true), 'tiles/Back.svg');
  }
  for (const t of ['3p', '9p', '5m', '6m', ...Array.from({ length: 9 }, (_, i) => `${i + 1}s`)]) {
    assert.ok(VAN_GOGH_APPROVED.includes(t));
    assert.match(tileImage(t, 'van-gogh'), /tiles\/van-gogh\/approved\//);
  }
});

test('orange exporter and preview include the new face without extending the fourteen-tile hand', () => {
  const exporter = read('web/scripts/export-van-gogh-tiles.mjs').toString();
  assert.ok(exporter.includes(`fourOranges: '${source}'`));
  assert.ok(exporter.includes("['Oranges A', '4p', 'Four disks', 'fourOranges', [0, 0, 300, 400]]"));
  const preview = read('web/public/tiles/van-gogh/preview.html').toString();
  assert.match(preview, /Oranges A/);
  assert.match(preview, /src="approved\/Pin4.svg"/);
  const rack = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/);
  assert.ok(rack);
  assert.equal((rack[1].match(/<img\b/g) || []).length, 14);
  for (const name of ['Pin4', 'Pin3', 'Pin9', 'Sou9', 'Sou6', 'Sou8', 'Man5', 'Man6']) {
    assert.ok(rack[1].includes(`approved/${name}.svg`));
  }
});
