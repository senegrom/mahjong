// One-off branch-only preparation; removed before the clean deployment commit.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, existsSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import vm from 'node:vm';
import { TILE_TYPES } from '../web/src/lib/tiles.js';

const root = process.cwd();
const branch = 'art/dali-four-disks-stairs-20261006';
assert.equal(process.env.GITHUB_REF, `refs/heads/${branch}`);
const read = file => readFileSync(path.join(root, file));
const text = file => read(file).toString();
const write = (file, value) => { mkdirSync(path.dirname(file), { recursive: true }); writeFileSync(file, value); };
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const manifestPath = 'web/public/tiles/dali/manifest.json';
const before = JSON.parse(read(manifestPath));
assert.equal(before.tiles.length, 20);
assert.equal(before.placeholders.length, 14);
assert.ok(!before.tiles.some(entry => entry.tile === '4p'));
const tracked = execFileSync('git', ['ls-files', '-z', 'docs/design', 'web/public/tiles']).toString().split('\0').filter(Boolean);
const originalArtwork = new Map(tracked.filter(file => /\.(png|webp|avif|svg|jpg|jpeg)$/i.test(file)).map(file => [file, hash(read(file))]));
const otherManifests = new Map(['matisse', 'van-gogh'].map(name => {
  const file = `web/public/tiles/${name}/manifest.json`;
  return [file, hash(read(file))];
}));
const base64 = Array.from({ length: 8 }, (_, i) => text(`.art-staging/part-${i}.txt`)).join('');
assert.match(base64, /^[A-Za-z0-9+/]+={0,2}$/);
const raster = Buffer.from(base64, 'base64');
assert.equal(raster.length, 66841);
assert.equal(hash(raster), '76e9743f0478a55bf14fba62c543ec013580eaddaeac6b23b4ecd130baf70a20');
assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
const ispe = raster.indexOf(Buffer.from('ispe'));
assert.ok(ispe >= 0);
assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [1086, 1448]);
const source = 'docs/design/dali/studies/four-disks-b-approved.svg';
const runtime = 'web/public/tiles/dali/approved/Pin4.svg';
const rasterSource = 'docs/design/dali/studies/four-disks-b-hires.avif';
const provenancePath = 'docs/design/dali/four-disks.json';
for (const file of [source, runtime, rasterSource, provenancePath]) assert.ok(!existsSync(file));
const metadata = {
  tile: '4p', name: 'Pin4', status: 'approved', direction: 'The Soft Staircase — B',
  selectedOption: 'B (middle), staircase', source, runtime,
  originalFilename: 'surreal_jade_medallions_amid_impossible_stairways.png',
  originalSha256: '09488a560c3d2299784740f85737f9430279a165f9bdf8c005835d2aa99e67d8',
  originalDimensions: [1086, 1448], originalCrop: { x: 0, y: 0, width: 1086, height: 1448 },
  originalStoredInRepository: false,
  originalLocation: 'Approved PNG conversation attachment; repository retains the full-dimension AVIF and self-contained SVG, not the original PNG bytes.',
  rasterDimensions: [1086, 1448], rasterMimeType: 'image/avif', rasterSource,
  rasterSha256: hash(raster),
  encoding: { encoder: 'Pillow 12.3.0 AVIF', quality: 50, speed: 0, lossless: false },
  processing: 'Approved middle staircase portrait encoded at its full 1086 x 1448 pixel dimensions; AVIF quality 50, speed 0; no resizing, crop, redraw, colour edit, added border or tile texture. The 300 x 400 SVG viewBox controls layout only, not embedded raster resolution.',
};
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">4 disks — Dali — The Soft Staircase — B</title><metadata>${JSON.stringify(metadata)}</metadata><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,${base64}"/></svg>\n`;
write(source, svg);
write(runtime, svg);
write(rasterSource, raster);
write(provenancePath, JSON.stringify({ ...metadata, svgSha256: hash(svg) }, null, 2) + '\n');
const replace = (value, from, to) => { assert.equal(value.split(from).length, 2, `Expected one occurrence: ${from}`); return value.replace(from, to); };
const exporterPath = 'web/scripts/export-dali-tiles.mjs';
let exporter = text(exporterPath);
const definition = "  ['Pin4', '4p', '4 disks', 'The Soft Staircase — B', 'four-disks-b-approved.svg', [0, 0, 1086, 1448]],\n";
exporter = replace(exporter, "  ['Pin5',", definition + "  ['Pin5',");
exporter = replace(exporter, '${manifest.tiles.length} approved faces. ', '${manifest.tiles.length} approved faces. Four disks uses B — The Soft Staircase: four jade medallions on an impossible ivory staircase, including one melting over a step. The complete approved painting is embedded at its full 1086 × 1448 resolution without adding a tile background. ');
exporter = replace(exporter, "'Sou9', 'Sou2', 'Sou6'", "'Sou9', 'Pin4', 'Sou6'");
write(exporterPath, exporter);
const manifest = structuredClone(before);
manifest.tiles.splice(manifest.tiles.findIndex(entry => entry.tile === '3p') + 1, 0, {
  name: 'Pin4', tile: '4p', label: '4 disks', direction: metadata.direction, status: 'approved',
  svg: 'approved/Pin4.svg', source, sourceSha256: hash(svg), svgSha256: hash(svg),
  crop: { x: 0, y: 0, width: 1086, height: 1448 },
});
// Execute the exporter's actual registry/manifest/preview writer without re-encoding older art.
const tail = exporter.slice(exporter.indexOf('const approved = manifest.tiles.map'));
assert.ok(tail.startsWith('const approved ='));
vm.runInNewContext(tail, { manifest, TILE_TYPES, writeFileSync, path, root, out: path.join(root, 'web/public/tiles/dali') });
const after = JSON.parse(read(manifestPath));
assert.equal(after.tiles.length, 21);
assert.equal(after.placeholders.length, 13);
for (const entry of before.tiles) assert.deepEqual(after.tiles.find(candidate => candidate.tile === entry.tile), entry);
assert.deepEqual(after.placeholders, before.placeholders.filter(entry => entry.tile !== '4p'));
const tests = ['dali-bamboo.test.js', 'dali-east-wind.test.js', 'dali-north-wind.test.js', 'dali-six-bamboo.test.js'];
for (const name of tests) {
  const file = `web/tests/${name}`;
  let content = text(file);
  assert.ok(content.includes('set.tiles.length, 20'));
  content = content.replaceAll('set.tiles.length, 20', 'set.tiles.length, 21')
    .replaceAll('set.placeholders.length, 14', 'set.placeholders.length, 13')
    .replaceAll('20 approved faces', '21 approved faces').replaceAll('remaining 14 tiles', 'remaining 13 tiles')
    .replaceAll('twenty approved identities and fourteen placeholders', 'twenty-one approved identities and thirteen placeholders');
  if (name === 'dali-bamboo.test.js') content = replace(content, "['1p', '3p', '5p',", "['1p', '3p', '4p', '5p',");
  write(file, content);
}
const faceTestPath = 'web/tests/tile-faces.test.js';
let faceTests = text(faceTestPath);
faceTests = replace(faceTests, 'Dali resolves twenty approved images', 'Dali resolves twenty-one approved images');
faceTests = replace(faceTests, "new Set(['1p', '3p', '5p',", "new Set(['1p', '3p', '4p', '5p',");
faceTests = replace(faceTests, "['3s', '4s'].includes(tile) ? /data:image\\/avif;base64,/", "['4p', '3s', '4s'].includes(tile) ? /data:image\\/avif;base64,/");
const pin3Test = "  assert.equal(tileImage('3p', 'dali'), 'tiles/dali/approved/Pin3.svg');";
faceTests = replace(faceTests, pin3Test, pin3Test + "\n  assert.equal(tileImage('4p', 'dali'), 'tiles/dali/approved/Pin4.svg');");
write(faceTestPath, faceTests);
const newTest = 'web/tests/dali-four-disks.test.js';
write(newTest, text('.art-staging/four-disks.test.txt').replace('__SVG_SHA__', hash(svg)));
const readmePath = 'docs/design/dali/README.md';
let readme = text(readmePath);
readme = replace(readme, 'Twenty faces are approved', 'Twenty-one faces are approved');
readme = replace(readme, 'The latest is six bamboo', 'The latest is four disks **The Soft Staircase — B**, the selected middle painting with four jade medallions on an impossible staircase and one softened disk. The complete 1086 × 1448 painting is retained at full pixel dimensions. The previously selected six bamboo');
readme = replace(readme, '| `Pin5.svg` |', '| `Pin4.svg` | 4 disks | The Soft Staircase — B: four jade medallions on an impossible ivory staircase, one melting over a step |\n| `Pin5.svg` |');
readme = replace(readme, 'The other 14 faces', 'The other 13 faces');
readme = readme.replaceAll('twenty approved faces plus', 'twenty-one approved faces plus');
const note = `Four disks uses [selected B — The Soft Staircase](studies/four-disks-b-approved.svg), from the approved middle image \`surreal_jade_medallions_amid_impossible_stairways.png\` (1086 × 1448; PNG SHA-256 \`${metadata.originalSha256}\`). The complete painting retains its four jade disks, melting disk, ivory stairs, clouds, seascapes and plain surround. The full-dimension AVIF is encoded at quality 50, speed 0, without resizing, cropping, repainting, a colour edit, or a baked tile frame. Compression is lossy; the original PNG remains in the conversation rather than being claimed as stored byte-for-byte. The repository retains the full-dimension AVIF and byte-identical self-contained source/runtime SVGs with the existing 300 × 400 logical canvas, rounded clipping and bleed. See [provenance](four-disks.json). Options A (portals) and C (detached shadows) are not deployed. All twenty previous Dalí faces and all other face sets remain unchanged.\n\n`;
readme = replace(readme, 'Three disks uses the exact', note + 'Three disks uses the exact');
write(readmePath, readme);
for (const [file, digest] of originalArtwork) assert.equal(hash(read(file)), digest, `Existing artwork changed: ${file}`);
for (const [file, digest] of otherManifests) assert.equal(hash(read(file)), digest, `Other set changed: ${file}`);
assert.deepEqual(read(source), read(runtime));
console.log(`Verified original/raster/source provenance and all ${originalArtwork.size} previous artwork files. New SVG SHA-256: ${hash(svg)}`);
execFileSync(process.execPath, ['--test', ...tests.map(file => `web/tests/${file}`), newTest], { stdio: 'inherit' });
for (const file of [...tests.map(name => `web/tests/${name}`), newTest, faceTestPath, exporterPath]) execFileSync(process.execPath, ['--check', file], { stdio: 'inherit' });
rmSync('.art-staging', { recursive: true });
rmSync('.github/workflows/prepare-dali-four-stairs.yml');
const expected = [readmePath, provenancePath, source, rasterSource, runtime, manifestPath,
  'web/public/tiles/dali/preview.html', exporterPath, 'web/src/lib/dali-faces.js',
  ...tests.map(file => `web/tests/${file}`), faceTestPath, newTest];
execFileSync('git', ['add', '-A']);
const changed = execFileSync('git', ['diff', '--cached', '--name-only', '1805bdc1aece832b490b9a4a1e007dfa1462d6eb']).toString().trim().split('\n');
assert.deepEqual(changed.sort(), expected.sort());
execFileSync('git', ['diff', '--cached', '--check'], { stdio: 'inherit' });
console.log('Prepared only the approved 4-disks artwork, provenance, registration, documentation and tests.');
