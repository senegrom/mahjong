import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';

const read = file => readFileSync(file, 'utf8');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const run = (exe, args) => execFileSync(exe, args, { stdio: 'inherit' });
const put = (file, value) => writeFileSync(file, value);
const json = file => JSON.parse(read(file));
const replaceOne = (s, a, b) => {
  assert.equal(s.split(a).length, 2, `Expected one anchor: ${a}`);
  return s.replace(a, b);
};
const tracked = execFileSync('git', ['ls-files', '-z', 'docs/design', 'web/public/tiles']).toString().split('\0').filter(Boolean);
const artwork = tracked.filter(file => /\.(svg|png|avif|webp)$/i.test(file));
const previousHashes = new Map(artwork.map(file => [file, hash(readFileSync(file))]));
const manifestPath = 'web/public/tiles/dali/manifest.json';
const previous = json(manifestPath);
assert.equal(previous.tiles.length, 22);
assert.equal(previous.placeholders.length, 12);
assert.equal(previous.tiles.some(tile => tile.tile === '6p'), false);
assert.equal(previous.fallback, 'text');
const metadata = {"tile":"6p","name":"Pin6","status":"approved","direction":"The Spiral Staircase — C","selectedOption":"C (third), spiral staircase; approved 3:4 recreation","source":"docs/design/dali/studies/six-disks-c-spiral-approved.svg","runtime":"web/public/tiles/dali/approved/Pin6.svg","originalFilename":"spiral_tower_of_celestial_orbs.png","originalSha256":"bbbdab7d9d3ed1435917f2e1901a9a018ea84bb69220fb8be2090f0cbf258dd3","originalDimensions":[1086,1448],"originalCrop":{"x":0,"y":0,"width":1086,"height":1448},"originalStoredInRepository":false,"originalLocation":"Approved PNG conversation attachment; repository retains the full-dimension AVIF and self-contained SVG, not the original PNG bytes.","rasterDimensions":[1086,1448],"rasterMimeType":"image/avif","rasterSource":"docs/design/dali/studies/six-disks-c-spiral-hires.avif","rasterSha256":"d0aa2a0d99ec502db5995bb52e945d7facaa8cf2a8ebeba0918a49af699b4d7f","encoding":{"encoder":"Pillow 12.3.0 AVIF","quality":50,"speed":6,"lossless":false},"processing":"Approved 3:4 spiral staircase portrait encoded at its full 1086 x 1448 pixel dimensions; AVIF quality 50, speed 6; no resizing, crop, redraw, colour edit, added border or tile texture. The 300 x 400 SVG viewBox controls layout only, not embedded raster resolution."};
const raster = Buffer.concat(Array.from({ length: 10 }, (_, i) => readFileSync(`.dali-six-spiral-prep/part${String(i).padStart(2, '0')}.bin`)));
assert.equal(raster.length, 116915);
assert.equal(hash(raster), metadata.rasterSha256);
assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
const ispe = raster.indexOf(Buffer.from('ispe'));
assert.ok(ispe >= 0);
assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [1086, 1448]);
for (const file of [metadata.source, metadata.runtime, metadata.rasterSource, 'docs/design/dali/six-disks.json', 'web/tests/dali-six-disks.test.js']) assert.equal(existsSync(file), false, file);
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">6 disks — Dali — The Spiral Staircase — C</title><metadata>${JSON.stringify(metadata)}</metadata><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,${raster.toString('base64')}"/></svg>\n`;
assert.equal(hash(svg), 'bfedb68eb1948f9fb2b0f5f193cebc27bf2379933f13448558c6021cc1b0dbdc');
put(metadata.rasterSource, raster);
put(metadata.source, svg);
put('docs/design/dali/six-disks.json', JSON.stringify({ ...metadata, svgSha256: hash(svg) }, null, 2) + '\n');

const exporterPath = 'web/scripts/export-dali-tiles.mjs';
let exporter = read(exporterPath);
const pin5 = "  ['Pin5', '5p', '5 disks', 'Levitation', board, [441, 110, 374, 492]],";
const pin6 = "  ['Pin6', '6p', '6 disks', 'The Spiral Staircase — C', 'six-disks-c-spiral-approved.svg', [0, 0, 1086, 1448]],";
exporter = replaceOne(exporter, pin5, pin5 + '\n' + pin6);
exporter = replaceOne(exporter, "const hand = ['Sou3', 'Sou4', 'Ton', 'Pei', 'Sou8',", "const hand = ['Sou3', 'Sou4', 'Ton', 'Pei', 'Pin6',");
exporter = replaceOne(exporter, '${manifest.tiles.length} approved faces. ', '${manifest.tiles.length} approved faces. Six disks uses C — The Spiral Staircase: the approved 3:4 portrait with six blue-and-green disks set into an ivory spiral tower over a floating island. The full 1086 × 1448 painting is retained without resizing or cropping. ');
exporter = exporter.replace('Ranks 1–5 of disks are complete.', 'Ranks 1–6 of disks are complete.');
put(exporterPath, exporter);
run(process.execPath, [exporterPath]);

// Only coverage assertions change in earlier tile tests; all old artwork pins remain.
const earlierTests = readdirSync('web/tests').filter(name => /^dali-.*\.test\.js$/.test(name));
for (const name of earlierTests) {
  const file = `web/tests/${name}`;
  let s = read(file);
  s = s.replaceAll('set.tiles.length, 22', 'set.tiles.length, 23')
    .replaceAll('set.placeholders.length, 12', 'set.placeholders.length, 11')
    .replaceAll('22 approved faces', '23 approved faces')
    .replaceAll('remaining 12 tiles', 'remaining 11 tiles')
    .replaceAll('twenty-two', 'twenty-three')
    .replaceAll('twelve placeholders', 'eleven placeholders')
    .replaceAll('twelve named fallbacks', 'eleven named fallbacks')
    .replaceAll("'1p', '2p', '3p', '4p', '5p', '1s'", "'1p', '2p', '3p', '4p', '5p', '6p', '1s'");
  put(file, s);
}
const facesPath = 'web/tests/tile-faces.test.js';
let faces = read(facesPath);
faces = replaceOne(faces, "test('Dali resolves twenty-two approved images", "test('Dali resolves twenty-three approved images");
faces = replaceOne(faces, "const approved = new Set(['1p', '2p', '3p', '4p', '5p', '1s'", "const approved = new Set(['1p', '2p', '3p', '4p', '5p', '6p', '1s'");
faces = replaceOne(faces, "['2p', '4p', '3s', '4s'].includes(tile)", "['2p', '4p', '6p', '3s', '4s'].includes(tile)");
faces = replaceOne(faces, "  assert.equal(tileImage('4p', 'dali'), 'tiles/dali/approved/Pin4.svg');", "  assert.equal(tileImage('4p', 'dali'), 'tiles/dali/approved/Pin4.svg');\n  assert.equal(tileImage('6p', 'dali'), 'tiles/dali/approved/Pin6.svg');");
put(facesPath, faces);
put('web/tests/dali-six-disks.test.js', read('.dali-six-spiral-prep/test.js'));

const notesPath = 'docs/design/dali/README.md';
let notes = read(notesPath);
notes = replaceOne(notes, 'Twenty-two faces are approved', 'Twenty-three faces are approved');
notes = replaceOne(notes, 'The latest is two disks', 'The latest is six disks **The Spiral Staircase — C**, the selected third concept recreated as a standalone 3:4 portrait: six blue-and-green disks inset into an ivory spiral tower on a floating island. This completes disks 1–6. The previously selected two disks');
notes = replaceOne(notes, '| `Sou1.svg` |', '| `Pin6.svg` | 6 disks | The Spiral Staircase — C: six blue-and-green disks inset into an ivory spiral tower on a floating island |\n| `Sou1.svg` |');
notes = replaceOne(notes, 'The other 12 tiles', 'The other 11 tiles');
notes = notes.replaceAll('twenty-two approved faces', 'twenty-three approved faces');
const paragraph = `Six disks uses [the selected spiral staircase C](studies/six-disks-c-spiral-approved.svg), recreated at the correct 3:4 aspect ratio in \`spiral_tower_of_celestial_orbs.png\` (1086 × 1448; original PNG SHA-256 \`${metadata.originalSha256}\`). This is the approved six-disk tower, not any seashell alternative. The complete painting is encoded as AVIF quality 50, speed 6 at its original pixel dimensions, without resizing, cropping, repainting, recolouring or adding a tile background. The full-dimension AVIF is stored separately and embedded in byte-identical source/runtime SVGs. Compression is lossy; the repository does not claim to preserve the original PNG bytes. Existing 300 × 400 layout, clipping and bleed are unchanged. See [provenance](six-disks.json). All previously approved artwork and fitting remain unchanged.\n\n`;
notes = replaceOne(notes, '## Presentation\n\n', '## Presentation\n\n' + paragraph);
put(notesPath, notes);

const current = json(manifestPath);
assert.equal(current.tiles.length, 23);
assert.equal(current.placeholders.length, 11);
for (const entry of previous.tiles) assert.deepEqual(current.tiles.find(item => item.tile === entry.tile), entry, entry.tile);
assert.deepEqual(current.placeholders, previous.placeholders.filter(item => item.tile !== '6p'));
for (const [file, expected] of previousHashes) assert.equal(hash(readFileSync(file)), expected, `Existing artwork changed: ${file}`);
assert.deepEqual(readFileSync(metadata.runtime), readFileSync(metadata.source));
const generated = [manifestPath, 'web/src/lib/dali-faces.js', 'web/public/tiles/dali/preview.html'];
const beforeRerun = new Map(generated.map(file => [file, hash(readFileSync(file))]));
run(process.execPath, [exporterPath]);
for (const [file, expected] of beforeRerun) assert.equal(hash(readFileSync(file)), expected, `Non-deterministic export: ${file}`);
for (const [file, expected] of previousHashes) assert.equal(hash(readFileSync(file)), expected, `Existing artwork changed on repeated export: ${file}`);
const allowed = new Set([notesPath, metadata.source, metadata.runtime, metadata.rasterSource, 'docs/design/dali/six-disks.json', exporterPath, facesPath, 'web/tests/dali-six-disks.test.js', ...generated, ...earlierTests.map(name => `web/tests/${name}`)]);
const changed = execFileSync('git', ['diff', '--name-only', '-z']).toString().split('\0').filter(Boolean);
for (const file of changed) assert.ok(allowed.has(file), `Out-of-scope modification: ${file}`);
console.log(`Verified ${previousHashes.size} existing artwork files and ${previous.tiles.length} manifest entries unchanged; source/runtime and full-resolution AVIF hashes exact; repeated exports identical.`);
