import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
const read = file => readFileSync(file, 'utf8');
const put = (file, content) => writeFileSync(file, content);
const json = file => JSON.parse(read(file));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const once = (s, a, b) => { assert.equal(s.split(a).length, 2, `Expected one anchor: ${a}`); return s.replace(a, b); };
const tracked = execFileSync('git', ['ls-files', '-z', 'docs/design', 'web/public/tiles']).toString().split('\0').filter(Boolean);
const artwork = tracked.filter(file => /\.(svg|png|avif|webp)$/i.test(file));
const saved = new Map(artwork.map(file => [file, hash(readFileSync(file))]));
const manifestPath = 'web/public/tiles/dali/manifest.json';
const previous = json(manifestPath);
assert.equal(previous.tiles.length, 23);
assert.equal(previous.placeholders.length, 11);
assert.equal(previous.tiles.some(tile => tile.tile === '5z'), false);
assert.equal(previous.fallback, 'text');
const metadata = json('.dali-white-a-prep/metadata.json');
const raster = Buffer.concat(Array.from({ length: 9 }, (_, i) => readFileSync(`.dali-white-a-prep/part${String(i).padStart(2, '0')}.bin`)));
assert.equal(raster.length, 105717);
assert.equal(hash(raster), metadata.rasterSha256);
assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
const ispe = raster.indexOf(Buffer.from('ispe'));
assert.ok(ispe >= 0);
assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [1086, 1448]);
const provenancePath = 'docs/design/dali/white-dragon.json';
const newTest = 'web/tests/dali-white-dragon.test.js';
for (const file of [metadata.source, metadata.runtime, metadata.rasterSource, provenancePath, newTest]) assert.equal(existsSync(file), false, file);
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">White dragon — Dali — The Fold That Woke Up — A</title><metadata>${JSON.stringify(metadata)}</metadata><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,${raster.toString('base64')}"/></svg>\n`;
assert.equal(hash(svg), '019442ea211bf281e98271a692e115dd529ec68a288bfb9303fe7a43ee4a74cb');
put(metadata.rasterSource, raster);
put(metadata.source, svg);
put(provenancePath, JSON.stringify({ ...metadata, svgSha256: hash(svg) }, null, 2) + '\n');
const exporterPath = 'web/scripts/export-dali-tiles.mjs';
let exporter = read(exporterPath);
const chun = "  ['Chun', '7z', 'Red dragon', 'Molten Ruby', board, [847, 656, 373, 513]],";
const haku = "  ['Haku', '5z', 'White dragon', 'The Fold That Woke Up — A', 'white-dragon-a-approved.svg', [0, 0, 1086, 1448]],";
exporter = once(exporter, chun, haku + '\n' + chun);
exporter = once(exporter, "'Pin6', 'Sou9', 'Pin4'", "'Pin6', 'Haku', 'Pin4'");
exporter = once(exporter, '${manifest.tiles.length} approved faces. ', '${manifest.tiles.length} approved faces. White dragon uses A — The Fold That Woke Up: the selected ivory silk dragon above mirror water, with four tiny figures holding a flat sheet. The complete 3:4 portrait is embedded at its original 1086 × 1448 pixel dimensions, without repainting, recolouring or adding a tile background. ');
put(exporterPath, exporter);
const earlier = readdirSync('web/tests').filter(name => /^dali-.*\.test\.js$/.test(name));
for (const name of earlier) {
  const file = `web/tests/${name}`;
  const s = read(file).replaceAll('set.tiles.length, 23', 'set.tiles.length, 24')
    .replaceAll('set.placeholders.length, 11', 'set.placeholders.length, 10')
    .replaceAll('23 approved faces', '24 approved faces')
    .replaceAll('remaining 11 tiles', 'remaining 10 tiles')
    .replaceAll('twenty-three', 'twenty-four')
    .replaceAll('eleven placeholders', 'ten placeholders')
    .replaceAll('eleven named fallbacks', 'ten named fallbacks')
    .replaceAll("'4z', '7z'", "'4z', '5z', '7z'");
  put(file, s);
}
const facesPath = 'web/tests/tile-faces.test.js';
let faces = read(facesPath);
faces = once(faces, "test('Dali resolves twenty-three approved images", "test('Dali resolves twenty-four approved images");
// Restrict edits to the Dali block; other artists have independent approved lists.
const begin = faces.indexOf("test('Dali resolves");
const end = faces.indexOf("test('hidden tiles", begin);
assert.ok(begin >= 0 && end > begin);
let block = faces.slice(begin, end);
block = once(block, "'4z', '7z'", "'4z', '5z', '7z'");
block = once(block, "['2p', '4p', '6p', '3s', '4s'].includes(tile)", "['2p', '4p', '6p', '3s', '4s', '5z'].includes(tile)");
const pei = "  assert.equal(tileImage('4z', 'dali'), 'tiles/dali/approved/Pei.svg');";
block = once(block, pei, pei + "\n  assert.equal(tileImage('5z', 'dali'), 'tiles/dali/approved/Haku.svg');");
put(facesPath, faces.slice(0, begin) + block + faces.slice(end));
put(newTest, read('.dali-white-a-prep/test.js'));
const notesPath = 'docs/design/dali/README.md';
let notes = read(notesPath);
notes = once(notes, 'Twenty-three faces are approved', 'Twenty-four faces are approved');
notes = once(notes, 'The latest is six disks', 'The latest is white dragon **The Fold That Woke Up — A**, the selected first painting of an ivory silk dragon over mirror water, with four tiny figures holding a flat sheet below. The complete 1086 × 1448 (3:4) painting is preserved. The previously selected six disks');
notes = once(notes, '| `Chun.svg` |', '| `Haku.svg` | White dragon | The Fold That Woke Up — A: a sinuous ivory silk dragon above mirror water and four tiny figures holding a flat sheet |\n| `Chun.svg` |');
notes = once(notes, 'The other 11 tiles', 'The other 10 tiles');
const paragraph = `White dragon uses [selected A — The Fold That Woke Up](studies/white-dragon-a-approved.svg), from the first painting \`silken_dragon_over_mirror_waters.png\` (1086 × 1448; PNG SHA-256 \`${metadata.originalSha256}\`). Its complete silk dragon, seascape, figures and flat sheet are retained. The full-dimension AVIF is encoded at quality 50, speed 0, without resizing, cropping, redrawing, recolouring or adding a tile background. Compression is lossy; the original PNG remains in the conversation, while the repository retains the AVIF and byte-identical self-contained source/runtime SVGs. Existing logical canvas, rounded clipping and bleed are unchanged. See [provenance](white-dragon.json). Options B (shadow dragon) and C (skeleton) are not deployed; all previously approved artwork, including six disks, remains unchanged. Seven disks is outside this change.\n\n`;
notes = once(notes, '## Presentation\n\n', '## Presentation\n\n' + paragraph);
put(notesPath, notes);
const runExport = () => execFileSync(process.execPath, [exporterPath], { stdio: 'inherit' });
runExport();
const current = json(manifestPath);
assert.equal(current.tiles.length, previous.tiles.length + 1);
assert.equal(current.placeholders.length, previous.placeholders.length - 1);
for (const entry of previous.tiles) assert.deepEqual(current.tiles.find(item => item.tile === entry.tile), entry, entry.tile);
assert.deepEqual(current.placeholders, previous.placeholders.filter(item => item.tile !== '5z'));
for (const [file, expected] of saved) assert.equal(hash(readFileSync(file)), expected, `Prior artwork changed: ${file}`);
assert.deepEqual(readFileSync(metadata.runtime), readFileSync(metadata.source));
const generated = [manifestPath, 'web/src/lib/dali-faces.js', 'web/public/tiles/dali/preview.html'];
const before = new Map(generated.map(file => [file, hash(readFileSync(file))]));
runExport();
for (const [file, expected] of before) assert.equal(hash(readFileSync(file)), expected, `Non-deterministic export: ${file}`);
for (const [file, expected] of saved) assert.equal(hash(readFileSync(file)), expected, `Prior artwork changed: ${file}`);
const allowed = new Set([notesPath, provenancePath, newTest, metadata.source, metadata.runtime, metadata.rasterSource, exporterPath, facesPath, ...generated, ...earlier.map(name => `web/tests/${name}`)]);
const changed = execFileSync('git', ['diff', '--name-only', '-z']).toString().split('\0').filter(Boolean);
for (const file of changed) assert.ok(allowed.has(file), `Out-of-scope change: ${file}`);
console.log(`Verified ${saved.size} existing artwork files and ${previous.tiles.length} prior Dali manifest entries unchanged. Exact full-dimension AVIF and identical source/runtime SVG hashes; repeated exports identical.`);
