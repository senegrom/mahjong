import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';

const read = file => readFileSync(file, 'utf8');
const put = (file, data) => writeFileSync(file, data);
const json = file => JSON.parse(read(file));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const once = (s, a, b) => { assert.equal(s.split(a).length, 2, `Expected one anchor: ${a}`); return s.replace(a, b); };
const prepared = 'eb36fce4be34dc1d2408fd9e2b9780c9705c8ae0';
const manifestPath = 'web/public/tiles/dali/manifest.json';
const previous = json(manifestPath);
assert.equal(previous.tiles.length, 24);
assert.equal(previous.placeholders.length, 10);
assert.ok(previous.tiles.some(tile => tile.tile === '7p'));
assert.equal(previous.tiles.some(tile => tile.tile === '5z'), false);
const tracked = execFileSync('git', ['ls-files', '-z', 'docs/design', 'web/public/tiles']).toString().split('\0').filter(Boolean);
const saved = new Map(tracked.filter(file => /\.(svg|png|avif|webp)$/i.test(file)).map(file => [file, hash(readFileSync(file))]));
const source = 'docs/design/dali/studies/white-dragon-a-approved.svg';
const runtime = 'web/public/tiles/dali/approved/Haku.svg';
const rasterPath = 'docs/design/dali/studies/white-dragon-a-hires.avif';
const provenancePath = 'docs/design/dali/white-dragon.json';
const newTest = 'web/tests/dali-white-dragon.test.js';
const recovered = [source, runtime, rasterPath, provenancePath, newTest];
for (const file of recovered) {
  assert.equal(existsSync(file), false, file);
  put(file, execFileSync('git', ['show', `${prepared}:${file}`], { maxBuffer: 8 * 1024 * 1024 }));
}
const metadata = json(provenancePath);
assert.equal(metadata.originalSha256, 'f1c12e186b3def7e5277c906c5cfb39f50cc9349bfcf9b66fa14f2d24578f193');
assert.equal(metadata.originalFilename, 'silken_dragon_over_mirror_waters.png');
assert.equal(hash(readFileSync(source)), '019442ea211bf281e98271a692e115dd529ec68a288bfb9303fe7a43ee4a74cb');
assert.equal(hash(readFileSync(rasterPath)), '7576d2f0519c65659d0cef8ffc39545da55471341e274592bbb31de97449ac15');
assert.deepEqual(readFileSync(source), readFileSync(runtime));
assert.deepEqual(metadata.originalDimensions, [1086, 1448]);
assert.deepEqual(metadata.rasterDimensions, [1086, 1448]);

const exporterPath = 'web/scripts/export-dali-tiles.mjs';
let exporter = read(exporterPath);
const chun = "  ['Chun', '7z', 'Red dragon', 'Molten Ruby', board, [847, 656, 373, 513]],";
const haku = "  ['Haku', '5z', 'White dragon', 'The Fold That Woke Up — A', 'white-dragon-a-approved.svg', [0, 0, 1086, 1448]],";
exporter = once(exporter, chun, haku + '\n' + chun);
exporter = once(exporter, '${manifest.tiles.length} approved faces. ', '${manifest.tiles.length} approved faces. White dragon uses A — The Fold That Woke Up: the selected ivory silk dragon above mirror water, with four tiny figures holding a flat sheet. The complete 3:4 portrait is embedded at its original 1086 × 1448 pixel dimensions, without repainting, recolouring or adding a tile background. ');
// Keep the existing fourteen-tile example hand, including the seven-disks agent's choice, unchanged.
put(exporterPath, exporter);
const earlier = readdirSync('web/tests').filter(name => /^dali-.*\.test\.js$/.test(name) && `web/tests/${name}` !== newTest);
for (const name of earlier) {
  const file = `web/tests/${name}`;
  const s = read(file).replaceAll('set.tiles.length, 24', 'set.tiles.length, 25')
    .replaceAll('set.placeholders.length, 10', 'set.placeholders.length, 9')
    .replaceAll('24 approved faces', '25 approved faces')
    .replaceAll('remaining 10 tiles', 'remaining 9 tiles')
    .replaceAll('twenty-four', 'twenty-five')
    .replaceAll('ten placeholders', 'nine placeholders')
    .replaceAll('ten named fallbacks', 'nine named fallbacks')
    .replaceAll("'4z', '7z'", "'4z', '5z', '7z'");
  put(file, s);
}
const facesPath = 'web/tests/tile-faces.test.js';
const faces = read(facesPath);
const begin = faces.indexOf("test('Dali resolves");
const end = faces.indexOf("test('hidden tiles", begin);
assert.ok(begin >= 0 && end > begin);
let block = faces.slice(begin, end);
block = once(block, 'twenty-four approved images', 'twenty-five approved images');
block = once(block, "'4z', '7z'", "'4z', '5z', '7z'");
block = once(block, "['2p', '4p', '6p', '7p', '3s', '4s'].includes(tile)", "['2p', '4p', '6p', '7p', '3s', '4s', '5z'].includes(tile)");
const pei = "  assert.equal(tileImage('4z', 'dali'), 'tiles/dali/approved/Pei.svg');";
block = once(block, pei, pei + "\n  assert.equal(tileImage('5z', 'dali'), 'tiles/dali/approved/Haku.svg');");
put(facesPath, faces.slice(0, begin) + block + faces.slice(end));

const notesPath = 'docs/design/dali/README.md';
let notes = read(notesPath);
notes = once(notes, 'Twenty-four faces are approved', 'Twenty-five faces are approved');
notes = once(notes, 'The latest is seven disks', 'The latest is white dragon **The Fold That Woke Up — A**, the selected first painting of an ivory silk dragon over mirror water, with four tiny figures holding a flat sheet below. The complete 1086 × 1448 (3:4) painting is retained. The previously selected seven disks');
notes = once(notes, '| `Chun.svg` |', '| `Haku.svg` | White dragon | The Fold That Woke Up — A: a sinuous ivory silk dragon above mirror water and four tiny figures holding a flat sheet |\n| `Chun.svg` |');
notes = once(notes, 'The other 10 tiles', 'The other 9 tiles');
const paragraph = `White dragon uses [selected A — The Fold That Woke Up](studies/white-dragon-a-approved.svg), from the first painting \`silken_dragon_over_mirror_waters.png\` (1086 × 1448; original PNG SHA-256 \`${metadata.originalSha256}\`). Its complete silk dragon, seascape, figures and flat sheet are retained. The full-dimension AVIF is encoded at quality 50, speed 0, without resizing, compositional cropping, redrawing, recolouring or adding a tile background. Compression is lossy; the original PNG remains in the conversation, while the repository retains the AVIF and byte-identical self-contained source/runtime SVGs. Existing logical canvas, rounded clipping and 1% bleed are unchanged. See [provenance](white-dragon.json). Options B (shadow dragon) and C (skeleton) are not deployed. All previously approved artwork, including the independently deployed Seven Disks and other artists' White Dragons, remains unchanged. Export this tile alone with \`node web/scripts/export-dali-tiles.mjs --only=5z\`.\n\n`;
notes = once(notes, '## Presentation\n\n', '## Presentation\n\n' + paragraph);
put(notesPath, notes);

const runExport = () => execFileSync(process.execPath, [exporterPath, '--only=5z'], { stdio: 'inherit' });
runExport();
const current = json(manifestPath);
assert.equal(current.tiles.length, 25);
assert.equal(current.placeholders.length, 9);
for (const entry of previous.tiles) assert.deepEqual(current.tiles.find(item => item.tile === entry.tile), entry, entry.tile);
assert.deepEqual(current.placeholders, previous.placeholders.filter(item => item.tile !== '5z'));
assert.equal(current.fallback, previous.fallback);
assert.deepEqual(current.facePresentation, previous.facePresentation);
assert.deepEqual(readFileSync(source), readFileSync(runtime));
const generated = [manifestPath, 'web/src/lib/dali-faces.js', 'web/public/tiles/dali/preview.html'];
const before = new Map(generated.map(file => [file, hash(readFileSync(file))]));
runExport();
for (const [file, expected] of before) assert.equal(hash(readFileSync(file)), expected, `Non-deterministic export: ${file}`);
for (const [file, expected] of saved) assert.equal(hash(readFileSync(file)), expected, `Prior artwork changed: ${file}`);
const allowed = new Set([notesPath, ...recovered, exporterPath, facesPath, ...generated, ...earlier.map(name => `web/tests/${name}`)]);
const changed = execFileSync('git', ['diff', '--name-only', '-z']).toString().split('\0').filter(Boolean);
for (const file of changed) assert.ok(allowed.has(file), `Out-of-scope change: ${file}`);
console.log(`Verified ${saved.size} existing artwork files and all ${previous.tiles.length} prior Dali manifest entries unchanged. Recovered approved White Dragon A with exact SVG and raster hashes. Two --only=5z exports were byte-identical. Seven Disks and every other artist's set are unchanged.`);
