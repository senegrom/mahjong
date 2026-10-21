import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';

const read = file => readFileSync(file, 'utf8');
const put = (file, value) => writeFileSync(file, value);
const json = file => JSON.parse(read(file));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const once = (s, a, b) => { assert.equal(s.split(a).length, 2, `Expected one anchor: ${a}`); return s.replace(a, b); };
const prep = '.dali-green-b-prep';
const manifestPath = 'web/public/tiles/dali/manifest.json';
const exporterPath = 'web/scripts/export-dali-tiles.mjs';
const notesPath = 'docs/design/dali/README.md';
const facesPath = 'web/tests/tile-faces.test.js';
const provenancePath = 'docs/design/dali/green-dragon.json';
const testPath = 'web/tests/dali-green-dragon.test.js';
const metadata = json(`${prep}/metadata.json`);
const expectedSvg = 'd25dd23f919dff9f8fc700c4c45527f31d944a858c8c2479bebd9a7684500205';
const generated = [manifestPath, 'web/src/lib/dali-faces.js', 'web/public/tiles/dali/preview.html'];
const snapshotPath = `${prep}/preservation.json`;
const previous = process.argv.includes('--verify') ? json(snapshotPath).manifest : json(manifestPath);
const tracked = execFileSync('git', ['ls-files', '-z']).toString().split('\0').filter(Boolean);
const artwork = tracked.filter(file => /^(docs\/design\/|web\/public\/tiles\/)/.test(file) && /\.(svg|png|avif|webp)$/i.test(file));
const unrelated = tracked.filter(file => /^(docs\/design\/(?:van-gogh|matisse)\/|web\/public\/tiles\/(?:van-gogh|matisse)\/|web\/src\/lib\/(?:van-gogh|matisse)-faces\.js$)/.test(file));
const saved = process.argv.includes('--verify') ? json(snapshotPath).files : Object.fromEntries([...new Set([...artwork, ...unrelated])].map(file => [file, hash(readFileSync(file))]));
const verify = () => {
  const current = json(manifestPath);
  assert.equal(current.tiles.length, previous.tiles.length + 1);
  assert.equal(current.placeholders.length, previous.placeholders.length - 1);
  for (const entry of previous.tiles) assert.deepEqual(current.tiles.find(item => item.tile === entry.tile), entry, entry.tile);
  assert.deepEqual(current.placeholders, previous.placeholders.filter(item => item.tile !== '6z'));
  for (const [file, expected] of Object.entries(saved)) assert.equal(hash(readFileSync(file)), expected, `Existing artwork/data changed: ${file}`);
  assert.deepEqual(readFileSync(metadata.runtime), readFileSync(metadata.source));
  assert.equal(hash(readFileSync(metadata.source)), expectedSvg);
  assert.equal(hash(readFileSync(metadata.rasterSource)), metadata.rasterSha256);
  const first = new Map([...generated, metadata.runtime].map(file => [file, hash(readFileSync(file))]));
  execFileSync(process.execPath, [exporterPath, '--only=6z'], { stdio: 'inherit' });
  for (const [file, expected] of first) assert.equal(hash(readFileSync(file)), expected, `Non-deterministic export: ${file}`);
  for (const [file, expected] of Object.entries(saved)) assert.equal(hash(readFileSync(file)), expected, `Existing artwork/data changed after export: ${file}`);
  const allowed = new Set([notesPath, provenancePath, testPath, metadata.source, metadata.runtime, metadata.rasterSource, exporterPath, facesPath, ...generated, ...readdirSync('web/tests').filter(name => /^dali-.*\.test\.js$/.test(name)).map(name => `web/tests/${name}`)]);
  for (const file of execFileSync('git', ['diff', '--name-only', '-z']).toString().split('\0').filter(Boolean)) assert.ok(allowed.has(file), `Out-of-scope change: ${file}`);
  console.log(`Verified ${Object.keys(saved).length} prior artwork/data files and ${previous.tiles.length} Dalí manifest entries unchanged; exact selected raster/SVG hashes and deterministic exports.`);
};
if (process.argv.includes('--verify')) {
  verify();
  process.exit(0);
}
assert.equal(previous.tiles.length, 27);
assert.equal(previous.placeholders.length, 7);
assert.equal(previous.tiles.some(tile => tile.tile === '6z'), false);
assert.equal(previous.fallback, 'text');
put(snapshotPath, JSON.stringify({ manifest: previous, files: saved }));
const raster = Buffer.concat(Array.from({ length: 7 }, (_, i) => readFileSync(`${prep}/part${String(i).padStart(2, '0')}.bin`)));
assert.equal(raster.length, 60307);
assert.equal(hash(raster), metadata.rasterSha256);
assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
const ispe = raster.indexOf(Buffer.from('ispe'));
assert.ok(ispe >= 0);
assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [606, 808]);
for (const file of [metadata.source, metadata.runtime, metadata.rasterSource, provenancePath, testPath]) assert.equal(existsSync(file), false, file);
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">Green dragon — Dali — The River That Escaped — Green B</title><metadata>${JSON.stringify(metadata)}</metadata><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,${raster.toString('base64')}"/></svg>\n`;
assert.equal(hash(svg), expectedSvg);
put(metadata.rasterSource, raster);
put(metadata.source, svg);
put(provenancePath, JSON.stringify({ ...metadata, svgSha256: hash(svg) }, null, 2) + '\n');
let exporter = read(exporterPath);
const chun = "  ['Chun', '7z', 'Red dragon', 'Molten Ruby', board, [847, 656, 373, 513]],";
const green = "  ['Hatsu', '6z', 'Green dragon', 'The River That Escaped — Green B', 'green-dragon-b-river-approved.svg', [0, 0, 606, 808]],";
exporter = once(exporter, chun, green + '\n' + chun);
exporter = once(exporter, '${manifest.tiles.length} approved faces. ', '${manifest.tiles.length} approved faces. Green dragon uses Green B — The River That Escaped: the selected middle painting from the latest greener triptych, with an emerald water dragon carrying boats and forests above a cloud-filled canyon and stone bridge. The 606 × 808 (3:4) crop removes separators, small side margins and bottom option letters, with no resizing, repainting or further recolouring. All three dragons are now approved. ');
put(exporterPath, exporter);
const earlier = readdirSync('web/tests').filter(name => /^dali-.*\.test\.js$/.test(name));
for (const name of earlier) {
  const file = `web/tests/${name}`;
  put(file, read(file).replaceAll('set.tiles.length, 27', 'set.tiles.length, 28')
    .replaceAll('set.placeholders.length, 7', 'set.placeholders.length, 6')
    .replaceAll('27 approved faces', '28 approved faces')
    .replaceAll('remaining 7 tiles', 'remaining 6 tiles')
    .replaceAll('twenty-seven', 'twenty-eight')
    .replaceAll('seven placeholders', 'six placeholders')
    .replaceAll('seven named fallbacks', 'six named fallbacks')
    .replaceAll("'4z', '5z', '7z'", "'4z', '5z', '6z', '7z'"));
}
let faces = read(facesPath);
const begin = faces.indexOf("test('Dali resolves");
const end = faces.indexOf("test('hidden tiles", begin);
assert.ok(begin >= 0 && end > begin);
let block = faces.slice(begin, end);
block = once(block, 'twenty-seven', 'twenty-eight');
block = once(block, "'4z', '5z', '7z'", "'4z', '5z', '6z', '7z'");
block = once(block, "'4s', '5z'].includes(tile)", "'4s', '5z', '6z'].includes(tile)");
const haku = "  assert.equal(tileImage('5z', 'dali'), 'tiles/dali/approved/Haku.svg');";
block = once(block, haku, haku + "\n  assert.equal(tileImage('6z', 'dali'), 'tiles/dali/approved/Hatsu.svg');");
put(facesPath, faces.slice(0, begin) + block + faces.slice(end));
put(testPath, read(`${prep}/test.js`));
let notes = read(notesPath);
notes = once(notes, 'Twenty-seven faces are approved', 'Twenty-eight faces are approved');
notes = once(notes, 'The latest is nine disks', 'The latest is green dragon **The River That Escaped — Green B**, the selected middle painting from the latest greener triptych: an emerald water dragon carrying boats and forests over a cloud-filled canyon and stone bridge. This completes the dragon trio. The previously selected nine disks');
notes = once(notes, '| `Chun.svg` |', '| `Hatsu.svg` | Green dragon | The River That Escaped — Green B: emerald water dragon carrying boats and forests above a cloud-filled canyon and stone bridge |\n| `Chun.svg` |');
notes = once(notes, 'The other 7 tiles', 'The other 6 tiles');
const paragraph = 'Green dragon uses [selected Green B — The River That Escaped](studies/green-dragon-b-river-approved.svg), the middle panel of the latest greener triptych `a_wide_triptych_fantasy_surreal_digital_painting.png` (1881 × 836). This is not either earlier standalone river-dragon image. The mechanical crop at (639, 0), 606 × 808 pixels, removes separators, small side margins and the bottom option letters while keeping the dragon, boats, waterfalls and stone bridge. No resizing, stretching, repainting or further recolouring is applied. The crop is encoded at its original pixel dimensions as AVIF quality 50, speed 0 (lossy), embedded in identical source/runtime SVGs. Existing logical canvas, rounded clipping and 1% bleed are unchanged. The original board and lossless crop are preserved in `dali-green-dragon-B-originals.zip` supplied in chat. See [provenance](green-dragon.json) for all hashes and extraction details. Pear A and porcelain C are not selected. All prior artwork, other sets and the fourteen-tile example hand remain unchanged. Reproduce only this tile with `node web/scripts/export-dali-tiles.mjs --only=6z`.\n\n';
notes = once(notes, '## Presentation\n\n', '## Presentation\n\n' + paragraph);
put(notesPath, notes);
execFileSync(process.execPath, [exporterPath, '--only=6z'], { stdio: 'inherit' });
verify();
