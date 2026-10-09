import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { TILE_TYPES, tileFile, tileWords } from '../src/lib/tiles.js';
import { faceImage, fitRecord, imageAttributes, placeStudyImage } from './face-fit.mjs';

// Export selected artwork; newer studies use documented optimized sources.
const root = fileURLToPath(new URL('../../', import.meta.url));
const out = path.join(root, 'web/public/tiles/van-gogh');
const facePresentation = { radius: 26, bleed: 3, preserveAspectRatio: 'none' };
// Paintings that are not 3:4 are cropped to the face rather than stretched to
// it, keeping their proportions to within 2%, or 4% where their counted objects
// or characters fill more of them (see face-fit.mjs). The 300 × 400
// studies for 2s, 3s, 4s, 7s, 8s, 3p, 4p, 9p and 6m were resized from other
// shapes without keeping proportions, so their fits name the original crop that
// docs/design/van-gogh records, and the fit undoes the squeeze as it crops.
const fits = {
  '1p': { stretch: -0.02, anchor: 0.5 },
  '5p': { stretch: -0.02, anchor: 0.5 },
  // Most of the cut falls on the meadow, where the stalks already disappear into
  // the grass, so the crowns keep more of their leaves against the sky.
  '3s': { painting: [439, 673], stretch: 0.02, anchor: 0.25 },
  // The leaves above 三's top branch and the foot of 萬's left leg leave little
  // height to cut, so the cut is shared evenly between them. A one-pixel side
  // bleed, which still hides the board's cream fringe in the outermost columns,
  // leaves enough height for a 2% fit.
  '3m': { stretch: 0.02, anchor: 0.418, bleedPixels: 1 },
  // A centred cut would clip the tip of the kingfisher's beak, at 96.6% of the
  // width; the bird's back reaches only 5% from the left edge.
  '1s': { stretch: -0.02, anchor: 0.6 },
  // Anchored right, so the orange sunflower loses nothing it showed before; its
  // outermost petal ends a pixel short of the board's cream fringe, which the
  // bleed must still hide. The cut falls on the yellow sunflower, whose petals
  // already run off the painting's left edge.
  '2p': { stretch: -0.02, anchor: 0.865 },
  '9p': { painting: [729, 1093], stretch: 0.02, anchor: 0.5 },
  '5z': { stretch: -0.02, anchor: 0.5 },
  '1z': { stretch: 0.02, anchor: 0.5 },
  // A centred cut would leave the upper roof of 二 against the top edge; this
  // one gives it and the hook of 萬 equal room.
  '2m': { stretch: 0.02, anchor: 0.466 },
  // The cypresses of 四萬 run from a flame tip near the top to the point of
  // 萬's foot near the bottom, so the face keeps the 4% stretch that any face
  // may show at most, which leaves both just touching the edges. The painting
  // reaches the canvas edge, so it needs no side bleed.
  '4m': { stretch: 0.04, anchor: 0.573, bleedPixels: 0 },
  '2s': { painting: [433, 667], stretch: 0.02, anchor: 0.5 },
  '4s': { painting: [438, 671], stretch: 0.02, anchor: 0.5 },
  '7s': { painting: [479, 793], stretch: 0.02, anchor: 0.5 },
  '8s': { painting: [1024, 1536], stretch: 0.02, anchor: 0.5 },
  // The lanterns hang through 72% of a painting almost twice as tall as it is
  // wide, so the face keeps the 4% stretch that any face may show at most,
  // which leaves the top lantern's finial and the bottom lantern's drop just
  // touching the edges. Most of the cut falls on the café terrace below them,
  // which is setting, and the side columns are clean painting, so there is no
  // side bleed.
  '3p': { painting: [442, 860], stretch: 0.04, anchor: 0.152, bleedPixels: 0 },
  '6m': { painting: [1295, 1214], stretch: -0.02, anchor: 0.5 },
  '4p': { painting: [442, 796], stretch: 0.02, anchor: 0.5 },
};
const sources = {
  first: 'docs/design/van-gogh/studies/01-van-gogh-concepts.png',
  second: 'docs/design/van-gogh/studies/02-van-gogh-concepts.png',
  east: 'docs/design/van-gogh/studies/03-east-wind-alternatives.png',
  north: 'docs/design/van-gogh/studies/04-north-wind-ribbons.png',
  characters: 'docs/design/van-gogh/studies/06-three-characters-new-directions.png',
  nightCafe: 'docs/design/van-gogh/studies/07-two-characters-night-cafe.png',
  cypressFields: 'docs/design/van-gogh/studies/08-four-characters-cypress-fields.png',
  gardenRhythmGreen: 'docs/design/van-gogh/studies/11-two-bamboo-garden-rhythm-green.webp',
  tripleShootsGreen: 'docs/design/van-gogh/studies/12-three-bamboo-triple-shoots-green.webp',
  moonlitFourGreen: 'docs/design/van-gogh/studies/13-four-bamboo-moonlit-four-green.webp',
  sevenIrises: 'docs/design/van-gogh/studies/14-seven-bamboo-irises-c.webp',
  bambooRaft: 'docs/design/van-gogh/studies/14-eight-bamboo-raft-approved.svg',
  copperFive: 'docs/design/van-gogh/studies/15-five-bamboo-copper-sunset.webp',
  sixStillLife: 'docs/design/van-gogh/studies/15-six-bamboo-green-still-life-b-approved.svg',
  nineWindChime: 'docs/design/van-gogh/studies/16-nine-bamboo-wind-chime-a-approved.svg',
  threeLanterns: 'docs/design/van-gogh/studies/17-three-disks-cafe-lanterns-b-approved.svg',
  pottersTable: 'docs/design/van-gogh/studies/17-nine-disks-potters-table-c-approved.svg',
  nineStarsOriginal: 'docs/design/van-gogh/superseded/nine-stars-Pin9.png',
  fiveVineyard: 'docs/design/van-gogh/studies/18-five-characters-vineyard-a-approved.svg',
  sixLemonTerrace: 'docs/design/van-gogh/studies/19-six-characters-lemon-terrace-approved.svg',
  sevenCharactersIrises: 'docs/design/van-gogh/studies/20-seven-characters-irises-c-approved.svg',
  fourOranges: 'docs/design/van-gogh/studies/20-four-disks-oranges-a-approved.svg',
};
const definitions = [
  ['A', '1p', 'One disk', 'first', [54, 122, 357, 462]],
  ['B', '5p', 'Five disks', 'first', [449, 122, 356, 463]],
  ['Bamboo A (green)', '3s', 'Three bamboo', 'tripleShootsGreen', [0, 0, 300, 400]],
  ['Characters A', '3m', 'Three characters', 'characters', [25, 93, 523, 782]],
  ['E', '7z', 'Red dragon', 'first', [449, 649, 356, 473]],
  ['G', '1s', 'One bamboo', 'second', [38, 100, 380, 471]],
  ['H', '2p', 'Two disks', 'second', [439, 100, 377, 471]],
  ['Disks C', '9p', 'Nine disks', 'pottersTable', [0, 0, 300, 400]],
  ['Six B (green)', '6s', 'Six bamboo', 'sixStillLife', [0, 0, 300, 400]],
  ['L', '5z', 'White dragon', 'second', [838, 645, 380, 495]],
  ['East A', '1z', 'East wind', 'east', [22, 118, 526, 737]],
  ['North B', '4z', 'North wind', 'north', [0, 0, 1086, 1448]],
  ['Characters B', '2m', 'Two characters', 'nightCafe', [0, 0, 1022, 1539]],
  ['Characters C', '4m', 'Four characters', 'cypressFields', [0, 0, 1024, 1536]],
  ['Bamboo B (green)', '2s', 'Two bamboo', 'gardenRhythmGreen', [0, 0, 300, 400]],
  ['Bamboo C (green)', '4s', 'Four bamboo', 'moonlitFourGreen', [0, 0, 300, 400]],
  ['Seven C', '7s', 'Seven bamboo', 'sevenIrises', [0, 0, 300, 400]],
  ['Bamboo Raft', '8s', 'Eight bamboo', 'bambooRaft', [0, 0, 300, 400]],
  ['Copper Sunset', '5s', 'Five bamboo', 'copperFive', [0, 0, 300, 400]],
  ['Wind Chime A', '9s', 'Nine bamboo', 'nineWindChime', [0, 0, 300, 400]],
  ['Lanterns B', '3p', 'Three disks', 'threeLanterns', [0, 0, 300, 400]],
  ['Vineyard A', '5m', 'Five characters', 'fiveVineyard', [0, 0, 300, 400]],
  ['Lemon Terrace', '6m', 'Six characters', 'sixLemonTerrace', [0, 0, 300, 400]],
  ['Irises C', '7m', 'Seven characters', 'sevenCharactersIrises', [0, 0, 300, 400]],
  ['Oranges A', '4p', 'Four disks', 'fourOranges', [0, 0, 300, 400]],
];
const onlyArgument = process.argv.find(argument => argument.startsWith('--only='));
const onlyTiles = onlyArgument ? new Set(onlyArgument.slice(7).split(',')) : null;
if (onlyTiles && [...onlyTiles].some(tile => !definitions.some(definition => definition[1] === tile))) {
  throw new Error('Unknown tile in --only selection');
}
const fullFaceSources = new Set(['nightCafe', 'cypressFields', 'nineStarsOriginal']);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const manifest = {
  version: 1, id: 'van-gogh', name: 'Van Gogh',
  canvas: { width: 300, height: 400 }, facePresentation,
  approval: 'Carl approved A–E, G–J and L, excluding K, then selected East A and North B. On 13 September 2026 he selected Almond Branches (Characters A) to replace D for 3m and requested deployment. Night Cafe (Characters B) for 2m and Cypress Fields (Characters C) for 4m are also approved for deployment. The user also approved the green Garden Rhythm B for 2s and selected green Triple Shoots A to overwrite C for 3s. The user also approved greener Moonlit Four C for 4s. On 29 September 2026 Carl selected Seven with Irises C for 7s and requested deployment. The later L is the active white dragon; F and D remain studies. Carl selected the raft concept for 8s and approved deployment of the resulting eight-bamboo painting without recolouring. Carl approved Copper Sunset for 5s with four green stalks and a reddish-brown central stalk, replacing the earlier all-green concept. Carl approved the corrected all-green B still life with six bamboo stalks to replace Green Rhythm J for 6s; its complete composition and colours are preserved. Carl selected the first nine-bamboo option, Moonlit Bamboo Wind Chime, and explicitly approved deployment; the nine hanging tubes and complete painted composition are preserved. Carl approved B, Three Cafe Lanterns, for 3 disks and explicitly requested deployment. The complete selected middle-panel crop is preserved without repainting or recolouring in a documented quality-90 WebP game export. Carl selected C — The Potter’s Table to replace Nine Stars I for 9p and explicitly requested deployment. The nine patterned plates were deployed and the old face archived. The intervening Nine Stars restoration was a misunderstanding. Carl explicitly clarified that The Potter’s Table C must replace Nine Stars as 9 disks. The approved Potter SVG is restored byte-for-byte; Nine Stars remains archived. The eight-star adaptation is not deployed and 8 disks is unchanged. Carl selected the first option, The Red Vineyard, for 5 characters (5m / 五萬) and explicitly approved GitHub deployment. The complete approved composition and colours are preserved; the wheat and iris alternatives are not used. Carl approved the latest red 六萬 calligraphy with lemons, cypresses, lake and village as 6 characters (6m). The complete approved canvas and colours are retained; the sunflower study is not selected. Carl selected the last seven-characters option with blue iris flowers and golden 萬, Irises at Dusk C, and explicitly requested GitHub deployment as 7m / 七萬. The complete composition and proportions are preserved without repainting; the olive and wheat alternatives are not used. Carl selected A, Four Oranges, the left panel with four whole oranges on blue cloth, for 4 disks (4p / Pin4) and explicitly requested GitHub deployment. The complete approved crop is retained without repainting or recolouring; the bowl and orange-slice alternatives are not used.',
  fallback: 'text',
  sources: Object.entries(sources).map(([id, source]) => ({ id, source, sha256: hash(readFileSync(path.join(root, source))) })),
  tiles: [],
  rejected: [{ candidate: 'K', tile: '1z', label: 'East wind', source: sources.second }],
  superseded: [
    { candidate: 'F', tile: '5z', activeCandidate: 'L', source: sources.first },
    { candidate: 'D', tile: '3m', activeCandidate: 'Characters A', source: sources.first },
    { candidate: 'C', tile: '3s', activeCandidate: 'Bamboo A (green)', source: sources.first },
    { candidate: 'J', tile: '6s', activeCandidate: 'Six B (green)', source: sources.second },
    { candidate: 'I', tile: '9p', activeCandidate: 'Disks C', source: sources.second, archivedPng: 'docs/design/van-gogh/superseded/nine-stars-Pin9.png', archivedSvg: 'docs/design/van-gogh/superseded/nine-stars-Pin9.svg' },
  ],
};
mkdirSync(path.join(out, 'approved'), { recursive: true });
for (const [candidate, tile, label, sourceId, crop] of definitions) {
  const name = tileFile(tile), source = sources[sourceId];
  const [x, y, width, height] = crop;
  const png = `approved/${name}.png`, svg = `approved/${name}.svg`;
  const selected = !onlyTiles || onlyTiles.has(tile);
  const fit = fits[tile];
  const image = faceImage(fit, [width, height]);
  const fitted = fit ? { fit: fitRecord(fit, [width, height]) } : {};
  // Self-contained approved SVG studies are copied exactly, never rasterized or
  // repainted; a fit moves only their picture's box.
  if (source.endsWith('.svg')) {
    const artwork = readFileSync(path.join(root, source));
    const match = artwork.toString('utf8').match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
    if (!match) throw new Error(`Missing embedded WebP in ${source}`);
    const raster = Buffer.from(match[1], 'base64');
    const face = fit ? placeStudyImage(artwork, image) : artwork;
    if (selected) writeFileSync(path.join(out, svg), face);
    manifest.tiles.push({ candidate, tile, name, label, source, status: 'approved', svg,
      crop: { x, y, width, height }, ...fitted, svgSha256: hash(face),
      rasterMimeType: 'image/webp', rasterSha256: hash(raster) });
    continue;
  }
  const raster = !selected
    ? readFileSync(path.join(out, png))
    : fullFaceSources.has(sourceId)
      ? readFileSync(path.join(root, source))
      : execFileSync('convert', [path.join(root, source), '-crop', `${width}x${height}+${x}+${y}`, '+repage', '-strip', 'PNG:-'], { maxBuffer: 8 * 1024 * 1024 });
  if (selected) {
    writeFileSync(path.join(out, png), raster);
    const { radius, preserveAspectRatio } = facePresentation;
    writeFileSync(path.join(out, svg), `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">${label} — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="${radius}"/></clipPath></defs><image clip-path="url(#face)" ${imageAttributes(image)} preserveAspectRatio="${preserveAspectRatio}" href="data:image/png;base64,${raster.toString('base64')}"/></svg>\n`);
  }
  manifest.tiles.push({ candidate, tile, name, label, source, status: 'approved', png, svg,
    crop: { x, y, width, height }, ...fitted, pngSha256: hash(raster) });
}
const approved = manifest.tiles.map(entry => entry.tile);
manifest.remaining = TILE_TYPES.filter(tile => !approved.includes(tile));
writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
writeFileSync(path.join(root, 'web/src/lib/van-gogh-faces.js'), `// Generated by web/scripts/export-van-gogh-tiles.mjs. Includes Four Oranges A (4p), Irises at Dusk C (7m), Lemon Terrace (6m), The Red Vineyard A (5m), The Potter’s Table C (9p), restored after the clarified selection, Three Cafe Lanterns B (3p), Moonlit Wind Chime A (9s), completing the bamboo suit, Green Still Life B (6s), replacing J, Copper Sunset (5s) and Bamboo Raft (8s); all other selections remain unchanged.\nexport const VAN_GOGH_APPROVED = Object.freeze(${JSON.stringify(approved)});\n`);
const gallery = manifest.tiles.map(entry => `<figure><img src="${entry.svg}" alt="${entry.label}" width="300" height="400"><figcaption>${entry.candidate} · ${entry.label}</figcaption></figure>`).join('');
const featured = ['4p', '7m', '6m', '5m', '2m', '3m', '4m', '9p', '3p', '9s', '6s', '8s'];
// The hand shows painted faces only: the game writes out the others' names
// until they are painted, which a picture here cannot show.
const handTiles = [...featured.filter(tile => approved.includes(tile)), ...approved.filter(tile => !featured.includes(tile))].slice(0, 14);
const hand = handTiles.map(tile => {
  const entry = manifest.tiles.find(entry => entry.tile === tile);
  return `<img src="${entry.svg}" alt="${entry.label}" width="300" height="400">`;
}).join('');
writeFileSync(path.join(out, 'preview.html'), `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Van Gogh Mahjong — Approved Tiles</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f5efdf;color:#173457;font:16px/1.5 system-ui,sans-serif}main{max-width:1000px;margin:auto;padding:32px 20px 56px}h1{font:48px/1.1 Georgia,serif;margin:8px 0 16px}h2{font-size:22px;margin:36px 0 16px}p{max-width:680px}a{color:inherit}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:24px}figure{margin:0}figure img{width:100%;height:auto;display:block}figcaption{font-size:14px;margin-top:8px}.scroll{overflow-x:auto;padding:12px 4px 24px}.rack{display:flex;gap:2px;width:390px;padding:20px 12px;background:#173d34;border-radius:12px}.rack img{width:calc((100% - 26px)/14);height:auto;aspect-ratio:3/4;min-width:0;flex:none;border-radius:2px}.notes{color:#5d655f;font-size:14px}select{font:inherit;padding:6px;background:#fff;border:1px solid #aaa;border-radius:6px}@media(max-width:500px){main{padding:24px 16px}.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}}</style></head>
<body><main><a href="../../">← Mahjong</a><h1>Van Gogh</h1><p>${approved.length} approved faces, including Four Oranges A for 4 disks, Irises at Dusk C for 7 characters, Lemon Terrace for 6 characters, The Red Vineyard for 5 characters, The Potter’s Table C for 9 disks, Three Café Lanterns for 3 disks, Moonlit Wind Chime for 9 bamboo, Green Still Life B for 6 bamboo, Copper Sunset for 5 bamboo, the Bamboo Raft for 8 bamboo, Seven with Irises C for 7 bamboo, green Moonlit Four for 4 bamboo, green Triple Shoots for 3 bamboo, green Garden Rhythm for 2 bamboo, Night Cafe for 2 of characters, Almond Branches for 3, Cypress Fields for 4, Blazing Dawn for East and Wind Ribbons for North. Choose <strong>Options → Tile face → Van Gogh</strong> in the game. The remaining ${manifest.remaining.length} tiles show their names until their artwork is approved: ${manifest.remaining.map(tileWords).join(', ')}.</p><h2>Approved artwork</h2><div class="gallery">${gallery}</div>
<h2>A mixed hand</h2><label>Hand width <select id="width"><option value="390">390 px · compact</option><option value="844">844 px · landscape</option></select></label><div class="scroll"><div class="rack" id="rack">${hand}</div></div>
<p class="notes">East wind K is excluded. White dragon L keeps its pale painted dragon under the normal dora ring and foil sheen.</p><p class="notes">Earlier selected art is preserved in lossless PNG crops. Lemon Terrace embeds its entire approved 1295 × 1214 canvas and the face crops it at the sides to keep its proportions; the original is separately preserved. The Potter’s Table C uses its existing approved 300 × 400 quality-95 WebP game export, copied byte-for-byte without repainting or recompression. Nine Stars I remains preserved in the design archive. The newer green bamboo studies use web-optimized crops. Irises at Dusk (7 characters), The Red Vineyard (5 characters), Three Café Lanterns (3 disks), Moonlit Wind Chime (9 bamboo), Green Still Life B (6 bamboo) and Bamboo Raft preserve their complete approved compositions and colours in 300 × 400 WebP images embedded in their SVGs; these are not the full-resolution source PNGs. A painting that is not 3:4, including one squeezed into a 300 × 400 export, is cropped to the face rather than stretched, keeping its proportions to within 2%, or within 4% where its counted objects and characters fill more of it; the manifest records each fit. Source checksums and processing are documented. <a href="manifest.json">Export manifest</a></p></main><script>document.getElementById('width').addEventListener('change',event=>{document.getElementById('rack').style.width=event.target.value+'px'});</script></body></html>\n`);
console.log(`Exported ${approved.length} approved Van Gogh faces; ${manifest.remaining.length} identities show their names until painted.`);
