import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { TILE_TYPES, tileFile } from '../src/lib/tiles.js';

// Export selected artwork; newer studies use documented optimized sources.
const root = fileURLToPath(new URL('../../', import.meta.url));
const out = path.join(root, 'web/public/tiles/van-gogh');
const facePresentation = { radius: 26, bleed: 3, preserveAspectRatio: 'none' };
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
  fiveVineyard: 'docs/design/van-gogh/studies/18-five-characters-vineyard-a-approved.svg',
  sixLemonTerrace: 'docs/design/van-gogh/studies/19-six-characters-lemon-terrace-approved.svg',
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
];
const onlyArgument = process.argv.find(argument => argument.startsWith('--only='));
const onlyTiles = onlyArgument ? new Set(onlyArgument.slice(7).split(',')) : null;
if (onlyTiles && [...onlyTiles].some(tile => !definitions.some(definition => definition[1] === tile))) {
  throw new Error('Unknown tile in --only selection');
}
const fullFaceSources = new Set(['nightCafe', 'cypressFields']);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const manifest = {
  version: 1, id: 'van-gogh', name: 'Van Gogh',
  canvas: { width: 300, height: 400 }, facePresentation,
  approval: 'Carl approved A–E, G–J and L, excluding K, then selected East A and North B. On 13 September 2026 he selected Almond Branches (Characters A) to replace D for 3m and requested deployment. Night Cafe (Characters B) for 2m and Cypress Fields (Characters C) for 4m are also approved for deployment. The user also approved the green Garden Rhythm B for 2s and selected green Triple Shoots A to overwrite C for 3s. The user also approved greener Moonlit Four C for 4s. On 29 September 2026 Carl selected Seven with Irises C for 7s and requested deployment. The later L is the active white dragon; F and D remain studies. Carl selected the raft concept for 8s and approved deployment of the resulting eight-bamboo painting without recolouring. Carl approved Copper Sunset for 5s with four green stalks and a reddish-brown central stalk, replacing the earlier all-green concept. Carl approved the corrected all-green B still life with six bamboo stalks to replace Green Rhythm J for 6s; its complete composition and colours are preserved. Carl selected the first nine-bamboo option, Moonlit Bamboo Wind Chime, and explicitly approved deployment; the nine hanging tubes and complete painted composition are preserved. Carl approved B, Three Cafe Lanterns, for 3 disks and explicitly requested deployment. The complete selected middle-panel crop is preserved without repainting or recolouring in a documented quality-90 WebP game export. Carl selected C — The Potter’s Table to replace Nine Stars I for 9p and explicitly requested deployment. The nine patterned plates retain the selected painting; the old face is archived. Carl selected the first option, The Red Vineyard, for 5 characters (5m / 五萬) and explicitly approved GitHub deployment. The complete approved composition and colours are preserved; the wheat and iris alternatives are not used. Carl approved the latest red 六萬 calligraphy with lemons, cypresses, lake and village as 6 characters (6m). The complete approved canvas and colours are retained; the sunflower study is not selected.',
  fallback: 'classic',
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
  // Self-contained approved SVG studies are copied exactly, never rasterized or repainted.
  if (source.endsWith('.svg')) {
    const artwork = readFileSync(path.join(root, source));
    const match = artwork.toString('utf8').match(/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/);
    if (!match) throw new Error(`Missing embedded WebP in ${source}`);
    const raster = Buffer.from(match[1], 'base64');
    if (selected) writeFileSync(path.join(out, svg), artwork);
    manifest.tiles.push({ candidate, tile, name, label, source, status: 'approved', svg,
      crop: { x, y, width, height }, svgSha256: hash(artwork),
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
    const { radius, bleed, preserveAspectRatio } = facePresentation;
    const verticalBleed = bleed * 4 / 3;
    writeFileSync(path.join(out, svg), `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">${label} — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="${radius}"/></clipPath></defs><image clip-path="url(#face)" x="${-bleed}" y="${-verticalBleed}" width="${300 + 2 * bleed}" height="${400 + 2 * verticalBleed}" preserveAspectRatio="${preserveAspectRatio}" href="data:image/png;base64,${raster.toString('base64')}"/></svg>\n`);
  }
  manifest.tiles.push({ candidate, tile, name, label, source, status: 'approved', png, svg,
    crop: { x, y, width, height }, pngSha256: hash(raster) });
}
const approved = manifest.tiles.map(entry => entry.tile);
manifest.remaining = TILE_TYPES.filter(tile => !approved.includes(tile));
writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
writeFileSync(path.join(root, 'web/src/lib/van-gogh-faces.js'), `// Generated by web/scripts/export-van-gogh-tiles.mjs. Includes Lemon Terrace (6m), The Red Vineyard A (5m), The Potter’s Table C (9p), replacing Nine Stars I, Three Cafe Lanterns B (3p), Moonlit Wind Chime A (9s), completing the bamboo suit, Green Still Life B (6s), replacing J, Copper Sunset (5s) and Bamboo Raft (8s); all other selections remain unchanged.\nexport const VAN_GOGH_APPROVED = Object.freeze(${JSON.stringify(approved)});\n`);
const gallery = manifest.tiles.map(entry => `<figure><img src="${entry.svg}" alt="${entry.label}" width="300" height="400"><figcaption>${entry.candidate} · ${entry.label}</figcaption></figure>`).join('');
const featured = ['6m', '5m', '2m', '3m', '4m', '9p', '3p', '9s', '6s', '5s', '8s', '7s'];
const handTiles = [...featured, ...approved.filter(tile => !featured.includes(tile)).slice(0, 12 - featured.length), '2z', '6z'];
const hand = handTiles.map(tile => {
  const entry = manifest.tiles.find(entry => entry.tile === tile);
  return `<img src="${entry ? entry.svg : `../${tileFile(tile)}.svg`}" alt="${entry?.label ?? tile}" width="300" height="400">`;
}).join('');
writeFileSync(path.join(out, 'preview.html'), `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Van Gogh Mahjong — Approved Tiles</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f5efdf;color:#173457;font:16px/1.5 system-ui,sans-serif}main{max-width:1000px;margin:auto;padding:32px 20px 56px}h1{font:48px/1.1 Georgia,serif;margin:8px 0 16px}h2{font-size:22px;margin:36px 0 16px}p{max-width:680px}a{color:inherit}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:24px}figure{margin:0}figure img{width:100%;height:auto;display:block}figcaption{font-size:14px;margin-top:8px}.scroll{overflow-x:auto;padding:12px 4px 24px}.rack{display:flex;gap:2px;width:390px;padding:20px 12px;background:#173d34;border-radius:12px}.rack img{width:calc((100% - 26px)/14);height:auto;aspect-ratio:3/4;min-width:0;flex:none;border-radius:2px}.notes{color:#5d655f;font-size:14px}select{font:inherit;padding:6px;background:#fff;border:1px solid #aaa;border-radius:6px}@media(max-width:500px){main{padding:24px 16px}.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}}</style></head>
<body><main><a href="../../">← Mahjong</a><h1>Van Gogh</h1><p>${approved.length} approved faces, including Lemon Terrace for 6 characters, The Red Vineyard for 5 characters, The Potter’s Table C for 9 disks, Three Café Lanterns for 3 disks, Moonlit Wind Chime for 9 bamboo, Green Still Life B for 6 bamboo, Copper Sunset for 5 bamboo, the Bamboo Raft for 8 bamboo, Seven with Irises C for 7 bamboo, green Moonlit Four for 4 bamboo, green Triple Shoots for 3 bamboo, green Garden Rhythm for 2 bamboo, Night Cafe for 2 of characters, Almond Branches for 3, Cypress Fields for 4, Blazing Dawn for East and Wind Ribbons for North. Choose <strong>Options → Tile face → Van Gogh</strong> in the game. The remaining ${manifest.remaining.length} tiles use Classic artwork.</p><h2>Approved artwork</h2><div class="gallery">${gallery}</div>
<h2>A mixed hand</h2><label>Hand width <select id="width"><option value="390">390 px · compact</option><option value="844">844 px · landscape</option></select></label><div class="scroll"><div class="rack" id="rack">${hand}</div></div>
<p class="notes">The last two tiles use Classic artwork. East wind K is excluded. White dragon L keeps its pale painted dragon under the normal dora ring and foil sheen.</p><p class="notes">Earlier selected art is preserved in lossless PNG crops. Lemon Terrace fits its entire approved 1295 × 1214 canvas to the shared 300 × 400 face; the original is separately preserved and the aspect ratio changes. The Potter’s Table C uses a documented quality-95 WebP crop; its former Nine Stars face is archived. The newer green bamboo studies use web-optimized crops. The Red Vineyard (5 characters), Three Café Lanterns (3 disks), Moonlit Wind Chime (9 bamboo), Green Still Life B (6 bamboo) and Bamboo Raft preserve their complete approved compositions and colours in 300 × 400 WebP images embedded in their SVGs; these are not the full-resolution source PNGs. Source checksums and processing are documented. <a href="manifest.json">Export manifest</a></p></main><script>document.getElementById('width').addEventListener('change',event=>{document.getElementById('rack').style.width=event.target.value+'px'});</script></body></html>\n`);
console.log(`Exported ${approved.length} approved Van Gogh faces; ${manifest.remaining.length} identities use Classic artwork.`);
