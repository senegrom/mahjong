import { readFileSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { TILE_TYPES, tileFile, tileWords } from '../src/lib/tiles.js';

// Export the approved source pixels, without regenerating or redrawing them.
// Run from any directory: node web/scripts/export-matisse-tiles.mjs
const root = fileURLToPath(new URL('../../', import.meta.url));
const out = path.join(root, 'web/public/tiles/matisse');
const sourceDir = 'docs/design/matisse/studies';
// The study crops include a physical tile, not just the printed motif. Fit
// that tile to the game's face instead of letterboxing one tile inside another.
// A small bleed hides the study's outside backdrop without trimming the art.
const facePresentation = { radius: 26, bleed: 3, preserveAspectRatio: 'none' };
const definitions = [
  ['Pin1', '1p', '1 dot', 'approved', '01-disk-and-red-dragon.png', [77, 232, 410, 554], 'Black disk and ivory rosette on yellow; direction B.'],
  ['Pin2', '2p', '2 disks', 'approved', 'two-disks-b-approved.png', [689, 60, 504, 663], 'B: two yellow and magenta cut-paper rosettes on ultramarine, with contrasting petals and blue centres.'],
  ['Pin3', '3p', '3 disks', 'approved', 'three-disks-c-approved.png', [1279, 80, 537, 636], 'C: three apricot, mint and carmine cut-paper rosettes in a loose S on deep purple, with coloured centres.'],
  ['Pin4', '4p', '4 disks', 'approved', 'four-disks-a-eclipse-approved.png', [26, 198, 483, 623], 'Éclipse A: four irregular ultramarine rings with vermilion wedges on warm ivory, in an asymmetric cut-paper dance. The lossless crop preserves the selected artwork and excludes the study label and surround, using the shared 3:4 face, bleed and rounded clipping.'],
  ['Pin5', '5p', '5 dots', 'approved', '02-five-dot-study.png', [77, 231, 410, 572], 'Five black and ivory rosettes on yellow.'],
  ['Pin6', '6p', '6 disks', 'approved', 'six-disks-c-approved.png', [1181, 89, 548, 713], 'C: six coral, chartreuse and orange cut-paper rosettes in three loose pairs on forest green, with broad open cuts and contrasting centres. The lossless crop excludes the study label and surround, using the shared 3:4 face, bleed and rounded clipping.'],
  ['Pin7', '7p', '7 disks', 'approved', 'seven-disks-a-approved.png', [31, 80, 514, 742], 'Ivory Garden A: seven blue, vermilion and green cut-paper rosettes in a loose 2-3-2 arrangement on ivory, with contrasting centres. The flat crop preserves the selected source pixels and excludes the study label and surround; it uses the shared 3:4 face, bleed and rounded clipping.'],
  ['Pin8', '8p', '8 disks', 'approved', 'eight-disks-c-approved.png', [1193, 77, 544, 729], 'Cut & Swap C: eight indigo and chartreuse cut-paper disks in four pairs across a sweeping divided colour field, with coral inserts. The lossless crop preserves all eight complete disks and excludes the study label and surround, using the shared 3:4 face, bleed and rounded clipping.'],
  ['Pin9', '9p', '9 disks', 'approved', 'nine-disks-b-approved.png', [670, 57, 541, 651], 'Orbit B: eight yellow, mint and pink cut-paper rosettes surround one larger apricot centre on deep purple. Exactly nine separate disks, with the shared full-face fit, bleed and rounded clipping.'],
  ['Sou2', '2s', '2 bamboo', 'approved', '08-two-bamboo-approved.png', [495, 174, 465, 769], 'Dance B: two sweeping green bamboo forms on a pale mint field. All printed artwork stays green for All Green; multiple shades are intentional.'],
  ['Sou3', '3s', '3 bamboo', 'approved', 'three-bamboo-original-b-approved.jpg', [614, 413, 219, 271], 'Original Chasuble B: three ivory jointed bamboo cut-outs, one above two, on a green cut-paper field. Selected from the original three-direction board. The crop excludes the photographed tile edge and uses the shared 3:4 face, bleed and rounded clipping. Green and neutral ivory preserve the All Green palette.'],
  ['Sou4', '4s', '4 bamboo', 'approved', 'four-bamboo-b-approved.png', [586, 119, 500, 700], 'Chapel B: four pale-green sculptural bamboo cut-outs on a deep forest-green field. All artwork stays green for All Green. The flat face excludes the study labels and surround, with the shared full-face fit and rounded clipping.'],
  ['Sou5', '5s', '5 bamboo', 'approved', '09-five-bamboo-approved.png', [963, 256, 448, 640], 'Fan C: four pale-green bamboo sprigs and a bold red centre on a deep forest-green field. Exactly five complete sprigs; the red accent distinguishes this tile from All Green tiles.'],
  ['Sou6', '6s', '6 bamboo', 'approved', 'six-bamboo-d-green-carnival-approved.png', [66, 70, 948, 1264], 'Green Carnival D: six distinct abstract green cut-outs dance around an open centre on deep forest green. The final crisp, glow-free artwork stays entirely green for All Green. The lossless 3:4 crop excludes the presentation label and surround, using the shared bleed and rounded clipping.'],
  ['Sou7', '7s', '7 bamboo', 'approved', '09-seven-bamboo-approved.png', [1037, 115, 468, 719], 'Cut-out C: one coral and six green abstract forms circling an open centre on butter yellow. Seven separated silhouettes preserve the tile count.'],
  ['Sou8', '8s', '8 bamboo', 'approved', 'eight-bamboo-c-wild-growth-approved.png', [60, 26, 967, 1281], 'Wild Growth C: eight mint and dark-forest bamboo fronds burst around an open centre on emerald green. All artwork stays green for All Green. The lossless crop preserves the selected artwork and excludes the study label and surround, using the shared 3:4 face, bleed and rounded clipping.'],
  ['Sou9', '9s', '9 bamboo', 'approved', 'nine-bamboo-b-approved.png', [531, 200, 477, 631], 'Vestment B: nine jointed bamboo ribbons in three staggered groups of three, green, ivory and chartreuse on an ultramarine cut-paper field.'],
  ['Ton', '1z', 'East wind', 'approved', 'east-wind-c-chapel-stencil-approved.png', [1028, 176, 469, 680], 'Chapel Stencil C from the second, bolder board: a broad deep-forest-green 東 with four coral-pink cut-out windows and an ivory upper crossbar on coral pink. The lossless crop preserves the selected source pixels and excludes the study label and surround, using the shared 3:4 face, bleed and rounded clipping with no inset border.'],
  ['Nan', '2z', 'South wind', 'approved', 'south-wind-c-approved.png', [975, 238, 395, 552], 'Chapel C: sculptural aubergine 南 with a carmine upper interior bar on saffron yellow. Both interior horizontal bars remain visible. The lossless crop excludes the study label, surround and photographed rim, using the shared 3:4 face, bleed and rounded clipping.'],
  ['Shaa', '3z', 'West wind', 'approved', 'west-wind-c-approved.png', [963, 213, 435, 602], 'Cut-out C: sculptural ivory 西 with a cobalt-blue top stroke on vermilion. The flat crop excludes the study label and surround, using the shared full-face fit, bleed and rounded clipping.'],
  ['Pei', '4z', 'North wind', 'approved', 'north-wind-b-approved.png', [579, 85, 523, 700], 'Ribbon Dance B: lemon and coral-pink 北 with a mint accent on ultramarine. The corrected five-stroke character includes the diagonal arm on the right. The flat crop preserves the selected artwork and excludes the label and surround, using the shared 3:4 face, bleed and rounded clipping.'],
  ['Chun', '7z', 'Red dragon', 'approved', 'red-dragon-c-chapel-flame-approved.png', [115, 117, 857, 1253], 'Chapel Flame C: a pale-pink sculptural enclosure and a sweeping lemon-yellow central ribbon form 中 on vermilion. The lossless crop preserves the selected artwork, excludes the study label and surround, and uses the shared 3:4 face, bleed and rounded clipping.'],
  ['Man1', '1m', '1 characters', 'approved', 'one-characters-b-approved.png', [513, 205, 423, 630], 'Dance B: a sweeping yellow 一 above sculptural vermilion 萬 on deep purple. The lossless centre-tile crop excludes the study label and surrounding board, with the shared 3:4 face, bleed and rounded clipping.'],
  ['Man2', '2m', '2 characters', 'approved', 'two-characters-b-approved.png', [510, 199, 429, 658], 'Dance B: coral-pink 萬 at upper right above two sweeping ivory strokes forming 二 on deep purple. The reversed composition preserves the approved centre tile exactly, using the shared 3:4 face, bleed and rounded clipping.'],
  ['Man3', '3m', '3 characters', 'approved', 'three-characters-a-jazz-approved.png', [35, 121, 459, 709], 'Jazz A: yellow, ivory and pink strokes form 三 above a vermilion-and-pink 萬 with an oversized kicking stroke on deep purple. The lossless left-tile crop preserves the approved artwork, excludes the label and surround, and uses the shared 3:4 face, bleed and rounded clipping.'],
  ['Man4', '4m', '4 characters', 'approved', 'four-characters-c-ivory-approved.png', [0, 0, 1086, 1448], 'Interlock C, with the requested ivory-white outer enclosure of 四, two upright yellow interior strokes and a sweeping pink 萬 on deep purple. The former red section is ivory-white. The complete flat 3:4 artwork is preserved at native resolution, using the shared bleed and rounded clipping.'],
  ['Man5', '5m', '5 characters', 'approved', 'five-characters-dance-approved.png', [0, 0, 1086, 1448], 'Dance: mint and vermilion cut-paper strokes form 五 with yellow 萬 and a cobalt crescent on deep purple. Reinterpreted from the selected fifth alternative and approved for play. The complete source is preserved with the shared 3:4 face, bleed and rounded clipping.'],
  ['Man6', '6m', '6 characters', 'approved', 'six-characters-chasuble-approved.png', [0, 0, 1086, 1448], 'Chasuble: purple 六 cut through a yellow silhouette, with vermilion 萬 and teal offcuts. Reinterpreted from the selected fourth alternative and approved for play. The complete source is preserved with the shared 3:4 face, bleed and rounded clipping.'],
  ['Man7', '7m', '7 characters', 'approved', '06-character-compositions.png', [965, 207, 432, 630], 'Cut-out C: pink 萬 at upper left and oversized pale-yellow 七 below on deep purple.'],
  ['Man9', '9m', '9 characters', 'approved', 'nine-characters-jazz-approved.png', [0, 0, 1086, 1448], 'Jazz: the exact sixth alternative selected by Carl, with fragmented coral 九 over a tilted cobalt field, ivory 萬 at lower left and a yellow wedge on deep purple. The complete flat source is preserved with the shared 3:4 face, bleed and rounded clipping.'],
  ['Sou1', '1s', '1 bamboo', 'approved', 'one-bamboo-c-carnival-approved.png', [62, 28, 962, 1300], 'Carnival C: one dancing pink bird with an oversized green wing, curling purple tail and yellow accents on vermilion. The lossless crop preserves the approved artwork and excludes the study label and surround, using the shared 3:4 face, bleed and rounded clipping.'],
  ['Hatsu', '6z', 'Green dragon', 'approved', 'green-dragon-b-chapel-approved.png', [62, 24, 963, 1308], 'Chapel B: forest-green 發 cut through an organic mint-green silhouette, with an emerald accent on a forest-green field. Every visible part of the face stays in shades of green for All Green. The lossless crop preserves the selected source pixels, excludes the label and surround, and uses the shared 3:4 face, bleed and rounded clipping.'],
  ['Haku', '5z', 'White dragon', 'approved', '07-white-dragon-approved.png', [158, 150, 540, 752], 'Approved blend of C and C1: a quiet ivory face with separated abstract dragon shapes, revealed in pearly silver for dora.'],
];

function exportCrop(source, crop, png, svg, label) {
  const [x, y, width, height] = crop;
  // Mechanical lossless extraction only: keep the artwork's native dimensions.
  const raster = execFileSync('convert', [path.join(root, source), '-crop', `${width}x${height}+${x}+${y}`, '+repage', '-strip', 'PNG:-'], { maxBuffer: 16 * 1024 * 1024 });
  writeFileSync(path.join(out, png), raster, { flush: true });
  const { radius, bleed, preserveAspectRatio } = facePresentation;
  const verticalBleed = bleed * 4 / 3;
  const svgText = `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">${label} — Matisse study</title><defs><clipPath id="face"><rect width="300" height="400" rx="${radius}"/></clipPath></defs><image clip-path="url(#face)" x="${-bleed}" y="${-verticalBleed}" width="${300 + 2 * bleed}" height="${400 + 2 * verticalBleed}" preserveAspectRatio="${preserveAspectRatio}" href="data:image/png;base64,${raster.toString('base64')}"/></svg>\n`;
  writeFileSync(path.join(out, svg), svgText);
  return svgText;
}

const manifest = { version: 2, canvas: { width: 300, height: 400 }, facePresentation, tiles: [], placeholders: [] };
const previews = [];
for (const [name, tile, label, status, sourceName, crop, notes] of definitions) {
  const group = status === 'approved' ? 'approved' : 'concepts';
  const previousGroup = status === 'approved' ? 'concepts' : 'approved';
  // Keep only the current status path when a design is approved or reconsidered.
  for (const extension of ['png', 'svg']) rmSync(path.join(out, previousGroup, `${name}.${extension}`), { force: true });
  mkdirSync(path.join(out, group), { recursive: true });
  const source = `${sourceDir}/${sourceName}`;
  const [x, y, width, height] = crop;
  const png = `${group}/${name}.png`;
  const svg = `${group}/${name}.svg`;
  exportCrop(source, crop, png, svg, label);
  const entry = { name, tile, label, status, png, svg, source, sourceSha256: createHash('sha256').update(readFileSync(path.join(root, source))).digest('hex'), crop: { x, y, width, height }, notes };
  if (name === 'Haku') {
    const foilCrop = [749, 150, 540, 752];
    entry.foil = { png: 'approved/Haku-foil.png', svg: 'approved/Haku-foil.svg', crop: { x: foilCrop[0], y: foilCrop[1], width: foilCrop[2], height: foilCrop[3] } };
    exportCrop(source, foilCrop, entry.foil.png, entry.foil.svg, 'White dragon in the light');
  }
  manifest.tiles.push(entry);
  // Reuse the shipped faces instead of duplicating every raster inside the HTML.
  previews.push({ name, label, status, src: svg });
}
const approved = manifest.tiles.filter(tile => tile.status === 'approved').map(tile => tile.tile);
writeFileSync(path.join(root, 'web/src/lib/matisse-faces.js'), `// Generated by web/scripts/export-matisse-tiles.mjs.\nexport const MATISSE_APPROVED = Object.freeze(${JSON.stringify(approved)});\n`);
mkdirSync(path.join(out, 'placeholders'), { recursive: true });
for (const tile of TILE_TYPES) {
  const name = tileFile(tile);
  const svg = `placeholders/${name}.svg`;
  if (approved.includes(tile)) {
    rmSync(path.join(out, svg), { force: true });
    continue;
  }
  const label = tileWords(tile);
  const lines = label.split(' ');
  const spans = lines.map((line, index) => `<tspan x="150" y="${175 + index * 65}">${line}</tspan>`).join('');
  writeFileSync(path.join(out, svg), `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">${label}</title><rect width="300" height="400" rx="${facePresentation.radius}" fill="#f5f1e4"/><text text-anchor="middle" fill="#000" font-family="Arial, sans-serif" font-size="43" font-weight="500">${spans}</text></svg>\n`);
  manifest.placeholders.push({ tile, name, label, svg });
}
writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');

const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Matisse Mahjong — Tile Studies</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f6f3ea;color:#202823;font:16px/1.5 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:32px 20px 56px}h1{font-size:32px;line-height:1.1;margin:8px 0 16px;letter-spacing:-1px}h2{font-size:20px;margin:32px 0 16px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#587162}.intro{max-width:650px;color:#566157}button,select,input{font:inherit}label{display:inline-flex;gap:8px;align-items:center;margin:4px 16px 4px 0}select{padding:6px;border:1px solid #adb5a7;border-radius:6px;background:#fffefa}.scroll{overflow-x:auto;padding:10px 0 18px}.table{width:390px;padding:28px 12px;background:#163d35;border-radius:14px;box-shadow:inset 0 1px 0 #ffffff15}.rack{display:flex;gap:2px;align-items:center}.rack img{height:auto;flex:0 0 calc((100% - 26px)/14);width:calc((100% - 26px)/14);aspect-ratio:3/4;min-width:0;border-radius:2px}.size{font-size:13px;color:#687366}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(135px,1fr));gap:24px 20px}.card{text-align:center}.card img{width:120px;height:auto;aspect-ratio:3/4;display:block;margin:0 auto 8px}.card p{margin:3px 0;font-size:14px}.badge{font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:#55725d}.badge.concept{color:#925427}footer{font-size:13px;color:#687366;margin-top:40px;max-width:760px}a{color:inherit}
@media(max-width:520px){main{padding:24px 14px}.gallery{grid-template-columns:repeat(2,minmax(0,1fr));gap:20px 10px}h1{font-size:29px}}
</style></head><body><main>
<div class="eyebrow">Chapelle du Rosaire · Cut-paper studies</div><h1>Matisse Mahjong</h1>
<p class="intro">${approved.length} approved faces. Select Matisse under Options → Tile face in the game. The other ${manifest.placeholders.length} tile types show their names in black until their artwork is approved.</p>
<h2>A mixed hand</h2><label>Hand width <select id="width"><option value="390">390 px · compact</option><option value="844">844 px · landscape</option></select></label>
<div class="scroll"><div class="table" id="table"><div class="rack" id="rack" aria-label="Fourteen-tile visual sample"></div></div></div><p class="size" id="size"></p>
<h2>Approved faces</h2><div class="gallery" id="approved"></div>
<footer>This hand is a visual sample. The approved character tiles use Dance B for 1 and 2 characters, Jazz A for 3, Interlock C with an ivory-white 四 for 4, Dance for 5, Chasuble for 6, Cut-out C for 7, and Jazz for 9, all on deep purple. White dragon uses the approved C/C1 blend, with a pearly reveal when it is dora. Raw PNG crops retain their source resolution; each SVG provides the game's 300 × 400 canvas.</footer>
</main><script>
const tiles=${JSON.stringify(previews)};
const byName=Object.fromEntries(tiles.map(tile=>[tile.name,tile]));
const rack=document.getElementById('rack');
const table=document.getElementById('table');
function picture(tile){const image=document.createElement('img');image.src=tile.src;image.alt=tile.label;image.title=tile.label;image.width=300;image.height=400;return image}
for(const tile of tiles){const card=document.createElement('div');card.className='card';card.append(picture(tile));const label=document.createElement('p');label.textContent=tile.label;card.append(label);const badge=document.createElement('span');badge.className='badge';badge.textContent='Approved';card.append(badge);document.getElementById('approved').append(card)}
function updateSize(){const w=rack.firstElementChild?.getBoundingClientRect().width||0;document.getElementById('size').textContent=w.toFixed(1)+' × '+(w*4/3).toFixed(1)+' CSS px per tile · 14 tiles · scroll horizontally if needed; the preview is not scaled down.'}
function draw(){const names=['Pin8','Man2','Man3','Man4','Man5','Man6','Man7','Man9','Pin4','Pin6','Sou1','Ton','Nan','Pei'];rack.replaceChildren(...names.map(name=>picture(byName[name])));table.style.width=document.getElementById('width').value+'px';requestAnimationFrame(updateSize)}
document.getElementById('width').addEventListener('change',draw);new ResizeObserver(updateSize).observe(rack);draw();
</script></body></html>`;
writeFileSync(path.join(out, 'preview.html'), html);
console.log(`Exported ${approved.length} approved tile faces, ${manifest.placeholders.length} placeholders and a hand preview.`);
