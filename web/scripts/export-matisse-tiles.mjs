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
  ['Pin3', '3p', '3 dots', 'approved', '03-dots-and-bamboo.png', [77, 99, 604, 835], 'Blue, red and green rosettes on ivory.'],
  ['Pin5', '5p', '5 dots', 'approved', '02-five-dot-study.png', [77, 231, 410, 572], 'Five black and ivory rosettes on yellow.'],
  ['Sou2', '2s', '2 bamboo', 'approved', '08-two-bamboo-approved.png', [495, 174, 465, 769], 'Dance B: two sweeping green bamboo forms on a pale mint field. All printed artwork stays green for All Green; multiple shades are intentional.'],
  ['Sou5', '5s', '5 bamboo', 'approved', '09-five-bamboo-approved.png', [963, 256, 448, 640], 'Fan C: four pale-green bamboo sprigs and a bold red centre on a deep forest-green field. Exactly five complete sprigs; the red accent distinguishes this tile from All Green tiles.'],
  ['Sou6', '6s', '6 bamboo', 'approved', 'six-bamboo-c-approved.png', [1036, 172, 472, 671], 'Approved C: six jointed cut-paper bamboo forms in three staggered pairs on pale sage. All printed artwork stays green for All Green; multiple shades are intentional.'],
  ['Sou7', '7s', '7 bamboo', 'approved', '09-seven-bamboo-approved.png', [1037, 115, 468, 719], 'Cut-out C: one coral and six green abstract forms circling an open centre on butter yellow. Seven separated silhouettes preserve the tile count.'],
  ['Sou8', '8s', '8 bamboo', 'approved', '03-dots-and-bamboo.png', [755, 99, 612, 835], 'Two sweeping fans of four jointed fronds.'],
  ['Ton', '1z', 'East wind', 'approved', '04-wind-calligraphy.png', [518, 225, 411, 579], 'Ribbon lettering B: blue with one red stroke on pale yellow.'],
  ['Chun', '7z', 'Red dragon', 'approved', '01-disk-and-red-dragon.png', [966, 232, 410, 554], 'Red cut-paper 中 on pink; direction B.'],
  ['Man7', '7m', '7 characters', 'approved', '06-character-compositions.png', [965, 207, 432, 630], 'Cut-out C: pink 萬 at upper left and oversized pale-yellow 七 below on deep purple.'],
  ['Sou1', '1s', '1 bamboo', 'approved', '05-characters-bird-green-dragon.png', [510, 210, 426, 630], 'Blue and green cut-paper bird on a bamboo perch.'],
  ['Hatsu', '6z', 'Green dragon', 'approved', '05-characters-bird-green-dragon.png', [964, 210, 433, 630], 'Ivory 發 cut out of emerald green, using style C.'],
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
  const svgText = exportCrop(source, crop, png, svg, label);
  const entry = { name, tile, label, status, png, svg, source, sourceSha256: createHash('sha256').update(readFileSync(path.join(root, source))).digest('hex'), crop: { x, y, width, height }, notes };
  if (name === 'Haku') {
    const foilCrop = [749, 150, 540, 752];
    entry.foil = { png: 'approved/Haku-foil.png', svg: 'approved/Haku-foil.svg', crop: { x: foilCrop[0], y: foilCrop[1], width: foilCrop[2], height: foilCrop[3] } };
    exportCrop(source, foilCrop, entry.foil.png, entry.foil.svg, 'White dragon in the light');
  }
  manifest.tiles.push(entry);
  previews.push({ name, label, status, src: `data:image/svg+xml;base64,${Buffer.from(svgText).toString('base64')}` });
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
<footer>This hand is a visual sample. The approved Cut-out composition is used for 7 characters. Dance lettering on deep purple is a direction for future character tiles. White dragon uses the approved C/C1 blend, with a pearly reveal when it is dora. Raw PNG crops retain their source resolution; each SVG provides the game's 300 × 400 canvas.</footer>
</main><script>
const tiles=${JSON.stringify(previews)};
const byName=Object.fromEntries(tiles.map(tile=>[tile.name,tile]));
const rack=document.getElementById('rack');
const table=document.getElementById('table');
function picture(tile){const image=document.createElement('img');image.src=tile.src;image.alt=tile.label;image.title=tile.label;image.width=300;image.height=400;return image}
for(const tile of tiles){const card=document.createElement('div');card.className='card';card.append(picture(tile));const label=document.createElement('p');label.textContent=tile.label;card.append(label);const badge=document.createElement('span');badge.className='badge';badge.textContent='Approved';card.append(badge);document.getElementById('approved').append(card)}
function updateSize(){const w=rack.firstElementChild?.getBoundingClientRect().width||0;document.getElementById('size').textContent=w.toFixed(1)+' × '+(w*4/3).toFixed(1)+' CSS px per tile · 14 tiles · scroll horizontally if needed; the preview is not scaled down.'}
function draw(){const names=['Man7','Pin1','Pin3','Pin5','Haku','Sou1','Sou2','Sou5','Sou6','Sou7','Sou8','Ton','Chun','Hatsu'];rack.replaceChildren(...names.map(name=>picture(byName[name])));table.style.width=document.getElementById('width').value+'px';requestAnimationFrame(updateSize)}
document.getElementById('width').addEventListener('change',draw);new ResizeObserver(updateSize).observe(rack);draw();
</script></body></html>`;
writeFileSync(path.join(out, 'preview.html'), html);
console.log(`Exported ${approved.length} approved tile faces, ${manifest.placeholders.length} placeholders and a standalone hand preview.`);
