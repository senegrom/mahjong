import { readFileSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

// Export the approved source pixels, without regenerating or redrawing them.
// Run from any directory: node web/scripts/export-matisse-tiles.mjs
const root = fileURLToPath(new URL('../../', import.meta.url));
const out = path.join(root, 'web/public/tiles/matisse');
const sourceDir = 'docs/design/matisse/studies';
const definitions = [
  ['Pin1', '1p', '1 dot', 'approved', '01-disk-and-red-dragon.png', [77, 232, 410, 554], 'Black disk and ivory rosette on yellow; direction B.'],
  ['Pin3', '3p', '3 dots', 'approved', '03-dots-and-bamboo.png', [77, 99, 604, 835], 'Blue, red and green rosettes on ivory.'],
  ['Pin5', '5p', '5 dots', 'approved', '02-five-dot-study.png', [77, 231, 410, 572], 'Five black and ivory rosettes on yellow.'],
  ['Sou8', '8s', '8 bamboo', 'approved', '03-dots-and-bamboo.png', [755, 99, 612, 835], 'Two sweeping fans of four jointed fronds.'],
  ['Ton', '1z', 'East wind', 'approved', '04-wind-calligraphy.png', [518, 225, 411, 579], 'Ribbon lettering B: blue with one red stroke on pale yellow.'],
  ['Chun', '7z', 'Red dragon', 'approved', '01-disk-and-red-dragon.png', [966, 232, 410, 554], 'Red cut-paper 中 on pink; direction B.'],
  ['Man7', '7m', '7 characters', 'concept', '05-characters-bird-green-dragon.png', [59, 210, 425, 630], 'Awaiting redesign: the stacked black 七 above vermilion 萬 was considered too restrained. Explore more expressive lettering and varied placement.'],
  ['Sou1', '1s', '1 bamboo', 'approved', '05-characters-bird-green-dragon.png', [510, 210, 426, 630], 'Blue and green cut-paper bird on a bamboo perch.'],
  ['Hatsu', '6z', 'Green dragon', 'approved', '05-characters-bird-green-dragon.png', [964, 210, 433, 630], 'Ivory 發 cut out of emerald green, using style C.'],
];

const manifest = { version: 1, canvas: { width: 300, height: 400 }, tiles: [] };
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
  // Mechanical lossless extraction only: keep the artwork's native dimensions.
  execFileSync('convert', [path.join(root, source), '-crop', `${width}x${height}+${x}+${y}`, '+repage', '-strip', path.join(out, png)]);
  const raster = readFileSync(path.join(out, png));
  const svgText = `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">${label} — Matisse study</title><defs><clipPath id="face"><rect width="300" height="400" rx="16"/></clipPath></defs><g clip-path="url(#face)"><rect width="300" height="400" fill="#f5f1e4"/><image x="0" y="0" width="300" height="400" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,${raster.toString('base64')}"/></g></svg>\n`;
  writeFileSync(path.join(out, svg), svgText);
  const entry = { name, tile, label, status, png, svg, source, sourceSha256: createHash('sha256').update(readFileSync(path.join(root, source))).digest('hex'), crop: { x, y, width, height }, notes };
  manifest.tiles.push(entry);
  previews.push({ name, label, status, src: `data:image/svg+xml;base64,${Buffer.from(svgText).toString('base64')}` });
}
writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');

const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Matisse Mahjong — Tile Studies</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f6f3ea;color:#202823;font:16px/1.5 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:32px 20px 56px}h1{font-size:32px;line-height:1.1;margin:8px 0 16px;letter-spacing:-1px}h2{font-size:20px;margin:32px 0 16px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#587162}.intro{max-width:650px;color:#566157}button,select,input{font:inherit}label{display:inline-flex;gap:8px;align-items:center;margin:4px 16px 4px 0}select{padding:6px;border:1px solid #adb5a7;border-radius:6px;background:#fffefa}.scroll{overflow-x:auto;padding:10px 0 18px}.table{width:390px;padding:28px 12px;background:#163d35;border-radius:14px;box-shadow:inset 0 1px 0 #ffffff15}.rack{display:flex;gap:2px;align-items:center}.rack img{height:auto;flex:0 0 calc((100% - 26px)/14);width:calc((100% - 26px)/14);aspect-ratio:3/4;min-width:0;border-radius:2px}.size{font-size:13px;color:#687366}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(135px,1fr));gap:24px 20px}.card{text-align:center}.card img{width:120px;height:auto;aspect-ratio:3/4;display:block;margin:0 auto 8px}.card p{margin:3px 0;font-size:14px}.badge{font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:#55725d}.badge.concept{color:#925427}footer{font-size:13px;color:#687366;margin-top:40px;max-width:760px}a{color:inherit}
@media(max-width:520px){main{padding:24px 14px}.gallery{grid-template-columns:repeat(2,minmax(0,1fr));gap:20px 10px}h1{font-size:29px}}
</style></head><body><main>
<div class="eyebrow">Chapelle du Rosaire · Cut-paper studies</div><h1>Matisse Mahjong</h1>
<p class="intro">Eight approved faces and one concept awaiting redesign. Rosettes, sweeping bamboo, a cut-paper bird and expressive lettering share an ivory tile body.</p>
<h2>A mixed hand</h2><label>Hand width <select id="width"><option value="390">390 px · compact</option><option value="844">844 px · landscape</option></select></label><label><input id="concepts" type="checkbox"> Include character concept</label>
<div class="scroll"><div class="table" id="table"><div class="rack" id="rack" aria-label="Fourteen-tile visual sample"></div></div></div><p class="size" id="size"></p>
<h2>Approved faces</h2><div class="gallery" id="approved"></div>
<h2>Character concept awaiting redesign</h2><div class="gallery" id="new"></div>
<footer>This preview uses the exported source artwork, with no stretching of the symbols. The hand is a visual sample, not a playable game state. The bird and green dragon are approved. The character concept remains for reference while more expressive lettering and varied placement are explored. Raw PNG crops retain their source resolution; each SVG provides the game's 300 × 400 canvas.</footer>
</main><script>
const tiles=${JSON.stringify(previews)};
const byName=Object.fromEntries(tiles.map(tile=>[tile.name,tile]));
const rack=document.getElementById('rack');
const table=document.getElementById('table');
function picture(tile){const image=document.createElement('img');image.src=tile.src;image.alt=tile.label;image.title=tile.label;image.width=300;image.height=400;return image}
for(const tile of tiles){const card=document.createElement('div');card.className='card';card.append(picture(tile));const label=document.createElement('p');label.textContent=tile.label;card.append(label);const badge=document.createElement('span');badge.className='badge '+tile.status;badge.textContent=tile.status==='approved'?'Approved':'Awaiting redesign';card.append(badge);document.getElementById(tile.status==='approved'?'approved':'new').append(card)}
function updateSize(){const w=rack.firstElementChild?.getBoundingClientRect().width||0;document.getElementById('size').textContent=w.toFixed(1)+' × '+(w*4/3).toFixed(1)+' CSS px per tile · 14 tiles · scroll horizontally if needed; the preview is not scaled down.'}
function draw(){const names=document.getElementById('concepts').checked?['Man7','Man7','Pin1','Pin3','Pin3','Pin5','Pin5','Sou1','Sou1','Sou8','Sou8','Ton','Chun','Hatsu']:['Pin1','Pin1','Pin3','Pin3','Pin5','Pin5','Sou1','Sou1','Sou8','Sou8','Ton','Ton','Chun','Hatsu'];rack.replaceChildren(...names.map(name=>picture(byName[name])));table.style.width=document.getElementById('width').value+'px';requestAnimationFrame(updateSize)}
document.getElementById('concepts').addEventListener('change',draw);document.getElementById('width').addEventListener('change',draw);new ResizeObserver(updateSize).observe(rack);draw();
</script></body></html>`;
writeFileSync(path.join(out, 'preview.html'), html);
console.log(`Exported ${definitions.length} tile faces and a standalone hand preview.`);
