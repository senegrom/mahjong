/** Render the real Tile component with its production CSS. The temporary
 * fixture never ships. Test mask motion (not just opacity), asset loading,
 * rotated geometry, prop changes, hidden tiles and reduced motion. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { DALI_APPROVED } from '../src/lib/dali-faces.js';
import { TILE_TYPES, tileShorthand, tileWords } from '../src/lib/tiles.js';

const web = fileURLToPath(new URL('../', import.meta.url));
const temporary = await mkdtemp(resolve(web, '.tile-effects-'));
const out = resolve(temporary, 'dist');
const evidence = resolve(web, 'test-results');
const { check, report } = browserChecks();
let browser, server;
try {
  await mkdir(evidence, {recursive:true});
  await writeFile(resolve(temporary, 'index.html'), '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Tile effects regression</title></head><body><div id="app"></div><script type="module" src="./main.js"></script></body></html>');
  await writeFile(resolve(temporary, 'main.js'), 'import { mount } from "svelte"; import Fixture from "./Fixture.svelte"; mount(Fixture, {target:document.getElementById("app")});');
  // A painted set for the tiles inside it, Matisse unless it names another,
  // whichever set the rest of the page shows.
  await writeFile(resolve(temporary, 'Painted.svelte'), '<script>import { setContext } from "svelte"; import { TILE_FACE_CONTEXT } from "../src/lib/tile-faces.js"; let { children, face = "matisse" } = $props(); setContext(TILE_FACE_CONTEXT, () => face);</script>{@render children()}');
  await writeFile(resolve(temporary, 'Fixture.svelte'), `<script>
import Painted from './Painted.svelte';
import Tile from '../src/lib/Tile.svelte';
import { setContext } from 'svelte';
import { TILE_FACE_CONTEXT } from '../src/lib/tile-faces.js';
import { DALI_APPROVED } from '../src/lib/dali-faces.js';
import { MATISSE_APPROVED } from '../src/lib/matisse-faces.js';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { TILE_TYPES } from '../src/lib/tiles.js';
let face = $state('classic');
setContext(TILE_FACE_CONTEXT, () => face);
// Tiles Dalí has not painted yet, whichever they are when this runs, at the
// sizes that show both words, a number over its suit's letter, an honour's
// first word, and an honour's letters, upright and turned.
const unpainted = TILE_TYPES.filter(tile => !DALI_APPROVED.includes(tile));
const suited = unpainted.find(tile => tile[1] !== 'z'), honour = unpainted.find(tile => tile[1] === 'z');
const named = [
  ['named', {tile:suited}], ['named-turned', {tile:honour,rotated:true,dora:true,size:'small'}],
  ['named-row', {tile:honour,size:'tiny',fromDraw:true,claimed:true}], ['named-small', {tile:suited,size:'small',claimed:true}],
  ['named-suited-turned', {tile:suited,rotated:true,claimed:true,size:'tiny'}],
].filter(([, props]) => props.tile);
const phones = [['named-phone', honour], ['named-suited-phone', suited]].filter(([, tile]) => tile);
let tile = $state('1m'); let marked = $state(true); let clicks = $state(0);
const cases = [
  ['blank', {tile:'5z'}], ['dora', {tile:'5z',dora:true}],
  ['small', {tile:'5z',dora:true,size:'small'}], ['tiny', {tile:'5z',dora:true,size:'tiny'}],
  ['rotated', {tile:'5z',dora:true,rotated:true}], ['hidden', {tile:'5z',dora:true,facedown:true}],
  ['other', {tile:'7z',dora:true}], ['muted', {tile:'5z',dora:true,disabled:true,onclick:()=>{}}],
  ['kept', {tile:'5z',size:'small'}], ['thrown', {tile:'5z',size:'small',fromDraw:true}],
  ['taken', {tile:'5z',size:'small',claimed:true}], ['both', {tile:'5z',size:'small',claimed:true,fromDraw:true}],
  ['taken-dora', {tile:'5z',size:'small',claimed:true,fromDraw:true,dora:true,rotated:true}],
  ['tiny-taken', {tile:'5z',size:'tiny',claimed:true}]
];
// Claimed tiles at every size the app draws a row at, upright and turned for
// riichi, each nudged by a fraction of a pixel as a row's layout may place it.
const nudged = [[54,'tiny'],[46,'small'],[52,'small'],[60,'small'],[46,'tiny'],[28,'tiny'],[23,'tiny']]
  .flatMap(([width,size]) => [0,0.3,0.55,0.8].flatMap(shift => [false,true].map(rotated => ({width,size,shift,rotated}))));
// The same, turned and unclaimed, in a set painted to every edge.
const turned = nudged.filter(({rotated}) => rotated);
// The greenest paintings of each artist's set, whose own edges are nearly the
// border's green, claimed at the sizes of the table's rows and a phone's,
// upright and turned for riichi.
const approved = {matisse:MATISSE_APPROVED, dali:DALI_APPROVED, 'van-gogh':VAN_GOGH_APPROVED};
const greens = [['matisse',['4s','6s','8s']],['dali',['2s','4s','8s']],['van-gogh',['2s','3s','4s','5s','6s']]]
  .flatMap(([set,tiles]) => tiles.filter(tile => approved[set].includes(tile)).map(tile => ({set,tile})));
const greenSizes = [[54,'tiny'],[46,'small'],[28,'tiny'],[23,'tiny']];
// Shown only for their own check, so their pictures add nothing to the
// others' screenshots or the time a change of pixel density takes to draw.
let greensShown = $state(false);
</script>
<h1>White dragon · dora foil</h1>
<div class="samples">{#each cases as [id, props]}<section id={id}><p>{id}</p><Tile {...props}/></section>{/each}<section id="phone" style="--tile-width:23px"><p>phone</p><Tile tile="5z" size="tiny" claimed/></section></div>
<div class="samples" id="nudged">{#each nudged as {width,size,shift,rotated}}<span style="--tile-width:{width}px;margin:{shift}px 0 0 {shift}px"><Tile tile="5z" {size} {rotated} claimed/></span>{/each}</div>
<Painted><div class="samples" id="painted">{#each turned as {width,size,shift}}<span style="--tile-width:{width}px;margin:{shift}px 0 0 {shift}px"><Tile tile="1p" {size} rotated/></span>{/each}</div></Painted>
<section id="dynamic"><Tile {tile} dora={marked} onclick={()=>clicks++}/></section>
<button id="identity" onclick={()=>tile=tile==='5z'?'1m':'5z'}>Change tile</button>
<button id="mark" onclick={()=>marked=!marked}>Toggle dora / hints</button><output>{clicks}</output>
<button id="faces" onclick={()=>face=face==='classic'?'matisse':'classic'}>Change tile face</button>
<button id="unpainted" onclick={()=>face='dali'}>Show a set still being painted</button>
<div class="samples">{#each named as [id, props]}<section id={id}><p>{id}</p><Tile {...props}/></section>{/each}{#each phones as [id, tile]}<section id={id} style="--tile-width:28px"><p>{id}</p><Tile {tile} size="tiny" claimed/></section>{/each}</div>
<button id="greens-shown" onclick={()=>greensShown=!greensShown}>Show the green paintings, claimed</button>
{#if greensShown}<div class="samples" id="greens">{#each greens as {set,tile}}<Painted face={set}>{#each greenSizes as [width,size]}{#each [false,true] as rotated}<span data-face="{set} {tile}" style="--tile-width:{width}px"><Tile {tile} {size} {rotated} claimed/></span>{/each}{/each}</Painted>{/each}</div>{/if}
<style>:global(body){margin:30px;background:#173e35;color:#fff;font:16px system-ui;--tile-width:60px;--ivory:#fffaf0;} .samples{display:flex;gap:26px;align-items:start;flex-wrap:wrap;margin-bottom:40px} section{min-width:70px} #dynamic{margin:25px 0} button{margin:10px;padding:10px}</style>`);
  await build({configFile:false,root:temporary,base:'/mahjong/',publicDir:false,plugins:[svelte()],logLevel:'warn',build:{outDir:out,target:'es2022'}});
  if (!process.argv.includes('--build-only')) {
    server = createServer(createFixtureHandler({root:out,publicRoot:resolve(web,'public')}));
    await new Promise(done=>server.listen(0,'127.0.0.1',done));
    browser = await launchChrome();
    const page = await browser.newPage(); const errors = [];
    page.on('pageerror', error=>errors.push(error.message));
    await page.setViewport({width:1100,height:700,deviceScaleFactor:2});
    await page.emulateMediaFeatures([{name:'prefers-reduced-motion',value:'no-preference'}]);
    await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/`,{waitUntil:'networkidle0',timeout:15000});
    const freeze = time=>page.evaluate(t=>{for(const animation of document.getAnimations()){animation.pause();animation.currentTime=t;}},time);
    await check('approved artwork loads from the project-scoped hashed URL',async()=>{
      const asset = await page.$eval('#dora .haku-dragon-reveal',async el=>{
        const url=getComputedStyle(el).backgroundImage.slice(5,-2); const image=new Image();image.src=url;await image.decode();
        return {path:new URL(url).pathname,width:image.naturalWidth,height:image.naturalHeight};
      });
      assert.match(asset.path,/^\/mahjong\/assets\/white-dragon-.*\.webp$/);
      assert.deepEqual([asset.width,asset.height],[192,256]);
    });
    await check('only face-up white-dragon dora tiles receive the dragon',async()=>{
      for(const id of ['blank','other','hidden']) assert.equal(await page.$(`#${id} .haku-dragon-reveal`),null);
      assert.equal(await page.$('#hidden .foil'),null);
      for(const id of ['dora','small','tiny','rotated','muted']) assert.ok(await page.$(`#${id} .haku-dragon-reveal`));
      assert.match(await page.$eval('#dora .tile',el=>el.getAttribute('aria-label')),/^white dragon, dora$/);
      assert.equal(await page.$eval('#dora .haku-dragon-reveal',el=>el.getAttribute('aria-hidden')),'true');
    });
    await check('mask travels with the shine while the dragon artwork stays still',async()=>{
      const positions=[];
      for(const time of [0,1250,2500,3750,4999]) {
        await freeze(time);
        const style=await page.$eval('#dora .face',el=>{
          const dragon=getComputedStyle(el.querySelector('.haku-dragon-reveal')),foil=getComputedStyle(el.querySelector('.foil'));
          return {mask:dragon.maskPosition,shine:foil.backgroundPosition,art:dragon.backgroundPosition};
        });
        assert.equal(style.mask,style.shine);assert.equal(style.art,'50% 50%');positions.push(style.mask);
      }
      assert.equal(new Set(positions).size,5);
    });
    await check('tile becomes blank at both endpoints and reveals actual dragon pixels mid-pass',async()=>{
      const foil=await page.addStyleTag({content:'.foil { visibility: hidden !important; }'});
      const tile=await page.$('#dora .face');const frames=[];
      for(const time of [0,2500,4999]) {await freeze(time);frames.push(Buffer.from(await tile.screenshot()).toString('base64'));}
      const difference=await page.evaluate(async frames=>{
        const pixels=await Promise.all(frames.map(async data=>{
          const bytes=Uint8Array.from(atob(data),c=>c.charCodeAt(0));const image=await createImageBitmap(new Blob([bytes],{type:'image/png'}));
          const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
          const context=canvas.getContext('2d');context.drawImage(image,0,0);return context.getImageData(0,0,image.width,image.height).data;
        }));
        return pixels.slice(1).map(p=>{let n=0;for(let i=0;i<p.length;i+=4)if(Math.abs(p[i]-pixels[0][i])+Math.abs(p[i+1]-pixels[0][i+1])+Math.abs(p[i+2]-pixels[0][i+2])>3)n++;return n;});
      },frames);
      await foil.dispose();await page.evaluate(()=>{for(const style of document.querySelectorAll('style'))if(style.textContent.includes('visibility: hidden !important'))style.remove();});
      assert.ok(difference[0]>100,`Mid-pass changed ${difference[0]} pixels`);
      assert.ok(difference[1]<10,`Off-face endpoint changed ${difference[1]} pixels`);
    });
    await check('artwork and shine rotate with the face and do not enlarge tile layout',async()=>{
      for(const id of ['dora','small','tiny','rotated','taken','taken-dora','phone']) {
        const [tile,face,...painted]=await page.$eval(`#${id} .tile`,el=>[el,el.querySelector('.face'),...el.querySelectorAll('.face > img,.haku-dragon-reveal,.foil')].map(e=>{const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height];}));
        for(let n=0;n<4;n++)assert.ok(Math.abs(face[n]-tile[n])<0.1,`${id}: the face fills the tile's box`);
        // A picture turned with its tile spares a pixel past each edge of
        // the face, which clips it; an upright one fits the face exactly.
        const spare=id.includes('rotated')||id==='taken-dora'?1:0;
        for(const [x,y,width,height] of painted) {
          assert.ok(Math.abs(x+width/2-(face[0]+face[2]/2))<0.1&&Math.abs(y+height/2-(face[1]+face[3]/2))<0.1,`${id}: the picture and shine are centred on the face`);
          assert.ok(Math.abs(width-face[2]-2*spare)<0.1&&Math.abs(height-face[3]-2*spare)<0.1,`${id}: the picture and shine cover the face: ${width} by ${height}`);
        }
      }
    });
    await check('a discard from the draw is darker in its own colours; a claimed one has a solid dark green border inside its edge; both combine',async()=>{
      const border=[14,110,51], solid=`solid rgb(${border.join(', ')})`, shade='rgba(0, 0, 0, 0.28)';
      // The hairline inside the border is the fixture's ivory.
      const line='rgb(255, 250, 240) 0px 0px 0px 1px inset';
      // Both marks are one layer over the face: its shade, and its border with the hairline inside it.
      const marks=await page.evaluate(()=>['kept','thrown','taken','both'].map(id=>{
        const tile=document.querySelector(`#${id} .tile`), face=tile.querySelector('.face'), layer=getComputedStyle(face,'::after');
        return [getComputedStyle(tile).opacity,getComputedStyle(face).filter,getComputedStyle(tile,'::after').content,
          layer.content==='none'?'none':layer.backgroundColor,layer.content==='none'||layer.borderTopStyle==='none'?'none':`${layer.borderTopStyle} ${layer.borderTopColor}`,
          layer.content==='none'?'none':layer.boxShadow];
      }));
      assert.deepEqual(marks,[['1','none','none','none','none','none'],['1','none','none',shade,'none','none'],
        ['1','none','none','rgba(0, 0, 0, 0)',solid,line],['1','none','none',shade,solid,line]]);
      // The border keeps to the tile's proportions: one pixel on a phone's
      // smallest row, two on the table's and three on the largest.
      const [phone,tiny,small]=await page.evaluate(()=>['phone','tiny-taken','taken'].map(id=>
        parseFloat(getComputedStyle(document.querySelector(`#${id} .face`),'::after').borderTopWidth)));
      assert.ok(phone>=1&&phone<1.5&&tiny>=2&&tiny<3&&small===3,`border widths: ${[phone,tiny,small]}`);
      // Read the actual pixels: the middle of each blank face, the felt
      // beside the tile, and one pixel above the tile's top edge (where a
      // ring shows), one below it (where the border is) and one to its left.
      await freeze(0);
      const pixels={};
      for(const id of ['kept','thrown','taken','both','taken-dora']) {
        const box=await (await page.$(`#${id} .tile`)).boundingBox(), margin=6;
        const clip={x:box.x-margin,y:box.y-margin,width:box.width+2*margin,height:box.height+2*margin};
        const data=await page.screenshot({clip,encoding:'base64'});
        pixels[id]=await page.evaluate(async(data,width,margin)=>{
          const bytes=Uint8Array.from(atob(data),c=>c.charCodeAt(0));const image=await createImageBitmap(new Blob([bytes],{type:'image/png'}));
          const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
          const context=canvas.getContext('2d');context.drawImage(image,0,0);
          const scale=image.width/width, at=(x,y)=>[...context.getImageData(Math.round(x*scale),Math.round(y*scale),1,1).data.slice(0,3)];
          // In CSS pixels: the middle of the face, a corner of felt, one pixel above the tile's top
          // edge and one below it, and one pixel to the left of the tile halfway down.
          const height=image.height/scale;
          return {face:at(width/2,height/2),felt:at(1,1),ring:at(width/2,margin-1),edge:at(width/2,margin+1),beside:at(margin-1,height/2)};
        },data,clip.width,margin);
      }
      const near=(actual,expected,label)=>actual.forEach((value,channel)=>assert.ok(Math.abs(value-expected[channel])<=6,`${label}: ${actual} against ${expected.map(Math.round)}`));
      const luminance=rgb=>rgb.map(x=>x/255).map(x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4).reduce((sum,x,i)=>sum+x*[.2126,.7152,.0722][i],0);
      const contrast=(a,b)=>{const x=luminance(a),y=luminance(b);return (Math.max(x,y)+.05)/(Math.min(x,y)+.05);};
      const {kept,thrown,taken,both}=pixels, dora=pixels['taken-dora'];
      near(thrown.face,kept.face.map(value=>value*0.72),'thrown from the draw: the same colours, darker');
      near(taken.face,kept.face,'claimed: the face keeps its colours');
      near(both.face,thrown.face,'claimed and from the draw: only the shade on the face');
      near(taken.edge,border,'claimed: the border lies inside the tile\'s edge');
      near(both.edge,border,'the shade leaves the border as it is');
      near(taken.beside,kept.beside,'the border stays inside the tile: the felt beside it is unchanged');
      // The border stands out from what it lies between: the felt, and the face plain or shaded.
      const ratios=[kept.felt,kept.face,thrown.face].map(colour=>contrast(border,colour));
      assert.ok(ratios[0]>=1.7&&ratios[1]>=4.5&&ratios[2]>=2.5,`border contrast with felt, face and shaded face: ${ratios.map(ratio=>ratio.toFixed(2))}`);
      // A claimed dora turned for riichi: the ring outside, the border inside it, both turned.
      assert.ok(dora.ring[0]>dora.ring[1]+40&&dora.ring[0]>dora.ring[2]+40,`the dora ring still shows red: ${dora.ring}`);
      near(dora.edge,border,'the border lies inside the ring, along the turned tile');
      near(dora.face,both.face,'the ring does not tint a claimed face');
    });
    await check('a claimed tile\'s border closes over the rim of its face, upright or turned, at any size, position and pixel density',async()=>{
      // The boxes agree to the fraction, so only the pixels can show this: a
      // face turned whole once fell between the device's pixels and showed
      // its rim past the border. Going in from the felt across the straight
      // part of each side, the first thing met must be the border, never the face.
      const faults=[];
      try {
        for(const scale of [1,2,3]) {
          await page.setViewport({width:1100,height:700,deviceScaleFactor:scale});
          await page.evaluate(()=>new Promise(done=>{scrollTo(0,0);requestAnimationFrame(()=>requestAnimationFrame(done));}));
          const {clip,boxes}=await page.$eval('#nudged',section=>{
            const box=element=>{const r=element.getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height};};
            return {clip:box(section),boxes:[...section.querySelectorAll('.tile')].map(tile=>({...box(tile.querySelector('.face')),
              radius:parseFloat(getComputedStyle(tile.querySelector('.face')).borderTopLeftRadius),
              label:`${tile.parentElement.getAttribute('style')} ${tile.classList.contains('rotated')?'turned':'upright'}`}))};
          });
          const margin=6, area={x:clip.x-margin,y:clip.y-margin,width:clip.width+2*margin,height:clip.height+2*margin};
          const data=await page.screenshot({clip:area,encoding:'base64',captureBeyondViewport:true});
          await writeFile(resolve(evidence,`claimed-borders-x${scale}.png`),Buffer.from(data,'base64'));
          faults.push(...await page.evaluate(async(data,area,boxes,scale)=>{
            const bytes=Uint8Array.from(atob(data),c=>c.charCodeAt(0));const image=await createImageBitmap(new Blob([bytes],{type:'image/png'}));
            const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
            const context=canvas.getContext('2d');context.drawImage(image,0,0);
            const pixels=context.getImageData(0,0,image.width,image.height).data;
            const at=(x,y)=>{const i=(y*image.width+x)*4;return [pixels[i],pixels[i+1],pixels[i+2]];};
            // The border, alone or blended with the felt; the face, ivory alone or blended with the felt.
            const border=([r,g,b])=>g-r>=40&&g-b>=20&&r<90, face=([r,g,b])=>r>110&&g>110&&b>100;
            const reach=Math.round(4*scale);
            return boxes.flatMap(({x,y,width,height,radius,label})=>{
              const [left,top,right,bottom]=[x-area.x,y-area.y,x+width-area.x,y+height-area.y].map(value=>Math.round(value*scale));
              // Every device pixel along the straight part of a side, clear of the rounded corners.
              const corner=Math.ceil(radius*scale)+1;
              const across=(from,to)=>{const start=from+corner,span=to-corner-start;return span>0?Array.from({length:span},(_,n)=>start+n):[Math.round((from+to)/2)];};
              // For each side, the lines that cross it and the step inwards along them.
              const sides={top:across(left,right).map(u=>[u,top,0,1]),bottom:across(left,right).map(u=>[u,bottom-1,0,-1]),
                left:across(top,bottom).map(v=>[left,v,1,0]),right:across(top,bottom).map(v=>[right-1,v,-1,0])};
              return Object.entries(sides).flatMap(([side,lines])=>{
                const met=lines.map(([u,v,du,dv])=>{
                  for(let n=-reach;n<=reach;n++){const colour=at(u+n*du,v+n*dv);if(border(colour))return 'border';if(face(colour))return `face ${colour}`;}
                  return 'nothing';
                });
                const wrong=met.filter(what=>what!=='border');
                return wrong.length?[`x${scale} ${label} ${side}: ${wrong.length}/${met.length} lines meet ${wrong[0]} first`]:[];
              });
            });
          },data,area,boxes,scale));
        }
      } finally {
        await page.setViewport({width:1100,height:700,deviceScaleFactor:2});
      }
      assert.deepEqual(faults,[]);
    });
    await check('a hairline of ivory parts a claimed tile\'s border from a painting of its own green, upright or turned, at any size and pixel density',async()=>{
      // The bamboo of every artist's set is painted to its edges in nearly the
      // border's green, and the border alone left such a tile looking
      // unclaimed. Going in from the felt across the straight part of each
      // side, the border must come first, then the hairline, then the
      // painting, and the mark must stand apart from the painting just inside
      // it: by 30 or more in CIE76 delta E, through whichever of its two
      // colours the painting is further from. The border alone managed about
      // 20 on these faces, and stands about 70 apart from an ivory one.
      const green=[14,110,51], ivory=[255,250,240], faults=[], found=[];
      await page.click('#greens-shown');
      await page.waitForFunction(()=>{const images=[...document.querySelectorAll('#greens img')];return images.length&&images.every(image=>image.complete&&image.naturalWidth>0);},{timeout:30000});
      try {
        for(const scale of [1,2,3]) {
          await page.setViewport({width:1100,height:700,deviceScaleFactor:scale});
          await page.evaluate(async()=>{
            scrollTo(0,0);
            await Promise.all([...document.querySelectorAll('#greens img')].map(image=>image.decode().catch(()=>null)));
            await new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)));
          });
          const {clip,boxes}=await page.$eval('#greens',section=>{
            const box=element=>{const r=element.getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height};};
            return {clip:box(section),boxes:[...section.querySelectorAll('.tile')].map(tile=>{
              const face=tile.querySelector('.face'), holder=tile.parentElement;
              return {...box(face),radius:parseFloat(getComputedStyle(face).borderTopLeftRadius),border:parseFloat(getComputedStyle(face,'::after').borderTopWidth),
                images:face.querySelectorAll('img').length,
                label:`${holder.dataset.face} at ${holder.style.getPropertyValue('--tile-width')} ${tile.classList.contains('small')?'small':'tiny'} ${tile.classList.contains('rotated')?'turned':'upright'}`};
            })};
          });
          const margin=6, area={x:clip.x-margin,y:clip.y-margin,width:clip.width+2*margin,height:clip.height+2*margin};
          const data=await page.screenshot({clip:area,encoding:'base64',captureBeyondViewport:true});
          await writeFile(resolve(evidence,`claimed-greens-x${scale}.png`),Buffer.from(data,'base64'));
          const measured=await page.evaluate(async(data,area,boxes,scale,green,ivory)=>{
            const bytes=Uint8Array.from(atob(data),c=>c.charCodeAt(0));const image=await createImageBitmap(new Blob([bytes],{type:'image/png'}));
            const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
            const context=canvas.getContext('2d');context.drawImage(image,0,0);
            const pixels=context.getImageData(0,0,image.width,image.height).data;
            const at=(x,y)=>{const i=(y*image.width+x)*4;return [pixels[i],pixels[i+1],pixels[i+2]];};
            // CIE76 delta E, from sRGB through XYZ (D65) to Lab.
            const lab=rgb=>{
              const [r,g,b]=rgb.map(v=>v/255).map(c=>c<=.04045?c/12.92:((c+.055)/1.055)**2.4);
              const f=t=>t>(6/29)**3?Math.cbrt(t):t/(3*(6/29)**2)+4/29;
              const x=f((.4124564*r+.3575761*g+.1804375*b)/.95047),y=f(.2126729*r+.7151522*g+.072175*b),z=f((.0193339*r+.119192*g+.9503041*b)/1.08883);
              return [116*y-16,500*(x-y),200*(y-z)];
            };
            const de=(a,b)=>{const p=lab(a),q=lab(b);return Math.hypot(p[0]-q[0],p[1]-q[1],p[2]-q[2]);};
            return boxes.map(({x,y,width,height,radius,border,images,label})=>{
              // The box as reported can sit a device pixel or two off the one
              // painted, so each line finds the tile's edge in the pixels, and
              // keeps clear of the rounded corners by as much again.
              const [left,top,right,bottom]=[x-area.x,y-area.y,x+width-area.x,y+height-area.y].map(value=>Math.round(value*scale));
              const corner=Math.ceil(radius*scale)+3, outer=Math.round(border*scale), line=Math.round(scale), inside=Math.round(2*scale), reach=Math.round(4*scale);
              const lines=[];
              for(let u=left+corner;u<right-corner;u++)lines.push([u,top,0,1],[u,bottom-1,0,-1]);
              for(let v=top+corner;v<bottom-corner;v++)lines.push([left,v,1,0],[right-1,v,-1,0]);
              let crossed=0;const apart=[];
              for(const [u,v,du,dv] of lines){
                const pick=n=>at(u+n*du,v+n*dv), isGreen=n=>de(pick(n),green)<=12, isIvory=n=>de(pick(n),ivory)<=12;
                let edge=-reach;
                while(edge<=reach&&!isGreen(edge))edge++;
                if(edge>reach)continue;
                // From the felt: the whole border, then the whole hairline, then the painting.
                let run=0;
                while(isGreen(edge+run))run++;
                let light=0;
                while(light<line&&isIvory(edge+run+light))light++;
                if(run===outer&&light===line)crossed++;
                // The colours actually drawn where the border and the hairline belong, against the painting inside them.
                const mean=(from,count)=>[0,1,2].map(k=>Array.from({length:count},(_,n)=>pick(from+n)[k]).reduce((sum,value)=>sum+value,0)/count);
                const [rim,hairline]=[mean(edge,outer),mean(edge+outer,line)];
                for(let n=0;n<inside;n++){const paint=pick(edge+outer+line+n);apart.push(Math.max(de(rim,paint),de(hairline,paint)));}
              }
              return {label,images,lines:lines.length,crossed,apart:apart.reduce((sum,value)=>sum+value,0)/apart.length};
            });
          },data,area,boxes,scale,green,ivory);
          for(const {label,images,lines,crossed,apart} of measured) {
            if(images!==1) faults.push(`x${scale} ${label}: not painted`);
            if(crossed<lines) faults.push(`x${scale} ${label}: ${lines-crossed}/${lines} lines do not cross the border and then the hairline`);
            if(!(apart>=30)) faults.push(`x${scale} ${label}: the mark stands ${apart.toFixed(1)} apart from the painting`);
            found.push({scale,label,apart:Math.round(apart*10)/10});
          }
        }
      } finally {
        await page.setViewport({width:1100,height:700,deviceScaleFactor:2});
        await page.click('#greens-shown');
      }
      await writeFile(resolve(evidence,'claimed-greens.json'),JSON.stringify(found.sort((a,b)=>a.apart-b.apart),null,1));
      assert.ok(found.length>=3*2*4*2,`the fixture shows the greenest faces: ${found.length}`);
      assert.deepEqual(faults,[]);
    });
    await check('a painting turned with its tile shows no ivory along its edges, at any size, position and pixel density',async()=>{
      // Only the picture turns, and it meets the screen's pixels apart from
      // the face around it, so it can fall a fraction short of an edge. What
      // shows there must never be a light line of ivory: change the ivory
      // alone, and no pixel of a face may change too, short of its rounded
      // corners, where an upright picture's edge blends with the face's just
      // the same.
      const changed=[];
      try {
        for(const scale of [1,2,3]) {
          await page.setViewport({width:1100,height:700,deviceScaleFactor:scale});
          const settle=()=>page.evaluate(()=>new Promise(done=>{scrollTo(0,0);requestAnimationFrame(()=>requestAnimationFrame(done));}));
          await settle();
          const {clip,boxes}=await page.$eval('#painted',section=>{
            const box=element=>{const r=element.getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height};};
            return {clip:box(section),boxes:[...section.querySelectorAll('.tile')].map(tile=>({...box(tile.querySelector('.face')),
              radius:parseFloat(getComputedStyle(tile.querySelector('.face')).borderTopLeftRadius),label:tile.parentElement.getAttribute('style')}))};
          });
          const area={x:clip.x-4,y:clip.y-4,width:clip.width+8,height:clip.height+8};
          const shot=()=>page.screenshot({clip:area,encoding:'base64',captureBeyondViewport:true});
          const before=await shot();
          await page.$eval('#painted',section=>section.style.setProperty('--ivory','#ff00ff'));
          await settle();
          const after=await shot();
          await page.$eval('#painted',section=>section.style.removeProperty('--ivory'));
          for(const [name,data] of [['plain',before],['magenta',after]])await writeFile(resolve(evidence,`turned-pictures-${name}-x${scale}.png`),Buffer.from(data,'base64'));
          changed.push(...await page.evaluate(async(shots,area,boxes,scale)=>{
            const [a,b]=await Promise.all(shots.map(async data=>{
              const bytes=Uint8Array.from(atob(data),c=>c.charCodeAt(0));const image=await createImageBitmap(new Blob([bytes],{type:'image/png'}));
              const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
              const context=canvas.getContext('2d');context.drawImage(image,0,0);return context.getImageData(0,0,image.width,image.height);
            }));
            return boxes.flatMap(({x,y,width,height,radius,label})=>{
              const [left,top,right,bottom]=[x-area.x,y-area.y,x+width-area.x,y+height-area.y].map(value=>Math.round(value*scale));
              const corner=Math.ceil(radius*scale)+1, inCorner=(column,row)=>(column<left+corner||column>=right-corner)&&(row<top+corner||row>=bottom-corner);
              let count=0;
              for(let row=top;row<bottom;row++)for(let column=left;column<right;column++){
                if(inCorner(column,row))continue;
                const i=(row*a.width+column)*4;
                if(Math.abs(a.data[i]-b.data[i])+Math.abs(a.data[i+1]-b.data[i+1])+Math.abs(a.data[i+2]-b.data[i+2])>12)count++;
              }
              return count?[`x${scale} ${label}: ${count} pixels show the face beneath`]:[];
            });
          },[before,after],area,boxes,scale));
        }
      } finally {
        await page.setViewport({width:1100,height:700,deviceScaleFactor:2});
      }
      assert.deepEqual(changed,[]);
    });
    await check('reused tiles restart both effects together and hint toggles remove them',async()=>{
      await page.click('#identity');await page.waitForSelector('#dynamic .haku-dragon-reveal');
      const clocks=await page.$eval('#dynamic .face',el=>el.getAnimations({subtree:true}).map(a=>a.startTime));
      assert.equal(clocks.length,2);assert.equal(clocks[0],clocks[1]);
      await page.click('#dynamic .tile');assert.equal(await page.$eval('output',el=>el.textContent),'1');
      await page.click('#mark');assert.equal(await page.$('#dynamic .haku-dragon-reveal'),null);
      await page.click('#mark');await page.waitForSelector('#dynamic .haku-dragon-reveal');
      await page.click('#identity');assert.equal(await page.$('#dynamic .haku-dragon-reveal'),null);
    });
    await freeze(2500);await page.screenshot({path:resolve(evidence,'white-dragon-fixture.png'),fullPage:true});
    const tile=await page.$('#dora .tile');
    for(let index=0;index<20;index++){await freeze(index*250);await tile.screenshot({path:resolve(evidence,`white-dragon-frame-${String(index).padStart(2,'0')}.png`)});}
    await check('reduced-motion stops both animations at the same static position',async()=>{
      await page.emulateMediaFeatures([{name:'prefers-reduced-motion',value:'reduce'}]);
      const styles=await page.$eval('#dora .face',el=>[...el.querySelectorAll('.haku-dragon-reveal,.foil')].map(e=>{const s=getComputedStyle(e);return {animation:s.animationName,position:e.classList.contains('foil')?s.backgroundPosition:s.maskPosition};}));
      assert.ok(styles.every(s=>s.animation==='none'));assert.equal(styles[0].position,styles[1].position);
    });
    await check('Matisse uses its approved quiet and lit faces with aligned foil and hidden tiles',async()=>{
      await page.click('#faces');
      await page.waitForSelector('#dora .haku-dragon-reveal.matisse');
      const state=await page.$eval('#dora .face',async el=>{
        const reveal=getComputedStyle(el.querySelector('.haku-dragon-reveal'));
        const image=new Image();image.src=reveal.backgroundImage.slice(5,-2);await image.decode();
        return {base:el.querySelector('img').getAttribute('src'),url:image.src,width:image.naturalWidth,height:image.naturalHeight,size:reveal.backgroundSize,blend:reveal.mixBlendMode,mask:reveal.maskPosition,shine:getComputedStyle(el.querySelector('.foil')).backgroundPosition};
      });
      assert.equal(state.base,'tiles/matisse/approved/Haku.svg');
      assert.match(state.url,/\/mahjong\/tiles\/matisse\/approved\/Haku-foil.svg$/);
      assert.deepEqual([state.width,state.height],[300,400]);
      assert.equal(state.size,'100% 100%');assert.equal(state.blend,'normal');assert.equal(state.mask,state.shine);
      assert.equal(await page.$('#blank .haku-dragon-reveal'),null);
      assert.equal(await page.$('#hidden .haku-dragon-reveal'),null);
      assert.match(await page.$eval('#hidden img',image=>image.src),/\/tiles\/Back.svg$/);
      await page.screenshot({path:resolve(evidence,'matisse-white-dragon-fixture.png'),fullPage:true});
    });
    await check('a tile its set has not painted is written out, upright, readable and inside its face, keeps its marks and loads nothing',async()=>{
      await page.click('#unpainted');
      await page.waitForSelector('#named .face.unpainted .name');
      // What a written tile shows at each width of its face: both words in a
      // hand; in a row, a number over its suit's letter or an honour's first
      // word; and on the table's rows and smaller, an honour's letters.
      const expected=(tile,width)=>{
        const [lead,rest]=tileWords(tile).split(' '), short=tileShorthand(tile);
        return width>40?[lead,rest]:tile[1]!=='z'?[lead,short.slice(lead.length)]:width>34?[lead]:[short];
      };
      const unpainted=(await page.$$eval('.samples section .tile[data-tile]',tiles=>tiles.map(tile=>tile.dataset.tile))).filter(tile=>!DALI_APPROVED.includes(tile));
      assert.ok(unpainted.length,'the fixture shows tiles Dalí has not painted');
      const tiers=new Set();
      for(const id of ['named','named-turned','named-row','named-small','named-suited-turned','named-phone','named-suited-phone']) {
        if(!await page.$(`#${id}`)) continue;
        const shape=await page.$eval(`#${id}`,section=>{
          const box=element=>{const r=element.getBoundingClientRect();return [r.x,r.y,r.width,r.height];};
          const tile=section.querySelector('.tile'),face=section.querySelector('.face'),name=section.querySelector('.name'),layer=getComputedStyle(face,'::after');
          const style=getComputedStyle(face);
          return {tile:tile.dataset.tile,width:Math.min(parseFloat(style.width),parseFloat(style.height)),face:box(face),name:box(name),label:tile.getAttribute('aria-label'),
            images:section.querySelectorAll('img').length,classes:[...tile.classList],
            border:layer.content==='none'?0:parseFloat(layer.borderTopWidth),
            // The hairline inside the border: the spread of its inset shadow.
            hairline:layer.content==='none'||layer.boxShadow==='none'?0:parseFloat(layer.boxShadow.replace(/rgba?\([^)]*\)/,'').trim().split(/\s+/)[3]),
            lines:[...name.children].filter(line=>getComputedStyle(line).display!=='none').map(line=>({text:line.textContent,box:box(line),size:parseFloat(getComputedStyle(line).fontSize)}))};
        });
        assert.equal(shape.images,0,`${id}: nothing is loaded for a tile without a picture`);
        assert.ok(shape.label.startsWith(tileWords(shape.tile)),`${id}: ${shape.label}`);
        assert.ok(shape.classes.includes('dali'),`${id}: the set's rounded corners`);
        const lines=expected(shape.tile,shape.width);
        assert.deepEqual(shape.lines.map(line=>line.text),lines,`${id} at ${shape.width}px`);
        tiers.add(lines.length===2&&lines[1].length===1?'number and letter':lines.length===2?'words':lines[0].length>2?'word':'letters');
        for(let n=0;n<4;n++)assert.ok(Math.abs(shape.name[n]-shape.face[n])<0.5,`${id}: the name covers its face`);
        // Inside the face, and inside a claimed tile's border and its hairline as well.
        const [fx,fy,fw,fh]=shape.face, inset=shape.border+shape.hairline;
        assert.equal(shape.border>0,shape.classes.includes('claimed'),`${id}: a border only on a claimed tile`);
        assert.equal(shape.hairline,shape.classes.includes('claimed')?1:0,`${id}: a hairline inside the border of a claimed tile`);
        for(const {text,box:[x,y,width,height],size} of shape.lines) {
          assert.ok(x>=fx+inset&&y>=fy+inset&&x+width<=fx+fw-inset&&y+height<=fy+fh-inset,`${id}: ${text} stays inside its face, border and hairline`);
          assert.ok(size>=6.5,`${id}: ${text} is ${size}px`);
        }
        // A suit's letter lies under its number, or beside it on a tile turned on its side.
        if(lines.length===2&&lines[1].length===1) {
          const [[nx,ny,nw],[lx,ly,lw]]=shape.lines.map(line=>line.box);
          if(shape.classes.includes('rotated')) assert.ok(lx>=nx+nw-0.5,`${id}: the letter follows its number`);
          else assert.ok(ly>ny&&Math.abs(lx+lw/2-(nx+nw/2))<1,`${id}: the letter stands under its number`);
        }
      }
      const left=TILE_TYPES.filter(tile=>!DALI_APPROVED.includes(tile));
      const possible=[...(left.some(tile=>tile[1]!=='z')?['number and letter','words']:[]),...(left.some(tile=>tile[1]==='z')?['letters','word']:[])];
      assert.deepEqual([...tiers].sort(),possible.sort(),'the fixture shows every way a tile is written');
      // A turned honour still reads across, and has the shine but no dragon.
      if(await page.$('#named-turned')) {
        const [width,height]=await page.$eval('#named-turned .name b',b=>{const r=b.getBoundingClientRect();return [r.width,r.height];});
        assert.ok(width>height,`a turned name reads upright: ${width} by ${height}`);
        assert.ok(await page.$('#named-turned .foil'));
        assert.equal(await page.$('#named-turned .haku-dragon-reveal'),null);
        // Held still for reduced motion, the shine rests on a corner of the
        // face rather than across the name, which has nothing else to read.
        const still=await page.$eval('#named-turned .foil',foil=>{const style=getComputedStyle(foil);return [style.animationName,style.backgroundPosition];});
        assert.deepEqual(still,['none','100% 0px']);
      }
      if(!DALI_APPROVED.includes('5z')) {
        // A white dragon written out, not painted, has the shine alone.
        await page.waitForSelector('#dora .face.unpainted');
        for(const id of ['dora','small','tiny','rotated','muted']) {
          assert.equal(await page.$(`#${id} .haku-dragon-reveal`),null,id);
          assert.ok(await page.$(`#${id} .foil`),id);
        }
      }
      assert.match(await page.$eval('#hidden img',image=>image.src),/\/tiles\/Back.svg$/);
      await freeze(2500);
      await page.screenshot({path:resolve(evidence,'unpainted-tiles-fixture.png'),fullPage:true});
    });
    assert.deepEqual(errors,[]);
  }
} finally {
  await browser?.close();if(server)await new Promise(done=>server.close(done));await rm(temporary,{recursive:true,force:true});
  await report('tile-effect checks',resolve(evidence,'white-dragon-report.json'));
}
