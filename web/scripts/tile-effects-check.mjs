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
import { tileShorthand, tileWords } from '../src/lib/tiles.js';

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
  await writeFile(resolve(temporary, 'Fixture.svelte'), `<script>
import Tile from '../src/lib/Tile.svelte';
import { setContext } from 'svelte';
import { TILE_FACE_CONTEXT } from '../src/lib/tile-faces.js';
import { DALI_APPROVED } from '../src/lib/dali-faces.js';
import { TILE_TYPES } from '../src/lib/tiles.js';
let face = $state('classic');
setContext(TILE_FACE_CONTEXT, () => face);
// Tiles Dalí has not painted yet, whichever they are when this runs, at the
// sizes that show both lines, the first alone, and an honour's letters.
const unpainted = TILE_TYPES.filter(tile => !DALI_APPROVED.includes(tile));
const suited = unpainted.find(tile => tile[1] !== 'z'), honour = unpainted.find(tile => tile[1] === 'z');
const named = [
  ['named', {tile:suited}], ['named-turned', {tile:honour,rotated:true,dora:true,size:'small'}],
  ['named-row', {tile:suited,size:'tiny',fromDraw:true,claimed:true}],
].filter(([, props]) => props.tile);
let tile = $state('1m'); let marked = $state(true); let clicks = $state(0);
const cases = [
  ['blank', {tile:'5z'}], ['dora', {tile:'5z',dora:true}],
  ['small', {tile:'5z',dora:true,size:'small'}], ['tiny', {tile:'5z',dora:true,size:'tiny'}],
  ['rotated', {tile:'5z',dora:true,rotated:true}], ['hidden', {tile:'5z',dora:true,facedown:true}],
  ['other', {tile:'7z',dora:true}], ['muted', {tile:'5z',dora:true,disabled:true,onclick:()=>{}}],
  ['kept', {tile:'5z',size:'small'}], ['thrown', {tile:'5z',size:'small',fromDraw:true}],
  ['taken', {tile:'5z',size:'small',claimed:true}], ['both', {tile:'5z',size:'small',claimed:true,fromDraw:true}],
  ['taken-dora', {tile:'5z',size:'small',claimed:true,fromDraw:true,dora:true,rotated:true}]
];
</script>
<h1>White dragon · dora foil</h1>
<div class="samples">{#each cases as [id, props]}<section id={id}><p>{id}</p><Tile {...props}/></section>{/each}</div>
<section id="dynamic"><Tile {tile} dora={marked} onclick={()=>clicks++}/></section>
<button id="identity" onclick={()=>tile=tile==='5z'?'1m':'5z'}>Change tile</button>
<button id="mark" onclick={()=>marked=!marked}>Toggle dora / hints</button><output>{clicks}</output>
<button id="faces" onclick={()=>face=face==='classic'?'matisse':'classic'}>Change tile face</button>
<button id="unpainted" onclick={()=>face='dali'}>Show a set still being painted</button>
<div class="samples">{#each named as [id, props]}<section id={id}><p>{id}</p><Tile {...props}/></section>{/each}{#if honour}<section id="named-phone" style="--tile-width:28px"><p>named-phone</p><Tile tile={honour} size="tiny"/></section>{/if}</div>
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
      for(const id of ['dora','small','tiny','rotated']) {
        const boxes=await page.$eval(`#${id} .tile`,el=>[el,...el.querySelectorAll('.face,.haku-dragon-reveal,.foil')].map(e=>{const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height];}));
        for(const box of boxes.slice(1))for(let n=0;n<4;n++)assert.ok(Math.abs(box[n]-boxes[0][n])<0.1,`${id} geometry changed`);
      }
    });
    await check('a discard from the draw is darker in its own colours; a claimed one lets the felt through; both combine',async()=>{
      const marks=await page.evaluate(()=>['kept','thrown','taken','both'].map(id=>{
        const tile=document.querySelector(`#${id} .tile`);
        return [getComputedStyle(tile).opacity,getComputedStyle(tile.querySelector('.face')).filter];
      }));
      assert.deepEqual(marks,[['1','none'],['1','brightness(0.86)'],['0.6','none'],['0.6','brightness(0.86)']]);
      // Read the actual pixels: the middle of each blank face, the felt
      // beside the tile and, on the dora tile, the ring just outside its edge.
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
          // In CSS pixels: the middle of the face, a corner of felt, and one pixel above the tile's top edge.
          return {face:at(width/2,image.height/scale/2),felt:at(1,1),ring:at(width/2,margin-1)};
        },data,clip.width,margin);
      }
      const near=(actual,expected,label)=>actual.forEach((value,channel)=>assert.ok(Math.abs(value-expected[channel])<=6,`${label}: ${actual} against ${expected.map(Math.round)}`));
      const mix=(top,under,alpha)=>top.map((value,channel)=>alpha*value+(1-alpha)*under[channel]);
      const {kept,thrown,taken,both}=pixels, felt=kept.felt, green=([r,g])=>g-r;
      near(thrown.face,kept.face.map(value=>value*0.86),'thrown from the draw: the same colours, darker');
      near(taken.face,mix(kept.face,felt,0.6),'claimed: the felt shows through');
      near(both.face,mix(thrown.face,felt,0.6),'claimed and from the draw: both marks');
      assert.ok(green(kept.face)<=0&&green(thrown.face)<=0&&green(taken.face)>=8,`only the claimed tile reads green: ${[kept,thrown,taken].map(p=>p.face)}`);
      // Fading the tile as one keeps a dora ring around the face, not through it.
      near(pixels['taken-dora'].face,both.face,'the ring does not tint a claimed face');
      const [r,g,b]=pixels['taken-dora'].ring;
      assert.ok(r>g+40&&r>b+40,`the dora ring still shows red: ${pixels['taken-dora'].ring}`);
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
      // Both lines in the hand, the first alone in a row, and an honour's
      // letters where even its word could not be read.
      const unpainted=(await page.$$eval('.samples section .tile[data-tile]',tiles=>tiles.map(tile=>tile.dataset.tile))).filter(tile=>!DALI_APPROVED.includes(tile));
      assert.ok(unpainted.length,'the fixture shows tiles Dalí has not painted');
      const shown={named:'both','named-turned':'lead','named-row':'lead','named-phone':'shorthand'};
      for(const [id,lines] of Object.entries(shown)) {
        if(!await page.$(`#${id}`)) continue;
        const shape=await page.$eval(`#${id}`,section=>{
          const box=element=>{const r=element.getBoundingClientRect();return [r.x,r.y,r.width,r.height];};
          const tile=section.querySelector('.tile'),name=section.querySelector('.name');
          return {tile:tile.dataset.tile,face:box(section.querySelector('.face')),name:box(name),label:tile.getAttribute('aria-label'),
            images:section.querySelectorAll('img').length,classes:[...tile.classList],
            lines:[...name.children].filter(line=>getComputedStyle(line).display!=='none').map(line=>({text:line.textContent,box:box(line),size:parseFloat(getComputedStyle(line).fontSize)}))};
        });
        const [lead,rest]=tileWords(shape.tile).split(' ');
        assert.equal(shape.images,0,`${id}: nothing is loaded for a tile without a picture`);
        assert.ok(shape.label.startsWith(tileWords(shape.tile)),`${id}: ${shape.label}`);
        assert.ok(shape.classes.includes('dali'),`${id}: the set's rounded corners`);
        assert.deepEqual(shape.lines.map(line=>line.text),{both:[lead,rest],lead:[lead],shorthand:[tileShorthand(shape.tile)]}[lines],id);
        for(let n=0;n<4;n++)assert.ok(Math.abs(shape.name[n]-shape.face[n])<0.5,`${id}: the name covers its face`);
        const [fx,fy,fw,fh]=shape.face;
        for(const {text,box:[x,y,width,height],size} of shape.lines) {
          assert.ok(x>=fx&&y>=fy&&x+width<=fx+fw&&y+height<=fy+fh,`${id}: ${text} stays inside its face`);
          assert.ok(size>=6.5,`${id}: ${text} is ${size}px`);
        }
      }
      // A turned honour still reads across, and has the shine but no dragon.
      if(await page.$('#named-turned')) {
        const [width,height]=await page.$eval('#named-turned .name b',b=>{const r=b.getBoundingClientRect();return [r.width,r.height];});
        assert.ok(width>height,`a turned name reads upright: ${width} by ${height}`);
        assert.ok(await page.$('#named-turned .foil'));
        assert.equal(await page.$('#named-turned .haku-dragon-reveal'),null);
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
