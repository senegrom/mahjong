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
  await writeFile(resolve(temporary, 'Fixture.svelte'), `<script>
import Tile from '../src/lib/Tile.svelte';
import { setContext } from 'svelte';
import { TILE_FACE_CONTEXT } from '../src/lib/tile-faces.js';
import { DALI_APPROVED } from '../src/lib/dali-faces.js';
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
</script>
<h1>White dragon · dora foil</h1>
<div class="samples">{#each cases as [id, props]}<section id={id}><p>{id}</p><Tile {...props}/></section>{/each}<section id="phone" style="--tile-width:23px"><p>phone</p><Tile tile="5z" size="tiny" claimed/></section></div>
<section id="dynamic"><Tile {tile} dora={marked} onclick={()=>clicks++}/></section>
<button id="identity" onclick={()=>tile=tile==='5z'?'1m':'5z'}>Change tile</button>
<button id="mark" onclick={()=>marked=!marked}>Toggle dora / hints</button><output>{clicks}</output>
<button id="faces" onclick={()=>face=face==='classic'?'matisse':'classic'}>Change tile face</button>
<button id="unpainted" onclick={()=>face='dali'}>Show a set still being painted</button>
<div class="samples">{#each named as [id, props]}<section id={id}><p>{id}</p><Tile {...props}/></section>{/each}{#each phones as [id, tile]}<section id={id} style="--tile-width:28px"><p>{id}</p><Tile {tile} size="tiny" claimed/></section>{/each}</div>
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
        const boxes=await page.$eval(`#${id} .tile`,el=>[el,...el.querySelectorAll('.face,.haku-dragon-reveal,.foil')].map(e=>{const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height];}));
        for(const box of boxes.slice(1))for(let n=0;n<4;n++)assert.ok(Math.abs(box[n]-boxes[0][n])<0.1,`${id} geometry changed`);
      }
    });
    await check('a discard from the draw is darker in its own colours; a claimed one has a solid dark green border inside its edge; both combine',async()=>{
      const border=[14,110,51], solid=`solid rgb(${border.join(', ')})`;
      const marks=await page.evaluate(()=>['kept','thrown','taken','both'].map(id=>{
        const tile=document.querySelector(`#${id} .tile`), edge=getComputedStyle(tile,'::after');
        return [getComputedStyle(tile).opacity,getComputedStyle(tile.querySelector('.face')).filter,
          edge.content==='none'?'none':`${edge.borderTopStyle} ${edge.borderTopColor}`];
      }));
      assert.deepEqual(marks,[['1','none','none'],['1','brightness(0.72)','none'],['1','none',solid],['1','brightness(0.72)',solid]]);
      // The border keeps to the tile's proportions: one pixel on a phone's
      // smallest row, two on the table's and three on the largest.
      const [phone,tiny,small]=await page.evaluate(()=>['phone','tiny-taken','taken'].map(id=>
        parseFloat(getComputedStyle(document.querySelector(`#${id} .tile`),'::after').borderTopWidth)));
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
          const tile=section.querySelector('.tile'),face=section.querySelector('.face'),name=section.querySelector('.name'),edge=getComputedStyle(tile,'::after');
          const style=getComputedStyle(face);
          return {tile:tile.dataset.tile,width:Math.min(parseFloat(style.width),parseFloat(style.height)),face:box(face),name:box(name),label:tile.getAttribute('aria-label'),
            images:section.querySelectorAll('img').length,classes:[...tile.classList],
            border:edge.content==='none'?0:parseFloat(edge.borderTopWidth),
            lines:[...name.children].filter(line=>getComputedStyle(line).display!=='none').map(line=>({text:line.textContent,box:box(line),size:parseFloat(getComputedStyle(line).fontSize)}))};
        });
        assert.equal(shape.images,0,`${id}: nothing is loaded for a tile without a picture`);
        assert.ok(shape.label.startsWith(tileWords(shape.tile)),`${id}: ${shape.label}`);
        assert.ok(shape.classes.includes('dali'),`${id}: the set's rounded corners`);
        const lines=expected(shape.tile,shape.width);
        assert.deepEqual(shape.lines.map(line=>line.text),lines,`${id} at ${shape.width}px`);
        tiers.add(lines.length===2&&lines[1].length===1?'number and letter':lines.length===2?'words':lines[0].length>2?'word':'letters');
        for(let n=0;n<4;n++)assert.ok(Math.abs(shape.name[n]-shape.face[n])<0.5,`${id}: the name covers its face`);
        // Inside the face, and inside a claimed tile's border as well.
        const [fx,fy,fw,fh]=shape.face, inset=shape.border;
        assert.equal(inset>0,shape.classes.includes('claimed'),`${id}: a border only on a claimed tile`);
        for(const {text,box:[x,y,width,height],size} of shape.lines) {
          assert.ok(x>=fx+inset&&y>=fy+inset&&x+width<=fx+fw-inset&&y+height<=fy+fh-inset,`${id}: ${text} stays inside its face and border`);
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
