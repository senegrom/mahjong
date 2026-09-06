/** Render the real Tile component with its production CSS. The temporary
 * fixture never ships. Test mask motion (not just opacity), asset loading,
 * rotated geometry, prop changes, hidden tiles and reduced motion. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import puppeteer from 'puppeteer-core';

const web = fileURLToPath(new URL('../', import.meta.url));
const temporary = await mkdtemp(resolve(web, '.tile-effects-'));
const out = resolve(temporary, 'dist');
const evidence = resolve(web, 'test-results');
const results = [];
let browser, server;
const check = async (name, test) => {
  try { await test(); results.push({name, ok:true}); }
  catch(error) { results.push({name, ok:false, error:error.stack}); }
};
try {
  await mkdir(evidence, {recursive:true});
  await writeFile(resolve(temporary, 'index.html'), '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Tile effects regression</title></head><body><div id="app"></div><script type="module" src="./main.js"></script></body></html>');
  await writeFile(resolve(temporary, 'main.js'), 'import { mount } from "svelte"; import Fixture from "./Fixture.svelte"; mount(Fixture, {target:document.getElementById("app")});');
  await writeFile(resolve(temporary, 'Fixture.svelte'), `<script>
import Tile from '../src/lib/Tile.svelte';
let tile = $state('1m'); let marked = $state(true); let clicks = $state(0);
const cases = [
  ['blank', {tile:'5z'}], ['dora', {tile:'5z',dora:true}],
  ['small', {tile:'5z',dora:true,size:'small'}], ['tiny', {tile:'5z',dora:true,size:'tiny'}],
  ['rotated', {tile:'5z',dora:true,rotated:true}], ['hidden', {tile:'5z',dora:true,facedown:true}],
  ['other', {tile:'7z',dora:true}], ['muted', {tile:'5z',dora:true,disabled:true,onclick:()=>{}}]
];
</script>
<h1>White dragon · dora foil</h1>
<div class="samples">{#each cases as [id, props]}<section id={id}><p>{id}</p><Tile {...props}/></section>{/each}</div>
<section id="dynamic"><Tile {tile} dora={marked} onclick={()=>clicks++}/></section>
<button id="identity" onclick={()=>tile=tile==='5z'?'1m':'5z'}>Change tile</button>
<button id="mark" onclick={()=>marked=!marked}>Toggle dora / hints</button><output>{clicks}</output>
<style>:global(body){margin:30px;background:#173e35;color:#fff;font:16px system-ui;--tile-width:60px;--ivory:#fffaf0;} .samples{display:flex;gap:26px;align-items:start;flex-wrap:wrap;margin-bottom:40px} section{min-width:70px} #dynamic{margin:25px 0} button{margin:10px;padding:10px}</style>`);
  await build({configFile:false,root:temporary,base:'/mahjong/',publicDir:false,plugins:[svelte()],logLevel:'warn',build:{outDir:out,target:'es2022'}});
  if (!process.argv.includes('--build-only')) {
    server = createServer(createFixtureHandler({root:out,publicRoot:resolve(web,'public')}));
    await new Promise(done=>server.listen(0,'127.0.0.1',done));
    const executablePath = process.env.CHROME_BIN || ['/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser'].find(existsSync);
    assert.ok(executablePath,'Set CHROME_BIN to Chrome/Chromium');
    browser = await puppeteer.launch({executablePath,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
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
    assert.deepEqual(errors,[]);
    console.log(`${results.filter(r=>r.ok).length}/${results.length} tile-effect checks passed`);
    for(const result of results)if(!result.ok)console.error(result.name,result.error);
    if(results.some(r=>!r.ok))process.exitCode=1;
  }
} finally {
  await writeFile(resolve(evidence,'white-dragon-report.json'),JSON.stringify(results,null,2));
  await browser?.close();if(server)await new Promise(done=>server.close(done));await rm(temporary,{recursive:true,force:true});
}
