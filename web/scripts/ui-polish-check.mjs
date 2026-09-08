import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { createFixtureHandler } from './static-fixture-server.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { unseenTileCounts } from '../src/lib/ui.js';

await init({module_or_path:readFileSync(new URL('../src/wasm/riichi_bg.wasm',import.meta.url))});
const web=fileURLToPath(new URL('../',import.meta.url)),output=resolve(web,'test-results');
const handler=createFixtureHandler({root:resolve(web,'dist'),publicRoot:resolve(web,'dist')});
const server=createServer(handler);let browser;const contexts=[],results=[];
function opening(){const m=new MatchSession(Game,81,'club');try{m.advance(false);return {snapshot:m.snapshot(),view:m.view};}finally{m.dispose();}}
function calling(){const m=new MatchSession(Game,1,'club');try{m.advance(false);for(const [kind,tile] of [['discard','9s'],['discard','3m'],['discard','8s'],['discard','7p']]){m.apply({type:'choose',kind,tile});m.advance(false);}return m.snapshot();}finally{m.dispose();}}
function won(){const m=new MatchSession(Game,16,'club');try{m.advance(false);for(let n=0;n<240&&m.view.phase!=='over';n++){const c=m.choices;const q=c.find(x=>['ron','tsumo'].includes(x.kind))??c.find(x=>x.kind==='riichi')??c.find(x=>x.kind==='pass')??c.find(x=>x.kind==='discard'&&x.tile===m.view.seats[0].drawn)??c.find(x=>x.kind==='discard')??c[0];assert.ok(q);m.apply({type:'choose',kind:q.kind,tile:q.tile??null});m.advance(false);}assert.equal(m.view.phase,'over');return m.snapshot();}finally{m.dispose();}}
const start=opening(),call=calling(),result=won();
async function open(snapshot=start.snapshot,width=390,height=844){const context=await browser.createBrowserContext();contexts.push(context);const p=await context.newPage();p.errors=[];p.on('pageerror',e=>p.errors.push(e.message));await p.setViewport({width,height,isMobile:width<600,hasTouch:width<600});await p.emulateMediaFeatures([{name:'prefers-reduced-motion',value:'reduce'}]);await p.evaluateOnNewDocument((key,settings,s)=>{Object.defineProperty(navigator,'serviceWorker',{value:undefined});localStorage.setItem(key,JSON.stringify(s));localStorage.setItem(settings,JSON.stringify({version:1,difficulty:'club',opponents:['club','club','club'],hints:true,confirmDiscards:true,shortcuts:true,tileFace:'classic'}));},SAVE_KEY,SETTINGS_KEY,snapshot);await p.goto(`http://127.0.0.1:${server.address().port}/mahjong/`,{waitUntil:'networkidle0'});await p.waitForSelector('.hand');return p;}
async function check(name,fn){try{await fn();results.push({name,passed:true});console.log('PASS '+name);}catch(e){results.push({name,passed:false,error:e.stack});console.error('FAIL '+name+'\n'+e.stack);}finally{while(contexts.length)await contexts.pop().close();}}
const shot=(p,name)=>p.screenshot({path:resolve(output,name+'.png'),fullPage:true});
try{await mkdir(output,{recursive:true});await new Promise(done=>server.listen(0,'127.0.0.1',done));const chrome=process.env.CHROME_BIN||['/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser'].find(existsSync);assert.ok(chrome);browser=await puppeteer.launch({executablePath:chrome,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 await check('phone header keeps one round display and accessible game controls',async()=>{
  const p=await open();
  assert.equal(await p.$eval('.bar h1',el=>el.textContent),'Riichi');
  assert.match(await p.$eval('.round-details',el=>el.textContent),/East 1[\s\S]*tiles left/);
  assert.equal(await p.$('.notice'),null,'restoring a match should not cover the table with a toast');
  assert.equal(await p.$eval('.opponents select',el=>el.disabled),false);
  assert.equal(await p.$eval('.preferences',el=>getComputedStyle(el).display),'none');
  assert.notEqual(await p.$eval('.settings-trigger',el=>getComputedStyle(el).display),'none');
  assert.equal(await p.$eval('.restart',el=>getComputedStyle(el).display),'none');
  await shot(p,'polish-phone-header');
  await p.click('.inspect');
  assert.ok(await p.$('.table-dialog[open]'));
  await p.keyboard.press('Escape');
  await p.click('.settings-trigger');
  assert.notEqual(await p.$eval('.preferences',el=>getComputedStyle(el).display),'none');
  assert.ok(await p.$('.mobile-new-game'));
  await shot(p,'polish-phone-settings');
  assert.deepEqual(p.errors,[]);
 });
 await check('hint counts sit below every hand tile and exactly match public unseen copies',async()=>{const p=await open(),expected=unseenTileCounts(start.view);const rows=await p.$$eval('.hand .hand-tile',els=>els.map(el=>{const button=el.querySelector('button.tile'),tile=button.dataset.tile,count=el.querySelector('.copy-count'),a=button.getBoundingClientRect(),b=count.getBoundingClientRect();return{tile,count:Number(count.textContent.trim()),below:b.top>=a.bottom-0.5};}));assert.equal(rows.length,14);for(const row of rows){assert.equal(row.count,expected.get(row.tile));assert.equal(row.below,true,row.tile);}await p.click('.settings-trigger');await p.click('.options summary');await p.click('.option-fields input[type=checkbox]');await p.waitForFunction(()=>!document.querySelector('.copy-count'));assert.equal(await p.$('.copy-count'),null);await shot(p,'polish-phone-game');assert.deepEqual(p.errors,[]);});
 await check('desktop table uses the expanded table surface and larger opponent zones',async()=>{const p=await open(start.snapshot,1440,1000);const box=await p.$eval('.board',el=>{const r=el.getBoundingClientRect(),s=getComputedStyle(el);return{w:r.width,h:r.height,bg:s.backgroundImage,radius:s.borderRadius};});assert.ok(box.w>1000);assert.ok(box.h>=380);assert.notEqual(box.bg,'none');assert.ok(parseFloat(box.radius)>=20);await shot(p,'polish-desktop-table');assert.deepEqual(p.errors,[]);});
 await check('call window presents the discard as a staged event before choices',async()=>{const p=await open(call);await p.waitForSelector('.call-stage');assert.match(await p.$eval('.call-stage',el=>el.textContent),/discarded/i);const tile=await p.$eval('.call-stage .tile',el=>el.getBoundingClientRect().width);assert.ok(tile>=40);assert.ok(await p.$('.call-options'));await shot(p,'polish-phone-call');assert.deepEqual(p.errors,[]);});
 await check('phone result is a fixed bottom sheet with hero value and can reveal the table',async()=>{const p=await open(result);await p.waitForSelector('.screen');const state=await p.$eval('.screen',el=>({position:getComputedStyle(el).position,maxHeight:getComputedStyle(el).maxHeight,bottom:el.getBoundingClientRect().bottom,hero:el.querySelector('.hero-score')?.textContent,buttons:getComputedStyle(el.querySelector('.buttons')).position}));assert.equal(state.position,'fixed');assert.ok(state.bottom<=845&&state.bottom>=842);assert.match(state.hero,/Ron|Tsumo/);assert.equal(state.buttons,'sticky');const review=await p.$('.screen .quiet');if(review){await review.click();await p.waitForSelector('.result-chip');assert.equal(await p.$eval('.screen',el=>getComputedStyle(el).display),'none');await p.click('.result-chip');assert.notEqual(await p.$eval('.screen',el=>getComputedStyle(el).display),'none');}await shot(p,'polish-phone-result');assert.deepEqual(p.errors,[]);});
}finally{await writeFile(resolve(output,'ui-polish-report.json'),JSON.stringify(results,null,2));for(const c of contexts)await c.close();await browser?.close();if(server.listening)await new Promise(done=>server.close(done));}
console.log(`${results.filter(r=>r.passed).length}/${results.length} UI-polish browser checks passed`);if(results.some(r=>!r.passed))process.exitCode=1;
