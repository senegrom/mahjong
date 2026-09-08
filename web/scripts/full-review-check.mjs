/** Full-app regressions on the production build. No test hooks are shipped.
 * The fixtures are legal commands replayed by the actual rebuilt WASM engine.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { readFileSync, existsSync } from 'node:fs';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
await init({module_or_path:readFileSync(new URL('../src/wasm/riichi_bg.wasm',import.meta.url))});
const web=fileURLToPath(new URL('../',import.meta.url)), dist=resolve(web,'dist'), output=resolve(web,'test-results');
await mkdir(output,{recursive:true});
const types={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.webp':'image/webp','.png':'image/png','.ico':'image/x-icon','.json':'application/json','.wasm':'application/wasm'};
const server=createServer(async(req,res)=>{
  try {
    const path=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    if(!path.startsWith('/mahjong/')){res.writeHead(404).end();return;}
    const file=resolve(dist,path.slice('/mahjong/'.length)||'index.html');
    if(!file.startsWith(dist+sep)){res.writeHead(403).end();return;}
    const data=await readFile(file);res.writeHead(200,{'Content-Type':types[extname(file)]??'application/octet-stream','Cache-Control':'no-store'});res.end(req.method==='HEAD'?undefined:data);
  }catch{res.writeHead(404).end();}
});
let browser;
const results=[],contexts=[];
const make=(seed,mode='club')=>{const m=new MatchSession(Game,seed,mode);m.advance(false);return m;};
const step=(m,kind,tile)=>{m.apply({type:'choose',kind,tile:tile??null});m.advance(false);};
function finishHand(m,riichi=true){
  for(let n=0;n<200&&m.view.phase!=='over';n++){
    const c=m.choices;
    const q=c.find(c=>['ron','tsumo'].includes(c.kind))??(riichi?c.find(c=>c.kind==='riichi'):null)??c.find(c=>c.kind==='pass')
      ??(riichi?c.find(c=>c.kind==='discard'&&c.tile===m.view.seats[0].drawn):null)??c.find(c=>c.kind==='discard')??c[0];
    assert.ok(q);step(m,q.kind,q.tile);
  }
  assert.equal(m.view.phase,'over');
}
function fixture(seed,commands=[]){const m=make(seed);try{for(const [kind,tile] of commands)step(m,kind,tile);return m.snapshot();}finally{m.dispose();}}
function won(seed){const m=make(seed);try{finishHand(m);return m.snapshot();}finally{m.dispose();}}
const initial=fixture(11),riichi=fixture(31,[['riichi','8m']]);
const beforeRiichi=fixture(81,[['discard','9s'],['discard','5z']]);
const waiting=fixture(81,[['discard','9s'],['discard','5z'],['riichi','2z']]);
const wins=[3,16,248].map(won);
const final=(()=>{
 const m=make(14,'beginner');let before,events,log,review;
 try{for(let n=0;n<40&&!m.over;n++){finishHand(m,false);before=m.snapshot();events=[...m.events];log=m.engine.log();review=m.engine.review();m.apply({type:'next'});m.advance(false);}assert.ok(m.over);return {before,after:m.snapshot(),events,log,review};}
 finally{m.dispose();}
})();
const saved=page=>page.evaluate(key=>JSON.parse(localStorage.getItem(key)),SAVE_KEY);
async function check(name,test){try{await test();results.push({name,passed:true});console.log(`PASS ${name}`);}catch(error){results.push({name,passed:false,error:error.stack});console.error(`FAIL ${name}\n${error.stack}`);}}
async function open(snapshot,{width=1100,height=900,confirm=false,hints=true}={}){
 const context=await browser.createBrowserContext();contexts.push(context);const page=await context.newPage();page.reviewErrors=[];
 page.on('pageerror',error=>page.reviewErrors.push(error.message));await page.setViewport({width,height});
 await page.evaluateOnNewDocument((key,settings,snapshot,confirm,hints)=>{
  if(!localStorage.getItem(key))localStorage.setItem(key,JSON.stringify(snapshot));
  localStorage.setItem(settings,JSON.stringify({version:1,difficulty:snapshot.difficulty,hints,confirmDiscards:confirm,shortcuts:true}));
 },SAVE_KEY,SETTINGS_KEY,snapshot,confirm,hints);
 await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/`,{waitUntil:'networkidle0'});
 await page.waitForSelector('.hand');return page;
}
const noErrors=page=>assert.deepEqual(page.reviewErrors,[]);
const shot=(page,name)=>page.screenshot({path:resolve(output,name+'.png'),fullPage:true});
try{
 await new Promise(done=>server.listen(0,'127.0.0.1',done));
 const chrome=process.env.CHROME_BIN||['/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser'].find(existsSync);assert.ok(chrome);
 browser=await puppeteer.launch({executablePath:chrome,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 await check('two consecutive keyboard turns retain usable hand focus',async()=>{
  const p=await open(initial);await p.focus('.hand');
  for(let n=1;n<=2;n++){
   await p.keyboard.press('ArrowLeft');const tile=await p.$eval('.hand .selected',el=>el.dataset.tile);
   assert.equal(await p.evaluate(()=>document.activeElement.dataset.tile),tile);await p.keyboard.press('Enter');
   await p.waitForFunction((key,n)=>JSON.parse(localStorage.getItem(key)).commands.length===n,{},SAVE_KEY,n);
   await p.waitForFunction(()=>document.activeElement===document.querySelector('.hand')&&document.querySelector('.hand button:not(:disabled)'));
   assert.equal((await saved(p)).commands.at(-1).tile,tile);
  }noErrors(p);
 });
 await check('riichi arrows never mark disabled concealed tiles',async()=>{
  const p=await open(riichi);assert.equal(await p.$$eval('.hand button:not(:disabled)',els=>els.length),1);
  await p.focus('.hand');for(const key of ['ArrowLeft','ArrowLeft','ArrowRight','ArrowLeft']){
   await p.keyboard.press(key);assert.equal(await p.$eval('.hand .selected',el=>el.dataset.tile),'2z');
   assert.equal(await p.evaluate(()=>document.activeElement.dataset.tile),'2z');
  }
  await p.keyboard.press('Enter');await p.waitForFunction(key=>JSON.parse(localStorage.getItem(key)).commands.length===2,{},SAVE_KEY);
  assert.equal((await saved(p)).commands.at(-1).tile,'2z');noErrors(p);
 });
 await check('returning hand focus respects a deliberately focused option control',async()=>{
  const p=await open(initial);await p.click('.options summary');await p.focus('.hand');await p.keyboard.press('ArrowLeft');
  await p.evaluate(()=>{
   document.activeElement.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true,cancelable:true}));
   document.querySelector('.option-fields input').focus();
  });
  await p.waitForFunction(key=>JSON.parse(localStorage.getItem(key)).commands.length===1,{},SAVE_KEY);
  await p.waitForFunction(()=>Boolean(document.querySelector('.hand button:not(:disabled)')));
  assert.ok(await p.evaluate(()=>document.activeElement===document.querySelector('.option-fields input')));
  await p.keyboard.press('Space');assert.equal((await saved(p)).commands.length,1);noErrors(p);
 });
 await check('riichi wait panel excludes the unrelated drawn North',async()=>{
  const p=await open(waiting,{width:390,height:844});
  assert.deepEqual(await p.$$eval('.hint .wait .tile',els=>els.map(el=>el.getAttribute('aria-label'))),['green dragon']);
  await shot(p,'riichi-correct-wait');noErrors(p);
 });
 await check('selecting a discard previews that hand without applying the move',async()=>{
  const p=await open(beforeRiichi,{confirm:true});await p.click('.hand button[data-tile="2z"]');
  assert.match(await p.$eval('.hint',el=>el.textContent),/After discarding south wind, waiting on/i);
  assert.deepEqual(await p.$$eval('.hint .wait .tile',els=>els.map(el=>el.getAttribute('aria-label'))),['green dragon']);
  assert.equal((await saved(p)).commands.length,2);noErrors(p);
 });
 await check('ura indicators are not exposed during an unfinished riichi hand',async()=>{
  const p=await open(waiting);assert.equal(await p.$('.ura-indicators'),null);assert.equal(await p.$('.indicator-row.ura'),null);noErrors(p);
 });
 for(let index=0;index<wins.length;index++)await check(`winning result shows correct per-player ura indicators (seed ${[3,16,248][index]})`,async()=>{
  const snapshot=wins[index],view=JSON.parse(snapshot.state)[0];const p=await open(snapshot,{hints:false,width:390,height:844});
  assert.equal(await p.$$('.win').then(els=>els.length),view.outcome.wins.length);
  for(let i=0;i<view.outcome.wins.length;i++){
   const win=view.outcome.wins[i];const actual=await p.$$eval('.win',(els,i)=>({
    ura:[...els[i].querySelectorAll('.indicator-row.ura img')].map(img=>img.getAttribute('src')),
    count:els[i].querySelector('.ura-count b')?.textContent??null,
    ordinary:els[i].querySelector('.dora-count b')?.textContent??null,
   }),i);
   assert.equal(actual.ura.length,win.ura_indicators.length);
   if(win.ura_indicators.length){assert.equal(actual.count,String(win.ura_dora));assert.equal(actual.ordinary,String(win.dora-win.ura_dora));}
   else assert.equal(actual.count,null);
  }
  const shown=view.outcome.wins.find(win=>win.ura_indicators.length)?.ura_indicators??[];
  assert.equal(await p.$$eval('.ura-indicators .tile',els=>els.length),shown.length);
  assert.ok(await p.evaluate(()=>[...document.querySelectorAll('.indicator-row img,.ura-indicators img')].every(img=>img.complete&&img.naturalWidth>0)));
  await shot(p,`ura-dora-result-${[3,16,248][index]}`);noErrors(p);
 });
 await check('final standings retain event history, an open review and export controls',async()=>{
  const p=await open(final.before);await p.click('.screen .quiet');await p.waitForSelector('.review');
  await p.click('.screen .primary');await p.waitForSelector('.standings');
  assert.ok(await p.$('.review'));assert.equal(await p.$$eval('.log p',els=>els.length),final.events.length);
  assert.ok(await p.$$eval('.screen button',els=>els.some(el=>el.textContent.trim()==='Save final hand')));
  await p.reload({waitUntil:'networkidle0'});await p.waitForSelector('.standings');
  const review=await p.$('.screen .quiet');assert.equal(await review.evaluate(el=>el.textContent),'Review final hand');
  await review.click();await p.waitForSelector('.review');assert.equal(await p.$$eval('.log p',els=>els.length),final.events.length);
  await shot(p,'final-standings-with-review');noErrors(p);
 });
 for(const [width,height] of [[320,568],[390,844],[844,390]])await check(`ura and final-hand results fit ${width}x${height}`,async()=>{
  for(const [label,snapshot] of [['ura',wins[1]],['final',final.after]]){
   const p=await open(snapshot,{width,height});
   const overflow=await p.evaluate(()=>({viewport:innerWidth,document:document.documentElement.scrollWidth,wide:[...document.querySelectorAll('*')].map(el=>{const r=el.getBoundingClientRect();return{tag:el.tagName,cls:el.className?.toString?.()??'',left:r.left,right:r.right,width:r.width,scroll:el.scrollWidth,client:el.clientWidth};}).filter(x=>x.right>innerWidth+1||x.left<-1||x.scroll>x.client+1).sort((a,b)=>Math.max(b.right-innerWidth,b.scroll-b.client)-Math.max(a.right-innerWidth,a.scroll-a.client)).slice(0,12)}));
   assert.ok(overflow.document<=overflow.viewport+1,JSON.stringify(overflow));
   assert.ok(await p.$$eval('.screen,.standings,.bonus-indicators',els=>els.every(el=>el.scrollWidth<=el.clientWidth+1)));
   await shot(p,`${label}-${width}x${height}`);noErrors(p);
  }
 });
}finally{
 await writeFile(resolve(output,'full-review-report.json'),JSON.stringify(results,null,2));
 for(const context of contexts)await context.close();await browser?.close();if(server.listening)await new Promise(done=>server.close(done));
}
console.log(`${results.filter(r=>r.passed).length}/${results.length} full-review browser checks passed`);
if(results.some(r=>!r.passed))process.exitCode=1;
