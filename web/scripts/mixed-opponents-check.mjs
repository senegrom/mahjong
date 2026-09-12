/** Production-build mixed controller tests, including the actual shipped
 * network. Fixtures use legal engine commands; no test API ships to users. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync, existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { createFixtureHandler } from './static-fixture-server.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { MODEL_FILES } from '../src/lib/model-package.js';
await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const web=fileURLToPath(new URL('../', import.meta.url));
const output=resolve(web,'test-results');
const handler=createFixtureHandler({root:resolve(web,'dist'),publicRoot:resolve(web,'dist')});
let modelGets=0;
const server=createServer((req,res)=>{
  if(req.url===`/mahjong/${MODEL_FILES.full}`&&req.method==='GET')modelGets++;
  void handler(req,res);
});
const results=[],contexts=[];
let browser;
function fixture(config='club') {
  const m=new MatchSession(Game,81,config);
  try {m.advance(false);return m.snapshot();} finally {m.dispose();}
}
const initial=fixture(), mixed=fixture(['neural','club','beginner']), recovery=fixture(['neural','beginner','neural']);
const saved=p=>p.evaluate(key=>JSON.parse(localStorage.getItem(key)),SAVE_KEY);
const labels=p=>p.evaluate(()=>['right','across','left'].map(side=>document.querySelector(`.place.${side} .opponent-type`)?.dataset.controller));
const shot=(p,name)=>p.screenshot({path:resolve(output, name+'.png'),fullPage:true});
async function open(snapshot=initial,{width=1100,height=900,mock=true,fail=false,missing=false}={}) {
  const context=await browser.createBrowserContext();contexts.push(context);
  const p=await context.newPage();p.errors=[];p.modelLoads=0;
  p.on('pageerror',e=>p.errors.push(e.message));
  p.on('request',req=>{if(req.url().endsWith(`/${MODEL_FILES.full}`)&&req.method()==='GET')p.modelLoads++;});
  await p.setViewport({width,height,isMobile:width<500||height<500,hasTouch:width<500||height<500});
  await p.emulateMediaFeatures([{name:'prefers-reduced-motion',value:'reduce'}]);
  await p.evaluateOnNewDocument((key,settings,snapshot)=>{
    if(!localStorage.getItem(key))localStorage.setItem(key,JSON.stringify(snapshot));
    if(!localStorage.getItem(settings))localStorage.setItem(settings,JSON.stringify({version:1,difficulty:'club',hints:true,confirmDiscards:true,shortcuts:true}));
  },SAVE_KEY,SETTINGS_KEY,snapshot);
  // Do not let a cached genuine worker bypass intentional AI failure mocks.
  if(mock || missing) await p.evaluateOnNewDocument(() => Object.defineProperty(navigator, 'serviceWorker', { value: undefined }));
  await p.setRequestInterception(true);
  p.on('request',req=>{
    if(missing&&req.url().endsWith(`/${MODEL_FILES.full}`))void req.respond({status:404,body:''});
    else if(mock&&req.url().includes('/assets/policy.worker-'))void req.respond({status:200,contentType:'text/javascript',body:fail
      ? 'self.onmessage=({data:d})=>self.postMessage({id:d.id,error:"Test network failure"});'
      : 'self.onmessage=({data:d})=>self.postMessage({id:d.id,action:d.mask[43]?43:d.mask[45]?45:d.mask.findIndex(Boolean)});'});
    else void req.continue();
  });
  await p.goto(`http://127.0.0.1:${server.address().port}/mahjong/`,{waitUntil:'networkidle0'});
  await p.waitForSelector('.hand');await settled(p);return p;
}
async function settled(p) {
  await p.waitForFunction(()=>document.querySelector('.failure')||document.querySelector('.standings')||document.querySelector('.screen')
    ||document.querySelector('.hand button:not(:disabled)')||document.querySelector('.call-options button:not(:disabled)'),{timeout:45000});
}
async function check(name,fn) {
  try {await fn();results.push({name,passed:true});console.log('PASS '+name);}
  catch(error){results.push({name,passed:false,error:error.stack});console.error('FAIL '+name+'\n'+error.stack);}
  finally{while(contexts.length)await contexts.pop().close();}
}
async function custom(p) { await p.select('select[aria-label="opponent strength"]','custom');await p.waitForSelector('.custom-dialog[open]'); }
const confirmStart=p=>p.click('.custom-dialog .primary');
try {
  await mkdir(output,{recursive:true});await new Promise(done=>server.listen(0,'127.0.0.1',done));
  const chrome=process.env.CHROME_BIN||['/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(chrome);browser=await puppeteer.launch({executablePath:chrome,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
  await check('custom setup is transactional and cancel leaves the active match unchanged',async()=>{
    const p=await open(),before=await saved(p);await custom(p);
    await p.select('select[aria-label="Left opponent"]','beginner');
    await p.select('select[aria-label="Right opponent"]','neural');
    assert.deepEqual(await labels(p),['club','club','club']);assert.deepEqual(await saved(p),before);
    await p.click('.custom-dialog button:not(.primary)');
    assert.equal(await p.$('.custom-dialog[open]'),null);
    assert.equal(await p.$eval('select[aria-label="opponent strength"]',el=>el.value),'club');
    assert.deepEqual(await saved(p),before);assert.deepEqual(p.errors,[]);
  });
  await check('a mixed table can be created, labelled, edited and restored without changing controllers',async()=>{
    const p=await open();
    // This check needs an unfinished match with progress, not the untouched
    // opening deal (which intentionally needs no abandonment confirmation).
    await p.click('.hand button:not(:disabled)');await p.click('.confirm-discard .primary');
    await p.waitForFunction(key=>JSON.parse(localStorage.getItem(key)).commands.length>0,{},SAVE_KEY);
    await settled(p);await custom(p);
    await p.select('select[aria-label="Left opponent"]','beginner');await p.select('select[aria-label="Right opponent"]','neural');
    const dismiss=d=>void d.dismiss();p.on('dialog',dismiss);const before=await saved(p);
    await confirmStart(p);assert.deepEqual(await saved(p),before);assert.ok(await p.$('.custom-dialog[open]'));
    p.off('dialog',dismiss);p.on('dialog',d=>void d.accept());await confirmStart(p);await settled(p);
    assert.equal(await p.$('.custom-dialog[open]'),null);
    assert.deepEqual(await labels(p),['neural','club','beginner']);assert.deepEqual((await saved(p)).opponents,['neural','club','beginner']);
    const after=await saved(p);await p.reload({waitUntil:'networkidle0'});await settled(p);
    assert.deepEqual(await labels(p),['neural','club','beginner']);assert.equal((await saved(p)).state,after.state);
    await p.click('.edit-table');assert.ok(await p.$('.custom-dialog[open]'));await shot(p,'mixed-table-setup');assert.deepEqual(p.errors,[]);
  });
  await check('the quick all-opponents selector still sets three matching controllers',async()=>{
    const p=await open(mixed);p.on('dialog',d=>void d.accept());
    await p.select('select[aria-label="opponent strength"]','beginner');await settled(p);
    assert.deepEqual(await labels(p),['beginner','beginner','beginner']);assert.equal((await saved(p)).difficulty,'beginner');
    assert.equal(await p.$('.edit-table'),null);assert.deepEqual(p.errors,[]);
  });
  await check('one-opponent recovery preserves the other trained player and the Beginner',async()=>{
    const p=await open(recovery,{fail:true});
    // The human starts as East in seed 81. Trigger the next trained turn
    // before expecting the deliberately failing worker to need recovery.
    await p.click('.hand button:not(:disabled)');await p.click('.confirm-discard .primary');
    await p.waitForSelector('.failure');
    const before=await labels(p),snapshot=await saved(p);const seats=JSON.parse(snapshot.state)[0].seats;
    const label=await p.$eval('.recovery-actions button:nth-child(2)',el=>el.textContent);
    assert.match(label,/opponent only/);const position=/right/.test(label)?0:/opposite/.test(label)?1:2;
    await p.click('.recovery-actions button:nth-child(2)');await settled(p);
    const expected=[...before];expected[position]='club';assert.deepEqual(await labels(p),expected);
    assert.equal(expected.filter(t=>t==='neural').length,1);assert.equal(expected[1],'beginner');
    const command=(await saved(p)).commands.find(c=>c.type==='opponent-club');assert.equal(command.player,seats[position+1].player);
    const r=MatchSession.restore(Game,JSON.stringify(await saved(p)));try{assert.deepEqual(r.opponents,expected);}finally{r.dispose();}
    await shot(p,'mixed-table-recovery');assert.deepEqual(p.errors,[]);
  });
  await check('missing trained assets do not prevent mixing the two built-in types',async()=>{
    const p=await open(initial,{missing:true});await custom(p);
    assert.ok(await p.$eval('select[aria-label="Left opponent"] option[value=neural]',el=>el.disabled));
    await p.select('select[aria-label="Left opponent"]','beginner');p.on('dialog',d=>void d.accept());
    await confirmStart(p);await settled(p);assert.deepEqual(await labels(p),['club','club','beginner']);assert.deepEqual(p.errors,[]);
  });
  await check('two trained players use the real model with one shared model load',async()=>{
    const loadsBefore=modelGets;
    const p=await open(fixture(['neural','club','neural']),{mock:false});
    for(let n=0;n<8;n++) {
      assert.equal(await p.$('.failure'),null);const current=await saved(p);
      if(JSON.parse(current.state)[0].phase==='over')break;
      const choice=await p.$('.call-options button[data-choice="ron"],.call-options button[data-choice="tsumo"],.call-options button[data-choice="pass"]');
      if(choice)await choice.click();
      else {await p.click('.hand button:not(:disabled)');await p.click('.confirm-discard .primary');}
      await p.waitForFunction((key,count)=>JSON.parse(localStorage.getItem(key)).commands.length>count,{},SAVE_KEY,current.commands.length);
      await settled(p);
    }
    assert.equal(modelGets-loadsBefore,1);assert.equal(await p.$('.failure'),null);
    const snapshot=await saved(p);assert.ok(snapshot.commands.filter(c=>c.type==='opponent').length>=4);
    assert.deepEqual(await labels(p),['neural','club','neural']);assert.deepEqual(p.errors,[]);
    const r=MatchSession.restore(Game,JSON.stringify(snapshot));try{assert.equal(r.stateKey(),snapshot.state);}finally{r.dispose();}
    const replay=new MatchSession(Game,snapshot.seed,snapshot.opponents), players=new Set();
    try {
      replay.advance(false);
      for(const command of snapshot.commands) {
        if(command.type==='opponent')players.add(replay.engine.opponent_player());
        replay.apply(command);replay.advance(false);
      }
      assert.equal(players.size,2,'Both trained players must have used the shared model');
    } finally {replay.dispose();}
    await shot(p,'mixed-table-real-network');
  });
  for(const [width,height] of [[320,568],[375,667],[390,844],[568,320],[844,390],[1440,1000]]) {
    await check(`mixed table labels and custom controls fit ${width}x${height}`,async()=>{
      const p=await open(mixed,{width,height});assert.deepEqual(await labels(p),['neural','club','beginner']);
      assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      await shot(p,`mixed-table-${width}x${height}`);
      const compact=width<=760||(width>=640&&height<=500);
      if(compact)await p.click('.settings-trigger');
      await p.click('.edit-table');
      assert.ok(await p.$eval('.custom-dialog',el=>el.scrollWidth<=el.clientWidth+1));
      assert.ok(await p.$$eval('.custom-dialog select',els=>els.every(el=>el.getBoundingClientRect().height>=44)));
      await shot(p,`mixed-setup-${width}x${height}`);assert.deepEqual(p.errors,[]);
    });
  }
} finally {
  await writeFile(resolve(output,'mixed-opponents-report.json'),JSON.stringify(results,null,2));
  while(contexts.length)await contexts.pop().close();await browser?.close();if(server.listening)await new Promise(done=>server.close(done));
}
console.log(`${results.filter(r=>r.passed).length}/${results.length} mixed-opponent browser checks passed`);
if(results.some(r=>!r.passed))process.exitCode=1;
