/** Deterministic interaction regressions on the production build and real WASM.
 * CI serves it at /mahjong/, just like Pages; screenshots are evidence, not
 * substitutes for behavioural assertions. No test hooks ship in the app.
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
import { heldSafeCount, callLabel } from '../src/lib/ui.js';
await init({module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url))});
const web = fileURLToPath(new URL('../', import.meta.url));
const output = resolve(web, 'test-results');
await mkdir(output, {recursive:true});
const types = {'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.png':'image/png','.ico':'image/x-icon','.json':'application/json','.webmanifest':'application/manifest+json','.wasm':'application/wasm','.onnx':'application/octet-stream'};
const dist = resolve(web, 'dist');
const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    if (!url.pathname.startsWith('/mahjong/')) { response.writeHead(404).end(); return; }
    const relative = decodeURIComponent(url.pathname.slice('/mahjong/'.length)) || 'index.html';
    const file = resolve(dist, relative);
    if (!file.startsWith(dist + sep)) { response.writeHead(403).end(); return; }
    const body = await readFile(file);
    response.writeHead(200, {'Content-Type': types[extname(file)] || 'application/octet-stream','Cache-Control':'no-store'});
    response.end(request.method === 'HEAD' ? undefined : body);
  } catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/mahjong/`;
const chrome = process.env.CHROME_BIN || ['/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser'].find(existsSync);
assert.ok(chrome, 'Set CHROME_BIN to a Chromium/Chrome executable');
const browser = await puppeteer.launch({executablePath:chrome,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
const results = [];
const contexts = [];
const problems = new WeakMap();

function make(seed=369,difficulty='club') { const match=new MatchSession(Game,seed,difficulty); match.advance(false); return match; }
function step(match,choice) { match.apply({type:'choose',kind:choice.kind,tile:choice.tile}); match.advance(false); }
function chooseSimple(match) {
  const choices=match.choices;
  return choices.find(c=>c.kind==='ron'||c.kind==='tsumo') ?? choices.find(c=>c.kind==='pass')
    ?? choices.find(c=>c.kind==='discard' && c.tile===match.view.seats[0].drawn) ?? choices.find(c=>c.kind==='discard') ?? choices[0];
}
function findFixture(predicate) {
  for(let seed=1;seed<=70;seed++) {
    const match=make(seed);
    for(let n=0;n<80;n++) {
      if(predicate(match)) { const snapshot=match.snapshot(); match.dispose(); return snapshot; }
      if(match.view.phase==='over') break;
      step(match,chooseSimple(match));
    }
    match.dispose();
  }
  throw new Error('Could not find bounded deterministic fixture');
}
const duplicate=make();
const original=JSON.parse(readFileSync(new URL('../tests/fixtures/duplicate-indicators.json',import.meta.url)));
for(const action of original.actions) step(duplicate,action);
const duplicateSave=duplicate.snapshot(); duplicate.dispose();
const initialMatch=make(11); const initial=initialMatch.snapshot(); initialMatch.dispose();
const callSave=findFixture(m=>m.view.phase==='call'&&m.choices.some(c=>c.kind==='pon'||c.kind==='chii'));
const safeSave=findFixture(m=>heldSafeCount(m.view)>0 && m.view.safe.length!==heldSafeCount(m.view));
const endSave=findFixture(m=>m.view.phase==='over');
const dormant=make(1,'neural'); const neuralSave=dormant.snapshot(); dormant.dispose();

async function open(saved=initial, {width=1100,height=850,dark=false,confirm=false,mock=null}={}) {
  const context=await browser.createBrowserContext(); contexts.push(context);
  const page=await context.newPage(); const errors=[]; problems.set(page, errors);
  page.on('pageerror',error=>errors.push(error.message));
  await page.setViewport({width,height,isMobile:width<500 || height<500,hasTouch:width<500 || height<500,deviceScaleFactor:1});
  await page.emulateMediaFeatures([{name:'prefers-color-scheme',value:dark?'dark':'light'},{name:'prefers-reduced-motion',value:'reduce'}]);
  await page.evaluateOnNewDocument((saveKey,settingsKey,saved,confirm)=>{
    if(!localStorage.getItem(settingsKey)) localStorage.setItem(settingsKey,JSON.stringify({version:1,difficulty:'club',hints:true,confirmDiscards:confirm,shortcuts:true}));
    if(!localStorage.getItem(saveKey)) localStorage.setItem(saveKey,JSON.stringify(saved));
  }, SAVE_KEY, SETTINGS_KEY, saved, confirm);
  if(mock) {
    await page.setCacheEnabled(false);
    await page.setRequestInterception(true);
    page.on('request',request=>{
      if(request.url().includes('/assets/policy.worker-')) {
        const text=mock.delay ? `self.onmessage=({data:d})=>setTimeout(()=>self.postMessage({id:d.id,action:d.mask.findIndex(Boolean)}),${mock.delay});`
          : mock.fail ? 'self.onmessage=({data:d})=>self.postMessage({id:d.id,error:"Simulated network failure"});'
          : 'self.onmessage=({data:d})=>self.postMessage({id:d.id,action:d.mask.findIndex(Boolean)});';
        void request.respond({status:200,contentType:'text/javascript',body:text});
      } else void request.continue();
    });
  }
  await page.goto(url,{waitUntil:'networkidle0'});
  await page.waitForSelector('.hand');
  return page;
}
const saved=page=>page.evaluate(key=>JSON.parse(localStorage.getItem(key)),SAVE_KEY);
const ui=page=>page.evaluate(()=>({failure:document.querySelector('.failure')?.textContent ?? '',selected:document.querySelectorAll('.hand .selected').length,opponents:document.querySelector('select').value}));
async function shot(page,name) { await page.screenshot({path:resolve(output,`${name}.png`),fullPage:true}); }
function noErrors(page) { assert.deepEqual(problems.get(page),[]); }
async function check(name,fn) {
  try { await fn(); results.push({name,passed:true}); console.log(`PASS ${name}`); }
  catch(error) { results.push({name,passed:false,error:error.stack}); console.error(`FAIL ${name}\n${error.stack}`); }
}
function contrast(a,b) {
  const luminance=s=>s.match(/[\d.]+/g).slice(0,3).map(Number).map(x=>x/255).map(x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4).reduce((sum,x,i)=>sum+x*[.2126,.7152,.0722][i],0);
  const x=luminance(a),y=luminance(b);return(Math.max(x,y)+.05)/(Math.min(x,y)+.05);
}

try {
  await check('duplicate dora indicators render without duplicate keys',async()=>{
    const page=await open(duplicateSave);
    assert.equal(await page.$$eval('.indicators .tile',nodes=>nodes.length),2);
    assert.deepEqual(JSON.parse((await saved(page)).state)[0].dora_indicators,['7z','7z']);
    await shot(page,'duplicate-indicators'); noErrors(page);
  });
  await check('cancelled opponent change leaves game and selector unchanged',async()=>{
    const page=await open(duplicateSave); const before=await saved(page);
    page.once('dialog',dialog=>dialog.dismiss()); await page.select('select','beginner');
    assert.equal((await ui(page)).opponents,'club'); assert.deepEqual(await saved(page),before); noErrors(page);
  });
  await check('focused checkbox, browser modifiers and held keys cannot discard',async()=>{
    const page=await open(duplicateSave); await page.focus('.hand'); await page.keyboard.press('ArrowLeft');
    assert.equal((await ui(page)).selected,1);
    await page.click('.options summary'); const checkbox='.option-fields input[type=checkbox]';
    const before=await saved(page); await page.focus(checkbox); const checked=await page.$eval(checkbox,n=>n.checked);
    await page.keyboard.press('Space'); assert.equal(await page.$eval(checkbox,n=>n.checked),!checked);
    assert.deepEqual(await saved(page),before);
    await page.focus('.hand');
    await page.evaluate(()=>{
      const hand=document.querySelector('.hand');
      for(const modifiers of [{ctrlKey:true},{altKey:true},{metaKey:true},{repeat:true}]) hand.dispatchEvent(new KeyboardEvent('keydown',{key:'1',bubbles:true,cancelable:true,...modifiers}));
    });
    assert.deepEqual(await saved(page),before); noErrors(page);
  });
  await check('keyboard marker clears after a synchronous Club turn',async()=>{
    const page=await open(duplicateSave); const count=(await saved(page)).commands.length;
    await page.focus('.hand'); await page.keyboard.press('ArrowLeft'); await page.keyboard.press('Enter');
    await page.waitForFunction((key,count)=>JSON.parse(localStorage.getItem(key)).commands.length>count,{},SAVE_KEY,count);
    assert.equal((await ui(page)).selected,0); noErrors(page);
  });
  await check('match state and preferences survive a real page reload',async()=>{
    const page=await open(initial,{confirm:true}); await page.click('.options summary');
    await page.$eval('.option-fields input',n=>n.click());
    const tile=await page.$('.hand button:not(:disabled)'); await tile.click();
    assert.equal((await saved(page)).commands.length,0,'First tap only selects');
    await page.click('.confirm-discard .primary');
    const before=await saved(page); assert.equal(before.commands.length,1);
    await page.reload({waitUntil:'networkidle0'}); await page.waitForSelector('.hand');
    assert.deepEqual(await saved(page),before);
    assert.equal(await page.$eval('.option-fields input',n=>n.checked),false);
    assert.equal(await page.$$eval('.option-fields input',n=>n[1].checked),true);
    noErrors(page);
  });
  await check('unfinished match requires confirmation between hands',async()=>{
    const page=await open(endSave); const before=await saved(page); let prompted=false;
    page.once('dialog',dialog=>{prompted=true;return dialog.dismiss();}); await page.click('.restart');
    assert.equal(prompted,true); assert.deepEqual(await saved(page),before);
    await page.click('.screen .primary'); await page.waitForFunction(key=>JSON.parse(localStorage.getItem(key)).commands.at(-1).type==='next',{},SAVE_KEY);
    const next=await saved(page); await page.reload({waitUntil:'networkidle0'}); await page.waitForSelector('.hand');
    assert.deepEqual(await saved(page),next); noErrors(page);
  });
  await check('trained errors allow a fresh retry without relabelling the match',async()=>{
    const mock={fail:true}; const page=await open(neuralSave,{mock});
    await page.waitForSelector('.recovery-actions'); assert.equal((await ui(page)).opponents,'neural');
    mock.fail=false; await page.click('.recovery-actions button');
    await page.waitForFunction(()=>!document.querySelector('.failure') && [...document.querySelectorAll('.hand button, .call-options button')].some(n=>!n.disabled));
    assert.ok((await saved(page)).commands.length>neuralSave.commands.length);
    assert.equal((await ui(page)).opponents,'neural'); noErrors(page);
  });
  await check('restarting during a pending trained answer cannot mutate the replacement match',async()=>{
    const page=await open(neuralSave,{mock:{delay:1800}});
    page.on('dialog',dialog=>dialog.accept());
    await page.select('select','club');
    await page.waitForFunction(()=>document.querySelector('select').value==='club' && !document.querySelector('.failure'));
    const before=await saved(page); assert.notEqual(before.seed,neuralSave.seed);
    await new Promise(resolve=>setTimeout(resolve,2200));
    assert.deepEqual(await saved(page),before); assert.equal((await ui(page)).failure,''); noErrors(page);
  });
  await check('Club recovery continues and reloads the same neural-origin match',async()=>{
    const page=await open(neuralSave,{mock:{fail:true}}); await page.waitForSelector('.recovery-actions');
    await page.$$eval('.recovery-actions button',n=>n[1].click());
    await page.waitForFunction(()=>document.querySelector('select').value==='club'&&!document.querySelector('.failure'));
    const state=await saved(page); assert.equal(state.seed,neuralSave.seed); assert.equal(state.commands.at(-1).type,'club');
    await page.reload({waitUntil:'networkidle0'}); assert.deepEqual(await saved(page),state); noErrors(page);
  });
  await check('global dora stays marked after discarding the last held copy',async()=>{
    const page=await open(initial); await page.click('.hand button[data-tile="6z"]');
    await page.waitForFunction(key=>JSON.parse(localStorage.getItem(key)).commands.length===1,{},SAVE_KEY);
    await page.click('.own-discards summary');
    assert.ok(await page.$('.own-discards [aria-label="green dragon, dora"]'));
    noErrors(page);
  });
  await check('safe count and accessible tile text retain the riichi qualification',async()=>{
    const page=await open(safeSave); const view=JSON.parse(safeSave.state)[0];
    const text=await page.$eval('.safe-note',n=>n.textContent);
    assert.match(text,new RegExp(`^${heldSafeCount(view)} held`)); assert.match(text,/declared riichi/);
    const labels=await page.$$eval('.hand .safe',nodes=>nodes.map(n=>n.getAttribute('aria-label')));
    assert.equal(labels.length,heldSafeCount(view)); assert.ok(labels.every(text=>text.includes('not guaranteed against undeclared'))); noErrors(page);
  });
  await check('call decisions show the offered tile, discarder and meld previews',async()=>{
    const page=await open(callSave,{width:390,height:844}); const view=JSON.parse(callSave.state)[0];
    const text=await page.$eval('.offered-tile',n=>n.textContent);
    assert.ok(text.toLowerCase().includes(view.pending_from));
    assert.ok(await page.$('.call-preview .tile'));
    const choice=JSON.parse(callSave.state)[1].find(c=>c.kind==='pon'||c.kind==='chii');
    assert.ok(await page.$(`button[aria-label="${callLabel(choice)}"]`));
    await shot(page,'phone-call'); noErrors(page);
  });
  for(const [width,height] of [[320,568],[375,667],[390,844],[568,320],[844,390],[1440,1000]]) {
    await check(`usable layout and no horizontal overflow at ${width}x${height}`,async()=>{
      const page=await open(initial,{width,height,confirm:width<500 || height<500});
      await page.evaluate(()=>window.scrollTo(0,0));
      const layout=await page.evaluate(()=>{
        const rect=selector=>document.querySelector(selector).getBoundingClientRect().toJSON();
        return {width:innerWidth,overflow:document.documentElement.scrollWidth,hand:rect('.hand'),controls:rect('.controls'),restart:rect('.restart')};
      });
      await shot(page,`layout-${width}x${height}`);
      assert.ok(layout.overflow<=width+1,JSON.stringify(layout));
      if(width<500) assert.ok(layout.hand.bottom<=height && layout.controls.bottom<=height,`Hand below fold: ${JSON.stringify(layout)}`);
      if(width===844) assert.ok(layout.hand.top<height && layout.controls.top<height,JSON.stringify(layout));
      assert.ok(layout.restart.height>=44);
      await page.click('.guide summary');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Expanded help overflows');
      await page.click('.guide summary'); await page.click('.inspect');
      assert.equal(await page.$eval('dialog',n=>n.open),true); assert.equal(await page.$$eval('.inspection-grid section',n=>n.length),4);
      await page.keyboard.press('Escape'); assert.equal(await page.$eval('dialog',n=>n.open),false); noErrors(page);
    });
  }
  await check('dark-mode result button contrast exceeds 4.5:1',async()=>{
    const page=await open(endSave,{width:390,height:844,dark:true});
    const colors=await page.$eval('.screen .primary',node=>{const style=getComputedStyle(node);return [style.color,style.backgroundColor];});
    const ratio=contrast(...colors); assert.ok(ratio>=4.5,`${ratio} ${colors}`);
    await shot(page,'phone-dark-result'); noErrors(page);
  });
  await check('history is expandable and keeps chronological event order',async()=>{
    const page=await open(duplicateSave); await page.click('.history summary');
    const actual=await page.$$eval('.log p',nodes=>nodes.map(n=>n.textContent));
    const match=MatchSession.restore(Game,JSON.stringify(duplicateSave)); assert.deepEqual(actual,match.events);match.dispose(); noErrors(page);
  });
} finally {
  await writeFile(resolve(output,'ui-report.json'),JSON.stringify(results,null,2));
  for(const context of contexts) await context.close();
  await browser.close(); await new Promise(resolve=>server.close(resolve));
}
if(results.some(test=>!test.passed)) process.exitCode=1;
console.log(`${results.filter(test=>test.passed).length}/${results.length} browser regressions passed`);
