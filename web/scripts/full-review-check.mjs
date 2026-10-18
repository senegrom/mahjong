/** Full-app regressions on the production build. No test hooks are shipped.
 * The fixtures are legal commands replayed by the actual rebuilt WASM engine.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdir } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import { createFixtureHandler } from './static-fixture-server.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { matchLabels, playerNames } from '../src/lib/game-log.js';
import { MANIFEST } from '../src/lib/model-manifest.js';
await init({module_or_path:readFileSync(new URL('../src/wasm/riichi_bg.wasm',import.meta.url))});
const web=fileURLToPath(new URL('../',import.meta.url)), dist=resolve(web,'dist'), output=resolve(web,'test-results');
await mkdir(output,{recursive:true});
// The trained runtime can be held back, so that a review by Trained AI, the
// default adviser, stays under way while a check looks at it. No check
// downloads the network: its host is out of the browser's reach.
const networkUrl=`${MANIFEST.origin}/${MANIFEST.object}`;
const runtime=new Set(JSON.parse(readFileSync(resolve(dist,'offline-manifest.json'),'utf8')).entries
 .filter(entry=>entry.group==='ai'&&entry.url.endsWith('.wasm')).map(entry=>entry.url));
let holding=false;const held=new Set();
const server=createServer(createFixtureHandler({root:dist,headers:{'Cache-Control':'no-store'},intercept:async name=>{
 if(holding&&runtime.has(name))await new Promise(done=>held.add(done));
}}));
function release(){holding=false;for(const done of held)done();held.clear();}
let browser;
const {check,openContext,report}=browserChecks();
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
// The same game three moves into its third hand.
const midway=(()=>{
 const m=make(14,'beginner');
 try{
  for(let n=0;n<2;n++){finishHand(m,false);m.apply({type:'next'});m.advance(false);}
  for(let n=0;n<3;n++){const c=m.choices,q=c.find(c=>c.kind==='pass')??c.find(c=>c.kind==='discard');step(m,q.kind,q.tile);}
  assert.notEqual(m.view.phase,'over');return m.snapshot();
 }finally{m.dispose();}
})();
// What the page exports from a save: the same engine, replaying the same
// commands, asked with the names the page gives the players.
function exported(snapshot){
 const m=MatchSession.restore(Game,JSON.stringify(snapshot));
 try{return {text:m.engine.game_log(playerNames(m.view,matchLabels(m.initialOpponents,m.opponents))),deal:m.engine.log().split('\n')[0],hands:m.engine.game_log_hands()};}
 finally{m.dispose();}
}
const saved=page=>page.evaluate(key=>JSON.parse(localStorage.getItem(key)),SAVE_KEY);
/** A page restored from the snapshot. `touch` makes it a touchscreen phone's;
 * `reduced` asks for reduced motion, so nothing moves while a check measures
 * where the page is; `adviser` null leaves the review's adviser at its
 * default, Trained AI, which `trained` puts on offer; `failReview` has the
 * engine fail to build the hand's review. */
async function open(snapshot,{width=1100,height=900,confirm=false,hints=true,files=false,touch=false,reduced=false,adviser='club',trained=false,failReview=false}={}){
 const page=await (await openContext(browser)).newPage();page.reviewErrors=[];
 page.on('pageerror',error=>page.reviewErrors.push(error.message));await page.setViewport({width,height,isMobile:touch,hasTouch:touch});
 if(reduced)await page.emulateMediaFeatures([{name:'prefers-reduced-motion',value:'reduce'}]);
 // The network answers the page's availability probe, so Trained AI is on
 // offer, but its download never ends.
 if(trained)await page.evaluateOnNewDocument(url=>{
  const realFetch=window.fetch;
  window.fetch=(input,options)=>{
   const target=input instanceof Request?input.url:String(input);
   if(target!==url)return realFetch(input,options);
   const method=String(options?.method??(input instanceof Request?input.method:'GET')).toUpperCase();
   return method==='HEAD'?Promise.resolve(new Response(null,{status:200})):new Promise(()=>{});
  };
 },networkUrl);
 // The engine fails to build the hand's review, as a broken one would: its
 // export behind Game.review throws, replaced as the page instantiates it.
 if(failReview)await page.evaluateOnNewDocument(()=>{
  const plant=result=>{
   if(!result?.instance?.exports?.game_review)return result;
   const exports={...result.instance.exports,game_review(){throw new Error('A planted review failure');}};
   return {module:result.module,instance:{exports}};
  };
  for(const name of ['instantiateStreaming','instantiate']){
   const original=WebAssembly[name].bind(WebAssembly);
   WebAssembly[name]=async(...args)=>plant(await original(...args));
  }
 });
 // A saved file is kept in the page, name and text, rather than downloaded.
 if(files)await page.evaluateOnNewDocument(()=>{
  const blobs=new Map(),create=URL.createObjectURL.bind(URL);
  URL.createObjectURL=object=>{const url=create(object);blobs.set(url,object);return url;};
  window.savedFiles=[];
  HTMLAnchorElement.prototype.click=function(){
   const file={name:this.download,text:null};window.savedFiles.push(file);
   void blobs.get(this.href)?.text().then(text=>{file.text=text;});
  };
 });
 await page.evaluateOnNewDocument((key,settings,snapshot,confirm,hints,adviser)=>{
  if(!localStorage.getItem(key))localStorage.setItem(key,JSON.stringify(snapshot));
  // Club reviews unless a check asks for the default: these checks open a
  // review without waiting on the network.
  const preferences={version:2,difficulty:snapshot.difficulty,hints,confirmDiscards:confirm,shortcuts:true};
  if(adviser)preferences.reviewAdviser=adviser;
  localStorage.setItem(settings,JSON.stringify(preferences));
 },SAVE_KEY,SETTINGS_KEY,snapshot,confirm,hints,adviser);
 await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/`,{waitUntil:'networkidle0'});
 await page.waitForSelector('.hand');return page;
}
const noErrors=page=>assert.deepEqual(page.reviewErrors,[]);
const shot=(page,name)=>page.screenshot({path:resolve(output,name+'.png'),fullPage:true});
async function saveFrom(page,selector){
 const before=await page.evaluate(()=>window.savedFiles.length);
 await page.click(selector);
 await page.waitForFunction(n=>window.savedFiles.length>n&&window.savedFiles.at(-1).text!==null,{},before);
 return page.evaluate(()=>window.savedFiles.at(-1));
}
const lines=text=>text.trimEnd().split('\n').map(line=>JSON.parse(line));
try{
 await new Promise(done=>server.listen(0,'127.0.0.1',done));
 browser=await launchChrome({args:[`--host-resolver-rules=MAP ${new URL(networkUrl).hostname} ~NOTFOUND`]});
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
 await check('a yaku explains itself and lifts the tiles that make it',async()=>{
  // Prefer a hand whose yaku has a shape, so the lifting is exercised and
  // not only the sentence.
  const SHAPED=['All Simples','Pinfu','Pure Double Sequence','Half Flush','Full Flush','All Triplets',
   'Mixed Triple Sequence','Pure Straight','Dragon Triplet','Seven Pairs','Half Outside Hand',
   'Seat Wind Triplet','Round Wind Triplet','Full Outside Hand','Triple Triplet'];
  const shapedFirst=[...wins].sort((a,b)=>{
   const has=snapshot=>JSON.parse(snapshot.state)[0].outcome.wins
    .some(win=>win.yaku.some(yaku=>SHAPED.includes(yaku.name)))?0:1;
   return has(a)-has(b);
  });
  const snapshot=shapedFirst[0],view=JSON.parse(snapshot.state)[0];
  const p=await open(snapshot,{hints:false,width:1100,height:900});
  const names=await p.$$eval('.win .yaku-name span:first-child',els=>els.map(el=>el.textContent.trim()));
  assert.deepEqual(names,view.outcome.wins[0].yaku.map(yaku=>yaku.name),'every yaku is a control');
  assert.equal(await p.$('.yaku-note'),null,'nothing is explained until it is asked about');
  // Hovering the name explains it; a yaku with a shape lifts its own tiles.
  const shaped=await p.evaluate(list=>{
   const rows=[...document.querySelectorAll('.win .yaku-name')];
   return rows.findIndex(row=>list.includes(row.querySelector('span').textContent.trim()));
  },SHAPED);
  const index=shaped>=0?shaped:0;
  await p.$$eval('.win .yaku-name',(els,i)=>els[i].dispatchEvent(new MouseEvent('mouseenter')),index);
  await p.waitForSelector('.yaku-note');
  const note=await p.$eval('.yaku-note',el=>el.textContent.trim());
  assert.ok(note.length>20,'the explanation is a sentence');
  const lit=await p.$$eval('.win .tile.in-shape',els=>els.length);
  if(shaped>=0)assert.ok(lit>0,`${names[index]} lights the tiles that make it`);
  await shot(p,'yaku-explained');
  // Leaving puts the hand back as it was.
  await p.$$eval('.win .yaku-name',(els,i)=>els[i].dispatchEvent(new MouseEvent('mouseleave')),index);
  await p.waitForFunction(()=>!document.querySelector('.yaku-note'));
  assert.equal(await p.$$eval('.win .tile.in-shape',els=>els.length),0);
  noErrors(p);
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
 // The result's buttons, and the headings they hand focus to as they go. A
 // press by keyboard and one by a mouse or a finger both move focus, but only
 // the keyboard's moves the page, to show where focus went.
 const SIZES=[[1100,900],[390,844],[320,568],[844,390],[568,320]];
 const resultButton=(p,text)=>p.evaluateHandle(text=>[...document.querySelectorAll('.screen .buttons button')].find(el=>el.textContent.trim()===text),text);
 const focusOn=p=>p.evaluate(()=>{const a=document.activeElement;return a?.matches('.review h3')?'the review':a?.matches('.standings h2')?'the standings'
  :a?.matches('.screen')?'the result':a?.matches('.result-chip')?'the chip':!a||a===document.body?'the page':a.outerHTML.slice(0,80);});
 const ringed=p=>p.evaluate(()=>document.activeElement?.matches(':focus-visible')??false);
 const sheetShown=p=>p.$eval('.screen',el=>getComputedStyle(el).display!=='none');
 const settle=p=>p.evaluate(()=>new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done))));
 // In view, and not under the result's chip or the game's bar.
 const uncovered=handle=>handle.evaluate(el=>{const box=el.getBoundingClientRect(),hit=document.elementFromPoint(box.left+box.width/2,box.top+box.height/2);
  return box.top>=0&&box.bottom<=innerHeight&&Boolean(hit)&&(hit===el||el.contains(hit));});
 // The ring that a keyboard's focus shows, drawn just outside the element,
 // and in view with it.
 const ringInView=handle=>handle.evaluate(el=>{const style=getComputedStyle(el),reach=parseFloat(style.outlineOffset)+parseFloat(style.outlineWidth),box=el.getBoundingClientRect();
  return el.matches(':focus-visible')&&style.outlineStyle!=='none'&&box.top-reach>=0&&box.bottom+reach<=innerHeight;});
 // The button where the player sees it, scrolled to first if need be.
 async function inReach(button){
  if(!await uncovered(button))await button.evaluate(el=>el.scrollIntoView({block:'center'}));
  assert.ok(await uncovered(button),'the button is in view to be pressed');
  return button;
 }
 // A mouse's click or a finger's tap on the button's middle, which scrolls
 // nothing first.
 async function pointerPress(p,button,touch){
  const box=await button.boundingBox(),x=box.x+box.width/2,y=box.y+box.height/2;
  if(touch)await p.touchscreen.tap(x,y);else await p.mouse.click(x,y);
 }
 // The result's review button goes as it opens the hand's review, and on a
 // phone, where the result is a sheet, the sheet steps aside with it. Pressed
 // by keyboard, focus must go on to the review's heading, which a screen
 // reader names, and the heading must be brought into view with its focus
 // ring, clear of the result's chip and the game's bar, with no chip's height
 // kept free below it where no chip sits. Focus must not drop to the page,
 // and the control before the review must lead back to the result: the
 // sheet's chip, which hands focus back to the result, or else the result's
 // own buttons.
 for(const [label,snapshot,pressed,sizes] of [['a hand',wins[0],'View table / my hand',SIZES],
  ['the last hand',final.before,'View table / my hand',[[1100,900],[390,844]]],['the final hand',final.after,'Review final hand',SIZES]])
  await check(`${pressed}, pressed by keyboard on ${label}'s result, hands focus to the review and back, at desktop and phone sizes`,async()=>{
  for(const [width,height] of sizes){
   const p=await open(snapshot,{width,height,reduced:true}),where=`at ${width}x${height}`;
   // The result takes focus as it appears, so the button is pressed after.
   await p.waitForFunction(()=>document.activeElement?.matches('.screen'));
   const sheet=await p.$eval('.screen',el=>getComputedStyle(el).position==='fixed');
   assert.equal(sheet,width<=760,`the result is a sheet up to 760 pixels wide, and only there, ${where}`);
   const button=await resultButton(p,pressed);
   assert.ok(await button.evaluate(el=>el instanceof HTMLButtonElement),`the result offers ${pressed} ${where}`);
   await button.focus();await p.keyboard.press('Enter');await p.waitForSelector('.review');
   if(sheet)assert.equal(await sheetShown(p),false,`the sheet steps aside ${where}`);
   await p.waitForFunction(()=>document.activeElement?.matches('.review h3'),{timeout:5000}).catch(()=>{});
   assert.equal(await focusOn(p),'the review',`focus goes on to the review ${where}`);
   // What a screen reader says of it: the hand's review, by its heading.
   const heading=await p.$('.review h3'),node=await p.accessibility.snapshot({root:heading,interestingOnly:false});
   assert.deepEqual([node?.role,node?.name,node?.level],['heading','Your hand, looked at again',3]);
   assert.equal(await heading.evaluate(el=>el.closest('section').getAttribute('aria-label')),'your decisions this hand');
   await settle(p);
   assert.ok(await uncovered(heading),`the review's heading is in view and uncovered ${where}`);
   assert.ok(await ringInView(heading),`the heading's focus ring is drawn, and in view ${where}`);
   // It lands clear of the chip where one sits; where none does, no chip's
   // height is kept free below it, only its ring's width and a little.
   const landed=await heading.evaluate(el=>{const chip=document.querySelector('.result-chip');
    return {innerHeight,bottom:el.getBoundingClientRect().bottom,margin:parseFloat(getComputedStyle(el).scrollMarginBottom),
     chip:chip&&getComputedStyle(chip).display!=='none'?chip.getBoundingClientRect().top:null};});
   if(landed.chip!==null)assert.ok(landed.bottom<=landed.chip,`the heading lands above the chip ${where}: ${JSON.stringify(landed)}`);
   else assert.ok(landed.margin<=12,`no chip's height is kept below the heading where no chip sits, ${where}: ${JSON.stringify(landed)}`);
   await shot(p,`result-review-focus-${label.replaceAll(' ','-')}-${width}x${height}`);
   // The way back to the result is the control before the review.
   await p.keyboard.down('Shift');await p.keyboard.press('Tab');await p.keyboard.up('Shift');
   if(!sheet)assert.ok(await p.evaluate(()=>document.activeElement?.matches('.screen button')),`Shift+Tab goes back to the result's buttons ${where}`);
   else{
    assert.equal(await focusOn(p),'the chip',`Shift+Tab reaches the result's chip ${where}`);
    await p.keyboard.press('Enter');
    await p.waitForFunction(()=>getComputedStyle(document.querySelector('.screen')).display!=='none',{timeout:5000});
    await p.waitForFunction(()=>document.activeElement?.matches('.screen'),{timeout:5000}).catch(()=>{});
    assert.equal(await focusOn(p),'the result',`the chip hands focus back to the result ${where}`);
   }
   noErrors(p);
  }
 });
 // Pressed by a mouse or a finger, the review's heading takes focus as well,
 // with no ring drawn, but the page stays where it was: the player asked to
 // see the table and their hand, which on a phone the sheet uncovered, and
 // has no ring to follow below them. (Held sideways on a wide phone, at
 // 844x390, the browser's own scroll anchoring moves the page as the review
 // opens, for every kind of press, so that size is not measured here.)
 for(const [label,snapshot,pressed,sizes] of [
  ['a hand',wins[0],'View table / my hand',[[1100,900,false],[390,844,true],[375,667,true],[360,640,true],[320,568,true],[568,320,true]]],
  ['the final hand',final.after,'Review final hand',[[1100,900,false],[390,844,true],[320,568,true]]]])
  await check(`${pressed}, pressed by mouse or touch on ${label}'s result, hands focus to the review and leaves the page where it was`,async()=>{
  for(const [width,height,touch] of sizes){
   const p=await open(snapshot,{width,height,touch,reduced:true}),where=`at ${width}x${height} by ${touch?'touch':'mouse'}`;
   await p.waitForFunction(()=>document.activeElement?.matches('.screen'));
   const button=await inReach(await resultButton(p,pressed)),before=await p.evaluate(()=>scrollY);
   await pointerPress(p,button,touch);await p.waitForSelector('.review');
   await p.waitForFunction(()=>document.activeElement?.matches('.review h3'),{timeout:5000}).catch(()=>{});
   assert.equal(await focusOn(p),'the review',`focus goes on to the review ${where}`);
   assert.equal(await ringed(p),false,`no focus ring is drawn for a pointer ${where}`);
   await settle(p);
   assert.equal(await p.evaluate(()=>scrollY),before,`the page stays where it was ${where}`);
   if(width<=760)assert.equal(await sheetShown(p),false,`the sheet steps aside ${where}`);
   noErrors(p);
  }
 });
 // Most players review with the default adviser, Trained AI, whose review is
 // still under way as the review opens, its region busy: the heading must
 // take focus and be named all the same.
 await check('View table / my hand, pressed by keyboard while Trained AI reviews, hands focus to the busy review',async()=>{
  for(const [width,height] of [[1100,900],[390,844],[320,568]]){
   const p=await open(wins[0],{width,height,reduced:true,adviser:null,trained:true}),where=`at ${width}x${height}`;
   try{
    // Trained AI is on offer before the review opens, as on a real visit.
    await p.waitForSelector('.opponents option[value="neural"]');
    await p.waitForFunction(()=>document.activeElement?.matches('.screen'));
    holding=true;
    const button=await resultButton(p,'View table / my hand');await button.focus();await p.keyboard.press('Enter');
    await p.waitForSelector('.review[aria-busy="true"]');
    assert.equal(await p.$eval('.review select',el=>el.value),'strong',`Trained AI reviews ${where}`);
    await p.waitForFunction(()=>document.activeElement?.matches('.review h3'),{timeout:5000}).catch(()=>{});
    assert.equal(await focusOn(p),'the review',`focus goes on to the busy review ${where}`);
    const heading=await p.$('.review h3'),node=await p.accessibility.snapshot({root:heading,interestingOnly:false});
    assert.deepEqual([node?.role,node?.name,node?.level,node?.focused],['heading','Your hand, looked at again',3,true]);
    assert.match(await p.$eval('.review [role="status"]',el=>el.textContent.trim()),/^Trained AI is reviewing your decisions/);
    await settle(p);
    assert.ok(await uncovered(heading),`the review's heading is in view and uncovered ${where}`);
    assert.ok(await ringInView(heading),`the heading's focus ring is drawn, and in view ${where}`);
    assert.equal(await p.$eval('.review',el=>el.getAttribute('aria-busy')),'true',`the review is still under way ${where}`);
    noErrors(p);
   }finally{release();}
  }
 });
 // Next hand on the last hand brings the standings in place of another
 // deal, and its button goes for the Final hand caption. Pressed by
 // keyboard, focus must go on to the standings' heading, which says where
 // the player finished, brought into view and uncovered: on a phone the
 // result, a sheet over the standings, steps aside for them, and its chip,
 // after the standings' buttons, brings it back with focus.
 await check('Next hand on the last hand, pressed by keyboard, hands focus to the standings, at desktop and phone sizes',async()=>{
  for(const [width,height] of SIZES){
   const p=await open(final.before,{width,height,reduced:true}),where=`at ${width}x${height}`,sheet=width<=760;
   await p.waitForFunction(()=>document.activeElement?.matches('.screen'));
   const button=await resultButton(p,'Next hand');await button.focus();await p.keyboard.press('Enter');
   await p.waitForSelector('.standings');
   await p.waitForFunction(()=>document.activeElement?.matches('.standings h2'),{timeout:5000}).catch(()=>{});
   assert.equal(await focusOn(p),'the standings',`focus goes on to the standings ${where}`);
   const heading=await p.$('.standings h2'),node=await p.accessibility.snapshot({root:heading,interestingOnly:false});
   assert.deepEqual([node?.role,node?.level],['heading',2]);assert.match(node?.name??'',/^You finished /);
   assert.equal(await heading.evaluate(el=>el.closest('section').getAttribute('aria-label')),'final standings');
   if(sheet)assert.equal(await sheetShown(p),false,`the sheet steps aside for the standings ${where}`);
   await settle(p);
   assert.ok(await uncovered(heading),`the standings' heading is in view and uncovered ${where}`);
   assert.ok(await ringInView(heading),`the heading's focus ring is drawn, and in view ${where}`);
   await shot(p,`standings-focus-${width}x${height}`);
   if(sheet){
    for(let presses=0;presses<6&&await focusOn(p)!=='the chip';presses++)await p.keyboard.press('Tab');
    assert.equal(await focusOn(p),'the chip',`Tab reaches the result's chip after the standings' buttons ${where}`);
    await p.keyboard.press('Enter');
    await p.waitForFunction(()=>document.activeElement?.matches('.screen'),{timeout:5000}).catch(()=>{});
    assert.equal(await focusOn(p),'the result',`the chip hands focus back to the result ${where}`);
   }
   noErrors(p);
  }
 });
 // Pressed by a mouse or a finger, the standings' heading takes focus as
 // well, with no ring drawn, and the page and the result stay as they were:
 // on a phone the result stays over the standings.
 await check('Next hand on the last hand, pressed by mouse or touch, hands focus to the standings and leaves the page where it was',async()=>{
  for(const [width,height,touch] of [[1100,900,false],[390,844,true],[320,568,true],[568,320,true]]){
   const p=await open(final.before,{width,height,touch,reduced:true}),where=`at ${width}x${height} by ${touch?'touch':'mouse'}`;
   await p.waitForFunction(()=>document.activeElement?.matches('.screen'));
   const button=await inReach(await resultButton(p,'Next hand')),before=await p.evaluate(()=>scrollY);
   await pointerPress(p,button,touch);await p.waitForSelector('.standings');
   await p.waitForFunction(()=>document.activeElement?.matches('.standings h2'),{timeout:5000}).catch(()=>{});
   assert.equal(await focusOn(p),'the standings',`focus goes on to the standings ${where}`);
   assert.equal(await ringed(p),false,`no focus ring is drawn for a pointer ${where}`);
   await settle(p);
   assert.equal(await p.evaluate(()=>scrollY),before,`the page stays where it was ${where}`);
   if(width<=760)assert.equal(await sheetShown(p),true,`the result stays over the standings ${where}`);
   noErrors(p);
  }
 });
 // A notice is spoken from a status line, and a screen reader hears a status
 // line only once it knows it: the line waits in the page, empty and taking
 // no room, before its words arrive. A hand's review that the engine could
 // not build is one such notice.
 await check('the notice that the review could not be built is spoken from a status line already in the page',async()=>{
  const p=await open(wins[0],{reduced:true,failReview:true});
  await p.waitForFunction(()=>document.activeElement?.matches('.screen'));
  const line=await p.$('[data-notice]');
  assert.ok(line,'the notice line is in the page before its words arrive');
  assert.deepEqual(await line.evaluate(el=>[el.getAttribute('role'),el.textContent.trim(),el.getBoundingClientRect().height<=1]),['status','',true],
   'the line waits empty, taking no room');
  assert.equal((await p.accessibility.snapshot({root:line,interestingOnly:false}))?.role,'status','a screen reader knows the line');
  const button=await resultButton(p,'View table / my hand');await button.focus();await p.keyboard.press('Enter');
  await p.waitForFunction(()=>document.querySelector('[data-notice]')?.textContent.trim(),{timeout:5000});
  assert.deepEqual(await line.evaluate(el=>[el.isConnected,el.textContent.trim(),el.getBoundingClientRect().height>1]),
   [true,'The review could not be built. Your match is unchanged.',true],'the same line speaks, and is seen');
  assert.equal(await p.$('.review'),null,'no review opens');noErrors(p);
 });
 await check('the game exports from East 1 from the settings, the score screen and the standings, never with the hand in play',async()=>{
  const fileName=/^riichi-game-\d{4}-\d{2}-\d{2}-\d{6}\.mjai\.jsonl$/;
  const opensAtEastOne=events=>{
   assert.equal(events[0].type,'start_game');assert.equal(events[0].names.filter(name=>name==='You').length,1);
   const deal=events.find(event=>event.type==='start_kyoku');
   assert.deepEqual([deal.bakaze,deal.kyoku,deal.honba,deal.oya],['E',1,0,0]);
  };
  // Three moves into the third hand: the two finished hands, from the
  // settings, and nothing of the hand on the table.
  const mid=exported(midway);assert.equal(mid.hands,2);
  let p=await open(midway,{files:true});
  await p.click('.game-export > summary');
  assert.match(await p.$eval('.game-export',el=>el.textContent),/hand being played is left out until it ends/);
  assert.equal(await p.$eval('.game-export [data-save-game]',el=>el.textContent.trim()),'Save 2 finished hands');
  let file=await saveFrom(p,'.game-export [data-save-game]');
  assert.match(file.name,fileName);assert.equal(file.text,mid.text+'\n');
  assert.ok(!file.text.includes(mid.deal),'the deal of the hand in play is not in the file');
  let events=lines(file.text);opensAtEastOne(events);
  assert.equal(events.filter(event=>event.type==='start_kyoku').length,2);
  assert.equal(events.at(-1).type,'end_kyoku');
  assert.equal(events.filter(event=>event.type==='start_game').length,1);
  await shot(p,'export-settings');noErrors(p);
  // At the score screen of the last hand: every hand, that one included,
  // and no close, since the game is not over until the hand is put away.
  const last=exported(final.before);
  p=await open(final.before,{files:true});
  assert.equal(await p.$eval('.screen [data-save-game]',el=>el.innerText.trim()),'Save game so far');
  file=await saveFrom(p,'.screen [data-save-game]');
  assert.match(file.name,fileName);assert.equal(file.text,last.text+'\n');
  events=lines(file.text);opensAtEastOne(events);
  assert.ok(file.text.includes(last.deal),'the finished hand is in the file');
  assert.equal(events.filter(event=>event.type==='start_kyoku').length,last.hands);
  assert.notEqual(events.at(-1).type,'end_game');noErrors(p);
  // At the standings: the whole game, close and all.
  const whole=exported(final.after),view=JSON.parse(final.after.state)[0];
  p=await open(final.after,{files:true});await p.waitForSelector('.standings');
  // Beside the standings the final result does not offer the file again.
  assert.equal(await p.$eval('.screen [data-save-game]',el=>el.getClientRects().length),0,'the standings offer the whole game');
  file=await saveFrom(p,'.standings [data-save-game]');
  assert.match(file.name,fileName);assert.equal(file.text,whole.text+'\n');
  events=lines(file.text);opensAtEastOne(events);
  assert.equal(events.filter(event=>event.type==='start_kyoku').length,view.hands_played);
  assert.equal(events.at(-1).type,'end_game');
  await shot(p,'export-standings');noErrors(p);
 });
 await check('on a phone the final result, which covers the standings, saves the whole game itself',async()=>{
  const whole=exported(final.after);
  for(const [width,height,label] of [[390,844,'Save game'],[320,568,'Save game']]){
   const p=await open(final.after,{width,height,files:true});await p.waitForSelector('.standings');
   assert.equal(await p.$eval('.screen [data-save-game]',el=>el.innerText.trim()),label);
   const file=await saveFrom(p,'.screen [data-save-game]');
   assert.match(file.name,/^riichi-game-\d{4}-\d{2}-\d{2}-\d{6}\.mjai\.jsonl$/);
   assert.equal(file.text,whole.text+'\n');assert.equal(lines(file.text).at(-1).type,'end_game');
   await shot(p,`export-final-${width}x${height}`);noErrors(p);
  }
 });
 for(const [width,height] of [[320,568],[390,844],[844,390]])await check(`ura and final-hand results fit ${width}x${height}`,async()=>{
  for(const [label,snapshot] of [['ura',wins[1]],['final',final.after]]){
   const p=await open(snapshot,{width,height});
   const overflow=await p.evaluate(()=>({viewport:innerWidth,document:document.documentElement.scrollWidth,wide:[...document.querySelectorAll('*')].map(el=>{const r=el.getBoundingClientRect();return{tag:el.tagName,cls:el.className?.toString?.()??'',left:r.left,right:r.right,width:r.width,scroll:el.scrollWidth,client:el.clientWidth};}).filter(x=>x.right>innerWidth+1||x.left<-1||x.scroll>x.client+1).sort((a,b)=>Math.max(b.right-innerWidth,b.scroll-b.client)-Math.max(a.right-innerWidth,a.scroll-a.client)).slice(0,12)}));
   assert.ok(overflow.document<=overflow.viewport+1,JSON.stringify(overflow));
   assert.ok(await p.$$eval('.screen,.standings,.bonus-indicators',els=>els.every(el=>el.scrollWidth<=el.clientWidth+1)));
   // Saving the game as well as the hand takes no extra row over the result.
   const rows=await p.$$eval('.screen .buttons button',els=>new Set(els.filter(el=>el.getClientRects().length).map(el=>Math.round(el.getBoundingClientRect().top))).size);
   assert.ok(rows<=3,`${rows} rows of buttons`);
   await shot(p,`${label}-${width}x${height}`);noErrors(p);
  }
 });
}finally{
 release();await browser?.close();if(server.listening)await new Promise(done=>server.close(done));
 await report('full-review browser checks',resolve(output,'full-review-report.json'));
}
