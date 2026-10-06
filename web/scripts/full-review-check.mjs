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
await init({module_or_path:readFileSync(new URL('../src/wasm/riichi_bg.wasm',import.meta.url))});
const web=fileURLToPath(new URL('../',import.meta.url)), dist=resolve(web,'dist'), output=resolve(web,'test-results');
await mkdir(output,{recursive:true});
const server=createServer(createFixtureHandler({root:dist,headers:{'Cache-Control':'no-store'}}));
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
async function open(snapshot,{width=1100,height=900,confirm=false,hints=true,files=false}={}){
 const page=await (await openContext(browser)).newPage();page.reviewErrors=[];
 page.on('pageerror',error=>page.reviewErrors.push(error.message));await page.setViewport({width,height});
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
 await page.evaluateOnNewDocument((key,settings,snapshot,confirm,hints)=>{
  if(!localStorage.getItem(key))localStorage.setItem(key,JSON.stringify(snapshot));
  localStorage.setItem(settings,JSON.stringify({version:1,difficulty:snapshot.difficulty,hints,confirmDiscards:confirm,shortcuts:true}));
 },SAVE_KEY,SETTINGS_KEY,snapshot,confirm,hints);
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
 browser=await launchChrome();
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
  assert.equal(await p.$('.screen [data-save-game]'),null,'the standings offer the whole game');
  file=await saveFrom(p,'.standings [data-save-game]');
  assert.match(file.name,fileName);assert.equal(file.text,whole.text+'\n');
  events=lines(file.text);opensAtEastOne(events);
  assert.equal(events.filter(event=>event.type==='start_kyoku').length,view.hands_played);
  assert.equal(events.at(-1).type,'end_game');
  await shot(p,'export-standings');noErrors(p);
 });
 for(const [width,height] of [[320,568],[390,844],[844,390]])await check(`ura and final-hand results fit ${width}x${height}`,async()=>{
  for(const [label,snapshot] of [['ura',wins[1]],['final',final.after]]){
   const p=await open(snapshot,{width,height});
   const overflow=await p.evaluate(()=>({viewport:innerWidth,document:document.documentElement.scrollWidth,wide:[...document.querySelectorAll('*')].map(el=>{const r=el.getBoundingClientRect();return{tag:el.tagName,cls:el.className?.toString?.()??'',left:r.left,right:r.right,width:r.width,scroll:el.scrollWidth,client:el.clientWidth};}).filter(x=>x.right>innerWidth+1||x.left<-1||x.scroll>x.client+1).sort((a,b)=>Math.max(b.right-innerWidth,b.scroll-b.client)-Math.max(a.right-innerWidth,a.scroll-a.client)).slice(0,12)}));
   assert.ok(overflow.document<=overflow.viewport+1,JSON.stringify(overflow));
   assert.ok(await p.$$eval('.screen,.standings,.bonus-indicators',els=>els.every(el=>el.scrollWidth<=el.clientWidth+1)));
   // Saving the game as well as the hand takes no extra row over the result.
   const rows=await p.$$eval('.screen .buttons button',els=>new Set(els.map(el=>Math.round(el.getBoundingClientRect().top))).size);
   assert.ok(rows<=3,`${rows} rows of buttons`);
   await shot(p,`${label}-${width}x${height}`);noErrors(p);
  }
 });
}finally{
 await browser?.close();if(server.listening)await new Promise(done=>server.close(done));
 await report('full-review browser checks',resolve(output,'full-review-report.json'));
}
