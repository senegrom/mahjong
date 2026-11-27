import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserChecks, launchChrome, openPanel, wordsContrast } from './browser-harness.mjs';
import { createFixtureHandler } from './static-fixture-server.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { unseenTileCounts } from '../src/lib/ui.js';
import { MANIFEST } from '../src/lib/model-manifest.js';

await init({module_or_path:readFileSync(new URL('../src/wasm/riichi_bg.wasm',import.meta.url))});
const web=fileURLToPath(new URL('../',import.meta.url)),output=resolve(web,'test-results');
const server=createServer(createFixtureHandler({root:resolve(web,'dist')}));let browser;const {check,openContext,report}=browserChecks();
function opening(){const m=new MatchSession(Game,81,'club');try{m.advance(false);return {snapshot:m.snapshot(),view:m.view};}finally{m.dispose();}}
function calling(){const m=new MatchSession(Game,1,'club');try{m.advance(false);for(const [kind,tile] of [['discard','9s'],['discard','3m'],['discard','8s'],['discard','7p']]){m.apply({type:'choose',kind,tile});m.advance(false);}return m.snapshot();}finally{m.dispose();}}
function won(){const m=new MatchSession(Game,16,'club');try{m.advance(false);for(let n=0;n<240&&m.view.phase!=='over';n++){const c=m.choices;const q=c.find(x=>['ron','tsumo'].includes(x.kind))??c.find(x=>x.kind==='riichi')??c.find(x=>x.kind==='pass')??c.find(x=>x.kind==='discard'&&x.tile===m.view.seats[0].drawn)??c.find(x=>x.kind==='discard')??c[0];assert.ok(q);m.apply({type:'choose',kind:q.kind,tile:q.tile??null});m.advance(false);}assert.equal(m.view.phase,'over');return m.snapshot();}finally{m.dispose();}}
// The player's turn late in a hand, an opponent in riichi: its declared tile lies
// sideways in its row, a third wider than the others, and the centre says the bet.
function riichiTable(){
  for(let seed=1;seed<=200;seed++){
    const m=new MatchSession(Game,seed,'club');
    try{
      m.advance(false);
      for(let n=0;n<90&&m.view.phase!=='over';n++){
        const mine=m.view.seats[0];
        if(m.view.phase==='act'&&mine.turn&&m.view.seats.slice(1).some(seat=>seat.discards.length>=7&&seat.discards.some(discard=>discard.riichi)))return m.snapshot();
        const c=m.choices;
        const q=c.find(x=>['ron','tsumo'].includes(x.kind))??c.find(x=>x.kind==='pass')??c.find(x=>x.kind==='discard'&&x.tile===mine.drawn)??c.find(x=>x.kind==='discard')??c[0];
        m.apply({type:'choose',kind:q.kind,tile:q.tile??null});m.advance(false);
      }
    }finally{m.dispose();}
  }
  throw new Error('No table with an opponent in riichi was found');
}
// A match whose first move is a Trained opponent's.
function trainedTable(){const m=new MatchSession(Game,1,'neural');try{m.advance(false);return m.snapshot();}finally{m.dispose();}}
const start=opening(),call=calling(),result=won(),riichi=riichiTable(),trained=trainedTable();
// The opponents' worker answers every question with a failure, so the page shows its recovery panel.
const failingWorker=theme=>async p=>{
  await p.emulateMediaFeatures([{name:'prefers-color-scheme',value:theme},{name:'prefers-reduced-motion',value:'reduce'}]);
  await p.setRequestInterception(true);
  p.on('request',request=>{
    if(request.url().includes('/assets/policy.worker-'))void request.respond({status:200,contentType:'text/javascript',body:'self.onmessage=({data:d})=>self.postMessage({id:d.id,error:"Simulated network failure"});'});
    else void request.continue();
  });
};
// Trained AI on offer, its worker a stand-in sure of the last legal move, so
// that the player's own moves differ and are flagged. Nothing is downloaded.
const STAND_IN=`self.onmessage=({data:d})=>{if(d.cancel!=null)return;self.postMessage({id:d.id,progress:'network ready',ready:true});
const one=m=>{let a=-1;for(let i=0;i<m.length;i++)if(m[i])a=i;return{action:a,weights:Array.from(m,(v,i)=>Number(i===a))};};
if(d.count!=null){const w=d.mask.length/d.count,o=[];for(let k=0;k<d.count;k++)o.push(one(d.mask.slice(k*w,(k+1)*w)));self.postMessage({id:d.id,analyses:o});}
else{const r=one(d.mask);self.postMessage({id:d.id,action:r.action,analysis:{...r,value:.12,hands:null}});}};`;
const networkUrl=`${MANIFEST.origin}/${MANIFEST.object}`;
async function trainedReviews(p){
  await p.setRequestInterception(true);
  p.on('request',request=>{
    const url=request.url();
    const answer=url.includes('/assets/policy.worker-')?request.respond({status:200,contentType:'text/javascript',body:STAND_IN})
      :url===networkUrl&&request.method()==='HEAD'?request.respond({status:200,headers:{'Access-Control-Allow-Origin':'*'},body:''})
      :url===networkUrl?request.abort():request.continue();
    void answer.catch(()=>{});
  });
}
function contrast(a,b){
  const luminance=s=>s.match(/[\d.]+/g).slice(0,3).map(Number).map(x=>x/255).map(x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4).reduce((sum,x,i)=>sum+x*[.2126,.7152,.0722][i],0);
  const x=luminance(a),y=luminance(b);return(Math.max(x,y)+.05)/(Math.min(x,y)+.05);
}
// A save the page wrote in format 6 at the result of North's mangan by self-draw, paid 4,000 by the dealer and 2,000 by each other player.
const selfDraw=JSON.parse(readFileSync(new URL('../tests/fixtures/format6-saves.json',import.meta.url))).selfDraw;
async function open(snapshot=start.snapshot,width=390,height=844,prepare=null){const p=await (await openContext(browser)).newPage();p.errors=[];p.on('pageerror',e=>p.errors.push(e.message));await p.setViewport({width,height,isMobile:width<600,hasTouch:width<600});await p.emulateMediaFeatures([{name:'prefers-reduced-motion',value:'reduce'}]);await p.evaluateOnNewDocument((key,settings,s)=>{Object.defineProperty(navigator,'serviceWorker',{value:undefined});localStorage.setItem(key,JSON.stringify(s));localStorage.setItem(settings,JSON.stringify({version:1,difficulty:'club',opponents:['club','club','club'],hints:true,confirmDiscards:true,shortcuts:true,tileFace:'classic',reviewAdviser:'club'}));},SAVE_KEY,SETTINGS_KEY,snapshot);if(prepare)await prepare(p);await p.goto(`http://127.0.0.1:${server.address().port}/mahjong/`,{waitUntil:'networkidle0'});await p.waitForSelector('.hand');return p;}
const shot=(p,name)=>p.screenshot({path:resolve(output,name+'.png'),fullPage:true});
try{await mkdir(output,{recursive:true});await new Promise(done=>server.listen(0,'127.0.0.1',done));browser=await launchChrome();
 await check('phone header keeps one round display and accessible game controls',async()=>{
  const p=await open();
  assert.equal(await p.$eval('.bar h1',el=>el.textContent),'Riichi');
  assert.match(await p.$eval('.round-details',el=>el.textContent),/East 1[\s\S]*tiles left/);
  assert.equal(await p.$('.notice'),null,'restoring a match should not cover the table with a toast');
  assert.equal(await p.$eval('.opponents select',el=>el.disabled),false);
  // The native closed dialog hides its children without changing their own display property.
  assert.equal(await p.$eval('.settings-dialog',el=>el.open),false);
  assert.equal(await p.$eval('.preferences',el=>el.getClientRects().length),0);
  assert.notEqual(await p.$eval('.settings-trigger',el=>getComputedStyle(el).display),'none');
  assert.equal(await p.$eval('.restart',el=>getComputedStyle(el).display),'none');
  await shot(p,'polish-phone-header');
  await p.click('.inspect');
  await p.waitForSelector('.table-dialog[open]');
  await p.keyboard.press('Escape');
  await p.click('.settings-trigger');
  await p.waitForSelector('.settings-dialog[open]');
  assert.equal(await p.$eval('.settings-dialog',el=>el.matches(':modal')),true);
  assert.ok(await p.$eval('.preferences',el=>el.getClientRects().length>0));
  assert.ok(await p.$('.mobile-new-game'));
  await shot(p,'polish-phone-settings');
  assert.deepEqual(p.errors,[]);
 });
 await check('hint counts sit below every hand tile and exactly match public unseen copies',async()=>{const p=await open(),expected=unseenTileCounts(start.view);const rows=await p.$$eval('.hand .hand-tile',els=>els.map(el=>{const button=el.querySelector('button.tile'),tile=button.dataset.tile,count=el.querySelector('.copy-count'),a=button.getBoundingClientRect(),b=count.getBoundingClientRect();return{tile,count:Number(count.textContent.trim()),below:b.top>=a.bottom-0.5};}));assert.equal(rows.length,14);for(const row of rows){assert.equal(row.count,expected.get(row.tile));assert.equal(row.below,true,row.tile);}await p.click('.settings-trigger');await openPanel(p,'.options summary');await p.click('.option-fields input[type=checkbox]');await p.waitForFunction(()=>!document.querySelector('.copy-count'));assert.equal(await p.$('.copy-count'),null);await shot(p,'polish-phone-game');assert.deepEqual(p.errors,[]);});
 await check('desktop table uses the expanded table surface and larger opponent zones',async()=>{const p=await open(start.snapshot,1440,1000);const box=await p.$eval('.board',el=>{const r=el.getBoundingClientRect(),s=getComputedStyle(el);return{w:r.width,h:r.height,bg:s.backgroundImage,radius:s.borderRadius};});assert.ok(box.w>1000);assert.ok(box.h>=380);assert.notEqual(box.bg,'none');assert.ok(parseFloat(box.radius)>=20);await shot(p,'polish-desktop-table');assert.deepEqual(p.errors,[]);});
 await check('call window presents the discard as a staged event before choices',async()=>{const p=await open(call);await p.waitForSelector('.call-stage');assert.match(await p.$eval('.call-stage',el=>el.textContent),/discarded/i);const tile=await p.$eval('.call-stage .tile',el=>el.getBoundingClientRect().width);assert.ok(tile>=40);assert.ok(await p.$('.call-options'));await shot(p,'polish-phone-call');assert.deepEqual(p.errors,[]);});
 await check('phone result is a fixed bottom sheet with hero value and can reveal the table',async()=>{const p=await open(result);await p.waitForSelector('.screen');const state=await p.$eval('.screen',el=>({position:getComputedStyle(el).position,maxHeight:getComputedStyle(el).maxHeight,bottom:el.getBoundingClientRect().bottom,hero:el.querySelector('.hero-score')?.textContent,buttons:getComputedStyle(el.querySelector('.buttons')).position}));assert.equal(state.position,'fixed');assert.ok(state.bottom<=845&&state.bottom>=842);assert.match(state.hero,/Ron|Tsumo/);assert.equal(state.buttons,'sticky');const review=await p.$('.screen .quiet');if(review){await review.click();await p.waitForSelector('.result-chip');assert.equal(await p.$eval('.screen',el=>getComputedStyle(el).display),'none');await p.click('.result-chip');assert.notEqual(await p.$eval('.screen',el=>getComputedStyle(el).display),'none');}await shot(p,'polish-phone-result');assert.deepEqual(p.errors,[]);});
 await check('a self-drawn win is headlined by what the hand is worth, not by its first payment',async()=>{
  const p=await open(selfDraw);await p.waitForSelector('.screen .hero-score');
  const hero=await p.$eval('.screen .hero-score',el=>({value:el.querySelector('b').textContent.trim(),label:el.getAttribute('aria-label')}));
  // The headline once read 4,000, the dealer's share alone.
  assert.deepEqual(hero,{value:'2,000 / 4,000',label:'North Tsumo, mangan, 2,000 / 4,000'});
  await shot(p,'polish-phone-tsumo');assert.deepEqual(p.errors,[]);
 });
 await check('a wide window too short for the table around its centre has the dense table, and the whole hand on its first screen',async()=>{
  // A 1366 by 768 laptop's browser is some 650 pixels high and a 1280 by 720
  // one's 600. With the opponents around the centre the hand lay under the
  // screen there as the turn began.
  for(const [width,height] of [[1366,650],[1280,600],[1536,730]]){
    for(const [name,snapshot] of [['an early turn',start.snapshot],['a late turn against a riichi',riichi]]){
      const p=await open(snapshot,width,height);
      const seen=await p.evaluate(()=>{
        const box=selector=>document.querySelector(selector).getBoundingClientRect();
        const discard=document.querySelector('.place .pool .tile:not(.rotated)');
        return{scrolled:scrollY,handTop:box('.hand').top,handBottom:box('.hand').bottom,screen:innerHeight,centre:box('.centre').width,board:box('.board').width,
          discard:discard?discard.getBoundingClientRect().width:null,overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth};
      });
      const where=`${name} at ${width}x${height}`;
      assert.equal(seen.scrolled,0,where);
      assert.ok(seen.handTop>=0&&seen.handBottom<=seen.screen,`the whole hand is on the screen for ${where}: ${seen.handTop} to ${seen.handBottom} of ${seen.screen}`);
      assert.ok(seen.centre>seen.board*.9,`the centre is a bar across the table for ${where}`);
      if(seen.discard!==null)assert.ok(seen.discard>=21.5,`the discards are a readable size for ${where}: ${seen.discard}`);
      assert.equal(seen.overflow,0,where);
      assert.deepEqual(p.errors,[]);
    }
  }
  // A taller window keeps the opponents around the centre, and the hand in view.
  const p=await open(riichi,1920,940);
  const tall=await p.evaluate(()=>({centre:document.querySelector('.centre').getBoundingClientRect().width,board:document.querySelector('.board').getBoundingClientRect().width,
    bottom:document.querySelector('.hand').getBoundingClientRect().bottom,screen:innerHeight}));
  assert.ok(tall.centre<tall.board/2,'from 900 pixels high the table has its room around the centre');
  assert.ok(tall.bottom<=tall.screen,`and the hand is on the screen: ${tall.bottom} of ${tall.screen}`);
 });
 await check('from 761 pixels wide the hand is one row, the tile just drawn in it',async()=>{
  // Fourteen tiles of 52 pixels need a window 889 wide. Under that the drawn
  // tile wrapped onto a row of its own, on an upright tablet as in a narrow
  // window, and the seat changed height at every turn. The seat is measured
  // as a container: 70 pixels of the window lie outside it, and its tiles
  // are 52 pixels wide from 822 inside, 48 from 766, 44 from 710, else 40.
  for(const [width,tile] of [[761,40],[768,40],[779,40],[780,44],[820,44],[835,44],[836,48],[891,48],[892,52],[1100,52]]){
    const p=await open(start.snapshot,width,1000);
    const hand=await p.$$eval('.hand [data-hand-index]',tiles=>tiles.map(tile=>{const box=tile.getBoundingClientRect();return{top:Math.round(box.top),width:box.width,right:box.right};}));
    const edge=await p.$eval('.hand',el=>el.getBoundingClientRect().right);
    assert.equal(hand.length,14,'the saved turn has a drawn tile');
    assert.equal(new Set(hand.map(tile=>tile.top)).size,1,`one row at ${width}: ${hand.map(tile=>tile.top).join(' ')}`);
    assert.ok(hand.every(shown=>shown.width===tile),`${tile} pixels a tile at ${width}: ${hand[0].width}`);
    assert.ok(hand.at(-1).right<=edge,`the drawn tile is inside the hand at ${width}`);
    assert.deepEqual(p.errors,[]);
  }
  // The same with a scrollbar's 17 pixels gone from the window, as a
  // desktop's has: each step then begins 17 pixels later, and these are each
  // step's first width and the pixel before it.
  for(const [width,tile] of [[761,40],[796,40],[797,44],[852,44],[853,48],[908,48],[909,52]]){
    const p=await open(start.snapshot,width,1000,async page=>{await page.evaluateOnNewDocument(()=>document.addEventListener('DOMContentLoaded',()=>{document.documentElement.style.paddingRight='17px';}));});
    const hand=await p.$$eval('.hand [data-hand-index]',tiles=>tiles.map(shown=>({top:Math.round(shown.getBoundingClientRect().top),width:shown.getBoundingClientRect().width})));
    assert.equal(new Set(hand.map(shown=>shown.top)).size,1,`one row at ${width} with a scrollbar: ${hand.map(shown=>shown.top).join(' ')}`);
    assert.ok(hand.every(shown=>shown.width===tile),`${tile} pixels a tile at ${width} with a scrollbar: ${hand[0].width}`);
  }
 });
 await check('a phone draws the opponents\u2019 discards as large as six to a row fit, and the modes in one row',async()=>{
  // They were 14 pixels wide on every phone, and 11.5 under 360. The first
  // two screens are narrower than any step but the least, and the last is a
  // phone held sideways, whose seats are 88 pixels wide: a full row with a
  // riichi's tile was 8.5 pixels wider than its seat there.
  for(const [width,height,least] of [[280,653,10.5],[300,653,11.5],[320,568,12.5],[360,740,15],[375,667,15.5],[390,844,16.5],[412,915,17.5],[430,932,18.5],[667,375,11.5]]){
    const p=await open(riichi,width,height);
    const seats=await p.$$eval('.place',places=>places.map(place=>{
      const seat=place.querySelector('.seat'),style=getComputedStyle(seat),inner=seat.getBoundingClientRect().right-parseFloat(style.paddingRight)-parseFloat(style.borderRightWidth);
      const tiles=[...place.querySelectorAll('.pool .tile')].map(tile=>({width:tile.getBoundingClientRect().width,right:tile.getBoundingClientRect().right,turned:tile.classList.contains('rotated')}));
      return{inner,tiles,row:Math.max(0,...tiles.map(tile=>tile.right))};
    }));
    assert.equal(seats.length,3);
    assert.ok(seats.some(seat=>seat.tiles.some(tile=>tile.turned)),'a riichi\u2019s tile lies sideways in one row');
    for(const seat of seats){
      const upright=seat.tiles.filter(tile=>!tile.turned);
      assert.ok(upright.length>=6,'each opponent has a full row');
      assert.ok(upright.every(tile=>tile.width>=least),`at ${width} a discard is at least ${least} pixels wide: ${upright[0].width}`);
      assert.ok(seat.row<=seat.inner+.5,`at ${width} no row is wider than its seat: ${seat.row} of ${seat.inner}`);
    }
    assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth-document.documentElement.clientWidth),0);
    // A ring around a discard is no wider than the room between two of them.
    const rows=await p.$$eval('.place .pool',pools=>pools.map(pool=>({gap:parseFloat(getComputedStyle(pool).columnGap),
      rings:[...pool.querySelectorAll('.tile')].map(tile=>parseFloat(getComputedStyle(tile).getPropertyValue('--ring-width')))})));
    for(const row of rows)assert.ok(row.gap>=1&&row.rings.length>=6&&row.rings.every(ring=>ring<=row.gap),`at ${width} a ring fits between two discards: ${row.rings[0]} in ${row.gap}`);
    const modes=await p.$eval('.game-modes',el=>({shown:getComputedStyle(el).display!=='none',height:el.getBoundingClientRect().height,
      names:[...el.querySelectorAll('button')].map(button=>[button.getAttribute('aria-label'),button.textContent,button.innerText.trim()])}));
    if(modes.shown&&width<620){
      // One word each, and their whole names still said and still their text.
      assert.ok(modes.height<=52,`the modes take one row at ${width}: ${modes.height}`);
      assert.deepEqual(modes.names,[['Play','Play','Play'],['Agent watch','Agent watch','Watch'],['Physical agent play','Physical agent play','Physical'],['Guided physical game','Guided physical game','Guided']]);
    }
    assert.deepEqual(p.errors,[]);
  }
  await shot(await open(riichi,390,844),'polish-phone-riichi-table');
 });
 await check('what went wrong with a Trained opponent, and the ways on, can be read in both themes',async()=>{
  // In the light theme the panel's buttons were its red on a grey wash of it,
  // 2.3 to 1, and 1.3 under the pointer.
  for(const theme of ['light','dark']){
    const p=await open(trained,1100,850,failingWorker(theme));
    await p.waitForSelector('.failure .recovery-actions button');
    const read=()=>p.$eval('.failure',panel=>{
      const pair=el=>[getComputedStyle(el).color,getComputedStyle(el).opacity];
      return{panel:getComputedStyle(panel).backgroundColor,words:[...panel.querySelectorAll('p')].map(pair),
        buttons:[...panel.querySelectorAll('button')].map(button=>[getComputedStyle(button).color,getComputedStyle(button).backgroundColor,button.getBoundingClientRect().height])};
    });
    const seen=await read();
    assert.ok(seen.words.length>=2&&seen.buttons.length>=3,theme);
    for(const [colour,opacity] of seen.words){
      assert.equal(opacity,'1',theme);
      assert.ok(contrast(colour,seen.panel)>=4.5,`the panel's words in the ${theme} theme: ${colour} on ${seen.panel}, ${contrast(colour,seen.panel).toFixed(2)}`);
    }
    for(const [colour,ground,height] of seen.buttons){
      assert.ok(contrast(colour,ground)>=4.5,`a button in the ${theme} theme: ${colour} on ${ground}, ${contrast(colour,ground).toFixed(2)}`);
      assert.ok(height>=44);
    }
    await p.hover('.failure .recovery-actions button');
    // Read once the pointer's colours are drawn, not the resting ones.
    await p.evaluate(()=>new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done))));
    assert.equal(await p.$eval('.failure .recovery-actions button',button=>button.matches(':hover')),true);
    const [colour,ground]=(await read()).buttons[0];
    assert.ok(contrast(colour,ground)>=4.5,`a button under the pointer in the ${theme} theme: ${colour} on ${ground}, ${contrast(colour,ground).toFixed(2)}`);
    if(theme==='light')await shot(p,'polish-recovery-light');
  }
 });
 await check('the reviews\u2019 words in gold can be read in both themes',async()=>{
  // In the borders' gold, Flagged on a line under the gold wash of the
  // decision on show was 3.6 to 1 in the light theme, and the player's own
  // weight, at 0.75, 4.4: under the 4.5 that words this small need.
  const p=await open(result,1200,900,trainedReviews);
  const theme=value=>p.emulateMediaFeatures([{name:'prefers-color-scheme',value},{name:'prefers-reduced-motion',value:'reduce'}]);
  const readable=async(selector,where)=>{
   for(const value of ['light','dark']){
    await theme(value);
    const seen=await wordsContrast(p,selector);
    assert.ok(seen.ratio>=4.5,`${where} in the ${value} theme: "${seen.text}" ${seen.colour} on ${seen.ground}, ${seen.ratio.toFixed(2)}`);
   }
   await theme('light');
  };
  await p.waitForSelector('.screen [data-review-game]');
  await p.click('.screen .buttons .quiet');
  await p.waitForSelector('.review select[aria-label="Review adviser"]');
  await p.select('.review select[aria-label="Review adviser"]','strong');
  await p.waitForSelector('.review .policy-preference .played strong');
  await readable('.review .policy-preference strong','Trained AI\u2019s preference and the player\u2019s');
  await p.click('.screen [data-review-game]');
  await p.waitForSelector('[data-game-review] [data-decision]');
  await p.click('.rail [data-filter-option="every"]');
  await p.waitForSelector('.rail .flag-mark');
  await readable('.rail .flag-mark','a flagged decision\u2019s line');
  assert.deepEqual(p.errors,[]);
 });
 await check('a window between two whole pixels is laid out as the whole pixel under it',async()=>{
  // At a browser zoom of 125 per cent a window 951 device pixels wide is 760.8
  // CSS pixels, and matched neither (max-width: 760px) nor (min-width: 761px):
  // the table, the seats and the hand fell back to styles meant for no size.
  // A frame zoomed by 1.25 is as wide as such a window. The built page's media
  // rules that match in it are those that match at the whole pixel under its
  // size, across and down.
  const p=await (await openContext(browser)).newPage();p.errors=[];p.on('pageerror',e=>p.errors.push(e.message));
  await p.setViewport({width:1400,height:1400});
  await p.evaluateOnNewDocument((key,settings,s)=>{
   if(location.protocol!=='http:')return;
   Object.defineProperty(navigator,'serviceWorker',{value:undefined});
   localStorage.setItem(key,JSON.stringify(s));
   localStorage.setItem(settings,JSON.stringify({version:1,difficulty:'club',opponents:['club','club','club'],hints:true,confirmDiscards:true,shortcuts:true,tileFace:'classic',reviewAdviser:'club'}));
  },SAVE_KEY,SETTINGS_KEY,start.snapshot);
  // The frame's host is a page of the site's own origin, so that the frame
  // keeps the storage it would have as a window of its own.
  const site=`http://127.0.0.1:${server.address().port}/mahjong/`;
  await p.goto(`${site}frame-host`);
  const matched=async(width,height)=>{
   await p.evaluate(html=>{document.body.innerHTML=html;},`<iframe style="border:0;zoom:1.25;width:${width}px;height:${height}px" src="${site}"></iframe>`);
   const frame=await (await p.waitForSelector('iframe')).contentFrame();
   await frame.waitForSelector('.hand');
   return frame.evaluate(()=>{
    const queries=new Set();
    const walk=rules=>{for(const rule of rules){if(rule.media)queries.add(rule.media.mediaText);if(rule.cssRules)walk(rule.cssRules);}};
    for(const sheet of document.styleSheets)walk(sheet.cssRules);
    return [...queries].filter(query=>matchMedia(query).matches).sort();
   });
  };
  for(const [whole,between] of [[[760,900],[760.8,900]],[[1000,500],[1000,500.8]]]){
   const there=await matched(...whole),near=await matched(...between);
   assert.ok(there.length>0);
   assert.deepEqual(near,there,`${between.join(' by ')} is laid out as ${whole.join(' by ')}`);
  }
  assert.deepEqual(p.errors,[]);
 });
 await check('a phone\u2019s settings keep an opened section where it was pressed, and its play area scrolls with the page',async()=>{
  const p=await open();
  await p.click('.settings-trigger');
  await p.waitForSelector('.settings-dialog[open]');
  const tops=()=>p.$$eval('.settings-dialog details',sections=>sections.map(section=>Math.round(section.getBoundingClientRect().top)));
  const before=await tops();
  assert.ok(before.length>=4);
  await openPanel(p,'.settings-dialog details:first-of-type > summary');
  const after=await tops();
  // The sections stand one under the other in the order they have in the
  // page, the opened one still first: it used to go to the foot of the list.
  assert.equal(await p.$eval('.settings-dialog details:first-of-type',section=>section.open),true);
  assert.deepEqual(after,after.toSorted((a,b)=>a-b),`${before} then ${after}`);
  assert.equal(new Set(after).size,after.length);
  await p.$eval('[data-close-settings]',button=>button.click());
  // The play area stayed six pixels from the top of the screen as the page
  // moved, under the game's bar, which stays there too.
  assert.equal(await p.$eval('.play-area',el=>getComputedStyle(el).position),'static');
  assert.deepEqual(p.errors,[]);
  // The result's chip lies over the foot of the screen: the page's last line can be scrolled clear of it.
  const over=await open(result);
  await over.waitForSelector('.screen');
  const review=await over.$('.screen .quiet');
  assert.ok(review);
  await review.click();
  await over.waitForSelector('.result-chip');
  // The page's foot makes room as the chip comes, and under reduced motion,
  // which these checks ask for, every change of style eases over a hundredth
  // of a millisecond (app.css): it is in effect at the next frame, not at
  // once. Measured before that frame, as a slow machine does, the foot was
  // still its 24 pixels and the history lay 30 under the chip (in CI, in
  // three runs). So the room is waited for, and its lack said as such.
  await over.waitForFunction(()=>parseFloat(getComputedStyle(document.querySelector('main')).paddingBottom)>=56,{timeout:10000})
    .catch(()=>assert.fail('the page\u2019s foot never made room for the chip'));
  // Scrolled to its end and measured in one step: the page grows a moment
  // after the chip shows, as the hand's review arrives, and measured a step
  // after the scroll it was no longer at its end (once, in CI).
  const foot=await over.evaluate(()=>{
    const root=document.documentElement;
    window.scrollTo(0,root.scrollHeight);
    return{chip:document.querySelector('.result-chip').getBoundingClientRect().top,chipHeight:document.querySelector('.result-chip').getBoundingClientRect().height,
      history:document.querySelector('.history summary').getBoundingClientRect().bottom,fromEnd:root.scrollHeight-root.scrollTop-innerHeight,
      room:getComputedStyle(document.querySelector('main')).paddingBottom,told:document.querySelector('main').classList.contains('chip'),inside:Boolean(document.querySelector('main .result-chip'))};
  });
  assert.ok(Math.abs(foot.fromEnd)<1,`the page is at its end: ${foot.fromEnd}`);
  assert.ok(foot.history<=foot.chip,`the hand's history ends above the chip: ${foot.history} against ${foot.chip}; the page's foot is ${foot.room}, told of the chip ${foot.told}, the chip inside it ${foot.inside}`);
  assert.ok(foot.chipHeight>=44);
  assert.deepEqual(over.errors,[]);
 });
}finally{await browser?.close();if(server.listening)await new Promise(done=>server.close(done));await report('UI-polish browser checks',resolve(output,'ui-polish-report.json'));}
