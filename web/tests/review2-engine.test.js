import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import init, {Game} from '../src/wasm/riichi.js';
import {MatchSession} from '../src/lib/session.js';
import {meldTiles,placeLabel} from '../src/lib/ui.js';
await init({module_or_path:readFileSync(new URL('../src/wasm/riichi_bg.wasm',import.meta.url))});
function make(seed,mode='club'){const m=new MatchSession(Game,seed,mode);m.advance(false);return m;}
function step(m,c){m.apply({type:'choose',kind:c.kind,tile:c.tile});m.advance(false);}
function finish(seed){const m=make(seed,'beginner');let previous,oldLog;
 for(let n=0;n<3500&&!m.over;n++){
  if(m.engine.hand_is_over()){previous=m.view;oldLog=m.engine.log();m.apply({type:'next'});m.advance(false);}
  else {const c=m.choices;step(m,c.find(c=>c.kind==='ron'||c.kind==='tsumo')??c.find(c=>c.kind==='pass')??c.find(c=>c.kind==='discard')??c[0]);}}
 assert.ok(m.over);return {m,previous,oldLog};}

test('shared neural call asks the other player after human Pass and can be restored mid-window',()=>{
 const m=make(287,'neural');let restored;
 try {
  m.apply({type:'opponent',action:1});m.advance(false);
  assert.equal(m.view.pending_discard,'2m');assert.ok(m.choices.some(c=>c.kind==='pon'));
  step(m,{kind:'pass'});
  assert.equal(m.view.phase,'call');assert.ok(m.engine.needs_opponent_move());assert.deepEqual(m.choices,[]);
  restored=MatchSession.restore(Game,JSON.stringify(m.snapshot()));assert.equal(restored.stateKey(),m.stateKey());
  const mask=m.engine.opponent_mask();assert.ok(mask[73],'The 2-3-4 characters Chii remains offered to South');
  const action=73;
  assert.ok(action>=0);m.apply({type:'opponent',action});m.advance(false);
  const events=m.engine.log().split('\n').map(JSON.parse);assert.ok(events.some(e=>e.type==='chi'&&e.pai==='2m'));
 } finally {m.dispose();restored?.dispose();}
});

test('Club recovery after a submitted human Pass gathers the remaining claim',()=>{
 const m=make(287,'neural');try{
  m.apply({type:'opponent',action:1});m.advance(false);step(m,{kind:'pass'});
  m.apply({type:'club'});m.advance(false);
  assert.ok(m.engine.log().split('\n').map(JSON.parse).some(e=>e.type==='chi'&&e.pai==='2m'));
  const r=MatchSession.restore(Game,JSON.stringify(m.snapshot()));assert.equal(r.stateKey(),m.stateKey());r.dispose();
 }finally{m.dispose();}
});

test('final table, last-hand log and standings preserve player identity after settlement and reload',()=>{
 const {m,previous,oldLog}=finish(14);try{
  assert.deepEqual(m.view.seats,previous.seats);
  assert.equal(m.engine.log(),oldLog,'Retained log must not rotate player ids');
  const rows=m.engine.standings();const yours=rows.find(r=>r.you);assert.equal(yours.score,previous.seats[0].score);assert.equal(yours.seat,previous.seats[0].seat);
  const r=MatchSession.restore(Game,JSON.stringify(m.snapshot()));assert.equal(r.stateKey(),m.stateKey());assert.equal(r.engine.log(),oldLog);r.dispose();
 }finally{m.dispose();}
});

test('equal final scores receive joint places and stable unique keys',()=>{
 const {m}=finish(2);try{
  const rows=m.engine.standings();const tied=rows.filter(r=>r.score===26600);
  assert.equal(tied.length,2);assert.equal(tied[0].place,tied[1].place);assert.ok(tied.every(r=>r.tied));assert.equal(placeLabel(tied[0]),'Joint 3rd');
  assert.equal(new Set(rows.map(r=>r.player)).size,4);
 }finally{m.dispose();}
});

test('Chii remembers the middle tile that was actually claimed',()=>{
 const m=make(1);try{
  for(const tile of ['9s','3m','8s','7p'])step(m,{kind:'discard',tile});
  assert.equal(m.view.pending_discard,'5m');step(m,{kind:'chii',tile:'4m'});
  const meld=m.view.seats[0].melds[0];assert.equal(meld.claimed_tile,'5m');assert.deepEqual(meldTiles(meld),['5m','4m','6m']);
  const r=MatchSession.restore(Game,JSON.stringify(m.snapshot()));assert.deepEqual(r.view.seats[0].melds,m.view.seats[0].melds);r.dispose();
 }finally{m.dispose();}
});

test('old compatible saves gain claimed-tile metadata without relaxing other state validation',()=>{
 const m=make(1);try{
  for(const tile of ['9s','3m','8s','7p'])step(m,{kind:'discard',tile});step(m,{kind:'chii',tile:'4m'});
  const old=m.snapshot();delete old.format;old.state=JSON.stringify(JSON.parse(old.state),(k,v)=>k==='claimed_tile'?undefined:v);
  const r=MatchSession.restore(Game,JSON.stringify(old));assert.equal(r.view.seats[0].melds[0].claimed_tile,'5m');r.dispose();
  const bad=JSON.parse(old.state);bad[0].seats[0].score++;assert.throws(()=>MatchSession.restore(Game,JSON.stringify({...old,state:JSON.stringify(bad)})));
 }finally{m.dispose();}
});

test('sequence layout handles claims on every position and leaves concealed quads unrotated',()=>{
 for(const claimed of ['4m','5m','6m'])assert.equal(meldTiles({kind:'chii',from:'left',claimed_tile:claimed,tiles:['4m','5m','6m']})[0],claimed);
 assert.deepEqual(meldTiles({kind:'concealed-kan',from:'self',claimed_tile:null,tiles:['1p','1p','1p','1p']}),['1p','1p','1p','1p']);
});

test('a legacy neural call save migrates its historical implicit Pass without replaying AI',()=>{
 const text=readFileSync(new URL('./fixtures/legacy-neural-call.json',import.meta.url),'utf8');
 const old=JSON.parse(text);assert.equal(old.format,undefined);assert.equal(old.commands.length,2);
 const m=MatchSession.restore(Game,text,{ai(){throw new Error('Do not rerun past neural choices');}});
 try {
  assert.equal(m.commands.length,3);assert.deepEqual(m.commands.at(-1),{type:'opponent',action:70});
  const r=MatchSession.restore(Game,JSON.stringify(m.snapshot()));assert.equal(r.stateKey(),m.stateKey());r.dispose();
 }finally{m.dispose();}
});
