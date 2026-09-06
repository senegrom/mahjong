import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { moveHandFocus } from '../src/lib/ui.js';
await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const make = (seed, difficulty = 'club') => { const m = new MatchSession(Game, seed, difficulty); m.advance(false); return m; };
const step = (m, kind, tile) => { m.apply({ type: 'choose', kind, tile: tile ?? null }); m.advance(false); };
function playHand(m, riichi = true) {
  for (let n = 0; n < 200 && m.view.phase !== 'over'; n++) {
    const c = m.choices;
    const q = c.find(c => ['ron', 'tsumo'].includes(c.kind))
      ?? (riichi ? c.find(c => c.kind === 'riichi') : null)
      ?? c.find(c => c.kind === 'pass')
      ?? (riichi ? c.find(c => c.kind === 'discard' && c.tile === m.view.seats[0].drawn) : null)
      ?? c.find(c => c.kind === 'discard') ?? c[0];
    assert.ok(q); step(m, q.kind, q.tile);
  }
  assert.equal(m.view.phase, 'over');
}
const mjaiTile = tile => tile?.endsWith('z') ? ['E','S','W','N','P','F','C'][Number(tile[0])-1] : tile;
const doraOf = tile => {
  const r = Number(tile[0]);
  return tile[1] === 'z' ? `${r <= 4 ? r % 4 + 1 : (r - 4) % 3 + 5}z` : `${r % 9 + 1}${tile[1]}`;
};

test('riichi waits stay on green dragon while unrelated North is drawn (seed 81)', () => {
  const m = make(81);
  try {
    step(m,'discard','9s'); step(m,'discard','5z'); step(m,'riichi','2z');
    assert.ok(m.view.seats[0].riichi); assert.equal(m.view.seats[0].drawn,'4z');
    assert.deepEqual(m.view.waits,['6z']);
    assert.deepEqual(m.engine.discard_hint('4z').waits,['6z']);
    assert.throws(() => m.engine.discard_hint('1p'));
  } finally { m.dispose(); }
});

test('discard previews are non-mutating and match the actual thirteen-tile hand', () => {
  const m = make(11);
  try {
    const before=m.stateKey();
    for(const c of m.choices.filter(c => c.kind === 'discard')) {
      const hint=m.engine.discard_hint(c.tile);
      assert.equal(m.stateKey(),before);
      const copy=MatchSession.restore(Game,JSON.stringify(m.snapshot()));
      try {
        copy.engine.choose('discard',c.tile);
        assert.deepEqual(hint.waits,copy.view.waits);
        assert.deepEqual(hint.waits_left,copy.view.waits_left);
        assert.equal(hint.shanten,copy.view.shanten);
      } finally {copy.dispose();}
    }
  } finally {m.dispose();}
});

test('old format-2 hints migrate without relaxing tile, score or action validation', () => {
  const original=JSON.parse(readFileSync(new URL('./fixtures/pre-hint-format2.json',import.meta.url)));
  assert.deepEqual(JSON.parse(original.state)[0].waits,['4z','6z']);
  const m=MatchSession.restore(Game,JSON.stringify(original));
  try {
    assert.deepEqual(m.view.waits,['6z']);assert.equal(m.snapshot().format,4);
    const restored=MatchSession.restore(Game,JSON.stringify(m.snapshot()));
    assert.equal(restored.stateKey(),m.stateKey());restored.dispose();
  } finally {m.dispose();}
  for(const corrupt of [state=>state[0].seats[0].score++,state=>state[0].seats[0].hand[0]='9m',state=>state[1].pop()]) {
    const state=JSON.parse(original.state);corrupt(state);
    assert.throws(()=>MatchSession.restore(Game,JSON.stringify({...original,state:JSON.stringify(state)})));
  }
});

for(const seed of [3,17,248])test(`exported settlement deltas balance event by event (seed ${seed})`,()=>{
  const m=make(seed);
  try {
    playHand(m);let balance;let wins=0;
    const before=m.view.seats.map(s=>s.score);
    for(const e of m.engine.log().trim().split('\n').map(JSON.parse)) {
      if(e.type==='start_kyoku')balance=[...e.scores];
      if(e.type==='reach_accepted')balance[e.actor]-=1000;
      if(['hora','ryukyoku'].includes(e.type)) {
        balance=balance.map((n,i)=>n+e.deltas[i]);
        assert.deepEqual(balance,e.scores,`${e.type} must report only this event's change`);
        if(e.type==='hora')wins++;
      }
    }
    if(seed===248)assert.equal(wins,2);
    const at=m.engine.player_index();
    assert.equal(balance[at],before[0]);
  } finally {m.dispose();}
});

test('ura indicators are absent during play and agree with scoring and exported markers on wins',()=>{
  let revealed=0,hidden=0,multiple=0;
  for(let seed=1;seed<=45;seed++) {
    const m=make(seed);
    try {
      assert.ok(!m.view.outcome);
      playHand(m);
      for(const win of m.view.outcome.wins) {
        const event=m.engine.log().split('\n').map(JSON.parse).find(e=>e.type==='hora'&&e.actor===['east','south','west','north'].indexOf(win.seat));
        assert.ok(event);
        const riichi=m.view.seats.find(s=>s.seat===win.seat).riichi;
        assert.deepEqual(win.dora_indicators,m.view.dora_indicators);
        assert.deepEqual(win.ura_indicators.map(mjaiTile),event.uradora_markers);
        if(riichi) {assert.equal(win.ura_indicators.length,win.dora_indicators.length);revealed++;if(win.ura_indicators.length>1)multiple++;}
        else {assert.deepEqual(win.ura_indicators,[]);hidden++;}
        const tiles=[...win.hand,win.winning_tile,...win.melds.flatMap(m=>m.tiles)];
        const count=indicators=>indicators.reduce((n,indicator)=>n+tiles.filter(t=>t===doraOf(indicator)).length,0);
        const scored=win.limit!=='yakuman';
        assert.equal(win.ura_dora,scored?count(win.ura_indicators):0);
        assert.equal(win.dora,scored?count(win.dora_indicators)+win.ura_dora:0);
      }
    } finally {m.dispose();}
  }
  assert.ok(revealed>0 && hidden>0,'Exercised riichi and non-riichi winners');
  // This deterministic sample includes kan-derived ura indicators.
  assert.ok(multiple>0,'Exercised multiple indicator slots');
});

test('opening final standings retains history, review, log and exact restoration',()=>{
  const m=make(14,'beginner');
  try {
    for(let hand=0;hand<40&&!m.over;hand++) {
      playHand(m,false);
      const before={events:[...m.events],review:m.engine.review(),log:m.engine.log(),scores:m.view.seats.map(s=>s.score)};
      m.apply({type:'next'});m.advance(false);
      if(m.over) {
        assert.ok(before.events.length>0);assert.ok(before.review.length>0);
        assert.deepEqual(m.events,before.events);assert.deepEqual(m.engine.review(),before.review);assert.equal(m.engine.log(),before.log);
        assert.deepEqual(m.view.seats.map(s=>s.score),before.scores);
        const restored=MatchSession.restore(Game,JSON.stringify(m.snapshot()));
        try {assert.deepEqual(restored.events,before.events);assert.deepEqual(restored.engine.review(),before.review);assert.equal(restored.engine.log(),before.log);}
        finally {restored.dispose();}
      }
    }
    assert.ok(m.over);
  } finally {m.dispose();}
});

function handFixture(disabled,focus=disabled.length-1) {
  const document={activeElement:null};
  const buttons=disabled.map((isDisabled,i)=>({disabled:isDisabled,dataset:{handIndex:String(i)},closest(){return this;},focus(){if(!this.disabled)document.activeElement=this;}}));
  const hand={ownerDocument:document,querySelectorAll:()=>buttons,contains:e=>e===hand||buttons.includes(e),closest:()=>null};
  document.activeElement=buttons[focus]??hand;
  return {hand,document,buttons};
}
test('arrow keys skip disabled tiles and keep a lone legal drawn tile selected',()=>{
  const {hand,document,buttons}=handFixture([true,true,true,false]);
  for(const dir of [-1,1,-1]) {assert.equal(moveHandFocus(hand,dir),3);assert.equal(document.activeElement,buttons[3]);}
  buttons[0].disabled=false;assert.equal(moveHandFocus(hand,-1),0);assert.equal(moveHandFocus(hand,1),3);
});
test('failed focus cannot move the marker to another tile or steal external focus',()=>{
  const {hand,document,buttons}=handFixture([false,false]);buttons[0].focus=()=>{};
  assert.equal(moveHandFocus(hand,-1),1);document.activeElement={};assert.equal(moveHandFocus(hand,1),null);
  document.activeElement=hand;buttons.forEach(b=>b.disabled=true);assert.equal(moveHandFocus(hand,-1),null);
});
