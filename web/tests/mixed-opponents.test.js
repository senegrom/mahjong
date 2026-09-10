import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, readSettings } from '../src/lib/session.js';
import { normalizeOpponents, opponentPreset, OPPONENT_TYPES } from '../src/lib/opponents.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const make = (seed, config) => new MatchSession(Game, seed, config);
const pick = choices => choices.find(c => c.kind === 'ron' || c.kind === 'tsumo')
  ?? choices.find(c => c.kind === 'riichi') ?? choices.find(c => c.kind === 'pass')
  ?? choices.find(c => c.kind === 'discard') ?? choices[0];
// Mortal's forty-six moves, which is what a trained opponent now answers in:
// 43 is the win, whether by draw or on a discard, and 45 the pass. Anything
// else falls back to the lowest legal move, which is a discard.
const neuralChoice = mask => mask[43] ? 43 : mask[45] ? 45 : mask.findIndex(Boolean);
// The same preference in our own seventy-eight, for the engine itself.
const enginePick = mask => mask[69] ? 69 : mask[68] ? 68 : mask[70] ? 70 : mask.findIndex(Boolean);
function turn(m) {
  if (m.engine.needs_opponent_move()) {
    const player = m.engine.opponent_player();
    const seat = m.view.seats.find(s => s.player === player);
    assert.equal(seat.controller, 'neural', 'Only a trained player may ask the network');
    m.apply({ type: 'opponent', action: neuralChoice(m.engine.opponent_mask_mortal()) });
    return player;
  }
  if (m.view.phase === 'over') m.apply({ type: 'next' });
  else { const c = pick(m.choices); assert.ok(c, 'Every settled position owes a decision'); m.apply({ type: 'choose', kind: c.kind, tile: c.tile }); }
  return null;
}
function restored(m) {
  const r = MatchSession.restore(Game, JSON.stringify(m.snapshot()));
  try { assert.equal(r.stateKey(), m.stateKey()); assert.deepEqual(r.opponents, m.opponents); }
  finally { r.dispose(); }
}

test('opponent configuration rejects partial, unknown and aliased inputs', () => {
  for (const bad of [null, 'custom', [], ['club'], ['club','club','wrong'], ['club','neural','club','club'], Array(3)]) {
    assert.throws(() => normalizeOpponents(bad), /Invalid opponents/);
  }
  const config = ['beginner','club','neural']; const copy = normalizeOpponents(config); copy[0] = 'club';
  assert.equal(config[0], 'beginner'); assert.equal(opponentPreset(config), 'custom');
  assert.deepEqual(normalizeOpponents('neural'), ['neural','neural','neural']);
  assert.throws(() => Game.with_opponents(1, 'club', 'other', 'neural'));
});

test('legacy and custom settings are read without retaining invalid assignments', () => {
  const read = value => readSettings({ getItem: () => JSON.stringify({ version:1, ...value }) });
  assert.equal(read({ difficulty:'neural' }).difficulty, 'neural');
  assert.deepEqual(read({ difficulty:'custom', opponents:['neural','club','beginner'] }).opponents, ['neural','club','beginner']);
  assert.equal(read({ difficulty:'custom', opponents:['neural','bad','club'] }).difficulty, 'club');
});

test('all 27 tables route only Trained decisions and round-trip pending calls and hands', () => {
  let external = 0, calls = 0, rotations = 0;
  for (const right of OPPONENT_TYPES) for (const across of OPPONENT_TYPES) for (const left of OPPONENT_TYPES) {
    const config = [right, across, left]; const m = make(287, config);
    try {
      m.advance(false); const ids = m.view.seats.map(s => s.player); const initialWind = m.view.seats[0].seat;
      let steps = 0;
      while (m.view.hands_played < 2 && !m.over && steps++ < 600) {
        assert.deepEqual(m.view.seats.slice(1).map(s => s.controller), config);
        assert.deepEqual(m.view.seats.map(s => s.player), ids);
        if (m.view.phase === 'call') calls++;
        if (steps % 31 === 0) restored(m);
        if (turn(m) !== null) external++;
        m.advance(false);
      }
      assert.ok(steps < 600); if (initialWind !== m.view.seats[0].seat) rotations++;
      assert.ok(m.view.hands_played >= 2 || m.over); restored(m);
    } finally { m.dispose(); }
  }
  assert.ok(external > 0 && calls > 0 && rotations > 0);
});

test('uniform custom engine construction produces identical decisions to the original constructor', () => {
  for (const type of OPPONENT_TYPES) for (const seed of [1,31,81,287]) {
    const a = new Game(seed, type), b = Game.with_opponents(seed, type, type, type);
    try {
      a.advance(); b.advance();
      for (let i = 0; i < 100 && !a.hand_is_over(); i++) {
        assert.deepEqual(b.view(), a.view()); assert.deepEqual(b.choices(), a.choices());
        if (a.needs_opponent_move()) {
          assert.equal(b.opponent_player(), a.opponent_player());
          const action = enginePick(a.opponent_mask()); a.play_opponent(action); b.play_opponent(action);
        } else { const c = pick(a.choices()); a.choose(c.kind,c.tile??undefined); b.choose(c.kind,c.tile??undefined); }
        a.advance(); b.advance();
      }
      assert.deepEqual(b.view(), a.view());
    } finally { a.free(); b.free(); }
  }
});

test('single-player recovery retains the other trained and Beginner assignments', async () => {
  const m = make(1, ['neural','beginner','neural']);
  try {
    m.ai = async () => { throw new Error('network failed'); };
    // Reach a trained decision even if the human started as East.
    for (let i=0;i<20&&!m.needsRecovery;i++) { if(i===0) await m.run(); else await m.choose(pick(m.choices)); }
    assert.ok(m.needsRecovery); const pending = m.pendingOpponent; assert.ok(pending);
    const index = m.view.seats.findIndex(s => s.player === pending.player) - 1;
    m.ai = async (_obs, mask) => neuralChoice(mask);
    const engine = m.engine; assert.equal(await m.continueOpponentWithClub(), true);
    assert.equal(m.engine,engine);
    const expected = ['neural','beginner','neural']; expected[index]='club';
    assert.deepEqual(m.opponents,expected); assert.deepEqual(m.view.seats.slice(1).map(s=>s.controller),expected);
    restored(m);
  } finally { m.dispose(); }
});

test('pending-claim recovery keeps all existing responses and leaves other controllers intact', () => {
  let found = false;
  for (let seed=1;seed<=10&&!found;seed++) {
    const m=make(seed,['neural','beginner','neural']);
    try {
      m.advance(false);
      for(let n=0;n<250&&m.view.phase!=='over';n++) {
        if(m.view.phase==='call'&&m.engine.needs_opponent_move()) {
          const player=m.pendingOpponent.player, before=[...m.opponents];
          const index=m.view.seats.findIndex(s=>s.player===player)-1;
          const handBefore=m.view.hands_played;
          m.apply({type:'opponent-club',player}); m.advance(false);
          before[index]='club'; assert.deepEqual(m.opponents,before); assert.equal(m.view.hands_played,handBefore);
          restored(m); found=true; break;
        }
        turn(m); m.advance(false);
      }
    } finally {m.dispose();}
  }
  assert.ok(found,'Must test recovery while a claim is unresolved');
});

test('fallback-all changes only trained opponents and rejects wrong-player commands', () => {
  const m=make(287,['beginner','neural','club']);
  try {
    m.advance(false);
    const snapshot=m.snapshot(); assert.throws(()=>m.apply({type:'opponent-club',player:99}));
    assert.equal(m.stateKey(),snapshot.state); assert.deepEqual(m.opponents,['beginner','neural','club']);
    m.apply({type:'club'}); m.advance(false);
    assert.deepEqual(m.opponents,['beginner','club','club']); restored(m);
  } finally {m.dispose();}
});

test('old format-3 saves migrate only additive opponent identity metadata', () => {
  let migrated=0,refused=0;
  for(const difficulty of OPPONENT_TYPES) {
    const m=make(81,difficulty);
    try {
      m.advance(false); for(let n=0;n<24&&m.view.phase!=='over';n++){turn(m);m.advance(false);}
      const old=m.snapshot(); old.format=3; delete old.opponents;
      const data=JSON.parse(old.state); for(const seat of data[0].seats){delete seat.player;delete seat.controller;}
      old.state=JSON.stringify(data);
      // Before format 5 a trained opponent's answer was written down in our
      // own seventy-eight moves. Replaying one as Mortal's forty-six would
      // play a different game, so such a save is refused, not migrated.
      if(old.commands.some(c=>c.type==='opponent')) {
        assert.throws(()=>MatchSession.restore(Game,JSON.stringify(old)),/Unsupported legacy opponent moves/);
        refused++; continue;
      }
      const r=MatchSession.restore(Game,JSON.stringify(old));
      try {assert.deepEqual(r.opponents,normalizeOpponents(difficulty));assert.equal(r.stateKey(),m.stateKey());}
      finally {r.dispose();}
      data[0].seats[0].score+=1000; old.state=JSON.stringify(data);
      assert.throws(()=>MatchSession.restore(Game,JSON.stringify(old)),/does not match/);
      migrated++;
    } finally {m.dispose();}
  }
  assert.ok(migrated>0&&refused>0,'Exercised both an old save that still migrates and one that cannot');
});

test('a complete mixed hanchan keeps player identity, settings and final restoration', () => {
  const m=make(14,['beginner','neural','club']);
  try {
    m.advance(false); const ids=m.view.seats.map(s=>s.player); const winds=new Set();
    for(let n=0;n<8000&&!m.over;n++) {
      winds.add(m.view.seats[0].seat); assert.deepEqual(m.view.seats.map(s=>s.player),ids);
      assert.deepEqual(m.view.seats.slice(1).map(s=>s.controller),m.opponents);
      turn(m); m.advance(false);
    }
    assert.ok(m.over); assert.equal(winds.size,4); restored(m);
    const bad=m.snapshot(); bad.opponents=['club','neural','beginner'];
    assert.throws(()=>MatchSession.restore(Game,JSON.stringify(bad)));
  } finally {m.dispose();}
});
