import assert from 'node:assert/strict';
import test from 'node:test';
import { MatchSession, readSettings } from '../src/lib/session.js';
import { heldSafeCount, callTiles, callLabel, unseenTileCounts } from '../src/lib/ui.js';

const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
class FakeGame {
  constructor(seed, difficulty) { this.seed = seed; this.external = difficulty === 'neural'; this.pending = this.external; this.moves = 0; this.applied = 0; this.freed = false; }
  view() { assert.ok(!this.freed); return { phase: 'act', hands_played: 0, seats: [{discards: Array(this.moves).fill('1m'), melds: [], riichi: false}], moves: this.moves, pending: this.pending }; }
  choices() { return this.pending ? [] : [{ kind: 'discard', tile: '1m' }]; }
  advance() { return []; }
  needs_opponent_move() { return this.pending; }
  opponent_mask() { return new Uint8Array([1, 0]); }
  opponent_observation() { return new Float32Array([0]); }
  play_opponent(action) { assert.ok(!this.freed); assert.equal(action, 0); this.applied++; this.pending = false; }
  continue_with_club() { this.external = false; this.pending = false; }
  choose() { this.moves++; this.pending = this.external; }
  game_is_over() { return false; }
  hand_is_over() { return false; }
  free() { this.freed = true; }
}

test('late AI answer cannot mutate a restarted match or notify its UI', async () => {
  const answer = deferred(); let changes = 0;
  const old = new MatchSession(FakeGame, 1, 'neural', {ai: () => answer.promise, onChange: () => changes++});
  const run = old.run(); const oldEngine = old.engine;
  old.dispose(); const checkpoint = changes;
  const replacement = new MatchSession(FakeGame, 2, 'club');
  await replacement.run(); answer.resolve(0); await run;
  assert.equal(oldEngine.applied, 0); assert.equal(changes, checkpoint);
  assert.equal(replacement.engine.moves, 0); assert.equal(replacement.failure, '');
  replacement.dispose();
});

test('late AI error cannot alter the replacement loading state', async () => {
  const answer = deferred(); const old = new MatchSession(FakeGame, 1, 'neural', {ai: () => answer.promise});
  const running = old.run(); old.dispose();
  const freshAnswer = deferred(); const fresh = new MatchSession(FakeGame, 2, 'neural', {ai: () => freshAnswer.promise});
  const freshRun = fresh.run(); answer.reject(new Error('old network error')); await running;
  assert.equal(fresh.busy, true); assert.equal(fresh.thinking, true); assert.equal(fresh.failure, '');
  freshAnswer.resolve(0); await freshRun; fresh.dispose();
});

test('AI failure remains explicit and retry succeeds without changing opponents', async () => {
  let fail = true;
  const match = new MatchSession(FakeGame, 1, 'neural', {ai: async () => {if (fail) throw new Error('offline'); return 0;}});
  await match.run(); assert.equal(match.needsRecovery, true); assert.equal(match.difficulty, 'neural'); assert.equal(match.busy, false);
  fail = false; await match.retry(); assert.equal(match.failure, ''); assert.equal(match.needsRecovery, false); match.dispose();
});

test('continue with Club changes the engine in place and survives restoration', async () => {
  const match = new MatchSession(FakeGame, 1, 'neural', {ai: async () => {throw new Error('offline');}});
  await match.run(); const engine = match.engine; await match.continueWithClub();
  assert.equal(match.engine, engine); assert.equal(engine.external, false); assert.equal(match.difficulty, 'club');
  const restored = MatchSession.restore(FakeGame, JSON.stringify(match.snapshot()));
  assert.equal(restored.difficulty, 'club'); assert.equal(restored.stateKey(), match.stateKey()); match.dispose(); restored.dispose();
});

test('busy sessions ignore additional player choices', async () => {
  const answer = deferred(); const match = new MatchSession(FakeGame, 1, 'neural', {ai: () => answer.promise});
  const run = match.run(); assert.equal(await match.choose({kind:'discard',tile:'1m'}), false);
  assert.equal(match.commands.length, 0); answer.resolve(0); await run; match.dispose();
});

test('illegal neural output is rejected rather than used as a fallback move', async () => {
  const match = new MatchSession(FakeGame, 1, 'neural', {ai: async () => 1});
  await match.run(); assert.match(match.failure, /Invalid opponent/); assert.equal(match.engine.applied, 0); match.dispose();
});

test('restoration rejects invalid versions, illegal commands and changed state', async () => {
  const match = new MatchSession(FakeGame, 1, 'club'); await match.run(); const saved = match.snapshot();
  assert.throws(() => MatchSession.restore(FakeGame, JSON.stringify({...saved, version: 99})));
  assert.throws(() => MatchSession.restore(FakeGame, JSON.stringify({...saved, commands: [{type:'choose',kind:'discard',tile:'9z'}]})));
  assert.throws(() => MatchSession.restore(FakeGame, JSON.stringify({...saved, state: 'different'})));
  assert.throws(() => MatchSession.restore(FakeGame, '{broken'));
  match.dispose();
});

test('preferences tolerate inaccessible or malformed storage', () => {
  assert.deepEqual(readSettings({getItem(){throw new Error('denied');}},true), {difficulty:'club',hints:true,confirmDiscards:true,shortcuts:true,tileFace:'classic',trainedModel:'quick',reviewAdviser:'club'});
  assert.equal(readSettings({getItem(){return '{broken';}}).hints, true);
  const value = {version:1,difficulty:'neural',hints:false,confirmDiscards:false,shortcuts:false};
  assert.deepEqual(readSettings({getItem(){return JSON.stringify(value);}}), {difficulty:'neural',hints:false,confirmDiscards:false,shortcuts:false,tileFace:'classic',trainedModel:'quick',reviewAdviser:'club'});
});

test('the trained opponent chosen is remembered, and nonsense is not', () => {
  const read = value => readSettings({ getItem: () => JSON.stringify(value) }).trainedModel;
  assert.equal(read({ version: 1, trainedModel: 'strong' }), 'strong');
  assert.equal(read({ version: 1, trainedModel: 'quick' }), 'quick');
  for (const trainedModel of [undefined, null, '', false, {}, 'huge', '../other']) {
    assert.equal(read({ version: 1, trainedModel }), 'quick');
  }
});

test('the review adviser is remembered independently of the opponent network', () => {
  for (const reviewAdviser of ['club', 'strong', 'quick', null, '../other']) {
    const settings = readSettings({ getItem: () => JSON.stringify({ version: 1, trainedModel: 'quick', reviewAdviser }) });
    assert.equal(settings.reviewAdviser, reviewAdviser === 'strong' ? 'strong' : 'club');
    assert.equal(settings.trainedModel, 'quick');
  }
});

test('safe count includes held copies, not absent globally safe kinds', () => {
  const view = {phase:'act',safe:['1m','2m','3m','4m','5m'],seats:[{hand:['1m','1m','9p'],drawn:'1m'}]};
  assert.equal(heldSafeCount(view), 3); assert.equal(heldSafeCount({...view,phase:'over'}), 0);
});

test('call previews use tile identity and explain mahjong terminology', () => {
  assert.deepEqual(callTiles({kind:'chii',tile:'5m'},'6m'), ['5m','6m','7m']);
  assert.equal(callLabel({kind:'chii',tile:'5m'}), 'Chii — 5–6–7 characters');
  assert.deepEqual(callTiles({kind:'pon'},'7z'), ['7z','7z','7z']);
  assert.deepEqual(callTiles({kind:'kan'},'1s'), ['1s','1s','1s','1s']);
  assert.match(callLabel({kind:'ron'}), /Ron.*win on discard/);
  assert.match(callLabel({kind:'tsumo'}), /Tsumo.*self-draw/);
});


test('remaining-copy hints count public information once, including claimed tiles via melds', () => {
  const left = unseenTileCounts({
    dora_indicators: ['1m'],
    seats: [
      { hand:['1m','2p'], drawn:'3s', discards:[], melds:[] },
      { hand:[], drawn:null, discards:[{tile:'4m',claimed:true},{tile:'5p',claimed:false}], melds:[{tiles:['4m','4m','4m']}] },
      { hand:[], drawn:null, discards:[{tile:'6z',claimed:false}], melds:[] },
      { hand:[], drawn:null, discards:[], melds:[] },
    ],
  });
  assert.equal(left.get('1m'), 2); // one held, one indicator
  assert.equal(left.get('2p'), 3);
  assert.equal(left.get('3s'), 3);
  assert.equal(left.get('4m'), 1); // claimed pond tile is not counted twice
  assert.equal(left.get('5p'), 3);
  assert.equal(left.get('6z'), 3);
  assert.equal(left.get('9s'), 4);
});
