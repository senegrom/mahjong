import assert from 'node:assert/strict';
import test from 'node:test';
import { MatchSession, readSettings, writeSettings, SETTINGS_KEY, DEFAULT_REVIEW_ADVISER } from '../src/lib/session.js';
import { DEFAULT_OPPONENT, normalizeOpponents, sameOpponents } from '../src/lib/opponents.js';
import { heldSafeCount, callTiles, callLabel, unseenTileCounts, waitingNote } from '../src/lib/ui.js';

const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
class FakeGame {
  constructor(seed, difficulty) { this.seed = seed; this.external = difficulty === 'neural'; this.pending = this.external; this.moves = 0; this.applied = 0; this.declared = 0; this.awaitingTile = false; this.freed = false; }
  view() { assert.ok(!this.freed); return { phase: 'act', hands_played: 0, seats: [{discards: Array(this.moves).fill('1m'), melds: [], riichi: false}], moves: this.moves, pending: this.pending }; }
  choices() { return this.pending ? [] : [{ kind: 'discard', tile: '1m' }]; }
  advance() { return []; }
  needs_opponent_move() { return this.pending; }
  /** Mortal's forty-six moves, which is what a trained opponent answers in.
   * Move 0 discards; move 37 declares a reach, which plays nothing until a
   * second answer names the tile it discards. */
  opponent_mask_mortal() { const mask = new Uint8Array(46); mask[0] = 1; if (!this.awaitingTile) mask[37] = 1; return mask; }
  opponent_observation_mortal() { return new Float32Array([this.awaitingTile ? 1 : 0]); }
  play_opponent_mortal(action) {
    assert.ok(!this.freed);
    if (action === 37) { this.declared++; this.awaitingTile = true; return true; }
    assert.equal(action, 0); this.applied++; this.awaitingTile = false; this.pending = false; return false;
  }
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

test('a declared reach is answered twice and replays without asking again', async () => {
  const match = new MatchSession(FakeGame, 1, 'neural', {ai: async (_planes, mask) => mask[37] ? 37 : 0});
  assert.equal(await match.run(), true);
  assert.deepEqual(match.commands, [{type:'opponent',action:37},{type:'opponent',action:0}]);
  assert.equal(match.engine.declared, 1); assert.equal(match.engine.applied, 1); assert.equal(match.failure, '');
  const restored = MatchSession.restore(FakeGame, JSON.stringify(match.snapshot()),
    {ai(){throw new Error('A saved reach must not be put to the network again');}});
  assert.equal(restored.stateKey(), match.stateKey());
  assert.equal(restored.engine.declared, 1); assert.equal(restored.engine.applied, 1);
  match.dispose(); restored.dispose();
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
  assert.deepEqual(readSettings({getItem(){throw new Error('denied');}},true), {difficulty:'neural',hints:true,confirmDiscards:true,shortcuts:true,tileFace:'classic',reviewAdviser:'strong'});
  assert.equal(readSettings({getItem(){return '{broken';}}).hints, true);
  const value = {version:1,difficulty:'club',hints:false,confirmDiscards:false,shortcuts:false};
  assert.deepEqual(readSettings({getItem(){return JSON.stringify(value);}}), {difficulty:'club',hints:false,confirmDiscards:false,shortcuts:false,tileFace:'classic',reviewAdviser:'strong'});
});

test('the review adviser is remembered, and a retired network preference is ignored', () => {
  for (const reviewAdviser of ['club', 'strong', 'quick', null, '../other']) {
    // Older builds also stored which of two networks played; one ships now.
    const settings = readSettings({ getItem: () => JSON.stringify({ version: 1, trainedModel: 'strong', reviewAdviser }) });
    assert.equal(settings.reviewAdviser, reviewAdviser === 'club' ? 'club' : 'strong');
    assert.equal(Object.hasOwn(settings, 'trainedModel'), false);
  }
});

// How App reads its opening table from the preferences.
const table = settings => normalizeOpponents(settings.opponents ?? settings.difficulty);

test('a first visit meets Trained opponents and is reviewed by Trained AI', () => {
  for (const storage of [null, { getItem: () => null }]) {
    const settings = readSettings(storage);
    assert.equal(settings.difficulty, DEFAULT_OPPONENT);
    assert.deepEqual(table(settings), ['neural', 'neural', 'neural']);
    assert.equal(settings.reviewAdviser, DEFAULT_REVIEW_ADVISER);
    assert.equal(DEFAULT_REVIEW_ADVISER, 'strong');
  }
  // A record this build cannot read is no choice at all.
  assert.deepEqual(table(readSettings({ getItem: () => JSON.stringify({ version: 2, difficulty: 'club' }) })), ['neural', 'neural', 'neural']);
});

test('saved choices stand, Club included, and only what a record lacks takes the new default', () => {
  const read = value => readSettings({ getItem: () => JSON.stringify(value) });
  // What the previous build wrote on a first visit: the old defaults, which
  // cannot be told from a player who chose Club. Both keep Club.
  const previous = read({ version: 1, difficulty: 'club', opponents: ['club', 'club', 'club'], hints: true,
    confirmDiscards: false, shortcuts: true, tileFace: 'classic', reviewAdviser: 'club' });
  assert.deepEqual(table(previous), ['club', 'club', 'club']);
  assert.equal(previous.difficulty, 'club');
  assert.equal(previous.reviewAdviser, 'club');
  // A table chosen seat by seat stays exactly as it was.
  assert.deepEqual(table(read({ version: 1, difficulty: 'custom', opponents: ['beginner', 'club', 'neural'] })), ['beginner', 'club', 'neural']);
  // From before seat-by-seat tables and before the review adviser existed:
  // the single strength stands, the adviser was never chosen.
  const older = read({ version: 1, difficulty: 'beginner' });
  assert.deepEqual(table(older), ['beginner', 'beginner', 'beginner']);
  assert.equal(older.reviewAdviser, 'strong');
  // A Club chosen in this build is written and read back as Club.
  const writes = new Map();
  writeSettings({ setItem: (key, value) => writes.set(key, value) },
    { difficulty: 'club', opponents: ['club', 'club', 'club'], hints: true, reviewAdviser: 'club' });
  const reread = readSettings({ getItem: key => writes.get(key) ?? null });
  assert.deepEqual(table(reread), ['club', 'club', 'club']);
  assert.equal(reread.reviewAdviser, 'club');
});

test('the table shows the network download while a Trained opponent waits for it', () => {
  // The worker's own note wins; before the worker starts, the page's offline
  // download is the wait, and nothing is said when nothing is downloading.
  assert.equal(waitingNote('loading the network', { phase: 'ai', progress: 40 }), 'loading the network');
  assert.equal(waitingNote('', { phase: 'ai', progress: 40 }), 'downloading the trained network 40%');
  assert.equal(waitingNote('', { phase: 'ai' }), 'downloading the trained network 0%');
  for (const phase of ['checking', 'ready', 'incomplete', 'unavailable']) assert.equal(waitingNote('', { phase, progress: 40 }), '');
  assert.equal(waitingNote('', null), '');
  // A first visit's download waits for the service worker to save the game
  // and its tile graphics, which reports no progress: that wait is named too.
  const installing = waitingNote('', { phase: 'checking', progress: 0, installing: true });
  assert.equal(installing, 'saving the game and tile graphics, then downloading the trained network');
  assert.equal(waitingNote('starting the network', { phase: 'checking', installing: true }), 'starting the network');
  assert.equal(waitingNote('', { phase: 'ai', progress: 5, installing: true }), 'downloading the trained network 5%');
  assert.equal(waitingNote('', { phase: 'checking', installing: false }), '');
});

test('preferences are written when they change, not each time the effect reruns', () => {
  const writes = [];
  let fail = false;
  const storage = { setItem(key, value) { if (fail) throw new Error('denied'); writes.push([key, value]); } };
  const preferences = { difficulty: 'club', opponents: ['club', 'club', 'club'], hints: true };
  let last = writeSettings(storage, preferences);
  // App's effect reruns with an equal copy after every move of a match.
  for (let move = 0; move < 5; move++) last = writeSettings(storage, { ...preferences, opponents: [...preferences.opponents] }, last);
  assert.deepEqual(writes, [[SETTINGS_KEY, JSON.stringify({ version: 1, ...preferences })]]);
  assert.deepEqual(readSettings({ getItem: () => writes[0][1] }).opponents, preferences.opponents);
  // Another tab's change stays until this tab changes something itself.
  last = writeSettings(storage, { ...preferences, hints: false }, last);
  assert.equal(writes.length, 2);
  fail = true;
  const kept = writeSettings(storage, { ...preferences, hints: true }, last);
  assert.equal(kept, last, 'a write that failed is not remembered as stored');
  fail = false;
  writeSettings(storage, { ...preferences, hints: true }, kept);
  assert.equal(writes.length, 3);
  assert.doesNotThrow(() => writeSettings(null, preferences));
});

test('an equal opponent table is not a change of table', () => {
  assert.equal(sameOpponents(['club', 'neural', 'beginner'], ['club', 'neural', 'beginner']), true);
  assert.equal(sameOpponents(['club', 'neural', 'beginner'], ['club', 'club', 'beginner']), false);
  assert.equal(sameOpponents(['club', 'club', 'club'], ['club', 'club']), false);
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

class ScoredGame extends FakeGame {
  view() { return { ...super.view(), outcome: { wins: [{ hand: ['1z'], winning_tile: '1z',
    han: 1, payment: '1000', scoring: { sets: [], pair: null },
    yaku: [{ name: 'Round Wind Triplet', han: 1, id: 'YakuhaiRoundWind', tile: '1z' }] }] } }; }
}
test('saved completed matches migrate old attribution only and keep strict new-format checking', () => {
  const match = new MatchSession(ScoredGame, 1, 'club');
  try {
    const current = match.snapshot();
    assert.equal(current.format, 6);
    const old = { ...current, format: 5 }, data = JSON.parse(old.state), win = data[0].outcome.wins[0];
    delete win.scoring; delete win.yaku[0].id; delete win.yaku[0].tile;
    old.state = JSON.stringify(data);
    const read = MatchSession.restore(ScoredGame, JSON.stringify(old));
    try { assert.deepEqual(read.snapshot(), current); } finally { read.dispose(); }
    for (const change of [w => { w.hand[0] = '2z'; }, w => { w.payment = '2000'; },
      w => { w.yaku[0].han = 2; }]) {
      const bad = JSON.parse(old.state); change(bad[0].outcome.wins[0]);
      assert.throws(() => MatchSession.restore(ScoredGame, JSON.stringify({ ...old, state: JSON.stringify(bad) })), /does not match/);
    }
    assert.throws(() => MatchSession.restore(ScoredGame, JSON.stringify({ ...old, format: 6 })), /does not match/);
  } finally { match.dispose(); }
});
