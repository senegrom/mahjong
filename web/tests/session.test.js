import assert from 'node:assert/strict';
import test from 'node:test';
import { MatchSession, readSettings, writeSettings, SETTINGS_KEY, DEFAULT_REVIEW_ADVISER } from '../src/lib/session.js';
import { DEFAULT_OPPONENT, normalizeOpponents, sameOpponents } from '../src/lib/opponents.js';
import { heldSafeCount, callTiles, callLabel, unseenTileCounts, waitingNote } from '../src/lib/ui.js';
import { MANIFEST, NETWORKS, networkFor } from '../src/lib/trained-network.js';

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
  assert.deepEqual(readSettings({getItem(){throw new Error('denied');}},true), {difficulty:'neural',hints:true,confirmDiscards:true,shortcuts:true,tileFace:'classic',reviewAdviser:'strong',reviewHideMove:false,reviewHideTiles:false,reviewFilter:'flagged'});
  assert.equal(readSettings({getItem(){return '{broken';}}).hints, true);
  const value = {version:2,difficulty:'club',hints:false,confirmDiscards:false,shortcuts:false};
  assert.deepEqual(readSettings({getItem(){return JSON.stringify(value);}}), {difficulty:'club',hints:false,confirmDiscards:false,shortcuts:false,tileFace:'classic',reviewAdviser:'strong',reviewHideMove:false,reviewHideTiles:false,reviewFilter:'flagged'});
});

test("the review's two hiding settings start off, and a record keeps them as it holds them", () => {
  const read = value => readSettings({ getItem: () => JSON.stringify(value) });
  // A first visit, and a record from before the settings were offered, hide
  // nothing: every review shows Trained AI's move and every hand.
  for (const settings of [readSettings(null), read({ version: 3, difficulty: 'club' }), read({ version: 2, hints: false })]) {
    assert.deepEqual([settings.reviewHideMove, settings.reviewHideTiles], [false, false]);
  }
  // Each is kept as the record holds it, the one apart from the other.
  for (const [reviewHideMove, reviewHideTiles] of [[true, false], [false, true], [true, true]]) {
    const settings = read({ version: 3, reviewHideMove, reviewHideTiles });
    assert.deepEqual([settings.reviewHideMove, settings.reviewHideTiles], [reviewHideMove, reviewHideTiles]);
  }
  // Anything but true or false is no choice: the review hides nothing.
  for (const held of ['true', 1, null, {}, ['yes']]) {
    const settings = read({ version: 3, reviewHideMove: held, reviewHideTiles: held });
    assert.deepEqual([settings.reviewHideMove, settings.reviewHideTiles], [false, false], JSON.stringify(held));
  }
  // Written with the other preferences, and read back the same.
  const writes = new Map();
  writeSettings({ setItem: (key, value) => writes.set(key, value) }, { hints: true, reviewHideMove: true, reviewHideTiles: false });
  const reread = readSettings({ getItem: key => writes.get(key) ?? null });
  assert.deepEqual([reread.reviewHideMove, reread.reviewHideTiles, reread.hints], [true, false, true]);
});

test("the review's filter starts on the flagged decisions, is kept as the player leaves it, and falls back from one it does not offer", () => {
  const read = value => readSettings({ getItem: () => JSON.stringify(value) });
  // A first visit, and a record from before the filter was kept.
  for (const settings of [readSettings(null), read({ version: 3, difficulty: 'club' }), read({ version: 2, hints: false })]) {
    assert.equal(settings.reviewFilter, 'flagged');
  }
  for (const reviewFilter of ['flagged', 'differed', 'every']) assert.equal(read({ version: 3, reviewFilter }).reviewFilter, reviewFilter);
  // A filter the review does not offer is no choice: it steps through the
  // flagged decisions, and the rest of the record stands.
  for (const reviewFilter of ['all', 'Flagged', '', null, 2, ['every'], { filter: 'every' }]) {
    const settings = read({ version: 3, reviewFilter, hints: false });
    assert.equal(settings.reviewFilter, 'flagged', JSON.stringify(reviewFilter));
    assert.equal(settings.hints, false);
  }
  // Written with the other preferences, and read back the same.
  const writes = new Map();
  writeSettings({ setItem: (key, value) => writes.set(key, value) }, { hints: true, reviewFilter: 'differed' });
  assert.equal(readSettings({ getItem: key => writes.get(key) ?? null }).reviewFilter, 'differed');
});

test('the review adviser is remembered, and a retired network preference is ignored', () => {
  for (const reviewAdviser of ['club', 'strong', 'quick', null, '../other']) {
    // Older builds also stored which of two networks played, as 'quick' or
    // 'strong'. That preference is retired: a network is picked by version.
    const settings = readSettings({ getItem: () => JSON.stringify({ version: 2, trainedModel: 'strong', reviewAdviser }) });
    assert.equal(settings.reviewAdviser, reviewAdviser === 'club' ? 'club' : 'strong');
    assert.equal(Object.hasOwn(settings, 'trainedModel'), false);
    assert.equal(Object.hasOwn(settings, 'trainedNetwork'), false);
    // Before 810 every review went back to Trained once (see the next tests).
    assert.equal(readSettings({ getItem: () => JSON.stringify({ version: 1, reviewAdviser }) }).reviewAdviser, 'strong');
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
  for (const version of [0, 5, '4', undefined]) {
    assert.deepEqual(table(readSettings({ getItem: () => JSON.stringify({ version, difficulty: 'club' }) })), ['neural', 'neural', 'neural']);
  }
});

test('a record from before 810 meets Trained once, and keeps everything else it holds', () => {
  const read = value => readSettings({ getItem: () => JSON.stringify(value) });
  // What builds before 810 wrote: whoever played and reviewed gives way to
  // Trained, Club, Beginner and seat-by-seat tables alike.
  for (const choice of [{ difficulty: 'club', opponents: ['club', 'club', 'club'], reviewAdviser: 'club' },
    { difficulty: 'custom', opponents: ['beginner', 'club', 'neural'] }, { difficulty: 'beginner' }]) {
    const previous = read({ version: 1, ...choice, hints: false, confirmDiscards: true, shortcuts: false, tileFace: 'van-gogh' });
    assert.deepEqual(table(previous), ['neural', 'neural', 'neural']);
    assert.equal(previous.difficulty, 'neural');
    assert.equal(previous.reviewAdviser, 'strong');
    assert.deepEqual([previous.hints, previous.confirmDiscards, previous.shortcuts, previous.tileFace], [false, true, false, 'van-gogh']);
  }
});

test('saved choices stand, Club included, and only what a record lacks takes the new default', () => {
  const read = value => readSettings({ getItem: () => JSON.stringify(value) });
  const chosen = read({ version: 2, difficulty: 'club', opponents: ['club', 'club', 'club'], hints: true,
    confirmDiscards: false, shortcuts: true, tileFace: 'classic', reviewAdviser: 'club' });
  assert.deepEqual(table(chosen), ['club', 'club', 'club']);
  assert.equal(chosen.difficulty, 'club');
  assert.equal(chosen.reviewAdviser, 'club');
  // A table chosen seat by seat stays exactly as it was.
  assert.deepEqual(table(read({ version: 2, difficulty: 'custom', opponents: ['beginner', 'club', 'neural'] })), ['beginner', 'club', 'neural']);
  // A record without seat-by-seat tables or a review adviser: the single
  // strength stands, and the adviser takes the default.
  const older = read({ version: 2, difficulty: 'beginner' });
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

// The network a page plays for these preferences, as App picks it.
const playing = settings => networkFor(settings.trainedNetwork).version;
// A pick of every network the build lists but the default.
const OLDER = NETWORKS.slice(1).map(network => network.version);

test('a picked trained network is kept by version, and without a listed pick the default plays', () => {
  const read = value => readSettings({ getItem: () => JSON.stringify(value) });
  // A first visit, and a record that holds no pick, play the default.
  for (const settings of [readSettings(null), readSettings({ getItem: () => null }), read({ version: 4, difficulty: 'club' })]) {
    assert.equal(Object.hasOwn(settings, 'trainedNetwork'), false);
    assert.equal(playing(settings), MANIFEST.version);
  }
  // Every listed network is read back as it was picked.
  assert.ok(OLDER.length >= 2, 'the networks played before the default stay listed');
  for (const version of NETWORKS.map(network => network.version)) {
    assert.equal(read({ version: 4, trainedNetwork: version }).trainedNetwork, version);
  }
  // Anything that is not a listed version is no pick: the default plays.
  for (const trainedNetwork of ['0.1.36', '0.1.4100', 'strong', 'full', '', 410, null, ['0.1.410'], { version: '0.1.410' }]) {
    const settings = read({ version: 4, trainedNetwork, hints: false });
    assert.equal(Object.hasOwn(settings, 'trainedNetwork'), false, JSON.stringify(trainedNetwork));
    assert.equal(playing(settings), MANIFEST.version);
    assert.equal(settings.hints, false, 'the rest of the record stands');
  }
  // A record from before 810 holds no pick; it would not be read if it did.
  assert.equal(Object.hasOwn(read({ version: 1, trainedNetwork: '0.1.410' }), 'trainedNetwork'), false);
  // Round trip: written with the other preferences, in the version 4 record.
  const writes = new Map();
  const storage = { setItem: (key, value) => writes.set(key, value), getItem: key => writes.get(key) ?? null };
  const preferences = { difficulty: 'neural', opponents: ['neural', 'neural', 'neural'], hints: true, confirmDiscards: false,
    shortcuts: true, tileFace: 'classic', reviewAdviser: 'strong', reviewHideMove: false, reviewHideTiles: false,
    reviewFilter: 'flagged' };
  writeSettings(storage, { ...preferences, trainedNetwork: '0.1.410' });
  assert.deepEqual(JSON.parse(writes.get(SETTINGS_KEY)), { version: 4, ...preferences, trainedNetwork: '0.1.410' });
  assert.deepEqual(readSettings(storage), { ...preferences, trainedNetwork: '0.1.410' });
  // A player who has picked nothing writes no pick, so a newer default reaches them.
  writeSettings(storage, { ...preferences, trainedNetwork: undefined });
  assert.equal(Object.hasOwn(JSON.parse(writes.get(SETTINGS_KEY)), 'trainedNetwork'), false);
  assert.equal(playing(readSettings(storage)), MANIFEST.version);
});

test('a record from before 1110 or 1410 plays the default once, whatever it picked, and keeps every other choice', () => {
  // Every other choice a record holds, none of them a default.
  const chosen = { difficulty: 'custom', opponents: ['beginner', 'club', 'neural'], hints: false, confirmDiscards: true,
    shortcuts: false, tileFace: 'van-gogh', reviewAdviser: 'club', reviewHideMove: true, reviewHideTiles: true, reviewFilter: 'every' };
  // What builds before 1110 wrote: 810 was the default then and was never
  // written as a pick, but a record that holds it gives it up all the same.
  // Builds before 1410 wrote version 3, with 1110 the default, and a record
  // of theirs gives up its pick the same way.
  const before = [2, 3].flatMap(record => [...OLDER, undefined, '0.1.36'].map(trainedNetwork => ({ record, trainedNetwork })));
  for (const { record, trainedNetwork } of before) {
    const writes = new Map([[SETTINGS_KEY, JSON.stringify({ version: record, ...chosen, trainedNetwork })]]);
    const storage = { setItem: (key, value) => writes.set(key, value), getItem: key => writes.get(key) ?? null };
    const settings = readSettings(storage);
    assert.deepEqual(settings, chosen, `version ${record}, ${trainedNetwork}`);
    assert.equal(playing(settings), MANIFEST.version);
    // The page writes the record back as it loads (App's effect), as version 4
    // and without a pick, so the next load reads the same.
    writeSettings(storage, settings);
    assert.deepEqual(JSON.parse(writes.get(SETTINGS_KEY)), { version: 4, ...chosen });
    assert.deepEqual(readSettings(storage), chosen);
    // A pick made after that is kept, load after load.
    for (const version of OLDER) {
      writeSettings(storage, { ...settings, trainedNetwork: version });
      for (let load = 0; load < 2; load++) {
        const again = readSettings(storage);
        assert.deepEqual(again, { ...chosen, trainedNetwork: version });
        assert.equal(playing(again), version);
        writeSettings(storage, again);
      }
    }
  }
  // A record from before 810 is read as it always was: its hints, keys and
  // tile face stand, and Trained plays and reviews. Its review's settings are
  // read as its hints are, though no build of then wrote any.
  assert.deepEqual(readSettings({ getItem: () => JSON.stringify({ version: 1, ...chosen, trainedNetwork: OLDER[0] }) }),
    { difficulty: 'neural', hints: false, confirmDiscards: true, shortcuts: false, tileFace: 'van-gogh', reviewAdviser: 'strong',
      reviewHideMove: true, reviewHideTiles: true, reviewFilter: 'every' });
});

test('a match saved with Trained opponents before the settings reset resumes, and asks the page for the next move', async () => {
  // A match's own save is unchanged: version 1, as before 810, 1110 and 1410.
  const match = new MatchSession(FakeGame, 1, 'neural', { ai: async (_planes, mask) => mask[37] ? 37 : 0 });
  await match.run();
  await match.choose({ kind: 'discard', tile: '1m' });
  const saved = match.snapshot();
  assert.equal(saved.version, 1);
  assert.deepEqual(saved.opponents, ['neural', 'neural', 'neural']);
  // Restored as a page would after the reload, its recorded moves are replayed
  // without the network, and the move still to come is asked of the network
  // the page now plays.
  let asked = 0;
  const restored = MatchSession.restore(FakeGame, JSON.stringify(saved), { ai: async (_planes, mask) => { asked++; return mask[37] ? 37 : 0; } });
  try {
    assert.equal(restored.stateKey(), match.stateKey());
    assert.deepEqual(restored.commands, saved.commands);
    assert.equal(asked, 0);
    assert.equal(await restored.choose({ kind: 'discard', tile: '1m' }), true);
    assert.equal(asked, 2, 'a reach and its discard');
  } finally { match.dispose(); restored.dispose(); }
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
  assert.deepEqual(writes, [[SETTINGS_KEY, JSON.stringify({ version: 4, ...preferences })]]);
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
    assert.equal(current.format, 7);
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
