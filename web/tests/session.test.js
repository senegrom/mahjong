import assert from 'node:assert/strict';
import test from 'node:test';
import { MatchSession, readSettings, writeSettings, SETTINGS_KEY, DEFAULT_REVIEW_ADVISER } from '../src/lib/session.js';
import { DEFAULT_OPPONENT, normalizeOpponents, sameOpponents, standInFor, tableAfterStandIn, trainedAtHand, trainedUnderWay } from '../src/lib/opponents.js';
import { heldSafeCount, callTiles, callLabel, partedNote, startupWords, unseenTileCounts, waitIsDownload, waitingNote } from '../src/lib/ui.js';
import { MANIFEST, NETWORKS, networkFor } from '../src/lib/trained-network.js';
import { average, swapMoves } from '../src/lib/suits.js';
import { DAMA, DECLINE, RULE_REFUSED } from '../src/lib/trained-move.js';

const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
/** Weights over Mortal's forty-six moves, nought where none is given. */
const weightsOf = entries => Array.from({ length: 46 }, (_, move) => entries[move] ?? 0);
/** An answer as analyzePolicy gives it, sure of its move. */
const sure = action => ({ action, weights: weightsOf({ [action]: 1 }) });
/** The commands a Trained opponent's Mortal moves are recorded as. */
const played = (...actions) => actions.map(action => ({ type: 'opponent', action }));
class FakeGame {
  constructor(seed, difficulty) {
    this.seed = seed; this.external = difficulty === 'neural'; this.pending = this.external; this.moves = 0; this.applied = 0; this.declared = 0; this.awaitingTile = false; this.freed = false;
    // The moves the opponent's first question opens, which a test may change;
    // the last hand's codes, nought unless a test sets them, `dama` by tile;
    // and how often the questions read only when wanted were read.
    this.open = [0, 37]; this.dama = {}; this.decline = 0; this.reads = { reachRenamed: 0, dama: 0, decline: 0 };
  }
  view() { assert.ok(!this.freed); return { phase: 'act', hands_played: 0, seats: [{discards: Array(this.moves).fill('1m'), melds: [], riichi: false}], moves: this.moves, pending: this.pending }; }
  choices() { return this.pending ? [] : [{ kind: 'discard', tile: '1m' }]; }
  advance() { return []; }
  needs_opponent_move() { return this.pending; }
  /** Mortal's forty-six moves, which is what a trained opponent answers in.
   * Move 0 discards; move 37 declares a reach, which plays nothing until a
   * second answer names the tile it discards; a test may open a win (43), a
   * pon (41) or the pass (45). */
  opponent_mask_mortal() { const mask = new Uint8Array(46); for (const move of this.awaitingTile ? [0] : this.open) mask[move] = 1; return mask; }
  opponent_observation_mortal() { return new Float32Array([this.awaitingTile ? 1 : 0]); }
  /** The renamed game's planes, other than the game's own. */
  opponent_observation_mortal_renamed() { this.renamedReads = (this.renamedReads ?? 0) + 1; assert.ok(!this.freed); return new Float32Array([this.awaitingTile ? 11 : 10]); }
  /** The reach's tile, asked before anything is recorded, of a preview that
   * reads as the pending reach does once its riichi is played. */
  opponent_observation_after_reach() { assert.ok(!this.freed); return new Float32Array([1]); }
  opponent_observation_after_reach_renamed() { this.reads.reachRenamed++; assert.ok(!this.freed); return new Float32Array([11]); }
  opponent_mask_after_reach() { assert.ok(!this.freed); const mask = new Uint8Array(46); mask[0] = 1; return mask; }
  opponent_all_last_dama() { this.reads.dama++; assert.ok(!this.freed); return Uint8Array.from({ length: 34 }, (_, tile) => this.dama[tile] ?? 0); }
  opponent_all_last_decline() { this.reads.decline++; assert.ok(!this.freed); return this.decline; }
  play_opponent_mortal(action) {
    assert.ok(!this.freed);
    assert.ok(this.opponent_mask_mortal()[action], `move ${action} is open`);
    if (action === 37) { this.declared++; this.awaitingTile = true; return true; }
    this.applied++; this.awaitingTile = false; this.pending = false; return false;
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
  await replacement.run(); answer.resolve(sure(0)); await run;
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
  freshAnswer.resolve(sure(0)); await freshRun; fresh.dispose();
});

test('AI failure remains explicit and retry succeeds without changing opponents', async () => {
  let fail = true;
  const match = new MatchSession(FakeGame, 1, 'neural', {ai: async () => {if (fail) throw new Error('offline'); return sure(0);}});
  await match.run(); assert.equal(match.needsRecovery, true); assert.equal(match.difficulty, 'neural'); assert.equal(match.busy, false);
  fail = false; await match.retry(); assert.equal(match.failure, ''); assert.equal(match.needsRecovery, false); match.dispose();
});

test('a deal whose first Trained move fails is recorded all the same, though the press answers false', async () => {
  // The page lets a finished hand's notes go by this, what the engine did,
  // not by whether the new hand's opponents then answered (App, nextHand).
  class Dealing extends FakeGame {
    constructor(...args) { super(...args); this.ended = true; this.pending = false; }
    hand_is_over() { return this.ended; }
    next_hand() { this.ended = false; this.pending = this.external; }
  }
  let fail = true;
  const match = new MatchSession(Dealing, 1, 'neural', { ai: async () => { if (fail) throw new Error('offline'); return sure(0); } });
  try {
    assert.equal(await match.nextHand(), false);
    assert.deepEqual(match.commands, [{ type: 'next' }]);
    assert.equal(match.engine.ended, false, 'the next hand is dealt');
    assert.equal(match.needsRecovery, true);
    // Retried, the hand goes on from the deal, which is not dealt again.
    fail = false;
    assert.equal(await match.retry(), true, match.failure);
    assert.deepEqual(match.commands, [{ type: 'next' }, ...played(0)]);
  } finally { match.dispose(); }
});

test('continue with Club changes the engine in place and survives restoration', async () => {
  const match = new MatchSession(FakeGame, 1, 'neural', {ai: async () => {throw new Error('offline');}});
  await match.run(); const engine = match.engine; await match.continueWithClub();
  assert.equal(match.engine, engine); assert.equal(engine.external, false); assert.equal(match.difficulty, 'club');
  const restored = MatchSession.restore(FakeGame, JSON.stringify(match.snapshot()));
  assert.equal(restored.difficulty, 'club'); assert.equal(restored.stateKey(), match.stateKey()); match.dispose(); restored.dispose();
});

test('Club meanwhile drops the question a Trained opponent waits on, and the same run goes on with Club, recorded once', async () => {
  // An answer that does not come, as while the network downloads: the
  // question ends only as its signal does, the way the page's request does.
  const asked = deferred(), saves = [];
  const match = new MatchSession(FakeGame, 1, 'neural', { onSave: save => saves.push(save.commands),
    ai: (_planes, _mask, signal) => new Promise((_, reject) => {
      signal.addEventListener('abort', () => reject(new DOMException('Match changed', 'AbortError')));
      asked.resolve(signal);
    }) });
  try {
    const run = match.run(), signal = await asked.promise;
    assert.equal(match.thinking, true);
    assert.equal(match.clubMeanwhile(), true);
    assert.equal(match.clubMeanwhile(), false, 'asked for once');
    assert.equal(signal.aborted, true, 'the question is dropped');
    assert.equal(await run, true, match.failure);
    assert.deepEqual(match.commands, [{ type: 'club' }]);
    assert.deepEqual(saves, [[], [{ type: 'club' }]]);
    assert.deepEqual([match.difficulty, match.stoodIn, match.failure, match.busy, match.thinking], ['club', true, '', false, false]);
    assert.deepEqual([match.engine.external, match.engine.applied], [false, 0]);
    assert.equal(match.abort.signal.aborted, false, 'the match is not closed by it');
    // It restores as any match continued with Club does, asking nobody.
    const restored = MatchSession.restore(FakeGame, JSON.stringify(match.snapshot()), { ai() { throw new Error('Club needs no network'); } });
    try { assert.deepEqual([restored.difficulty, restored.stateKey()], ['club', match.stateKey()]); } finally { restored.dispose(); }
    // The player's own moves go on as against Club.
    assert.equal(await match.choose({ kind: 'discard', tile: '1m' }), true);
    assert.equal(match.clubMeanwhile(), false, 'nobody is awaited');
  } finally { match.dispose(); }
});

test('Club meanwhile lets go of an answer that came as it was asked for, and is not taken where nobody is awaited', async () => {
  // The network answers after all, its request deaf to the signal: the
  // answer is not played, and Club takes the seat.
  const answer = deferred();
  const match = new MatchSession(FakeGame, 1, 'neural', { ai: () => answer.promise });
  try {
    const run = match.run();
    assert.equal(match.clubMeanwhile(), true);
    answer.resolve(sure(0));
    assert.equal(await run, true, match.failure);
    assert.deepEqual(match.commands, [{ type: 'club' }]);
    assert.equal(match.engine.applied, 0);
  } finally { match.dispose(); }
  // Against Club, at the player's own turn, after a failure and once closed,
  // there is nothing to stand in for: recovery has its own command.
  const club = new MatchSession(FakeGame, 1, 'club');
  await club.run();
  assert.equal(club.clubMeanwhile(), false);
  club.dispose();
  const failed = new MatchSession(FakeGame, 1, 'neural', { ai: async () => { throw new Error('offline'); } });
  await failed.run();
  assert.deepEqual([failed.needsRecovery, failed.clubMeanwhile(), failed.stoodIn], [true, false, false]);
  await failed.continueWithClub();
  assert.deepEqual([failed.difficulty, failed.stoodIn], ['club', false], 'Club after a failure is no stand-in');
  failed.dispose();
  assert.equal(failed.clubMeanwhile(), false);
  // A match closed while it waits is closed, whatever was asked of it.
  const pending = deferred(), closing = new MatchSession(FakeGame, 1, 'neural', { ai: () => pending.promise });
  const running = closing.run();
  assert.equal(closing.clubMeanwhile(), true);
  closing.dispose();
  pending.reject(new DOMException('Match changed', 'AbortError'));
  assert.equal(await running, false);
  assert.deepEqual([closing.commands, closing.failure], [[], '']);
});

test('Club meanwhile is not taken once the last answer is in, while the run still holds the save lock', async () => {
  // The run's guard, as the page's save lock does, lets go only some time
  // after the run's last move is played; the match is still thinking then.
  const answered = deferred(), release = deferred();
  const match = new MatchSession(FakeGame, 1, 'neural', { ai: async () => sure(0),
    guard: async task => { const result = await task(); answered.resolve(); await release.promise; return result; } });
  try {
    const run = match.run();
    await answered.promise;
    assert.equal(match.thinking, true, 'the run has not ended');
    assert.equal(match.clubMeanwhile(), false, 'nobody is awaited any more, so the press is not taken');
    release.resolve();
    assert.equal(await run, true, match.failure);
    assert.deepEqual(match.commands, played(0));
    assert.deepEqual([match.stoodIn, match.engine.external, match.abort.signal.aborted], [false, true, false]);
  } finally { match.dispose(); }
});

test("a stand-in is Club where the player's own table says Trained, and nothing else", () => {
  const trained = ['neural', 'neural', 'neural'], club = ['club', 'club', 'club'];
  assert.deepEqual(standInFor(trained, club), trained);
  assert.notEqual(standInFor(trained, club), trained, 'a copy');
  // A mixed table: only its Trained seats are stood in for.
  assert.deepEqual(standInFor(['neural', 'club', 'beginner'], ['club', 'club', 'beginner']), ['neural', 'club', 'beginner']);
  assert.deepEqual(standInFor(trained, ['club', 'neural', 'club']), trained, 'a seat still Trained');
  // The same table is no stand-in, and any other difference is the match's
  // own table: Beginner for Trained, Trained for Club, Club for Beginner.
  for (const [preferred, playing] of [[trained, trained], [club, club], [trained, ['beginner', 'beginner', 'beginner']], [club, trained],
    [['beginner', 'neural', 'neural'], ['club', 'club', 'club']], [trained, ['club', 'club', 'beginner']], [trained, ['club', 'club']]]) {
    assert.equal(standInFor(preferred, playing), null, `${preferred} against ${playing}`);
  }
  // After a reload a stand-in is known by the table the player's record
  // itself holds, as every page of this build writes it. A record a reset
  // read as Trained holds none, nor does one from before tables were stored
  // seat by seat: a saved Club match is no stand-in there, and its table
  // becomes the preference, as it always did.
  const read = value => readSettings({ getItem: () => JSON.stringify(value) });
  assert.deepEqual(standInFor(read({ version: 6, difficulty: 'neural', opponents: trained }).opponents ?? null, club), trained);
  for (const record of [{ version: 1, difficulty: 'club', opponents: club }, { version: 2, difficulty: 'neural' }]) {
    const settings = read(record);
    assert.equal(settings.difficulty, 'neural', JSON.stringify(record));
    assert.equal(standInFor(settings.opponents ?? null, club), null, JSON.stringify(record));
  }
  assert.equal(standInFor(null, club), null);
});

test('a new game after a stand-in: the chosen table where the network is at hand or nothing is fetching it, Club only while its download is under way', () => {
  const chosen = ['neural', 'club', 'neural'];
  // What the page knows of the network as the new game is dealt, in each
  // case, and whether the stand-in holds (Club again) or ends.
  const rows = [
    ['saved on this device', { network: 'ready' }, false],
    ['storage refused, the bytes held for the worker', { network: 'ready' }, false],
    ['storage refused, no bytes held, nothing fetching', { network: 'none' }, false],
    ['storage refused, no bytes held, the worker fetching', { network: 'none', workerNote: 'downloading the network 40%' }, true],
    ['nothing can be saved, the worker loaded', { network: 'none', workerLoaded: true }, false],
    ['nothing can be saved, the worker loading', { network: 'none', workerNote: 'downloading the network 12%' }, true],
    ['nothing can be saved, the worker gone', { network: 'none' }, false],
    ['the page downloading', { network: 'downloading' }, true],
    ['waiting behind another window', { network: 'waiting' }, true],
    ['the worker waiting behind another window', { network: 'none', workerNote: 'waiting for another window to download the network' }, true],
    ['the download failed', { network: 'none' }, false],
    ['the worker loading a stored network', { network: 'none', workerNote: 'loading the network' }, false],
    ['downloading, and the worker loaded already', { network: 'downloading', workerLoaded: true }, false],
  ];
  for (const [name, known, holds] of rows) {
    const where = { atHand: trainedAtHand(known), underWay: trainedUnderWay(known) };
    const table = tableAfterStandIn(chosen, where);
    assert.deepEqual(table, holds ? ['club', 'club', 'club'] : chosen, name);
    assert.deepEqual(standInFor(chosen, table), holds ? chosen : null, `${name}: the stand-in ${holds ? 'holds' : 'ends'}`);
  }
  // Only those two say the network is at hand: not offline's aiReady, nor a
  // worker that is merely not loading.
  assert.equal(trainedAtHand({ network: 'none', aiReady: true }), false);
  assert.equal(trainedAtHand({}), false);
  assert.equal(trainedUnderWay({}), false);
  // Seats that were never Trained keep their type either way.
  assert.deepEqual(tableAfterStandIn(['neural', 'beginner', 'neural'], { atHand: false, underWay: true }), ['club', 'beginner', 'club']);
});

test('busy sessions ignore additional player choices', async () => {
  const answer = deferred(); const match = new MatchSession(FakeGame, 1, 'neural', {ai: () => answer.promise});
  const run = match.run(); assert.equal(await match.choose({kind:'discard',tile:'1m'}), false);
  assert.equal(match.commands.length, 0); answer.resolve(sure(0)); await run; match.dispose();
});

test('illegal neural output is rejected rather than used as a fallback move', async () => {
  const match = new MatchSession(FakeGame, 1, 'neural', {ai: async () => sure(1)});
  await match.run(); assert.match(match.failure, /Invalid opponent/); assert.equal(match.engine.applied, 0); match.dispose();
});

test('a declared reach is answered twice and replays without asking again', async () => {
  const match = new MatchSession(FakeGame, 1, 'neural', {ai: async (_planes, mask) => sure(mask[37] ? 37 : 0)});
  assert.equal(await match.run(), true);
  assert.deepEqual(match.commands, [{type:'opponent',action:37},{type:'opponent',action:0}]);
  assert.equal(match.engine.declared, 1); assert.equal(match.engine.applied, 1); assert.equal(match.failure, '');
  const restored = MatchSession.restore(FakeGame, JSON.stringify(match.snapshot()),
    {ai(){throw new Error('A saved reach must not be put to the network again');}});
  assert.equal(restored.stateKey(), match.stateKey());
  assert.equal(restored.engine.declared, 1); assert.equal(restored.engine.applied, 1);
  match.dispose(); restored.dispose();
});

test('a Trained opponent that hesitates is asked the renamed game too, with its moves renamed, and plays the mean', async () => {
  // Move 0 (1m) at 0.6 and the reach at 0.4: hesitant. The renamed game,
  // where 1m is called 1p (move 9), puts 0.2 on it and 0.8 on the reach, and
  // the mean declares. The tile the reach then throws is its only move.
  const asked = [], plain = { action: 0, weights: weightsOf({ 0: 0.6, 37: 0.4 }) }, renamed = { action: 37, weights: weightsOf({ 9: 0.2, 37: 0.8 }) };
  const match = new MatchSession(FakeGame, 1, 'neural', { ai: async (planes, mask, signal) => {
    asked.push({ planes: Array.from(planes), mask: Array.from(mask), signal });
    return [plain, renamed, sure(0)][asked.length - 1];
  } });
  try {
    assert.equal(await match.run(), true, match.failure);
    const reachMask = Array.from(match.engine.opponent_mask_mortal(), (_, move) => Number([0, 37].includes(move)));
    assert.deepEqual(asked.map(({ planes, mask }) => ({ planes, mask })), [
      { planes: [0], mask: reachMask }, { planes: [10], mask: swapMoves(reachMask) }, { planes: [1], mask: Array.from(weightsOf({ 0: 1 })) }]);
    assert.ok(asked.every(({ signal }) => signal === match.abort.signal));
    assert.equal(average(plain.weights, renamed.weights, reachMask).action, 37);
    assert.deepEqual(match.commands, [{ type: 'opponent', action: 37 }, { type: 'opponent', action: 0 }]);
  } finally { match.dispose(); }
});

test('a sure Trained opponent is asked once, and one closed while it was asked reads no renamed game from a freed engine', async () => {
  const asked = [];
  const sureMatch = new MatchSession(FakeGame, 1, 'neural', { ai: async (planes, mask) => {
    asked.push(Array.from(planes));
    return { action: 0, weights: Array.from(mask, (open, move) => (!open ? 0 : move === 0 ? 0.95 : 0.05)) };
  } });
  try {
    assert.equal(await sureMatch.run(), true, sureMatch.failure);
    assert.deepEqual(asked, [[0]]);
    assert.deepEqual(sureMatch.commands, [{ type: 'opponent', action: 0 }]);
  } finally { sureMatch.dispose(); }
  // Answered hesitantly as the match closes: its engine is freed, and the
  // renamed planes are never read from it.
  const answer = deferred(), closing = new MatchSession(FakeGame, 1, 'neural', { ai: () => answer.promise });
  const run = closing.run(), engine = closing.engine;
  closing.dispose();
  answer.resolve({ action: 0, weights: weightsOf({ 0: 0.5, 37: 0.5 }) });
  assert.equal(await run, false);
  assert.deepEqual([engine.applied, engine.renamedReads ?? 0], [0, 0]);
});

test('a Trained opponent\'s riichi whose tile keeps it first is thrown undeclared: one discard, saved once, restored without asking', async () => {
  // Riichi is its best move, and the reach's one tile keeps it first.
  const saves = [], declare = async (_planes, mask) => sure(mask[37] ? 37 : 0);
  const match = new MatchSession(FakeGame, 1, 'neural', { ai: declare, onSave: save => saves.push(save.commands) });
  match.engine.dama = { 0: DAMA.KEEPS_FIRST };
  try {
    assert.equal(await match.run(), true, match.failure);
    assert.deepEqual(match.commands, played(0));
    assert.deepEqual([match.engine.declared, match.engine.applied], [0, 1], 'nothing is declared');
    assert.deepEqual(saves, [[], played(0)], 'saved as the run begins, then once for the move');
    const restored = MatchSession.restore(FakeGame, JSON.stringify(match.snapshot()),
      { ai() { throw new Error('A saved move must not be put to the network again'); } });
    try {
      assert.equal(restored.stateKey(), match.stateKey());
      assert.deepEqual([restored.engine.declared, restored.engine.applied], [0, 1]);
      assert.deepEqual(restored.engine.reads, { reachRenamed: 0, dama: 0, decline: 0 }, 'nor are the rules asked');
    } finally { restored.dispose(); }
  } finally { match.dispose(); }
  // Where the codes are nought the riichi stands, and it and its tile are
  // saved together.
  saves.length = 0;
  const reach = new MatchSession(FakeGame, 1, 'neural', { ai: declare, onSave: save => saves.push(save.commands) });
  try {
    assert.equal(await reach.run(), true, reach.failure);
    assert.deepEqual(saves, [[], played(37, 0)]);
  } finally { reach.dispose(); }
});

test('a Trained opponent passes a ron that would leave it last, though a pon ranks second', async () => {
  const match = new MatchSession(FakeGame, 1, 'neural', { ai: async () => ({ action: 43, weights: weightsOf({ 43: 0.95, 41: 0.04, 45: 0.01 }) }) });
  Object.assign(match.engine, { open: [41, 43, 45], decline: DECLINE.LAST });
  try {
    assert.equal(await match.run(), true, match.failure);
    assert.deepEqual(match.commands, played(45));
    assert.deepEqual(match.engine.reads, { reachRenamed: 0, dama: 0, decline: 1 });
  } finally { match.dispose(); }
});

test('a Trained opponent declines a self-draw that would leave it last for its best other move, a discard or a riichi and its tile', async () => {
  for (const [weights, moves] of [[{ 43: 0.95, 0: 0.04, 37: 0.01 }, [0]], [{ 43: 0.95, 37: 0.04, 0: 0.01 }, [37, 0]]]) {
    const match = new MatchSession(FakeGame, 1, 'neural', { ai: async (_planes, mask) => (mask[43] ? { action: 43, weights: weightsOf(weights) } : sure(0)) });
    Object.assign(match.engine, { open: [0, 37, 43], decline: DECLINE.LAST });
    try {
      assert.equal(await match.run(), true, match.failure);
      assert.deepEqual(match.commands, played(...moves));
    } finally { match.dispose(); }
  }
});

test('codes no legal move can have refuse the opponent\'s move: nothing is recorded, and it waits for Retry or Club', async () => {
  const match = new MatchSession(FakeGame, 1, 'neural', { ai: async (_planes, mask) => sure(mask[37] ? 37 : 0) });
  // The reach's one tile coded as one whose riichi is not legal.
  match.engine.dama = { 0: DAMA.NOT_WAITING };
  try {
    assert.equal(await match.run(), false);
    assert.equal(match.failure, RULE_REFUSED);
    assert.deepEqual(match.commands, []);
    assert.equal(match.needsRecovery, true);
  } finally { match.dispose(); }
});

test('a save stopped between a riichi and its tile restores, and asks only the tile, of the pending reach, reading no code', async () => {
  // As the page saved before the rules, after every command: the riichi
  // recorded, its tile not yet.
  const half = new MatchSession(FakeGame, 1, 'neural');
  half.apply({ type: 'opponent', action: 37 });
  half.advance(false);
  const saved = JSON.stringify(half.snapshot());
  half.dispose();
  const asked = [];
  const restored = MatchSession.restore(FakeGame, saved, { ai: async (planes, mask) => {
    asked.push({ planes: Array.from(planes), mask: Array.from(mask) });
    return sure(0);
  } });
  try {
    // The codes would throw the tile undeclared, were they asked.
    restored.engine.dama = { 0: DAMA.KEEPS_FIRST };
    assert.equal(await restored.run(), true, restored.failure);
    assert.deepEqual(asked, [{ planes: [1], mask: Array.from(weightsOf({ 0: 1 })) }]);
    assert.deepEqual(restored.engine.reads, { reachRenamed: 0, dama: 0, decline: 0 });
    assert.deepEqual(restored.commands, played(37, 0));
    assert.deepEqual([restored.engine.declared, restored.engine.applied], [1, 1]);
  } finally { restored.dispose(); }
});

test('a riichi whose tile fails once recorded is left alone: Retry saves it, then asks the tile of the pending reach', async () => {
  /** An engine that refuses the first tile played. */
  class Refusing extends FakeGame {
    play_opponent_mortal(action) {
      if (action !== 37 && !this.refused) { this.refused = true; throw new Error('The tile was refused'); }
      return super.play_opponent_mortal(action);
    }
  }
  const saves = [], asked = [];
  const match = new MatchSession(Refusing, 1, 'neural', { onSave: save => saves.push(save.commands), ai: async (planes, mask) => {
    asked.push(Array.from(planes));
    return sure(mask[37] ? 37 : 0);
  } });
  try {
    assert.equal(await match.run(), false);
    assert.equal(match.failure, 'The tile was refused');
    assert.deepEqual(match.commands, played(37));
    assert.deepEqual(saves, [[]], 'nothing is saved after the riichi');
    assert.equal(match.needsRecovery, true);
    assert.equal(await match.retry(), true, match.failure);
    assert.deepEqual(saves, [[], played(37), played(37, 0)]);
    assert.deepEqual(asked, [[0], [1], [1]], 'the riichi, its tile previewed, then its tile of the pending reach');
    assert.deepEqual(match.engine.reads, { reachRenamed: 0, dama: 1, decline: 0 }, 'the codes are read for the first answer only');
  } finally { match.dispose(); }
});

test('a match closed while its opponent\'s riichi tile is asked reads nothing more of its engine', async () => {
  const tile = deferred(), tileAsked = deferred();
  const match = new MatchSession(FakeGame, 1, 'neural', { ai: async (_planes, mask) => {
    if (mask[37]) return sure(37);
    tileAsked.resolve();
    return tile.promise;
  } });
  const run = match.run(), engine = match.engine;
  await tileAsked.promise;
  match.dispose();
  // Hesitant: asked on, it would read the renamed preview, then the codes.
  tile.resolve({ action: 0, weights: weightsOf({ 0: 0.5 }) });
  assert.equal(await run, false);
  assert.equal(match.failure, '');
  assert.deepEqual(engine.reads, { reachRenamed: 0, dama: 0, decline: 0 });
  assert.deepEqual([engine.declared, engine.applied], [0, 0]);
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
  for (const version of [0, 7, '6', undefined]) {
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
  for (const settings of [readSettings(null), readSettings({ getItem: () => null }), read({ version: 6, difficulty: 'club' })]) {
    assert.equal(Object.hasOwn(settings, 'trainedNetwork'), false);
    assert.equal(playing(settings), MANIFEST.version);
  }
  // Every listed network is read back as it was picked.
  assert.ok(OLDER.length >= 2, 'the networks played before the default stay listed');
  for (const version of NETWORKS.map(network => network.version)) {
    assert.equal(read({ version: 6, trainedNetwork: version }).trainedNetwork, version);
  }
  // Anything that is not a listed version is no pick: the default plays.
  for (const trainedNetwork of ['0.1.36', '0.1.4100', 'strong', 'full', '', 410, null, ['0.1.410'], { version: '0.1.410' }]) {
    const settings = read({ version: 6, trainedNetwork, hints: false });
    assert.equal(Object.hasOwn(settings, 'trainedNetwork'), false, JSON.stringify(trainedNetwork));
    assert.equal(playing(settings), MANIFEST.version);
    assert.equal(settings.hints, false, 'the rest of the record stands');
  }
  // A record from before 810 holds no pick; it would not be read if it did.
  assert.equal(Object.hasOwn(read({ version: 1, trainedNetwork: '0.1.410' }), 'trainedNetwork'), false);
  // Round trip: written with the other preferences, in the version 6 record.
  const writes = new Map();
  const storage = { setItem: (key, value) => writes.set(key, value), getItem: key => writes.get(key) ?? null };
  const preferences = { difficulty: 'neural', opponents: ['neural', 'neural', 'neural'], hints: true, confirmDiscards: false,
    shortcuts: true, tileFace: 'classic', reviewAdviser: 'strong', reviewHideMove: false, reviewHideTiles: false,
    reviewFilter: 'flagged' };
  writeSettings(storage, { ...preferences, trainedNetwork: '0.1.410' });
  assert.deepEqual(JSON.parse(writes.get(SETTINGS_KEY)), { version: 6, ...preferences, trainedNetwork: '0.1.410' });
  assert.deepEqual(readSettings(storage), { ...preferences, trainedNetwork: '0.1.410' });
  // A player who has picked nothing writes no pick, so a newer default reaches them.
  writeSettings(storage, { ...preferences, trainedNetwork: undefined });
  assert.equal(Object.hasOwn(JSON.parse(writes.get(SETTINGS_KEY)), 'trainedNetwork'), false);
  assert.equal(playing(readSettings(storage)), MANIFEST.version);
});

test('a record from before 1110, 1410, 2610 or 3000 plays the default once, whatever it picked, and keeps every other choice', () => {
  // Every other choice a record holds, none of them a default.
  const chosen = { difficulty: 'custom', opponents: ['beginner', 'club', 'neural'], hints: false, confirmDiscards: true,
    shortcuts: false, tileFace: 'van-gogh', reviewAdviser: 'club', reviewHideMove: true, reviewHideTiles: true, reviewFilter: 'every' };
  // What builds before 1110 wrote: 810 was the default then and was never
  // written as a pick, but a record that holds it gives it up all the same.
  // Builds before 1410 wrote version 3, with 1110 the default, builds before
  // 2610 version 4, with 1410 the default, and builds before 3000 version 5,
  // with 2610 the default; a record of any of them gives up its pick the same
  // way.
  const before = [2, 3, 4, 5].flatMap(record => [...OLDER, undefined, '0.1.36'].map(trainedNetwork => ({ record, trainedNetwork })));
  for (const { record, trainedNetwork } of before) {
    const writes = new Map([[SETTINGS_KEY, JSON.stringify({ version: record, ...chosen, trainedNetwork })]]);
    const storage = { setItem: (key, value) => writes.set(key, value), getItem: key => writes.get(key) ?? null };
    const settings = readSettings(storage);
    assert.deepEqual(settings, chosen, `version ${record}, ${trainedNetwork}`);
    assert.equal(playing(settings), MANIFEST.version);
    // The page writes the record back as it loads (App's effect), as version 6
    // and without a pick, so the next load reads the same.
    writeSettings(storage, settings);
    assert.deepEqual(JSON.parse(writes.get(SETTINGS_KEY)), { version: 6, ...chosen });
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
  // A match's own save is unchanged: version 1, as before 810, 1110, 1410,
  // 2610 and 3000.
  const match = new MatchSession(FakeGame, 1, 'neural', { ai: async (_planes, mask) => sure(mask[37] ? 37 : 0) });
  await match.run();
  await match.choose({ kind: 'discard', tile: '1m' });
  const saved = match.snapshot();
  assert.equal(saved.version, 1);
  assert.deepEqual(saved.opponents, ['neural', 'neural', 'neural']);
  // Restored as a page would after the reload, its recorded moves are replayed
  // without the network, and the move still to come is asked of the network
  // the page now plays.
  let asked = 0;
  const restored = MatchSession.restore(FakeGame, JSON.stringify(saved), { ai: async (_planes, mask) => { asked++; return sure(mask[37] ? 37 : 0); } });
  try {
    assert.equal(restored.stateKey(), match.stateKey());
    assert.deepEqual(restored.commands, saved.commands);
    assert.equal(asked, 0);
    assert.equal(await restored.choose({ kind: 'discard', tile: '1m' }), true);
    assert.equal(asked, 2, 'a reach and its discard');
  } finally { match.dispose(); restored.dispose(); }
});

test('the table shows the network download while a Trained opponent waits for it, and nothing once the network is in hand', () => {
  // The worker's own note wins; before the worker starts, the page's offline
  // download of the network is the wait, and nothing is said when the
  // network is not downloading.
  assert.equal(waitingNote('loading the network', { phase: 'ai', network: 'downloading', progress: 40 }), 'loading the network');
  // The download line says how much the download is, once: the size of the
  // network the page plays as it is fetched and kept, in whole megabytes.
  const size = `${Math.round((MANIFEST.storedBytes ?? MANIFEST.bytes) / 1e6)} MB`;
  assert.equal(waitingNote('', { phase: 'ai', network: 'downloading', progress: 40 }), `downloading the trained network (${size}) 40%`);
  assert.equal(waitingNote('', { phase: 'ai', network: 'downloading' }), `downloading the trained network (${size}) 0%`);
  // Of the network given, and of its own bytes where its listing says no
  // more.
  for (const entry of [{ bytes: 116935691, storedBytes: 107878377 }, { bytes: 116935691 }]) {
    const megabytes = Math.round((entry.storedBytes ?? entry.bytes) / 1e6);
    assert.equal(waitingNote('', { network: 'downloading', progress: 7 }, entry), `downloading the trained network (${megabytes} MB) 7%`);
  }
  // Behind another window's download of it.
  assert.equal(waitingNote('', { phase: 'ai', network: 'waiting', progress: 0 }), 'waiting for another window to download the trained network');
  assert.equal(waitingNote('starting the network', { phase: 'ai', network: 'waiting' }), 'starting the network');
  // The network in hand, the save still waiting for its runtime at 99: the
  // opponent is playing, and the table says nothing of a download.
  assert.equal(waitingNote('', { phase: 'ai', network: 'ready', progress: 99 }), '');
  // Nor while a first visit's install saves the game beside it, or after the
  // network failed, or where none was asked for.
  assert.equal(waitingNote('', { phase: 'ai', network: 'ready', progress: 99, installing: true }), '');
  assert.equal(waitingNote('', { phase: 'checking', network: 'none', progress: 0, installing: true }), '');
  for (const phase of ['checking', 'ready', 'incomplete', 'unavailable']) assert.equal(waitingNote('', { phase, network: 'none', progress: 40 }), '');
  assert.equal(waitingNote('', { phase: 'ai', progress: 40 }), '', 'a state that does not say where the network stands says no download');
  assert.equal(waitingNote('', null), '');
});

test("a waiting note is parted from the share of a download it ends with, which the prompt shows outside its live region", () => {
  assert.deepEqual(partedNote('downloading the trained network (108 MB) 37%'), { words: 'downloading the trained network (108 MB)', figure: '37%' });
  assert.deepEqual(partedNote('downloading the network 100%'), { words: 'downloading the network', figure: '100%' });
  // A note without one is all words, the megabytes in it no figure.
  for (const note of ['saving the game, then downloading the trained network (108 MB)', 'starting the network', 'network ready', '37% done']) {
    assert.deepEqual(partedNote(note), { words: note, figure: '' }, note);
  }
  for (const none of ['', null, undefined]) assert.deepEqual(partedNote(none), { words: '', figure: '' });
});

test('a wait is a download while the page saves the network or its worker fetches it, and not while the network starts or thinks', () => {
  // By the page: where the network itself stands, downloading or waiting its
  // turn behind another window.
  assert.equal(waitIsDownload('', { phase: 'ai', network: 'downloading', progress: 3 }), true);
  assert.equal(waitIsDownload('', { phase: 'ai', network: 'waiting', progress: 0 }), true);
  // Not where it is in hand, or none is asked for, has failed, or is left to
  // the worker: whatever the save's phase, and never by an install.
  for (const network of ['ready', 'none', undefined]) {
    for (const phase of ['checking', 'ready', 'ai', 'incomplete', 'unavailable']) {
      assert.equal(waitIsDownload('', { phase, network, progress: 40 }), false, `${phase}, ${network}`);
      assert.equal(waitIsDownload('', { phase, network, progress: 40, installing: true }), false, `${phase}, ${network}, installing`);
    }
  }
  assert.equal(waitIsDownload('', null), false);
  // By the opponent's worker, whose note then stands for the page's.
  assert.equal(waitIsDownload('downloading the network 12%', { phase: 'unavailable', network: 'none' }), true);
  assert.equal(waitIsDownload('waiting for another window to download the network', { network: 'none' }), true);
  for (const note of ['loading the network', 'starting the network', 'network ready', 'retrying the agent with more memory']) {
    assert.equal(waitIsDownload(note, { phase: 'ai', network: 'downloading', progress: 50 }), false, note);
  }
});

test('ordinary thinking, with the network in hand and the save still at 99, shows no wait and offers no Club meanwhile', () => {
  // The opponents play while the runtime is still being saved, on a first
  // visit beside the install: that is thinking, not a download.
  const thinking = { phase: 'ai', network: 'ready', progress: 99, installing: true };
  assert.equal(waitingNote('', thinking), '');
  assert.equal(waitIsDownload('', thinking), false);
  assert.equal(waitingNote('loading the network', thinking), 'loading the network');
  assert.equal(waitIsDownload('loading the network', thinking), false);
});

test('the start-up line says which of the two it still waits for: the tile graphics, or the engine once they are in', () => {
  assert.equal(startupWords(null, false, false), 'Loading the game and selected tile graphics…');
  assert.equal(startupWords({ done: 12, total: 37 }, false, false), 'Loading selected tile graphics… 12/37');
  // The engine in first changes nothing: the graphics are what is waited for.
  assert.equal(startupWords({ done: 12, total: 37 }, false, true), 'Loading selected tile graphics… 12/37');
  assert.equal(startupWords(null, false, true), 'Loading the game and selected tile graphics…');
  // The last graphic counted is not yet all of them in; once they are, and
  // the engine is not, the line no longer reads 37/37.
  assert.equal(startupWords({ done: 37, total: 37 }, false, false), 'Loading selected tile graphics… 37/37');
  assert.equal(startupWords({ done: 37, total: 37 }, true, false), 'Loading the game engine…');
  assert.equal(startupWords(null, true, false), 'Loading the game engine…', 'a face with nothing to fetch');
});

test('preferences are written when they change, not each time the effect reruns', () => {
  const writes = [];
  let fail = false;
  const storage = { setItem(key, value) { if (fail) throw new Error('denied'); writes.push([key, value]); } };
  const preferences = { difficulty: 'club', opponents: ['club', 'club', 'club'], hints: true };
  let last = writeSettings(storage, preferences);
  // App's effect reruns with an equal copy after every move of a match.
  for (let move = 0; move < 5; move++) last = writeSettings(storage, { ...preferences, opponents: [...preferences.opponents] }, last);
  assert.deepEqual(writes, [[SETTINGS_KEY, JSON.stringify({ version: 6, ...preferences })]]);
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

test('a change writes only what this window changed, over the record as another window left it', () => {
  const writes = new Map();
  const storage = { setItem: (key, value) => writes.set(key, value), getItem: key => writes.get(key) ?? null };
  const stored = () => JSON.parse(writes.get(SETTINGS_KEY));
  const start = { difficulty: 'custom', opponents: ['club', 'club', 'beginner'], hints: true, confirmDiscards: false, shortcuts: true,
    tileFace: 'classic', reviewAdviser: 'strong', reviewHideMove: false, reviewHideTiles: false, reviewFilter: 'flagged' };
  // Two windows load the same record, and each writes it back whole as it loads.
  let a = writeSettings(storage, start), b = writeSettings(storage, start);
  assert.deepEqual(stored(), { version: 6, ...start });
  // One picks another network; the other, still open, then turns the hints
  // off. The pick is still there for the first window's reload to read.
  a = writeSettings(storage, { ...start, trainedNetwork: OLDER[0] }, a);
  b = writeSettings(storage, { ...start, hints: false }, b);
  assert.deepEqual(stored(), { version: 6, ...start, hints: false, trainedNetwork: OLDER[0] });
  assert.equal(readSettings(storage).trainedNetwork, OLDER[0]);
  // What a window did not change it leaves as it finds it, though its own
  // copy is the older: the first window knows nothing of the hints.
  a = writeSettings(storage, { ...start, trainedNetwork: OLDER[0], reviewFilter: 'every' }, a);
  assert.deepEqual(stored(), { version: 6, ...start, hints: false, trainedNetwork: OLDER[0], reviewFilter: 'every' });
  // A pick set back to the default is taken out of the record, alone.
  a = writeSettings(storage, { ...start, reviewFilter: 'every' }, a);
  assert.deepEqual(stored(), { version: 6, ...start, hints: false, reviewFilter: 'every' });
  // The table is one choice stored as two. One window makes it Trained; the
  // other then changes a seat of the table it still shows, whose name does
  // not change: both are written, so the record never names one table and
  // seats another.
  writeSettings(storage, { ...start, hints: false, difficulty: 'neural', opponents: ['neural', 'neural', 'neural'] }, b);
  assert.deepEqual([stored().difficulty, stored().opponents], ['neural', ['neural', 'neural', 'neural']]);
  a = writeSettings(storage, { ...start, reviewFilter: 'every', opponents: ['club', 'beginner', 'beginner'] }, a);
  assert.deepEqual(stored(), { version: 6, ...start, hints: false, reviewFilter: 'every', difficulty: 'custom', opponents: ['club', 'beginner', 'beginner'] });
  // Nothing changed, nothing written: another window's record stays whole.
  const other = JSON.stringify({ ...stored(), tileFace: 'matisse' });
  writes.set(SETTINGS_KEY, other);
  assert.equal(writeSettings(storage, { ...start, reviewFilter: 'every', opponents: ['club', 'beginner', 'beginner'] }, a), a);
  assert.equal(writes.get(SETTINGS_KEY), other);
});

test('over a record this build did not write, a change writes the whole of the window\'s preferences', () => {
  const start = { difficulty: 'club', opponents: ['club', 'club', 'club'], hints: true, shortcuts: true };
  // A record of another version, one that is no record, and none at all.
  for (const held of [JSON.stringify({ version: 5, hints: false, trainedNetwork: OLDER[0] }), JSON.stringify({ version: 7, hints: false }),
    '{broken', '[]', 'null', null]) {
    const writes = new Map();
    const storage = { setItem: (key, value) => writes.set(key, value), getItem: key => writes.get(key) ?? null };
    const last = writeSettings(storage, start);
    if (held === null) writes.delete(SETTINGS_KEY); else writes.set(SETTINGS_KEY, held);
    writeSettings(storage, { ...start, shortcuts: false }, last);
    assert.deepEqual(JSON.parse(writes.get(SETTINGS_KEY)), { version: 6, ...start, shortcuts: false }, String(held));
  }
  // Storage that cannot be read back, or refuses to be, is written whole too.
  for (const getItem of [undefined, () => { throw new Error('denied'); }]) {
    const writes = [];
    const storage = { setItem: (_key, value) => writes.push(value), getItem };
    const last = writeSettings(storage, start);
    writeSettings(storage, { ...start, hints: false }, last);
    assert.deepEqual(JSON.parse(writes.at(-1)), { version: 6, ...start, hints: false });
  }
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
