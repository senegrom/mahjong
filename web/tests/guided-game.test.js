import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { PhysicalAnalysis, settle_physical } from '../src/wasm/riichi.js';
import { parseTiles, PHYSICAL_KEY } from '../src/lib/physical-position.js';
import { emptyGuided, editGuided, guidedEvent, undoGuided, parseGuided, GUIDED_FORMAT, GUIDED_KEY, doraTiles } from '../src/lib/guided-game.js';
import { PhysicalStore } from '../src/lib/physical-store.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const inspect = (game, read = e => e.agent_choices()) => {
  const engine = new PhysicalAnalysis(game.state.position);
  try { return read(engine); } finally { engine.free(); }
};
const validate = position => { const engine = new PhysicalAnalysis(position); engine.free(); };
const act = (g, e) => guidedEvent(g, e, validate, settle_physical);
function start(seat = 3, tiles = '123m456p789s1123z', indicator = '7z') {
  let g = emptyGuided();
  g = editGuided(g, s => { s.position.seat = seat; });
  g = act(g, { type: 'setup' });
  g = editGuided(g, s => { s.position.players[seat].hand = parseTiles(tiles); });
  g = act(g, { type: 'hand' });
  return act(g, { type: 'indicator', tile: indicator });
}
function choose(g, kind, tile) {
  const choices = inspect(g), choice = choices.find(c => c.kind === kind && (!tile || c.tile === tile));
  assert.ok(choice, `Expected legal ${kind} ${tile ?? ''}`);
  return act(g, { type: 'choice', choices, choice });
}
const pass = g => choose(g, 'pass');
const next = g => act(g, { type: 'continue' });
const discard = (g, tile, options = {}) => act(g, { type: 'discard', tile, ...options });

test('North guide asks East, South, West, then the real draw and remembers the chosen discard', () => {
  let g = start();
  assert.equal(g.state.position.wall, 70);
  for (const [seat, tile] of ['9m', '8m', '7m'].entries()) {
    assert.equal(g.state.stage, 'turn'); assert.equal(g.state.nextSeat, seat);
    g = discard(g, tile);
    assert.equal(g.state.stage, 'decision');
    assert.equal(g.state.position.wall, 69 - seat);
    assert.equal(g.state.position.players[seat].hand.length, 0);
    g = next(pass(g));
  }
  assert.equal(g.state.nextSeat, 3);
  const before = structuredClone(g);
  g = act(g, { type: 'draw', tile: '4z' });
  assert.equal(g.state.position.wall, 66);
  assert.equal(g.state.position.players[3].hand.length, 14);
  assert.deepEqual(undoGuided(g), before);
  g = choose(g, 'discard', '4z');
  assert.equal(g.state.position.players[3].hand.length, 13);
  assert.equal(g.state.position.players[3].discards[0].order, 3);
  assert.equal(g.state.position.players[3].discards[0].drawn, true);
  assert.equal(next(g).state.nextSeat, 0);
  const text = GUIDED_FORMAT.encode(g);
  assert.deepEqual(parseGuided(text), g);
  assert.deepEqual(undoGuided(parseGuided(text)), undoGuided(g));
});

test('dealer starts with 13 tiles and explicitly enters the extra tile before any decision', () => {
  let g = start(0);
  assert.equal(g.state.nextSeat, 0);
  assert.equal(g.state.position.players[0].hand.length, 13);
  assert.throws(() => discard(g, '9m'), /legal moves/);
  g = act(g, { type: 'draw', tile: '4z' });
  assert.equal(g.state.position.wall, 69);
  assert.ok(inspect(g).some(c => c.kind === 'discard'));
});

test('my pon consumes held tiles and goes straight to legal discards without a draw', () => {
  let g = start(3, '123m456p789s1155z');
  g = discard(g, '5z');
  const before = g;
  g = choose(g, 'pon');
  assert.equal(g.state.stage, 'decision');
  assert.equal(g.state.position.players[3].hand.length, 11);
  assert.equal(g.state.position.players[0].discards[0].claimed, true);
  assert.equal(g.state.position.players[3].melds[0].from, 1);
  assert.equal(g.state.position.wall, 69);
  g = choose(g, 'discard', '1z');
  assert.equal(g.state.position.wall, 69);
  assert.equal(next(g).state.nextSeat, 0);
  assert.deepEqual(undoGuided(undoGuided(g)), before);
});

test('opponent pon skips seats and their next discard does not consume another wall tile', () => {
  let g = pass(discard(start(), '5z'));
  g = act(g, { type: 'call', seat: 2, kind: 'pon' });
  assert.equal(g.state.nextSeat, 2); assert.equal(g.state.needsDraw, false);
  assert.equal(g.state.position.players[0].discards[0].claimed, true);
  assert.equal(g.state.position.players[2].melds[0].from, 2);
  assert.throws(() => act(g, { type: 'kan', kind: 'concealed-kan', tile: '1p' }), /after a draw/);
  assert.throws(() => discard(g, '6z', { riichi: true }), /closed hand/);
  g = discard(g, '6z');
  assert.equal(g.state.position.wall, 69);
  assert.equal(next(pass(g)).state.nextSeat, 3);
});

test('opponent chii enforces the left-hand source, sequence and available copies', () => {
  const g = pass(discard(start(), '3m'));
  assert.throws(() => act(g, { type: 'call', seat: 2, kind: 'chii', tile: '1m' }), /from the left/);
  assert.throws(() => act(g, { type: 'call', seat: 1, kind: 'chii', tile: '4m' }), /sequence/);
  const called = act(g, { type: 'call', seat: 1, kind: 'chii', tile: '1m' });
  assert.equal(called.state.nextSeat, 1);
  assert.equal(discard(called, '8p').state.position.wall, 69);
  assert.equal(g.state.position.players[1].melds.length, 0, 'failed entries do not mutate their source');
});

test('opponent open kan prompts for dora and counts the replacement once', () => {
  let g = pass(discard(start(), '5z'));
  g = act(g, { type: 'call', seat: 2, kind: 'kan' });
  assert.equal(g.state.stage, 'indicator'); assert.equal(g.state.position.wall, 69);
  g = act(g, { type: 'indicator', tile: '6z' });
  assert.equal(g.state.nextSeat, 2); assert.equal(g.state.position.after_quad, true);
  g = discard(g, '8p');
  assert.equal(g.state.position.wall, 68);
  assert.equal(g.state.position.indicators.length, 2);
  assert.ok(inspect(g).some(c => c.kind === 'pass'));
});

test('opponent concealed and added kans have a robbery decision before the indicator', () => {
  let g = start();
  g = act(g, { type: 'kan', kind: 'concealed-kan', tile: '1p' });
  assert.equal(g.state.stage, 'decision'); assert.equal(g.state.position.wall, 69);
  assert.equal(g.state.position.pending_kind, 'concealed-kan');
  g = next(pass(g));
  assert.equal(g.state.stage, 'indicator');
  g = act(g, { type: 'indicator', tile: '6z' });
  g = discard(g, '8p');
  assert.equal(g.state.position.wall, 68);
  let added = pass(discard(start(), '5z'));
  added = act(added, { type: 'call', seat: 2, kind: 'pon' });
  added = next(pass(discard(added, '6z')));
  added = choose(act(added, { type: 'draw', tile: '4z' }), 'discard', '4z');
  added = next(added);
  for (const tile of ['9m', '8m']) added = next(pass(discard(added, tile)));
  added = act(added, { type: 'kan', kind: 'extended-kan', tile: '5z' });
  assert.equal(added.state.position.pending_kind, 'extended-kan');
  assert.equal(added.state.position.players[2].melds.length, 1);
  assert.equal(added.state.position.players[2].melds[0].kind, 'extended-kan');
  assert.equal(next(pass(added)).state.stage, 'indicator');
});

test('my concealed kan records the new indicator and real replacement before the next advice', () => {
  let g = start(0, '111m456p789s1123z');
  g = act(g, { type: 'draw', tile: '1m' });
  g = choose(g, 'concealed-kan', '1m');
  assert.equal(g.state.stage, 'kan-response');
  g = next(g); g = act(g, { type: 'indicator', tile: '6z' });
  g = act(g, { type: 'draw', tile: '4z' });
  assert.equal(g.state.position.wall, 68);
  assert.equal(g.state.position.after_quad, true);
  assert.equal(g.state.position.players[0].hand.length, 11);
  assert.ok(inspect(g).some(c => c.kind === 'discard'));
});

test('passing ron keeps furiten until my real draw and riichi pays exactly one stick', () => {
  let g = discard(start(3, '123m456p789s1112z'), '2z', { riichi: true });
  assert.equal(g.state.position.players[0].score, 29000);
  assert.equal(g.state.position.players[0].riichi, 'double');
  assert.equal(g.state.position.riichi_sticks, 1);
  assert.ok(inspect(g).some(c => c.kind === 'ron'));
  g = next(pass(g));
  assert.equal(g.state.position.players[3].furiten, true);
  for (const tile of ['9m', '8m']) g = next(pass(discard(g, tile)));
  g = act(g, { type: 'draw', tile: '2z' });
  assert.equal(g.state.position.players[3].furiten, false);
  g = choose(g, 'tsumo');
  assert.equal(g.state.stage, 'over');
});

test('EMA 2025: guided riichi and the real engine agree at one remaining live tile', () => {
  for (const remaining of [0, 1, 2, 3]) {
    let own = start(0, '123m456p789s1112z');
    own.state.position.wall = remaining + 1;
    own.state.position.first_turns = false;
    own = act(own, { type: 'draw', tile: '9m' });
    const riichi = inspect(own).find(c => c.kind === 'riichi' && c.tile === '9m');
    assert.equal(Boolean(riichi), remaining > 0);
    let other = start();
    other.state.position.wall = remaining + 1;
    other.state.position.first_turns = false;
    if (remaining === 0) assert.throws(() => discard(other, '9m', { riichi: true }), /one live tile/);
    else {
      own = choose(own, 'riichi', '9m');
      other = discard(other, '9m', { riichi: true });
      assert.equal(other.state.position.wall, own.state.position.wall);
      assert.equal(other.state.position.riichi_sticks, own.state.position.riichi_sticks);
      assert.equal(other.state.position.players[0].score, own.state.position.players[0].score);
    }
  }
});

test('rejects fifth copies, incomplete setup, stale events and rolls back invalid decisions', () => {
  let g = start(0, '1111m456p789s123z');
  assert.throws(() => act(g, { type: 'draw', tile: '1m' }), /four copies/);
  assert.equal(g.state.position.wall, 70);
  assert.throws(() => act(g, { type: 'continue' }), /no longer/);
  g = emptyGuided(); g.state.position.players[0].score = null;
  assert.throws(() => act(g, { type: 'setup' }), /points/);
  g.state.position.players[0].score = 30000;
  g = act(g, { type: 'setup' });
  assert.throws(() => act(g, { type: 'hand' }), /exactly 13/);
  assert.throws(() => act(start(), { type: 'discard', tile: '9z' }), /Choose a tile/);
});

test('exhaustion waits for last-discard calls, then next hand rotates scores and the followed seat', () => {
  let g = start(3, '123m456p789s1112z');
  g.state.position.wall = 1; g.state.position.first_turns = false;
  g = discard(g, '2z');
  assert.equal(g.state.position.wall, 0); assert.ok(inspect(g).some(c => c.kind === 'ron'));
  g = next(pass(g)); assert.equal(g.state.stage, 'over');
  g = act(g, { type: 'settle', input: { tenpai: [false, false, false, true] } });
  const finished = structuredClone(g);
  g = act(g, { type: 'next-hand', repeat: false });
  assert.equal(g.state.position.seat, 2);
  assert.deepEqual(g.state.position.players.map(p => p.score), [29000,29000,33000,29000]);
  assert.equal(g.state.position.kyoku, 2); assert.equal(g.state.position.counters, 1);
  assert.equal(g.state.position.wall, 70);
  assert.ok(g.state.position.players.every(p => !p.hand.length && !p.melds.length && !p.discards.length));
  assert.deepEqual(undoGuided(g), finished);
});

test('saved guide is separate from the position editor and conflicting tabs preserve the latest game', async () => {
  const values = new Map([[PHYSICAL_KEY, 'physical editor untouched']]);
  const storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  const locks = { request: async (name, options, action) => { assert.equal(name, GUIDED_FORMAT.lock); return action(); } };
  const conflicts = [];
  const a = new PhysicalStore(storage, locks, { format: GUIDED_FORMAT });
  const b = new PhysicalStore(storage, locks, { format: GUIDED_FORMAT, onConflict: m => conflicts.push(m) });
  await a.read(); await b.read();
  const g = discard(start(), '9m');
  assert.equal(await a.save(g), true);
  assert.equal(await b.save(start(0)), false);
  assert.equal(conflicts.length, 1);
  assert.deepEqual(await b.read(), g);
  assert.deepEqual(undoGuided(parseGuided(values.get(GUIDED_KEY))), undoGuided(g));
  assert.equal(values.get(PHYSICAL_KEY), 'physical editor untouched');
  a.close(); b.close();
});

test('unreadable guides are rejected and dora cycles use winds and dragons separately', () => {
  for (const text of ['{broken', '{}', JSON.stringify({ version: 9, game: emptyGuided() }),
    GUIDED_FORMAT.encode({ ...emptyGuided(), state: { ...emptyGuided().state, stage: 'unknown' } })]) assert.equal(parseGuided(text), null);
  assert.deepEqual(doraTiles({ indicators: ['9m', '4z', '7z', '5z'] }), ['1m', '1z', '5z', '6z']);
});
