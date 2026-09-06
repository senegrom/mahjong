import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { analyzeDiscards } from '../src/lib/ui.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const opening = [['discard','6p'], ['discard','9s'], ['discard','8s'], ['discard','7p'],
  ['chii','4m'], ['discard','4z'], ['chii','2m'], ['discard','6z']];
function make(seed, commands = []) {
  const m = new MatchSession(Game, seed, 'club');
  m.advance(false);
  for (const [kind, tile] of commands) {
    m.apply({ type: 'choose', kind, tile });
    m.advance(false);
  }
  return m;
}

test('readiness and selected waits are computed once per distinct legal discard', () => {
  const calls = [];
  const expected = { shanten: 0, waits: ['6z'], waits_left: [3] };
  const engine = { discard_hint(tile) { calls.push(tile); return expected; } };
  const hints = analyzeDiscards(engine, [
    { kind: 'discard', tile: '2z' }, { kind: 'discard', tile: '2z' },
    { kind: 'riichi', tile: '2z' }, { kind: 'ron' }, { kind: 'pass' },
  ]);
  assert.deepEqual(calls, ['2z']);
  assert.equal(hints.get('2z'), expected);
  assert.equal(hints.get('2z'), expected); // Selection reads the cache, not the engine.
  assert.deepEqual(calls, ['2z']);
  assert.equal(analyzeDiscards(null).size, 0);
  assert.equal(analyzeDiscards(engine, []).size, 0);
});

test('different selections in the same seven-pairs hand produce different exact waits', () => {
  const m = make(81, [['discard','9s'], ['discard','5z']]);
  try {
    const saved = JSON.stringify(m.snapshot());
    const hints = analyzeDiscards(m.engine, m.choices);
    assert.deepEqual(hints.get('2z'), { shanten: 0, waits: ['6z'], waits_left: [3] });
    assert.deepEqual(hints.get('6z'), { shanten: 0, waits: ['2z'], waits_left: [3] });
    assert.equal(hints.get('7m').shanten, 1);
    assert.deepEqual(hints.get('7m').waits, []);
    assert.equal(JSON.stringify(m.snapshot()), saved);
  } finally { m.dispose(); }
});

test('each border describes the actual hand after discarding, not the current 14-tile shanten', () => {
  for (const [seed, commands] of [[11, []], [81, [['discard','9s'], ['discard','5z']]], [1, opening]]) {
    const m = make(seed, commands);
    try {
      const saved = JSON.stringify(m.snapshot());
      for (const [tile, hint] of analyzeDiscards(m.engine, m.choices)) {
        const copy = MatchSession.restore(Game, saved);
        try {
          copy.engine.choose('discard', tile); // Do not draw or play opponents yet.
          assert.equal(hint.shanten, copy.view.shanten);
          assert.deepEqual(hint.waits, copy.view.waits);
          assert.deepEqual(hint.waits_left, copy.view.waits_left);
        } finally { copy.dispose(); }
      }
      assert.equal(JSON.stringify(m.snapshot()), saved);
    } finally { m.dispose(); }
  }
});

test('open hands can be gold-ready without any riichi action', () => {
  const m = make(1, opening);
  try {
    assert.equal(m.view.seats[0].melds.length, 2);
    assert.ok(!m.choices.some(c => c.kind === 'riichi'));
    const hints = analyzeDiscards(m.engine, m.choices);
    assert.deepEqual(hints.get('7z'), { shanten: 0, waits: ['2m'], waits_left: [3] });
    assert.equal(hints.get('1m').shanten, 1);
  } finally { m.dispose(); }
});

test('post-call restrictions and riichi do not produce hints for forbidden discards', () => {
  const afterCall = make(1, opening.slice(0, 7));
  try {
    assert.ok(afterCall.view.seats[0].hand.includes('3m'));
    const hints = analyzeDiscards(afterCall.engine, afterCall.choices);
    assert.ok(!hints.has('3m')); // Swap-calling is forbidden.
    assert.equal(hints.get('6z').shanten, 1);
    assert.equal(hints.get('1p').shanten, 2);
  } finally { afterCall.dispose(); }
  const riichi = make(31, [['riichi','8m']]);
  try {
    const hints = analyzeDiscards(riichi.engine, riichi.choices);
    assert.deepEqual([...hints.keys()], ['2z']);
    assert.equal(hints.get('2z').shanten, 0);
  } finally { riichi.dispose(); }
});

test('call windows have no discard hints and repeated analysis cannot change saves or events', () => {
  const m = make(1, opening.slice(0, 4));
  try {
    assert.equal(m.view.phase, 'call');
    const before = { save: JSON.stringify(m.snapshot()), log: m.engine.log(), events: [...m.events] };
    for (let i = 0; i < 20; i++) assert.equal(analyzeDiscards(m.engine, m.choices).size, 0);
    assert.equal(JSON.stringify(m.snapshot()), before.save);
    assert.equal(m.engine.log(), before.log);
    assert.deepEqual(m.events, before.events);
  } finally { m.dispose(); }
});
