import assert from 'node:assert/strict';
import test from 'node:test';
import { MatchSession } from '../src/lib/session.js';

// A saved human call with two unasked opponents. No historical network
// actions are supplied: those old action schemas must still be rejected.
class LegacyCallGame {
  constructor() { this.chosen = false; this.remaining = 0; this.freed = false; }
  view() {
    return { phase: !this.chosen || this.remaining ? 'call' : 'act', hands_played: 0,
      seats: [{ discards: [], melds: [], riichi: false }], remaining: this.remaining };
  }
  choices() { return !this.chosen ? [{ kind: 'pass' }] : this.remaining ? [] : [{ kind: 'discard', tile: '1m' }]; }
  choose(kind) { assert.equal(kind, 'pass'); this.chosen = true; this.remaining = 2; }
  advance() { return []; }
  needs_opponent_move() { return this.remaining > 0; }
  opponent_mask_mortal() { const mask = new Uint8Array(46); mask[45] = 1; return mask; }
  play_opponent_mortal(action) { assert.equal(action, 45); this.remaining--; }
  opponent_mask() { throw new Error('Legacy engine-space mask must not be consulted'); }
  game_is_over() { return false; }
  free() { this.freed = true; }
}

function legacySave() {
  const expected = new MatchSession(LegacyCallGame, 1, 'neural');
  expected.apply({ type: 'choose', kind: 'pass' });
  expected.apply({ type: 'opponent', action: 45 });
  expected.apply({ type: 'opponent', action: 45 });
  const text = JSON.stringify({ version: 1, seed: 1, difficulty: 'neural',
    commands: [{ type: 'choose', kind: 'pass' }], state: expected.stateKey() });
  expected.dispose();
  return text;
}

test('implicit legacy passes migrate to legal Mortal passes and restore twice', () => {
  const saved = legacySave();
  const match = MatchSession.restore(LegacyCallGame, saved, { ai() { throw new Error('No new inference'); } });
  assert.deepEqual(match.commands, [
    { type: 'choose', kind: 'pass', tile: null },
    { type: 'opponent', action: 45 }, { type: 'opponent', action: 45 },
  ]);
  const restored = MatchSession.restore(LegacyCallGame, JSON.stringify(match.snapshot()));
  assert.equal(restored.stateKey(), match.stateKey());
  match.dispose(); restored.dispose();
});

test('an unavailable historical pass fails closed without applying another move', () => {
  let game;
  class RefusesPass extends LegacyCallGame {
    constructor() { super(); game = this; }
    opponent_mask_mortal() { const mask = new Uint8Array(46); mask[43] = 1; return mask; }
    play_opponent_mortal() { assert.fail('No substitute action may be played'); }
  }
  const text = legacySave();
  assert.throws(() => MatchSession.restore(RefusesPass, text), /Cannot migrate a historical claim/);
  assert.equal(game.freed, true);
  assert.equal(game.remaining, 2);
  assert.equal(text, legacySave());
});

test('explicit old-schema network actions remain rejected', () => {
  const saved = JSON.parse(legacySave());
  saved.commands.push({ type: 'opponent', action: 70 });
  assert.throws(() => MatchSession.restore(LegacyCallGame, JSON.stringify(saved)), /Unsupported legacy opponent moves/);
});
