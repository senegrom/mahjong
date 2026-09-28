import assert from 'node:assert/strict';
import test from 'node:test';
import { MatchSession } from '../src/lib/session.js';

// Before format 5 a trained opponent's move was recorded in the engine's own
// seventy-eight actions, not Mortal's forty-six. Such saves are refused
// rather than replayed as different moves. (A neural save in which the player
// answered a discard always holds that discard as such a move, so older
// implicit passes after a call never reach the restore loop either.)
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

test('explicit old-schema network actions remain rejected', () => {
  const saved = JSON.parse(legacySave());
  saved.commands.push({ type: 'opponent', action: 70 });
  assert.throws(() => MatchSession.restore(LegacyCallGame, JSON.stringify(saved)), /Unsupported legacy opponent moves/);
});
