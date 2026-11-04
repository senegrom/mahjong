import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

// What the page reads from the engine for the last hand's rules: Mortal's
// planes, 1012 over the 34 tile kinds; its 46 moves; and the codes, a byte a
// tile kind for dama and one number for decline. The rules themselves, the
// preview's equality with the pending reach and the codes in South 4 are the
// native tests' (engine/riichi-wasm/src/last_hand.rs and lib.rs).
const PLANES = 1012 * 34, MOVES = 46, KINDS = 34;

/** Dama codes as the page reads them, which outside South 4 are all zero. */
function noDama(codes, at) {
  assert.ok(codes instanceof Uint8Array, `${at}: dama codes are a Uint8Array`);
  assert.deepEqual([...codes], new Array(KINDS).fill(0), at);
}

/** A decline code as the page reads it, which outside South 4 is 0. */
function noDecline(code, at) {
  assert.equal(typeof code, 'number', `${at}: a decline code is a number`);
  assert.equal(code, 0, at);
}

test('the last hand\'s bindings answer the page in the types it reads', () => {
  const match = new MatchSession(Game, 81, 'neural');
  const engine = match.engine;
  try {
    match.advance(false);
    // The person playing throws what they may until a Trained opponent owes
    // a decision.
    for (let steps = 0; !engine.needs_opponent_move(); steps++) {
      assert.ok(steps < 20, 'a Trained opponent comes to a decision');
      const choice = match.choices.find(c => c.kind === 'discard') ?? match.choices.find(c => c.kind === 'pass');
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null });
      match.advance(false);
    }
    for (const planes of [engine.opponent_observation_after_reach(), engine.opponent_observation_after_reach_renamed()]) {
      assert.ok(planes instanceof Float32Array, 'a previewed reach is planes');
      assert.equal(planes.length, PLANES);
    }
    const reach = engine.opponent_mask_after_reach();
    assert.ok(reach instanceof Uint8Array, 'a previewed reach\'s mask is a Uint8Array');
    assert.equal(reach.length, MOVES);
    noDama(engine.opponent_all_last_dama(), 'the opponent');
    noDecline(engine.opponent_all_last_decline(), 'the opponent');

    // The opponents pass where they may, and otherwise play their first open
    // move, until the person playing decides.
    for (let steps = 0; engine.needs_opponent_move(); steps++) {
      assert.ok(steps < 20, 'the person playing comes to a decision');
      const mask = engine.opponent_mask_mortal();
      match.apply({ type: 'opponent', action: mask[45] ? 45 : mask.findIndex(Boolean) });
      match.advance(false);
    }
    const choice = match.choices.find(c => c.kind === 'discard') ?? match.choices.find(c => c.kind === 'pass');
    assert.ok(choice, 'the person playing decides');
    noDama(engine.agent_all_last_dama(), 'the followed seat');
    noDecline(engine.agent_all_last_decline(), 'the followed seat');
    match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null });

    const recorded = engine.review().length;
    assert.ok(recorded > 0, 'a decision is recorded');
    for (let index = 0; index < recorded; index++) {
      noDama(engine.review_all_last_dama(index), `decision ${index + 1}`);
      noDecline(engine.review_all_last_decline(index), `decision ${index + 1}`);
    }
    for (const method of ['review_all_last_dama', 'review_all_last_decline']) {
      assert.throws(() => engine[method](recorded), /No recorded decision at this index/, method);
    }
    const analysis = new PhysicalAnalysis(engine.review_position(0));
    try {
      noDama(analysis.agent_all_last_dama(), 'a typed-in table');
      noDecline(analysis.agent_all_last_decline(), 'a typed-in table');
    } finally {
      analysis.free();
    }
  } finally {
    match.dispose();
  }
});
