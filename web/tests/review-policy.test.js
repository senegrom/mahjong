import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { reviewWithStrong } from '../src/lib/review-policy.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

/** Mortal's reach, which names no tile of its own, and its kan. */
const MORTAL_REACH = 37, MORTAL_KAN = 42;
/** Our own concealed quad, which is what Mortal's kan means here. */
const CONCEALED_KAN = 76;

function finishHand(match) {
  const decisions = [];
  for (let turn = 0; turn < 250 && match.view.phase !== 'over'; turn++) {
    const choices = match.choices;
    const choice = choices.find(c => ['ron', 'tsumo', 'riichi'].includes(c.kind))
      ?? choices.find(c => c.kind === 'pass') ?? choices.find(c => c.kind === 'discard') ?? choices[0];
    assert.ok(choice);
    // The same position twice over: as our own encoder reads it, and as
    // Mortal's does, which is what the trained network is asked. Where a
    // reach is open, the position its declaration would leave behind is
    // kept too, for the second question a declaration asks.
    const mortalMask = match.engine.agent_mask_mortal();
    decisions.push({
      planes: match.engine.agent_observation(), mask: match.engine.agent_mask(), choices: match.engine.agent_choices(), choice,
      mortalPlanes: match.engine.agent_observation_mortal(), mortalMask,
      reachPlanes: mortalMask[MORTAL_REACH] ? match.engine.agent_observation_after_reach() : null,
      reachMask: mortalMask[MORTAL_REACH] ? match.engine.agent_mask_after_reach() : null,
    });
    match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
  }
  assert.equal(match.view.phase, 'over');
  return decisions;
}

function assertInputs(engine, expected) {
  const notes = engine.review();
  assert.equal(notes.length, expected.length);
  expected.forEach((decision, index) => {
    assert.deepEqual(engine.review_observation(index), decision.planes, `historical observation ${index}`);
    assert.deepEqual(engine.review_mask(index), decision.mask);
    assert.deepEqual(engine.review_observation_mortal(index), decision.mortalPlanes, `Mortal observation ${index}`);
    assert.deepEqual(engine.review_mask_mortal(index), decision.mortalMask);
    if (decision.reachMask) {
      assert.deepEqual(engine.review_observation_after_reach(index), decision.reachPlanes, `declared observation ${index}`);
      assert.deepEqual(engine.review_mask_after_reach(index), decision.reachMask);
    }
    assert.deepEqual(engine.review_choices(index), decision.choices);
    assert.equal(notes[index].played_kind, decision.choice.kind);
    assert.equal(notes[index].played_tile ?? null, decision.choice.tile ?? null);
  });
  for (const method of ['review_observation', 'review_mask', 'review_choices',
    'review_observation_mortal', 'review_mask_mortal', 'review_observation_after_reach', 'review_mask_after_reach']) {
    assert.throws(() => engine[method](expected.length), /No recorded decision/);
  }
}

test('review inputs retain the exact pre-move information through saves and hand boundaries', () => {
  for (const seed of [1, 31, 81]) {
    const match = new MatchSession(Game, seed, 'club');
    try {
      match.advance(false);
      const first = finishHand(match), before = match.snapshot();
      assert.ok(first.length > 1);
      assertInputs(match.engine, first);
      assert.deepEqual(match.snapshot(), before, 'reviewing cannot change the game or its history');
      const restored = MatchSession.restore(Game, JSON.stringify(before));
      try { assertInputs(restored.engine, first); } finally { restored.dispose(); }
      match.apply({ type: 'next' }); match.advance(false);
      assert.equal(match.engine.review().length, 0);
      const second = finishHand(match);
      assertInputs(match.engine, second);
    } finally { match.dispose(); }
  }
});

test('the trained review sends historical positions sequentially and matches moves by kind as well as tile', async () => {
  const match = new MatchSession(Game, 31, 'club');
  try {
    match.advance(false); const expected = finishHand(match), notes = match.engine.review();
    const before = match.snapshot(), progress = [];
    assert.ok(notes.some(note => note.call_tile), 'call responses must be included');
    const abort = new AbortController();
    let calls = 0, running = false;
    const results = await reviewWithStrong(match.engine, notes, async (planes, mask, signal, model) => {
      assert.equal(running, false); running = true;
      assert.equal(signal, abort.signal); assert.equal(model, 'full');
      const decision = expected[calls++];
      assert.deepEqual(planes, decision.mortalPlanes); assert.deepEqual(mask, decision.mortalMask);
      // Prefer the matching plain discard over riichi, where it is available.
      const preferred = decision.choices.find(c => c.kind === 'discard' && c.tile === decision.choice.tile)
        ?? decision.choices.find(c => c.index != null);
      // Answered in Mortal's moves, which is the space the network speaks.
      const chosen = match.engine.mortal_action_of(preferred.index);
      assert.ok(mask[chosen], 'The answer must be one the mask allows');
      const allowed = Array.from(mask).filter(Boolean).length;
      const weights = Array.from(mask, valid => valid && allowed > 1 ? .1 / (allowed - 1) : 0);
      weights[chosen] = allowed === 1 ? 1 : .9;
      structuredClone(planes, { transfer: [planes.buffer] });
      await new Promise(resolve => setImmediate(resolve));
      running = false; return { action: chosen, weights };
    }, { signal: abort.signal, onProgress: completed => progress.push(completed) });
    assert.equal(results.length, notes.length); assert.equal(calls, notes.length);
    assert.equal(progress.at(-1), notes.length);
    for (let index = 0; index < notes.length; index++) {
      assert.equal(results[index].played, notes[index].played);
      assert.equal(results[index].call_tile, notes[index].call_tile);
      assert.equal(results[index].call_from, notes[index].call_from);
      assert.ok(results[index].preferred_weight >= .9);
      assert.equal('shanten_advised' in results[index], false, 'Club metrics do not explain a neural choice');
    }
    const riichi = notes.findIndex(note => note.played_kind === 'riichi');
    assert.ok(riichi >= 0); assert.equal(results[riichi].agreed, false, 'riichi and discard are different moves');
    assert.deepEqual(match.snapshot(), before);
  } finally { match.dispose(); }
});

test('leaving the hand cancels the trained review before another inference or late result', async () => {
  const match = new MatchSession(Game, 1, 'club'); match.advance(false); finishHand(match);
  const abort = new AbortController(), progress = [];
  let resolve, calls = 0;
  const pending = reviewWithStrong(match.engine, match.engine.review(), () => {
    calls++; return new Promise(done => { resolve = done; });
  }, { signal: abort.signal, onProgress: count => progress.push(count) });
  const rejected = assert.rejects(pending, { name: 'AbortError' });
  assert.equal(calls, 1);
  match.dispose(); abort.abort(); resolve({});
  await rejected;
  assert.equal(calls, 1); assert.deepEqual(progress, [0]);
});

test('a reviewed reach is asked again which tile it discards, and keeps the declaration weight', async () => {
  const match = new MatchSession(Game, 31, 'club');
  try {
    match.advance(false); const expected = finishHand(match), notes = match.engine.review();
    const declared = expected.findIndex(decision => decision.reachMask);
    assert.ok(declared >= 0, 'This hand offers a reach');
    let index = 0, naming = false, reaches = 0, calls = 0;
    const results = await reviewWithStrong(match.engine, notes, async (planes, mask) => {
      const decision = expected[index]; calls++;
      if (naming) {
        // The second question: only the tiles that declaration may discard.
        assert.deepEqual(planes, decision.reachPlanes); assert.deepEqual(mask, decision.reachMask);
        naming = false; index++;
        return { action: Array.from(mask).findIndex(Boolean), weights: null };
      }
      assert.deepEqual(planes, decision.mortalPlanes); assert.deepEqual(mask, decision.mortalMask);
      if (mask[MORTAL_REACH]) {
        naming = true; reaches++;
        return { action: MORTAL_REACH, weights: Array.from(mask, (valid, i) => valid && i === MORTAL_REACH ? .8 : 0) };
      }
      index++;
      const action = Array.from(mask).findIndex(Boolean);
      return { action, weights: Array.from(mask, (valid, i) => valid && i === action ? .95 : 0) };
    });
    assert.ok(reaches > 0); assert.equal(calls, notes.length + reaches, 'Each declaration asks one extra question');
    const tile = Array.from(expected[declared].reachMask).findIndex(Boolean);
    const advised = match.engine.review_choices(declared)
      .find(choice => choice.index === match.engine.review_action_from_mortal(declared, tile, true));
    assert.equal(advised.kind, 'riichi', 'A declaration discards as a riichi, not as a plain discard');
    assert.equal(results[declared].advised, advised.label);
    assert.equal(results[declared].advised_tile, advised.tile);
    // The weight belongs to the declaration: the tile is only asked once the
    // reach is already decided, so its own answer carries no probability.
    assert.equal(results[declared].preferred_weight, .8);
  } finally { match.dispose(); }
});

test('a second legal kan is reviewed without inventing a probability for an unscored move', async () => {
  const first = { kind: 'concealed-kan', tile: '1m', index: CONCEALED_KAN, label: 'conceals a quad of 1 characters' };
  const second = { kind: 'concealed-kan', tile: '2m', index: null, label: 'conceals a quad of 2 characters' };
  // Mortal has one kan, so the second quad is a move it cannot name at all.
  const engine = {
    review_observation_mortal: () => new Float32Array(1012 * 34),
    review_mask_mortal: () => Uint8Array.from({ length: 46 }, (_, i) => Number(i === MORTAL_KAN)),
    review_choices: () => [first, second],
    review_action_from_mortal: (_index, action, afterReach) => !afterReach && action === MORTAL_KAN ? CONCEALED_KAN : -1,
    mortal_action_of: ours => ours === CONCEALED_KAN ? MORTAL_KAN : -1,
  };
  const notes = [{ played_kind: second.kind, played_tile: second.tile, played: second.label, turn: 1 }];
  const result = await reviewWithStrong(engine, notes, async () => ({ action: MORTAL_KAN, weights: Array.from({ length: 46 }, (_, i) => Number(i === MORTAL_KAN)) }));
  assert.equal(result[0].played_weight, null); assert.equal(result[0].preferred_weight, 1);
  assert.equal(result[0].agreed, false);
});
