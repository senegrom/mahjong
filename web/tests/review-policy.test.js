import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { reviewWithStrong } from '../src/lib/review-policy.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

function finishHand(match) {
  const decisions = [];
  for (let turn = 0; turn < 250 && match.view.phase !== 'over'; turn++) {
    const choices = match.choices;
    const choice = choices.find(c => ['ron', 'tsumo', 'riichi'].includes(c.kind))
      ?? choices.find(c => c.kind === 'pass') ?? choices.find(c => c.kind === 'discard') ?? choices[0];
    assert.ok(choice);
    if (match.view.phase === 'act') decisions.push({
      planes: match.engine.agent_observation(), mask: match.engine.agent_mask(), choices: match.engine.agent_choices(), choice,
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
    assert.deepEqual(engine.review_choices(index), decision.choices);
    assert.equal(notes[index].played_kind, decision.choice.kind);
    assert.equal(notes[index].played_tile ?? null, decision.choice.tile ?? null);
  });
  for (const method of ['review_observation', 'review_mask', 'review_choices']) {
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

test('Strong review sends historical positions sequentially and matches moves by kind as well as tile', async () => {
  const match = new MatchSession(Game, 31, 'club');
  try {
    match.advance(false); const expected = finishHand(match), notes = match.engine.review();
    const before = match.snapshot(), progress = [];
    const abort = new AbortController();
    let calls = 0, running = false;
    const results = await reviewWithStrong(match.engine, notes, async (planes, mask, signal, model) => {
      assert.equal(running, false); running = true;
      assert.equal(signal, abort.signal); assert.equal(model, 'strong');
      const decision = expected[calls++];
      assert.deepEqual(planes, decision.planes); assert.deepEqual(mask, decision.mask);
      // Prefer the matching plain discard over riichi, where it is available.
      const chosen = decision.choices.find(c => c.kind === 'discard' && c.tile === decision.choice.tile)
        ?? decision.choices.find(c => c.index != null);
      const allowed = Array.from(mask).filter(Boolean).length;
      const weights = Array.from(mask, valid => valid && allowed > 1 ? .1 / (allowed - 1) : 0);
      weights[chosen.index] = allowed === 1 ? 1 : .9;
      structuredClone(planes, { transfer: [planes.buffer] });
      await new Promise(resolve => setImmediate(resolve));
      running = false; return { action: chosen.index, weights };
    }, { signal: abort.signal, onProgress: completed => progress.push(completed) });
    assert.equal(results.length, notes.length); assert.equal(calls, notes.length);
    assert.equal(progress.at(-1), notes.length);
    for (let index = 0; index < notes.length; index++) {
      assert.equal(results[index].played, notes[index].played);
      assert.ok(results[index].preferred_weight >= .9);
      assert.equal('shanten_advised' in results[index], false, 'Club metrics do not explain a neural choice');
    }
    const riichi = notes.findIndex(note => note.played_kind === 'riichi');
    assert.ok(riichi >= 0); assert.equal(results[riichi].agreed, false, 'riichi and discard are different moves');
    assert.deepEqual(match.snapshot(), before);
  } finally { match.dispose(); }
});

test('leaving the hand cancels Strong review before another inference or late result', async () => {
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

test('a second legal kan is reviewed without inventing a probability for an unscored move', async () => {
  const first = { kind: 'concealed-kan', tile: '1m', index: 71, label: 'conceals a quad of 1 characters' };
  const second = { kind: 'concealed-kan', tile: '2m', index: null, label: 'conceals a quad of 2 characters' };
  const engine = { review_observation: () => new Float32Array(3298), review_mask: () => [1], review_choices: () => [first, second] };
  const notes = [{ played_kind: second.kind, played_tile: second.tile, played: second.label, turn: 1 }];
  const result = await reviewWithStrong(engine, notes, async () => ({ action: 71, weights: Array.from({ length: 79 }, (_, i) => Number(i === 71)) }));
  assert.equal(result[0].played_weight, null); assert.equal(result[0].preferred_weight, 1);
  assert.equal(result[0].agreed, false);
});
