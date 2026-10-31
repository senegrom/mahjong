import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { reviewWithStrong } from '../src/lib/review-policy.js';
import { MOVES, average, swapMoves } from '../src/lib/suits.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

/** Mortal's reach, which names no tile of its own, and its kan. */
const MORTAL_REACH = 37, MORTAL_KAN = 42;
/** Our own concealed quad, which is what Mortal's kan means here. */
const CONCEALED_KAN = 76;
/** A table's planes by their bits, in a word. */
function digest(planes) {
  let hash = 2166136261;
  for (const word of new Uint32Array(Float32Array.from(planes).buffer)) hash = Math.imul(hash ^ word, 16777619) >>> 0;
  return hash;
}
/** The moves a mask opens. */
const open = mask => Array.from(mask).flatMap((allowed, move) => (allowed ? [move] : []));
/** Weights over a mask's moves: `top` on `best`, the rest shared evenly. */
const spread = (mask, best, top) => Array.from(mask, (allowed, move) => (!allowed ? 0 : move === best ? (open(mask).length > 1 ? top : 1)
  : (1 - top) / (open(mask).length - 1)));

function finishHand(match) {
  const decisions = [];
  for (let turn = 0; turn < 250 && match.view.phase !== 'over'; turn++) {
    const choices = match.choices;
    const choice = choices.find(c => ['ron', 'tsumo', 'riichi'].includes(c.kind))
      ?? choices.find(c => c.kind === 'pass') ?? choices.find(c => c.kind === 'discard') ?? choices[0];
    assert.ok(choice);
    // The position as Mortal's encoder reads it, which is what the trained
    // network is asked. Where a reach is open, the position its declaration
    // would leave behind is kept too, for the second question a declaration
    // asks.
    const mortalMask = match.engine.agent_mask_mortal();
    decisions.push({
      choices: match.engine.agent_choices(), choice,
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
  for (const method of ['review_choices', 'review_observation_mortal', 'review_mask_mortal',
    'review_observation_after_reach', 'review_mask_after_reach']) {
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
    const results = await reviewWithStrong(match.engine, notes, async (...args) => {
      const [planes, mask, signal] = args;
      assert.equal(running, false); running = true;
      assert.equal(signal, abort.signal); assert.equal(args.length, 3, 'one network: no model is named');
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
    let index = 0, step = 'first', reaches = 0, calls = 0;
    const results = await reviewWithStrong(match.engine, notes, async (planes, mask) => {
      const decision = expected[index]; calls++;
      if (step === 'renamed') {
        // Hesitant over the declaration, Trained AI asks the renamed game
        // too, its moves renamed; it answers here as the plain answer did.
        assert.deepEqual(planes, match.engine.review_observation_mortal_renamed(index));
        assert.deepEqual(Array.from(mask), swapMoves(Array.from(decision.mortalMask)));
        step = 'naming';
        return { action: MORTAL_REACH, weights: swapMoves(spread(decision.mortalMask, MORTAL_REACH, .8)) };
      }
      if (step === 'naming') {
        // The second question: only the tiles that declaration may discard.
        assert.deepEqual(planes, decision.reachPlanes); assert.deepEqual(mask, decision.reachMask);
        step = 'first'; index++;
        const tile = Array.from(mask).findIndex(Boolean);
        return { action: tile, weights: Array.from(mask, (_, i) => Number(i === tile)) };
      }
      assert.deepEqual(planes, decision.mortalPlanes); assert.deepEqual(mask, decision.mortalMask);
      if (mask[MORTAL_REACH]) {
        step = 'renamed'; reaches++;
        return { action: MORTAL_REACH, weights: spread(mask, MORTAL_REACH, .8) };
      }
      index++;
      const action = Array.from(mask).findIndex(Boolean);
      return { action, weights: spread(mask, action, .95) };
    });
    assert.ok(reaches > 0); assert.equal(calls, notes.length + 2 * reaches, 'Each declaration asks one extra question, and its renamed game one more');
    const tile = Array.from(expected[declared].reachMask).findIndex(Boolean);
    const advised = match.engine.review_choices(declared)
      .find(choice => choice.index === match.engine.review_action_from_mortal(declared, tile, true));
    assert.equal(advised.kind, 'riichi', 'A declaration discards as a riichi, not as a plain discard');
    assert.equal(results[declared].advised, advised.label);
    assert.equal(results[declared].advised_tile, advised.tile);
    // The weight belongs to the declaration: the tile is only asked once the
    // reach is already decided, so its own answer carries no probability.
    // The renamed game answering as the plain one did, the mean is the same.
    assert.ok(Math.abs(results[declared].preferred_weight - .8) < 1e-12);
  } finally { match.dispose(); }
});

test('a review in batches asks the decisions it hesitates over again, renamed, in a batch of their own, and judges them by the mean', async () => {
  const match = new MatchSession(Game, 31, 'club');
  try {
    match.advance(false); finishHand(match);
    const engine = match.engine, notes = engine.review();
    const masks = notes.map((_, index) => Array.from(engine.review_mask_mortal(index), Number));
    // Every decision by its planes, and by the renamed game's.
    const decisionOf = new Map(notes.map((_, index) => [digest(engine.review_observation_mortal(index)), index]));
    const renamedOf = new Map(notes.map((_, index) => [digest(engine.review_observation_mortal_renamed(index)), index]));
    // Trained AI hesitates over the decisions of even index, putting 0.6 on
    // its first legal move, and is sure of the rest at 0.95. The renamed game
    // puts 0.7 on the plain answer's second move, which the mean then plays.
    const plain = new Map(), renamed = new Map(), sent = [];
    const analyze = async positions => {
      sent.push(positions.map(({ planes, mask }) => ({ planes: digest(planes), mask: Array.from(mask, Number) })));
      return positions.map(({ planes, mask }) => {
        const key = digest(planes);
        if (decisionOf.has(key)) {
          const index = decisionOf.get(key), first = open(mask)[0];
          plain.set(index, { action: first, weights: spread(mask, first, index % 2 ? .95 : .6) });
          return plain.get(index);
        }
        const index = renamedOf.get(key);
        assert.ok(index !== undefined, 'a batch asks the decisions, or their renamed games');
        const second = MOVES[open(masks[index])[1]];
        renamed.set(index, { action: second, weights: spread(mask, second, .7) });
        return renamed.get(index);
      });
    };
    const results = await reviewWithStrong(engine, notes, null, { skipForced: true, batch: { size: () => 4, analyze } });
    assert.equal(results.length, notes.length);
    // Each group of four in a batch, its forced moves unasked, and then those
    // of it Trained AI hesitated over, renamed, with their moves renamed.
    const expected = [];
    for (let start = 0; start < notes.length; start += 4) {
      const group = notes.map((_, index) => index).slice(start, start + 4).filter(index => open(masks[index]).length > 1);
      if (!group.length) continue;
      expected.push(group.map(index => ({ planes: digest(engine.review_observation_mortal(index)), mask: masks[index] })));
      const again = group.filter(index => index % 2 === 0);
      if (again.length) expected.push(again.map(index => ({ planes: digest(engine.review_observation_mortal_renamed(index)), mask: swapMoves(masks[index]) })));
    }
    assert.deepEqual(sent, expected);
    assert.ok(renamed.size > 0 && renamed.size < plain.size, `${renamed.size} of ${plain.size} asked again`);
    for (const [index, answer] of plain) {
      const mean = renamed.has(index) ? average(answer.weights, renamed.get(index).weights, masks[index]) : answer;
      if (renamed.has(index)) assert.equal(mean.action, open(masks[index])[1], 'the mean plays the renamed game\'s choice');
      const chosen = engine.review_action_from_mortal(index, mean.action, false);
      assert.equal(results[index].advised, engine.review_choices(index).find(choice => choice.index === chosen).label);
      assert.equal(results[index].preferred_weight, mean.weights[mean.action]);
    }
  } finally { match.dispose(); }
});

test('a second legal kan is reviewed without inventing a probability for an unscored move', async () => {
  const first = { kind: 'concealed-kan', tile: '1m', index: CONCEALED_KAN, label: 'conceals a quad of 1 characters' };
  const second = { kind: 'concealed-kan', tile: '2m', index: null, label: 'conceals a quad of 2 characters' };
  // Mortal has one kan, so the second quad is a move it cannot name at all.
  const engine = {
    review_observation_mortal: () => new Float32Array(1012 * 34),
    // Never read: a sure answer is not asked again.
    review_observation_mortal_renamed: () => new Float32Array(1012 * 34).fill(1),
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
