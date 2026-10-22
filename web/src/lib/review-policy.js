import { weightsByChoice, MORTAL_REACH } from './action-weights.js';

/** Review captured decisions in order, sharing the ordinary bounded AI worker.
 *
 * The network reads Mortal's planes and answers in Mortal's forty-six moves,
 * so each recorded decision is rebuilt by replaying the events that had
 * happened by then, and the answer is turned back into a move the review can
 * name. A reach is asked in two steps here as everywhere else: the
 * declaration first, then which tile it discards.
 *
 * A hand's review asks one position at a time. A whole-game review passes
 * `batch`, `{ size, analyze(positions, signal) }`, and asks that many
 * positions in one forward pass, then the second question of every
 * declaration among them in another. Either way each decision is judged the
 * same. With `skipForced`, a decision with one legal move is answered
 * without the network: the weights over one move are that move's whole. */
const aborted = signal => {
  if (signal?.aborted) throw new DOMException('Review changed', 'AbortError');
};

/** The answer the network would give a position with one legal move. */
function forcedAnswer(mask) {
  const legal = Array.from(mask, allowed => (allowed ? 1 : 0));
  return legal.reduce((sum, value) => sum + value, 0) === 1
    ? { action: legal.indexOf(1), weights: legal } : null;
}

export async function reviewWithStrong(engine, notes, analyze, { signal, onProgress = () => {}, batch = null, skipForced = false } = {}) {
  aborted(signal);
  // Copy the small observations before awaiting anything. The engine calls
  // that follow an await, for a reach's second question and to translate the
  // answers back, are guarded only by the abort signal checked after each
  // await, so the caller must abort the review when the hand moves on or the
  // engine is freed. Each input is released as it is consumed; no engine
  // positions are cloned or retained.
  const inputs = notes.map((_, index) => ({
    planes: engine.review_observation_mortal(index),
    mask: engine.review_mask_mortal(index),
    choices: engine.review_choices(index),
  }));
  const size = batch ? batch.size : 1;
  // A group's positions in one request, or one request each, in order.
  const ask = async positions => {
    const answers = positions.map(position => (skipForced ? forcedAnswer(position.mask) : null));
    const asked = positions.filter((_, index) => !answers[index]);
    if (asked.length) {
      const replies = batch ? await batch.analyze(asked, signal)
        : [await analyze(asked[0].planes, asked[0].mask, signal)];
      aborted(signal);
      if (!Array.isArray(replies) || replies.length !== asked.length) {
        throw new Error('The trained network returned the wrong number of answers');
      }
      let next = 0;
      for (let index = 0; index < answers.length; index++) answers[index] ??= replies[next++];
    }
    return answers;
  };
  const reviewed = [];
  for (let start = 0; start < notes.length; start += size) {
    aborted(signal);
    const group = [];
    for (let index = start; index < Math.min(notes.length, start + size); index++) {
      group.push({ index, ...inputs[index] });
      inputs[index] = null;
    }
    onProgress(start, notes.length);
    const firsts = await ask(group);
    // A declaration is asked again which tile it discards, from the position
    // the declaration leaves behind.
    const reaching = group.filter((_, slot) => firsts[slot].action === MORTAL_REACH);
    const seconds = reaching.length ? await ask(reaching.map(({ index }) => ({
      planes: engine.review_observation_after_reach(index), mask: engine.review_mask_after_reach(index),
    }))) : [];
    group.forEach(({ index, mask, choices }, slot) => {
      const first = firsts[slot], second = first.action === MORTAL_REACH ? seconds[reaching.findIndex(entry => entry.index === index)] : null;
      reviewed.push(reviewedNote(engine, index, notes[index], choices, mask, first, second));
    });
    onProgress(Math.min(notes.length, start + size), notes.length);
  }
  return reviewed;
}

/** One decision judged from the network's answers: the first question's, and
 * the second's where it declared a reach. */
function reviewedNote(engine, index, note, choices, mask, first, second) {
  const action = second ? second.action : first.action;
  const chosen = engine.review_action_from_mortal(index, action, Boolean(second));
  const weighted = weightsByChoice(engine, choices, first.weights,
    action => engine.review_action_from_mortal(index, action, false),
    second?.weights ? { weights: second.weights, fromMortal: action => engine.review_action_from_mortal(index, action, true) } : null);
  const preferred = weighted.find(choice => choice.index === chosen);
  // The weight belongs to the declaration, not to the tile it names: the
  // second question is asked only once the first is decided. Keep its
  // conditional weight separate rather than claiming it is a first-stage move.
  const weight = preferred?.weight;
  if (!preferred || !Number.isFinite(weight) || weight < 0 || weight > 1) {
    throw new Error('The trained network returned an invalid review choice');
  }
  const played = weighted.find(choice => choice.kind === note.played_kind
    && (choice.tile ?? null) === (note.played_tile ?? null));
  if (!played) throw new Error('The recorded move is missing from this decision');
  return {
    turn: note.turn, played: note.played, played_kind: note.played_kind, played_tile: note.played_tile,
    call_tile: note.call_tile, call_from: note.call_from,
    dora_types: note.dora_types, advised: preferred.label, advised_tile: preferred.tile,
    agreed: played.kind === preferred.kind && (played.tile ?? null) === (preferred.tile ?? null),
    preferred_weight: weight, played_weight: played.weight,
    advised_kind: preferred.kind,
    preferred_conditional_weight: preferred.conditionalWeight ?? null,
    played_conditional_weight: played.conditionalWeight ?? null,
    // Every move that was open and its weight, which Trained AI's view lists
    // at every reviewed decision of a replay, and by which the step bar
    // weighs both moves (moveWeight); and whether there was a choice at
    // all, which a forced move, such as a discard after riichi, is not.
    weighted,
    forced: Array.from(mask).filter(Boolean).length === 1,
  };
}
