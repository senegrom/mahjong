/** Review captured decisions in order, sharing the ordinary bounded AI worker.
 *
 * The network reads Mortal's planes and answers in Mortal's forty-six moves,
 * so each recorded decision is rebuilt by replaying the events that had
 * happened by then, and the answer is turned back into a move the review can
 * name. A reach is asked in two steps here as everywhere else: the
 * declaration first, then which tile it discards. */
const aborted = signal => {
  if (signal?.aborted) throw new DOMException('Review changed', 'AbortError');
};

/** Mortal's reach, which names no tile of its own. */
const MORTAL_REACH = 37;

export async function reviewWithStrong(engine, notes, analyze, { signal, onProgress = () => {} } = {}) {
  aborted(signal);
  // Copy the small observations before awaiting anything. Advancing the hand or
  // freeing its engine cannot invalidate a review already in flight. Each input
  // is released as it is consumed; no engine positions are cloned or retained.
  const inputs = notes.map((_, index) => ({
    planes: engine.review_observation_mortal(index),
    mask: engine.review_mask_mortal(index),
    choices: engine.review_choices(index),
  }));
  const reviewed = [];
  for (let index = 0; index < notes.length; index++) {
    aborted(signal);
    const { planes, mask, choices } = inputs[index];
    inputs[index] = null;
    onProgress(index, notes.length);
    let { action, weights } = await analyze(planes, mask, signal, 'full');
    aborted(signal);
    let afterReach = false;
    if (action === MORTAL_REACH) {
      afterReach = true;
      ({ action } = await analyze(
        engine.review_observation_after_reach(index), engine.review_mask_after_reach(index), signal, 'full',
      ));
      aborted(signal);
    }
    const chosen = engine.review_action_from_mortal(index, action, afterReach);
    const preferred = choices.find(choice => choice.index === chosen);
    // The weight belongs to the declaration, not to the tile it names: the
    // second question is asked only once the first is decided.
    const weight = weights?.[afterReach ? MORTAL_REACH : action];
    if (!preferred || !Number.isFinite(weight) || weight < 0 || weight > 1) {
      throw new Error('The trained network returned an invalid review choice');
    }
    const note = notes[index];
    const played = choices.find(choice => choice.kind === note.played_kind
      && (choice.tile ?? null) === (note.played_tile ?? null));
    if (!played) throw new Error('The recorded move is missing from this decision');
    const playedAction = played.index == null ? -1 : engine.mortal_action_of(played.index);
    reviewed.push({
      turn: note.turn, played: note.played, played_kind: note.played_kind, played_tile: note.played_tile,
      call_tile: note.call_tile, call_from: note.call_from,
      dora_types: note.dora_types, advised: preferred.label, advised_tile: preferred.tile,
      agreed: played.kind === preferred.kind && (played.tile ?? null) === (preferred.tile ?? null),
      preferred_weight: weight, played_weight: playedAction < 0 ? null : weights[playedAction],
    });
    onProgress(index + 1, notes.length);
  }
  return reviewed;
}
