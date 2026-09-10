/** Review captured decisions in order, sharing the ordinary bounded AI worker. */
const aborted = signal => {
  if (signal?.aborted) throw new DOMException('Review changed', 'AbortError');
};

export async function reviewWithStrong(engine, notes, analyze, { signal, onProgress = () => {} } = {}) {
  aborted(signal);
  // Copy the small observations before awaiting anything. Advancing the hand or
  // freeing its engine cannot invalidate a review already in flight. Each input
  // is released as it is consumed; no engine positions are cloned or retained.
  const inputs = notes.map((_, index) => ({
    planes: engine.review_observation(index), mask: engine.review_mask(index), choices: engine.review_choices(index),
  }));
  const reviewed = [];
  for (let index = 0; index < notes.length; index++) {
    aborted(signal);
    const { planes, mask, choices } = inputs[index];
    inputs[index] = null;
    onProgress(index, notes.length);
    const { action, weights } = await analyze(planes, mask, signal, 'strong');
    aborted(signal);
    const preferred = choices.find(choice => choice.index === action);
    const weight = weights?.[action];
    if (!preferred || !Number.isFinite(weight) || weight < 0 || weight > 1) {
      throw new Error('Strong returned an invalid review choice');
    }
    const note = notes[index];
    const played = choices.find(choice => choice.kind === note.played_kind
      && (choice.tile ?? null) === (note.played_tile ?? null));
    if (!played) throw new Error('The recorded move is missing from this decision');
    reviewed.push({
      turn: note.turn, played: note.played, played_kind: note.played_kind, played_tile: note.played_tile,
      call_tile: note.call_tile, call_from: note.call_from,
      dora_types: note.dora_types, advised: preferred.label, advised_tile: preferred.tile,
      agreed: played.kind === preferred.kind && (played.tile ?? null) === (preferred.tile ?? null),
      preferred_weight: weight, played_weight: played.index == null ? null : weights[played.index],
    });
    onProgress(index + 1, notes.length);
  }
  return reviewed;
}
