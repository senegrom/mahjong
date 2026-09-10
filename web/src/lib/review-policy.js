/** Review captured decisions in order, sharing the ordinary bounded AI worker.
 *
 * The network reads Mortal's planes and answers in Mortal's forty-six moves,
 * so each recorded decision is rebuilt by replaying the events that had
 * happened by then, and the answer is turned back into a move the review can
 * name. A reach is asked in two steps here as everywhere else: the
 * declaration first, then which tile it discards. */
import { MORTAL_REACH, weightsOverOurMoves } from './mortal-space.js';

const aborted = signal => {
  if (signal?.aborted) throw new DOMException('Review changed', 'AbortError');
};

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
    // Asked wherever a reach was legal, not only where it was preferred:
    // the tiles a declaration would throw are what turn its one weight into
    // a weight for each riichi, and both the advice and the move played are
    // read off the same distribution.
    let after = null;
    if (mask[MORTAL_REACH]) {
      after = await analyze(
        engine.review_observation_after_reach(index), engine.review_mask_after_reach(index), signal, 'full',
      );
      aborted(signal);
    }
    const afterReach = action === MORTAL_REACH;
    if (afterReach) action = after.action;
    const chosen = engine.review_action_from_mortal(index, action, afterReach);
    const preferred = choices.find(choice => choice.index === chosen);
    const spread = weightsOverOurMoves(weights, mask, after,
      (entry, reach) => engine.review_action_from_mortal(index, entry, reach));
    const weight = preferred == null ? undefined : spread.get(preferred.index) ?? 0;
    if (!preferred || !Number.isFinite(weight) || weight < 0 || weight > 1) {
      throw new Error('The trained network returned an invalid review choice');
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
      preferred_weight: weight,
      played_weight: played.index == null ? null : spread.get(played.index) ?? 0,
    });
    onProgress(index + 1, notes.length);
  }
  return reviewed;
}
