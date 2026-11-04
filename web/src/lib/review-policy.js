import { weightsByChoice } from './action-weights.js';
import { reviewTable, trainedMoves } from './trained-move.js';

/** Review captured decisions in order, sharing the ordinary bounded AI worker.
 *
 * The network reads Mortal's planes and answers in Mortal's forty-six moves,
 * so each recorded decision is rebuilt by replaying the events that had
 * happened by then, and the answer is turned back into a move the review can
 * name. Each decision is judged by Trained AI's move, built as everywhere
 * else (trained-move.js): each question Trained AI hesitates over asked again
 * of the renamed game and judged by the mean of the two answers (suits.js); a
 * reach asked in two steps, the declaration first, then which tile it
 * discards; and the last hand's rules over that.
 *
 * A hand's review asks one position at a time. A whole-game review passes
 * `batch`, `{ size(), analyze(positions, signal) }`, and asks as many
 * positions as `size()` says in one forward pass, asked again before each,
 * since the review sizes its passes by the speed it measures as it goes;
 * then those it hesitates over, renamed, in another; then the second
 * question of every declaration among them, and those of these it hesitates
 * over, the same way. Either way each decision is judged the same. With
 * `skipForced`, a decision with one legal move is answered without the
 * network: the weights over one move are that move's whole. */
const aborted = signal => {
  if (signal?.aborted) throw new DOMException('Review changed', 'AbortError');
};

export async function reviewWithStrong(engine, notes, analyze, { signal, onProgress = () => {}, batch = null, skipForced = false } = {}) {
  aborted(signal);
  // Copy the small observations before awaiting anything. The engine calls
  // that follow an await, for a reach's second question, for the renamed
  // game of a question Trained AI hesitates over, for the last hand's codes
  // and to translate the answers back, are guarded only by the abort signal
  // checked after each await, so the caller must abort the review when the
  // hand moves on or the engine is freed. Each input is released as it is
  // consumed; no engine positions are cloned or retained.
  const inputs = notes.map((_, index) => ({ ...reviewTable(engine, index), choices: engine.review_choices(index) }));
  // Positions in one request, or one position without a batch.
  const request = async positions => {
    const replies = batch ? await batch.analyze(positions, signal)
      : [await analyze(positions[0].planes, positions[0].mask, signal)];
    aborted(signal);
    if (!Array.isArray(replies) || replies.length !== positions.length) {
      throw new Error('The trained network returned the wrong number of answers');
    }
    return replies;
  };
  const reviewed = [];
  for (let start = 0, size; start < notes.length; start += size) {
    aborted(signal);
    size = batch ? batch.size() : 1;
    const group = [];
    for (let index = start; index < Math.min(notes.length, start + size); index++) {
      group.push({ index, ...inputs[index] });
      inputs[index] = null;
    }
    onProgress(start, notes.length);
    // Trained AI's moves at the group's decisions: its first questions in one
    // request, those it hesitates over again, renamed, in another, and the
    // tile of each declaration among them, from the position the declaration
    // would leave behind, the same way, in two more at most.
    const moves = await trainedMoves(request, group, { signal, skipForced });
    aborted(signal);
    group.forEach(({ index, mask, choices }, slot) => reviewed.push(reviewedNote(engine, index, notes[index], choices, mask, moves[slot])));
    onProgress(Math.min(notes.length, start + size), notes.length);
  }
  return reviewed;
}

/** One decision judged by Trained AI's move there (trainedMoves): weighed by
 * the first question's answer, and the second's where it asked which tile a
 * reach discards. */
function reviewedNote(engine, index, note, choices, mask, { first, second, final, ruled }) {
  // A riichi only where the move is one: a riichi the last hand's rules threw
  // undeclared is the plain discard of its tile.
  const chosen = engine.review_action_from_mortal(index, final.action, final.afterReach);
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
    // Which of the last hand's rules made Trained AI's move, if any did: its
    // weight then did not choose it (game-review.js, judgement).
    ruled: ruled ? { rules: ruled.rules } : null,
  };
}
