/** Reading Mortal's forty-six moves as weights over ours.
 *
 * The network answers in the space it was trained in, and the page plays in
 * ours. The two do not line up one for one, so a weight cannot simply be
 * looked up:
 *
 *  - Mortal names a red five apart from the plain tile, and our rules have
 *    no red fives. Two of its moves mean the same discard, and both shares
 *    belong to it; taking only one understates every five on the table.
 *  - A reach names no tile. Its weight is the weight of declaring, and
 *    which tile the declaration throws is a second question, asked from the
 *    position the declaration leaves behind. The declaration's weight
 *    belongs across the tiles in the proportions that second answer gives.
 *  - A quad is one move of Mortal's and three of ours, and only one of the
 *    three can be legal at a time.
 *
 * Gathered this way every legal move of Mortal's is counted exactly once,
 * so the weights over our moves are a distribution and can be read as one.
 */

/** Mortal's reach, which names no tile of its own. */
export const MORTAL_REACH = 37;

/**
 * Weight per action of ours, as a map.
 *
 * `translate(action, afterReach)` is the engine's own answer for which of
 * our actions one of Mortal's means here, or a negative number for none.
 * `after` is the second question's weights, needed only where a reach is
 * legal; without it a legal reach contributes nothing rather than landing
 * all of its weight on one arbitrary tile.
 */
export function weightsOverOurMoves(weights, mask, after, translate) {
  const ours = new Map();
  const add = (index, weight) => {
    if (index >= 0 && weight > 0) ours.set(index, (ours.get(index) ?? 0) + weight);
  };
  for (let action = 0; action < mask.length; action += 1) {
    if (!mask[action]) continue;
    if (action === MORTAL_REACH) {
      for (let tile = 0; tile < 34; tile += 1) {
        add(translate(tile, true), weights[action] * (after?.weights[tile] ?? 0));
      }
      continue;
    }
    add(translate(action, false), weights[action]);
  }
  return ours;
}
