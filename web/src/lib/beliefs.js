/** What the network thinks it cannot see.
 *
 * The trained network answers three things about a position at once: which
 * move it prefers, what the hand is worth, and what each opponent is
 * holding. The third is the belief head, and this turns its raw numbers
 * into something a page can show and a search can sample from.
 *
 * A row is a distribution over the thirty-four kinds: the chance that a
 * tile drawn at random from that opponent's concealed hand is of each kind.
 * That is exactly what the head was trained against, so multiplying a row
 * by how many tiles the opponent actually holds gives expected copies, and
 * those are the numbers worth reading.
 *
 * The rows are in turn order from the seat that was observed: the next
 * player, the one across, then the previous player. That is Mortal's order
 * and our own, and they agree.
 */

/** The thirty-four kinds, in the order the network writes them. */
export const KINDS = Object.freeze([
  ...Array.from({ length: 9 }, (_, i) => `${i + 1}m`),
  ...Array.from({ length: 9 }, (_, i) => `${i + 1}p`),
  ...Array.from({ length: 9 }, (_, i) => `${i + 1}s`),
  'E', 'S', 'W', 'N', 'P', 'F', 'C',
]);

export const POSITIONS = KINDS.length;
export const OPPONENTS = 3;

/** Named for the seat that was asked, not for the table. */
export const RELATIVE = Object.freeze(['Next', 'Across', 'Previous']);

function softmax(row) {
  const best = row.reduce((a, b) => (b > a ? b : a), -Infinity);
  if (!Number.isFinite(best)) throw new Error('The belief head returned invalid weights');
  const weights = row.map(value => Math.exp(value - best));
  const total = weights.reduce((a, b) => a + b, 0);
  return weights.map(value => value / total);
}

/**
 * Three rows of thirty-four chances, from the head's raw output.
 *
 * `held` says how many concealed tiles each opponent has, in the same
 * order; where it is given, each row also carries the expected number of
 * copies, which is what a person reads. A seat holding nothing — it has
 * won, or the hand is over — is left out.
 */
export function readBeliefs(hands, held = null) {
  if (!hands || hands.length !== OPPONENTS * POSITIONS) {
    throw new Error('The belief head returned the wrong shape');
  }
  return Array.from({ length: OPPONENTS }, (_, seat) => {
    const row = Array.from(hands.slice(seat * POSITIONS, (seat + 1) * POSITIONS));
    const chance = softmax(row);
    const tiles = held?.[seat] ?? null;
    return {
      seat,
      relative: RELATIVE[seat],
      held: tiles,
      chance,
      expected: tiles == null ? null : chance.map(value => value * tiles),
    };
  });
}

/**
 * The kinds a belief puts the most weight on, largest first, for showing a
 * few of them rather than all thirty-four.
 */
export function likeliest(belief, count = 6) {
  return belief.chance
    .map((chance, index) => ({ tile: KINDS[index], chance, expected: belief.expected?.[index] ?? null }))
    .sort((a, b) => b.chance - a.chance)
    .slice(0, count);
}

/**
 * The value head reads in the same units the critic was trained in: the
 * hand's return to the seat that was asked, where a full return is one
 * placement's worth. Reported as it is, not dressed up as a win chance,
 * because it is not one.
 */
export function readValue(value) {
  if (value == null) return null;
  const scalar = typeof value === 'number' ? value : value[0];
  return Number.isFinite(scalar) ? scalar : null;
}
