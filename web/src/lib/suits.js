/** Characters and circles renamed: Trained AI asks a question it hesitates
 * over of the renamed game too, and plays the mean of the two answers, as the
 * ruler's averaging player does (neural/suits.py, docs/RULER.md).
 *
 * Under our rules characters and circles are interchangeable: there are no
 * red fives, and All Green, the one yaku that names a suit, names bamboo. A
 * game with every 1m-9m named 1p-9p and every 1p-9p named 1m-9m is another
 * game, played the same way, so its best move is the renamed best move of the
 * other. The network knows that only as well as it learned it; asking both
 * namings takes the accident out of a decision.
 *
 * The renamed game is the engine's own encoding of it, every event renamed
 * as Mortal's augmentation renames it, not the game's planes permuted: the
 * efficiency lookahead breaks ties by tile index, so the two differ in about
 * half the questions. The engine writes it (the `*_renamed` observations);
 * its mask is the game's with Mortal's moves renamed (swapMoves).
 *
 * Only a question whose plain answer hesitates, its top weight below GATE,
 * is asked again, about one in six, each on its own answer, a reach's tile
 * included, and in a request of its own after the plain one. The answer is
 * then the softmax of the mean of the two answers' logits, the renamed one
 * renamed back, which the page works out from the two answers' weights
 * (average). Every place the page asks Trained AI asks so, so that it is one
 * player everywhere: the opponents, the adviser, the hand's notes, the game's
 * review and its reasoning. Its value and its belief about the other hands
 * stay the plain answer's. */

/** The plain answer's top weight below which a question is asked of the
 * renamed game too: neural/suits.py's GATE. */
export const GATE = 0.9;

/** Mortal's move `m` of a game is move `MOVES[m]` of the renamed game: the
 * discards of 1m-9m (0-8) and of 1p-9p (9-17) trade places, and so do those
 * of the red fives 5mr and 5pr (34, 35), which our tables never deal; bamboo,
 * the honours, 5sr and the moves that name no tile, riichi to pass, stay.
 * Its own inverse. */
export const MOVES = Object.freeze([
  ...Array.from({ length: 9 }, (_, k) => 9 + k), ...Array.from({ length: 9 }, (_, k) => k),
  ...Array.from({ length: 16 }, (_, k) => 18 + k), 35, 34, ...Array.from({ length: 10 }, (_, k) => 36 + k),
]);

/** Anything over Mortal's forty-six moves renamed: a game's mask as the
 * renamed game's, or the renamed game's weights as the game's. The same call
 * maps either way. */
export function swapMoves(values) {
  if (values?.length !== MOVES.length) throw new Error('The policy returned the wrong number of choices');
  return MOVES.map(move => values[move]);
}

/** Whether Trained AI hesitates over an answer, given its weights over the
 * legal moves: its top weight below GATE, strictly, as neural/suits.py has
 * it. An answer with one legal move never does. */
export function hesitates(weights) {
  if (!weights?.length) throw new Error('The trained network returned no weights');
  return Math.max(...weights) < GATE;
}

/** The answer of the mean of two answers' logits over the legal moves
 * (`mask`), the game's and the renamed game's renamed back, from the two
 * answers' weights: a move's weight is the square root of its two weights'
 * product, normalised, which is the softmax of the mean of its two logits,
 * each answer's own normalisation cancelling. Worked out in double precision;
 * of moves weighed alike the first is best, as numpy's argmax takes it. */
export function average(plain, renamed, mask) {
  if (plain?.length !== MOVES.length || renamed?.length !== MOVES.length || mask?.length !== MOVES.length) {
    throw new Error('The policy returned the wrong number of choices');
  }
  const weights = Array.from(mask, (open, move) => (open ? Math.sqrt(plain[move] * renamed[MOVES[move]]) : 0));
  let action = -1, total = 0;
  for (let move = 0; move < weights.length; move++) {
    if (!mask[move]) continue;
    total += weights[move];
    if (action < 0 || weights[move] > weights[action]) action = move;
  }
  if (!(total > 0) || !Number.isFinite(total)) throw new Error('The policy returned invalid weights');
  return { action, weights: weights.map(weight => weight / total) };
}

/** Trained AI's answers to `positions`, each `{ planes, mask, renamed }`, as
 * `ask(positions)` answers them in one request, in order, each as
 * analyzePolicy does (`{ action, weights, … }`): the plain answer where it is
 * sure enough, and where it hesitates the average of it and the renamed
 * game's, asked of `ask` in a request of its own, `renamed()` giving a
 * position's renamed planes only then. The plain answer's other fields, its
 * value and its belief, are kept. Once `signal` has stopped the asking the
 * renamed planes are not read: the engine they are read from may be gone. */
export async function averagedAnswers(ask, positions, signal) {
  if (!positions.length) return [];
  const answers = await ask(positions);
  const again = positions.flatMap((_, row) => (hesitates(answers[row].weights) ? [row] : []));
  if (!again.length) return answers;
  if (signal?.aborted) throw new DOMException('Asking stopped', 'AbortError');
  const renamed = await ask(again.map(row => ({ planes: positions[row].renamed(), mask: swapMoves(positions[row].mask) })));
  return answers.map((answer, row) => {
    const at = again.indexOf(row);
    return at < 0 ? answer : { ...answer, ...average(answer.weights, renamed[at].weights, positions[row].mask) };
  });
}

/** Trained AI's answer to one position, `{ planes, mask, renamed }`, asked
 * as averagedAnswers asks: `ask(planes, mask, signal)` is analyzePolicy. */
export async function averagedAnswer(ask, position, signal) {
  const [answer] = await averagedAnswers(async ([asked]) => [await ask(asked.planes, asked.mask, signal)], [position], signal);
  return answer;
}
