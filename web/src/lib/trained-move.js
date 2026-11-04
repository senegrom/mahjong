import { MORTAL_REACH } from './action-weights.js';
import { averagedAnswers } from './suits.js';

/** Trained AI's move: the network's answer averaged over the suits
 * (suits.js), then the last hand's two rules, taken greedily, the best move
 * played, as the ruler plays them. Every place the page asks Trained AI for
 * a move asks it here: Play's opponents and Watch's other seats
 * (session.js), Watch's followed seat and the advisers (agents.js), and the
 * hand's notes and the game's review (review-policy.js).
 *
 * South 4 is the last hand: a win by anyone but the dealer ends the game,
 * and the places are then the points. Decline-last (neural/late_rules.py):
 * where the answer's best move is a win that would leave the player alone in
 * last, the win is declined, a ron for the pass whatever ranks second, a
 * self-draw for the best other move of the same answer. All-last dama
 * (neural/all_last.py): where the move is then a riichi, the tile its reach
 * would throw is asked, from a preview of the reach, and where that tile,
 * thrown undeclared, keeps the player first through every win the hand
 * could then make, it is thrown and nothing is declared. Decline comes
 * first because a declined self-draw can leave a riichi, which dama must
 * judge too.
 *
 * Which wins leave the player last or keep them first is the engine's to
 * say (riichi_core::all_last), as codes asked lazily, as Python asks them:
 * decline's only where the answer's best is a win, dama's only where the
 * move is a riichi. The engine answers nought outside South 4, for the
 * dealer and for a seat that does not owe the decision, so nothing changes
 * there. Nothing is asked again once a win is struck: the suit average's
 * gate is read once, on the plain answer, and the move is the best of what
 * the average left, as neural/late_rules.py strikes the win from the values
 * neural/suits.py returned; the averaged weights order the moves as the mean
 * logits do. Codes that no legal move can have would be the engine's fault:
 * the move is refused (RULE_REFUSED) and nothing is played, as Python
 * raises. */

/** Mortal's moves the rules name besides riichi (MORTAL_REACH, 37, kept in
 * action-weights.js): a win, by self-draw or ron, and the pass. */
export const MORTAL_AGARI = 43, MORTAL_PASS = 45;
/** The engine's codes by number, riichi_core::all_last::Dama, one a tile
 * kind, and ::Decline, one a win, as neural/all_last.py and
 * neural/late_rules.py number them; nought is not asked. Dama: the tile,
 * thrown undeclared, keeps the player first through every win; a wait with
 * no yaku without riichi; a win that leaves another player level or ahead;
 * no live wait; and a tile held whose riichi is not legal. Decline: the win
 * leaves the player alone in last, and it does not. */
export const DAMA = Object.freeze({ ABSENT: 0, KEEPS_FIRST: 1, NO_YAKU: 2, SHORT: 3, DEAD: 4, NOT_WAITING: 5 });
export const DECLINE = Object.freeze({ ABSENT: 0, LAST: 1, NOT_LAST: 2 });
/** The rules' names, as the ruler and the learner give them. */
export const DAMA_RULE = 'all-last-dama', DECLINE_RULE = 'decline-last';
/** What a refused move says, on its own or after the review's "Review
 * stopped:". The rule, its codes and the reach's tile go in the error's
 * cause, for a developer; no code reaches the message. */
export const RULE_REFUSED = 'A last-hand rule could not be applied';
/** The words for a move a rule made: `tag`, in place of its weight, and
 * each rule's line, which is the tag's title, the note in Trained AI's view
 * and what a screen reader hears (ruleLine, ruleSaid). "It" is the seat
 * Trained AI plays or advises. */
export const RULE_WORDS = Object.freeze({
  tag: 'by rule',
  [DECLINE_RULE]: 'Last hand: this win would leave it last',
  [DAMA_RULE]: 'Last hand: every win keeps it first, so no riichi',
});

/** The best of the moves `mask` opens by `weights`, `struck` left out, the
 * first of moves weighed alike, as numpy's argmax takes it; -1 where none
 * is open. */
export function bestOpen(weights, mask, struck = -1) {
  let best = -1;
  for (let move = 0; move < mask.length; move++) {
    if (!mask[move] || move === struck) continue;
    if (best < 0 || weights[move] > weights[best]) best = move;
  }
  return best;
}

/** The answer the network would give a position with one legal move, or
 * null: the weights over one move are that move's whole. */
export function forcedAnswer(mask) {
  const legal = Array.from(mask, allowed => (allowed ? 1 : 0));
  return legal.reduce((sum, value) => sum + value, 0) === 1
    ? { action: legal.indexOf(1), weights: legal } : null;
}

// A table is one decision's questions: the first question as suits.js takes
// a position, `{ planes, mask, renamed }`, its planes and mask read as the
// table is built; and, read only when asked, `afterReach()`, the position a
// reach would leave, previewed before anything is recorded, as another such
// position, and the seat's codes, `dama()`, a byte a tile kind, and
// `decline()`, a number. Each builder names one engine's questions.

/** The questions of the Trained opponent that owes a decision, of a Game. */
export const opponentTable = game => ({
  planes: game.opponent_observation_mortal(), mask: game.opponent_mask_mortal(),
  renamed: () => game.opponent_observation_mortal_renamed(),
  afterReach: () => ({ planes: game.opponent_observation_after_reach(), mask: game.opponent_mask_after_reach(),
    renamed: () => game.opponent_observation_after_reach_renamed() }),
  dama: () => game.opponent_all_last_dama(),
  decline: () => game.opponent_all_last_decline(),
});

/** The questions of the seat Watch follows, of a Game, or of the seat a
 * typed-in table asks about, of a PhysicalAnalysis, which name them alike. */
export const agentTable = engine => ({
  planes: engine.agent_observation_mortal(), mask: engine.agent_mask_mortal(),
  renamed: () => engine.agent_observation_mortal_renamed(),
  afterReach: () => ({ planes: engine.agent_observation_after_reach(), mask: engine.agent_mask_after_reach(),
    renamed: () => engine.agent_observation_after_reach_renamed() }),
  dama: () => engine.agent_all_last_dama(),
  decline: () => engine.agent_all_last_decline(),
});

/** The questions of the `index`th decision a Game's review recorded. */
export const reviewTable = (game, index) => ({
  planes: game.review_observation_mortal(index), mask: game.review_mask_mortal(index),
  renamed: () => game.review_observation_mortal_renamed(index),
  afterReach: () => ({ planes: game.review_observation_after_reach(index), mask: game.review_mask_after_reach(index),
    renamed: () => game.review_observation_after_reach_renamed(index) }),
  dama: () => game.review_all_last_dama(index),
  decline: () => game.review_all_last_decline(index),
});

const stopped = signal => {
  if (signal?.aborted) throw new DOMException('Asking stopped', 'AbortError');
};

/** A move refused: the engine's codes and the legal moves disagree. */
const refused = cause => new Error(RULE_REFUSED, { cause });

/** Trained AI's answers to `positions`, in order: with `skipForced`, one
 * with a single legal move without the network (forcedAnswer); the rest in
 * one request of `ask`, and those it hesitates over again, renamed, in
 * another (averagedAnswers). */
async function answered(ask, positions, skipForced, signal) {
  const answers = positions.map(position => (skipForced ? forcedAnswer(position.mask) : null));
  const asked = positions.filter((_, row) => !answers[row]);
  if (asked.length) {
    const replies = await averagedAnswers(ask, asked, signal);
    let next = 0;
    for (let row = 0; row < answers.length; row++) answers[row] ??= replies[next++];
  }
  return answers;
}

/** Decline-last at a table whose first answer is `first`: the move it
 * leaves, and whether it declined a win. Its code is read only where the
 * answer's best is a win. */
function declined(table, first) {
  if (first.action !== MORTAL_AGARI) return { action: first.action, fired: false };
  const code = table.decline();
  if (code === DECLINE.ABSENT || code === DECLINE.NOT_LAST) return { action: MORTAL_AGARI, fired: false };
  if (code !== DECLINE.LAST) throw refused({ rule: DECLINE_RULE, decline: code });
  // A claim window, where the pass is always open: the pass, whatever ranks
  // second, a call included.
  if (table.mask[MORTAL_PASS]) return { action: MORTAL_PASS, fired: true };
  // The player's own turn, where a discard always is: the best other move of
  // the same weights, the win struck. Nothing is asked again, and whether to
  // average was settled on the plain answer, before the strike.
  const action = bestOpen(first.weights, table.mask, MORTAL_AGARI);
  if (action < 0) throw refused({ rule: DECLINE_RULE, decline: code });
  return { action, fired: true };
}

/** All-last dama at a table whose move is a riichi, `tile` the best tile of
 * the reach's own answer: the moves to play, and the codes where the rule
 * throws the tile undeclared, else null. Codes all nought were not asked:
 * the reach stands. */
function dama(table, tile) {
  const codes = Array.from(table.dama());
  if (!codes.some(Boolean)) return { moves: [MORTAL_REACH, tile], codes: null };
  const code = codes[tile];
  if (code === DAMA.KEEPS_FIRST) {
    // The tile's plain discard, read without the reach.
    if (!table.mask[tile]) throw refused({ rule: DAMA_RULE, dama: codes, tile });
    return { moves: [tile], codes };
  }
  if (code === DAMA.NO_YAKU || code === DAMA.SHORT || code === DAMA.DEAD) return { moves: [MORTAL_REACH, tile], codes: null };
  throw refused({ rule: DAMA_RULE, dama: codes, tile });
}

/** Trained AI's moves at `tables` (see above), asked through
 * `ask(positions)`, which answers a request of positions in order, each as
 * analyzePolicy answers it: a review's batches, or one position at a time.
 *
 * The first questions are asked in one request, and those Trained AI
 * hesitates over again, renamed, in another; with `skipForced` a question
 * with one legal move is answered without the network. Then decline-last at
 * each table whose answer is a win; then the reach's tile at each table
 * whose move is a riichi, the network's or one a declined self-draw left,
 * asked the same way, in two requests more at most; then all-last dama at
 * those. Each table's answer:
 *
 * - `first`, its first answer, averaged where it hesitated; its value and
 *   its belief are the plain answer's;
 * - `second`, the answer about the reach's tile, where one was asked, else
 *   null;
 * - `moves`, Mortal's moves to play, in order: one, or a riichi and its tile;
 * - `final`, `{ action, afterReach }`, the move as the engine translates it
 *   (`*_action_from_mortal`): the tile and true for a reach, the tile and
 *   false for a tile thrown undeclared, the move and false otherwise;
 * - `ruled`, null where no rule changed the move, else `{ rules, dama }`:
 *   the rules that did, in the order they did, and the dama codes where
 *   dama did, else null.
 *
 * Each code is read at most once, and every lazy question only right after
 * the check of `signal` that follows the answers before it, so nothing is
 * read from an engine freed while the network was asked. */
export async function trainedMoves(ask, tables, { signal, skipForced = false } = {}) {
  const firsts = await answered(ask, tables, skipForced, signal);
  stopped(signal);
  const rows = tables.map((table, row) => {
    const { action, fired } = declined(table, firsts[row]);
    return { first: firsts[row], second: null, action, rules: fired ? [DECLINE_RULE] : [] };
  });
  const reaching = rows.flatMap((row, at) => (row.action === MORTAL_REACH ? [at] : []));
  if (reaching.length) {
    const seconds = await answered(ask, reaching.map(at => tables[at].afterReach()), skipForced, signal);
    stopped(signal);
    reaching.forEach((at, slot) => { rows[at].second = seconds[slot]; });
  }
  return rows.map(({ first, second, action, rules }, at) => {
    if (action !== MORTAL_REACH) {
      return { first, second, moves: [action], final: { action, afterReach: false },
        ruled: rules.length ? { rules, dama: null } : null };
    }
    const tile = second.action;
    const { moves, codes } = dama(tables[at], tile);
    const ruledBy = codes ? [...rules, DAMA_RULE] : rules;
    return { first, second, moves, final: { action: tile, afterReach: moves.length === 2 },
      ruled: ruledBy.length ? { rules: ruledBy, dama: codes } : null };
  });
}

/** Trained AI's move at one table, asked through `analyze(planes, mask,
 * signal)`, which is analyzePolicy: trainedMoves's answer for it. */
export const trainedMove = async (analyze, table, signal) =>
  (await trainedMoves(async ([position]) => [await analyze(position.planes, position.mask, signal)], [table], { signal }))[0];

/** Whether a rule would decide a table whose answer's best open move is
 * `action`, the table being `engine`'s seat (agentTable's engine), asked
 * without a preview: a riichi any of whose tiles keeps the player first, or
 * a win that would leave them last. The reasoning edits tables and judges
 * them by the network's own answer, which is Trained AI's move only where
 * this does not hold. It asks no forward: the codes are the engine's. */
export function ruleDecides(action, engine) {
  if (action === MORTAL_REACH) return Array.from(engine.agent_all_last_dama()).includes(DAMA.KEEPS_FIRST);
  if (action === MORTAL_AGARI) return engine.agent_all_last_decline() === DECLINE.LAST;
  return false;
}

/** The line of a move the rules in `ruled.rules` made (trainedMoves's
 * `ruled`): one rule's, or, where both did, decline's and then dama's. */
export function ruleLine(ruled) {
  return [DECLINE_RULE, DAMA_RULE].filter(rule => ruled.rules.includes(rule)).map(rule => RULE_WORDS[rule]).join(' · ');
}

/** What a screen reader hears in place of a ruled move's weight. */
export function ruleSaid(ruled) {
  const line = ruleLine(ruled);
  return `, ${RULE_WORDS.tag} (${line.charAt(0).toLowerCase()}${line.slice(1)})`;
}
