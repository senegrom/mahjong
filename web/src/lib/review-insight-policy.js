/** The game review's reasoning about a discard (review-insight.js), asked of
 * Trained AI through the page's one policy worker.
 *
 * Every table is written by the engine as Mortal reads it (PhysicalAnalysis):
 * a table the rules refuse is answered as refused and never sent. The rest go
 * to the worker through analyzeBatch, in passes of 32, which fit its memory
 * ceiling, and each answer is read back into the engine's moves, a riichi
 * being one move whichever tile it names.
 *
 * The worker answers its requests in turn, and the review of the game so far
 * and the hand's own notes ask it too. A search holds a spell of asking of
 * its own (`asking`); the review of the rest of the game, given the gate of
 * both spells (`anyGate`), waits at it as it waits for the notes. Before each
 * pass the search waits in turn while the notes ask (`notes`), so that of the
 * notes' questions only one asked while a pass is under way waits behind it:
 * at the review's slow threshold, half a second a position, a pass of 32
 * takes 16 seconds, inside that question's 20.
 */
import { MORTAL_REACH } from './action-weights.js';
import { PASS, explainWhy, movesOf, whatWouldChangeItsMind } from './review-insight.js';
import { MatchSession } from './session.js';

const stopped = () => new DOMException('Search stopped', 'AbortError');

/** A table as the worker is asked it: Mortal's planes and mask, and the
 * engine's move for each of Mortal's open moves. Or why the rules refused it. */
function written(PhysicalAnalysis, position) {
  let analysis;
  try {
    analysis = new PhysicalAnalysis(position);
    const planes = analysis.agent_observation_mortal(), mask = analysis.agent_mask_mortal(), choices = analysis.agent_choices();
    const moves = Array.from(mask, (open, action) => {
      if (!open) return null;
      if (action === MORTAL_REACH) return { kind: 'riichi', tile: null };
      const choice = choices.find(entry => entry.index === analysis.agent_action_from_mortal(action, false));
      return choice ? { kind: choice.kind, tile: choice.tile ?? null } : null;
    });
    return { planes, mask, moves };
  } catch (error) {
    return { refused: String(error?.message ?? error) };
  } finally { analysis?.free(); }
}

/** Waits while `gate()` answers a promise, as the review waits at its own:
 * asked again each time it opens, and given up at once when `signal` stops
 * the search. */
async function passGate(gate, signal) {
  for (let shut = gate(); shut; shut = gate()) {
    await new Promise((resolve, reject) => {
      const stop = () => reject(stopped());
      if (signal?.aborted) return stop();
      signal?.addEventListener('abort', stop, { once: true });
      shut.then(resolve, reject).finally(() => signal?.removeEventListener('abort', stop));
    });
  }
}

/** Trained AI's answers to typed-in tables, `ask(positions, signal)` as
 * review-insight.js asks: the tables in passes of at most 32, each sent once
 * `gate()` is open, every answer read into moves. */
export function trainedAsk({ PhysicalAnalysis, analyzeBatch, gate = () => null }) {
  return async (positions, signal) => {
    const answers = [];
    for (let start = 0; start < positions.length; start += PASS) {
      const tables = positions.slice(start, start + PASS).map(position => written(PhysicalAnalysis, position));
      const sent = tables.filter(table => !table.refused);
      let replies = [];
      if (sent.length) {
        await passGate(gate, signal);
        replies = await analyzeBatch(sent.map(({ planes, mask }) => ({ planes, mask })), signal);
        if (signal?.aborted) throw stopped();
        if (!Array.isArray(replies) || replies.length !== sent.length) throw new Error('The trained network returned the wrong number of answers');
      }
      let next = 0;
      for (const table of tables) {
        if (table.refused) { answers.push({ refused: table.refused }); continue; }
        const { weights } = replies[next++];
        answers.push({ moves: table.moves.flatMap((move, action) => (move ? [{ ...move, weight: weights[action] }] : [])) });
      }
    }
    return answers;
  };
}

/** The gate of several spells of asking: shut while any of them is under
 * way, the first one's answered first. */
export const anyGate = (...askings) => () => askings.map(asking => asking.gate()).find(Boolean) ?? null;

/** The table at each of a reviewed hand's decisions, as the engine writes it
 * (Game.review_position), from the match the review replays: the saved
 * match's commands played again, from its first deal to the hand's end, on
 * an engine of its own, so the review's own replay is left where it is. */
export function handTables(Game, snapshot, hand) {
  const match = new MatchSession(Game, snapshot.seed, snapshot.opponents);
  try {
    match.advance(false);
    for (const command of snapshot.commands.slice(0, hand.end)) {
      match.apply(command);
      match.advance(false);
    }
    return hand.frames.map((_, index) => match.engine.review_position(index));
  } finally { match.dispose(); }
}

/** What a reviewed decision asks: its table (`handTables`), the player's move
 * and Trained AI's choice, and the review's own weights. */
export function questionOf(decision, position) {
  return { position, played: { kind: decision.played_kind, tile: decision.played_tile ?? null },
    preferred: { kind: decision.advised_kind, tile: decision.advised_tile ?? null }, weights: movesOf(decision.weighted) };
}

/** The review's reasoning about a decision's question (`questionOf`), asked
 * of the page's worker: `why(question)` for the Why line and
 * `search(question)` for what would change Trained AI's mind, each stopped by
 * its `signal`. Whatever they ask, they ask inside a spell of `asking`, begun
 * at their first question and ended however they end, and before each pass
 * they wait while `notes` is under way. The two must be spells apart: a
 * search waiting for its own spell to end would never ask. */
export function reviewInsight({ PhysicalAnalysis, review_facts, analyzeBatch, asking, notes }) {
  if (asking === notes) throw new Error('A search needs a spell of asking apart from the notes\'');
  const ask = trainedAsk({ PhysicalAnalysis, analyzeBatch, gate: () => notes.gate() });
  const spelled = async work => {
    let end = null;
    try {
      return await work((positions, signal) => { end ??= asking.begin(); return ask(positions, signal); });
    } finally { end?.(); }
  };
  return {
    why: (question, { signal } = {}) => spelled(spellAsk => explainWhy({ ...question, facts: review_facts, ask: spellAsk, signal })),
    search: (question, { signal, onProgress } = {}) => spelled(spellAsk => whatWouldChangeItsMind({ ...question, facts: review_facts,
      ask: spellAsk, signal, onProgress })),
  };
}
