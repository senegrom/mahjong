/** The game review's reasoning about a discard (review-insight.js), asked of
 * Trained AI through the page's one policy worker.
 *
 * Every table is written by the engine as Mortal reads it (PhysicalAnalysis):
 * a table the rules refuse is answered as refused and never sent. The rest go
 * to the worker through analyzeBatch, and each answer is read back into the
 * engine's moves as the review reads its own (weightsByChoice), so that the
 * unedited table's answer can be set beside the review's weights move for
 * move.
 *
 * The worker answers its requests one at a time, in turn, and runs one it has
 * begun to its end, whatever is cancelled; a request's deadline runs from the
 * moment it is sent, its wait in the worker's queue included (policy.js): 20
 * seconds for a question of the hand's notes or a Trained opponent's move,
 * more for a batch. The review of the game so far and the hand's notes ask it
 * too, and the next deal's opponents once the player goes on, so the
 * reasoning keeps its passes out of the worker's queue:
 *  - A pass waits its turn in the page, where no deadline runs. It is sent
 *    only while no batch is with the worker, the review's or another pass,
 *    whichever question sent it (`batches`, countedBatches), and while the
 *    notes do not ask (`notes`); a question stopped while it waits sends
 *    nothing more.
 *  - So a pass goes to a worker with nothing to do, which begins it at once
 *    and would run it to its end whatever is cancelled: it is sent without
 *    the question's signal. It is no longer than the review's own batch,
 *    fewer on a device so slow that a question sent just behind it would run
 *    out of its time (passFor), and whatever is sent while it runs, the
 *    notes' first question or the next deal's first move, waits behind that
 *    one pass at most.
 *  - It holds a spell of asking of its own (`asking`), and the review of the
 *    rest of the game, given the gate of both spells (`anyGate`), waits at it
 *    as it waits for the notes. A question stopped while one of its passes is
 *    still with the worker keeps the spell until the worker has answered that
 *    pass, so the review never sends a batch to wait behind a pass nobody
 *    wants, its own deadline running.
 * The page sends the review's batches through countedBatches too, whose
 * spell a stopped batch keeps until the worker has answered it, and stops the
 * questions (their `signal`) wherever it stops the review for the next deal:
 * at the next hand, whose Trained opponents ask the same worker, and on
 * leaving Play.
 */
import { MORTAL_REACH, weightsByChoice } from './action-weights.js';
import { PASS, explainWhy, movesOf, whatWouldChangeItsMind } from './review-insight.js';
import { MatchSession } from './session.js';

const stopped = () => new DOMException('Asking stopped', 'AbortError');

/** How long a question sent just behind a pass may wait for it, the notes'
 * first question or the next deal's first move: either has 20 seconds in all
 * (policy.js), and this leaves it time to be answered itself. */
const WAIT_MS = 15000;

/** The positions in each pass on a device that takes `positionMs` to answer a
 * position (the review's own measure, MatchReview.positionMs): the review's
 * batch, or fewer where a question sent behind a whole batch would wait more
 * than WAIT_MS; the batch where nothing has been measured yet. */
export function passFor(positionMs) {
  return positionMs > 0 ? Math.max(1, Math.min(PASS, Math.floor(WAIT_MS / positionMs))) : PASS;
}

/** A table as the worker is asked it: Mortal's planes and mask, and how to
 * read an answer into the engine's moves, as the review reads its own
 * (weightsByChoice): Mortal has one kan, which the engine gives to the first
 * kan open, so a second kan open at once weighs nothing on either side. Or
 * why the rules refused the table. */
function written(PhysicalAnalysis, position) {
  let analysis;
  try {
    analysis = new PhysicalAnalysis(position);
    const planes = analysis.agent_observation_mortal(), mask = analysis.agent_mask_mortal(), choices = analysis.agent_choices();
    // What the reading needs of the engine, kept, since the analysis is freed
    // long before the answer comes back.
    const actions = new Map(choices.map(choice => [choice.index, analysis.mortal_action_of(choice.index)]));
    const owners = Array.from(mask, (open, action) => (open && action !== MORTAL_REACH ? analysis.agent_action_from_mortal(action, false) : -1));
    const engine = { mortal_action_of: index => actions.get(index) ?? -1 };
    return { planes, mask, read: weights => movesOf(weightsByChoice(engine, choices, weights, action => owners[action])) };
  } catch (error) {
    return { refused: String(error?.message ?? error) };
  } finally { analysis?.free(); }
}

/** Waits for `promise`, given up at once when `signal` stops the asking. */
function unlessStopped(promise, signal) {
  return new Promise((resolve, reject) => {
    const stop = () => reject(stopped());
    signal?.addEventListener('abort', stop, { once: true });
    promise.then(resolve, reject).finally(() => signal?.removeEventListener('abort', stop));
    if (signal?.aborted) stop();
  });
}

/** `analyzeBatch` with each batch it sends counted as a spell of `batches`
 * (Asking) until the worker has answered it or it has failed: what the page
 * sends the review's batches through, and the reasoning its passes, so that a
 * pass is sent only while no batch is with the worker. A batch is sent
 * without its signal: the worker runs one it has begun to its end whatever is
 * cancelled, and only its answer says that it is free, so a batch stopped by
 * its signal is let go at once and its spell kept until then. A review
 * stopped with its Stop button, the questions going on, then holds them back
 * as a running review would. One stopped before it is sent is not sent. The
 * batches it carries go to a worker with nothing to do, as the review's and
 * the passes wait at their gates, so none waits in its queue for a cancel to
 * drop. */
export function countedBatches(analyzeBatch, batches) {
  return (positions, signal) => {
    if (signal?.aborted) return Promise.reject(stopped());
    const end = batches.begin();
    let answer;
    try { answer = Promise.resolve(analyzeBatch(positions)); } catch (error) { answer = Promise.reject(error); }
    answer.then(end, end);
    return signal ? unlessStopped(answer, signal) : answer;
  };
}

/** Trained AI's answers to typed-in tables, `ask(positions, signal)` as
 * review-insight.js asks: the tables in passes of at most `size`, each sent
 * once `gate()` is open, every answer read into moves. The gate is asked
 * again each time it opens, and a pass is sent in the same turn as the gate
 * is found open, before anything else can look, so that of two questions
 * waiting at one gate only one goes. A pass is sent without the signal, the
 * worker running it to its end anyway, and `sent(pass)` hears each one as it
 * goes; the asker is let go at once when the signal stops it. */
export function trainedAsk({ PhysicalAnalysis, analyzeBatch, gate = () => null, size = PASS, sent = () => {} }) {
  return async (positions, signal) => {
    const answers = [];
    for (let start = 0; start < positions.length; start += size) {
      const tables = positions.slice(start, start + size).map(position => written(PhysicalAnalysis, position));
      const asked = tables.filter(table => !table.refused);
      let replies = [];
      if (asked.length) {
        for (let shut = gate(); shut; shut = gate()) await unlessStopped(shut, signal);
        if (signal?.aborted) throw stopped();
        const pass = Promise.resolve(analyzeBatch(asked.map(({ planes, mask }) => ({ planes, mask }))));
        sent(pass);
        replies = await unlessStopped(pass, signal);
        if (signal?.aborted) throw stopped();
        if (!Array.isArray(replies) || replies.length !== asked.length) throw new Error('The trained network returned the wrong number of answers');
      }
      let next = 0;
      for (const table of tables) answers.push(table.refused ? { refused: table.refused } : { moves: table.read(replies[next++].weights) });
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
 * at their first question and ended once they are over and the worker has
 * answered every pass they sent; and each pass waits until neither `notes`
 * nor `batches`, the spells of every batch with the worker (countedBatches),
 * is under way. The three must be spells apart: a search waiting for its own
 * spell to end would never ask. `positionMs()` is the review's time a
 * position, read as each begins, which sizes its passes (passFor). */
export function reviewInsight({ PhysicalAnalysis, review_facts, analyzeBatch, asking, notes, batches, positionMs = () => null }) {
  if (asking === notes) throw new Error('A search needs a spell of asking apart from the notes\'');
  if (!batches || batches === asking || batches === notes) throw new Error('The worker\'s batches need spells of their own (countedBatches)');
  const send = countedBatches(analyzeBatch, batches), gate = anyGate(notes, batches);
  const spelled = async work => {
    let end = null, out = 0, over = false;
    const close = () => { if (over && !out) end?.(); };
    const settled = () => { out--; close(); };
    const size = passFor(positionMs());
    const ask = trainedAsk({ PhysicalAnalysis, analyzeBatch: send, gate, size, sent: pass => { out++; pass.then(settled, settled); } });
    try {
      return await work((positions, signal) => { end ??= asking.begin(); return ask(positions, signal); }, size);
    } finally {
      over = true;
      close();
    }
  };
  return {
    why: (question, { signal } = {}) => spelled(spellAsk => explainWhy({ ...question, facts: review_facts, ask: spellAsk, signal })),
    search: (question, { signal, onProgress } = {}) => spelled((spellAsk, size) => whatWouldChangeItsMind({ ...question, facts: review_facts,
      ask: spellAsk, signal, onProgress, size })),
  };
}
