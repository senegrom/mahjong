import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { eachDecision, reviewCheckGame } from './review-check-game.js';
import { applyEdits, sameMove, scoreEdit, singleEdits } from '../../src/lib/review-insight.js';

/** The "What would change its mind?" prototype's eight questions, as it ran
 * them on Trained AI 0.1.1110 (insight/prototype/out/results-run1.json):
 * six decisions of the review check's game and the owner's why-7 table from
 * Agent Watch, asked twice. Each has the two tiles in question, the review's
 * own weights, and Trained AI's answer to every table version 1 asks, by the
 * change's words: `[words, played, preferred, top kind, top tile, top weight,
 * margin, flips, ideas, planes]`, `planes` being the digest of the planes the
 * table was asked on (planesDigest). The question's own `planes` are its
 * unedited table's, the review's, and a threshold's are by score step. */
export const RECORD = JSON.parse(readFileSync(new URL('./review-insight-answers.json', import.meta.url), 'utf8'));

/** A short digest of a table as Mortal reads it, its planes and its mask:
 * what a recorded answer was asked on, to set beside what the page's engine
 * writes for the same change. */
export function planesDigest(planes, mask) {
  return createHash('sha256').update(Uint8Array.from(mask, open => (open ? 1 : 0)))
    .update(new Uint8Array(Float32Array.from(planes).buffer)).digest('hex').slice(0, 16);
}

/** The one recorded table the page writes otherwise, whose answer is kept.
 * In this pair of South 3 #4 the prototype counted the points on the table
 * as they stood before North's riichi gave back its thousand, so it left
 * South a thousand short (23,300 where the page, counting them after, has
 * 24,300); the planes differ in rows 9 and 10 alone. */
const WRITTEN_OTHERWISE = new Set(['North had not declared riichi, and you had had 8,000 points fewer, taken evenly from the others']);

/** Each recorded question as the review would put it: the table as the
 * engine writes the decision (the typed-in table for why-7), the player's
 * discard and Trained AI's, and the review's own weights as moves; and
 * `written(table)`, the digest of the planes the page's engine writes for a
 * table, worked out once a table. */
export function recordedQuestions(Game, PhysicalAnalysis) {
  const tables = new Map(), digests = new Map();
  eachDecision(Game, reviewCheckGame(Game), ({ hand, index, position }) => tables.set(`${hand.index}-${index}`, position));
  const written = table => {
    const key = JSON.stringify(table);
    if (!digests.has(key)) {
      const analysis = new PhysicalAnalysis(table);
      try { digests.set(key, planesDigest(analysis.agent_observation_mortal(), analysis.agent_mask_mortal())); } finally { analysis.free(); }
    }
    return digests.get(key);
  };
  return RECORD.questions.map(record => ({
    record, id: record.id, title: record.title, written,
    position: record.position ?? tables.get(`${record.hand}-${record.decision}`),
    played: { kind: 'discard', tile: record.played }, preferred: { kind: 'discard', tile: record.preferred },
    weights: [...record.weights.map(([tile, weight]) => ({ kind: 'discard', tile, weight })),
      ...(record.riichi == null ? [] : [{ kind: 'riichi', tile: null, weight: record.riichi }])],
  }));
}

/** One recorded answer as Trained AI's moves. The record keeps the weights a
 * panel reads, the player's tile, Trained AI's and the one now rated
 * highest, and the margin of the player's tile over the best other move.
 * Where the player's tile now leads and that best other is not Trained AI's
 * tile, the margin gives its weight but not its tile, so it is given to
 * another tile of the hand; no other move carries weight, which keeps every
 * judgement the record makes. */
function movesOf([, played, preferred, kind, tile, weight, margin, flips], question, table) {
  const P = question.played.tile, A = question.preferred.tile;
  const moves = [{ kind: 'discard', tile: P, weight: played }];
  if (preferred != null) moves.push({ kind: 'discard', tile: A, weight: preferred });
  const top = { kind, tile, weight };
  if (!moves.some(move => sameMove(move, top))) moves.push(top);
  if (flips && (preferred == null || Math.abs(Math.log(played / preferred) - margin) > 6e-4)) {
    const other = table.players[table.seat].hand.find(held => held !== P && held !== A);
    moves.push({ kind: 'discard', tile: other, weight: played / Math.exp(margin) });
  }
  return { moves };
}

/** Trained AI as the recorded run answered, for one question: `ask` answers
 * the unedited table with the review's own weights, and every changed table
 * the record has with its recorded answer, the table made again from the
 * change's words. A score step the halving asked between the recorded steps
 * is answered as the step at the threshold was where it reaches it, and as
 * the unedited table where it falls short, which is all the halving reads:
 * the run found that it flipped there and not before. A table is answered
 * only where the page's engine writes the planes the run asked (`planes`);
 * one written otherwise is noted in `otherwise`, by its words, and like a
 * table the run never asked is answered as refused and noted in
 * `unanswered`. `asked` notes how many tables each pass asked; `answers`
 * holds every table answered, by its JSON. */
export function recordedAsk(question) {
  const { position, played, preferred, record, written } = question;
  const recorded = new Map(record.answers.map(entry => [entry[0], entry]));
  const known = new Map(), otherwise = [];
  // `answer` is a recorded answer, or the unedited table's moves.
  const learn = (table, planes, answer, words) => {
    if (written(table) !== planes && !WRITTEN_OTHERWISE.has(words)) otherwise.push(words);
    else known.set(JSON.stringify(table), Array.isArray(answer) ? movesOf(answer, question, table) : answer);
  };
  const singles = new Map(singleEdits(position, { played, preferred }).map(edit => [edit.words, edit]));
  for (const [words, entry] of recorded) {
    if (singles.has(words)) { learn(applyEdits(position, singles.get(words)), entry[9], entry, words); continue; }
    // A pair's words are its two changes', joined by ", and ".
    for (let at = words.indexOf(', and '); at >= 0; at = words.indexOf(', and ', at + 1)) {
      const [first, second] = [singles.get(words.slice(0, at)), singles.get(words.slice(at + 6))];
      if (first && second) { learn(applyEdits(position, first, second), entry[9], entry, words); break; }
    }
  }
  for (const { at, answer, planes } of record.thresholds) {
    for (let step = 1000; step <= 16000; step += 1000) {
      const edit = scoreEdit(position, step * Math.sign(at));
      if (recorded.has(edit.words)) continue;
      learn(applyEdits(position, edit), planes[step], step >= Math.abs(at) ? answer : { moves: question.weights }, edit.words);
    }
  }
  // The unedited table last, so that no change recorded as asking it again
  // answers for it.
  learn(position, record.planes, { moves: question.weights }, 'the unedited table');
  const asked = [], unanswered = [];
  const ask = async positions => {
    asked.push(positions.length);
    return positions.map(table => {
      const answer = known.get(JSON.stringify(table));
      if (!answer) unanswered.push(table);
      return answer ?? { refused: 'Not asked in the recorded run' };
    });
  };
  return { ask, asked, unanswered, answers: known, otherwise };
}
