import { applyEdits, whatWouldChangeItsMind } from '../../src/lib/review-insight.js';

/** A search's findings, or a Why line, as lines of plain text, the way the
 * tests read a panel: the lead the panel sets in bold with what follows it,
 * each item on a line of its own, and what moved nothing. The page lays the
 * parts out itself, the lead apart, so this is the tests' alone. */
export function saidLines(said) {
  if ('label' in said) return [`${said.label}: ${said.text}`];
  return [[said.lead, said.detail].filter(Boolean).join(' '), ...said.items.map(item => `- ${item}`), ...(said.still ? [said.still] : [])];
}

/** Weights standing in for Trained AI's, for a test that asks it about tables
 * no run recorded: the player's tile `P` and Trained AI's `A` the only moves
 * it weighs. They set only the figures a panel gives, never which change it
 * names or what it says a change did, which is what such a test reads. */
export const standIn = (P, A, played, preferred) => [{ kind: 'discard', tile: P, weight: played }, { kind: 'discard', tile: A, weight: preferred }];

/** A question about a table, with Trained AI's weights stood in for: it
 * throws `A` over the player's `P` by four to one. */
export function standInQuestion(position, P, A) {
  return { position, played: { kind: 'discard', tile: P }, preferred: { kind: 'discard', tile: A }, weights: standIn(P, A, 0.2, 0.8) };
}

/** What the search finds for `question` where Trained AI, stood in for,
 * throws the player's tile on the table that `edits` make together and on no
 * other: each of two changes alone brings it nearer than any other change,
 * so that the two are the pair tried, and every other table is answered as
 * the unedited one. A table the rules refuse is answered as refused, as the
 * page answers it without asking (trainedAsk). `facts` is review_facts,
 * which refuses what the rules refuse. */
export function searchWhereOnly(question, facts, ...edits) {
  const P = question.played.tile, A = question.preferred.tile, key = table => JSON.stringify(table);
  const answers = new Map();
  if (edits.length > 1) for (const edit of edits) answers.set(key(applyEdits(question.position, edit)), standIn(P, A, 0.4, 0.6));
  for (const order of [edits, [...edits].reverse()]) answers.set(key(applyEdits(question.position, ...order)), standIn(P, A, 0.9, 0.1));
  const answer = table => {
    try { facts(table); } catch (error) { return { refused: String(error?.message ?? error) }; }
    return { moves: answers.get(key(table)) ?? question.weights };
  };
  return whatWouldChangeItsMind({ ...question, facts, ask: async positions => positions.map(answer) });
}

/** The words in brackets in which a panel's line says what its change did, or
 * null where it says nothing of it. */
export const saidOfChange = line => line.match(/\(([^()]*)\): [^()]*$/)?.[1] ?? null;
