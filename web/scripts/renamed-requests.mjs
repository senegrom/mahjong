/** What the browser checks that intercept Trained AI's requests see of the
 * renamed game (src/lib/suits.js): that a question Trained AI hesitates over
 * is asked again of the renamed game, and that what is sent as the renamed
 * game is the renamed game. Its planes must be the question's own planes
 * with characters and circles renamed on every row but the efficiency
 * lookahead's (tests/fixtures/renamed-planes.js), and not the question's own
 * planes; its moves the question's renamed.
 *
 * `watchRenamedRequests(page)` installs `window.renamedRequests` in every
 * document the page loads, before the check's own scripts. The check's
 * Worker tells it of each request it lets through (`posted(data)`) and each
 * answer (`answered(data)`). Questions asked one at a time, as an opponent,
 * the adviser and the hand's notes ask them, are followed by their renamed
 * games at once: after an answer that hesitates, the next question must be
 * its renamed game, and `renamedRequests(page)` says how many were checked
 * so, what was wrong with any, and whether one is still awaited. Batches,
 * which the game's review and its reasoning send among each other, are not
 * paired here: `prints(planes, count)` gives each position of a batch three
 * digests to pair them by, `full` of its planes, `exact` of its rows but the
 * lookahead's, and `renamedExact` of those rows renamed, which a renamed
 * position's `exact` must equal. */
import { GATE, MOVES } from '../src/lib/suits.js';
import { EXACT, KINDS, TILE_ROWS } from '../tests/fixtures/renamed-planes.js';

const TABLES = { exact: [...EXACT], kinds: [...KINDS], rows: [...TILE_ROWS], gate: GATE, moves: [...MOVES] };

function install({ exact, kinds, rows, gate, moves }) {
  const TILES = 34;
  const bits = planes => new Uint32Array(Float32Array.from(planes).buffer);
  const digest = (words, at) => {
    let hash = 2166136261;
    for (const index of at) hash = Math.imul(hash ^ words[index], 16777619) >>> 0;
    return hash;
  };
  const exactAt = exact.flatMap(row => Array.from({ length: TILES }, (_, tile) => row * TILES + tile));
  const renamedAt = exact.flatMap(row => kinds.map(kind => rows[row] * TILES + kind));
  /** What is wrong with `asked` as the renamed game of `plain`, or null. */
  const fault = (plain, asked) => {
    if (asked.mask.some((open, move) => open !== plain.mask[moves[move]])) return 'its moves are not the question\'s renamed';
    for (let at = 0; at < exactAt.length; at++) {
      if (asked.words[exactAt[at]] !== plain.words[renamedAt[at]]) return `its planes are not the question's renamed, row ${exact[Math.floor(at / TILES)]}`;
    }
    return asked.words.some((word, at) => word !== plain.words[at]) ? null : 'its planes are the question\'s own';
  };
  const state = { checked: 0, faults: [], hesitant: null, sent: new Map() };
  window.renamedRequests = {
    posted(data) {
      if (!data?.planes || data.count) return;
      const asked = { words: bits(data.planes), mask: Array.from(data.mask, open => (open ? 1 : 0)) };
      if (state.hesitant) {
        const problem = fault(state.hesitant, asked);
        if (problem) state.faults.push(problem); else state.checked++;
        state.hesitant = null;
        state.sent.set(data.id, null);
      } else state.sent.set(data.id, asked);
    },
    answered(data) {
      // The worker's notes about a request come before its answer.
      if (data?.progress !== undefined || !state.sent.has(data?.id)) return;
      const asked = state.sent.get(data.id);
      state.sent.delete(data.id);
      const weights = data.analysis?.weights;
      if (asked && weights && Math.max(...weights) < gate) state.hesitant = asked;
    },
    prints(planes, count = 1) {
      const words = bits(planes), width = words.length / count;
      return Array.from({ length: count }, (_, row) => {
        const own = words.subarray(row * width, (row + 1) * width);
        return { full: digest(own, own.keys()), exact: digest(own, exactAt), renamedExact: digest(own, renamedAt) };
      });
    },
    report: () => ({ checked: state.checked, faults: [...state.faults], awaited: Boolean(state.hesitant) }),
  };
}

/** Installs the watch in every document `page` loads from now on. */
export const watchRenamedRequests = page => page.evaluateOnNewDocument(install, TABLES);

/** What the watch has seen in the page's document: how many renamed
 * questions it checked, what was wrong with any, and whether a question
 * that hesitated still awaits its renamed game. */
export const renamedRequests = page => page.evaluate(() => window.renamedRequests.report());
