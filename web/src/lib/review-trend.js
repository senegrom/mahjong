/** How often the player has matched the trained network, game by game, kept
 * in this browser only. It is a convenience: a browser that keeps nothing,
 * or refuses to, still reviews every game, and shows no trend. A game is
 * entered once, when its whole review is done, and entered again only to
 * replace itself.
 *
 * Each game names the version of the network that judged it. A player can
 * pick another version, which judges differently, so a game judged again by
 * another version is entered beside its first entry, not over it, and a
 * figure is compared only with games the same version judged. */
export const TREND_KEY = 'riichi.review-trend.v1';
const VERSION = 1;
/** The games kept, the newest last. */
export const TREND_GAMES = 50;
/** The games an average is taken over. */
export const TREND_WINDOW = 10;

const entry = value => value && typeof value === 'object' && typeof value.id === 'string' && typeof value.network === 'string'
  && Number.isSafeInteger(value.at) && Number.isSafeInteger(value.matched) && Number.isSafeInteger(value.decisions)
  && value.decisions > 0 && value.matched >= 0 && value.matched <= value.decisions;

/** The games recorded so far, oldest first; none when storage cannot be read. */
export function readTrend(storage) {
  try {
    const value = JSON.parse(storage?.getItem(TREND_KEY) ?? 'null');
    if (value?.version !== VERSION || !Array.isArray(value.games)) return [];
    return value.games.filter(entry).slice(-TREND_GAMES);
  } catch { return []; }
}

/** Enters one reviewed game, replacing an earlier entry for the same game
 * judged by the same network, and returns the games as they now stand,
 * whether or not they could be kept. */
export function recordTrend(storage, game) {
  const games = [...readTrend(storage).filter(other => other.id !== game.id || other.network !== game.network), game]
    .filter(entry).slice(-TREND_GAMES);
  try { storage?.setItem(TREND_KEY, JSON.stringify({ version: VERSION, games })); } catch { /* Kept for this page only. */ }
  return games;
}

/** The games judged by this version of the network, oldest first. */
export const judgedBy = (games, network) => games.filter(game => game.network === network);

/** The share of decisions matched over these games, decision-weighted, so a
 * short game counts for what it held; null for no decisions at all. */
export function matchedShare(games) {
  const decisions = games.reduce((sum, game) => sum + game.decisions, 0);
  return decisions ? games.reduce((sum, game) => sum + game.matched, 0) / decisions : null;
}
