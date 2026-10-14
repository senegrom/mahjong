import test from 'node:test';
import assert from 'node:assert/strict';
import { TREND_GAMES, TREND_KEY, TREND_WINDOW, readTrend, recordTrend, trendSummary } from '../src/lib/review-trend.js';

const storage = (initial = {}) => {
  const values = new Map(Object.entries(initial));
  return { values, getItem: key => values.get(key) ?? null, setItem: (key, value) => { values.set(key, String(value)); } };
};
const game = (id, matched, decisions = 100, at = 1) => ({ id, at, matched, decisions, flagged: 0 });

test('a browser that keeps nothing, or something else, has no trend', () => {
  assert.deepEqual(readTrend(null), []);
  assert.deepEqual(readTrend(storage()), []);
  assert.deepEqual(readTrend(storage({ [TREND_KEY]: 'not json' })), []);
  assert.deepEqual(readTrend(storage({ [TREND_KEY]: JSON.stringify({ version: 2, games: [game('a', 1)] }) })), []);
  assert.deepEqual(readTrend({ getItem: () => { throw new Error('denied'); } }), []);
  // Entries that could not have been written are left out, not trusted.
  const mixed = [game('a', 5, 10), { id: 'b', at: 1, matched: 11, decisions: 10 }, { id: 3 }, null, game('c', 0, 4)];
  assert.deepEqual(readTrend(storage({ [TREND_KEY]: JSON.stringify({ version: 1, games: mixed }) })).map(entry => entry.id), ['a', 'c']);
});

test('each reviewed game is entered once, the newest last, and only the last fifty are kept', () => {
  const kept = storage();
  assert.deepEqual(recordTrend(kept, game('a', 50)).map(entry => entry.id), ['a']);
  recordTrend(kept, game('b', 60));
  // Reviewing the same game again replaces its entry rather than adding one.
  assert.deepEqual(recordTrend(kept, game('a', 70, 100, 2)).map(entry => [entry.id, entry.matched]), [['b', 60], ['a', 70]]);
  assert.deepEqual(readTrend(kept).map(entry => entry.id), ['b', 'a']);
  for (let index = 0; index < TREND_GAMES + 5; index++) recordTrend(kept, game(`g${index}`, index));
  const games = readTrend(kept);
  assert.equal(games.length, TREND_GAMES);
  assert.equal(games.at(-1).id, `g${TREND_GAMES + 4}`);
  assert.equal(JSON.parse(kept.values.get(TREND_KEY)).version, 1);
  // A refused write still answers with the games, for this page.
  const refusing = { getItem: () => null, setItem: () => { throw new Error('quota'); } };
  assert.deepEqual(recordTrend(refusing, game('x', 1)).map(entry => entry.id), ['x']);
  assert.deepEqual(recordTrend(refusing, { id: 'bad' }), []);
});

test('the trend averages the last ten games by decision, beside the ten before them', () => {
  assert.deepEqual(trendSummary([]), { games: 0, recent: null, recentGames: 0, earlier: null });
  // A short game counts for the decisions it held.
  const two = trendSummary([game('a', 10, 20), game('b', 90, 100)]);
  assert.equal(two.recent, 100 / 120);
  assert.equal(two.earlier, null, 'no earlier window without ten games in it');
  const games = [...Array.from({ length: TREND_WINDOW }, (_, i) => game(`old${i}`, 50)),
    ...Array.from({ length: TREND_WINDOW }, (_, i) => game(`new${i}`, 70))];
  assert.deepEqual(trendSummary(games), { games: 2 * TREND_WINDOW, recent: 0.7, recentGames: TREND_WINDOW, earlier: 0.5 });
});
