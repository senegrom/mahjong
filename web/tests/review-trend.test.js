import test from 'node:test';
import assert from 'node:assert/strict';
import { TREND_GAMES, TREND_KEY, judgedBy, matchedShare, readTrend, recordTrend } from '../src/lib/review-trend.js';

const storage = (initial = {}) => {
  const values = new Map(Object.entries(initial));
  return { values, getItem: key => values.get(key) ?? null, setItem: (key, value) => { values.set(key, String(value)); } };
};
/** A game as the review enters it now, judged averaging over the suits. */
const game = (id, matched, decisions = 100, at = 1, network = '0.1.810') => ({ id, network, at, matched, decisions, flagged: 0, suits: true });
/** A game as it was entered before the review averaged: plain, unsaid. */
const plainGame = (...args) => { const { suits: _, ...entered } = game(...args); return entered; };

test('a browser that keeps nothing, or something else, has no trend', () => {
  assert.deepEqual(readTrend(null), []);
  assert.deepEqual(readTrend(storage()), []);
  assert.deepEqual(readTrend(storage({ [TREND_KEY]: 'not json' })), []);
  assert.deepEqual(readTrend(storage({ [TREND_KEY]: JSON.stringify({ version: 2, games: [game('a', 1)] }) })), []);
  assert.deepEqual(readTrend({ getItem: () => { throw new Error('denied'); } }), []);
  // Entries that could not have been written are left out, not trusted: a
  // game is always entered with the network that judged it.
  const { network: _, ...unjudged } = game('d', 1);
  const mixed = [game('a', 5, 10), { id: 'b', network: '0.1.810', at: 1, matched: 11, decisions: 10 }, { id: 3 }, null, unjudged, game('c', 0, 4),
    { ...game('e', 1), suits: 'yes' }, plainGame('f', 1)];
  assert.deepEqual(readTrend(storage({ [TREND_KEY]: JSON.stringify({ version: 1, games: mixed }) })).map(entry => entry.id), ['a', 'c', 'f']);
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

test('a game judged again by another version of the network is entered beside its first entry', () => {
  const kept = storage();
  recordTrend(kept, game('a', 50));
  recordTrend(kept, game('b', 60, 100, 1, '0.1.410'));
  // The same game, judged by the other version, is a figure of its own.
  const games = recordTrend(kept, game('a', 40, 100, 2, '0.1.410'));
  assert.deepEqual(games.map(entry => [entry.id, entry.network, entry.matched]),
    [['a', '0.1.810', 50], ['b', '0.1.410', 60], ['a', '0.1.410', 40]]);
  // Judged again by the same version, it replaces only that version's entry.
  recordTrend(kept, game('a', 45, 100, 3, '0.1.410'));
  assert.deepEqual(readTrend(kept).map(entry => [entry.id, entry.network, entry.matched]),
    [['a', '0.1.810', 50], ['b', '0.1.410', 60], ['a', '0.1.410', 45]]);
  assert.deepEqual(judgedBy(readTrend(kept), '0.1.410').map(entry => entry.id), ['b', 'a']);
  assert.deepEqual(judgedBy(readTrend(kept), '0.1.810').map(entry => entry.id), ['a']);
  assert.deepEqual(judgedBy(readTrend(kept), '0.2.0'), []);
});

test('a game judged averaging over the suits is compared only with games judged so, an older entry counting as plain', () => {
  const kept = storage({ [TREND_KEY]: JSON.stringify({ version: 1, games: [plainGame('a', 50), plainGame('b', 60)] }) });
  // Judged again, averaging, the same game is entered beside its plain entry.
  const games = recordTrend(kept, game('a', 55, 100, 2));
  assert.deepEqual(games.map(entry => [entry.id, entry.matched, entry.suits ?? null]), [['a', 50, null], ['b', 60, null], ['a', 55, true]]);
  assert.deepEqual(judgedBy(games, '0.1.810').map(entry => entry.id), ['a'], 'the page\'s review averages');
  assert.deepEqual(judgedBy(games, '0.1.810', false).map(entry => [entry.id, entry.matched]), [['a', 50], ['b', 60]]);
  // Judged again alike, each replaces only its own entry.
  recordTrend(kept, game('a', 58, 100, 3));
  recordTrend(kept, plainGame('b', 61, 100, 3));
  assert.deepEqual(readTrend(kept).map(entry => [entry.id, entry.matched, entry.suits ?? null]), [['a', 50, null], ['a', 58, true], ['b', 61, null]]);
});

test('games are averaged by decision, so a short game counts for what it held', () => {
  assert.equal(matchedShare([]), null);
  assert.equal(matchedShare([game('a', 10, 20), game('b', 90, 100)]), 100 / 120);
  assert.equal(matchedShare([game('a', 0, 4)]), 0);
});
