import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { tilesFor, noteFor, readings, YAKU_NOTES } from '../src/lib/yaku.js';

// A yaku the engine can print must have something to say about it, or the
// result screen offers a person an empty explanation.
test('every yaku the engine names has a note', () => {
  const source = readFileSync(new URL('../../engine/riichi-core/src/yaku.rs', import.meta.url), 'utf8');
  const named = [...source.matchAll(/=> "([^"]+)"/g)].map(match => match[1]);
  assert.ok(named.length > 30, 'the engine names its yaku in one place');
  for (const name of named) assert.ok(noteFor(name), `no note for ${name}`);
});

const win = (hand, winning, melds = []) => ({ hand, winning_tile: winning, melds });

test('a hand reads as sets and a pair, called sets included', () => {
  const all = readings(win(['2m', '3m', '4m', '5p', '5p', '5p', '7s', '8s', '3z', '3z'], '9s',
    [{ kind: 'chi', tiles: ['1m', '2m', '3m'] }]));
  assert.ok(all.length >= 1);
  const [first] = all;
  assert.equal(first.sets.length, 4, 'four sets');
  assert.deepEqual(first.pair.faces, ['3z', '3z']);
  assert.equal(first.sets.filter(set => set.kind === 'run').length, 3);
});

test('a double sequence points at the two sequences that made it', () => {
  const hand = win(['2m', '2m', '3m', '3m', '4m', '4m', '6p', '7p', '8p', '1s', '1s', '5z', '5z'], '5z');
  const places = tilesFor('Pure Double Sequence', hand);
  assert.equal(places.length, 6, 'two sequences of three');
  assert.ok(places.every(place => place.startsWith('hand:')));
});

test('a mixed triple sequence points at one sequence in each suit', () => {
  const hand = win(['3m', '4m', '5m', '3p', '4p', '5p', '3s', '4s', '5s', '1z', '1z', '9m', '9m'], '9m');
  const places = tilesFor('Mixed Triple Sequence', hand);
  assert.equal(places.length, 9);
});

test('a pure straight points at the three sequences of one suit', () => {
  const hand = win(['1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p', '2z', '2z', '5s', '5s'], '5s');
  assert.equal(tilesFor('Pure Straight', hand).length, 9);
});

test('a dragon triplet points at the dragons alone, in hand or called', () => {
  const concealed = win(['7z', '7z', '7z', '2m', '3m', '4m', '5p', '6p', '7p', '1s', '1s', '2s', '3s'], '4s');
  assert.deepEqual(tilesFor('Dragon Triplet', concealed).sort(),
    ['hand:0', 'hand:1', 'hand:2']);
  const claimed = win(['2m', '3m', '4m', '5p', '6p', '7p', '1s', '1s', '2s', '3s'], '4s',
    [{ kind: 'pon', tiles: ['6z', '6z', '6z'] }]);
  assert.deepEqual(tilesFor('Dragon Triplet', claimed).sort(),
    ['meld:0:0', 'meld:0:1', 'meld:0:2']);
});

test('all simples lights the whole hand, called tiles included', () => {
  const hand = win(['2m', '3m', '4m', '5p', '6p', '7p', '3s', '3s'], '3s',
    [{ kind: 'pon', tiles: ['4s', '4s', '4s'] }, { kind: 'chi', tiles: ['6m', '7m', '8m'] }]);
  const places = tilesFor('All Simples', hand);
  assert.equal(places.length, 9 + 6, 'nine in hand and two called sets');
  assert.ok(places.includes('won'));
});

test('a yaku about how the hand was won points at no tiles', () => {
  const hand = win(['2m', '3m', '4m', '5p', '6p', '7p', '3s', '4s', '5s', '1z', '1z', '2p', '3p'], '4p');
  for (const name of ['Riichi', 'Ippatsu', 'Under the Sea', 'Robbing a Quad', 'Fully Concealed Hand']) {
    assert.deepEqual(tilesFor(name, hand), [], name);
  }
  assert.ok(noteFor('Riichi').length > 10, 'but it still explains itself');
});

test('an unreadable hand asks for nothing rather than guessing', () => {
  assert.deepEqual(tilesFor('Pure Straight', win(['1m'], '2m')), []);
  assert.deepEqual(tilesFor('All Simples', null), []);
  assert.equal(noteFor('Not A Yaku'), '');
});

test('the notes are sentences, not labels', () => {
  for (const [name, note] of Object.entries(YAKU_NOTES)) {
    assert.ok(note.length > 20 && note.endsWith('.'), name);
  }
});
