import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { tilesFor, noteFor, YAKU_NOTES } from '../src/lib/yaku.js';

// A yaku the engine can print must have something to say about it, or the
// result screen offers a person an empty explanation.
test('every yaku the engine names has a note', () => {
  const source = readFileSync(new URL('../../engine/riichi-core/src/yaku.rs', import.meta.url), 'utf8');
  const named = [...source.matchAll(/=> "([^"]+)"/g)].map(match => match[1]);
  assert.ok(named.length > 30, 'the engine names its yaku in one place');
  for (const name of named) assert.ok(noteFor(name), `no note for ${name}`);
});

const win = (hand, winning, melds = []) => ({ hand, winning_tile: winning, melds });

test('all simples lights the whole hand, called tiles included', () => {
  const hand = win(['2m', '3m', '4m', '5p', '6p', '7p', '3s', '3s'], '3s',
    [{ kind: 'pon', tiles: ['4s', '4s', '4s'] }, { kind: 'chi', tiles: ['6m', '7m', '8m'] }]);
  const places = tilesFor({ name: 'All Simples' }, hand);
  assert.equal(places.length, 9 + 6, 'nine in hand and two called sets');
  assert.ok(places.includes('won'));
});

test('a yaku about how the hand was won points at no tiles', () => {
  const hand = win(['2m', '3m', '4m', '5p', '6p', '7p', '3s', '4s', '5s', '1z', '1z', '2p', '3p'], '4p');
  for (const name of ['Riichi', 'Ippatsu', 'Under the Sea', 'Robbing a Quad', 'Fully Concealed Hand']) {
    assert.deepEqual(tilesFor({ name }, hand), [], name);
  }
  assert.ok(noteFor('Riichi').length > 10, 'but it still explains itself');
});

test('an unreadable hand asks for nothing rather than guessing', () => {
  assert.deepEqual(tilesFor({ name: 'Pure Straight' }, win(['1m'], '2m')), []);
  assert.deepEqual(tilesFor({ name: 'All Simples' }, null), []);
  assert.equal(noteFor('Not A Yaku'), '');
});

test('the notes are sentences, not labels', () => {
  for (const [name, note] of Object.entries(YAKU_NOTES)) {
    assert.ok(note.length > 20 && note.endsWith('.'), name);
  }
});
