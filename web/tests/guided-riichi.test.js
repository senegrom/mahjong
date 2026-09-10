import test from 'node:test';
import assert from 'node:assert/strict';
import { emptyGuided, guidedEvent, undoGuided, parseGuided, GUIDED_FORMAT } from '../src/lib/guided-game.js';
import { parseTiles } from '../src/lib/physical-position.js';

function beforeDiscard(wall) {
  let game = emptyGuided();
  game.state.position.seat = 3;
  game.state.position.players[3].hand = parseTiles('123m456p789s1123z');
  for (const event of [{ type: 'setup' }, { type: 'hand' }, { type: 'indicator', tile: '7z' }]) {
    game = guidedEvent(game, event);
  }
  game.state.position.wall = wall;
  game.state.position.first_turns = false;
  return game;
}

test('EMA 2025: opponent riichi is recorded with one to three tiles left after the hidden draw', () => {
  for (const remaining of [1, 2, 3]) {
    const before = beforeDiscard(remaining + 1);
    const game = guidedEvent(before, { type: 'discard', tile: '9m', riichi: true });
    assert.equal(game.state.position.wall, remaining);
    assert.equal(game.state.position.players[0].riichi, 'riichi');
    assert.equal(game.state.position.players[0].score, 29000);
    assert.equal(game.state.position.riichi_sticks, 1);
    assert.deepEqual(parseGuided(GUIDED_FORMAT.encode(game)), game);
    assert.deepEqual(undoGuided(game), before);
  }
});

test('EMA 2025: the final live-wall draw cannot declare riichi or debit a stick', () => {
  const game = beforeDiscard(1), before = structuredClone(game);
  assert.throws(() => guidedEvent(game, { type: 'discard', tile: '9m', riichi: true }), /one live tile/);
  assert.deepEqual(game, before);
  const discarded = guidedEvent(game, { type: 'discard', tile: '9m' });
  assert.equal(discarded.state.position.wall, 0);
  assert.equal(discarded.state.position.players[0].score, 30000);
  assert.equal(discarded.state.position.riichi_sticks, 0);
});
