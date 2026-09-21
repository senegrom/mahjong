import test from 'node:test';
import assert from 'node:assert/strict';
import { emptyPosition, recordChoice, parsePhysical } from '../src/lib/physical-position.js';
import { emptyGuided, guidedEvent, undoGuided, parseGuided, GUIDED_FORMAT, setTiles } from '../src/lib/guided-game.js';

function pending(offered, seat = 1) {
  const position = emptyPosition();
  Object.assign(position, { seat, turn: 0, phase: 'call', pending: offered, wall: 60 });
  position.players[0].discards.push({ tile: offered, order: 0, drawn: false, riichi: false, claimed: false });
  return position;
}

for (const offered of ['1m', '2m', '3m']) {
  test(`own chii on ${offered} preserves and displays the actual claim through save and undo`, () => {
    let game = emptyGuided();
    game.state.position = pending(offered);
    game.state.stage = 'decision';
    game.state.position.players[1].hand = ['1m', '2m', '3m'].filter(tile => tile !== offered);
    const choice = { kind: 'chii', tile: '1m' };
    const before = structuredClone(game);
    game = guidedEvent(game, { type: 'choice', choice, choices: [choice] });
    assert.equal(game.state.position.players[1].melds.length, 0, 'do not commit before higher-priority responses');
    game = guidedEvent(game, { type: 'continue' });
    const meld = game.state.position.players[1].melds[0];
    assert.equal(meld.claimed_tile, offered);
    assert.equal(setTiles(meld)[0], offered, 'GuidedPlay uses this order for the sideways left tile');
    assert.deepEqual([...setTiles(meld)].sort(), ['1m', '2m', '3m']);
    const restored = parseGuided(GUIDED_FORMAT.encode(game));
    assert.ok(restored);
    assert.equal(restored.state.position.players[1].melds[0].claimed_tile, offered);
    assert.deepEqual(undoGuided(undoGuided(restored)), before);
  });

  test(`opponent chii on ${offered} retains the same metadata and display order`, () => {
    const game = emptyGuided();
    game.state.position = pending(offered, 3);
    game.state.stage = 'responses';
    const queued = guidedEvent(game, { type: 'call', seat: 1, kind: 'chii', tile: '1m' });
    const called = guidedEvent(queued, { type: 'continue' });
    const restored = parseGuided(GUIDED_FORMAT.encode(called));
    assert.ok(restored);
    const meld = restored.state.position.players[1].melds[0];
    assert.equal(meld.claimed_tile, offered);
    assert.equal(meld.from, 3);
    assert.equal(setTiles(meld)[0], offered);
    assert.deepEqual(undoGuided(restored), queued);
    assert.equal(game.state.position.players[0].discards[0].claimed, false);
  });
}

test('old drafts remain readable without guessed claim metadata', () => {
  const p = pending('2m');
  p.players[1].melds.push({ kind: 'chii', tile: '1m', from: 3 });
  const restored = parsePhysical(JSON.stringify({ version: 1, position: p }));
  assert.ok(restored);
  assert.equal(Object.hasOwn(restored.players[1].melds[0], 'claimed_tile'), false);
  assert.deepEqual(setTiles(restored.players[1].melds[0]), ['1m', '2m', '3m']);
  restored.players[1].melds[0].claimed_tile = '0z';
  assert.equal(parsePhysical(JSON.stringify({ version: 1, position: restored })), null);
});

test('a set that excludes the offered tile cannot be recorded', () => {
  const p = pending('9m');
  p.players[1].hand = ['1m', '2m'];
  const before = structuredClone(p), choice = { kind: 'chii', tile: '1m' };
  assert.throws(() => recordChoice(p, choice, [choice]), /include the offered tile/);
  assert.deepEqual(p, before);
});

test('extending a pon retains its original claimed tile and source', () => {
  const p = emptyPosition();
  p.players[0].hand = ['2m'];
  p.players[0].melds.push({ kind: 'pon', tile: '2m', from: 2, claimed_tile: '2m' });
  const choice = { kind: 'extended-kan', tile: '2m' };
  const result = recordChoice(p, choice, [choice]);
  assert.deepEqual(result.players[0].melds[0], { kind: 'extended-kan', tile: '2m', from: 2, claimed_tile: '2m' });
  assert.deepEqual(setTiles(result.players[0].melds[0]), ['2m', '2m', '2m', '2m']);
});
