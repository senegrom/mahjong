import test from 'node:test';
import assert from 'node:assert/strict';
import { emptyPosition, parseTiles, recordChoice, recordDiscard, recordDraw } from '../src/lib/physical-position.js';

function callPosition(kind, wall = 10) {
  const p = emptyPosition();
  Object.assign(p, { seat: 1, turn: 0, phase: 'call', pending: '3m', wall, first_turns: false });
  p.players[0].discards = [{ tile: '3m', order: 0, drawn: false, riichi: false, claimed: false }];
  p.players[1].hand = parseTiles(kind === 'pon' ? '333m456p789s1122z' : '123m456p789s1122z');
  const choice = { kind, tile: kind === 'pon' ? '3m' : '1m' };
  return recordChoice(p, choice, [choice]);
}

for (const kind of ['pon', 'chii']) {
  for (const known of [false, true]) {
    test(`${known ? 'known' : 'unknown'} hand's post-${kind} discard does not consume a draw`, () => {
      const p = callPosition(kind);
      if (!known) p.players[1].hand = [];
      const before = structuredClone(p);
      const after = recordDiscard(p, 1, '4p');
      assert.equal(after.wall, 10);
      assert.equal(after.phase, 'call');
      assert.equal(after.pending, '4p');
      assert.equal(after.just_claimed, null);
      assert.equal(after.players[1].discards.at(-1).drawn, false);
      assert.deepEqual(p, before, 'recording does not mutate the saved draft');
      // The following unknown seat still needs its ordinary draw.
      assert.equal(recordDiscard(after, 2, '2p').wall, 9);
    });
  }
}

test('ordinary unknown opponents include exactly one draw per discard', () => {
  const p = emptyPosition();
  Object.assign(p, { wall: 10, turn: 0, phase: 'call', pending: '1m' });
  const first = recordDiscard(p, 1, '2p');
  const second = recordDiscard(first, 2, '3p');
  assert.equal(first.wall, 9);
  assert.equal(second.wall, 8);
  assert.deepEqual(second.players.map(player => player.discards.length), [0, 1, 1, 0]);
});

test('the initial dealer act decision does not count its already included draw twice', () => {
  const p = emptyPosition();
  assert.equal(recordDiscard(p, 0, '2p').wall, 69);
});

test('a recorded draw is not counted again when the hand is hidden before discarding', () => {
  const p = emptyPosition();
  p.wall = 10;
  p.players[0].hand = parseTiles('123m456p789s1122z');
  const drawn = recordDraw(p, 0, '3z');
  assert.equal(drawn.wall, 9);
  drawn.players[0].hand = [];
  const discarded = recordDiscard(drawn, 0, '3z');
  assert.equal(discarded.wall, 9);
  assert.equal(discarded.players[0].discards.at(-1).drawn, true);
});

for (const kind of ['kan', 'extended-kan', 'concealed-kan']) {
  test(`${kind}: implicit and explicit replacement draws each consume one wall tile`, () => {
    const p = emptyPosition();
    Object.assign(p, { seat: 1, turn: 1, wall: 10, after_quad: true,
      phase: kind === 'kan' ? 'draw' : 'call', pending_kind: kind });
    p.players[1].melds = [{ kind, tile: '5m', from: kind === 'concealed-kan' ? 0 : 3 }];
    const implicit = recordDiscard(p, 1, '2z');
    assert.equal(implicit.wall, 9);
    assert.equal(implicit.after_quad, false);
    p.players[1].hand = parseTiles('123m456p789s1z');
    const drawn = recordDraw(p, 1, '2z');
    assert.equal(drawn.wall, 9);
    assert.equal(drawn.after_quad, true);
    assert.equal(recordDiscard(drawn, 1, '2z').wall, 9);
  });
}

test('empty or incomplete walls reject implicit draws without changing the draft', () => {
  for (const wall of [0, null, undefined]) {
    const p = emptyPosition();
    Object.assign(p, { turn: 0, phase: 'call', pending: '1m', wall });
    const before = structuredClone(p);
    assert.throws(() => recordDiscard(p, 1, '2p', true), /wall/i);
    assert.deepEqual(p, before, 'failed riichi must not charge points or add a discard');
  }
});

test('a final recorded draw may be discarded when no live tiles remain', () => {
  const p = emptyPosition();
  p.wall = 1;
  p.players[0].hand = parseTiles('123m456p789s1122z');
  const drawn = recordDraw(p, 0, '3z');
  assert.equal(drawn.wall, 0);
  assert.equal(recordDiscard(drawn, 0, '3z').wall, 0);
});
