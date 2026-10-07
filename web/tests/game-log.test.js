import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { finishedHands, gameFileName, matchLabels, playerNames } from '../src/lib/game-log.js';
await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

const make = (seed, opponents = 'club') => { const match = new MatchSession(Game, seed, opponents); match.advance(false); return match; };
const events = text => text.split('\n').map(line => JSON.parse(line));
const deals = text => events(text).filter(event => event.type === 'start_kyoku');
/** One step of the match as the page takes it: the player's move, a
 * trained opponent's answer (the first move its mask allows) or the deal. */
function step(match) {
  if (match.engine.hand_is_over()) match.apply({ type: 'next' });
  else if (match.engine.needs_opponent_move()) {
    match.apply({ type: 'opponent', action: match.engine.opponent_mask_mortal().findIndex(Boolean) });
  } else {
    const choices = match.choices;
    const choice = choices.find(c => c.kind === 'ron' || c.kind === 'tsumo') ?? choices.find(c => c.kind === 'riichi')
      ?? choices.find(c => c.kind === 'pon') ?? choices.find(c => c.kind === 'pass') ?? choices.find(c => c.kind === 'discard') ?? choices[0];
    match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null });
  }
  match.advance(false);
}

test('the game log is every finished hand from East 1, never the hand being played', () => {
  const match = make(14, 'beginner');
  try {
    const hands = [];
    for (let steps = 0; !match.over; steps++) {
      assert.ok(steps < 4000, 'the match finishes');
      const log = match.engine.game_log([]);
      if (match.engine.hand_is_over()) {
        // The score screen: the hand just over is the last one in the log.
        hands.push(match.engine.log());
        assert.ok(log.endsWith(`\n${match.engine.log()}`));
      } else {
        // Its deal names every tile all four hold.
        const deal = match.engine.log().split('\n')[0];
        assert.ok(!log.includes(deal), 'the hand being played is not in the log');
      }
      assert.equal(log, [JSON.stringify({ type: 'start_game', names: ['player 0', 'player 1', 'player 2', 'player 3'] }), ...hands].join('\n'));
      assert.equal(match.engine.game_log_hands(), hands.length);
      step(match);
    }
    const log = match.engine.game_log([]);
    assert.equal(events(log).at(-1).type, 'end_game');
    assert.equal(deals(log).length, match.view.hands_played);
    assert.ok(hands.length >= 8, 'two rounds');
    const [first] = deals(log);
    assert.deepEqual([first.bakaze, first.kyoku, first.honba, first.kyotaku, first.oya], ['E', 1, 0, 0, 0]);
    assert.deepEqual(first.scores, [30000, 30000, 30000, 30000]);
    // The standings come from the same game: the last settlement's points.
    const settled = events(log).filter(event => event.type === 'hora' || event.type === 'ryukyoku').at(-1).scores;
    for (const row of match.engine.standings()) assert.equal(row.score, settled[row.player]);
  } finally { match.dispose(); }
});

test('a restored match exports from East 1 again, whatever format the save was written in', () => {
  const match = make(14, 'beginner');
  try {
    while (match.view.hands_played < 3) step(match);
    for (let n = 0; n < 6; n++) step(match);
    assert.notEqual(match.view.phase, 'over', 'the fourth hand is being played');
    const log = match.engine.game_log([]);
    assert.equal(deals(log).length, 3);
    const saved = match.snapshot();
    // The save is the deal and every command since: a restore replays the
    // hands from the first, so even a save from before the game log had
    // existed exports the whole game. No history needs storing.
    for (const format of [undefined, 2, 3, 4, 5, saved.format]) {
      const old = { ...saved, format };
      if (format === undefined) delete old.format;
      const restored = MatchSession.restore(Game, JSON.stringify(old));
      try {
        assert.equal(restored.engine.game_log([]), log, `format ${format}`);
        assert.equal(restored.engine.game_log_hands(), 3);
      } finally { restored.dispose(); }
    }
  } finally { match.dispose(); }
});

test('a trained table exports the same game after a reload, without asking the network again', () => {
  const match = make(1, ['neural', 'club', 'neural']);
  try {
    while (match.view.hands_played < 2) step(match);
    const log = match.engine.game_log([]);
    assert.equal(deals(log).length, 2);
    assert.ok(match.commands.some(command => command.type === 'opponent'));
    const restored = MatchSession.restore(Game, JSON.stringify(match.snapshot()), { ai() { throw new Error('no inference during replay'); } });
    try { assert.equal(restored.engine.game_log([]), log); } finally { restored.dispose(); }
  } finally { match.dispose(); }
});

test('players are named by number from where they sat, the person first', () => {
  const match = make(3, ['club', 'neural', 'beginner']);
  try {
    const names = playerNames(match.view, matchLabels(['club', 'neural', 'beginner'], ['club', 'club', 'beginner']));
    const [opening] = events(match.engine.game_log(names));
    assert.equal(opening.type, 'start_game');
    const seats = match.view.seats;
    assert.equal(opening.names[seats[0].player], 'You');
    assert.equal(opening.names[seats[1].player], 'Right (Club)');
    assert.equal(opening.names[seats[2].player], 'Opposite (Trained, then Club)');
    assert.equal(opening.names[seats[3].player], 'Left (Beginner)');
  } finally { match.dispose(); }
  assert.deepEqual(playerNames({ seats: [{ player: 2 }, { player: 3 }, { player: 0 }, { player: 1 }] }, ['a', 'b', 'c', 'd']), ['c', 'd', 'a', 'b']);
});

test('the file is named for the game and the local time it was saved', () => {
  assert.equal(gameFileName(new Date(2026, 9, 6, 9, 5, 7)), 'riichi-game-2026-10-06-090507.mjai.jsonl');
  assert.equal(finishedHands(1), '1 finished hand');
  assert.equal(finishedHands(7), '7 finished hands');
});
