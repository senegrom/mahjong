import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { callLabel, percent, quadOnOffer, signed, winValue } from '../src/lib/ui.js';
await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

test('a win is headlined by what it is worth: a ron by its total, a self-draw by its shares', () => {
  const win = (by, total, fromDealer, fromEachOther) => ({ by, payments: { total, from_dealer: fromDealer, from_each_other: fromEachOther } });
  // A mangan: 8,000 for anyone else, paid 2,000 by each other player and
  // 4,000 by the dealer; 12,000 for the dealer, 4,000 from each.
  assert.equal(winValue(win('self-draw', 8000, 4000, 2000)), '2,000 / 4,000');
  assert.equal(winValue(win('self-draw', 12000, 0, 4000)), '4,000 all');
  assert.equal(winValue(win('discard', 8000, 0, 0)), '8,000');
  // A counter adds 100 to every share of a self-draw.
  assert.equal(winValue(win('self-draw', 2300, 1100, 600)), '600 / 1,100');
});

test('the result sheet, the standings and the advisers share one way of writing their figures', () => {
  assert.equal(signed(8000), '+8,000');
  assert.equal(signed(-2000), '-2,000');
  assert.equal(signed(0), '0', 'no change has no sign');
  assert.equal(percent(0.8), '80.0%');
  assert.equal(percent(0), '0.0%');
  assert.equal(percent(0.0004), '<0.1%', 'a weight too small to show is not shown as none');
});

test('a call window offers a quad to be robbed only where the seat offering it holds that quad', () => {
  const quad = (kind, tile = '7z') => ({ kind, tiles: Array(4).fill(tile), from: kind === 'concealed-kan' ? 'self' : 'left', claimed_tile: null });
  const offering = melds => ({ phase: 'call', pending_discard: '7z', pending_from: 'west',
    seats: [{ seat: 'south', melds: [] }, { seat: 'west', melds }, { seat: 'north', melds: [quad('claimed-kan', '2m')] }] });
  assert.equal(quadOnOffer(offering([quad('concealed-kan')])), 'concealed-kan');
  assert.equal(quadOnOffer(offering([quad('pon', '1m'), quad('extended-kan')])), 'extended-kan');
  assert.equal(quadOnOffer(offering([quad('extended-kan', '1m')])), null, 'a quad of another tile');
  assert.equal(quadOnOffer(offering([])), null, 'an ordinary discard');
  assert.equal(quadOnOffer({ ...offering([quad('concealed-kan')]), phase: 'act' }), null, 'the quad went ahead');
  assert.equal(callLabel({ kind: 'ron' }, true), 'Ron — rob the quad');
  assert.equal(callLabel({ kind: 'ron' }), 'Ron — win on discard');
});

// Saves taken before format 7, replayed on an engine that shows the same
// position the way format 7 does. The engine stands in for the real one so
// each difference can be set out on its own.
const quad = { kind: 'extended-kan', tiles: ['1p', '1p', '1p', '1p'], from: 'left', claimed_tile: '1p' };
const others = ['south', 'west', 'north'].map((wind, index) => ({ player: index + 1, controller: 'club', seat: wind, hand_size: 13,
  hand: [], drawn: null, melds: [], discards: [], score: 25000, riichi: false, turn: false }));
/** East, the player, has added the drawn fourth 1 circles to its triplet. */
const declared = (extra = {}) => ({ round: 'east', kyoku: 1, hands_played: 0, counters: 0, riichi_sticks: 0, wall: 40,
  dora_indicators: ['3z'], phase: 'call', pending_discard: '1p', pending_from: 'east', dora_types: ['4z'], outcome: null,
  waits: ['9s'], waits_left: [3], dora: [], shanten: 0, furiten: false, safe: [],
  seats: [{ player: 0, controller: 'human', seat: 'east', hand_size: 10, hand: ['2m', '3m', '4m', '5m', '6m', '7m', '7s', '8s', '9s', '9s'],
    drawn: null, melds: [quad], discards: [], score: 25000, riichi: false, turn: true }, ...structuredClone(others)], ...extra });
/** The same once South has robbed the quad. */
const robbed = (line, payments = true) => declared({ phase: 'over', pending_discard: '1p', outcome: { kind: 'win', line, tenpai: [],
  changes: [-1300, 1300, 0, 0], wins: [{ seat: 'south', by: 'discard', from: 'east', hand: ['2p', '3p'], melds: [], winning_tile: '1p',
    yaku: [{ id: 'RobbingAQuad', tile: null, name: 'Robbing a Quad', han: 1 }], scoring: null, han: 2, dora: 0, dora_types: ['4z'],
    dora_indicators: ['3z'], ura_indicators: [], ura_dora: 0, fu: 30, limit: null, payment: '1300 from the discarder', bets: 0,
    ...(payments ? { payments: { total: 1300, from_dealer: 0, from_each_other: 0 } } : {}) }] } });
const replaying = view => class {
  view() { return structuredClone(view); }
  choices() { return []; }
  advance() { return []; }
  needs_opponent_move() { return false; }
  game_is_over() { return false; }
  hand_is_over() { return view.phase === 'over'; }
  free() {}
};
const save = (view, format = 6) => JSON.stringify({ version: 1, format, seed: 5, difficulty: 'club', opponents: ['club', 'club', 'club'],
  commands: [], state: JSON.stringify([view, [], false]) });
const restores = (now, before, format) => {
  const session = MatchSession.restore(replaying(now), save(before, format));
  try { return session.snapshot(); } finally { session.dispose(); }
};
/** What a format-6 engine showed: East's hand with the tile it had drawn
 * before the quad held apart, whether or not the hand still held it. */
const held = (view, drawn, hand = view.seats[0].hand) => {
  const old = structuredClone(view);
  Object.assign(old.seats[0], { drawn, hand });
  return old;
};

test('a format-6 save taken while the player\'s quad could be robbed still restores, and is saved as format 7', () => {
  // The quad took the drawn tile: the old view held a fifth 1 circles apart.
  const now = declared();
  assert.equal(restores(now, held(now, '1p')).format, 7);
  // The quad took a tile held from before, and the tile drawn this turn is
  // in the hand: the old view held it apart, and read no waits from the
  // hand without it.
  const kept = declared({ seats: [{ ...declared().seats[0], hand: ['2m', '3m', '4m', '5m', '6m', '7m', '5s', '7s', '8s', '9s'] }, ...others] });
  const old = held(kept, '5s', ['2m', '3m', '4m', '5m', '6m', '7m', '7s', '8s', '9s']);
  Object.assign(old, { waits: [], waits_left: [] });
  assert.equal(restores(kept, old).format, 7);
  // Every tile is still compared: a different hand is refused, and so is the
  // old view in a format-7 save, which shows no tile held apart there.
  assert.throws(() => restores(now, held(now, '1p', ['2m', '3m', '4m', '5m', '6m', '7m', '7s', '8s', '9s', '9p'])), /does not match/);
  assert.throws(() => restores(now, held(now, '1p'), 7), /does not match/);
  // In the middle of a turn the tile held apart is the drawn tile, as before.
  const acting = declared({ phase: 'act', pending_discard: null, pending_from: null });
  assert.throws(() => restores(acting, held(acting, '1p')), /does not match/);
});

test('a format-6 save of a robbed quad\'s result still restores, without the numbers or words format 7 added', () => {
  const now = robbed("South robs East's quad");
  assert.equal(restores(now, held(robbed("South wins on East's discard", false), '1p')).format, 7);
  // The sentence and every other part of the win are still compared.
  const other = held(robbed("South wins on East's discard", false), '1p');
  other.outcome.wins[0].payment = '2000 from the discarder';
  assert.throws(() => restores(now, other), /does not match/);
  assert.throws(() => restores(now, held(robbed("West wins on East's discard", false), '1p')), /does not match/);
});

// Real saves the page wrote in format 6, at 4374bc0: a non-dealer's mangan by
// self-draw; the player's added quad robbed, once with the fourth tile just
// drawn and once with it held from before and another tile just drawn; and a
// window to rob a Trained opponent's added quad.
const saves = JSON.parse(readFileSync(new URL('./fixtures/format6-saves.json', import.meta.url)));
const restore = saved => MatchSession.restore(Game, JSON.stringify(saved));

test('real format-6 saves restore and show what format 7 adds: the value, no fifth tile, a robbed quad', () => {
  const checks = {
    selfDraw(view, old) {
      const win = view.outcome.wins[0];
      assert.equal(win.payment, old.outcome.wins[0].payment);
      assert.equal(winValue(win), '2,000 / 4,000', 'the old headline read 4,000');
    },
    robbedDrawn(view, old) {
      // The quad holds all four 4 characters, and the old view held a fifth apart.
      assert.equal(old.seats[0].drawn, '4m');
      assert.equal(view.seats[0].drawn ?? null, null);
      assert.deepEqual(view.seats[0].hand, old.seats[0].hand);
      assert.equal(old.outcome.line, "North wins on South's discard");
      assert.equal(view.outcome.line, "North robs South's quad");
    },
    robbedHeld(view, old) {
      // The 6 bamboo just drawn is in the hand, no longer held apart.
      assert.equal(old.seats[0].drawn, '6s');
      assert.equal(view.seats[0].drawn ?? null, null);
      assert.deepEqual([...view.seats[0].hand].sort(), [...old.seats[0].hand, '6s'].sort());
      assert.equal(view.outcome.line, "South robs North's quad");
    },
    robbing(view, old, match) {
      // Compared as a save holds it, where a field with no value is absent.
      assert.deepEqual(JSON.parse(JSON.stringify(view)), old, 'nothing in the window differs');
      assert.equal(quadOnOffer(view), 'extended-kan');
      assert.deepEqual(match.choices.map(choice => choice.kind), ['ron', 'pass']);
      match.apply({ type: 'choose', kind: 'ron', tile: null });
      match.advance(false);
      assert.equal(match.view.outcome.line, "South robs North's quad");
      assert.ok(match.events.includes('South robs the quad'), match.events.join('\n'));
    },
  };
  assert.deepEqual(Object.keys(saves), Object.keys(checks));
  for (const [name, saved] of Object.entries(saves)) {
    assert.equal(saved.format, 6, name);
    const match = restore(saved);
    try {
      checks[name](match.view, JSON.parse(saved.state)[0], match);
      // Saved again, a match is format 7 and restores exactly.
      const again = restore(match.snapshot());
      try { assert.equal(again.stateKey(), match.stateKey()); assert.equal(again.snapshot().format, 7); } finally { again.dispose(); }
    } finally { match.dispose(); }
    // Read as format 7, each result is refused: the rules for older saves are
    // what let it in. The window looks the same in both.
    if (name !== 'robbing') assert.throws(() => restore({ ...saved, format: 7 }), /does not match/, name);
  }
});

test('every call and win after the player passes on a discard is in the hand history', () => {
  const seat = name => name[0].toUpperCase() + name.slice(1);
  let checked = 0;
  for (let seed = 1; seed <= 150; seed++) {
    const game = new Game(seed, 'club');
    try {
      game.advance();
      for (let n = 0; n < 500 && !game.hand_is_over(); n++) {
        const view = game.view(), choices = game.choices();
        if (view.phase !== 'call') {
          const discard = choices.find(choice => choice.kind === 'discard') ?? choices[0];
          game.choose(discard.kind, discard.tile ?? undefined);
          game.advance();
          continue;
        }
        const melds = view.seats.map(other => other.melds.length);
        game.choose('pass');
        const now = game.view(), lines = game.advance();
        const caller = now.seats.findIndex((other, index) => index > 0 && other.melds.length > melds[index]);
        if (caller > 0) {
          checked++;
          assert.ok(lines.some(line => line.startsWith(`${seat(now.seats[caller].seat)} calls a`)), `seed ${seed}: ${lines}`);
        }
        for (const win of now.outcome?.wins ?? []) {
          if (win.by !== 'discard') continue;
          checked++;
          assert.ok(lines.some(line => [`${seat(win.seat)} wins on the discard`, `${seat(win.seat)} robs the quad`].includes(line)),
            `seed ${seed}: ${lines}`);
        }
      }
    } finally { game.free(); }
  }
  assert.ok(checked >= 5, `only ${checked} calls and wins followed a pass`);
});
