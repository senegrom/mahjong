import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, review_facts } from '../src/wasm/riichi.js';
import { eachDecision, reviewCheckGame } from './fixtures/review-check-game.js';
import { prototypeFacts } from './fixtures/prototype-facts.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

const snapshot = reviewCheckGame(Game);
const decisions = visit => eachDecision(Game, snapshot, visit);
const WINDS = ['east', 'south', 'west', 'north'];

/** The prototype's facts as review_facts hands them over: without the hand's
 * own distance, the dora by name and each discard's improving kinds, which
 * the engine keeps for its own checks (discard_facts.rs) since the page reads
 * none of them. The copies of the improving kinds left are still compared. */
function handedOver(position) {
  const { shanten: _shanten, dora: _dora, discards, ...facts } = prototypeFacts(position);
  return { ...facts, discards: discards.map(({ improving: _improving, ...discard }) => discard) };
}

test('review_facts agrees with the prototype\'s engine facts at every turn of the review check\'s game', () => {
  let turns = 0, tiles = 0;
  decisions(({ hand, index, position }) => {
    if (position.phase !== 'act') return;
    const facts = review_facts(position);
    // The prototype measured every kind held; the engine measures the tiles
    // the player may throw, which with no call and no riichi standing, as in
    // every turn of this game, are the same.
    assert.deepEqual(facts, handedOver(position), `${hand.name}, decision ${index + 1}`);
    turns++;
    tiles += facts.discards.length;
  });
  assert.equal(turns, 114);
  assert.equal(tiles, 1337);
});

test('review_facts gives the review notes\' own numbers for the moves played and advised, and the table\'s safe tiles', () => {
  let tiles = 0, safeSets = 0;
  decisions(({ hand, index, view, note, position }) => {
    if (position.phase !== 'act' || !['discard', 'riichi'].includes(note.played_kind)) return;
    const facts = review_facts(position), at = `${hand.name}, decision ${index + 1}`;
    const measured = tile => {
      const found = facts.discards.find(entry => entry.tile === tile);
      return found && [found.shanten, found.acceptance];
    };
    assert.deepEqual(measured(note.played_tile), [note.shanten_played, note.acceptance_played], at);
    tiles++;
    if (note.advised_tile) {
      assert.deepEqual(measured(note.advised_tile), [note.shanten_advised, note.acceptance_advised], at);
      tiles++;
    }
    if (!facts.threats.length) return;
    // The seats in riichi are the view's, and the tiles held that are safe
    // against every one of them are the view's safe tiles.
    assert.deepEqual(facts.threats.map(threat => threat.seat),
      view.seats.slice(1).filter(seat => seat.riichi).map(seat => WINDS.indexOf(seat.seat)).sort(), at);
    const safe = facts.discards.filter(entry => entry.tiers.every(exposure => exposure.tier === 'safe')).map(entry => entry.tile);
    assert.deepEqual(safe, view.safe.filter(tile => facts.discards.some(entry => entry.tile === tile)), at);
    safeSets++;
  });
  assert.equal(tiles, 228);
  assert.equal(safeSets, 17);
});

test('review_facts names nothing to throw while the player answers another seat\'s tile', () => {
  let answering = 0;
  decisions(({ position }) => {
    if (position.phase !== 'call') return;
    const facts = review_facts(position);
    assert.deepEqual(facts.discards, []);
    assert.deepEqual(facts, { ...handedOver(position), discards: [] });
    answering++;
  });
  assert.ok(answering > 0, 'the game has decisions answering a tile');
});

test('review_facts refuses a table the rules refuse, saying why', () => {
  let refused = false;
  decisions(({ position }) => {
    if (refused || position.phase !== 'act') return;
    // Four more copies of a tile the player holds, in the next seat's river.
    const tile = position.players[position.seat].hand[0], next = (position.seat + 1) % 4;
    const order = Math.max(-1, ...position.players.flatMap(player => player.discards.map(discard => discard.order))) + 1;
    const fifth = structuredClone(position);
    fifth.players[next].discards.push(...Array.from({ length: 4 }, (_, k) =>
      ({ tile, order: order + k, drawn: false, riichi: false, claimed: false })));
    assert.throws(() => review_facts(fifth), /More than four copies/);
    refused = true;
  });
  assert.ok(refused);
});
