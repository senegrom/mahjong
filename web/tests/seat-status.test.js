import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { calledSet, seatAnnouncement } from '../src/lib/ui.js';

test('a set called in front of the table is announced', () => {
  assert.equal(calledSet([], ['chii']), 'Chii');
  assert.equal(calledSet([], ['pon']), 'Pon');
  assert.equal(calledSet([], ['claimed-kan']), 'Kan');
  assert.equal(calledSet([], ['concealed-kan']), 'Kan');
  assert.equal(calledSet(['chii'], ['chii', 'pon']), 'Pon');
  // A triplet extended into a quad changes a set in place; it is a call.
  assert.equal(calledSet(['pon', 'chii'], ['extended-kan', 'chii']), 'Kan');
});

test('sets that were already there when the seat came into view say nothing', () => {
  assert.equal(calledSet(['pon'], ['pon']), '', 'nothing changed');
  assert.equal(calledSet([], []), '', 'a closed hand on both sides');
  assert.equal(calledSet(['pon', 'chii'], []), '', 'the next deal');
  assert.equal(calledSet(null, ['pon']), '', 'the first look at this seat');
  assert.equal(calledSet(['pon'], undefined), '', 'nothing to compare with');
});

test('a screen reader is told which seat called or declared riichi', () => {
  assert.equal(seatAnnouncement('West', false, 'Pon'), 'West called pon');
  assert.equal(seatAnnouncement('North', false, 'Kan'), 'North called kan');
  assert.equal(seatAnnouncement('South', true, ''), 'South declared riichi');
  // Riichi outlasts a call, as the badge shows it.
  assert.equal(seatAnnouncement('South', true, 'Kan'), 'South declared riichi');
  assert.equal(seatAnnouncement('East', false, ''), '', 'nothing to say');
});

test('the opponent seat keeps riichi and lets a call fade on its own', async () => {
  const source = await readFile(new URL('../src/lib/Seat.svelte', import.meta.url), 'utf8');
  // Riichi is a state and stays; a call is an event and is cleared again.
  assert.match(source, /seat\.riichi \? 'Riichi' : called/);
  assert.match(source, /setTimeout\(\(\) => \{ called = ''; \}/);
  // The badge must never be read straight off the melds, which would leave
  // a call from ten turns ago still announced.
  assert.doesNotMatch(source, /\$derived\([^\n]*seat\.melds\.at/);
});
