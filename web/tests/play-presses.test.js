import test from 'node:test';
import assert from 'node:assert/strict';
import { handKeyActs, heldMove, pressedByKeyboard, repeatedClick } from '../src/lib/ui.js';

test('only the second and later presses of a multiple click are let go, never a single press or a key\'s', () => {
  // A click's detail is its count: one for a single press, two for a double
  // click's second, and none for the click a browser makes for Enter or Space.
  for (const [detail, again] of [[0, false], [1, false], [2, true], [3, true]]) assert.equal(repeatedClick({ detail }), again, `detail ${detail}`);
  // No event, or one without a count, is a press like any other.
  for (const event of [undefined, null, {}]) assert.equal(repeatedClick(event), false);
  // A key's press is never taken for a second click.
  assert.equal(pressedByKeyboard({ detail: 0 }) && repeatedClick({ detail: 0 }), false);
});

/** An element inside what each of `selectors` names, as `closest` finds one
 * around it: a button, a button in a row of the player's choices. */
const inside = (...selectors) => ({ closest: selector => (selectors.includes(selector) ? {} : null) });
const enter = (target, values = {}) => ({ key: 'Enter', repeat: true, target, ...values });

test('Enter held down presses a tile of the hand or one of the player\'s choices once, and is left alone elsewhere', () => {
  const tile = inside('button'), rack = inside(), hand = { contains: node => node === tile || node === rack };
  const call = inside('[data-moves] button', 'button'), other = inside('button');
  // In the hand: on a tile, not on the hand itself, which is no button.
  assert.equal(heldMove(enter(tile), hand), true);
  assert.equal(heldMove(enter(rack), hand), false);
  // On a call or the confirmation's buttons, with or without a hand to name.
  assert.equal(heldMove(enter(call), hand), true);
  assert.equal(heldMove(enter(call), null), true);
  // Whatever is held with it: the keys' own guard leaves a key with Shift alone.
  assert.equal(heldMove(enter(tile, { shiftKey: true }), hand), true);
  // Any other button is its own business: the settings', the result's.
  assert.equal(heldMove(enter(other), hand), false);
  // The first press goes through, and so does every other key held down.
  assert.equal(heldMove(enter(tile, { repeat: false }), hand), false);
  for (const key of [' ', 'ArrowDown', 'Tab', '1']) assert.equal(heldMove(enter(tile, { key }), hand), false, key);
  assert.equal(heldMove(enter(null), hand), false);
});

test('a held key is cancelled in the hand only where the hand\'s keys act on it', () => {
  for (const key of ['Enter', ' ', '0', '1', '9', 'r', 'R', 't', 'T']) assert.equal(handKeyActs(key), true, key);
  // The browser keeps the rest: scrolling, moving focus, its own shortcuts.
  for (const key of ['ArrowDown', 'ArrowUp', 'PageDown', 'PageUp', 'Home', 'End', 'Tab', 'a', 's', 'F5', 'Escape', '', undefined]) {
    assert.equal(handKeyActs(key), false, String(key));
  }
});
