import test from 'node:test';
import assert from 'node:assert/strict';
import { handKeyActs, heldPress, land, pressedByKeyboard, repeatedClick } from '../src/lib/ui.js';

test('only the second and later presses of a multiple click are let go, never a single press or a key\'s', () => {
  // A click's detail is its count: one for a single press, two for a double
  // click's second, and none for the click a browser makes for Enter or Space.
  for (const [detail, again] of [[0, false], [1, false], [2, true], [3, true]]) assert.equal(repeatedClick({ detail }), again, `detail ${detail}`);
  // No event, or one without a count, is a press like any other.
  for (const event of [undefined, null, {}]) assert.equal(repeatedClick(event), false);
  // A key's press is never taken for a second click.
  assert.equal(pressedByKeyboard({ detail: 0 }) && repeatedClick({ detail: 0 }), false);
});

const enter = (target, values = {}) => ({ key: 'Enter', repeat: true, target, ...values });

/** An element that is `kind`, as `closest` finds it from inside: a selector
 * list names it when one of its parts is that kind. */
const element = kind => ({ closest: selector => (selector.split(',').map(part => part.trim()).includes(kind) ? {} : null) });

test('Enter held down presses nothing twice: a tile, a call, any button, a summary, a link or a role="button", and leaves text fields and selects alone', () => {
  // A tile of the hand, a call, any other button (New game, the result's),
  // a summary, a link, and anything with role="button".
  for (const kind of ['button', 'summary', 'a[href]', '[role="button"]']) assert.equal(heldPress(enter(element(kind))), true, kind);
  // Whatever is held with it.
  assert.equal(heldPress(enter(element('button'), { shiftKey: true })), true);
  // Not on what Enter does not press: the hand itself, a text field, a select.
  for (const kind of ['div', 'input', 'textarea', 'select']) assert.equal(heldPress(enter(element(kind))), false, kind);
  // The first press goes through, and so does every other key held down:
  // Space presses as it is let go, once.
  assert.equal(heldPress(enter(element('button'), { repeat: false })), false);
  for (const key of [' ', 'ArrowDown', 'Tab', '1']) assert.equal(heldPress(enter(element('button'), { key })), false, key);
  for (const target of [null, undefined, {}]) assert.equal(heldPress(enter(target)), false);
  assert.equal(heldPress(null), false);
});

test('landing on a heading always moves focus, and moves the page only when asked: by as little as shows it, or to the top of the screen', () => {
  const done = [];
  const heading = { focus: options => done.push(['focus', options]), scrollIntoView: options => done.push(['scroll', options]) };
  const focused = ['focus', { preventScroll: true }];
  // A pointer's press where nothing was covered: focus alone.
  land(heading, false);
  assert.deepEqual(done.splice(0), [focused]);
  // A keyboard's press: the heading and its ring, by the least movement.
  land(heading, true);
  assert.deepEqual(done.splice(0), [focused, ['scroll', { block: 'nearest' }]]);
  // The final standings on a phone, as the result's sheet steps aside for a
  // tap: the heading at the top, what it opens on the screen under it.
  land(heading, true, 'start');
  assert.deepEqual(done.splice(0), [focused, ['scroll', { block: 'start' }]]);
  // A heading not drawn yet is no error.
  assert.doesNotThrow(() => land(null, true, 'start'));
});

test('a held key is cancelled in the hand only where the hand\'s keys act on it', () => {
  for (const key of ['Enter', ' ', '0', '1', '9', 'r', 'R', 't', 'T']) assert.equal(handKeyActs(key), true, key);
  // The browser keeps the rest: scrolling, moving focus, its own shortcuts.
  for (const key of ['ArrowDown', 'ArrowUp', 'PageDown', 'PageUp', 'Home', 'End', 'Tab', 'a', 's', 'F5', 'Escape', '', undefined]) {
    assert.equal(handKeyActs(key), false, String(key));
  }
});
