import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { parse } from 'svelte/compiler';
import { REVIEW_WIDE, heldEnter, reviewKey } from '../src/lib/ui.js';

/** An element of the page, inside a field where `field` says so, as
 * `closest` finds one around it. */
const element = ({ field = false } = {}) => ({ closest: selector => (field && selector.includes('input') ? {} : null) });
/** A keydown as the review's window hears it, on an element outside any
 * field unless `target` says otherwise. */
const key = (name, values = {}) => ({ key: name, shiftKey: false, ctrlKey: false, altKey: false, metaKey: false, isComposing: false,
  defaultPrevented: false, target: element(), ...values });

test('in a review the arrows step and change hand, or move along a live hand, s shows the answer, Esc goes back, and the digits pick only from a live hand', () => {
  for (const [name, action] of [['ArrowLeft', 'previous'], ['ArrowRight', 'next'], ['Escape', 'back'], ['s', 'reveal'], ['S', 'reveal']]) {
    assert.equal(reviewKey(key(name)), action, name);
  }
  assert.equal(reviewKey(key('ArrowLeft', { shiftKey: true })), 'previous-hand');
  assert.equal(reviewKey(key('ArrowRight', { shiftKey: true })), 'next-hand');
  for (const name of ['1', '9', '0', 'a', 'Enter', ' ']) assert.equal(reviewKey(key(name)), null, `${name} with no hand to pick from`);
  // A live hand: the digits pick its tiles, 0 the tile just drawn, as in play.
  const tile = element(), hand = { contains: node => node === tile };
  assert.equal(reviewKey(key('1'), { hand }), 'pick-1');
  assert.equal(reviewKey(key('9'), { hand }), 'pick-9');
  assert.equal(reviewKey(key('0'), { hand }), 'pick-drawn');
  assert.equal(reviewKey(key('1', { shiftKey: true }), { hand }), null, 'a digit with Shift, as play leaves it');
  // In the hand the arrows move along it, as in play, and Shift still changes
  // hand; outside it they step.
  assert.equal(reviewKey(key('ArrowRight', { target: tile }), { hand }), 'next-tile');
  assert.equal(reviewKey(key('ArrowLeft', { target: tile }), { hand }), 'previous-tile');
  assert.equal(reviewKey(key('ArrowRight', { target: tile, shiftKey: true }), { hand }), 'next-hand');
  assert.equal(reviewKey(key('ArrowRight'), { hand }), 'next');
  assert.equal(reviewKey(key('s', { target: tile }), { hand }), 'reveal');
});

test("a review's keys are left alone with Ctrl, Alt or Meta held, while text is composed, once handled, in a field and under a dialog", () => {
  for (const values of [{ ctrlKey: true }, { altKey: true }, { metaKey: true }, { isComposing: true }, { defaultPrevented: true },
    { target: element({ field: true }) }]) {
    for (const name of ['ArrowRight', 'ArrowLeft', 'Escape', 's', '1']) {
      assert.equal(reviewKey(key(name, values), { hand: { contains: () => true } }), null, `${name} with ${JSON.stringify(Object.keys(values))}`);
    }
  }
  for (const name of ['ArrowRight', 'Escape', 's']) assert.equal(reviewKey(key(name), { dialogOpen: true }), null, `${name} under a dialog`);
  // Chrome sends a keydown with no key as it fills a field in for the player.
  assert.equal(reviewKey(key(undefined)), null);
  assert.equal(REVIEW_WIDE, '(min-width: 1100px) and (min-height: 600px)');
});

test("a held Enter presses a button of the review's step bar, of Trained AI's reasoning or of the review's controls once, and is left alone anywhere else", () => {
  // A button inside the element `around` names, as `closest` finds it,
  // matching the selector as the page's own does, any of its parts.
  const inside = around => ({ closest: selector => (selector.split(', ').includes(around) ? {} : null) });
  const bar = inside('[data-steps]'), reasoning = inside('[data-insight]');
  assert.equal(heldEnter(key('Enter', { repeat: true, target: bar })), true, 'held on the bar');
  // "What would change its mind?" and Cancel hand focus to each other, so a
  // held Enter would start and stop the search by turns.
  assert.equal(heldEnter(key('Enter', { repeat: true, target: reasoning })), true, "held on the reasoning's buttons");
  // Stop and Continue take each other's place, so a held Enter would stop and
  // start the review by turns.
  assert.equal(heldEnter(key('Enter', { repeat: true, target: inside('[data-review-controls]') })), true, "held on the review's controls");
  assert.equal(heldEnter(key('Enter', { repeat: false, target: bar })), false, 'pressed once, it presses');
  assert.equal(heldEnter(key('Enter', { repeat: false, target: reasoning })), false, 'pressed once, it presses there too');
  assert.equal(heldEnter(key('ArrowRight', { repeat: true, target: bar })), false, 'held arrows step on');
  assert.equal(heldEnter(key(' ', { repeat: true, target: reasoning })), false, 'Space presses as it is let go');
  assert.equal(heldEnter(key('Enter', { repeat: true })), false, 'held anywhere else');
  assert.equal(heldEnter(key('Enter', { repeat: true, target: null })), false, 'with nothing focused');
});

test("the review's controls, which take each other's place, let go of the second press of a double click or tap", () => {
  // The page's own handlers, from its source, pressed on a stand-in review.
  const source = readFileSync(new URL('../src/lib/GameReview.svelte', import.meta.url), 'utf8');
  const ast = parse(source, { modern: true });
  const all = (node, predicate, found = []) => {
    if (!node || typeof node !== 'object') return found;
    if (predicate(node)) found.push(node);
    for (const [name, value] of Object.entries(node)) if (name !== 'parent') all(value, predicate, found);
    return found;
  };
  const [controls] = all(ast.fragment, node => node.type === 'RegularElement' && node.attributes.some(a => a.name === 'data-review-controls'));
  assert.ok(controls, 'the controls are marked for heldEnter');
  const buttons = all(controls, node => node.type === 'RegularElement' && node.name === 'button');
  assert.deepEqual(buttons.map(button => button.attributes.find(a => a.name.startsWith('data-'))?.name),
    ['data-stop-review', 'data-next-hand', 'data-rest', 'data-continue']);
  const [again] = all(ast.instance, node => node.type === 'VariableDeclarator' && node.id.name === 'again'
    && node.init?.type === 'ArrowFunctionExpression');
  assert.ok(again);
  for (const button of buttons) {
    const pressed = [];
    const review = new Proxy({}, { get: (_, name) => () => pressed.push(name) });
    const context = vm.createContext({ review });
    context.again = vm.runInContext(source.slice(again.init.start, again.init.end), context);
    const handler = button.attributes.find(a => a.name === 'onclick').value.expression;
    const click = vm.runInContext(`(${source.slice(handler.start, handler.end)})`, context);
    click({ detail: 2 });
    assert.deepEqual(pressed, [], 'the second press of a double click');
    click({ detail: 1 });
    click({ detail: 0 });
    assert.equal(pressed.length, 2, 'a click, and Enter or Space');
  }
});
