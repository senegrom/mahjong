import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse } from 'svelte/compiler';

// A number field whose value is written back after every keystroke loses a
// leading minus: the browser reports the lone "-" as an empty value, the
// field stores null, and Svelte writes the empty value back, so "-500" became
// 500. A function binding compares the bound value with the field's own
// reading and leaves the partial input alone, as PhysicalPlay's fields do.
// Negative scores are legal: setup accepts down to -100000.
const source = readFileSync(new URL('../src/lib/GuidedPlay.svelte', import.meta.url), 'utf8');

function inputs(node, found = []) {
  if (Array.isArray(node)) { for (const child of node) inputs(child, found); return found; }
  if (!node || typeof node !== 'object') return found;
  if (node.type === 'RegularElement' && node.name === 'input') found.push(node);
  for (const [key, value] of Object.entries(node)) if (key !== 'parent') inputs(value, found);
  return found;
}
const text = attribute => Array.isArray(attribute.value) ? attribute.value.map(part => part.data ?? '').join('') : attribute.value;
const numbers = inputs(parse(source, { modern: true }).fragment)
  .filter(input => input.attributes.some(a => a.type === 'Attribute' && a.name === 'type' && text(a) === 'number'));

test('every guided number field binds its value, so a typed minus sign survives', () => {
  assert.equal(numbers.length, 5, 'your points, hand number, honba, riichi sticks, other and settled points');
  for (const input of numbers) {
    const snippet = source.slice(input.start, input.end);
    assert.equal(input.attributes.some(a => a.type === 'Attribute' && ['value', 'oninput'].includes(a.name)), false, snippet);
    const binding = input.attributes.find(a => a.type === 'BindDirective' && a.name === 'value');
    assert.ok(binding, snippet);
    assert.equal(binding.expression.type, 'SequenceExpression', `a getter and a setter: ${snippet}`);
    const [getter, setter] = binding.expression.expressions;
    assert.equal(getter.type, 'ArrowFunctionExpression');
    // A blank or partial field is kept as a saved null, never as 0.
    assert.match(source.slice(setter.start, setter.end), /\?\? null/);
  }
});
