import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { parse } from 'svelte/compiler';
import { emptyPosition } from '../src/lib/physical-position.js';

function component(path) {
  const source = readFileSync(new URL(path, import.meta.url), 'utf8');
  return { source, ast: parse(source, { modern: true }) };
}
function find(node, predicate) {
  if (!node || typeof node !== 'object') return null;
  if (predicate(node)) return node;
  for (const value of Object.values(node)) {
    const result = find(value, predicate);
    if (result) return result;
  }
  return null;
}
const physical = component('../src/lib/PhysicalPlay.svelte');
const watch = component('../src/lib/AgentWatch.svelte');
const app = component('../src/App.svelte');

// Execute the actual inline event handlers with the OLD component state and
// the NEW selected value. Svelte dispatches onchange before updating bindings.
function change({ source, ast }, label, state, value) {
  const select = find(ast.fragment, node => node.name === 'select'
    && node.attributes?.some(a => a.name === 'aria-label' && a.value?.[0]?.data === label));
  assert.ok(select, `Missing select: ${label}`);
  const expression = select.attributes.find(a => a.name === 'onchange')?.value?.expression;
  assert.ok(expression, `Missing change handler: ${label}`);
  const context = vm.createContext(state);
  const handler = vm.runInContext(`(${source.slice(expression.start, expression.end)})`, context);
  handler({ currentTarget: { value } });
}
function editor() {
  const state = { position: emptyPosition(), index: 0, slot: 0 };
  state.edit = fn => {
    const next = structuredClone(state.position);
    state.position = fn(next) ?? next;
  };
  return state;
}

test('changing the analysed seat uses the new wind and clears the previous turn tile', () => {
  const state = editor(); state.position.drawn = '1m';
  change(physical, 'Analyse seat', state, '2');
  assert.equal(state.position.seat, 2);
  assert.equal(state.position.turn, 2);
  assert.equal(state.position.drawn, null);
  state.position.phase = 'call'; state.position.turn = 3; state.position.pending = '5z';
  change(physical, 'Analyse seat', state, '1');
  assert.equal(state.position.seat, 1);
  assert.equal(state.position.turn, 3, 'changing the respondent preserves the discard source');
  assert.equal(state.position.pending, '5z');
});

test('selecting a draw replaces the old call context and clearing it stores null', () => {
  const state = editor(); state.position.just_claimed = '5z';
  change(physical, 'Drawn tile', state, '9m');
  assert.equal(state.position.drawn, '9m');
  assert.equal(state.position.just_claimed, null);
  change(physical, 'Drawn tile', state, '');
  assert.equal(state.position.drawn, null);
});

test('changing the meld kind immediately applies the correct call direction', () => {
  const state = editor();
  state.position.players[0].melds.push({ kind: 'pon', tile: '1m', from: 1 });
  change(physical, 'Set kind', state, 'concealed-kan');
  assert.equal(state.position.players[0].melds[0].from, 0);
  change(physical, 'Set kind', state, 'pon');
  assert.equal(state.position.players[0].melds[0].from, 3);
  state.position.players[0].melds[0].from = 1;
  change(physical, 'Set kind', state, 'chii');
  assert.equal(state.position.players[0].melds[0].from, 3);
});

test('watch pace reschedules with the newly selected delay', () => {
  const delays = [];
  const state = { speed: 1400, watch: { delay: 1400, schedule() { delays.push(this.delay); } } };
  change(watch, 'Watch pace', state, '3000');
  change(watch, 'Watch pace', state, '700');
  assert.deepEqual(delays, [3000, 700]);
  assert.equal(state.speed, 700);
});

test('direct agent-mode entry leaves regular saves untouched until Play is opened', () => {
  const effect = find(app.ast.instance, node => node.type === 'CallExpression' && node.callee?.name === '$effect'
    && app.source.slice(node.start, node.end).includes('playStarted'));
  assert.ok(effect);
  const callback = effect.arguments[0];
  for (const saved of [null, 'saved-match']) {
    const calls = [];
    const state = { ready: true, mode: 'physical', playStarted: false,
      matchStore: { read() { calls.push('read'); return saved; } },
      start() { calls.push('start'); }, update() {}, Game: {}, callbacks: {},
      MatchSession: { restore(_Game, text) {
        assert.equal(text, 'saved-match'); calls.push('restore');
        return { opponents: ['neural'], run() { calls.push('run'); } };
      } },
      downloadAi() { calls.push('download'); }, session: null, notice: '', failure: '',
    };
    const context = vm.createContext(state);
    const run = vm.runInContext(`(${app.source.slice(callback.start, callback.end)})`, context);
    run(); state.mode = 'watch'; run();
    assert.deepEqual(calls, []);
    state.mode = 'play'; run();
    const expected = saved ? ['read', 'restore', 'download', 'run'] : ['read', 'start'];
    assert.deepEqual(calls, expected);
    state.mode = 'physical'; run(); state.mode = 'play'; run();
    assert.deepEqual(calls, expected, 'switching back keeps the same regular session');
  }
});
