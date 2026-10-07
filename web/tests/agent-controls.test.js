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

test('watch alternative confirmation pauses autoplay, cancels safely, and applies the requested choice once', () => {
  const declaration = find(watch.ast.instance, node => node.type === 'FunctionDeclaration' && node.id?.name === 'chooseAlternative');
  for (const accepted of [false, true]) {
    const calls = [], choice = { kind: 'discard', tile: '1m', label: 'discards the 1 characters' }, analysis = {};
    const owner = { analysis, autoplay: true, closed: false,
      setAutoplay(value) { this.autoplay = value; calls.push(value); },
      choose(actual, expected) { assert.equal(actual, choice); assert.equal(expected, analysis); calls.push('choose'); },
    };
    const state = { watch: owner, analysis, busy: false, confirm(message) {
      assert.equal(owner.autoplay, false, 'pause before opening the dialog');
      assert.match(message, /Play "discards the 1 characters"/); return accepted;
    } };
    const handler = vm.runInContext(`(${watch.source.slice(declaration.start, declaration.end)})`, vm.createContext(state));
    handler(choice);
    assert.deepEqual(calls, accepted ? [false, 'choose', true] : [false, true]);
    state.confirm = () => { state.analysis = owner.analysis = {}; return true; };
    calls.length = 0; handler(choice);
    assert.deepEqual(calls, [false], 'a changed position invalidates the confirmation');
  }
});

test('watch setup defaults every seat to Trained, and to Club only while the network is absent', () => {
  const effect = find(watch.ast.instance, node => node.type === 'CallExpression' && node.callee?.name === '$effect');
  const callback = effect.arguments[0];
  for (const [trainedAvailable, expected] of [[false, 'club'], [true, 'full']]) {
    // Play's table is not the watch's: a Club or mixed Play table changes nothing here.
    const state = { ready: true, configured: false, trainedAvailable,
      opponents: ['club', 'beginner', 'club'], lineup: [] };
    vm.runInContext(`(${watch.source.slice(callback.start, callback.end)})()`, vm.createContext(state));
    assert.deepEqual(Array.from(state.lineup), [expected, expected, expected, expected]);
    assert.equal(state.configured, trainedAvailable, 'only a Trained lineup is final; a Club one waits for the network');
  }
  // An edited or started draft is never replaced.
  const state = { configured: true, trainedAvailable: true, lineup: ['beginner', 'club', 'club', 'club'] };
  vm.runInContext(`(${watch.source.slice(callback.start, callback.end)})()`, vm.createContext(state));
  assert.deepEqual(state.lineup, ['beginner', 'club', 'club', 'club']);
});

test('a watch whose network cannot load deals the same table again with Club in the Trained seats', () => {
  const declaration = find(watch.ast.instance, node => node.type === 'FunctionDeclaration' && node.id?.name === 'watchWithClub');
  assert.ok(declaration);
  const started = [];
  const state = { lineup: ['full', 'full', 'full', 'full'], watch: { lineup: ['full', 'beginner', 'full', 'club'] },
    isTrained: agent => agent === 'full', start() { started.push([...state.lineup]); } };
  vm.runInContext(`(${watch.source.slice(declaration.start, declaration.end)})()`, vm.createContext(state));
  assert.deepEqual(started, [['club', 'beginner', 'club', 'club']], 'the failed game\'s own seats, Trained ones as Club');
  // Offered beside Retry agent, and only when a seat is Trained.
  assert.match(watch.source, /\{#if watch\?\.lineup\.some\(isTrained\)\}<button data-watch-club[^>]*onclick=\{watchWithClub\}>Watch with Club instead<\/button>/);
});

test('a new guided game keeps the adviser the last one used', async () => {
  const guided = component('../src/lib/GuidedPlay.svelte');
  const declaration = find(guided.ast.instance, node => node.type === 'FunctionDeclaration' && node.id?.name === 'restart');
  const { emptyGuided } = await import('../src/lib/guided-game.js');
  for (const agent of ['club', 'beginner', 'full']) {
    const state = { loaded: true, conflict: '', unreadable: false, failure: 'old', game: null, emptyGuided,
      state: { agent }, window: { confirm: () => true } };
    await vm.runInContext(`(${guided.source.slice(declaration.start, declaration.end)})()`, vm.createContext(state));
    assert.equal(state.game.state.agent, agent);
    assert.equal(state.game.state.stage, 'setup');
    assert.equal(state.failure, '');
  }
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
      // Svelte's untrack, which the effect uses to keep its dependencies to readiness and the mode.
      untrack: task => task(),
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
