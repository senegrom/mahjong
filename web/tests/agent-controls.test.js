import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { parse } from 'svelte/compiler';
import { emptyPosition, withoutStaleDraw } from '../src/lib/physical-position.js';

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

test('physical play keeps a Trained agent in its select, disabled, while the network is unavailable', () => {
  const select = find(physical.ast.fragment, node => node.name === 'select'
    && node.attributes?.some(a => a.name === 'aria-label' && a.value?.[0]?.data === 'Physical play agent'));
  const shown = find(select, node => node.type === 'IfBlock');
  const option = find(shown.consequent, node => node.name === 'option');
  const disabled = option.attributes.find(a => a.name === 'disabled')?.value?.expression;
  const run = (expression, state) => vm.runInContext(physical.source.slice(expression.start, expression.end),
    vm.createContext({ isTrained: key => key === 'full', ...state }));
  for (const [agent, listed] of [['full', true], ['club', false]]) {
    const state = { key: 'full', agent, trainedAvailable: false };
    assert.equal(run(shown.test, state), listed, `the Trained option is listed only while it is the pick (${agent})`);
    assert.ok(disabled && run(disabled, state), 'and cannot be picked again while unavailable');
  }
  assert.equal(run(shown.test, { key: 'full', agent: 'club', trainedAvailable: true }), true);
  assert.equal(run(disabled, { key: 'full', agent: 'club', trainedAvailable: true }), false);
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

test('changing your seat at a guided setup takes the points you typed to the new wind', () => {
  const guided = component('../src/lib/GuidedPlay.svelte');
  const game = { position: emptyPosition() };
  const state = { edit: fn => fn(game) };
  game.position.players[0].score = 25000;
  game.position.players[2].score = 28000;
  change(guided, 'Your seat', state, '1');
  assert.equal(game.position.seat, 1);
  assert.deepEqual(game.position.players.map(player => player.score), [30000, 25000, 28000, 30000]);
  change(guided, 'Your seat', state, '1');
  assert.deepEqual(game.position.players.map(player => player.score), [30000, 25000, 28000, 30000], 'the same seat moves nothing');
});

test('the typed-in modes ask the engine only as it takes what a draft keeps of a riichi and a kan', () => {
  // A draft keeps two facts the engine is told only where it and its replay
  // take them (analysisOf): a table handed to the engine any other way would
  // be told none, or be refused for one an edit made unfit.
  for (const path of ['../src/lib/PhysicalPlay.svelte', '../src/lib/GuidedPlay.svelte']) {
    const { source } = component(path);
    assert.doesNotMatch(source, /new PhysicalAnalysis\(/, `${path} builds no engine of its own`);
    assert.match(source, /analysisOf\(PhysicalAnalysis, /, path);
  }
});

test('physical play\'s editor leaves a riichi it moved no draw of the turn it left, and any other change its draw', () => {
  // Every change to the table goes through the component's own edit, run
  // here as it is written.
  const declaration = find(physical.ast.instance.content, node => node.type === 'FunctionDeclaration' && node.id?.name === 'edit');
  assert.ok(declaration, 'the table is changed through edit');
  const state = { closed: false, loaded: true, saveConflict: false, unreadable: false, history: [], numericEdit: null, failure: '',
    position: emptyPosition(), withoutStaleDraw, structuredClone, JSON };
  state.snapshot = () => JSON.parse(JSON.stringify(state.position));
  const edit = vm.runInContext(`(${physical.source.slice(declaration.start, declaration.end)})`, vm.createContext(state));
  Object.assign(state.position.players[0], { riichi: 'riichi', drew_at_riichi: '1m', discards: [
    { tile: '9p', order: 0, drawn: false, riichi: true, claimed: false }, { tile: '4z', order: 4, drawn: true, riichi: false, claimed: false }] });
  assert.equal(edit(p => { p.players[0].hand = ['1m']; }), true);
  assert.equal(state.position.players[0].drew_at_riichi, '1m');
  assert.equal(edit(p => { p.players[0].discards[0].riichi = false; p.players[0].discards[1].riichi = true; }), true);
  assert.equal(state.position.players[0].drew_at_riichi, undefined);
  assert.equal(state.history.at(-1).players[0].drew_at_riichi, '1m', 'Undo brings the table back with it');
});

test("Watch, a guided game, physical play and both reviews show the worker's own download as Play's table does", async () => {
  const { waitingNote } = await import('../src/lib/ui.js');
  const app = component('../src/App.svelte');
  const note = 'downloading the network 37%';
  for (const [name, tag] of [['AgentWatch', 'AgentWatch'], ['GuidedPlay', 'GuidedPlay'], ['PhysicalPlay', 'PhysicalPlay'], ['Review', 'Review'],
    ['GameReview', 'GameReview']]) {
    const { source, ast } = component(`../src/lib/${name}.svelte`);
    const download = find(ast.instance, node => node.type === 'VariableDeclarator' && node.id.name === 'download');
    const shown = download.init.arguments[0];
    const state = { waitingNote, workerNote: note, offline: null, isTrained: agent => agent === 'full',
      state: { agent: 'full' }, agent: 'full', watch: { lineup: ['full', 'club', 'club', 'club'] } };
    assert.equal(vm.runInContext(source.slice(shown.start, shown.end), vm.createContext(state)), note, name);
    // The page hands each the note.
    const used = find(app.ast.fragment, node => node.type === 'Component' && node.name === tag);
    assert.ok(used?.attributes.some(attribute => attribute.name === 'workerNote'), `App gives ${tag} the worker's note`);
  }
  assert.match(app.source, /reportLoading\(\(note\) => \{ if \(mounted\) workerNote = note; \}\);/);
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

test("a guided step hands focus on to the next prompt's heading, and a refused one leaves it where it was pressed", async () => {
  // Each step draws the next prompt in place of the one pressed, so the
  // pressed control is gone: the component's own step and arrive are run
  // here as they are written, the table event, the page's tick and land
  // stood in for.
  const guided = component('../src/lib/GuidedPlay.svelte');
  const written = name => {
    const declaration = find(guided.ast.instance, node => node.type === 'FunctionDeclaration' && node.id?.name === name);
    return guided.source.slice(declaration.start, declaration.end);
  };
  const landed = [], heading = { name: 'the prompt heading' };
  const state = { closed: false, promptHeading: null, taken: true, act: () => state.taken,
    // The next prompt, and its heading, are drawn by the time the tick is over.
    tick: async () => { state.promptHeading = heading; },
    pressedByKeyboard: press => press?.detail === 0, land: (...where) => landed.push(where) };
  const step = vm.runInContext(`${written('arrive')}\n${written('step')}\nstep`, vm.createContext(state));
  const settled = () => new Promise(resolve => setImmediate(resolve));
  // A keyboard's press moves the page to show the heading; a tap does not.
  assert.equal(step({ type: 'setup' }, { detail: 0 }), true); await settled();
  assert.equal(step({ type: 'hand' }, { detail: 1 }), true); await settled();
  assert.deepEqual(landed, [[heading, true], [heading, false]]);
  // A refused step draws no new prompt: focus stays beside the line that says why.
  state.taken = false;
  assert.equal(step({ type: 'hand' }, { detail: 0 }), false); await settled();
  // Nor is focus moved from another mode's page, once this one has been left.
  state.taken = true; state.closed = true;
  step({ type: 'hand' }, { detail: 0 }); await settled();
  assert.equal(landed.length, 2);
  // Every press that takes a step goes through it, with its click.
  assert.doesNotMatch(guided.source.slice(guided.ast.instance.end), /\bact\(/);
  assert.equal((guided.source.match(/<h3 tabindex="-1" bind:this=\{promptHeading\}>/g) ?? []).length,
    (guided.source.match(/<h3[ >]/g) ?? []).length, 'every prompt heading can take focus');
});

/** The page's own effect that restores the saved match as Play is first
 * entered, from its source, run on a stand-in for the page's state: what it
 * calls goes to `calls`, and `values` replaces or adds to the state. */
async function restoring(calls, saved, values = {}) {
  const { standInFor } = await import('../src/lib/opponents.js');
  const effect = find(app.ast.instance, node => node.type === 'CallExpression' && node.callee?.name === '$effect'
    && app.source.slice(node.start, node.end).includes('playStarted'));
  assert.ok(effect);
  const callback = effect.arguments[0];
  const state = { ready: true, mode: 'physical', playStarted: false,
    matchStore: { read() { calls.push('read'); return saved; } },
    start() { calls.push('start'); }, update() {}, Game: {}, callbacks: {},
    restored: { opponents: ['neural', 'neural', 'neural'], over: false },
    MatchSession: { restore(_Game, text) {
      assert.equal(text, 'saved-match'); calls.push('restore');
      return { ...state.restored, run() { calls.push('run'); } };
    } },
    downloadAi() { calls.push('download'); }, session: null, notice: '', failure: '',
    // What a restored match is read against, and where a finished one lands
    // on a phone: no table of the player's own in the record, and no phone.
    standInFor, storedTable: null, returnTable: undefined, matchMedia: () => ({ matches: false }), RESULT_SHEET: '(max-width: 760px)',
    tick: () => Promise.resolve(), land(...args) { calls.push(['land', ...args]); }, standingsHeading: 'the heading',
    // Svelte's untrack, which the effect uses to keep its dependencies to readiness and the mode.
    untrack: task => task(), ...values,
  };
  const context = vm.createContext(state);
  return { state, run: vm.runInContext(`(${app.source.slice(callback.start, callback.end)})`, context) };
}

test('direct agent-mode entry leaves regular saves untouched until Play is opened', async () => {
  for (const saved of [null, 'saved-match']) {
    const calls = [], { state, run } = await restoring(calls, saved);
    run(); state.mode = 'watch'; run();
    assert.deepEqual(calls, []);
    state.mode = 'play'; run();
    const expected = saved ? ['read', 'restore', 'download', 'run'] : ['read', 'start'];
    assert.deepEqual(calls, expected);
    assert.equal(state.failure, '');
    state.mode = 'physical'; run(); state.mode = 'play'; run();
    assert.deepEqual(calls, expected, 'switching back keeps the same regular session');
  }
});

test('a restored match is a stand-in only against a table the record itself holds, and a finished one lands on its standings on a phone', async () => {
  const trained = ['neural', 'neural', 'neural'], club = ['club', 'club', 'club'];
  // Club where the player's stored table says Trained: the table stays their
  // choice, and the network they wait for goes on being fetched.
  let calls = [], { state, run } = await restoring(calls, 'saved-match', { mode: 'play', storedTable: trained, restored: { opponents: club, over: false } });
  run();
  assert.deepEqual(state.returnTable, trained);
  assert.deepEqual(calls, ['read', 'restore', 'download', 'run']);
  // No table in the record, as after a reset or under an address that asks
  // for one: the match's own table stands, and Club asks for no network.
  calls = [];
  ({ state, run } = await restoring(calls, 'saved-match', { mode: 'play', restored: { opponents: club, over: false } }));
  run();
  assert.equal(state.returnTable, null);
  assert.deepEqual(calls, ['read', 'restore', 'run']);
  // The same table is no stand-in.
  ({ state, run } = await restoring([], 'saved-match', { mode: 'play', storedTable: trained }));
  run();
  assert.equal(state.returnTable, null);
  // A finished match where the result is a sheet over the standings opens
  // on them: focus and the top of the screen go to their heading, once the
  // page has drawn them. Elsewhere the result takes focus itself.
  for (const [sheet, over, landed] of [[true, true, true], [true, false, false], [false, true, false]]) {
    calls = [];
    ({ state, run } = await restoring(calls, 'saved-match', { mode: 'play', restored: { opponents: club, over },
      matchMedia: query => ({ matches: sheet && query === '(max-width: 760px)' }) }));
    run();
    await Promise.resolve();
    assert.deepEqual(calls.filter(Array.isArray), landed ? [['land', 'the heading', true, 'start']] : [], `sheet ${sheet}, over ${over}`);
    assert.equal(state.failure, '');
  }
});
