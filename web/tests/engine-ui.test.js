import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const fixture = JSON.parse(readFileSync(new URL('./fixtures/duplicate-indicators.json', import.meta.url)));

function make(seed = 369, difficulty = 'club') { const match = new MatchSession(Game, seed, difficulty); match.advance(false); return match; }
function step(match, choice) { match.apply({type:'choose',kind:choice.kind,tile:choice.tile}); match.advance(false); }

test('duplicate physical indicators are valid and produce global multipliers', () => {
  const match = make(); for (const choice of fixture.actions) step(match, choice);
  assert.deepEqual(match.view.dora_indicators, ['7z','7z']);
  assert.deepEqual(match.view.dora_types, ['5z','5z']);
  assert.equal(match.view.kyoku, 1);
  const restored = MatchSession.restore(Game, JSON.stringify(match.snapshot()));
  assert.equal(restored.stateKey(), match.stateKey()); restored.dispose(); match.dispose();
});

test('discarding your last dora does not change global dora markings', () => {
  let found = false;
  for (let seed = 1; seed <= 500 && !found; seed++) {
    const match = make(seed); const view = match.view;
    const choice = match.choices.find((c) => c.kind === 'discard' && view.dora.includes(c.tile));
    if (choice) {
      const before = view.dora_types.slice(); step(match, choice);
      assert.ok(match.view.dora_types.includes(choice.tile));
      if (match.view.dora_indicators.length === view.dora_indicators.length) assert.deepEqual(match.view.dora_types, before);
      const notes = match.engine.review(); assert.deepEqual(notes.at(-1).dora_types, before);
      found = true;
    }
    match.dispose();
  }
  assert.ok(found, 'A real dora discard was exercised');
});

test('neural fallback preserves the pending position, scores and seat', () => {
  const match = make(1,'neural');
  while (!match.engine.needs_opponent_move()) step(match, match.choices.find(c=>c.kind==='discard') ?? match.choices[0]);
  const before = match.view;
  assert.equal(before.phase, 'act');
  match.apply({type:'club'});
  assert.deepEqual(match.view.seats.slice(1).map(seat => seat.controller), ['club','club','club']);
  const physicalState = view => JSON.parse(JSON.stringify(view, (key, value) => key === 'controller' ? undefined : value));
  assert.deepEqual(physicalState(match.view), physicalState(before), 'Switching the controller does not redeal');
  assert.equal(match.engine.needs_opponent_move(), false);
  match.advance(false);
  assert.ok(match.choices.length > 0 || match.view.phase === 'over');
  const restored = MatchSession.restore(Game, JSON.stringify(match.snapshot()));
  assert.equal(restored.stateKey(), match.stateKey()); restored.dispose(); match.dispose();
});

test('real neural decisions replay without running inference again', () => {
  const match = make(1,'neural');
  for (let n=0; n<25 && match.view.phase !== 'over'; n++) {
    if (match.engine.needs_opponent_move()) {
      match.apply({type:'opponent',action:match.engine.opponent_mask().findIndex(Boolean)});
      match.advance(false);
    } else step(match, match.choices.find(c=>c.kind==='discard') ?? match.choices[0]);
  }
  const restored = MatchSession.restore(Game, JSON.stringify(match.snapshot()), {ai(){throw new Error('Inference must not run during replay');}});
  assert.equal(restored.stateKey(), match.stateKey()); restored.dispose(); match.dispose();
});

test('an entire match including repeat deals and standings restores exactly', () => {
  const match = make(14,'beginner'); let steps = 0;
  while (!match.over && steps++ < 3500) {
    if (match.engine.hand_is_over()) {
      assert.equal(match.progressed, true, 'Between-hand abandonment needs confirmation');
      match.apply({type:'next'}); match.advance(false);
    } else {
      const choices = match.choices;
      step(match, choices.find(c=>c.kind==='ron'||c.kind==='tsumo') ?? choices.find(c=>c.kind==='pass') ?? choices.find(c=>c.kind==='discard') ?? choices[0]);
    }
  }
  assert.ok(match.over, 'Match completed within the deterministic bound');
  const restored = MatchSession.restore(Game, JSON.stringify(match.snapshot()));
  assert.equal(restored.stateKey(), match.stateKey());
  assert.deepEqual(restored.engine.standings(), match.engine.standings());
  assert.throws(()=>restored.apply({type:'next'}));
  restored.dispose(); match.dispose();
});

test('Club recovery resolves a pending opponent claim and restores exactly', () => {
  let found = false;
  for (let seed = 1; seed <= 20 && !found; seed++) {
    const match = make(seed, 'neural');
    for (let n = 0; n < 200 && match.view.phase !== 'over'; n++) {
      if (match.view.phase === 'call' && match.engine.needs_opponent_move()) {
        const wall = match.view.wall;
        match.apply({type:'club'}); match.advance(false);
        assert.equal(match.engine.needs_opponent_move(), false);
        assert.ok(match.view.wall <= wall, 'Recovery must not deal a fresh wall');
        assert.ok(match.choices.length || match.view.phase === 'over');
        const restored = MatchSession.restore(Game, JSON.stringify(match.snapshot()));
        assert.equal(restored.stateKey(), match.stateKey());
        restored.dispose(); found = true; break;
      }
      if (match.engine.needs_opponent_move()) {
        match.apply({type:'opponent',action:match.engine.opponent_mask().findIndex(Boolean)});
        match.advance(false);
      } else step(match, match.choices.find(c=>c.kind==='pass') ?? match.choices.find(c=>c.kind==='discard') ?? match.choices[0]);
    }
    match.dispose();
  }
  assert.ok(found, 'Exercised an actual external claim window');
});
