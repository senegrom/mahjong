import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis } from '../src/wasm/riichi.js';
import { policyWeights } from '../src/lib/policy-weights.js';
import { WatchSession } from '../src/lib/watch-session.js';
import { emptyPosition, parseTiles, recordChoice, recordDraw, recordDiscard, readPhysical } from '../src/lib/physical-position.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

const basic = () => {
  const p = emptyPosition();
  p.players[0].hand = parseTiles('123m456p789s11234z');
  p.drawn = '4z'; p.indicators = ['5z'];
  return p;
};
const callPosition = (hand = '123m456p789s1155z', pending = '5z') => {
  const p = basic(); p.players[0].hand = parseTiles(hand);
  p.phase = 'call'; p.turn = 3; p.pending = pending; p.drawn = null; p.indicators = ['7z']; p.first_turns = false;
  p.players[3].discards = [{ tile: pending, order: 0, drawn: true, riichi: false, claimed: false }];
  return p;
};
function inspect(p, fn) { const a = new PhysicalAnalysis(p); try { return fn(a); } finally { a.free(); } }
const pick = mask => mask[69] ? 69 : mask[68] ? 68 : mask[70] ? 70 : mask.findIndex(Boolean);
const builtin = async (engine, agent) => ({ choice: engine.agent_pick(agent), choices: engine.agent_choices() });

test('policy percentages are finite, normalized and masked, even for very large logits', () => {
  const result = policyWeights([10000, 10000 + Math.log(3), 1e30], [1, 1, 0]);
  assert.equal(result.action, 1);
  assert.ok(Math.abs(result.weights[0] - .25) < 1e-12);
  assert.ok(Math.abs(result.weights[1] - .75) < 1e-12);
  assert.equal(result.weights[2], 0);
  assert.throws(() => policyWeights([NaN], [1]), /invalid weights/);
  assert.throws(() => policyWeights([Infinity], [1]), /invalid weights/);
  assert.throws(() => policyWeights([1], [0]), /no legal choices/);
  assert.throws(() => policyWeights([1, 2], [1]), /wrong number/);
});

test('physical positions reject impossible tiles, incomplete hands, calls and history without trapping WASM', () => {
  const invalid = [
    p => { p.players[0].hand.pop(); },
    p => { p.players[1].hand = Array(13).fill('1m'); },
    p => { p.drawn = '9z'; },
    p => { p.drawn = '9m'; },
    p => { p.indicators = []; },
    p => { p.wall = 71; },
    p => { p.seat = 9; },
    p => { p.players[0].riichi = 'riichi'; },
    p => { p.players[0].ippatsu = true; },
    p => { p.players[1].melds = [{ kind: 'chii', tile: '9m', from: 3 }]; },
    p => { p.players[1].melds = [{ kind: 'chii', tile: '1z', from: 3 }]; },
    p => { p.players[1].melds = [{ kind: 'pon', tile: '1m', from: 0 }]; },
    p => { p.players[1].discards = Array.from({ length: 100 }, (_, order) => ({ tile: '1m', order, drawn: false, claimed: false, riichi: false })); },
  ];
  for (const mutate of invalid) { const p = basic(); mutate(p); assert.throws(() => new PhysicalAnalysis(p)); }
  inspect(basic(), a => assert.ok(a.agent_choices().some(c => c.kind === 'discard')));
});

test('manually entered starting positions have exactly the same observation and legal mask as live play', () => {
  for (const seed of [1, 2, 3, 7, 19, 31, 81, 287]) {
    const g = new Game(seed, 'club');
    try {
      g.advance();
      for (let turn = 0; turn < 12 && !g.hand_is_over(); turn++) {
        const view = g.view();
        if (view.phase === 'act') {
          const p = emptyPosition(), winds = ['east','south','west','north'];
          p.seat = winds.indexOf(view.seats[0].seat); p.turn = p.seat;
          p.round = winds.indexOf(view.round); p.kyoku = view.kyoku; p.wall = view.wall;
          p.counters = view.counters; p.riichi_sticks = view.riichi_sticks; p.indicators = view.dora_indicators;
          p.drawn = view.seats[0].drawn; p.first_turns = false;
          const events = g.log().split('\n').map(line => JSON.parse(line));
          const discards = events.filter(e => e.type === 'dahai');
          for (const seat of view.seats) {
            const index = winds.indexOf(seat.seat), player = p.players[index];
            player.hand = [...seat.hand, ...(seat.drawn ? [seat.drawn] : [])]; player.score = seat.score;
            player.melds = seat.melds.map(m => ({ kind: m.kind === 'claimed-kan' ? 'kan' : m.kind, tile: m.tiles[0], from: ['self','right','across','left'].indexOf(m.from) }));
            player.riichi = seat.riichi ? 'riichi' : 'none';
            let used = 0;
            player.discards = discards.map((e, order) => ({ e, order })).filter(({ e }) => e.actor === seat.player)
              .map(({ order }) => ({ ...seat.discards[used++], order }));
          }
          // No calls made by the followed bot in this test: every turn has a draw.
          assert.ok(p.drawn);
          inspect(p, a => {
            assert.deepEqual(a.agent_observation(), g.agent_observation(), `observation, seed ${seed}, turn ${turn}`);
            assert.deepEqual(a.agent_mask(), g.agent_mask(), `mask, seed ${seed}, turn ${turn}`);
          });
        }
        const choices = g.choices();
        const choice = choices.find(c => c.kind === 'pass') ?? choices.find(c => c.kind === 'discard') ?? choices[0];
        g.choose(choice.kind, choice.tile ?? undefined); g.advance();
      }
    } finally { g.free(); }
  }
});

test('a physical pon consumes only held tiles, preserves the claimed discard and offers legal follow-up discards', () => {
  const p = callPosition();
  p.players[0].furiten = true;
  const choices = inspect(p, a => a.agent_choices());
  const pon = choices.find(c => c.kind === 'pon'); assert.ok(pon);
  const next = recordChoice(p, pon, choices);
  assert.equal(next.players[0].hand.length, 11);
  assert.equal(next.players[3].discards[0].claimed, true);
  assert.equal(next.players[0].melds[0].from, 3);
  assert.equal(next.just_claimed, '5z'); assert.equal(next.drawn, null);
  assert.equal(next.players[0].furiten, false, 'taking a call clears temporary furiten');
  inspect(next, a => { assert.ok(a.agent_choices().some(c => c.kind === 'discard')); assert.equal(a.agent_choices().some(c => c.kind === 'riichi'), false); });
  assert.equal(p.players[0].hand.length, 13, 'recording does not mutate the previous position');
});

test('ron, pass, furiten, tsumo and riichi use the engine legality for a physical table', () => {
  const p = callPosition('123m456p789s1112z', '2z');
  p.players[0].hand = parseTiles('123m456p789s1112z');
  const choices = inspect(p, a => a.agent_choices());
  assert.ok(choices.some(c => c.kind === 'ron'));
  const next = recordChoice(p, choices.find(c => c.kind === 'pass'), choices);
  assert.equal(next.pending, '2z', 'other players can still respond to this discard');
  assert.equal(next.players[0].furiten, true);
  inspect(next, a => assert.equal(a.agent_choices().some(c => c.kind === 'ron'), false));
  const drawn = recordDraw(next, 0, '2z');
  inspect(drawn, a => assert.ok(a.agent_choices().some(c => c.kind === 'tsumo')));
  const ready = basic(); ready.players[0].hand = parseTiles('123m456p789s11123z'); ready.drawn = '3z';
  const options = inspect(ready, a => a.agent_choices());
  const riichi = options.find(c => c.kind === 'riichi'); assert.ok(riichi);
  const declared = recordChoice(ready, riichi, options);
  assert.equal(declared.riichi_sticks, 1); assert.equal(declared.players[0].score, 29000);
  assert.equal(declared.players[0].riichi, 'double', 'an unbroken first-turn declaration is double riichi');
  assert.ok(declared.players[0].discards.at(-1).riichi);
});

test('passing a complete shape without yaku still marks temporary furiten', () => {
  const p = callPosition('2245m456p789s', '6m');
  p.players[0].melds = [{ kind: 'chii', tile: '1m', from: 3 }];
  const choices = inspect(p, a => a.agent_choices());
  assert.equal(choices.some(c => c.kind === 'ron'), false, 'the complete open hand has no yaku');
  const pass = choices.find(c => c.kind === 'pass');
  assert.equal(pass.causes_furiten, true);
  const next = recordChoice(p, pass, choices);
  assert.equal(next.players[0].furiten, true);
  assert.equal(recordDraw(next, 0, '8p').players[0].furiten, false);
});

test('recording a call follows edited discard chronology rather than array order', () => {
  const p = callPosition();
  p.players[3].discards[0].order = 2;
  p.players[3].discards.push({ tile: '9m', order: 0, drawn: true, riichi: false, claimed: false });
  p.players[1].discards.push({ tile: '8m', order: 1, drawn: true, riichi: false, claimed: false });
  const choices = inspect(p, a => a.agent_choices());
  const next = recordChoice(p, choices.find(c => c.kind === 'pon'), choices);
  assert.equal(next.players[3].discards[0].claimed, true);
  assert.equal(next.players[3].discards[1].claimed, false);
  inspect(next, a => assert.ok(a.agent_choices().some(c => c.kind === 'discard')));
});

test('kan robbery preserves ippatsu until the replacement and requires the correct kan kind', () => {
  const p = basic(); p.players[0].hand = parseTiles('1111m456p789s1123z'); p.drawn = '1m';
  p.players[1].riichi = 'riichi'; p.players[1].ippatsu = true;
  p.players[1].discards.push({ tile: '8p', order: 0, drawn: false, riichi: true, claimed: false });
  const choices = inspect(p, a => a.agent_choices());
  const announced = recordChoice(p, choices.find(c => c.kind === 'concealed-kan'), choices);
  assert.equal(announced.players[1].ippatsu, true);
  const respondent = structuredClone(announced);
  respondent.seat = 2; respondent.players[2].hand = parseTiles('234567m123p3456s');
  inspect(respondent, a => assert.deepEqual(a.agent_choices().map(c => c.kind), ['pass']));
  respondent.pending_kind = 'extended-kan';
  assert.throws(() => new PhysicalAnalysis(respondent), /tile and kind/);
  const fifth = callPosition('123456p789s1122z', '5m');
  fifth.pending_kind = 'concealed-kan';
  fifth.players[1].melds = ['1m', '2m', '3m', '4m'].map(tile => ({ kind: 'concealed-kan', tile, from: 0 }));
  fifth.players[3].melds = [{ kind: 'concealed-kan', tile: '5m', from: 0 }];
  fifth.players[3].discards = [];
  fifth.indicators = ['5z', '6z', '7z', '8p', '9p'];
  assert.throws(() => new PhysicalAnalysis(fifth), /At most four kans/);
  const next = recordDraw(announced, 0, '4z');
  assert.equal(next.players[1].ippatsu, false);
  assert.equal(next.after_quad, true);
  assert.equal(next.wall, p.wall - 1);
  next.indicators.push('6z');
  inspect(next, a => assert.ok(a.agent_choices().some(c => c.kind === 'discard')));
});

test('physical chii direction, completed kan indicators, and robbery windows are enforced', () => {
  const p = callPosition('12m456p789s11223z', '3m');
  assert.ok(inspect(p, a => a.agent_choices()).some(c => c.kind === 'chii'));
  p.players[2].discards = p.players[3].discards; p.players[3].discards = []; p.turn = 2;
  assert.equal(inspect(p, a => a.agent_choices()).some(c => c.kind === 'chii'), false);
  const kan = basic(); kan.players[0].hand = parseTiles('1111m456p789s1123z'); kan.drawn = '1m';
  const choices = inspect(kan, a => a.agent_choices());
  const choice = choices.find(c => c.kind === 'concealed-kan'); assert.ok(choice);
  const announced = recordChoice(kan, choice, choices);
  assert.equal(announced.pending_kind, 'concealed-kan'); assert.equal(announced.phase, 'call');
  const replacement = recordDraw(announced, 0, '4z');
  assert.throws(() => new PhysicalAnalysis(replacement), /indicator/);
  replacement.indicators.push('6z');
  inspect(replacement, a => assert.ok(a.agent_choices().some(c => c.kind === 'discard')));
});

test('manual entry preserves unknown hands and refuses duplicate riichi and stale actions', () => {
  const p = basic();
  const next = recordDiscard(p, 1, '9m');
  assert.equal(next.players[1].hand.length, 0); assert.equal(next.wall, p.wall - 1);
  assert.throws(() => recordChoice(p, { kind: 'discard', tile: '9m' }, []), /Analyse/);
  assert.throws(() => recordDiscard({ ...p, players: p.players.map(player => ({ ...player, riichi: 'riichi' })) }, 0, '1m', true), /already declared/);
  assert.deepEqual(parseTiles('123m 456p,77z'), ['1m','2m','3m','4p','5p','6p','7z','7z']);
  assert.throws(() => parseTiles('89z'));
  assert.throws(() => parseTiles('1m garbage 2p'));
  assert.deepEqual(readPhysical({ getItem: () => '{broken' }), emptyPosition());
});

test('a second legal kan stays recordable without inventing another trained-policy weight', () => {
  const p = basic(); p.players[0].hand = parseTiles('1111m2222p3459s11z'); p.drawn = '9s';
  const choices = inspect(p, a => a.agent_choices());
  const kans = choices.filter(c => c.kind === 'concealed-kan');
  assert.equal(kans.length, 2);
  assert.equal(kans[0].index, 76);
  assert.equal(kans[1].index ?? null, null);
  const next = recordChoice(p, kans[1], choices);
  assert.equal(next.players[0].melds[0].tile, '2p');
});

test('a manually entered kan leads to a replacement win and ends ippatsu', () => {
  const p = basic();
  p.players[0].hand = parseTiles('345p456789s1z');
  p.players[0].melds = [{ kind: 'extended-kan', tile: '2m', from: 3 }];
  p.players[1].hand = parseTiles('456m123p123789s1z');
  p.players[1].riichi = 'riichi'; p.players[1].ippatsu = true;
  p.players[1].discards = [{ tile: '6z', order: 0, drawn: false, riichi: true, claimed: false }];
  p.indicators = ['7z']; p.drawn = null; p.first_turns = false;
  p.seat = 1; p.turn = 0; p.phase = 'call'; p.pending_kind = 'extended-kan'; p.pending = '2m';
  inspect(p, a => assert.ok(a.agent_choices().some(choice => choice.kind === 'pass')));
  const next = recordDraw(p, 0, '1z'); next.indicators.push('6z');
  assert.equal(next.after_quad, true);
  assert.equal(next.players[1].ippatsu, false);
  inspect(next, a => assert.ok(a.agent_choices().some(choice => choice.kind === 'tsumo')));
  next.after_quad = false;
  inspect(next, a => assert.equal(a.agent_choices().some(choice => choice.kind === 'tsumo'), false,
    'the open hand needs the replacement-draw yaku to win'));
});

test('watch uses the retained choice, passes each trained opponent its own model, and finishes hands', async () => {
  const models = new Set();
  const w = new WatchSession(Game, 287, ['beginner', 'quick', 'strong', 'club'], {
    ai: async (_planes, mask, _signal, model) => { models.add(model); return pick(mask); }, evaluate: builtin,
  });
  try {
    await w.prepare();
    let decisions = 0, calls = 0;
    while (w.match.view.hands_played < 2 && !w.match.over && decisions++ < 500) {
      if (w.match.view.phase === 'over') { assert.equal(await w.step(), true); continue; }
      const choice = { ...w.analysis.choice }, before = w.match.commands.length;
      if (w.match.view.phase === 'call') calls++;
      assert.equal(await w.step(), true, w.failure);
      assert.deepEqual(w.match.commands[before], { type: 'choose', kind: choice.kind, tile: choice.tile ?? null });
    }
    assert.ok(decisions < 500 && decisions > 1); assert.ok(calls > 0);
    assert.deepEqual([...models].sort(), ['quick', 'strong']);
  } finally { w.dispose(); }
});

test('a disposed watcher cannot apply a late recommendation or run its autoplay timer', async () => {
  let resolve;
  const w = new WatchSession(Game, 1, ['club', 'club', 'club', 'club'], { evaluate: () => new Promise(done => { resolve = done; }) });
  const pending = w.prepare();
  await new Promise(done => setImmediate(done));
  const commands = w.match.commands.length;
  w.setAutoplay(true); w.dispose(); resolve({ choice: { kind: 'discard', tile: '1m' } });
  assert.equal(await pending, false);
  assert.equal(w.match.commands.length, commands);
  assert.equal(w.analysis, null);
  assert.equal(await w.step(), false);
});
