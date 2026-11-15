import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis, settle_physical } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { scanMatch } from '../src/lib/game-review.js';
import { handTables } from '../src/lib/review-insight-policy.js';
import { analysisInput, analysisOf, emptyPosition, parsePhysical, parseTiles, recordChoice, recordDiscard, recordDraw, withoutStaleDraw } from '../src/lib/physical-position.js';
import { emptyGuided, guidedEvent, parseGuided, GUIDED_FORMAT } from '../src/lib/guided-game.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

// Two facts of Mortal's replay that a typed-in table knows as it records the
// move, and that a game played here carries: the tile a seat drew on the
// turn it declared riichi (`drew_at_riichi` in a draft, the engine's
// `riichi_draw`), and the turn of a quad a seat declared itself (a set's
// `declared_turn`, the engine's `order`: the number of the discard the turn
// ends with). A draft keeps them under names the engine does not read, and
// only analysisInput gives them to the engine, which is asked with them only
// where it takes them (analysisOf).

/** The engine as the page asks it (GuidedPlay, PhysicalPlay). */
const validate = p => { analysisOf(PhysicalAnalysis, p).free(); };
const act = (game, event) => guidedEvent(game, event, validate, settle_physical);
const asked = (p, read) => {
  const engine = analysisOf(PhysicalAnalysis, p);
  try { return read(engine); } finally { engine.free(); }
};
const choicesOf = p => asked(p, engine => engine.agent_choices());
const planesOf = p => asked(p, engine => Array.from(engine.agent_observation_mortal()));
/** Mortal's planes of a table the engine wrote itself, read as it wrote it. */
const playedPlanes = p => { const engine = new PhysicalAnalysis(p); try { return Array.from(engine.agent_observation_mortal()); } finally { engine.free(); } };
const same = (a, b) => a.length === b.length && a.every((value, at) => value === b[at]);
const thrown = (tile, order, extra = {}) => ({ tile, order, drawn: false, riichi: false, claimed: false, ...extra });

/** East to act with 1m drawn, ready on 1m-4m once 9p goes. */
function readyAfterDraw(drawn = '1m') {
  const position = emptyPosition();
  position.players[0].hand = parseTiles('2345678m11p456s9p1m');
  Object.assign(position, { drawn, first_turns: false, wall: 50, indicators: ['7z'] });
  return position;
}

test('a riichi recorded from a known hand keeps the tile drawn that turn, and one from an unknown hand keeps none', () => {
  const declared = recordDiscard(readyAfterDraw(), 0, '9p', true);
  assert.equal(declared.players[0].drew_at_riichi, '1m');
  assert.equal(declared.players[0].riichi_draw, undefined, 'never under the engine\'s name');
  assert.equal(recordDiscard(readyAfterDraw(), 0, '9p').players[0].drew_at_riichi, undefined, 'a plain discard records none');
  const unknown = emptyPosition();
  Object.assign(unknown, { turn: 3, phase: 'call', pending: '5z', first_turns: false, wall: 50 });
  assert.equal(recordDiscard(unknown, 0, '9p', true).players[0].drew_at_riichi, undefined);
});

test('a riichi declared without a known draw keeps none of an earlier declaration the editor took back', () => {
  // The table's editor can take a riichi back and leave what its turn drew;
  // declared again with no drawn tile marked, the seat keeps no draw, and a
  // plain discard leaves a standing riichi's draw alone.
  const edited = readyAfterDraw(null);
  edited.players[0].drew_at_riichi = '7z';
  const again = recordDiscard(edited, 0, '9p', true);
  assert.equal(again.players[0].riichi, 'riichi');
  assert.equal('drew_at_riichi' in again.players[0], false);
  const standing = recordDiscard(readyAfterDraw(), 0, '9p', true);
  const later = recordDiscard(standing, 1, '2s');
  assert.equal(later.players[0].drew_at_riichi, '1m');
});

test('a riichi keeps ippatsu whether or not its draw is known, and the next discard ends it', () => {
  // An opponent's riichi, its hand unknown.
  const unknown = emptyPosition();
  Object.assign(unknown, { turn: 3, phase: 'call', pending: '5z', first_turns: false, wall: 50 });
  const theirs = recordDiscard(unknown, 0, '9p', true);
  assert.equal(theirs.players[0].ippatsu, true);
  // A known hand, with its drawn tile marked and without.
  for (const drawn of ['1m', null]) {
    const declared = recordDiscard(readyAfterDraw(drawn), 0, '9p', true);
    assert.equal(declared.players[0].ippatsu, true, `drawn ${drawn}`);
    assert.equal(declared.players[0].drew_at_riichi, drawn ?? undefined);
  }
  // The declaring seat's next discard ends it.
  const next = { ...structuredClone(theirs), turn: 0, phase: 'act', drawn: null };
  assert.equal(recordDiscard(next, 0, '3s').players[0].ippatsu, false);
});

test('a quad a seat declares records the turn its discard will end, at the table and in a guided game through the real engine', () => {
  const p = emptyPosition();
  p.players[1].discards.push(thrown('9m', 0, { drawn: true }));
  p.players[2].discards.push(thrown('1z', 4, { drawn: true }));
  p.players[0].hand = parseTiles('1111m234p567s555z2z');
  Object.assign(p, { drawn: '2z', first_turns: false, wall: 50 });
  const concealed = recordChoice(p, { kind: 'concealed-kan', tile: '1m' }, [{ kind: 'concealed-kan', tile: '1m' }]);
  assert.equal(concealed.players[0].melds[0].declared_turn, 5);
  assert.equal(concealed.players[0].melds[0].order, undefined, 'never under the engine\'s name');
  // The replacement, then the discard that ends the turn carries that number.
  assert.equal(recordDiscard(recordDraw(concealed, 0, '7z'), 0, '7z').players[0].discards.at(-1).order, 5);
  const added = structuredClone(p);
  added.players[0].hand = parseTiles('234p567s555z2z1m');
  added.players[0].melds.push({ kind: 'pon', tile: '1m', from: 1, claimed_tile: '1m' });
  added.players[1].discards[0] = thrown('1m', 0, { drawn: true, claimed: true });
  assert.equal(recordChoice(added, { kind: 'extended-kan', tile: '1m' }, [{ kind: 'extended-kan', tile: '1m' }]).players[0].melds[0].declared_turn, 5);
  // A turn past what a draft reads is not recorded, and the draft still reads.
  const late = structuredClone(p);
  late.players[2].discards[0].order = 399;
  const unrecorded = recordChoice(late, { kind: 'concealed-kan', tile: '1m' }, [{ kind: 'concealed-kan', tile: '1m' }]);
  assert.equal(unrecorded.players[0].melds[0].declared_turn, undefined);
  assert.ok(parsePhysical(JSON.stringify({ version: 1, position: unrecorded })));
  // An opponent's quad at a guided table, the player North, asked through the
  // page's own validator, then each later table asked as the page asks it.
  let game = emptyGuided('club');
  game.state.position.seat = 3;
  game.state.position.players[3].hand = parseTiles('123m456p789s1123z');
  for (const event of [{ type: 'setup' }, { type: 'hand' }, { type: 'indicator', tile: '7z' }]) game = act(game, event);
  game = act(game, { type: 'kan', kind: 'concealed-kan', tile: '2s' });
  assert.equal(game.state.position.players[0].melds[0].declared_turn, 0, 'the first discard of the hand is still to come');
  assert.equal(asked(game.state.position, engine => engine.agent_choices()).length > 0, true, 'the window to rob it');
  const pass = g => { const choices = choicesOf(g.state.position); return act(g, { type: 'choice', choice: choices.find(c => c.kind === 'pass'), choices }); };
  game = act(pass(game), { type: 'continue' });
  game = act(game, { type: 'indicator', tile: '6z' });
  game = act(game, { type: 'discard', tile: '9m' });
  assert.equal(analysisInput(game.state.position).players[0].melds[0].order, 0, 'told once its turn has ended');
  game = act(pass(game), { type: 'continue' });
  game = act(game, { type: 'discard', tile: '8m' });
  assert.equal(planesOf(game.state.position).length > 0, true);
});

test('the engine is told a riichi draw for the seat analysed alone, and a quad turn only while the engine would take it', () => {
  const declared = recordDiscard(readyAfterDraw(), 0, '9p', true);
  assert.equal(analysisInput(declared).players[0].riichi_draw, '1m', 'East is analysed');
  assert.equal(analysisInput(declared).players[0].drew_at_riichi, undefined);
  // South answers East's discard: East's draw is East's to know.
  const south = { ...structuredClone(declared), seat: 1 };
  south.players[1].hand = parseTiles('123p456p789p1199s');
  assert.equal(analysisInput(south).players[0].riichi_draw, undefined);
  validate(south);
  // A riichi taken back by an edit takes its draw with it.
  const taken = structuredClone(declared);
  taken.players[0].riichi = 'none';
  taken.players[0].discards[0].riichi = false;
  assert.equal(analysisInput(taken).players[0].riichi_draw, undefined);
  // A quad's turn goes with the discard it names, or with the set's kind.
  const quad = emptyPosition();
  quad.players[0].melds.push({ kind: 'concealed-kan', tile: '1m', from: 0, declared_turn: 3 });
  quad.players[0].discards.push(thrown('9p', 3, { drawn: true }));
  assert.equal(analysisInput(quad).players[0].melds[0].order, 3);
  const removed = structuredClone(quad);
  removed.players[0].discards = [];
  assert.equal(analysisInput(removed).players[0].melds[0].order, undefined, 'its discard removed in the editor');
  const renamed = structuredClone(quad);
  renamed.players[0].melds[0].kind = 'pon';
  assert.equal(analysisInput(renamed).players[0].melds[0].order, undefined, 'no longer a quad of its own');
  // The turn still being played: the replacement held.
  const playing = structuredClone(removed);
  Object.assign(playing, { after_quad: true, phase: 'act', turn: 0 });
  playing.players[1].discards.push(thrown('1z', 2, { drawn: true }));
  assert.equal(analysisInput(playing).players[0].melds[0].order, 3);
});

/** South, analysed, has a pon of East's 9m (order 0) and threw 1z after it
 * (order 1); a round later it declares a quad as a move, `kind` of `tile`
 * with `hand` held and `drawn` drawn, throws its replacement (order 5), the
 * table plays a round on and South draws again. */
function quadAfterPon(kind, tile, hand, drawn) {
  const p = emptyPosition();
  Object.assign(p, { seat: 1, turn: 1, phase: 'act', first_turns: false, wall: 55, indicators: ['7z'], drawn });
  p.players[0].discards = [thrown('9m', 0, { claimed: true }), thrown('4z', 4)];
  p.players[1].discards = [thrown('1z', 1)];
  p.players[2].discards = [thrown('2z', 2)];
  p.players[3].discards = [thrown('3z', 3)];
  p.players[1].melds = [{ kind: 'pon', tile: '9m', from: 3, claimed_tile: '9m' }];
  p.players[1].hand = parseTiles(hand);
  const choices = choicesOf(p);
  let q = recordChoice(p, choices.find(choice => choice.kind === kind && choice.tile === tile), choices);
  q.indicators.push('6z');
  q = recordDiscard(recordDraw(q, 1, '5z'), 1, '5z');
  for (const [seat, thrownTile] of [[2, '6z'], [3, '6z'], [0, '5z']]) q = recordDiscard(q, seat, thrownTile);
  return recordDraw(q, 1, '2s');
}
const addedKan = () => quadAfterPon('extended-kan', '9m', '123p456p789p1s9m', '9m');
/** The table less both facts, as a draft held it before they were kept. */
function bare(position) {
  const p = structuredClone(position);
  for (const player of p.players) {
    delete player.drew_at_riichi;
    for (const meld of player.melds) delete meld.declared_turn;
  }
  return p;
}

test("an added kan whose pon's discard was renumbered after it is guessed, not refused", () => {
  // South adds to its pon of East's 9m as a move, plays on a turn, and East's
  // claimed 9m is then given a later number in the editor.
  const q = addedKan();
  assert.equal(analysisInput(q).players[1].melds[0].order, 5);
  validate(q);
  const edited = structuredClone(q);
  edited.players[0].discards[0].order = 20;
  assert.equal(analysisInput(edited).players[1].melds[0].order, undefined);
  validate(edited);
  // Told the turn, the engine refuses the table.
  const told = analysisInput(edited);
  told.players[1].melds[0].order = 5;
  assert.throws(() => new PhysicalAnalysis(told), /records a turn before its pon was called/);
  // The engine pairs a triplet with a claimed discard of the set's own tile,
  // whatever tile the set was first recorded as taking: the kan retyped 8m in
  // the editor, East's claimed 9m removed and an 8m typed in, claimed and
  // numbered after the kan's turn, the turn is left out too.
  const retyped = structuredClone(q);
  retyped.players[1].melds[0].tile = '8m';
  retyped.players[0].discards.splice(0, 1);
  retyped.players[0].discards.push(thrown('8m', 9, { claimed: true }));
  assert.equal(retyped.players[1].melds[0].claimed_tile, '9m', 'the editor leaves what the set first took');
  assert.equal(analysisInput(retyped).players[1].melds[0].order, undefined);
  new PhysicalAnalysis(analysisInput(retyped)).free();
});

test("a quad's turn that Mortal's replay will not take costs a table none of its advice", () => {
  // The replay tells a quad on a turn that began with a draw, and a discard
  // that follows a call is none. Which discard follows a call is the
  // engine's own pairing of sets and discards to say, so the page cannot
  // tell: it asks the engine, and tells the table without the turn where
  // the replay refuses it, as the table was told before a draft kept one.
  for (const [name, table] of [['an added kan', addedKan()],
    ['a concealed kan', quadAfterPon('concealed-kan', '1s', '123p456p1s1s1s7p1s', '1s')]]) {
    const turn = table.players[1].melds.find(meld => meld.kind !== 'pon').declared_turn;
    assert.equal(turn, 5, name);
    assert.ok(same(planesOf(table), playedPlanes(analysisInput(table))), `${name}: the table as recorded is told with its turn`);
    // The editor renumbers the 1z South threw after its pon: the discard of
    // the quad's turn is now South's first after the call.
    const edited = structuredClone(table);
    edited.players[1].discards[0].order = 12;
    const told = analysisInput(edited);
    assert.equal(told.players[1].melds.find(meld => meld.kind !== 'pon').order, 5, `${name}: nothing the page sees is wrong with the turn`);
    const refused = new PhysicalAnalysis(told);
    try { assert.throws(() => refused.agent_observation_mortal(), /cannot be replayed as a hand/, name); } finally { refused.free(); }
    // As the page asks: the table without the turn, with advice from every adviser.
    assert.ok(same(planesOf(edited), playedPlanes(bare(edited))), `${name}: told as before the turn was kept`);
    assert.ok(choicesOf(edited).length > 0, name);
    validate(edited);
  }
  // A table the engine refuses with or without the facts is refused as before.
  const broken = addedKan();
  broken.players[1].hand.push('1m');
  let message = null;
  try { new PhysicalAnalysis(bare(broken)).free(); } catch (error) { message = String(error.message ?? error); }
  assert.ok(message, 'the table is refused without the facts');
  assert.throws(() => validate(broken), error => String(error.message ?? error) === message);
});

test('a draft keeps the facts under its own names: older drafts read as before, and the engine is told none of its own names from a draft', () => {
  const draft = position => JSON.stringify({ version: 1, position });
  const old = emptyPosition();
  old.players[0].melds.push({ kind: 'concealed-kan', tile: '1m', from: 0 });
  assert.deepEqual(parsePhysical(draft(old)), old);
  const recorded = structuredClone(old);
  recorded.players[0].melds[0].declared_turn = 7;
  recorded.players[0].drew_at_riichi = '5p';
  assert.deepEqual(parsePhysical(draft(recorded)), recorded);
  for (const spoil of [p => { p.players[0].melds[0].declared_turn = -1; }, p => { p.players[0].melds[0].declared_turn = 1.5; },
    p => { p.players[0].melds[0].declared_turn = '7'; }, p => { p.players[0].melds[0].declared_turn = 400; },
    p => { p.players[0].drew_at_riichi = '0m'; }, p => { p.players[0].drew_at_riichi = 5; }]) {
    const bad = structuredClone(recorded);
    spoil(bad);
    assert.equal(parsePhysical(draft(bad)), null, JSON.stringify(bad.players[0]));
  }
  // The engine's own names are no draft's. No page writes them, and one
  // that held them reads as it is, like any key a draft does not know; the
  // engine is told the facts from the draft's own names alone.
  const named = structuredClone(old);
  named.players[0].discards.push(thrown('9p', 7, { drawn: true, riichi: true }));
  named.players[0].riichi = 'riichi';
  named.players[0].melds[0].order = 7;
  named.players[0].riichi_draw = '5p';
  assert.deepEqual(parsePhysical(draft(named)), named);
  const told = analysisInput(named);
  assert.equal(told.players[0].melds[0].order, undefined);
  assert.equal(told.players[0].riichi_draw, undefined);
});

test('a riichi the editor moves or takes back keeps no draw of the turn it left', () => {
  // East declared on 9p having drawn 1m, and a turn later threw 4z.
  const declared = recordDiscard(readyAfterDraw(), 0, '9p', true);
  declared.players[0].hand.push('4z');
  const later = recordDiscard({ ...structuredClone(declared), turn: 0, phase: 'act', drawn: '4z' }, 0, '4z');
  assert.equal(later.players[0].drew_at_riichi, '1m');
  const edit = change => { const next = structuredClone(later); change(next.players[0]); return withoutStaleDraw(later, next); };
  // The mark moved to the 4z: the 1m was not drawn on that turn.
  assert.equal(edit(east => { east.discards[0].riichi = false; east.discards[1].riichi = true; }).players[0].drew_at_riichi, undefined);
  // The declaration taken back, and the discard that made it removed.
  assert.equal(edit(east => { east.riichi = 'none'; }).players[0].drew_at_riichi, undefined);
  assert.equal(edit(east => { east.discards.splice(0, 1); }).players[0].drew_at_riichi, undefined);
  // What leaves the declaration where it was leaves the draw: another
  // discard, the hand, an ordinary riichi made a double one.
  assert.equal(edit(east => { east.discards[1].tile = '3z'; }).players[0].drew_at_riichi, '1m');
  assert.equal(edit(east => { east.hand.pop(); }).players[0].drew_at_riichi, '1m');
  assert.equal(edit(east => { east.riichi = 'double'; }).players[0].drew_at_riichi, '1m');
  // A riichi recorded as a move brings its own draw, which is kept.
  const before = readyAfterDraw();
  assert.equal(withoutStaleDraw(before, recordDiscard(before, 0, '9p', true)).players[0].drew_at_riichi, '1m');
});

/** East, the player, draws 1m and declares riichi on 9p from the adviser's list. */
function afterRiichi() {
  let game = emptyGuided('club');
  game.state.position.players[0].hand = parseTiles('2345678m11p456s9p');
  for (const event of [{ type: 'setup' }, { type: 'hand' }, { type: 'indicator', tile: '7z' }, { type: 'draw', tile: '1m' }]) game = act(game, event);
  const choices = choicesOf(game.state.position);
  game = act(game, { type: 'choice', choice: choices.find(choice => choice.kind === 'riichi' && choice.tile === '9p'), choices });
  assert.equal(game.state.position.players[0].drew_at_riichi, '1m');
  assert.equal(game.state.position.players[0].ippatsu, true);
  return act(game, { type: 'continue' });
}

test('a guided hand in which the player declared riichi settles whoever ends it, and an older page reads its draft', () => {
  const quiet = [false, false, false, false];
  const settled = (game, input) => {
    const done = act(game, { type: 'settle', input });
    assert.equal(done.state.stage, 'over');
    assert.ok(done.state.settlement, 'settled');
    return done;
  };
  // South draws and wins by tsumo.
  settled(act(afterRiichi(), { type: 'finish', result: 'South: tsumo', kind: 'tsumo', winner: 1 }),
    { winners: [1], hands: [[], parseTiles('234p567p789s2233z'), [], []], winning_tile: '3z', ura: [], tenpai: quiet });
  // South discards 5z: West wins by ron, and West and North both.
  const thrownBySouth = () => {
    const game = act(afterRiichi(), { type: 'discard', tile: '5z' });
    const choices = choicesOf(game.state.position);
    return act(game, { type: 'choice', choice: choices.find(choice => choice.kind === 'pass'), choices });
  };
  settled(act(thrownBySouth(), { type: 'finish', result: 'West: ron', kind: 'ron', winner: 2 }),
    { winners: [2], hands: [[], [], parseTiles('234p567p678s99s55z'), []], winning_tile: '5z', ura: [], tenpai: quiet, confirmed_no_furiten: true });
  settled(act(thrownBySouth(), { type: 'finish', result: 'West: ron', kind: 'ron', winner: 2 }),
    { winners: [2, 3], hands: [[], [], parseTiles('234p567p678s99s55z'), parseTiles('123m456m789m111s5z')], winning_tile: '5z', ura: [], tenpai: quiet,
      confirmed_no_furiten: true });
  // The wall runs out on South's discard, the player ready.
  const last = thrownBySouth();
  last.state.position.wall = 0;
  settled(act(last, { type: 'finish', result: 'Exhaustive draw', kind: 'draw' }),
    { winners: [], hands: [[], [], [], []], winning_tile: null, ura: [], tenpai: [true, false, false, false] });
  // The draft as saved, read and asked as a page before these facts asks:
  // the position handed to the engine as it is, with no analysisInput.
  const saved = parseGuided(GUIDED_FORMAT.encode(thrownBySouth()), settle_physical);
  assert.ok(saved);
  const position = saved.state.position;
  assert.equal(position.players[0].drew_at_riichi, '1m');
  new PhysicalAnalysis(position).free();
  const ending = { kind: 'ron', winners: [2], position: structuredClone(position), nextSeat: saved.state.nextSeat, needsDraw: false };
  assert.ok(settle_physical(ending, { winners: [2], hands: [[], [], parseTiles('234p567p678s99s55z'), []], winning_tile: '5z', ura: [], tenpai: quiet,
    confirmed_no_furiten: true }));
});

/** A game the player plays with Club's moves, declaring every riichi it can. */
function clubGame(seed) {
  const match = new MatchSession(Game, seed, 'club');
  try {
    match.advance(false);
    for (let step = 0; step < 5000 && !match.over; step++) {
      if (match.engine.hand_is_over()) { match.apply({ type: 'next' }); match.advance(false); continue; }
      const choices = match.choices, pick = match.engine.agent_pick('club');
      const choice = choices.find(c => c.kind === 'riichi') ?? choices.find(c => c.kind === pick.kind && (c.tile ?? null) === (pick.tile ?? null)) ?? choices[0];
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
    }
    return match.snapshot();
  } finally { match.dispose(); }
}

/** A game the player plays throwing the tile drawn, calling every set and quad offered. */
function callingGame(seed) {
  const match = new MatchSession(Game, seed, 'beginner');
  try {
    match.advance(false);
    for (let step = 0; step < 5000 && !match.over; step++) {
      if (match.engine.hand_is_over()) { match.apply({ type: 'next' }); match.advance(false); continue; }
      const choices = match.choices, drawn = match.view.seats[0].drawn;
      const choice = choices.find(c => ['ron', 'tsumo', 'riichi'].includes(c.kind))
        ?? choices.find(c => ['concealed-kan', 'extended-kan', 'kan', 'pon', 'chii'].includes(c.kind))
        ?? choices.find(c => c.kind === 'discard' && c.tile === drawn) ?? choices.find(c => c.kind === 'discard') ?? choices[0];
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
    }
    return match.snapshot();
  } finally { match.dispose(); }
}

/** Plays each hand of `game` in which the player made a move of `kinds` on
 * at the page's own table from that move: the move recorded as the page
 * records it (recordChoice), then each later discard (recordDiscard, marked
 * from the draw as the player marks it), the player's draws and replacement
 * (recordDraw) and the player's moves, until a call or a new indicator the
 * walk does not follow. At each decision it reaches, the page's table asked
 * through analysisInput is compared with the played game's own, and the same
 * table without the two facts. */
function walkAtTheTable(game, kinds) {
  let compared = 0, differed = 0;
  for (const hand of scanMatch(Game, game)) {
    const at = hand.frames.findIndex(frame => kinds.includes(frame.played.kind));
    if (at < 0) continue;
    const tables = handTables(Game, game, hand);
    let page = recordChoice(tables[at], hand.frames[at].played, [hand.frames[at].played]);
    for (let step = at + 1; step < tables.length; step++) {
      const target = tables[step];
      const seen = new Set(page.players.flatMap(player => player.discards.map(entry => entry.order)));
      const fresh = target.players.flatMap((player, seat) => player.discards.filter(entry => !seen.has(entry.order)).map(entry => ({ ...entry, seat })))
        .sort((a, b) => a.order - b.order);
      const sameTable = JSON.stringify(target.indicators) === JSON.stringify(page.indicators)
        && target.players.every((player, seat) => player.melds.length === page.players[seat].melds.length);
      if (!sameTable && page.phase === 'call' && target.phase === 'act' && target.after_quad && !fresh.length) {
        page = recordDraw({ ...page, indicators: target.indicators }, page.seat, target.drawn);
      } else if (sameTable && !fresh.some(entry => entry.claimed) && (target.phase !== 'act' || target.drawn)) {
        for (const entry of fresh) {
          page = recordDiscard(page, entry.seat, entry.tile, entry.riichi);
          page.players[entry.seat].discards.at(-1).drawn = entry.drawn;
        }
        if (target.phase === 'act') page = recordDraw(page, target.seat, target.drawn);
      } else break;
      assert.ok(same(planesOf(page), playedPlanes(target)), `hand ${hand.index}, decision ${step}: the played game's planes`);
      const bare = structuredClone(page);
      bare.players.forEach(player => { delete player.drew_at_riichi; player.melds.forEach(meld => { delete meld.declared_turn; }); });
      compared++;
      differed += same(planesOf(bare), playedPlanes(target)) ? 0 : 1;
      page = recordChoice(page, hand.frames[step].played, [hand.frames[step].played]);
    }
  }
  return { compared, differed };
}

test("after a riichi recorded at the table, the page's own later tables give Mortal the played game's planes", () => {
  const { compared, differed } = walkAtTheTable(clubGame(1), ['riichi']);
  assert.ok(compared > 5 && differed > 0, `${differed} of ${compared} later decisions read otherwise without the riichi draw`);
});

test("after a quad recorded at the table, the page's own later tables give Mortal the played game's planes", () => {
  const { compared, differed } = walkAtTheTable(callingGame(21), ['concealed-kan', 'extended-kan']);
  assert.ok(compared > 5 && differed > 0, `${differed} of ${compared} later decisions read otherwise without the quad's turn`);
});
