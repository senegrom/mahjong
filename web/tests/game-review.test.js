import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { reviewWithStrong } from '../src/lib/review-policy.js';
import { Asking, BATCH, FILTERS, FILTER_LABELS, FILTER_TITLES, FLAG_PLAYED_BELOW, FLAG_PREFERRED_FROM, MatchReview, SLOW_MS, agreement,
  decisionInBrief, decisionKey, differed, filterCounts, filterSaid, handEntry, handName, judgement, landing, listedIn, moveToken, moveWeight,
  neighbour, playlist, rankDecisions, scanMatch, seatNotes, stepSaid, tableSaid } from '../src/lib/game-review.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });


/** A whole game, played to its standings by a player who takes every win,
 * declares every riichi and otherwise throws the tile just drawn. With
 * `calls` the player also declares every quad and calls every set offered. */
function finishedGame(seed, opponents = 'beginner', { calls = false } = {}) {
  const match = new MatchSession(Game, seed, opponents);
  try {
    match.advance(false);
    for (let step = 0; step < 5000 && !match.over; step++) {
      if (match.engine.hand_is_over()) { match.apply({ type: 'next' }); match.advance(false); continue; }
      const choices = match.choices, drawn = match.view.seats[0].drawn;
      const choice = choices.find(c => ['ron', 'tsumo', 'riichi'].includes(c.kind))
        ?? (calls ? choices.find(c => ['concealed-kan', 'extended-kan', 'kan', 'pon', 'chii'].includes(c.kind)) : undefined)
        ?? choices.find(c => c.kind === 'pass')
        ?? choices.find(c => c.kind === 'discard' && c.tile === drawn) ?? choices.find(c => c.kind === 'discard') ?? choices[0];
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
    }
    assert.ok(match.over, 'the game reaches its standings');
    return match.snapshot();
  } finally { match.dispose(); }
}
const game = finishedGame(14);

/** A stand-in network that is sure of one legal move: the last one its mask
 * allows, at 0.95, with the rest shared among the others. Its answer depends
 * on the position alone, so a batch and a single request agree. */
function decisive(mask) {
  const legal = Array.from(mask).flatMap((allowed, index) => (allowed ? [index] : []));
  const action = legal.at(-1);
  const weights = Array.from(mask, (allowed, index) => (!allowed ? 0 : index === action
    ? (legal.length === 1 ? 1 : 0.95) : 0.05 / (legal.length - 1)));
  return { action, weights };
}
const network = (log = []) => async (positions, signal) => {
  assert.ok(positions.length >= 1 && positions.length <= BATCH, 'a batch is at most BATCH positions');
  log.push(positions.length);
  await new Promise(resolve => setImmediate(resolve));
  if (signal?.aborted) throw new DOMException('Review changed', 'AbortError');
  return positions.map(position => decisive(position.mask));
};
const settled = review => new Promise(resolve => {
  const check = () => (review.busy ? setImmediate(check) : resolve());
  setImmediate(check);
});
const turn = () => new Promise(resolve => setImmediate(resolve));
/** Waits a turn of the event loop at a time until `ready()` holds. */
async function until(ready, what) {
  for (let count = 0; count < 1000 && !ready(); count++) await turn();
  assert.ok(ready(), what);
}

/** A review whose gate is the notes' spells of asking, `asking`, as the page
 * has it, noting in `waits` the hand it was on each time it found the gate
 * shut, and in `sizes` every batch it sent. `onBatch(sent)` is told as each
 * batch is sent, with how many were sent before it, so that the notes can
 * begin to ask while that batch is under way. */
function gatedReview(snapshot, asking, { sizes = [], waits = [], onBatch = () => {}, analyzeBatch = network(sizes), ...options } = {}) {
  const review = new MatchReview(Game, snapshot, { ...options,
    analyzeBatch: (positions, signal) => { onBatch(sizes.length); return analyzeBatch(positions, signal); },
    gate: () => {
      const shut = asking.gate();
      if (shut) waits.push(review.current);
      return shut;
    } });
  return review;
}

/** The batches an ungated review of `snapshot` sends, in order. */
async function ungatedSizes(snapshot) {
  const sizes = [], review = new MatchReview(Game, snapshot, { analyzeBatch: network(sizes) });
  try {
    review.review();
    await settled(review);
    return sizes;
  } finally { review.dispose(); }
}

/** The replayed engine at the end of one hand, for comparisons. */
function atHandEnd(snapshot, end) {
  const match = new MatchSession(Game, snapshot.seed, snapshot.opponents);
  match.advance(false);
  for (const command of snapshot.commands.slice(0, end)) { match.apply(command); match.advance(false); }
  return match;
}

/** The match as it was saved after its first `count` commands: at a hand's
 * result where `count` is that hand's end. */
function savedAfter(snapshot, count) {
  const match = atHandEnd(snapshot, count);
  try { return match.snapshot(); } finally { match.dispose(); }
}

/** A review of `snapshot` whose network notes, in `order`, each hand it is
 * asked about as the review goes from one hand to the next, and in `sizes`
 * every batch. */
function notingReview(snapshot, order, { sizes = [], onComplete } = {}) {
  const answer = network(sizes);
  const review = new MatchReview(Game, snapshot, { onComplete, analyzeBatch: (positions, signal) => {
    if (order.at(-1) !== review.current) order.push(review.current);
    return answer(positions, signal);
  } });
  return review;
}

test('a finished match replays hand by hand, with the table at every decision and what was chosen there', () => {
  const hands = scanMatch(Game, game);
  const nexts = game.commands.flatMap((command, at) => (command.type === 'next' ? [at] : []));
  assert.deepEqual(hands.map(hand => hand.end), nexts, 'each hand ends at the command that dealt the next');
  assert.equal(hands[0].name, 'East 1');
  let start = 0;
  for (const hand of hands) {
    const chosen = game.commands.slice(start, hand.end).flatMap((command, offset) => (command.type === 'choose' ? [start + offset] : []));
    assert.deepEqual(hand.frames.map(frame => frame.command), chosen);
    for (const frame of hand.frames) {
      const command = game.commands[frame.command];
      assert.deepEqual(frame.played, { kind: command.kind, tile: command.tile ?? null });
      assert.ok(['act', 'call'].includes(frame.view.phase));
      assert.ok(frame.choices.some(choice => choice.kind === command.kind && (choice.tile ?? null) === (command.tile ?? null)),
        'the move played was one of the choices shown');
    }
    const ended = atHandEnd(game, hand.end);
    try {
      const view = ended.view;
      assert.equal(view.phase, 'over');
      assert.equal(hand.outcome, view.outcome.line);
      assert.equal(hand.change, view.outcome.changes[0]);
      assert.equal(handName(view), hand.name);
    } finally { ended.dispose(); }
    start = hand.end + 1;
  }
  assert.ok(hands.some(hand => hand.frames.some(frame => frame.lines.length)), 'the history between decisions is kept');
  const moved = { ...game, state: game.state.replace(/"score":(\d+)/, (_, score) => `"score":${Number(score) + 100}`) };
  assert.throws(() => scanMatch(Game, moved), /^Error: The replay did not end where the match did$/);
  assert.throws(() => scanMatch(Game, { ...game, commands: game.commands.slice(0, -1) }), /^Error: The replay did not end where the match did$/);
});

test("a match waiting at a hand's result replays to that hand as the finished match has it, and a hand in play is refused", () => {
  // The tile-throwing game, and one whose player calls every set and quad.
  for (const finished of [game, finishedGame(16, 'beginner', { calls: true })]) {
    const whole = scanMatch(Game, finished);
    for (const hand of [whole[0], whole[2], whole.at(-1)]) {
      const saved = savedAfter(finished, hand.end), hands = scanMatch(Game, saved);
      assert.deepEqual(hands, whole.slice(0, hand.index + 1), `seed ${finished.seed}, at the result of ${hand.name}`);
      // The hand closes where the command that deals the next will go, the
      // command it closes at once the match has gone on.
      assert.equal(hands.at(-1).end, saved.commands.length);
    }
  }
  // While a hand is played, and once one is dealt, its tiles are what the
  // table hides: one move into the first hand, just after the first hand's
  // result, and before any move at all.
  const first = game.commands.findIndex(command => command.type === 'choose');
  for (const count of [first + 1, scanMatch(Game, game)[0].end + 1, 0]) {
    assert.throws(() => scanMatch(Game, savedAfter(game, count)), /^Error: A hand is still being played$/, `after ${count} commands`);
  }
  // A replay that went another way says so, though it stopped in a hand too.
  const playing = savedAfter(game, first + 1);
  const astray = { ...playing, state: playing.state.replace(/"score":(\d+)/, (_, score) => `"score":${Number(score) + 100}`) };
  assert.notEqual(astray.state, playing.state);
  assert.throws(() => scanMatch(Game, astray), /^Error: The replay did not end where the match did$/);
});

test('every hand of a game has a name of its own, a repeated hand told apart by its honba', () => {
  assert.equal(handName({ round: 'east', kyoku: 1, counters: 0 }), 'East 1');
  assert.equal(handName({ round: 'south', kyoku: 3, counters: 2 }), 'South 3 · 2 honba');
  // A game in which East 2 comes back, with one honba and then two.
  const names = scanMatch(Game, finishedGame(5)).map(hand => hand.name);
  const bare = names.map(name => name.replace(/ · \d+ honba$/, ''));
  assert.ok(new Set(bare).size < bare.length, `a round and number come back: ${names.join(', ')}`);
  assert.equal(new Set(names).size, names.length, names.join(', '));
});

test('a decision is flagged only where the player had under 5% and the network at least 60%, riichi in two stages', () => {
  const row = values => ({ agreed: false, forced: false, played_kind: 'discard', advised_kind: 'discard',
    played_weight: 0.01, preferred_weight: 0.8, played_conditional_weight: null, preferred_conditional_weight: null, ...values });
  assert.equal(FLAG_PLAYED_BELOW, 0.05); assert.equal(FLAG_PREFERRED_FROM, 0.6);
  assert.deepEqual(judgement(row({})), { stage: 'move', played: 0.01, preferred: 0.8, gap: 0.8 - 0.01, flagged: true });
  assert.equal(judgement(row({ played_weight: 0.05 })).flagged, false, 'five percent is not under five percent');
  assert.equal(judgement(row({ preferred_weight: 0.59 })).flagged, false, 'an undecided network flags nothing');
  assert.equal(judgement(row({ agreed: true })), null);
  assert.equal(judgement(row({ forced: true })), null, 'a forced move was no choice');
  assert.equal(judgement(row({ played_weight: null })), null, 'an unscored move is not compared');
  // Both declared: the declaration agreed, so the tile is what is judged.
  const tile = judgement(row({ played_kind: 'riichi', advised_kind: 'riichi', played_weight: 0.9, preferred_weight: 0.9,
    played_conditional_weight: 0.02, preferred_conditional_weight: 0.7 }));
  assert.deepEqual({ stage: tile.stage, flagged: tile.flagged }, { stage: 'tile', flagged: true });
  // One declared and the other did not: the declaration itself is judged.
  const declared = judgement(row({ played_kind: 'riichi', played_weight: 0.03, preferred_weight: 0.9, played_conditional_weight: 0.5 }));
  assert.deepEqual({ stage: declared.stage, flagged: declared.flagged, played: declared.played }, { stage: 'move', flagged: true, played: 0.03 });
  const withheld = judgement(row({ advised_kind: 'riichi', played_weight: 0.2, preferred_weight: 0.75 }));
  assert.equal(withheld.flagged, false);
});

test("the filters' buttons say in a word what each keeps, and the Flagged and Differed buttons' titles what keeps them", () => {
  assert.deepEqual(FILTERS, ['flagged', 'differed', 'every']);
  assert.deepEqual(FILTER_LABELS, { flagged: 'Flagged', differed: 'Differed', every: 'All' });
  assert.deepEqual(FILTER_TITLES, { flagged: "Your move under 5%, Trained AI's choice 60% or more", differed: 'Trained AI chose another move' });
});

test('flagged decisions are counted by the gap, ties in the order played, and agreement leaves forced moves out', () => {
  const decision = (name, gap, flagged = true) => ({ name, judgement: { gap, flagged } });
  const ranked = rankDecisions([decision('a', 0.7), decision('b', 0.9), decision('c', 0.99, false), decision('d', 0.7), { name: 'e', judgement: null }]);
  assert.deepEqual(ranked.map(entry => entry.name), ['b', 'a', 'd']);
  assert.deepEqual(agreement([{ agreed: true }, { agreed: false }, { agreed: true, forced: true }, { agreed: true }]),
    { matched: 2, decisions: 3, share: 2 / 3 });
  assert.deepEqual(agreement([]), { matched: 0, decisions: 0, share: null });
});

test('a filter counts the flagged decisions, those where Trained AI chose differently, and every one', () => {
  const decision = (name, values = {}) => ({ name, agreed: false, forced: false, judgement: null, ...values });
  const decisions = [
    decision('agreed', { agreed: true }),
    decision('near', { judgement: { gap: 0.5, flagged: false } }),
    decision('flag', { judgement: { gap: 0.9, flagged: true } }),
    decision('forced', { forced: true, agreed: true }),
    decision('unscored'),
    decision('wide', { judgement: { gap: 0.95, flagged: false } }),
    decision('forced and other', { forced: true }),
    decision('flag2', { judgement: { gap: 0.7, flagged: true } }),
    decision('unscored2'),
  ];
  // A forced move never differs, and a move that could not be compared does.
  assert.deepEqual(filterCounts(decisions), { flagged: 2, differed: 6, every: 9 });
  assert.deepEqual(filterCounts([]), { flagged: 0, differed: 0, every: 0 });
  assert.equal(differed(decisions[6]), false);
  assert.equal(differed(decisions[4]), true);
});

test('the review says what its filter keeps as it is chosen, in no order but the order played', () => {
  const counts = { flagged: 7, differed: 64, every: 182 };
  assert.equal(filterSaid('flagged', counts), '7 flagged decisions.');
  assert.equal(filterSaid('differed', counts), '64 decisions where Trained AI differed, 7 of them flagged.');
  assert.equal(filterSaid('every', counts), 'All 182 decisions.');
  assert.equal(filterSaid('flagged', { flagged: 1, differed: 3, every: 9 }), '1 flagged decision.');
  assert.equal(filterSaid('differed', { flagged: 0, differed: 1, every: 9 }), '1 decision where Trained AI differed.');
  assert.equal(filterSaid('every', { flagged: 0, differed: 0, every: 1 }), 'The one decision.');
  // Where the filter keeps nothing.
  assert.equal(filterSaid('flagged', { flagged: 0, differed: 3, every: 4 }), 'None flagged.');
  assert.equal(filterSaid('differed', { flagged: 0, differed: 0, every: 4 }), 'Trained AI agreed at every decision.');
  assert.equal(filterSaid('every', { flagged: 0, differed: 0, every: 0 }), 'No decisions.');
});

test("the table at a step is said in a sentence, each seat says what it was doing, and a hand's line gives both moves", () => {
  const seat = (wind, values = {}) => ({ seat: wind, turn: false, riichi: false, hand: [], drawn: null, ...values });
  const turn = { phase: 'act', wall: 45, pending_discard: null, pending_from: null, dora_types: [],
    seats: [seat('east', { turn: true, drawn: '3m' }), seat('south', { riichi: true }), seat('west'), seat('north', { riichi: true })] };
  assert.equal(tableSaid(turn), 'Your turn, 45 tiles left. South and North declared riichi.');
  assert.deepEqual(seatNotes(turn), ['To play', '', '', '']);
  const answer = { ...turn, phase: 'call', wall: 1, pending_discard: '5s', pending_from: 'west',
    seats: [seat('east', { riichi: true }), seat('south'), seat('west', { turn: true }), seat('north')] };
  assert.equal(tableSaid(answer), 'West discarded the 5 bamboo, 1 tile left. You declared riichi.');
  assert.deepEqual(seatNotes(answer), ['To answer', '', 'Discarded', '']);
  // A line of the hand's decisions: both moves as tiles and words, and its
  // name in the engine's words. The choices name the moves as the engine does.
  const choices = [{ kind: 'discard', tile: '7m', label: 'discards the 7 characters' }, { kind: 'discard', tile: '8s', label: 'discards the 8 bamboo' },
    { kind: 'riichi', tile: '3s', label: 'declares riichi on the 3 bamboo' }, { kind: 'tsumo', tile: null, label: 'wins by self-draw' },
    { kind: 'pass', tile: null, label: 'passes' }, { kind: 'chii', tile: '4s', label: 'calls a sequence from the 4 bamboo' }];
  const frame = (played, view = turn) => ({ played, view, choices });
  const decision = (advised, values = {}) => ({ agreed: false, forced: false, judgement: null, advised_kind: advised.kind,
    advised_tile: advised.tile, advised: advised.label, ...values });
  assert.deepEqual(decisionInBrief(frame({ kind: 'discard', tile: '7m' }), decision(choices[1], { judgement: { flagged: true } })),
    { you: { word: '', tiles: ['7m'] }, ai: { word: '', tiles: ['8s'] }, offered: null, verdict: 'Flagged', flagged: true, agreed: false,
      forced: false, reviewed: true, hidden: false, said: 'You: discards the 7 characters. Trained AI: discards the 8 bamboo. Flagged.' });
  const riichi = decisionInBrief(frame({ kind: 'riichi', tile: '3s' }), decision(choices[3]));
  assert.deepEqual([riichi.you, riichi.ai, riichi.verdict], [{ word: 'Riichi', tiles: ['3s'] }, { word: 'Tsumo', tiles: [] }, 'Trained AI differed']);
  assert.equal(riichi.said, 'You: declares riichi on the 3 bamboo. Trained AI: wins by self-draw.');
  // Agreed, there is no other move to give, and an only move was no choice.
  const agreed = decisionInBrief(frame({ kind: 'discard', tile: '7m' }), decision(choices[0], { agreed: true }));
  assert.deepEqual([agreed.ai, agreed.verdict, agreed.said], [null, 'Trained AI agreed', 'You: discards the 7 characters. Trained AI agreed.']);
  const forced = decisionInBrief(frame({ kind: 'discard', tile: '7m' }), decision(choices[0], { agreed: true, forced: true }));
  assert.deepEqual([forced.ai, forced.verdict, forced.forced, forced.said], [null, 'Only move', true, 'You: discards the 7 characters. Only move.']);
  // An answer to another's discard is on that tile.
  const pass = decisionInBrief(frame({ kind: 'pass', tile: null }, answer), decision(choices[5]));
  assert.deepEqual([pass.offered, pass.you, pass.ai], ['5s', { word: 'Pass', tiles: [] }, { word: 'Chii', tiles: ['4s', '5s', '6s'] }]);
  assert.equal(pass.said, 'On the 5 bamboo. You: passes. Trained AI: calls a sequence from the 4 bamboo.');
  // Before the hand is reviewed only the player's move is known.
  const waiting = decisionInBrief(frame({ kind: 'discard', tile: '7m' }), null);
  assert.deepEqual([waiting.reviewed, waiting.ai, waiting.verdict, waiting.said], [false, null, '', 'You: discards the 7 characters. Not reviewed yet.']);
});

test("with Trained AI's move hidden, a hand's line gives the player's move and the flag, and keeps back Trained AI's move and that it agreed, heard too", () => {
  const seat = wind => ({ seat: wind, turn: false, riichi: false, hand: [], drawn: null });
  const turn = { phase: 'act', wall: 45, pending_discard: null, pending_from: null, seats: ['east', 'south', 'west', 'north'].map(seat) };
  const answer = { ...turn, phase: 'call', pending_discard: '5s', pending_from: 'west' };
  const choices = [{ kind: 'discard', tile: '7m', label: 'discards the 7 characters' }, { kind: 'discard', tile: '8s', label: 'discards the 8 bamboo' },
    { kind: 'pass', tile: null, label: 'passes' }, { kind: 'pon', tile: null, label: 'calls a triplet' }];
  const frame = (played, view = turn) => ({ played, view, choices });
  const decision = (advised, values = {}) => ({ agreed: false, forced: false, judgement: null, advised_kind: advised.kind,
    advised_tile: advised.tile, advised: advised.label, ...values });
  const hidden = { hidden: true }, thrown = { kind: 'discard', tile: '7m' };
  // Flagged: the flag stays and Trained AI's move goes, from the line and
  // from its name alike.
  assert.deepEqual(decisionInBrief(frame(thrown), decision(choices[1], { judgement: { flagged: true } }), hidden),
    { you: { word: '', tiles: ['7m'] }, ai: null, offered: null, verdict: 'Flagged', flagged: true, agreed: false, forced: false,
      reviewed: true, hidden: true, said: "You: discards the 7 characters. Trained AI's move hidden. Flagged." });
  // Agreed: no tick and no word of it, which would give the answer away.
  const agreed = decisionInBrief(frame(thrown), decision(choices[0], { agreed: true }), hidden);
  assert.deepEqual([agreed.ai, agreed.agreed, agreed.verdict, agreed.hidden, agreed.said],
    [null, false, '', true, "You: discards the 7 characters. Trained AI's move hidden."]);
  // An answer to another's discard still names the tile on offer, which the
  // table shows.
  const pass = decisionInBrief(frame({ kind: 'pass', tile: null }, answer), decision(choices[3]), hidden);
  assert.deepEqual([pass.offered, pass.you, pass.ai], ['5s', { word: 'Pass', tiles: [] }, null]);
  assert.equal(pass.said, "On the 5 bamboo. You: passes. Trained AI's move hidden.");
  // An only move and a hand not yet reviewed have nothing to hide.
  const forced = decision(choices[0], { agreed: true, forced: true });
  assert.deepEqual(decisionInBrief(frame(thrown), forced, hidden), decisionInBrief(frame(thrown), forced));
  assert.deepEqual(decisionInBrief(frame(thrown), null, hidden), decisionInBrief(frame(thrown), null));
  assert.equal(decisionInBrief(frame(thrown), forced, hidden).hidden, false);
});

test('a filter holds the decisions it names, and its list runs through the hands in the order played, a hand not yet reviewed held only by the list of every decision', () => {
  assert.equal(decisionKey(3, 12), '3:12');
  const decision = (values = {}) => ({ agreed: false, forced: false, judgement: null, ...values });
  const flagged = decision({ judgement: { gap: 0.9, flagged: true } }), near = decision({ judgement: { gap: 0.4, flagged: false } });
  const agreed = decision({ agreed: true }), forced = decision({ agreed: true, forced: true });
  for (const [filter, held] of [['flagged', [true, false, false, false, false]], ['differed', [true, true, false, false, false]],
    ['every', [true, true, true, true, true]]]) {
    assert.deepEqual([flagged, near, agreed, forced, null].map(entry => listedIn(entry, filter)), held, filter);
  }
  const hand = (index, decisions, frames = decisions?.length ?? 2) => ({ index, frames: Array(frames).fill({}), decisions });
  // A hand reviewed, one not yet reviewed, one won before the player's first
  // turn, and another reviewed.
  const hands = [hand(0, [near, flagged, agreed]), hand(1, null, 2), hand(2, [], 0), hand(3, [flagged, forced])];
  assert.deepEqual(playlist(hands, 'flagged'), [{ hand: 0, step: 1 }, { hand: 3, step: 0 }]);
  assert.deepEqual(playlist(hands, 'differed'), [{ hand: 0, step: 0 }, { hand: 0, step: 1 }, { hand: 3, step: 0 }]);
  assert.deepEqual(playlist(hands, 'every'), [{ hand: 0, step: 0 }, { hand: 0, step: 1 }, { hand: 0, step: 2 },
    { hand: 1, step: 0 }, { hand: 1, step: 1 }, { hand: 3, step: 0 }, { hand: 3, step: 1 }]);
});

test('Previous and Next go to the place the list holds before or after the one shown, across hands, from a place it does not hold too, and stop at the ends', () => {
  const list = [{ hand: 0, step: 1 }, { hand: 2, step: 0 }, { hand: 2, step: 4 }];
  assert.deepEqual(neighbour(list, { hand: 0, step: 1 }, 1), { hand: 2, step: 0 }, 'on into a later hand');
  assert.deepEqual(neighbour(list, { hand: 2, step: 0 }, -1), { hand: 0, step: 1 }, 'back into an earlier hand');
  assert.deepEqual(neighbour(list, { hand: 2, step: 0 }, 1), { hand: 2, step: 4 });
  // A place the list does not hold: in a hand it holds nothing of, and
  // between two decisions of a hand it holds.
  assert.deepEqual(neighbour(list, { hand: 1, step: 3 }, 1), { hand: 2, step: 0 });
  assert.deepEqual(neighbour(list, { hand: 1, step: 3 }, -1), { hand: 0, step: 1 });
  assert.deepEqual(neighbour(list, { hand: 2, step: 2 }, -1), { hand: 2, step: 0 });
  assert.deepEqual(neighbour(list, { hand: 2, step: 2 }, 1), { hand: 2, step: 4 });
  // The ends, from a place the list holds or one beyond it.
  assert.equal(neighbour(list, { hand: 0, step: 1 }, -1), null);
  assert.equal(neighbour(list, { hand: 0, step: 0 }, -1), null);
  assert.equal(neighbour(list, { hand: 2, step: 4 }, 1), null);
  assert.equal(neighbour(list, { hand: 3, step: 0 }, 1), null);
  assert.equal(neighbour([], { hand: 0, step: 0 }, 1), null);
});

test("a hand opens at its first decision the filter holds, or at its first, and a review at the newest hand's", () => {
  const decision = (values = {}) => ({ agreed: false, forced: false, judgement: { gap: 0.5, flagged: false }, ...values });
  const hand = (index, decisions, frames = decisions?.length ?? 3) => ({ index, frames: Array(frames).fill({}), decisions });
  const agreed = decision({ agreed: true, judgement: null });
  const reviewed = hand(0, [agreed, decision(), decision({ judgement: { gap: 0.9, flagged: true } })]), waiting = hand(1, null);
  assert.deepEqual(handEntry([reviewed], 'flagged', 0), { hand: 0, step: 2 });
  assert.deepEqual(handEntry([reviewed], 'differed', 0), { hand: 0, step: 1 });
  assert.deepEqual(handEntry([reviewed], 'every', 0), { hand: 0, step: 0 });
  // A hand the filter holds nothing of, or one not yet reviewed, opens at its
  // first decision; one in which the player made none has nowhere to open.
  assert.deepEqual(handEntry([hand(0, [agreed], 1)], 'flagged', 0), { hand: 0, step: 0 });
  assert.deepEqual(handEntry([reviewed, waiting], 'flagged', 1), { hand: 1, step: 0 });
  assert.equal(handEntry([hand(0, [], 0)], 'every', 0), null);
  // The review opens on the newest hand with a decision of the player's,
  // reviewed or not, passing over a last hand won before their first turn.
  assert.deepEqual(landing([reviewed, waiting], 'flagged'), { hand: 1, step: 0 });
  assert.deepEqual(landing([hand(0, null), hand(1, reviewed.decisions)], 'flagged'), { hand: 1, step: 2 });
  assert.deepEqual(landing([reviewed, hand(1, [], 0)], 'differed'), { hand: 0, step: 1 });
  assert.equal(landing([hand(0, [], 0)], 'every'), null);
  assert.equal(landing([], 'every'), null);
});

test('each move is a word and the tiles it names, and a discard its tile alone', () => {
  assert.deepEqual(moveToken('discard', '5z'), { word: '', tiles: ['5z'] });
  assert.deepEqual(moveToken('riichi', '3s'), { word: 'Riichi', tiles: ['3s'] });
  assert.deepEqual(moveToken('chii', '3m'), { word: 'Chii', tiles: ['3m', '4m', '5m'] });
  assert.deepEqual(moveToken('concealed-kan', '7z'), { word: 'Kan', tiles: ['7z'] });
  assert.deepEqual(moveToken('extended-kan', '2p'), { word: 'Kan', tiles: ['2p'] });
  for (const [kind, word] of [['pon', 'Pon'], ['kan', 'Kan'], ['ron', 'Ron'], ['tsumo', 'Tsumo'], ['pass', 'Pass']]) {
    assert.deepEqual(moveToken(kind, null), { word, tiles: [] }, kind);
  }
});

test('a step says in one line where it is, the table, both moves with their weights, a pick, and whether it was flagged', () => {
  const seat = (wind, values = {}) => ({ seat: wind, turn: false, riichi: false, hand: [], drawn: null, ...values });
  const turn = { phase: 'act', wall: 36, pending_discard: null, pending_from: null,
    seats: [seat('north', { turn: true, drawn: '7z' }), seat('east'), seat('south'), seat('west')] };
  const offer = { ...turn, phase: 'call', wall: 64, pending_discard: '7p', pending_from: 'south',
    seats: [seat('north'), seat('east'), seat('south', { turn: true }), seat('west')] };
  const choice = (kind, tile, label, weight, conditionalWeight = null) => ({ kind, tile, label, weight, conditionalWeight });
  const discards = [choice('discard', '7z', 'discards the red dragon', 0.02), choice('discard', '8s', 'discards the 8 bamboo', 0.955),
    choice('discard', '1z', 'discards the east wind', 0.012)];
  const reaches = [choice('riichi', '3s', 'declares riichi on the 3 bamboo', 0.9, 0.02), choice('riichi', '6m', 'declares riichi on the 6 characters', 0.9, 0.9),
    choice('discard', '3s', 'discards the 3 bamboo', 0.1)];
  const calls = [choice('pon', null, 'calls a triplet', 0.99), choice('pass', null, 'passes', 0.003)];
  const frame = (view, choices, kind, tile = null) => ({ view, choices, played: { kind, tile }, lines: [] });
  const hand = { name: 'East 4', frames: [frame(offer, calls, 'pass'), frame(turn, discards, 'discard', '7z'),
    frame(turn, reaches, 'riichi', '3s'), frame(turn, discards, 'discard', '8s')] };
  const decision = (step, advised, values = {}) => ({ played_kind: hand.frames[step].played.kind, played_tile: hand.frames[step].played.tile,
    advised_kind: advised.kind, advised_tile: advised.tile, advised: advised.label, agreed: false, forced: false,
    judgement: { flagged: false }, weighted: hand.frames[step].choices, ...values });
  const said = (step, decision, options = {}) => stepSaid({ hand, step, decision, ...options });
  const flagged = decision(1, discards[1], { judgement: { flagged: true } });

  assert.equal(said(1, flagged), 'East 4, decision 2 of 4. Your turn, 36 tiles left. You: discards the red dragon, 2.0%. '
    + 'Trained AI: discards the 8 bamboo, 95.5%. Flagged.');
  // A move the player picked there, with its weight, once the answer is shown.
  assert.equal(said(1, flagged, { pick: { kind: 'discard', tile: '1z' } }), 'East 4, decision 2 of 4. Your turn, 36 tiles left. '
    + 'You: discards the red dragon, 2.0%. Trained AI: discards the 8 bamboo, 95.5%. Now: discards the east wind, 1.2%. Flagged.');
  // Hidden, the player's weight goes with Trained AI's move, and the flag stays.
  assert.equal(said(1, flagged, { hidden: true }), 'East 4, decision 2 of 4. Your turn, 36 tiles left. You: discards the red dragon. '
    + "Trained AI's move hidden. Flagged.");
  // Not reviewed yet, the move is known from the table; Trained AI's is not.
  assert.equal(said(1, null), 'East 4, decision 2 of 4. Your turn, 36 tiles left. You: discards the red dragon. Trained AI: not reviewed yet.');
  assert.equal(said(3, decision(3, discards[1], { agreed: true, judgement: null })),
    'East 4, decision 4 of 4. Your turn, 36 tiles left. You: discards the 8 bamboo, 95.5%. Trained AI agreed.');
  assert.equal(said(3, decision(3, discards[1], { agreed: true, forced: true, judgement: null })),
    'East 4, decision 4 of 4. Your turn, 36 tiles left. You: discards the 8 bamboo. Only move.');
  // An answer to another's discard: the table says whose tile it was.
  assert.equal(said(0, decision(0, calls[0])), 'East 4, decision 1 of 4. South discarded the 7 circles, 64 tiles left. '
    + 'You: passes, 0.3%. Trained AI: calls a triplet, 99.0%.');
  // Both declared riichi: the tiles are weighed given the declaration.
  const riichi = decision(2, reaches[1], { judgement: { flagged: true } });
  assert.equal(said(2, riichi), 'East 4, decision 3 of 4. Your turn, 36 tiles left. You: declares riichi on the 3 bamboo, 2.0% given riichi. '
    + 'Trained AI: declares riichi on the 6 characters, 90.0% given riichi. Flagged.');
  assert.deepEqual(moveWeight(riichi, { kind: 'riichi', tile: '3s' }), { weight: 0.02, given: true });
  assert.deepEqual(moveWeight(riichi, { kind: 'discard', tile: '3s' }), { weight: 0.1, given: false });
  // Trained AI alone declared: the declaration itself is weighed, as judged.
  const declared = { ...riichi, played_kind: 'discard', played_tile: '3s' };
  assert.deepEqual(moveWeight(declared, { kind: 'riichi', tile: '6m' }), { weight: 0.9, given: false });
  assert.deepEqual(moveWeight({ ...flagged, weighted: [{ ...discards[0], weight: null }] }, discards[0]), { weight: null, given: false },
    'a move Trained AI gave no weight');
});

test("a real review's lists hold what their filters count, in the order played: the flagged among the different moves, and every decision", async () => {
  const review = new MatchReview(Game, game, { analyzeBatch: network() });
  try {
    review.review();
    await settled(review);
    const decisions = review.decisions, counts = filterCounts(decisions);
    const at = place => review.hands[place.hand].decisions[place.step];
    const flagged = playlist(review.hands, 'flagged').map(at), other = playlist(review.hands, 'differed').map(at);
    assert.equal(counts.every, decisions.length);
    assert.equal(counts.flagged, flagged.length);
    assert.equal(counts.differed, other.length);
    assert.ok(counts.flagged > 0 && counts.differed >= counts.flagged && counts.every > counts.differed, JSON.stringify(counts));
    assert.ok(flagged.every(decision => decision.judgement.flagged && other.includes(decision)), 'every flagged decision differs');
    assert.ok(other.every(decision => !decision.agreed && !decision.forced));
    // Hand by hand, each hand's decisions in the order they were made.
    const order = decision => decisions.indexOf(decision);
    for (const list of [flagged, other]) assert.deepEqual(list.map(order), [...list.map(order)].sort((a, b) => a - b));
    assert.deepEqual(playlist(review.hands, 'every').map(at), decisions, 'every decision in the order played');
  } finally { review.dispose(); }
});

test('the whole game is reviewed in batches, and judged as the hand review judges each hand', async () => {
  const sizes = [];
  const review = new MatchReview(Game, game, { analyzeBatch: network(sizes) });
  try {
    review.review();
    await settled(review);
    assert.equal(review.failure, '');
    assert.ok(review.complete);
    assert.equal(review.reviewedHands, review.hands.length);
    assert.ok(sizes.some(size => size === BATCH), 'full batches are asked');
    const decisions = review.decisions;
    assert.equal(decisions.length, review.hands.reduce((sum, hand) => sum + hand.frames.length, 0));
    const forced = decisions.filter(decision => decision.forced).length;
    const reaches = decisions.filter(decision => !decision.forced && decision.preferred_conditional_weight !== null).length;
    assert.equal(sizes.reduce((sum, size) => sum + size, 0), decisions.length - forced + reaches,
      'a forced move is not asked, and a declaration is asked its tile');
    // One hand, reviewed one position at a time as the hand review does it.
    for (const hand of [review.hands[0], review.hands.at(-1)]) {
      const match = atHandEnd(game, hand.end);
      try {
        const single = await reviewWithStrong(match.engine, match.engine.review(), async (planes, mask) => decisive(mask));
        assert.equal(hand.decisions.length, single.length);
        hand.decisions.forEach((decision, index) => {
          const { hand: _hand, index: _index, frame, judgement: judged, measures: _measures, ...row } = decision;
          assert.deepEqual(row, single[index]);
          assert.equal(frame, hand.frames[index]);
          assert.deepEqual(judged, judgement(row));
        });
      } finally { match.dispose(); }
    }
    const flagged = rankDecisions(decisions);
    assert.ok(flagged.length > 5, 'a network sure of other moves flags many');
    assert.ok(flagged.every(decision => decision.judgement.flagged && !decision.agreed));
  } finally { review.dispose(); }
});

test('a flagged turn carries the review numbers of every move that was open, the one played as its note gives them', async () => {
  const review = new MatchReview(Game, game, { analyzeBatch: network() });
  try {
    review.review([0]);
    await settled(review);
    const hand = review.hands[0];
    const match = atHandEnd(game, hand.end);
    try {
      const notes = match.engine.review();
      let checked = 0;
      hand.decisions.forEach((decision, index) => {
        if (decision.call_tile || decision.agreed || decision.forced) { assert.equal(decision.measures, undefined); return; }
        const note = notes[index];
        const played = decision.measures.find(measure => measure.kind === note.played_kind && measure.tile === (note.played_tile ?? null));
        assert.deepEqual(played, { kind: note.played_kind, tile: note.played_tile ?? null, shanten: note.shanten_played,
          acceptance: note.acceptance_played, danger: note.danger_played, dora: note.dora_played });
        assert.ok(decision.measures.some(measure => measure.kind === decision.advised_kind && measure.tile === (decision.advised_tile ?? null)));
        if (decision.judgement?.flagged) assert.equal(decision.measures.length, decision.weighted.length);
        checked++;
      });
      assert.ok(checked > 0);
    } finally { match.dispose(); }
  } finally { review.dispose(); }
});

/** Each player's concealed tiles, the tile just drawn among them, rebuilt
 * from a hand's mjai log by player number, which is all a replayer of the
 * log has to go on: the deal, the draws, the discards and the tiles each
 * call or quad takes from the hand. */
function tilesFromLog(log) {
  const ours = name => ('ESWNPFC'.includes(name) ? `${'ESWNPFC'.indexOf(name) + 1}z` : name);
  const held = [[], [], [], []];
  const take = (actor, tiles) => {
    for (const tile of tiles) {
      const at = held[actor].indexOf(ours(tile));
      assert.ok(at >= 0, `player ${actor} holds the ${tile} they give up`);
      held[actor].splice(at, 1);
    }
  };
  for (const event of log.split('\n').filter(Boolean).map(line => JSON.parse(line))) {
    if (event.type === 'start_kyoku') held.splice(0, 4, ...event.tehais.map(tiles => tiles.map(ours)));
    else if (event.type === 'tsumo') held[event.actor].push(ours(event.pai));
    else if (event.type === 'dahai' || event.type === 'kakan') take(event.actor, [event.pai]);
    else if (['chi', 'pon', 'daiminkan', 'ankan'].includes(event.type)) take(event.actor, event.consumed);
  }
  return held.map(tiles => tiles.sort());
}

test('every hand at every step of a replay is the one its log rebuilds, and every tile of the set is accounted for', () => {
  // A game that throws its draws, and one that calls every set and declares
  // every quad, so the hands shrink by calls and quads on both sides. Both
  // players declare every riichi and take every win they are offered, though
  // such play seldom gets one: of the first 24 seeds, only the calling game
  // of seed 16 offers its player a win at a decision.
  const seen = { decisions: 0, answers: 0, calls: 0, quads: 0, riichi: 0, drawn: 0, wins: 0 };
  for (const snapshot of [game, finishedGame(16, 'beginner', { calls: true })]) {
    const frames = new Map(scanMatch(Game, snapshot).flatMap(hand => hand.frames.map(frame => [frame.command, frame])));
    const match = new MatchSession(Game, snapshot.seed, snapshot.opponents);
    try {
      match.advance(false);
      let checked = 0;
      for (const [at, command] of snapshot.commands.entries()) {
        const frame = frames.get(at);
        if (frame) {
          // An engine of the test's own, played to the decision: its hand's
          // log so far, and its table.
          const { view } = frame, rebuilt = tilesFromLog(match.engine.log());
          assert.deepEqual(match.view, view, 'the frame is the table at the decision');
          assert.equal(frame.hands.length, 4);
          let counted = view.wall + 14;
          frame.hands.forEach((held, slot) => {
            const seat = view.seats[slot], tiles = [...held.hand, ...(held.drawn ? [held.drawn] : [])];
            assert.deepEqual([held.seat, held.player], [seat.seat, seat.player], 'the seats in the order of the view');
            assert.deepEqual([...tiles].sort(), rebuilt[seat.player], `${seat.seat} at command ${at}`);
            assert.equal(tiles.length, seat.hand_size);
            assert.ok(!held.drawn || (seat.turn && view.phase === 'act'), 'only the seat to play holds a tile apart');
            counted += tiles.length + seat.melds.reduce((sum, meld) => sum + meld.tiles.length, 0)
              + seat.discards.filter(discard => !discard.claimed).length;
            if (slot && seat.melds.length) seen.calls++;
            if (seat.melds.some(meld => meld.tiles.length === 4)) seen.quads++;
            if (seat.riichi) seen.riichi++;
            if (held.drawn) seen.drawn++;
          });
          assert.deepEqual([frame.hands[0].hand, frame.hands[0].drawn], [view.seats[0].hand, view.seats[0].drawn],
            "the player's own tiles as the table showed them");
          // The hands, the sets, the discards still lying, the wall and the
          // dead wall hold the whole set between them.
          assert.equal(counted, 136, `command ${at}`);
          if (['tsumo', 'ron'].includes(frame.played.kind)) seen.wins++;
          if (view.phase === 'call') seen.answers++;
          checked++;
        }
        match.apply(command);
        match.advance(false);
      }
      assert.equal(checked, frames.size, 'every decision of every hand');
      seen.decisions += checked;
    } finally { match.dispose(); }
  }
  // Each kind of position was seen: answers to discards, opponents holding
  // fewer tiles for their calls, a quad on the table, riichi declared, a
  // tile just drawn, and wins.
  for (const [what, count] of Object.entries(seen)) assert.ok(count > 0, `${what}: ${JSON.stringify(seen)}`);
});

test('the hands at a decision are refused while that hand is still being played', () => {
  const match = new MatchSession(Game, game.seed, game.opponents);
  try {
    match.advance(false);
    const first = game.commands.findIndex(command => command.type === 'choose');
    for (const command of game.commands.slice(0, first + 1)) { match.apply(command); match.advance(false); }
    assert.equal(match.engine.review().length, 1);
    assert.throws(() => match.engine.review_hands(0), /the hand is still being played/);
  } finally { match.dispose(); }
});

test('a hand asked for out of order is reviewed first, and an earlier one after it replays from the deal', async () => {
  const review = new MatchReview(Game, game, { analyzeBatch: network() });
  try {
    const last = review.hands.length - 1;
    review.review([last]);
    await settled(review);
    assert.deepEqual(review.hands.map(hand => hand.status), review.hands.map((_, index) => (index === last ? 'done' : 'waiting')));
    review.review([1]);
    await settled(review);
    assert.equal(review.hands[1].status, 'done');
    const match = atHandEnd(game, review.hands[1].end);
    try {
      const single = await reviewWithStrong(match.engine, match.engine.review(), async (planes, mask) => decisive(mask));
      assert.deepEqual(review.hands[1].decisions.map(decision => decision.advised), single.map(row => row.advised));
    } finally { match.dispose(); }
    assert.equal(review.complete, false);
  } finally { review.dispose(); }
});

test('a slow device holds after each hand until the player asks for the next one or the rest', async () => {
  let clock = 0;
  const slow = async (positions, signal) => {
    clock += positions.length * (SLOW_MS + 100);
    return network()(positions, signal);
  };
  const review = new MatchReview(Game, game, { analyzeBatch: slow, now: () => clock });
  try {
    review.review();
    await settled(review);
    assert.ok(review.slow);
    assert.equal(review.held, true);
    assert.equal(review.reviewedHands, 1);
    assert.ok(review.estimate > 0, 'the rest is estimated from the speed seen');
    review.resume();
    await settled(review);
    assert.equal(review.reviewedHands, 2);
    assert.equal(review.held, true);
    review.resume({ patient: true });
    await settled(review);
    assert.ok(review.complete);
    assert.equal(review.held, false);
  } finally { review.dispose(); }
});

test('stopping puts the hand being reviewed back, and a failure says what failed and can be retried', async () => {
  // The final hand, asked about first, and the first hand are answered; the
  // second hand's first batch waits until the review is stopped.
  let review = null;
  const waiting = async (positions, signal) => {
    if (review.current === 1) {
      await new Promise((_, reject) => signal.addEventListener('abort',
        () => reject(new DOMException('Review changed', 'AbortError')), { once: true }));
    }
    return positions.map(position => decisive(position.mask));
  };
  review = new MatchReview(Game, game, { analyzeBatch: waiting });
  try {
    review.review();
    for (let turn = 0; turn < 1000 && review.current !== 1; turn++) await new Promise(resolve => setImmediate(resolve));
    assert.equal(review.hands[1].status, 'reviewing');
    review.stop();
    await settled(review);
    assert.deepEqual(review.hands.slice(0, 3).map(hand => hand.status), ['done', 'waiting', 'waiting']);
    assert.equal(review.failure, '');
    assert.deepEqual(review.queue, []);
  } finally { review.dispose(); }

  let fail = true;
  const flaky = async (positions, signal) => {
    if (fail) throw new Error('The trained opponent did not answer in time');
    return network()(positions, signal);
  };
  const retried = new MatchReview(Game, game, { analyzeBatch: flaky });
  try {
    retried.review();
    await settled(retried);
    assert.equal(retried.failure, 'The trained opponent did not answer in time');
    assert.ok(retried.hands.every(hand => hand.status === 'waiting'));
    fail = false;
    retried.review();
    await settled(retried);
    assert.equal(retried.failure, '');
    assert.ok(retried.complete);
  } finally { retried.dispose(); }
});

test("the notes' spells of asking keep the gate shut until the last has ended, each ended once however often it is told", async () => {
  const asking = new Asking();
  assert.equal(asking.gate(), null, 'open while nothing is asked');
  const first = asking.begin(), shut = asking.gate();
  assert.ok(shut instanceof Promise);
  const second = asking.begin();
  assert.equal(asking.gate(), shut, 'one gate for the spells under way together');
  let opened = false;
  void shut.then(() => { opened = true; });
  // The first spell told twice that it has ended: counted out twice, it
  // would open the gate on the second, still under way.
  first();
  first();
  await turn();
  assert.equal(asking.gate(), shut);
  assert.equal(opened, false);
  second();
  assert.equal(asking.gate(), null, 'open again before whoever waits there goes on');
  await turn();
  assert.equal(opened, true);
  // The next spell shuts it afresh, and an old one told again leaves it so.
  const third = asking.begin();
  assert.ok(asking.gate() instanceof Promise && asking.gate() !== shut);
  second();
  assert.notEqual(asking.gate(), null);
  third();
  assert.equal(asking.gate(), null);
});

test('a review waits at its gate while the notes ask, and goes on where it was, the time it waited not held against the device', async () => {
  // At the third hand's result, which is asked about first, then the first
  // two. The notes begin to ask while the review's second batch is under
  // way, so its third waits, and a clock that moves only at the gate stands
  // for the notes' time there: a minute, which, held against the device,
  // would make it slow and have it hold after the hand.
  const hands = scanMatch(Game, game), saved = savedAfter(game, hands[2].end), alone = await ungatedSizes(saved);
  const asking = new Asking(), sizes = [], waits = [];
  let clock = 0, stopAsking = null;
  const review = gatedReview(saved, asking, { sizes, waits, now: () => clock,
    onBatch: sent => { if (sent === 1) stopAsking = asking.begin(); } });
  try {
    review.review();
    await until(() => waits.length > 0, 'the review comes to its gate');
    for (let count = 0; count < 50; count++) await turn();
    assert.deepEqual(sizes, alone.slice(0, 2), 'nothing more is sent while the notes ask');
    // Nothing is given up while it waits: the hand under review and how far
    // it has got, the hands queued, and the player's word as it was.
    const reviewing = review.hands[2];
    assert.deepEqual({ busy: review.busy, current: review.current, status: reviewing.status, queue: review.queue,
      requested: review.requested, patient: review.patient, held: review.held, failure: review.failure },
    { busy: true, current: 2, status: 'reviewing', queue: [0, 1], requested: true, patient: false, held: false, failure: '' });
    assert.ok(reviewing.done > 0 && reviewing.done < reviewing.frames.length, `the hand is part reviewed: ${reviewing.done}`);
    assert.ok(new Set(waits).size === 1 && waits[0] === 2, 'only ever at the hand under review');
    clock += 60000;
    stopAsking();
    await settled(review);
    assert.equal(review.failure, '');
    assert.ok(review.complete, 'it goes on to the end by itself');
    assert.equal(review.slow, false, 'the time at the gate is not held against the device');
    // Every batch an ungated review sends, each once: the hand went on from
    // its third batch, and nothing was asked again or left out.
    assert.deepEqual(sizes, alone);
  } finally { review.dispose(); }
});

test("a slow device waits at the gate as any other, and still holds after the hand for the player's word", async () => {
  const hands = scanMatch(Game, game), saved = savedAfter(game, hands[2].end);
  const asking = new Asking(), sizes = [], waits = [], answer = network(sizes);
  let clock = 0, stopAsking = null;
  const review = gatedReview(saved, asking, { sizes, waits, now: () => clock,
    analyzeBatch: (positions, signal) => {
      clock += positions.length * (SLOW_MS + 100);
      return answer(positions, signal);
    },
    onBatch: sent => { if (sent === 1) stopAsking = asking.begin(); } });
  try {
    review.review();
    await until(() => waits.length > 0, 'the review comes to its gate');
    stopAsking();
    await settled(review);
    // The hand just played is reviewed, and the review holds after it as it
    // would have without the notes, the rest queued for the player's word.
    assert.equal(review.slow, true);
    assert.deepEqual({ held: review.held, queue: review.queue, requested: review.requested, patient: review.patient,
      reviewed: review.reviewedHands, failure: review.failure },
    { held: true, queue: [0, 1], requested: true, patient: false, reviewed: 1, failure: '' });
    // The player asks for the rest while the notes ask again: the review
    // comes to its gate before its first batch, the word kept meanwhile.
    stopAsking = asking.begin();
    const waited = waits.length, sent = sizes.length;
    review.resume({ patient: true });
    await until(() => waits.length > waited, 'the review comes to its gate again');
    assert.deepEqual({ busy: review.busy, current: review.current, queue: review.queue, patient: review.patient, held: review.held, sent: sizes.length },
      { busy: true, current: 0, queue: [1], patient: true, held: false, sent });
    stopAsking();
    await settled(review);
    assert.equal(review.failure, '');
    assert.ok(review.complete, "the player's word holds: no hold after the first hand");
  } finally { review.dispose(); }
});

test('a review stopped or closed while it waits at its gate stops waiting at once', async () => {
  // Notes that do not stop asking until the end: the gate shuts while the
  // review's first batch is under way, and stays shut. A review that went on
  // waiting would never settle, so it is given a few turns, not for ever.
  const hands = scanMatch(Game, game), saved = savedAfter(game, hands[2].end);
  const asking = new Asking(), sizes = [], waits = [];
  let stopAsking = null;
  const review = gatedReview(saved, asking, { sizes, waits, onBatch: sent => { if (sent === 0) stopAsking = asking.begin(); } });
  try {
    review.review();
    await until(() => waits.length > 0, 'the review comes to its gate');
    review.stop();
    await until(() => !review.busy, 'the stopped review stops waiting');
    // Stopped as at any other moment: the hand under review goes back to
    // waiting, nothing is left queued, and nothing is said to have failed.
    assert.deepEqual(review.hands.map(hand => hand.status), ['waiting', 'waiting', 'waiting']);
    assert.deepEqual({ queue: review.queue, current: review.current, failure: review.failure }, { queue: [], current: null, failure: '' });
    // Asked for again, it comes to the gate again, before its first batch,
    // and is closed there.
    const waited = waits.length, sent = sizes.length;
    review.review();
    await until(() => waits.length > waited, 'the review comes to its gate again');
    review.dispose();
    await until(() => !review.busy, 'the closed review stops waiting');
    assert.equal(review.replay, null, 'its replay is freed');
    // The notes stop at last: a closed review asks nothing more.
    stopAsking();
    for (let count = 0; count < 50; count++) await turn();
    assert.equal(sizes.length, sent);
    assert.equal(review.busy, false);
  } finally { review.dispose(); }
});

test('a whole-game review held at its gate goes on by itself once let through, and the game joins the trend once', async () => {
  // The review of the finished game, from the standings. The last hand's
  // notes ask Trained AI while the fourth hand is reviewed, and stop only to
  // ask again at once, as Svelte runs them again when what they review
  // changes under them: the gate opens and shuts in the same turn, and the
  // review, looking again, waits on.
  const alone = await ungatedSizes(game), whole = [];
  const ungated = new MatchReview(Game, game, { analyzeBatch: network(), onComplete: done => whole.push(done.summary) });
  try {
    ungated.review();
    await settled(ungated);
  } finally { ungated.dispose(); }
  const asking = new Asking(), sizes = [], waits = [], reported = [];
  let stopAsking = null;
  const review = gatedReview(game, asking, { sizes, waits, onComplete: done => reported.push(done.summary),
    onBatch: () => { if (review.current === 3 && !stopAsking) stopAsking = asking.begin(); } });
  try {
    review.review();
    await until(() => waits.length > 0, 'the review comes to its gate');
    const sent = sizes.length;
    stopAsking();
    const stopAgain = asking.begin();
    for (let count = 0; count < 50; count++) await turn();
    assert.equal(sizes.length, sent, 'nothing is sent while the notes ask again');
    assert.equal(review.current, 3);
    stopAgain();
    // Nothing asks for the review again: it goes on to the end by itself.
    await settled(review);
    assert.equal(review.failure, '');
    assert.ok(review.complete);
    assert.deepEqual(sizes, alone, 'every batch once');
    assert.deepEqual(reported, whole, 'the game joins the trend as an ungated review has it');
    review.review();
    await settled(review);
    assert.equal(reported.length, 1, 'and only once');
  } finally { review.dispose(); }
});

test('a gate that fails stops the review, saying why, rather than leaving it waiting', async () => {
  let failing = true;
  const review = new MatchReview(Game, game, { analyzeBatch: network(),
    gate: () => (failing ? Promise.reject(new Error('The notes could not say whether they ask')) : null) });
  try {
    review.review();
    await settled(review);
    assert.equal(review.failure, 'The notes could not say whether they ask');
    assert.ok(review.hands.every(hand => hand.status === 'waiting'));
    failing = false;
    review.review();
    await settled(review);
    assert.equal(review.failure, '');
    assert.ok(review.complete);
  } finally { review.dispose(); }
});

test('a finished review is reported once, with what the trend keeps of it', async () => {
  const reported = [];
  const review = new MatchReview(Game, game, { analyzeBatch: network(), onComplete: done => reported.push(done.summary) });
  try {
    review.review([0, 1]);
    await settled(review);
    assert.deepEqual(reported, [], 'a part of the game is not the game');
    review.review();
    await settled(review);
    review.review();
    await settled(review);
    const { matched, decisions } = agreement(review.decisions);
    assert.deepEqual(reported, [{ id: `${game.seed}:${game.commands.length}`, matched, decisions,
      flagged: rankDecisions(review.decisions).length }]);
  } finally { review.dispose(); }
});

test('a review asks about the newest hand first, the hand just played or the final one, and keeps the game so far out of the trend', async () => {
  const hands = scanMatch(Game, game), saved = savedAfter(game, hands[2].end);
  const order = [], reported = [];
  const review = notingReview(saved, order, { onComplete: done => reported.push(done.summary) });
  try {
    assert.equal(review.over, false);
    assert.equal(review.id, `${game.seed}:${hands[2].end}`);
    review.review();
    await settled(review);
    assert.equal(review.failure, '');
    assert.deepEqual(order, [2, 0, 1], 'the hand just played, then the rest from the first deal');
    assert.equal(review.complete, true);
    assert.deepEqual(reported, [], 'the game so far is not a game for the trend');
  } finally { review.dispose(); }
  // A finished game is asked about its final hand first, the hand its review
  // opens on, then from the first deal.
  const asked = [], finished = notingReview(game, asked);
  try {
    assert.equal(finished.over, true);
    finished.review();
    await settled(finished);
    const indices = finished.hands.map(hand => hand.index);
    assert.deepEqual(asked, [indices.at(-1), ...indices.slice(0, -1)]);
  } finally { finished.dispose(); }
});

test('a review taken on to later hands keeps what it reviewed, and asks only about the new hands, the newest first', async () => {
  const hands = scanMatch(Game, game), later = savedAfter(game, hands[3].end);
  const order = [], sizes = [];
  const review = notingReview(savedAfter(game, hands[1].end), order, { sizes });
  try {
    review.review();
    await settled(review);
    const kept = review.hands.map(hand => hand.decisions);
    review.extend(later);
    assert.deepEqual(review.hands.map(hand => hand.status), ['done', 'done', 'waiting', 'waiting']);
    assert.deepEqual(review.hands.map(hand => hand.end), hands.slice(0, 4).map(hand => hand.end));
    kept.forEach((decisions, index) => assert.equal(review.hands[index].decisions, decisions, `hand ${index} keeps its verdicts`));
    assert.equal(review.requested, false, 'the new hands wait for the review to be opened');
    assert.equal(review.over, false);
    order.length = 0;
    sizes.length = 0;
    review.review();
    await settled(review);
    assert.equal(review.failure, '');
    assert.deepEqual(order, [3, 2]);
    assert.ok(review.complete);
    // As much as a review begun at the later result asks of those two hands.
    const fresh = [], alone = new MatchReview(Game, later, { analyzeBatch: network(fresh) });
    try {
      alone.review([3, 2]);
      await settled(alone);
      const total = list => list.reduce((sum, size) => sum + size, 0);
      assert.equal(total(sizes), total(fresh));
    } finally { alone.dispose(); }
  } finally { review.dispose(); }
});

test('only the same match, saved later, is taken on', async () => {
  const hands = scanMatch(Game, game), saved = savedAfter(game, hands[1].end), later = savedAfter(game, hands[3].end);
  const review = new MatchReview(Game, saved, { analyzeBatch: network() });
  try {
    review.review();
    await settled(review);
    const altered = { ...later, commands: later.commands.map((command, at) => (at === 0 ? { ...command, tile: command.tile === '1m' ? '2m' : '1m' } : command)) };
    for (const [what, other] of [['another seed', { ...later, seed: later.seed + 1 }], ['an earlier move changed', altered],
      ['fewer commands', savedAfter(game, hands[0].end)]]) {
      assert.throws(() => review.extend(other), /not the match under review/, what);
      assert.equal(review.hands.length, 2, what);
    }
    // The same match, saved at the same result: nothing to take on.
    review.extend(savedAfter(game, hands[1].end));
    assert.equal(review.hands.length, 2);
    assert.equal(review.requested, true);
  } finally { review.dispose(); }
});

test('a review taken on while it reviews a hand plays on along the longer match', async () => {
  const hands = scanMatch(Game, game), reported = [];
  const review = new MatchReview(Game, savedAfter(game, hands[1].end), { analyzeBatch: network(), onComplete: done => reported.push(done.summary) });
  try {
    // The hand just played first, then the first hand, which replays from the
    // deal along the commands saved at the second hand's result.
    review.review();
    for (let turn = 0; turn < 1000 && review.current !== 0; turn++) await new Promise(resolve => setImmediate(resolve));
    assert.equal(review.current, 0);
    assert.ok(review.busy);
    review.extend(game);
    assert.equal(review.over, true);
    await settled(review);
    assert.equal(review.failure, '');
    assert.deepEqual(review.hands.slice(0, 2).map(hand => hand.status), ['done', 'done']);
    // The later hands, the final one first, play on along the whole game.
    review.review();
    await settled(review);
    assert.equal(review.failure, '');
    assert.ok(review.complete);
    assert.equal(reported.length, 1);
  } finally { review.dispose(); }
});

test('a game joins the trend once, as the match ends, also when every hand was reviewed before it did', async () => {
  const hands = scanMatch(Game, game), sizes = [], reported = [];
  const review = new MatchReview(Game, savedAfter(game, hands.at(-1).end), { analyzeBatch: network(sizes), onComplete: done => reported.push(done.summary) });
  try {
    review.review();
    await settled(review);
    assert.ok(review.complete);
    assert.deepEqual(reported, [], 'not while the last hand waits at its result');
    const asked = sizes.length;
    review.extend(game);
    assert.equal(review.over, true);
    assert.equal(review.id, `${game.seed}:${game.commands.length}`);
    assert.equal(reported.length, 1, 'reported the moment the match ends');
    assert.equal(sizes.length, asked, 'nothing is asked again');
    await settled(review);
    assert.equal(reported.length, 1);
  } finally { review.dispose(); }
  // The same figure as a review of the finished game.
  const fresh = new MatchReview(Game, game, { analyzeBatch: network() });
  let whole;
  try {
    fresh.review();
    await settled(fresh);
    whole = fresh.summary;
  } finally { fresh.dispose(); }
  assert.deepEqual(reported, [whole]);

  // From an earlier result, the rest is asked for once the match is over,
  // and the game reported as that review completes.
  const finished = [];
  const earlier = new MatchReview(Game, savedAfter(game, hands[1].end), { analyzeBatch: network(), onComplete: done => finished.push(done.summary) });
  try {
    earlier.review();
    await settled(earlier);
    earlier.extend(game);
    assert.deepEqual(finished, []);
    assert.equal(earlier.requested, false);
    earlier.review();
    await settled(earlier);
    assert.deepEqual(finished, [whole]);
  } finally { earlier.dispose(); }
});

test('a closed review asks nothing more and frees its replay', async () => {
  const review = new MatchReview(Game, game, { analyzeBatch: network() });
  review.review();
  review.dispose();
  await settled(review);
  assert.equal(review.replay, null);
  const before = review.decisions.length;
  review.review();
  await settled(review);
  assert.equal(review.decisions.length, before);
});
