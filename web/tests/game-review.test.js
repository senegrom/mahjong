import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession } from '../src/lib/session.js';
import { reviewWithStrong } from '../src/lib/review-policy.js';
import { BATCH, FLAG_PLAYED_BELOW, FLAG_PREFERRED_FROM, LEADING, MatchReview, SLOW_MS, agreement, handName, judgement,
  movedNote, ordinal, rankDecisions, scanMatch } from '../src/lib/game-review.js';

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

/** The replayed engine at the end of one hand, for comparisons. */
function atHandEnd(snapshot, end) {
  const match = new MatchSession(Game, snapshot.seed, snapshot.opponents);
  match.advance(false);
  for (const command of snapshot.commands.slice(0, end)) { match.apply(command); match.advance(false); }
  return match;
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
  assert.throws(() => scanMatch(Game, moved), /could not be replayed/);
  assert.throws(() => scanMatch(Game, { ...game, commands: game.commands.slice(0, -1) }), /could not be replayed/);
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

test('flagged decisions rank by the gap, ties in the order played, and agreement leaves forced moves out', () => {
  const decision = (name, gap, flagged = true) => ({ name, judgement: { gap, flagged } });
  const ranked = rankDecisions([decision('a', 0.7), decision('b', 0.9), decision('c', 0.99, false), decision('d', 0.7), { name: 'e', judgement: null }]);
  assert.deepEqual(ranked.map(entry => entry.name), ['b', 'a', 'd']);
  assert.equal(LEADING, 5);
  assert.deepEqual(agreement([{ agreed: true }, { agreed: false }, { agreed: true, forced: true }, { agreed: true }]),
    { matched: 2, decisions: 3, share: 2 / 3 });
  assert.deepEqual(agreement([]), { matched: 0, decisions: 0, share: null });
});

test('a decision ranked lower by the hands reviewed since is told where it went, in places as they are said', () => {
  assert.deepEqual([1, 2, 3, 4, 10, 11, 12, 13, 21, 22, 23, 101, 111, 112].map(ordinal),
    ['1st', '2nd', '3rd', '4th', '10th', '11th', '12th', '13th', '21st', '22nd', '23rd', '101st', '111th', '112th']);
  // The place only: how many are flagged grows with every hand reviewed, and
  // a count said once would go stale while the place still held.
  assert.equal(movedNote(6, true),
    'Decisions reviewed since departed further from Trained AI: this one is now 7th, and every flagged decision is shown.');
  assert.equal(movedNote(2, false), 'Decisions reviewed since departed further from Trained AI: this one is now 3rd.');
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
          const { hand: _hand, index: _index, frame, judgement: judged, measures: _measures, hands: _hands, ...row } = decision;
          assert.deepEqual(row, single[index]);
          assert.equal(frame, hand.frames[index]);
          assert.deepEqual(judged, judgement(row));
        });
      } finally { match.dispose(); }
    }
    const flagged = rankDecisions(decisions);
    assert.ok(flagged.length > LEADING, 'a network sure of other moves flags many');
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
 * call takes from the hand. */
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

test('every hand at every decision is the one its log rebuilds, and every tile of the set is accounted for', async () => {
  // A game that throws its draws, and one that calls every set and declares
  // every quad, so the hands shrink by calls and quads on both sides.
  const calling = finishedGame(5, 'beginner', { calls: true });
  let quads = 0, called = 0;
  for (const snapshot of [game, calling]) {
    const review = new MatchReview(Game, snapshot, { analyzeBatch: network() });
    const match = new MatchSession(Game, snapshot.seed, snapshot.opponents);
    try {
      review.review();
      await settled(review);
      assert.ok(review.complete);
      const byCommand = new Map(review.decisions.map(decision => [decision.frame.command, decision]));
      let checked = 0;
      match.advance(false);
      for (const [at, command] of snapshot.commands.entries()) {
        const decision = byCommand.get(at);
        if (decision) {
          // A replay of the game's commands on an engine of the test's own,
          // read at the decision: its hand's log, and its table.
          const { view } = decision.frame, rebuilt = tilesFromLog(match.engine.log());
          assert.deepEqual(match.view, view, 'the frame is the table at the decision');
          assert.equal(decision.hands.length, 4);
          let counted = view.wall + 14;
          decision.hands.forEach((held, slot) => {
            const seat = view.seats[slot], tiles = [...held.hand, ...(held.drawn ? [held.drawn] : [])];
            assert.deepEqual([held.seat, held.player], [seat.seat, seat.player], 'the seats in the order of the view');
            assert.deepEqual([...tiles].sort(), rebuilt[seat.player], `${seat.seat} at ${decision.hand}:${decision.index}`);
            assert.equal(tiles.length, seat.hand_size);
            assert.ok(!held.drawn || seat.turn, 'only the seat whose turn it is holds a tile apart');
            counted += tiles.length + seat.melds.reduce((sum, meld) => sum + meld.tiles.length, 0)
              + seat.discards.filter(discard => !discard.claimed).length;
            if (slot && seat.melds.length) called++;
            if (seat.melds.some(meld => meld.tiles.length === 4)) quads++;
          });
          assert.deepEqual([decision.hands[0].hand, decision.hands[0].drawn], [view.seats[0].hand, view.seats[0].drawn],
            "the player's own tiles as the table showed them");
          // The hands, the sets, the discards still lying, the wall and the
          // dead wall hold the whole set between them.
          assert.equal(counted, 136, `${decision.hand}:${decision.index}`);
          checked++;
        }
        match.apply(command);
        match.advance(false);
      }
      assert.equal(checked, review.decisions.length, 'every decision of every hand');
    } finally { review.dispose(); match.dispose(); }
  }
  assert.ok(called > 0 && quads > 0, `opponents after a call at ${called} decisions, a quad on the table at ${quads}`);
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
  // The first hand is answered; the second hand's first batch waits until
  // the review is stopped.
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
