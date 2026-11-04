import test from 'node:test';
import assert from 'node:assert/strict';
import { MORTAL_REACH } from '../src/lib/action-weights.js';
import { policyWeights } from '../src/lib/policy-weights.js';
import { average, hesitates, swapMoves } from '../src/lib/suits.js';
import {
  DAMA, DAMA_RULE, DECLINE, DECLINE_RULE, MORTAL_AGARI, MORTAL_PASS, RULE_REFUSED, RULE_WORDS, agentTable, bestOpen,
  forcedAnswer, opponentTable, reviewTable, ruleDecides, ruleLine, ruleSaid, trainedMove, trainedMoves,
} from '../src/lib/trained-move.js';

// The last hand's rules over Trained AI's averaged answer (src/lib/trained-move.js),
// with the network and the engine stood in for: each question is answered from
// logits set by hand, made weights as the worker makes them, and each table's
// codes are set by hand. The rules against the Python wrappers they are written
// from are the parity check's (tests/last-hand-rules-parity.mjs); the engine's
// codes are its native tests'.

const { KEEPS_FIRST, NO_YAKU, SHORT, DEAD, NOT_WAITING } = DAMA;
const { LAST, NOT_LAST } = DECLINE;
/** Mortal's moves `open`, as the engine's masks give them. */
const mask = open => Uint8Array.from({ length: 46 }, (_, move) => (open.includes(move) ? 1 : 0));
/** Logits over Mortal's moves: those named, nought elsewhere. */
const logits = named => Array.from({ length: 46 }, (_, move) => named[move] ?? 0);
/** Dama codes, a byte a tile kind: those named, nought elsewhere. */
const codes = named => Uint8Array.from({ length: 34 }, (_, tile) => named[tile] ?? 0);
const NO_READS = Object.freeze({ renamed: 0, afterReach: 0, reachRenamed: 0, dama: 0, decline: 0 });

/** A decision as trainedMoves reads it, named `name`: the moves its first
 * question opens, the tiles its reach may throw and its codes, every read of
 * a lazy question counted in `reads`. Its questions are its name, "<name>
 * renamed", "<name> reach" and "<name> reach renamed". */
function table(name, { open, tiles = [], dama = {}, decline = 0 }) {
  const reads = { ...NO_READS };
  return {
    reads, planes: name, mask: mask(open),
    renamed: () => { reads.renamed++; return `${name} renamed`; },
    afterReach: () => {
      reads.afterReach++;
      return { planes: `${name} reach`, mask: mask(tiles), renamed: () => { reads.reachRenamed++; return `${name} reach renamed`; } };
    },
    dama: () => { reads.dama++; return codes(dama); },
    decline: () => { reads.decline++; return decline; },
  };
}

/** An asker that answers each question by its name from `answers`, logits
 * made weights over the mask it is sent, as the worker answers; every
 * request is noted, by the questions' names, and shown to `onAsk`. */
function asker(answers, onAsk = () => {}) {
  const requests = [];
  const ask = async positions => {
    requests.push(positions.map(position => position.planes));
    onAsk(positions);
    return positions.map(({ planes, mask: open }) => {
      assert.ok(answers[planes], `nothing answers ${planes}`);
      return policyWeights(logits(answers[planes]), open);
    });
  };
  return { requests, ask };
}

/** One decision's move asked alone, as an opponent or an adviser asks it,
 * and the requests it made. */
async function alone(decision, answers) {
  const { requests, ask } = asker(answers);
  const move = await trainedMove(async (planes, open) => (await ask([{ planes, mask: open }]))[0], decision);
  return { move, requests };
}

/** A refusal: the one message, and the rule, its codes and the reach's tile
 * in its cause. */
const refusal = cause => error => {
  assert.equal(error.message, RULE_REFUSED);
  assert.deepEqual(error.cause, cause);
  return true;
};

test('the codes, the moves and the rules\' names are numbered and named as the engine and Python have them', () => {
  assert.deepEqual(DAMA, { ABSENT: 0, KEEPS_FIRST: 1, NO_YAKU: 2, SHORT: 3, DEAD: 4, NOT_WAITING: 5 });
  assert.deepEqual(DECLINE, { ABSENT: 0, LAST: 1, NOT_LAST: 2 });
  assert.ok([DAMA, DECLINE, RULE_WORDS].every(Object.isFrozen));
  assert.deepEqual([MORTAL_REACH, MORTAL_AGARI, MORTAL_PASS], [37, 43, 45]);
  assert.deepEqual([DAMA_RULE, DECLINE_RULE], ['all-last-dama', 'decline-last']);
});

test('the best open move is the first of moves weighed alike, a struck move left out, and a lone legal move is forced', () => {
  const weights = [0.1, 0.4, 0.4, 0.05, 0.05];
  assert.equal(bestOpen(weights, [1, 1, 1, 1, 1]), 1);
  assert.equal(bestOpen(weights, [1, 0, 1, 1, 1]), 2);
  assert.equal(bestOpen(weights, [1, 1, 1, 1, 1], 1), 2);
  assert.equal(bestOpen(weights, [0, 1, 0, 0, 0], 1), -1, 'nothing left open');
  assert.equal(bestOpen([0, 0, 0], [0, 1, 1]), 1, 'an open move weighed nought is still a move');
  assert.deepEqual(forcedAnswer(mask([7])), { action: 7, weights: Array.from(mask([7])) });
  assert.equal(forcedAnswer(mask([7, 9])), null);
});

test('each table reads its first question as it is built, and the rest only when asked, each of its own engine', () => {
  /** An engine whose every method answers its own name, each call noted. */
  const recorder = calls => new Proxy({}, { get: (_, name) => (...args) => { calls.push([name, ...args]); return name; } });
  const names = seat => [`${seat}_observation_mortal`, `${seat}_mask_mortal`, `${seat}_observation_mortal_renamed`,
    `${seat}_observation_after_reach`, `${seat}_mask_after_reach`, `${seat}_observation_after_reach_renamed`,
    `${seat}_all_last_dama`, `${seat}_all_last_decline`];
  for (const [build, asked, args] of [[opponentTable, names('opponent'), []], [agentTable, names('agent'), []],
    [engine => reviewTable(engine, 7), names('review'), [7]]]) {
    const calls = [];
    const built = build(recorder(calls));
    assert.deepEqual(calls, asked.slice(0, 2).map(name => [name, ...args]), `${asked[0]}: only the first question is read`);
    assert.deepEqual([built.planes, built.mask], asked.slice(0, 2));
    assert.equal(built.renamed(), asked[2]);
    const reach = built.afterReach();
    assert.deepEqual([reach.planes, reach.mask, reach.renamed()], asked.slice(3, 6));
    assert.deepEqual([built.dama(), built.decline()], asked.slice(6));
    assert.deepEqual(calls, asked.map(name => [name, ...args]));
  }
});

test('with no rule to apply, a discard reads no code and asks no reach, and a reach whose codes are nought is declared', async () => {
  const discard = table('discard', { open: [3, 5, 9, 37] });
  let { move, requests } = await alone(discard, { discard: { 9: 6 } });
  assert.deepEqual(move, { first: policyWeights(logits({ 9: 6 }), mask([3, 5, 9, 37])), second: null, moves: [9],
    final: { action: 9, afterReach: false }, ruled: null });
  assert.deepEqual(requests, [['discard']]);
  assert.deepEqual(discard.reads, NO_READS);

  const declare = table('declare', { open: [3, 5, 9, 37], tiles: [5, 9] });
  ({ move, requests } = await alone(declare, { declare: { 37: 6 }, 'declare reach': { 5: 6 } }));
  assert.deepEqual([move.moves, move.final, move.ruled, move.second.action], [[MORTAL_REACH, 5], { action: 5, afterReach: true }, null, 5]);
  assert.deepEqual(requests, [['declare'], ['declare reach']]);
  assert.deepEqual(declare.reads, { ...NO_READS, afterReach: 1, dama: 1 });
});

test('dama throws the tile the reach would throw, read from the reach\'s own answer, not the best other move', async () => {
  // The reach is best, its answer throws the 5, and the 3 keeps first too;
  // the best discard of the first answer is a third tile, the 9. Thrown
  // undeclared, the move is the plain discard of the 5, and translated as
  // one: no riichi.
  const dama = { 3: KEEPS_FIRST, 5: KEEPS_FIRST, 9: SHORT, 20: NOT_WAITING };
  const thrown = table('thrown', { open: [3, 5, 9, 20, 37], tiles: [3, 5, 9], dama });
  const { move, requests } = await alone(thrown, { thrown: { 37: 8, 9: 5 }, 'thrown reach': { 5: 6, 3: 2 } });
  assert.equal(bestOpen(move.first.weights, thrown.mask, MORTAL_REACH), 9, 'the best other move is another tile');
  assert.deepEqual(move.moves, [5]);
  assert.deepEqual(move.final, { action: 5, afterReach: false });
  assert.deepEqual(move.ruled, { rules: [DAMA_RULE], dama: Array.from(codes(dama)) });
  assert.equal(move.second.action, 5, 'the reach\'s answer is kept, its tile weights with it');
  assert.deepEqual(requests, [['thrown'], ['thrown reach']]);
  // Where the reach's answer throws the 9, which is short, the riichi is
  // declared, though the 3 and the 5 would keep first.
  const declared = table('declared', { open: [3, 5, 9, 20, 37], tiles: [3, 5, 9], dama });
  const { move: reach } = await alone(declared, { declared: { 37: 8, 9: 5 }, 'declared reach': { 9: 6 } });
  assert.deepEqual([reach.moves, reach.final, reach.ruled], [[MORTAL_REACH, 9], { action: 9, afterReach: true }, null]);
});

test('codes 2, 3 and 4 at the reach\'s tile declare it; nought there among others, 5 there, or a tile that keeps first and cannot be thrown are refused', async () => {
  for (const code of [NO_YAKU, SHORT, DEAD]) {
    const kept = table('kept', { open: [3, 5, 9, 37], tiles: [5, 9], dama: { 3: NOT_WAITING, 5: code, 9: KEEPS_FIRST } });
    const { move } = await alone(kept, { kept: { 37: 6 }, 'kept reach': { 5: 6 } });
    assert.deepEqual([move.moves, move.final, move.ruled], [[MORTAL_REACH, 5], { action: 5, afterReach: true }, null], `code ${code}`);
  }
  for (const [open, dama] of [
    // Nought at the reach's tile, where other tiles are coded.
    [[3, 5, 9, 37], { 3: NOT_WAITING, 9: KEEPS_FIRST }],
    // The reach's tile coded as one whose riichi is not legal.
    [[3, 5, 9, 37], { 3: NOT_WAITING, 5: NOT_WAITING, 9: KEEPS_FIRST }],
    // It keeps first, but its plain discard is not open.
    [[3, 9, 37], { 5: KEEPS_FIRST, 9: SHORT }],
  ]) {
    const refused = table('refused', { open, tiles: [5, 9], dama });
    await assert.rejects(alone(refused, { refused: { 37: 6 }, 'refused reach': { 5: 6 } }),
      refusal({ rule: DAMA_RULE, dama: Array.from(codes(dama)), tile: 5 }));
  }
});

test('a win that would leave it last is passed in a claim window, whatever ranks second: a pon, a chi, or nothing at a robbed quad', async () => {
  for (const [name, open, second] of [['pon', [41, 43, 45], 41], ['chi', [38, 43, 45], 38], ['robbed quad', [43, 45], null]]) {
    const claim = table(name, { open, decline: LAST });
    const { move } = await alone(claim, { [name]: second === null ? { 43: 8 } : { 43: 8, [second]: 5 } });
    if (second !== null) assert.equal(bestOpen(move.first.weights, claim.mask, MORTAL_AGARI), second, `${name}: the call ranks second`);
    assert.deepEqual([move.moves, move.final, move.ruled], [[MORTAL_PASS], { action: MORTAL_PASS, afterReach: false },
      { rules: [DECLINE_RULE], dama: null }], name);
    assert.deepEqual(claim.reads, { ...NO_READS, decline: 1 }, name);
  }
});

test('a self-draw that would leave it last gives way to the best other move of the same answer: the first of two alike, a quad, the mean\'s', async () => {
  for (const [open, named, best] of [
    [[3, 9, 43], { 43: 8, 9: 3, 3: 1 }, 9],
    [[3, 9, 43], { 43: 8, 3: 2, 9: 2 }, 3],
    [[3, 42, 43], { 43: 8, 42: 3, 3: 1 }, 42],
  ]) {
    const { move } = await alone(table('turn', { open, decline: LAST }), { turn: named });
    assert.deepEqual([move.moves, move.final, move.ruled], [[best], { action: best, afterReach: false }, { rules: [DECLINE_RULE], dama: null }]);
  }
  // A hesitant answer, the win best in the mean too. Struck, the plain
  // answer would throw 1m; the mean, which the renamed game moves, throws 1p.
  const plain = { 43: 2, 0: 0.5 }, renamed = { 43: 2, 0: 1.5, 9: -1 };
  assert.equal(bestOpen(policyWeights(logits(plain), mask([0, 9, 43])).weights, mask([0, 9, 43]), MORTAL_AGARI), 0);
  const mean = table('mean', { open: [0, 9, 43], decline: LAST });
  const { move, requests } = await alone(mean, { mean: plain, 'mean renamed': renamed });
  assert.deepEqual(requests, [['mean'], ['mean renamed']]);
  assert.deepEqual(move.moves, [9]);
});

test('codes 0 and 2 leave the win; any other code, or a declined self-draw with no other move, is refused', async () => {
  for (const code of [DECLINE.ABSENT, NOT_LAST]) {
    for (const open of [[3, 9, 43], [41, 43, 45]]) {
      const win = table('win', { open, decline: code });
      const { move } = await alone(win, { win: { 43: 8 } });
      assert.deepEqual([move.moves, move.final, move.ruled], [[MORTAL_AGARI], { action: MORTAL_AGARI, afterReach: false }, null]);
      assert.deepEqual(win.reads, { ...NO_READS, decline: 1 });
    }
  }
  await assert.rejects(alone(table('three', { open: [3, 9, 43], decline: 3 }), { three: { 43: 8 } }),
    refusal({ rule: DECLINE_RULE, decline: 3 }));
  await assert.rejects(alone(table('only', { open: [43], decline: LAST }), { only: { 43: 1 } }),
    refusal({ rule: DECLINE_RULE, decline: LAST }));
});

test('the gate is read once, on the plain answer, before the win is struck: nothing is asked again of the moves left', async () => {
  // A win at 0.95 is sure, so the renamed game is never asked; the moves
  // after it, a riichi at 0.03 and the 3m at 0.02, would hesitate on their
  // own, and in the renamed game's mean the 3m would rank above the riichi.
  const open = mask([2, 37, 43]);
  const plain = { 43: Math.log(0.95), 37: Math.log(0.03), 2: Math.log(0.02) };
  const renamed = { 43: Math.log(0.95), 37: Math.log(0.01), 11: Math.log(0.04) };
  const plainWeights = policyWeights(logits(plain), open).weights;
  assert.equal(hesitates(plainWeights), false);
  assert.equal(hesitates([plainWeights[37], plainWeights[2]].map(weight => weight / (plainWeights[37] + plainWeights[2]))), true);
  const mean = average(plainWeights, policyWeights(logits(renamed), swapMoves(open)).weights, open);
  assert.equal(bestOpen(mean.weights, open, MORTAL_AGARI), 2, 'the mean would throw the 3m');
  const sure = table('sure', { open: [2, 37, 43], tiles: [2], decline: LAST });
  const { move, requests } = await alone(sure, { sure: plain, 'sure renamed': renamed, 'sure reach': { 2: 0 } });
  assert.deepEqual(requests, [['sure'], ['sure reach']], 'one request, then the reach\'s tile');
  assert.equal(sure.reads.renamed, 0);
  assert.deepEqual([move.moves, move.ruled], [[MORTAL_REACH, 2], { rules: [DECLINE_RULE], dama: null }]);
});

test('decline-last comes first, then dama on the reach a declined self-draw left: thrown undeclared where its tile keeps first', async () => {
  for (const [dama, moves, rules] of [
    [{ 3: NOT_WAITING, 5: KEEPS_FIRST, 9: SHORT }, [5], [DECLINE_RULE, DAMA_RULE]],
    [{ 3: NOT_WAITING, 5: SHORT, 9: KEEPS_FIRST }, [MORTAL_REACH, 5], [DECLINE_RULE]],
  ]) {
    const chain = table('chain', { open: [3, 5, 9, 37, 43], tiles: [5, 9], dama, decline: LAST });
    const { move, requests } = await alone(chain, { chain: { 43: 9, 37: 6, 9: 2 }, 'chain reach': { 5: 3 } });
    assert.deepEqual(requests, [['chain'], ['chain reach']]);
    assert.deepEqual(move.moves, moves);
    assert.deepEqual(move.final, { action: 5, afterReach: moves.length === 2 });
    assert.deepEqual(move.ruled, { rules, dama: rules.includes(DAMA_RULE) ? Array.from(codes(dama)) : null });
    assert.deepEqual(chain.reads, { ...NO_READS, afterReach: 1, dama: 1, decline: 1 });
  }
});

test('the rules act on the mean of a hesitant answer and its renamed game\'s: a win or a riichi the mean makes best, a tile the mean throws', async () => {
  // A pon the plain answer prefers, a win the mean does.
  for (const [decline, moves] of [[LAST, [MORTAL_PASS]], [DECLINE.ABSENT, [MORTAL_AGARI]]]) {
    const claim = table('claim', { open: [41, 43, 45], decline });
    const { move, requests } = await alone(claim, { claim: { 41: 1, 43: 0.8 }, 'claim renamed': { 43: 2 } });
    assert.deepEqual([requests, move.moves], [[['claim'], ['claim renamed']], moves]);
  }
  // A discard of 3s the plain answer prefers, a riichi the mean does; its
  // tile, 1p, keeps first.
  const turn = table('turn', { open: [0, 9, 20, 37], tiles: [0, 9], dama: { 0: SHORT, 9: KEEPS_FIRST, 20: NOT_WAITING } });
  let { move, requests } = await alone(turn, { turn: { 20: 1, 37: 0.8 }, 'turn renamed': { 37: 2 }, 'turn reach': { 9: 4 } });
  assert.deepEqual([requests, move.moves, move.ruled.rules], [[['turn'], ['turn renamed'], ['turn reach']], [9], [DAMA_RULE]]);
  // A hesitant reach: the plain answer throws 1m, which keeps first; the
  // mean throws 1p, which is short, and the riichi is declared.
  const tile = table('tile', { open: [0, 9, 37], tiles: [0, 9], dama: { 0: KEEPS_FIRST, 9: SHORT } });
  ({ move, requests } = await alone(tile, { tile: { 37: 6 }, 'tile reach': { 0: 0.4 }, 'tile reach renamed': { 0: 1.5 } }));
  assert.deepEqual(requests, [['tile'], ['tile reach'], ['tile reach renamed']]);
  assert.deepEqual([move.moves, move.final, move.ruled], [[MORTAL_REACH, 9], { action: 9, afterReach: true }, null]);
});

test('codes are read only where a rule may change the move, each once, and the reach\'s tile only where the move is a riichi', async () => {
  const decisions = [
    table('discard', { open: [3, 9] }),
    table('kept win', { open: [3, 9, 43], decline: NOT_LAST }),
    table('declare', { open: [3, 5, 37], tiles: [5], dama: { 3: NOT_WAITING, 5: SHORT } }),
    table('declined', { open: [3, 5, 37, 43], tiles: [5], dama: { 3: NOT_WAITING, 5: SHORT }, decline: LAST }),
    table('hesitant', { open: [0, 9] }),
  ];
  const { requests, ask } = asker({ discard: { 9: 5 }, 'kept win': { 43: 8 }, declare: { 37: 6 }, 'declare reach': { 5: 1 },
    declined: { 43: 9, 37: 6 }, 'declined reach': { 5: 1 }, hesitant: { 0: 0.5 }, 'hesitant renamed': {} });
  const moves = await trainedMoves(ask, decisions);
  assert.deepEqual(moves.map(move => move.moves), [[9], [MORTAL_AGARI], [MORTAL_REACH, 5], [MORTAL_REACH, 5], [0]]);
  assert.deepEqual(decisions.map(decision => decision.reads), [NO_READS, { ...NO_READS, decline: 1 },
    { ...NO_READS, afterReach: 1, dama: 1 }, { ...NO_READS, afterReach: 1, dama: 1, decline: 1 }, { ...NO_READS, renamed: 1 }]);
  assert.deepEqual(requests, [['discard', 'kept win', 'declare', 'declined', 'hesitant'], ['hesitant renamed'],
    ['declare reach', 'declined reach']]);
});

test('with skipForced a question with one legal move is not asked: neither a lone move nor the tile of a reach that has one', async () => {
  const dama = { 3: NOT_WAITING, 5: KEEPS_FIRST };
  const answers = { lone: {}, 'one tile': { 37: 6 }, 'one tile reach': { 5: 0 } };
  const lone = table('lone', { open: [7] }), one = table('one tile', { open: [3, 5, 37], tiles: [5], dama });
  const skipped = asker(answers);
  const [forced, thrown] = await trainedMoves(skipped.ask, [lone, one], { skipForced: true });
  assert.deepEqual(skipped.requests, [['one tile']]);
  assert.deepEqual(forced, { first: forcedAnswer(mask([7])), second: null, moves: [7], final: { action: 7, afterReach: false }, ruled: null });
  assert.deepEqual([thrown.moves, thrown.second, thrown.ruled], [[5], forcedAnswer(mask([5])), { rules: [DAMA_RULE], dama: Array.from(codes(dama)) }]);
  assert.deepEqual(one.reads, { ...NO_READS, afterReach: 1, dama: 1 });
  // Without it, both are asked, and play alike.
  const asked = asker(answers);
  const both = await trainedMoves(asked.ask, [table('lone', { open: [7] }), table('one tile', { open: [3, 5, 37], tiles: [5], dama })]);
  assert.deepEqual(asked.requests, [['lone', 'one tile'], ['one tile reach']]);
  assert.deepEqual(both.map(move => move.moves), [[7], [5]]);
});

test('a batch asks the first questions, those renamed, the reaches\' tiles and those renamed, and answers each table as alone', async () => {
  const answers = {
    sure: { 9: 5 },
    hesitant: { 0: 0.5 }, 'hesitant renamed': { 0: 0.2 },
    declare: { 37: 6 }, 'declare reach': { 5: 0.3 }, 'declare reach renamed': { 0: 1 },
    declined: { 43: 9, 37: 6 }, 'declined reach': { 5: 1 },
    win: { 43: 8 },
  };
  const decisions = () => [
    table('sure', { open: [3, 9] }),
    table('hesitant', { open: [0, 9] }),
    table('declare', { open: [3, 5, 9, 37], tiles: [5, 9] }),
    table('declined', { open: [3, 5, 37, 43], tiles: [5], dama: { 3: NOT_WAITING, 5: SHORT }, decline: LAST }),
    table('win', { open: [3, 9, 43], decline: NOT_LAST }),
  ];
  const { requests, ask } = asker(answers);
  const batch = await trainedMoves(ask, decisions());
  assert.deepEqual(requests, [['sure', 'hesitant', 'declare', 'declined', 'win'], ['hesitant renamed'],
    ['declare reach', 'declined reach'], ['declare reach renamed']]);
  assert.deepEqual(batch.map(move => move.moves), [[9], [0], [MORTAL_REACH, 9], [MORTAL_REACH, 5], [MORTAL_AGARI]]);
  for (const [row, decision] of decisions().entries()) assert.deepEqual(batch[row], (await alone(decision, answers)).move, decision.planes);
});

test('stopped while the network is asked, it throws AbortError and reads nothing more of the table', async () => {
  // Stopped during the first question: its answer is a win, whose code is
  // not read.
  let stop = new AbortController();
  const win = table('win', { open: [3, 9, 43], decline: LAST });
  let { ask } = asker({ win: { 43: 8 } }, () => stop.abort());
  await assert.rejects(trainedMoves(ask, [win], { signal: stop.signal }), { name: 'AbortError' });
  assert.deepEqual(win.reads, NO_READS);
  // Stopped during the reach's tile, sure or hesitant: its codes are not
  // read, nor its renamed game.
  for (const tile of [{ 5: 6 }, { 5: 0.3 }]) {
    stop = new AbortController();
    const reach = table('reach', { open: [3, 5, 9, 37], tiles: [5, 9], dama: { 5: KEEPS_FIRST, 9: SHORT } });
    ({ ask } = asker({ reach: { 37: 6 }, 'reach reach': tile }, positions => { if (positions[0].planes === 'reach reach') stop.abort(); }));
    const signals = [];
    const analyze = async (planes, open, signal) => { signals.push(signal); return (await ask([{ planes, mask: open }]))[0]; };
    await assert.rejects(trainedMove(analyze, reach, stop.signal), { name: 'AbortError' });
    assert.deepEqual(reach.reads, { ...NO_READS, afterReach: 1 });
    assert.deepEqual(signals, [stop.signal, stop.signal], 'each question is asked with the signal');
  }
});

test('ruleDecides holds for a riichi any of whose tiles keeps first and a win that leaves it last, reading only the code its move needs', () => {
  for (const [action, dama, decline, decides, reads] of [
    [MORTAL_REACH, { 3: NOT_WAITING, 5: KEEPS_FIRST, 9: SHORT }, 0, true, { dama: 1, decline: 0 }],
    [MORTAL_REACH, { 3: NOT_WAITING, 5: NO_YAKU, 9: SHORT, 12: DEAD }, 0, false, { dama: 1, decline: 0 }],
    [MORTAL_REACH, {}, LAST, false, { dama: 1, decline: 0 }],
    [MORTAL_AGARI, { 5: KEEPS_FIRST }, LAST, true, { dama: 0, decline: 1 }],
    [MORTAL_AGARI, {}, NOT_LAST, false, { dama: 0, decline: 1 }],
    [MORTAL_AGARI, {}, DECLINE.ABSENT, false, { dama: 0, decline: 1 }],
    [9, { 9: KEEPS_FIRST }, LAST, false, { dama: 0, decline: 0 }],
    [MORTAL_PASS, { 9: KEEPS_FIRST }, LAST, false, { dama: 0, decline: 0 }],
  ]) {
    const read = { dama: 0, decline: 0 };
    const engine = {
      agent_all_last_dama: () => { read.dama++; return codes(dama); },
      agent_all_last_decline: () => { read.decline++; return decline; },
    };
    assert.equal(ruleDecides(action, engine), decides, `move ${action}`);
    assert.deepEqual(read, reads, `move ${action}`);
  }
});

test('a ruled move\'s words: each rule\'s line, both with decline\'s first, the tag, and what a screen reader hears', () => {
  const dama = 'Last hand: every win keeps it first, so no riichi', decline = 'Last hand: this win would leave it last';
  assert.equal(RULE_WORDS.tag, 'by rule');
  assert.equal(ruleLine({ rules: [DAMA_RULE] }), dama);
  assert.equal(ruleLine({ rules: [DECLINE_RULE] }), decline);
  assert.equal(ruleLine({ rules: [DECLINE_RULE, DAMA_RULE] }), `${decline} · ${dama}`);
  assert.equal(ruleLine({ rules: [DAMA_RULE, DECLINE_RULE] }), `${decline} · ${dama}`, 'decline\'s line first');
  assert.equal(ruleSaid({ rules: [DAMA_RULE] }), ', by rule (last hand: every win keeps it first, so no riichi)');
  assert.equal(ruleSaid({ rules: [DECLINE_RULE] }), ', by rule (last hand: this win would leave it last)');
  assert.equal(ruleSaid({ rules: [DECLINE_RULE, DAMA_RULE] }),
    ', by rule (last hand: this win would leave it last · Last hand: every win keeps it first, so no riichi)');
  assert.equal(RULE_REFUSED, 'A last-hand rule could not be applied');
});
