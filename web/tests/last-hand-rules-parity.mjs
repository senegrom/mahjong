// The last hand's rules as the page plays them (src/lib/trained-move.js), held
// to the Python wrappers they are written from. Each case of
// neural/tests/export_rules_fixture.py was played there by
// neural/late_rules.py's decline-last around neural/suits.py's average, and
// neural/all_last.py's dama around that, through zoo.choose_in_mortal_space
// with the reach's tile previewed, greedy, with skip_forced off, as the page's
// opponents, adviser and hand's notes ask, and on, as the game's review asks.
// No wasm and no network: a question is answered with the logits Python was
// given, made weights as the worker makes them (policyWeights).
//
// 1. The layer: trainedMoves over the cases in groups of eight, and each case
//    alone, play the moves Python plays, with the rules that fired, and refuse
//    where Python refuses; each asks the renamed game where the plain answer
//    hesitates, and only there, and the reach's tile only where the move is a
//    riichi.
// 2. Play's opponents (session.js), skip_forced off: a match whose Trained
//    opponent owes the case records the moves Python plays, a riichi and its
//    tile back to back, asking what the layer asks, and records nothing where
//    Python refuses.
// 3. The adviser (agents.js), skip_forced off: the move advised is the one
//    Python plays, translated as a riichi only where it is one, asking what
//    the layer asks, and the advice is refused where Python refuses.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { MORTAL_REACH, weightsByChoice } from '../src/lib/action-weights.js';
import { policyWeights } from '../src/lib/policy-weights.js';
import { MatchSession } from '../src/lib/session.js';
import { GATE, MOVES, swapMoves } from '../src/lib/suits.js';
import {
  DAMA, DAMA_RULE, DECLINE, DECLINE_RULE, MORTAL_AGARI, MORTAL_PASS, RULE_REFUSED, agentTable, trainedMove, trainedMoves,
} from '../src/lib/trained-move.js';
import { loadModule } from './fixtures/load-module.js';

const folder = resolve(process.argv[2]);
const fixture = JSON.parse(await readFile(resolve(folder, 'rules.json'), 'utf8'));
const { cases } = fixture;
/** The settings each case was played with, by skip_forced, as `expected` lists them. */
const SETTINGS = [false, true];
const settingName = skipForced => `skip_forced ${skipForced ? 'on' : 'off'}`;

// The page numbers and names what Python does.
assert.deepEqual(DAMA, fixture.codes.dama);
assert.deepEqual(DECLINE, fixture.codes.decline);
assert.equal(GATE, fixture.gate);
assert.deepEqual(MOVES, fixture.moves);
assert.deepEqual([MORTAL_REACH, MORTAL_AGARI, MORTAL_PASS], [fixture.mortal.reach, fixture.mortal.agari, fixture.mortal.pass]);
assert.deepEqual([DAMA_RULE, DECLINE_RULE], [fixture.rules.dama, fixture.rules.decline]);

/** Each question a case can be asked: its logits, in its own game's move
 * order, and the mask it must be asked with. */
const QUESTIONS = {
  first: { logits: item => item.logits, mask: item => item.mask },
  'first renamed': { logits: item => item.renamed_logits, mask: item => swapMoves(item.mask) },
  preview: { logits: item => item.preview, mask: item => item.after_mask },
  'preview renamed': { logits: item => item.renamed_preview, mask: item => swapMoves(item.after_mask) },
};
assert.deepEqual(Object.keys(QUESTIONS), fixture.questions);

/** Case `at` as the layer reads a decision, each question's planes naming
 * the case and the question. */
function tableOf(at) {
  const item = cases[at];
  return {
    planes: { at, question: 'first' }, mask: Uint8Array.from(item.mask),
    renamed: () => ({ at, question: 'first renamed' }),
    afterReach: () => ({ planes: { at, question: 'preview' }, mask: Uint8Array.from(item.after_mask),
      renamed: () => ({ at, question: 'preview renamed' }) }),
    dama: () => Uint8Array.from(item.dama),
    decline: () => item.decline,
  };
}

/** An asker that answers each question with its case's logits over the mask
 * it is sent, which must be the question's own, as the worker answers; each
 * request is noted by its questions. */
function asker() {
  const requests = [];
  const ask = async positions => {
    requests.push(positions.map(({ planes }) => planes));
    return positions.map(({ planes: { at, question }, mask }) => {
      const item = cases[at];
      assert.deepEqual(Array.from(mask, Number), Array.from(QUESTIONS[question].mask(item), Number), `${item.name}: the ${question} question's mask`);
      return policyWeights(QUESTIONS[question].logits(item), mask);
    });
  };
  return { requests, ask };
}

/** That the layer's answer at case `at` (`asking`) plays as Python played it
 * with `skipForced`: Mortal's moves, the rules that fired, the move as the
 * engine translates it and our move, or the refusal of the same rule. */
async function plays(at, skipForced, asking, how) {
  const item = cases[at], want = item.expected[Number(skipForced)];
  const label = `${item.name}, ${settingName(skipForced)}, ${how}`;
  if (want.refused) {
    await assert.rejects(asking, error => error.message === RULE_REFUSED && error.cause?.rule === want.refused, label);
    return;
  }
  const move = await asking;
  assert.deepEqual(move.moves, want.moves, `${label}: Mortal's moves`);
  assert.deepEqual(move.ruled?.rules ?? [], want.rules, `${label}: the rules that fired`);
  const reach = move.moves.length === 2;
  assert.deepEqual(move.final, { action: move.moves.at(-1), afterReach: reach }, `${label}: the move as the engine translates it`);
  assert.equal((reach ? item.riichi : item.meaning)[move.final.action], want.move, `${label}: our move`);
}

/** How many cases played each kind the generator counts, with `skipForced`,
 * from what they must play. */
function counted(skipForced) {
  const counts = Object.fromEntries(Object.keys(fixture.minimum).map(name => [name, 0]));
  for (const item of cases) {
    const { rules, moves, refused, asks } = item.expected[Number(skipForced)];
    const claim = Boolean(item.mask[MORTAL_PASS]), declined = rules.includes(DECLINE_RULE);
    const reached = refused === DAMA_RULE || rules.includes(DAMA_RULE) || moves?.[0] === MORTAL_REACH;
    counts['dama fired'] += rules.includes(DAMA_RULE);
    counts['decline in a claim window'] += declined && claim;
    counts['decline at a turn'] += declined && !claim;
    counts['decline at a turn, the win above 0.999'] += declined && !claim
      && Math.max(...policyWeights(item.logits, item.mask).weights) > 0.999;
    counts['decline, then dama'] += rules.join() === [DECLINE_RULE, DAMA_RULE].join();
    counts['a reach with one tile'] += reached && item.after_mask.filter(Boolean).length === 1;
    counts.refused += Boolean(refused);
    counts['renamed game asked'] += asks.includes('first renamed') || asks.includes('preview renamed');
  }
  return counts;
}

for (const skipForced of SETTINGS) {
  const setting = Number(skipForced);
  // The fixture exercises every kind of firing, as the generator counted it.
  const counts = counted(skipForced);
  assert.deepEqual(counts, fixture.counts[setting], `${settingName(skipForced)}: the cases' kinds, as the generator counted them`);
  for (const [name, least] of Object.entries(fixture.minimum)) {
    assert.ok(counts[name] >= least, `${settingName(skipForced)}: ${counts[name]} cases of ${name}, fewer than ${least}`);
  }
  // Each case alone, as an opponent, the adviser or the hand's notes asks, or
  // the review a decision at a time: the questions it asks, one a request.
  for (const [at, item] of cases.entries()) {
    const want = item.expected[setting];
    for (const [question, hesitant] of [['first', item.hesitant[0]], ['preview', item.hesitant[1]]]) {
      assert.equal(want.asks.includes(`${question} renamed`), want.asks.includes(question) && hesitant,
        `${item.name}: the ${question} question's renamed game is asked where its plain answer hesitates, and only there`);
    }
    const { requests, ask } = asker();
    const asking = skipForced ? trainedMoves(ask, [tableOf(at)], { skipForced }).then(([move]) => move)
      : trainedMove(async (planes, mask) => (await ask([{ planes, mask }]))[0], tableOf(at));
    await plays(at, skipForced, asking, 'alone');
    assert.deepEqual(requests, want.asks.map(question => [{ at, question }]), `${item.name}, ${settingName(skipForced)}: the questions asked`);
  }
  // In groups of eight, as the game's review asks: the first questions, those
  // renamed, the reaches' tiles and those renamed, a request each, holding
  // the cases that ask them. A refusal stops a whole group, as it stops a
  // whole batch in Python, so the cases Python refuses are asked alone only.
  const played = cases.flatMap((item, at) => (item.expected[setting].refused ? [] : [at]));
  for (let start = 0; start < played.length; start += 8) {
    const group = played.slice(start, start + 8);
    const { requests, ask } = asker();
    const moves = await trainedMoves(ask, group.map(tableOf), { skipForced });
    for (const [slot, at] of group.entries()) await plays(at, skipForced, Promise.resolve(moves[slot]), 'in a group of eight');
    assert.deepEqual(requests, fixture.questions.map(question => group.filter(at => cases[at].expected[setting].asks.includes(question))
      .map(at => ({ at, question }))).filter(request => request.length), `${settingName(skipForced)}: the requests of cases ${group}`);
  }
}

/** A Game, as a match makes one, whose one Trained opponent owes case `at`:
 * its first question, the reach's preview and the codes, as the engine asks
 * them; once a riichi is played, the reach waits for its tile, and asks the
 * preview's question with no code, as the engine's pending reach does. */
const opponentGame = at => class {
  constructor() { this.item = cases[at]; this.reached = false; this.played = false; }
  view() { return { reached: this.reached, played: this.played }; }
  choices() { return []; }
  advance() { return []; }
  game_is_over() { return false; }
  needs_opponent_move() { return !this.played; }
  opponent_observation_mortal() { return { at, question: this.reached ? 'preview' : 'first' }; }
  opponent_observation_mortal_renamed() { return { at, question: this.reached ? 'preview renamed' : 'first renamed' }; }
  opponent_mask_mortal() { return Uint8Array.from(this.reached ? this.item.after_mask : this.item.mask); }
  opponent_observation_after_reach() { return { at, question: 'preview' }; }
  opponent_observation_after_reach_renamed() { return { at, question: 'preview renamed' }; }
  opponent_mask_after_reach() { return Uint8Array.from(this.item.after_mask); }
  opponent_all_last_dama() { return Uint8Array.from(this.reached ? new Uint8Array(34) : this.item.dama); }
  opponent_all_last_decline() { return this.reached ? 0 : this.item.decline; }
  play_opponent_mortal(action) {
    assert.ok(!this.played, `${this.item.name}: a move after the decision`);
    if (action === MORTAL_REACH && !this.reached) this.reached = true;
    else this.played = true;
  }
  free() {}
};

// 2. Each case as Play's opponents ask it, with skip_forced off: the match
// records Mortal's moves, the riichi and its tile together, and nothing where
// Python refuses, which it says as the layer says it.
for (const [at, item] of cases.entries()) {
  const want = item.expected[0], label = `${item.name}, as Play's opponent`;
  const { requests, ask } = asker();
  const match = new MatchSession(opponentGame(at), 0, 'neural', { ai: async (planes, mask) => (await ask([{ planes, mask }]))[0] });
  try {
    const ran = await match.run();
    if (want.refused) assert.deepEqual([ran, match.failure, match.commands], [false, RULE_REFUSED, []], `${label}: refused`);
    else {
      assert.equal(ran, true, `${label}: ${match.failure}`);
      assert.deepEqual(match.commands, want.moves.map(action => ({ type: 'opponent', action })), `${label}: the commands recorded`);
    }
    assert.deepEqual(requests, want.asks.map(question => [{ at, question }]), `${label}: the questions asked`);
  } finally { match.dispose(); }
}

/** Mortal's move for each of our moves, as the cases' maps give them: a
 * riichi on any tile is Mortal's 37, and every other move one move of
 * Mortal's, the same in every case (zoo.first_meaning). */
const MORTAL_OF = new Map();
for (const item of cases) {
  item.meaning.forEach((ours, move) => {
    if (ours < 0) return;
    assert.equal(MORTAL_OF.get(ours) ?? move, move, `our move ${ours} is one of Mortal's`);
    MORTAL_OF.set(ours, move);
  });
  for (const ours of item.riichi.filter(ours => ours >= 0)) {
    assert.equal(MORTAL_OF.get(ours) ?? MORTAL_REACH, MORTAL_REACH, `our move ${ours} is a riichi`);
    MORTAL_OF.set(ours, MORTAL_REACH);
  }
}

/** Case `at` as an engine puts a seat's decision to the adviser (agentTable):
 * its choices our moves, named by their numbers, and Mortal's moves
 * translated to ours by the case's maps. */
function adviserEngine(at) {
  const item = cases[at];
  const choices = [...item.meaning, ...item.riichi].filter(ours => ours >= 0)
    .map(index => ({ index, kind: MORTAL_OF.get(index) === MORTAL_REACH ? 'riichi' : 'move', tile: String(index), label: `move ${index}` }));
  return {
    agent_choices: () => choices,
    agent_observation_mortal: () => ({ at, question: 'first' }),
    agent_mask_mortal: () => Uint8Array.from(item.mask),
    agent_observation_mortal_renamed: () => ({ at, question: 'first renamed' }),
    agent_observation_after_reach: () => ({ at, question: 'preview' }),
    agent_mask_after_reach: () => Uint8Array.from(item.after_mask),
    agent_observation_after_reach_renamed: () => ({ at, question: 'preview renamed' }),
    agent_all_last_dama: () => Uint8Array.from(item.dama),
    agent_all_last_decline: () => item.decline,
    agent_action_from_mortal: (action, afterReach) => (afterReach ? item.riichi : item.meaning)[action] ?? -1,
    mortal_action_of: index => MORTAL_OF.get(index) ?? -1,
  };
}

// 3. Each case as the adviser asks it, with skip_forced off: the move advised
// is the one Python plays, and a refusal is the layer's.
let adviserAsks = null;
const { evaluateAgent } = loadModule('agents.js', { analyzePolicy: (...question) => adviserAsks(...question), agentTable, trainedMove,
  weightsByChoice, readValue: () => null, readBeliefs: () => null }, ['evaluateAgent']);
for (const [at, item] of cases.entries()) {
  const want = item.expected[0], label = `${item.name}, as the adviser`;
  const { requests, ask } = asker();
  adviserAsks = async (planes, mask) => (await ask([{ planes, mask }]))[0];
  const advising = evaluateAgent(adviserEngine(at), 'full');
  if (want.refused) await assert.rejects(advising, error => error.message === RULE_REFUSED && error.cause?.rule === want.refused, `${label}: refused`);
  else {
    const analysis = await advising;
    assert.equal(analysis.choice.index, want.move, `${label}: the move advised`);
    assert.deepEqual(analysis.ruled?.rules ?? [], want.rules, `${label}: the rules that fired`);
  }
  assert.deepEqual(requests, want.asks.map(question => [{ at, question }]), `${label}: the questions asked`);
}

const said = setting => Object.entries(fixture.counts[setting]).map(([name, count]) => `${name} ${count}`).join(', ');
console.log(`PASS: ${cases.length} cases of the last hand's rules, played by the page's layer as the composed Python wrappers play them, `
  + `each alone and in groups of eight, by Play's opponents and by the adviser. skip_forced off: ${said(0)}. skip_forced on: ${said(1)}`);
