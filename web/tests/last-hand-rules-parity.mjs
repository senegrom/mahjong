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
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { MORTAL_REACH } from '../src/lib/action-weights.js';
import { policyWeights } from '../src/lib/policy-weights.js';
import { GATE, MOVES, swapMoves } from '../src/lib/suits.js';
import { DAMA, DAMA_RULE, DECLINE, DECLINE_RULE, MORTAL_AGARI, MORTAL_PASS, RULE_REFUSED, trainedMove, trainedMoves } from '../src/lib/trained-move.js';

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

const said = setting => Object.entries(fixture.counts[setting]).map(([name, count]) => `${name} ${count}`).join(', ');
console.log(`PASS: ${cases.length} cases of the last hand's rules, played by the page's layer as the composed Python wrappers play them, `
  + `each alone and in groups of eight. skip_forced off: ${said(0)}. skip_forced on: ${said(1)}`);
