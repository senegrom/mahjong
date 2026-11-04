import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { PhysicalAnalysis } from '../src/wasm/riichi.js';
import { emptyPosition, parseTiles } from '../src/lib/physical-position.js';
import { emptyGuided, editGuided, guidedEvent } from '../src/lib/guided-game.js';
import { policyWeights } from '../src/lib/policy-weights.js';
import { readBeliefs, readValue } from '../src/lib/beliefs.js';
import { weightsByChoice } from '../src/lib/action-weights.js';
import { DAMA, DAMA_RULE, DECLINE_RULE, agentTable, trainedMove } from '../src/lib/trained-move.js';
import { loadModule } from './fixtures/load-module.js';

// agents.js loads the policy client, whose offline storage resolves the
// page's folder as it loads. Nothing here constructs a worker or downloads
// the trained model: Trained is asked through adviserWith below.
globalThis.document = { baseURI: 'https://example.invalid/mahjong/' };
const { evaluateAgent } = await import('../src/lib/agents.js');
await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

/** Mortal's pass, the last of its forty-six moves. */
const MORTAL_PASS = 45;

/** The adviser's own code with the network stood in for. The engine's
 * questions and the worker's softmax over the legal moves are the real ones;
 * only the network's raw preferences are made up, the same in either naming,
 * so a question it hesitates over, asked again of the renamed game, is
 * answered by the same preferences over that game's moves. */
function adviserWith(logits) {
  const analyzePolicy = async (planes, mask) => {
    assert.ok(planes.length / 34 > 900, 'the network is asked about Mortal planes');
    // What policy.worker.js answers with, by the same arithmetic.
    return { ...policyWeights(logits, mask), value: 0.25, hands: new Float32Array(102) };
  };
  return loadModule('agents.js', { analyzePolicy, agentTable, trainedMove, readBeliefs, readValue, weightsByChoice },
    ['evaluateAgent']).evaluateAgent;
}
/** Raw preferences over Mortal's moves: those named, nought elsewhere. */
const logitsOf = named => Float32Array.from({ length: 46 }, (_, move) => named[move] ?? 0);

test('a physical position builds the trained observation and keeps its legal choices', async () => {
  const position = emptyPosition();
  position.players[0].hand = parseTiles('123m456p789s11234z');
  position.drawn = '4z'; position.indicators = ['5z'];
  const original = structuredClone(position), engine = new PhysicalAnalysis(position);
  try {
    const choices = engine.agent_choices();
    // The observation the network would be asked, built from the replay:
    // Mortal's planes over 34 tiles, and a mask over Mortal's 46 moves.
    const planes = engine.agent_observation_mortal();
    assert.equal(planes.length % 34, 0);
    assert.ok(planes.length / 34 > 900, 'not Mortal planes');
    assert.equal(engine.agent_mask_mortal().length, 46);
    assert.deepEqual(engine.agent_choices(), choices);
    for (const agent of ['beginner', 'club']) {
      const answer = await evaluateAgent(engine, agent);
      assert.equal(answer.agent, agent);
      assert.equal(answer.kind, 'selection');
      assert.ok(choices.some(choice => choice.kind === answer.choice.kind && choice.tile === answer.choice.tile));
    }
    assert.deepEqual(position, original);
  } finally { engine.free(); }
});

const thrown = (tile, order, claimed = false) => ({ tile, order, drawn: false, riichi: false, claimed });
/** East to act on its draw after two rounds nobody called, with no hand but
 * East's typed in. */
function twoRounds() {
  const position = emptyPosition();
  position.players[0].hand = parseTiles('123m456p789s11235z');
  position.drawn = '5z'; position.indicators = ['3s']; position.wall = 61; position.first_turns = false;
  [['9m', '8m'], ['9p', '8p'], ['9s', '8s'], ['4z', '6z']].forEach((river, seat) => river.forEach((tile, turn) =>
    position.players[seat].discards.push(thrown(tile, 4 * turn + seat))));
  return position;
}
async function asking(position, ask) {
  const engine = new PhysicalAnalysis(position);
  try { return await ask(engine); } finally { engine.free(); }
}

test('trained advice refuses a typed-in set whose claimed discard is not marked', async () => {
  // The network's replay tells a called set at the discard it took. West
  // shows a pon of red dragons taken from North, but nothing in North's river
  // is marked Claimed, so the network would read a table without the pon.
  const trained = adviserWith(new Float32Array(46));
  const unmarked = twoRounds();
  unmarked.players[2].melds.push({ kind: 'pon', tile: '7z', from: 1 });
  await asking(unmarked, async engine => {
    await assert.rejects(trained(engine, 'full'),
      /^West's pon of 7z needs the discard it took marked Claimed in North's river$/);
    // Club and Beginner read the sets themselves, so they still advise.
    assert.equal((await evaluateAgent(engine, 'club')).kind, 'selection');
  });
  // Marked: North's second discard was the red dragon West called, and West
  // discarded after calling it.
  const marked = structuredClone(unmarked);
  marked.players[3].discards[1] = thrown('7z', 7, true);
  marked.players[2].discards.push(thrown('1p', 8));
  marked.players[3].discards.push(thrown('2p', 9));
  marked.wall = 60;
  await asking(marked, async engine => assert.equal((await trained(engine, 'full')).kind, 'policy'));
  // An open quad is refused the same way; the indicator it turned would have
  // been left out with it.
  const quad = twoRounds();
  quad.players[2].melds.push({ kind: 'kan', tile: '7z', from: 1 });
  quad.indicators.push('9p'); quad.wall = 60;
  await asking(quad, engine => assert.rejects(trained(engine, 'full'),
    /^West's open kan of 7z needs the discard it took marked Claimed in North's river$/));
  // A chii is named by all three of its tiles.
  const chii = twoRounds();
  chii.players[1].melds.push({ kind: 'chii', tile: '1m', from: 3 });
  await asking(chii, engine => assert.rejects(trained(engine, 'full'),
    /^South's chii of 123m needs the discard it took marked Claimed in East's river$/));
});

test('trained advice answers an opponent discard the player cannot claim', async () => {
  // North, after East lets go of a tile North can do nothing with. The guided
  // game stops at every discard, and its adviser analyses each one at once.
  const validate = position => new PhysicalAnalysis(position).free();
  let game = emptyGuided();
  game = editGuided(game, state => { state.position.seat = 3; });
  game = guidedEvent(game, { type: 'setup' }, validate);
  game = editGuided(game, state => { state.position.players[3].hand = parseTiles('123m456p789s1123z'); });
  game = guidedEvent(game, { type: 'hand' }, validate);
  game = guidedEvent(game, { type: 'indicator', tile: '7z' }, validate);
  game = guidedEvent(game, { type: 'discard', tile: '9m' }, validate);
  assert.equal(game.state.stage, 'decision');
  const engine = new PhysicalAnalysis(game.state.position);
  try {
    assert.deepEqual(engine.agent_choices().map(choice => choice.kind), ['pass']);
    const open = Array.from(engine.agent_mask_mortal()).flatMap((allowed, action) => allowed ? [action] : []);
    assert.deepEqual(open, [MORTAL_PASS], 'the trained mask offers the pass the page lists');
    // Whatever the network would rather do, a pass is all there is.
    const logits = Float32Array.from({ length: 46 }, (_, action) => action === 0 ? 9 : 0);
    const answer = await adviserWith(logits)(engine, 'full');
    assert.equal(answer.kind, 'policy');
    assert.equal(answer.choice.kind, 'pass');
    assert.deepEqual(answer.choices.map(choice => [choice.kind, choice.weight]), [['pass', 1]]);
    // Nobody else's hand was typed in: hidden, not empty. East has just let
    // its tile go, and nobody has called, so each holds thirteen.
    assert.deepEqual(answer.beliefs.map(belief => belief.held), [13, 13, 13]);
    assert.ok(answer.beliefs.every(belief => Math.abs(belief.expected.reduce((a, b) => a + b, 0) - 13) < 1e-9));
  } finally { engine.free(); }
});

// The last hand's tables of riichi_core::all_last's tests, typed in: South 4,
// thirty tiles left, the south wind the indicator, nothing thrown before, and
// nobody's tiles but the deciding seat's.
function southFour({ seat = 3, hand, drawn = null, scores }) {
  const position = emptyPosition();
  Object.assign(position, { seat, turn: seat, round: 1, kyoku: 4, wall: 30, indicators: ['2z'], drawn, first_turns: false });
  position.players.forEach((player, at) => { player.score = scores[at]; });
  position.players[seat].hand = parseTiles(hand);
  return position;
}
/** North to move on 23456m 456p 678s 33s with the 9p drawn, the 9p its one
 * riichi tile; East on 20,000, South on 36,000 + `gap`, West on 28,000 -
 * `gap` and North on 36,000. Thrown undeclared, the 9p keeps North first
 * through every win while `gap` is under 1,000. */
const probe = gap => southFour({ hand: '23456m456p678s33s9p', drawn: '9p', scores: [20000, 36000 + gap, 28000 - gap, 36000] });
/** North to move on 123456m 456p 678s 33s with the 1m drawn, which wins:
 * pinfu and menzen tsumo, which leaves North alone in last while West holds
 * 12,000 or more. Every tile is a riichi tile, and short. */
const tsumo = west => southFour({ hand: '123456m456p678s33s', drawn: '1m', scores: [52000 - west, 38000, west, 10000] });
/** West has just thrown the 1m, which North, on 23456m 456p 678s 33s, may
 * chii or ron: pinfu, which leaves North alone in last while West holds more
 * than 12,000. */
function ron(west) {
  const position = southFour({ hand: '23456m456p678s33s', scores: [52000 - west, 38000, west, 10000] });
  Object.assign(position, { turn: 2, phase: 'call', pending: '1m' });
  position.players[2].discards.push(thrown('1m', 0));
  return position;
}
const advised = answer => [answer.choice.kind, answer.choice.tile ?? null, answer.ruled?.rules ?? null];

test('Trained advice in the last hand throws undeclared a riichi whose tile keeps it first, and declares one whose tile does not', async () => {
  const advise = adviserWith(logitsOf({ 37: 8 }));
  await asking(probe(0), async engine => assert.deepEqual(advised(await advise(engine, 'full')), ['discard', '9p', [DAMA_RULE]]));
  await asking(probe(1500), async engine => assert.deepEqual(advised(await advise(engine, 'full')), ['riichi', '9p', null]));
});

test('Trained advice in the last hand declines a self-draw that leaves it last for its best other move, a riichi judged as any', async () => {
  // The 3s second, a discard.
  await asking(tsumo(12000), async engine => assert.deepEqual(advised(await adviserWith(logitsOf({ 43: 12, 20: 9, 37: 5 }))(engine, 'full')),
    ['discard', '3s', [DECLINE_RULE]]));
  // Riichi second: its tile, the 3s, is asked of a preview of the reach, and
  // judged, short, so the riichi stands.
  await asking(tsumo(12000), async engine => {
    const reads = { dama: 0, decline: 0 };
    const counted = Object.assign(Object.create(engine), {
      agent_all_last_dama: () => { reads.dama++; return engine.agent_all_last_dama(); },
      agent_all_last_decline: () => { reads.decline++; return engine.agent_all_last_decline(); },
    });
    const answer = await adviserWith(logitsOf({ 43: 12, 37: 9, 20: 5 }))(counted, 'full');
    assert.deepEqual(advised(answer), ['riichi', '3s', [DECLINE_RULE]]);
    assert.deepEqual(reads, { dama: 1, decline: 1 });
    assert.equal(engine.agent_all_last_dama()[20], DAMA.SHORT);
    assert.ok(answer.choices.some(choice => choice.kind === 'riichi' && choice.conditionalWeight > 0), 'its tiles are weighed');
  });
  // West on 11,900: the tsumo would leave North level, and stands.
  await asking(tsumo(11900), async engine => assert.deepEqual(advised(await adviserWith(logitsOf({ 43: 12, 20: 9 }))(engine, 'full')),
    ['tsumo', null, null]));
});

test('Trained advice in the last hand passes a ron that leaves it last, though a chii ranks second', async () => {
  const advise = adviserWith(logitsOf({ 43: 12, 38: 9 }));
  await asking(ron(12100), async engine => assert.deepEqual(advised(await advise(engine, 'full')), ['pass', null, [DECLINE_RULE]]));
  await asking(ron(12000), async engine => assert.deepEqual(advised(await advise(engine, 'full')), ['ron', null, null]));
});

test('Trained advice in the last hand asks no rule of the dealer, and reads a furiten tick as play leaves the hand', async () => {
  const advise = adviserWith(logitsOf({ 37: 8 }));
  // The probe with the seats turned one place: East, the dealer, decides.
  const dealer = southFour({ seat: 0, hand: '23456m456p678s33s9p', drawn: '9p', scores: [36000, 20000, 36000, 28000] });
  await asking(dealer, async engine => assert.deepEqual(advised(await advise(engine, 'full')), ['riichi', '9p', null]));
  // North ticked furiten, which play clears at the draw: the same move.
  const ticked = probe(0);
  ticked.players[3].furiten = true;
  await asking(ticked, async engine => assert.deepEqual(advised(await advise(engine, 'full')), ['discard', '9p', [DAMA_RULE]]));
});
