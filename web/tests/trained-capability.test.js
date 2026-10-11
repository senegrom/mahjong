import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis } from '../src/wasm/riichi.js';
import { emptyPosition, parseTiles } from '../src/lib/physical-position.js';
import { emptyGuided, editGuided, guidedEvent } from '../src/lib/guided-game.js';
import { policyWeights } from '../src/lib/policy-weights.js';
import { readBeliefs, readValue } from '../src/lib/beliefs.js';
import { weightsByChoice, MORTAL_REACH } from '../src/lib/action-weights.js';
import { loadModule } from './fixtures/load-module.js';

// Loading policy.js resolves URLs, but an unsupported adapter must never
// construct a worker or download the trained model.
globalThis.document = { baseURI: 'https://example.invalid/mahjong/' };
const { evaluateAgent, supportsTrainedAgent, TRAINED_HISTORY_REQUIRED } = await import('../src/lib/agents.js');
await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

/** Mortal's pass, the last of its forty-six moves. */
const MORTAL_PASS = 45;

/** The adviser's own code with the network stood in for. The engine's
 * questions and the worker's softmax over the legal moves are the real ones;
 * only the network's raw preferences are made up. */
function adviserWith(logits) {
  const analyzePolicy = async (planes, mask) => {
    assert.ok(planes.length / 34 > 900, 'the network is asked about Mortal planes');
    // What policy.worker.js answers with, by the same arithmetic.
    return { ...policyWeights(logits, mask), value: 0.25, hands: new Float32Array(102) };
  };
  return loadModule('agents.js', { analyzePolicy, readBeliefs, readValue, weightsByChoice, MORTAL_REACH },
    ['evaluateAgent']).evaluateAgent;
}

test('trained capability is met by the live game and by a typed-in position alike', () => {
  // A position typed into the guided or physical table is replayed into
  // the events Mortal's encoder needs (mortal_log.rs), so the position-only
  // engine answers the same six questions the live game does.
  assert.equal(supportsTrainedAgent(Game.prototype), true);
  assert.equal(supportsTrainedAgent(PhysicalAnalysis.prototype), true);
  assert.equal(supportsTrainedAgent(null), false);
  assert.equal(supportsTrainedAgent({}), false);
});

test('trained capability requires both riichi stages and action translation', () => {
  const names = ['agent_observation_mortal', 'agent_mask_mortal',
    'agent_observation_after_reach', 'agent_mask_after_reach',
    'agent_action_from_mortal', 'mortal_action_of'];
  const complete = Object.fromEntries(names.map(name => [name, () => {}]));
  assert.equal(supportsTrainedAgent(complete), true);
  for (const name of names) {
    assert.equal(supportsTrainedAgent({ ...complete, [name]: undefined }), false, name);
    assert.equal(supportsTrainedAgent({ ...complete, [name]: true }), false, name);
  }
});

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
    // An engine that cannot answer the six questions is still refused by
    // name rather than being asked half of one.
    await assert.rejects(evaluateAgent({ agent_choices: () => choices }, 'full'), { message: TRAINED_HISTORY_REQUIRED });
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
