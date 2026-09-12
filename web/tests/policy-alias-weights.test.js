import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import init, { Game } from '../src/wasm/riichi.js';
import { weightsByChoice, MORTAL_REACH } from '../src/lib/action-weights.js';
import { readBeliefs, readValue } from '../src/lib/beliefs.js';
import { reviewWithStrong } from '../src/lib/review-policy.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
// Exercise the actual evaluator with only its worker call replaced. The same
// real weight mapper and actual WASM action translation run in both paths.
const source = readFileSync(new URL('../src/lib/agents.js', import.meta.url), 'utf8')
  .replace(/^import .*\n/gm, '').replace(/^export /gm, '');
const evaluator = analyze => vm.runInNewContext(source + '\nevaluateAgent', {
  analyzePolicy: analyze, weightsByChoice, MORTAL_REACH, readBeliefs, readValue,
});

for (const [alias, canonical] of [[34, 4], [35, 13], [36, 22]]) {
  test(`live advice and historical review retain both five-tile actions ${canonical}/${alias}`, async () => {
    let game;
    for (let seed = 1; seed <= 100; seed++) {
      game = new Game(seed, 'club'); game.advance();
      if (game.agent_mask_mortal()[alias]) break;
      game.free(); game = null;
    }
    assert.ok(game, 'bounded fixtures must offer this five discard');
    try {
      const choices = game.agent_choices();
      const other = choices.find(choice => choice.kind === 'discard' && choice.index !== canonical);
      assert.ok(other);
      const weights = Array(46).fill(0);
      weights[alias] = .7; weights[canonical] = .1;
      weights[game.mortal_action_of(other.index)] = .2;
      const analyze = async () => ({ action: alias, weights });
      const result = await evaluator(analyze)(game, 'full');
      assert.equal(result.choice.index, canonical, 'alias selection still translates to the original best move');
      const preferred = result.choices.find(choice => choice.index === canonical);
      assert.ok(Math.abs(preferred.weight - .8) < 1e-12);
      assert.equal(result.choices[0].index, canonical);
      assert.ok(Math.abs(result.choices.reduce((sum, choice) => sum + (choice.weight ?? 0), 0) - 1) < 1e-12);
      game.choose(result.choice.kind, result.choice.tile);
      const review = await reviewWithStrong(game, game.review().slice(0, 1), analyze);
      assert.equal(review[0].agreed, true);
      assert.ok(Math.abs(review[0].preferred_weight - .8) < 1e-12);
      assert.equal(review[0].played_weight, review[0].preferred_weight);
    } finally { game.free(); }
  });
}

test('reach remains a declaration weight and unnameable extra kans remain unscored', () => {
  const engine = { mortal_action_of: index => index >= 34 ? 37 : index };
  const choices = [{ index: 34 }, { index: 35 }, { index: 2 }, { index: null }];
  const weights = Array(46).fill(0); weights[37] = .8; weights[2] = .2;
  const result = weightsByChoice(engine, choices, weights, action => action === 2 ? 2 : -1);
  assert.deepEqual(result.map(choice => choice.weight), [.8, .8, .2, null]);
});

test('invalid alias weights and excessive combined mass fail explicitly', () => {
  const engine = { mortal_action_of: () => 4 };
  for (const value of [NaN, Infinity, -.1, 1.1]) {
    const weights = Array(46).fill(0); weights[34] = value;
    assert.throws(() => weightsByChoice(engine, [{ index: 4 }], weights, () => 4), /invalid/);
  }
  const weights = Array(46).fill(0); weights[4] = .6; weights[34] = .6;
  assert.throws(() => weightsByChoice(engine, [{ index: 4 }], weights, () => 4), /invalid/);
});
