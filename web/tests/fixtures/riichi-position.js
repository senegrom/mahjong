import assert from 'node:assert/strict';
import { MatchSession } from '../../src/lib/session.js';
import { MOVES, average, swapMoves } from '../../src/lib/suits.js';

export const RIICHI_SEED = 2;
const commands = ['7z', '6z', '2z', '9m', null, '9s', null, '1p', null]
  .map(tile => ({ type: 'choose', kind: tile ? 'discard' : 'pass', tile }));
const weights = entries => Array.from({ length: 46 }, (_, action) => entries[action] ?? 0);
const allowed = mask => Array.from(mask, Boolean);

/** A real played position, not a hand/mask/translator mock. The fixed legal
 * history reaches two riichi discards; fail explicitly if that fixture changes.
 * `plan` is every answer the followed seat's adviser is given, in the order
 * it asks: one sure answer a move up to the riichi, then the declaration and
 * its tile, each hesitant and so asked again of the renamed game, whose
 * answer comes after it in its own moves (`firstRenamed`, `secondRenamed`).
 * Their means (`declared`, `tile`) declare and throw `preferred`; ranked by
 * the two stages' product, the plain discard would win instead. */
export function riichiPosition(Game) {
  const match = new MatchSession(Game, RIICHI_SEED, 'club'), plan = [];
  try {
    match.advance(false);
    for (const command of commands) {
      const choice = match.engine.agent_choices().find(choice => choice.kind === command.kind
        && (choice.tile ?? null) === command.tile);
      assert.ok(choice, 'fixture command must still be legal');
      const action = match.engine.mortal_action_of(choice.index);
      const mask = allowed(match.engine.agent_mask_mortal());
      assert.ok(mask[action] && action !== 37);
      plan.push({ action, weights: weights({ [action]: 1 }), mask });
      match.apply(command); match.advance(false);
    }
    const choices = match.engine.agent_choices();
    const riichi = choices.filter(choice => choice.kind === 'riichi');
    assert.equal(riichi.length, 2, 'fixture must offer exactly two riichi discards');
    const afterMask = allowed(match.engine.agent_mask_after_reach());
    const actions = afterMask.flatMap((valid, action) => valid ? [action] : []);
    assert.equal(actions.length, 2);
    const translated = actions.map(action => choices.find(choice =>
      choice.index === match.engine.agent_action_from_mortal(action, true)));
    assert.ok(translated.every(choice => choice?.kind === 'riichi'));
    const [alternative, preferred] = translated;
    const first = { action: 37, weights: weights({ 37: .55, [actions[0]]: .45 }),
      value: -1.25, mask: allowed(match.engine.agent_mask_mortal()) };
    const firstRenamed = { action: 37, weights: weights({ 37: .5, [MOVES[actions[0]]]: .5 }), value: 7, mask: swapMoves(first.mask) };
    const second = { action: actions[1], weights: weights({ [actions[0]]: .25, [actions[1]]: .75 }),
      value: 99, mask: afterMask };
    const secondRenamed = { action: MOVES[actions[1]], weights: weights({ [MOVES[actions[0]]]: .3, [MOVES[actions[1]]]: .7 }),
      value: 98, mask: swapMoves(afterMask) };
    assert.ok(first.mask[actions[0]], 'plain discard is a separate first-stage alternative');
    const declared = average(first.weights, firstRenamed.weights, first.mask), tile = average(second.weights, secondRenamed.weights, afterMask);
    assert.equal(declared.action, 37);
    assert.equal(tile.action, actions[1]);
    return { match, plan: [...plan, first, firstRenamed, second, secondRenamed], first, firstRenamed, second, secondRenamed,
      declared, tile, actions, alternative, preferred, warmup: commands.length };
  } catch (error) { match.dispose(); throw error; }
}
