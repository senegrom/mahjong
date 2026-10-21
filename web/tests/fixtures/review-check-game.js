import assert from 'node:assert/strict';
import { MatchSession } from '../../src/lib/session.js';
import { scanMatch } from '../../src/lib/game-review.js';

/** The review check's game (scripts/game-review-check.mjs): seed 5 against
 * Beginner players, played to its standings by a player who takes every win,
 * declares every riichi and otherwise throws the tile just drawn. It is the
 * game the "What would change its mind?" prototype was measured on. */
export function reviewCheckGame(Game) {
  const match = new MatchSession(Game, 5, 'beginner');
  try {
    match.advance(false);
    for (let step = 0; step < 5000 && !match.over; step++) {
      if (match.engine.hand_is_over()) { match.apply({ type: 'next' }); match.advance(false); continue; }
      const choices = match.choices, drawn = match.view.seats[0].drawn;
      const choice = choices.find(c => ['ron', 'tsumo', 'riichi'].includes(c.kind)) ?? choices.find(c => c.kind === 'pass')
        ?? choices.find(c => c.kind === 'discard' && c.tile === drawn) ?? choices.find(c => c.kind === 'discard') ?? choices[0];
      match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); match.advance(false);
    }
    assert.ok(match.over, 'the game reaches its standings');
    return match.snapshot();
  } finally { match.dispose(); }
}

/** A whole game against Club players in which the player makes, at every
 * decision, the move the page's Club adviser picks for it. It comes to riichi
 * far more often than the review check's player, who throws the tile just
 * drawn, so a riichi's second question is asked in hands after the first. */
export function clubAdvisedGame(Game, seed) {
  const match = new MatchSession(Game, seed, 'club');
  try {
    match.advance(false);
    for (let step = 0; step < 20000 && !match.over; step++) {
      if (match.engine.hand_is_over()) { match.apply({ type: 'next' }); match.advance(false); continue; }
      const advised = match.engine.agent_pick('club');
      match.apply({ type: 'choose', kind: advised.kind, tile: advised.tile ?? null }); match.advance(false);
    }
    assert.ok(match.over, 'the game reaches its standings');
    return match.snapshot();
  } finally { match.dispose(); }
}

/** Shows `visit` every decision of a finished game as the review reads it:
 * on a replay played on to each hand's end, where the engine still keeps the
 * hand's decisions, with the decision's table as the page showed it
 * (`view`), its review note, and its written position copied as plain data,
 * the way the page passes a position on. */
export function eachDecision(Game, snapshot, visit) {
  const hands = scanMatch(Game, snapshot);
  const match = new MatchSession(Game, snapshot.seed, snapshot.opponents);
  try {
    match.advance(false);
    let at = 0;
    for (const hand of hands) {
      for (; at < hand.end; at++) { match.apply(snapshot.commands[at]); match.advance(false); }
      const engine = match.engine, notes = engine.review();
      hand.frames.forEach((frame, index) => {
        const position = JSON.parse(JSON.stringify(engine.review_position(index)));
        visit({ hand, index, view: frame.view, note: notes[index], engine, position });
      });
    }
  } finally { match.dispose(); }
}
