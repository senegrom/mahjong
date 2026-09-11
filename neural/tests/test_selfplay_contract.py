"""The promises self-play has to keep before anything is learned from it.

Every one of these was a real defect rather than a hypothetical: a hand
that ended on a step nobody recorded went unpaid and its decisions were
then paid by the next hand; a round that ran out of steps turned games
still in progress into final placements; an index naming no legal move was
replaced by one that was legal while the record kept the move that was
asked for; imagining a world drew from the generator that deals the cards,
so thinking harder changed the deal; and a round too small for one
minibatch trained on nothing and said so nowhere.

    python -m unittest neural.tests.test_selfplay_contract -v
"""

from __future__ import annotations

import unittest

import numpy as np

import riichi_py

GAMES = 8
SEED = 4_242


class FirstLegal:
    """A player that always takes the first move the rules allow.

    Enough to drive the arena without a checkpoint: these tests are about
    what the collector records and refuses, not about how well anyone
    plays.
    """

    kind = "engine"

    def eval(self):
        return self

    def choose(self, views, rows, players, legal):
        return np.asarray(legal, dtype=bool).argmax(axis=1).astype(np.int64)

    def everything(self, planes, mask):
        """Policy, value and belief, shaped as the collector expects.

        The policy puts all its weight on the first legal move, so play is
        deterministic and never names an action the rules forbid — which
        matters now that the arena refuses one rather than substituting.
        """
        import torch

        rows = mask.shape[0]
        logits = torch.where(mask, 0.0, float("-inf"))
        logits[:, 0] = torch.where(mask[:, 0], torch.tensor(1.0), logits[:, 0])
        first = mask.float().argmax(dim=1)
        logits = torch.full_like(logits, float("-inf"))
        logits[torch.arange(rows), first] = 0.0
        value = torch.zeros(rows)
        guessed = torch.zeros(rows, riichi_py.OPPONENTS, riichi_py.POSITIONS)
        return logits, value, guessed


def played_out(arena, games, limit=4000):
    """Plays every seat's first legal move until the games end."""
    steps = 0
    while not arena.all_finished() and steps < limit:
        steps += 1
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        if not (seats != 0xFF).any():
            break
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, -1).astype(bool)
        live = seats != 0xFF
        choice = np.where(live, mask.argmax(axis=1), 0).astype(np.int64)
        arena.step(choice.tolist())
    return steps


class StrictActions(unittest.TestCase):
    """An index that names no legal move must not be quietly replaced."""

    def test_a_lenient_arena_substitutes_and_carries_on(self):
        arena = riichi_py.Arena(games=1, seed=SEED, bot_places=[])
        self.assertFalse(arena.strict, "a page wants to shrug at a stray index")
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(1, -1).astype(bool)
        illegal = int(np.nonzero(~mask[0])[0][0])
        arena.step([illegal])  # does not raise

    def test_a_strict_arena_refuses_rather_than_substitute(self):
        arena = riichi_py.Arena(games=1, seed=SEED, bot_places=[])
        arena.strict = True
        self.assertTrue(arena.strict)
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(1, -1).astype(bool)
        illegal = int(np.nonzero(~mask[0])[0][0])
        with self.assertRaises(ValueError) as caught:
            arena.step([illegal])
        self.assertIn("strict", str(caught.exception))

    def test_strict_play_is_otherwise_unchanged(self):
        """Legal moves are played the same either way, so turning it on
        does not change what a run does, only what it refuses."""
        lenient = riichi_py.Arena(games=GAMES, seed=SEED, bot_places=[])
        strict = riichi_py.Arena(games=GAMES, seed=SEED, bot_places=[])
        strict.strict = True
        played_out(lenient, GAMES)
        played_out(strict, GAMES)
        self.assertEqual(
            np.frombuffer(lenient.final_scores(), dtype=np.int32).tolist(),
            np.frombuffer(strict.final_scores(), dtype=np.int32).tolist(),
        )


class SeparateStreams(unittest.TestCase):
    """What is imagined must not disturb what is dealt."""

    def deal_after_imagining(self, times: int):
        arena = riichi_py.Arena(games=GAMES, seed=SEED, bot_places=[])
        beliefs = np.full(GAMES * riichi_py.HANDS, 1.0 / riichi_py.POSITIONS, dtype=np.float32)
        for _ in range(times):
            arena.imagined_hands_bytes(beliefs.tobytes())
        played_out(arena, GAMES)
        return np.frombuffer(arena.final_scores(), dtype=np.int32).tolist()

    def test_imagining_more_does_not_change_the_games(self):
        none = self.deal_after_imagining(0)
        some = self.deal_after_imagining(1)
        many = self.deal_after_imagining(25)
        self.assertEqual(none, some, "one imagined world changed the deal")
        self.assertEqual(none, many, "twenty-five imagined worlds changed the deal")

    def test_imagining_is_still_random(self):
        """Separate does not mean frozen: the worlds must still vary."""
        arena = riichi_py.Arena(games=GAMES, seed=SEED, bot_places=[])
        beliefs = np.full(GAMES * riichi_py.HANDS, 1.0 / riichi_py.POSITIONS, dtype=np.float32)
        first = arena.imagined_hands_bytes(beliefs.tobytes())
        second = arena.imagined_hands_bytes(beliefs.tobytes())
        self.assertNotEqual(bytes(first), bytes(second), "every world came out the same")


class TruncationIsNotAnEnding(unittest.TestCase):
    """A game that did not finish has no result to report."""

    def test_self_play_refuses_a_round_that_ran_out_of_steps(self):
        from neural import selfplay, zoo  # noqa: F401  (import cost is the point)

        with self.assertRaises(RuntimeError) as caught:
            selfplay.play(FirstLegal(), games=4, seed=SEED, device="cpu", max_steps=3)
        self.assertIn("unfinished", str(caught.exception))

    def test_the_duel_refuses_a_run_that_ran_out_of_steps(self):
        from neural import duel

        with self.assertRaises(RuntimeError) as caught:
            duel.table(FirstLegal(), FirstLegal(), games=4, seed=SEED, place=0,
                       device="cpu", max_steps=3)
        self.assertIn("unfinished", str(caught.exception))

    def test_a_generous_limit_finishes_and_reports(self):
        """The guard must not fire on an ordinary run."""
        from neural import duel

        scores = duel.table(FirstLegal(), FirstLegal(), games=4, seed=SEED, place=0,
                            device="cpu")
        self.assertEqual(scores.shape, (4, 4))


class EveryHandIsPaidOnce(unittest.TestCase):
    """The defect this guards against, in its own words.

    When every live game's next decision belonged to a seated foreign
    opponent, the collector stepped the arena and went straight round the
    loop, skipping the block that pays out a hand that has just ended. The
    engine forgets a hand ended as soon as the next step is taken, so those
    points were lost — and the decisions waiting for them stayed waiting,
    to be paid instead by the hand after, which credits a decision with
    points from a hand it was not part of.

    `play` now asserts that nothing is still waiting once the games are
    over, so exercising the path is enough to catch a regression. A
    foreign opponent is seated in every game to make the path common: with
    two games, the steps where both owe their decision to that one player
    come round often.
    """

    def test_a_hand_ending_on_an_opponent_only_step_is_still_paid(self):
        from neural import selfplay

        batch = selfplay.play(
            FirstLegal(),
            games=2,
            seed=SEED,
            device="cpu",
            opponents=[FirstLegal()],
            opponent_share=1.0,
        )
        self.assertGreater(batch.hands, 0, "no hand was played")
        self.assertGreater(batch.decisions, 0, "nothing was recorded")
        self.assertEqual(len(batch.returns), batch.decisions)

    def test_the_same_holds_without_any_foreign_opponent(self):
        from neural import selfplay

        batch = selfplay.play(FirstLegal(), games=2, seed=SEED, device="cpu")
        self.assertGreater(batch.hands, 0)


class RewardsAreVersioned(unittest.TestCase):
    def test_a_round_says_which_objective_built_it(self):
        from neural import selfplay

        self.assertEqual(selfplay.REWARD_VERSION, 1)
        batch = selfplay.play(FirstLegal(), games=2, seed=SEED, device="cpu")
        self.assertEqual(batch.reward_version, selfplay.REWARD_VERSION)
        self.assertEqual(len(batch.returns), batch.decisions)


if __name__ == "__main__":
    unittest.main()
