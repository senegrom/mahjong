"""What the evaluator and the promotion gate have to guarantee.

The evaluator's own symmetry is the first thing to check, because every
claim of strength rests on it: sit a deterministic network against three
copies of itself, in all four seats over the same deals, and the four
placements must sum to ten on every deal. Mean exactly 2.5, spread exactly
zero. Anything else and the harness has a bias of its own, and every
comparison it has ever reported is off by that amount.

    python -m unittest neural.tests.test_promotion -v
"""

from __future__ import annotations

import unittest

import numpy as np

import riichi_py

from neural import duel, promote

SEED = 5_150


class FirstLegal:
    """A deterministic player: always the first move the rules allow."""

    kind = "engine"

    def eval(self):
        return self

    def choose(self, views, rows, players, legal):
        return np.asarray(legal, dtype=bool).argmax(axis=1).astype(np.int64)


class TheEvaluatorIsSymmetric(unittest.TestCase):
    def test_a_network_against_itself_is_exactly_level(self):
        report = duel.duel(FirstLegal(), FirstLegal(), games=24, seed=SEED, device="cpu")
        self.assertAlmostEqual(report["placement"], 2.5, places=9)
        self.assertAlmostEqual(report["standard_error"], 0.0, places=9)

    def test_every_deal_sums_to_ten_across_the_four_seatings(self):
        """The identity the symmetry rests on, checked deal by deal rather
        than only in the average, where two errors could cancel."""
        rows = [
            duel.placements(
                duel.table(FirstLegal(), FirstLegal(), games=16, seed=SEED, place=place,
                           device="cpu"),
                place,
            ).astype(float)
            for place in range(4)
        ]
        total = np.stack(rows).sum(axis=0)
        self.assertTrue(np.all(total == 10.0), f"a deal summed to {sorted(set(total))}")


class TheGateHoldsItsGround(unittest.TestCase):
    def test_a_candidate_no_better_than_the_champion_is_not_promoted(self):
        verdict = promote.gate(
            FirstLegal(), FirstLegal(), games=16, device="cpu",
            candidate_name="same", champion_name="same",
        )
        self.assertFalse(verdict.promoted, verdict.reason)
        self.assertIn("short of", verdict.reason)

    def test_the_verdict_records_what_it_was_decided_on(self):
        verdict = promote.gate(FirstLegal(), FirstLegal(), games=8, device="cpu")
        self.assertEqual(len(verdict.directions), 2, "both directions are measured")
        self.assertNotEqual(
            verdict.directions[0].seed, verdict.directions[1].seed,
            "the two directions must not share their deals",
        )
        for row in verdict.directions:
            self.assertGreaterEqual(row.seed, promote.EVALUATION_SEED_BASE,
                                    "evaluation must not borrow a training seed")
        self.assertIn("promoted", verdict.as_json())

    def test_attempts_move_the_deals(self):
        """Testing the same candidate again must not re-run the same games,
        or a gate becomes a coin flipped until it lands."""
        first = promote.gate(FirstLegal(), FirstLegal(), games=8, device="cpu", attempt=1)
        second = promote.gate(FirstLegal(), FirstLegal(), games=8, device="cpu", attempt=2)
        self.assertNotEqual(first.directions[0].seed, second.directions[0].seed)


class SeedsDoNotCollide(unittest.TestCase):
    def test_training_seeds_are_far_below_evaluation_seeds(self):
        """`train_combined` measures at 7_000_000 + generation and duels
        elsewhere; nothing training uses may reach the evaluation range."""
        self.assertGreater(promote.EVALUATION_SEED_BASE, 100_000_000)


if __name__ == "__main__":
    unittest.main()
