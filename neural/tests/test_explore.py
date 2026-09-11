"""Wandering off the policy, and telling the truth about having done so.

The value head only ever sees positions the policy reached, and a search
asks it what a move the policy would not have chosen is worth. Its error
there is a bias that correlates with the move being considered, which no
amount of averaging over imagined worlds can remove, because every world
shares it. Forcing a legal move now and then is the direct attack: it shows
the value head the positions the search will ask about.

The trap is what gets written down. PPO divides by the probability the
behaviour gave the action it took. Forcing a move and then recording the
policy's own probability for it says the policy chose something it did not,
and the ratio is then wrong on exactly the decisions this is for.

    python -m unittest neural.tests.test_explore -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from neural.selfplay import explore


def legal_mask(rows: int, allowed: list[int], width: int = 8) -> torch.Tensor:
    mask = torch.zeros(rows, width, dtype=torch.bool)
    for index in allowed:
        mask[:, index] = True
    return mask


def masked_logits(rows: int, allowed: list[int], width: int = 8) -> torch.Tensor:
    logits = torch.full((rows, width), float("-inf"))
    for position, index in enumerate(allowed):
        logits[:, index] = float(position)
    return logits


class NothingChangesAtZero(unittest.TestCase):
    def test_the_action_is_the_policy_s_own(self):
        torch.manual_seed(0)
        logits = masked_logits(256, [1, 3, 5])
        mask = legal_mask(256, [1, 3, 5])
        chosen, _ = explore(logits, mask, 0.0, np.random.default_rng(0))
        self.assertTrue(bool(mask.gather(1, chosen.unsqueeze(1)).all()), "an illegal move")

    def test_the_probability_is_the_policy_s_own_to_the_last_bit(self):
        logits = masked_logits(64, [0, 2, 7])
        mask = legal_mask(64, [0, 2, 7])
        torch.manual_seed(3)
        chosen, recorded = explore(logits, mask, 0.0, np.random.default_rng(0))
        wanted = torch.distributions.Categorical(logits=logits).log_prob(chosen)
        self.assertTrue(torch.allclose(recorded, wanted, atol=0, rtol=0))


class TheRecordedProbabilityIsTheBehaviour(unittest.TestCase):
    def test_it_is_the_mixture_not_the_policy(self):
        epsilon = 0.25
        allowed = [1, 3, 5]
        logits = masked_logits(512, allowed)
        mask = legal_mask(512, allowed)
        torch.manual_seed(11)
        chosen, recorded = explore(logits, mask, epsilon, np.random.default_rng(5))

        policy = torch.distributions.Categorical(logits=logits)
        share = torch.exp(policy.log_prob(chosen))
        wanted = torch.log((1 - epsilon) * share + epsilon / len(allowed))
        self.assertTrue(torch.allclose(recorded, wanted, atol=1e-6))

    def test_it_is_always_at_least_the_floor_the_mixture_guarantees(self):
        """No action the behaviour could take may be recorded as less
        likely than the even draw alone makes it."""
        epsilon = 0.1
        allowed = [0, 1, 2, 3]
        logits = masked_logits(256, allowed)
        mask = legal_mask(256, allowed)
        torch.manual_seed(7)
        _chosen, recorded = explore(logits, mask, epsilon, np.random.default_rng(1))
        floor = math.log(epsilon / len(allowed))
        self.assertTrue(bool((recorded >= floor - 1e-6).all()))

    def test_a_move_the_policy_hates_is_recorded_as_possible(self):
        """The whole point. The policy gives this move about nothing; the
        behaviour gives it epsilon over the legal moves, and that is what
        PPO must divide by."""
        epsilon = 0.5
        allowed = [0, 1]
        logits = torch.full((1, 8), float("-inf"))
        logits[0, 0] = 20.0   # everything
        logits[0, 1] = -20.0  # nothing
        mask = legal_mask(1, allowed)
        found = False
        for trial in range(200):
            torch.manual_seed(trial)
            chosen, recorded = explore(logits, mask, epsilon, np.random.default_rng(trial))
            if int(chosen[0]) == 1:
                found = True
                self.assertGreater(
                    float(recorded[0]), math.log(epsilon / 2) - 1e-6,
                    "the hated move was recorded as less likely than the even draw",
                )
                self.assertLess(float(recorded[0]), 0.0)
        self.assertTrue(found, "the hated move was never forced")


class ItOnlyEverPlaysLegalMoves(unittest.TestCase):
    def test_forced_moves_respect_the_mask(self):
        allowed = [2, 4, 6]
        logits = masked_logits(1024, allowed)
        mask = legal_mask(1024, allowed)
        torch.manual_seed(2)
        chosen, _ = explore(logits, mask, 1.0, np.random.default_rng(9))
        self.assertTrue(bool(mask.gather(1, chosen.unsqueeze(1)).all()))
        self.assertEqual(set(chosen.tolist()) - set(allowed), set())

    def test_a_full_share_draws_the_legal_moves_evenly(self):
        allowed = [0, 3, 7]
        logits = masked_logits(20_000, allowed)
        mask = legal_mask(20_000, allowed)
        torch.manual_seed(4)
        chosen, _ = explore(logits, mask, 1.0, np.random.default_rng(13))
        for index in allowed:
            share = float((chosen == index).float().mean())
            self.assertAlmostEqual(share, 1 / 3, delta=0.02)

    def test_one_legal_move_is_always_that_move(self):
        logits = masked_logits(32, [5])
        mask = legal_mask(32, [5])
        torch.manual_seed(1)
        chosen, recorded = explore(logits, mask, 0.7, np.random.default_rng(0))
        self.assertTrue(bool((chosen == 5).all()))
        self.assertTrue(torch.allclose(recorded, torch.zeros_like(recorded), atol=1e-6),
                        "the only move must be recorded as certain")


class ARoundActuallyWanders(unittest.TestCase):
    def test_exploring_changes_what_is_played(self):
        from neural import selfplay
        from neural.tests.test_selfplay_contract import FirstLegal

        calm = selfplay.play(FirstLegal(), games=4, seed=808, device="cpu")
        # FirstLegal puts all its weight on one move, so anything else
        # played can only have come from wandering.
        busy = selfplay.play(
            FirstLegal(), games=4, seed=808, device="cpu", explore_share=0.5
        )
        self.assertNotEqual(calm.decisions, busy.decisions,
                            "the round played out identically, so nothing wandered")


if __name__ == "__main__":
    unittest.main()
