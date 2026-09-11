"""The account of what each decision was worth, and the loop that uses it.

F2 of the review: the search chose its moves with the critic, the policy
learned those choices, and the critic was never told what any of it came
to. The value was fetched and thrown away. That is not a loop, it is a
network agreeing with itself, and the missing half is an outcome the games
actually paid.

    python -m unittest neural.tests.test_ledger -v
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

import riichi_py

from neural import ledger


class TheAccountBalances(unittest.TestCase):
    def test_a_decision_is_paid_its_own_hand_and_its_own_game(self):
        account = ledger.Ledger(games=1)
        row = account.open(0, 2)
        self.assertEqual(row, 0)
        self.assertEqual(account.rewards, [0.0])

    def test_settling_pays_everyone_waiting_and_clears_them(self):
        class Ending:
            def hand_ended(self):
                return bytes([1])

            def hand_result(self):
                return np.array([[4000, -1000, -1000, -2000]], dtype=np.int32).tobytes()

        account = ledger.Ledger(games=1)
        first = account.open(0, 0)
        second = account.open(0, 1)
        self.assertEqual(account.settle(Ending()), 1)
        self.assertAlmostEqual(account.rewards[first], 1.0, places=6)
        self.assertAlmostEqual(account.rewards[second], -0.25, places=6)
        self.assertEqual(account.waiting[0][0], [], "paid twice by the next hand")

    def test_a_missed_ending_is_refused_rather_than_paid_by_the_next_hand(self):
        class Over:
            def hands_done(self):
                return [3]

            def final_scores(self):
                return np.array([[40000, 30000, 20000, 10000]], dtype=np.int32).tobytes()

        class Ending:
            def hand_ended(self):
                return bytes([1])

            def hand_result(self):
                return np.array([[1000, -1000, 0, 0]], dtype=np.int32).tobytes()

        account = ledger.Ledger(games=1)
        account.open(0, 0)
        # The decision is paid, so nothing is left waiting -- which is
        # exactly what makes this fault invisible without the engine's own
        # count. Its points came from the wrong hand.
        account.settle(Ending())
        self.assertEqual(account.waiting[0][0], [])
        with self.assertRaises(RuntimeError) as caught:
            account.close(Over())
        self.assertIn("went by unseen", str(caught.exception))

    def test_a_decision_still_waiting_at_the_end_is_refused(self):
        class Over:
            def hands_done(self):
                return [0]

            def final_scores(self):
                return np.array([[40000, 30000, 20000, 10000]], dtype=np.int32).tobytes()

        account = ledger.Ledger(games=1)
        account.open(0, 0)
        with self.assertRaises(RuntimeError) as caught:
            account.close(Over())
        self.assertIn("still waiting", str(caught.exception))


class TheSearchedLoopCloses(unittest.TestCase):
    """The critic has to be moved by what the games paid, or the evaluator
    the search leans on is never corrected by anything outside itself."""

    def test_an_outcome_target_moves_the_value_head(self):
        torch.manual_seed(0)
        value = torch.nn.Linear(8, 1)
        before = value.weight.detach().clone()
        planes = torch.randn(32, 8)
        returns = torch.randn(32)
        loss = torch.nn.functional.mse_loss(value(planes).squeeze(1), returns)
        loss.backward()
        torch.optim.SGD(value.parameters(), lr=0.1).step()
        self.assertGreater(
            float((value.weight - before).abs().max()), 1e-6,
            "the outcome reached nothing",
        )

    def test_an_improvement_distribution_keeps_what_the_search_did_not_rank(self):
        """A hard label throws away the policy's opinion about every move
        the search never looked at. Moving part of the mass keeps it."""
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        chosen = torch.tensor([2])
        for improve in (0.25, 0.5):
            base = torch.softmax(logits, dim=1)
            wanted = base * (1.0 - improve)
            wanted.scatter_add_(
                1, chosen.unsqueeze(1),
                torch.full_like(chosen, improve, dtype=base.dtype).unsqueeze(1),
            )
            wanted = wanted / wanted.sum(dim=1, keepdim=True)
            self.assertAlmostEqual(float(wanted.sum()), 1.0, places=6)
            self.assertGreater(float(wanted[0, 2]), float(base[0, 2]),
                               "the searched move gained nothing")
            self.assertGreater(float(wanted[0, 0]), 0.0,
                               "the policy's own favourite was erased")
            self.assertGreater(float(wanted[0, 0]), float(wanted[0, 3]),
                               "the ordering of the unranked moves was lost")

    def test_a_full_improvement_is_the_old_hard_label(self):
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        chosen = torch.tensor([2])
        base = torch.softmax(logits, dim=1)
        wanted = base * 0.0
        wanted.scatter_add_(
            1, chosen.unsqueeze(1),
            torch.full_like(chosen, 1.0, dtype=base.dtype).unsqueeze(1),
        )
        wanted = wanted / wanted.sum(dim=1, keepdim=True)
        self.assertAlmostEqual(float(wanted[0, 2]), 1.0, places=6)


class AnUnfinishedRoundIsRefused(unittest.TestCase):
    def test_it_says_how_many_were_left(self):
        class Midway:
            def all_finished(self):
                return False

            def seats(self):
                return bytes([0, 0xFF, 1])

        with self.assertRaises(RuntimeError) as caught:
            ledger.refuse_if_unfinished(Midway(), 3, 4000, "the searched round")
        self.assertIn("2 of 3 games", str(caught.exception))

    def test_a_finished_round_passes(self):
        class Done:
            def all_finished(self):
                return True

        ledger.refuse_if_unfinished(Done(), 3, 10, "the searched round")


class TheObjectiveIsVersioned(unittest.TestCase):
    def test_the_ledger_says_which_rewards_it_makes(self):
        self.assertEqual(ledger.REWARD_VERSION, 1)
        self.assertEqual(ledger.PLACEMENT_VALUE, tuple(riichi_py.PLACEMENT_VALUE))


if __name__ == "__main__":
    unittest.main()
