"""Tie semantics must be invariant to player identities in every evaluator."""
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from neural import arena, duel, searched, selfplay
from neural.outcomes import placement_rewards, placements, win_shares, validate_budget


class OutcomeTests(unittest.TestCase):
    def test_ties_share_ranks_rewards_and_first_place(self):
        scores = np.array([[40, 30, 20, 10], [40, 40, 20, 10], [40, 30, 10, 10],
                           [40, 40, 40, 10], [0, 0, 0, 0]])
        expected = np.array([[1, 2, 3, 4], [1.5, 1.5, 3, 4], [1, 2, 3.5, 3.5],
                             [2, 2, 2, 4], [2.5, 2.5, 2.5, 2.5]])
        np.testing.assert_array_equal(placements(scores), expected)
        np.testing.assert_allclose(placement_rewards(scores, [1.5, .5, -.5, -1.5]), 2.5 - expected)
        np.testing.assert_allclose(win_shares(scores), [[1, 0, 0, 0], [.5, .5, 0, 0],
                                                       [1, 0, 0, 0], [1/3, 1/3, 1/3, 0], [.25]*4])
        # Nonlinear rewards must be averaged, not interpolated at a mean rank.
        np.testing.assert_allclose(placement_rewards(scores[-1:], [8, 4, 0, 0]), [[3]*4])
        for order in itertools.permutations(range(4)):
            order = list(order)
            np.testing.assert_array_equal(placements(scores[:, order]), expected[:, order])
            np.testing.assert_allclose(win_shares(scores[:, order]), win_shares(scores)[:, order])
            for module in (arena, duel, searched):
                for player in range(4):
                    np.testing.assert_array_equal(module.placements(scores[:, order], player), expected[:, order[player]])

    def test_duplicate_and_duel_reports_share_tied_wins(self):
        scores = np.zeros((2, 4), dtype=np.int32)
        with patch.object(arena, 'evaluate_games', return_value=(scores, 16)):
            result = arena.duplicate(None, 2, 1, 'cpu')
        self.assertEqual(result['placement'], 2.5)
        self.assertEqual(result['wins'], .25)
        with patch.object(duel, 'table', return_value=scores):
            result = duel.duel(None, None, 2, 1, 'cpu')
        self.assertEqual(result['placement'], 2.5)
        self.assertTrue(all(row['wins'] == .25 for row in result['by_seat']))

    def test_measure_reports_tie_convention(self):
        engine = SimpleNamespace(all_finished=lambda: True,
                                 final_scores=lambda: np.zeros((1, 4), dtype=np.int32).tobytes())
        net = SimpleNamespace(kind='engine', eval=lambda: None)
        with patch.object(selfplay.riichi_py, 'Arena', return_value=engine):
            result = selfplay.measure(net, 1, 1, 'cpu')
        self.assertEqual(result['placement'], 2.5)
        self.assertEqual(result['wins'], .25)

    def test_invalid_scores_and_budgets_are_rejected(self):
        for scores in (np.zeros(4), np.zeros((2, 3)), [[0, 1, 2, np.nan]], [[0, 1, 2, np.inf]]):
            with self.assertRaises(ValueError):
                placements(scores)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                validate_budget(1, bad)
            with self.assertRaises(ValueError):
                validate_budget(bad, 1)
