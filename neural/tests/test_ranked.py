"""A player that asks the sibling head keeps the policy's choice until the
head has a reason, and takes the head's when it does.

    python -m unittest neural.tests.test_ranked -v
"""

from __future__ import annotations

import unittest

import numpy as np
import riichi_py
import torch
from torch import nn

from neural import contract, duel, ranked, sibling_head, zoo
from neural.model import MORTAL_PLANES, PolicyValueNet
from neural.observe import Views


class RankedPlayerTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(6)
        self.net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46).eval()

    def tearDown(self):
        torch.set_num_threads(self.previous)

    def test_a_head_that_has_learned_nothing_plays_the_policys_game(self):
        """With every move scored even, the player is the policy, move for
        move: the same deals against the same opponent end the same way."""
        plain = zoo.MortalSpacePlayer(self.net, "cpu")
        head = sibling_head.Ranker(self.net.channels)
        player = ranked.RankedPlayer(self.net, head, k=4, margin=0.05, device="cpu")
        with_head = duel.table(player, plain, games=2, seed=31, place=0, device="cpu")
        without = duel.table(plain, plain, games=2, seed=31, place=0, device="cpu")
        np.testing.assert_array_equal(with_head, without)
        self.assertGreater(player.asked, 0)
        self.assertEqual(player.overrode, 0, "an even head never overrides")

    def test_the_heads_favourite_is_taken_only_past_the_margin(self):
        """A head that scores our moves by a fixed vector: among the
        policy's first four legal moves, its favourite is chosen when it
        beats the policy's move by the margin, else the policy's stands."""
        fixed = torch.zeros(riichi_py.ACTIONS)
        fixed[:34] = torch.linspace(0.0, 1.0, 34)  # later discards score higher

        class Fixed(nn.Module):
            def forward(self, features, pooled):
                return fixed.unsqueeze(0).expand(features.shape[0], -1)

        games = 3
        arena = riichi_py.Arena(games=games, seed=44, bot_places=[])
        views = Views(arena, games, {"mortal"})
        views.advance()
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        rows = np.nonzero(seats != 0xFF)[0]
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
        deciding = players[rows, np.minimum(seats[rows], 3)].astype(np.int64)
        views.prepare(rows, deciding)
        legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, riichi_py.ACTIONS).astype(bool)

        for margin, sure in ((0.0, 1.0), (10.0, 1.0), (0.0, 0.0)):
            player = ranked.RankedPlayer(self.net, Fixed(), k=4, margin=margin, device="cpu", sure=sure)
            choice = player.choose(views, rows, deciding, legal[rows])
            order, _l, _v, _g = contract.root_order(player.served, None, views, rows, deciding, legal, "cpu")
            for at, game in enumerate(rows):
                candidates = order[game][:4]
                candidates = candidates[legal[game][candidates]]
                worth = fixed[candidates].numpy()
                best = int(np.argmax(worth))
                asked = sure > 0.0 and len(candidates) > 1
                expected = candidates[best] if asked and best and worth[best] - worth[0] > margin else candidates[0]
                self.assertEqual(int(choice[at]), int(expected), f"game {game} at margin {margin}, sure {sure}")
                self.assertTrue(legal[game][choice[at]], "the choice is legal")
            if margin == 10.0:
                self.assertEqual(player.overrode, 0, "nothing clears a margin of ten")
            if sure == 0.0:
                self.assertEqual(player.asked, 0, "a policy sure of every move is never second-guessed")
                self.assertEqual(player.taken_sure, len(rows))


if __name__ == "__main__":
    unittest.main()
