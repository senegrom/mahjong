"""A table nobody follows keeps no mjai log.

The arena writes every game's mjai events for Mortal's follower and holds
them until `mjai_all` hands them over. Only the follower asks, and `Views`
builds one only for a network that sees Mortal's planes, so at a table of
older networks and heuristic players every game's whole log stayed in
memory until the arena was dropped: 22,118 lines and 1.1 MB over the
sixteen games below, 40 to 50 MB over an evaluation of five hundred.

    python -m unittest neural.tests.test_unfollowed_tables -v
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

import riichi_py

from neural import selfplay
from neural.observe import Views
from neural.tests.test_selfplay_contract import FirstLegal

NATIVE_ARENA = riichi_py.Arena
GAMES = 16
SEED = 99


class UnfollowedTablesTests(unittest.TestCase):
    def test_an_evaluation_nobody_follows_holds_no_log(self):
        """Three heuristic players and a network of the engine's kind, as
        `evaluate_games` seats an old checkpoint: once the games are over
        nothing is held, and they went move for move as at a table that
        keeps its log."""
        built = []

        def build(*args, **kwargs):
            built.append(NATIVE_ARENA(*args, **kwargs))
            return built[-1]

        with patch.object(selfplay.riichi_py, "Arena", side_effect=build):
            scores, hands = selfplay.evaluate_games(FirstLegal(), GAMES, SEED, device="cpu")
        (arena,) = built
        self.assertTrue(arena.all_finished())
        self.assertEqual(sum(len(lines) for lines in arena.mjai_all()), 0)

        # The same games at a table that keeps its log, each deciding seat
        # taking the first legal move as the evaluation's player did.
        kept = NATIVE_ARENA(games=GAMES, seed=SEED, bot_places=[1, 2, 3])
        while not kept.all_finished():
            mask = np.frombuffer(kept.legal_mask(), dtype=np.uint8).reshape(GAMES, -1)
            kept.step(mask.argmax(axis=1).tolist())
        np.testing.assert_array_equal(
            scores, np.frombuffer(kept.final_scores(), dtype=np.int32).reshape(GAMES, 4))
        self.assertEqual(hands, sum(kept.hands_done()))
        # Every game's whole log, which the evaluation no longer holds.
        for lines in kept.mjai_all():
            self.assertEqual(json.loads(lines[0])["type"], "start_game")
            self.assertEqual(json.loads(lines[-1])["type"], "end_game")

    def test_only_a_table_without_a_follower_stops_its_log(self):
        """A table that seats a network of Mortal's planes, alone or beside
        an older one, is followed and writes its log to the end of every
        game. A table of older networks alone stops writing it, and stays
        stopped: not even the end of a game is written."""
        for kinds, followed in (({"engine"}, False), ({"engine", "mortal"}, True), (None, True)):
            with self.subTest(kinds=kinds):
                arena = NATIVE_ARENA(games=2, seed=SEED)
                views = Views(arena, 2, kinds)
                self.assertEqual(views.observer is not None, followed)
                self.assertEqual([bool(lines) for lines in arena.mjai_all()], [followed] * 2)
                steps = 0
                while not arena.all_finished():
                    steps += 1
                    self.assertLess(steps, 4000, "the games should finish")
                    mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(2, -1)
                    arena.step(mask.argmax(axis=1).tolist())
                held = arena.mjai_all()
                if followed:
                    self.assertEqual([json.loads(lines[-1])["type"] for lines in held],
                                     ["end_game"] * 2)
                else:
                    self.assertEqual(held, [[], []])


if __name__ == "__main__":
    unittest.main()
