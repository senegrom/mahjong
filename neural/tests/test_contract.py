"""What a search is allowed to assume about the network it is searching.

F1 of the review. `neural.searched` assumed the engine's planes, our own
action space, and that `from_payload` rebuilt whatever it was given. A
network of the current lineage satisfies none of those, and the search ran
anyway — one backbone out of two, the wrong planes, the wrong moves, and a
placement figure at the end that looked exactly like a measurement.

    python -m unittest neural.tests.test_contract -v
"""

from __future__ import annotations

import unittest

import riichi_py

from neural import contract


class OursIsReadCorrectly(unittest.TestCase):
    def test_an_engine_plane_network(self):
        class Ours:
            kind = "engine"
            actions = riichi_py.ACTIONS

            def planes(self):
                return riichi_py.PLANES

            def everything(self, planes, mask):
                raise NotImplementedError

        found = contract.of(Ours())
        self.assertEqual(found.reads, "engine")
        self.assertEqual(found.planes, riichi_py.PLANES)
        self.assertTrue(found.speaks_our_moves)
        self.assertTrue(found.has_value)
        self.assertIn("policy", found.parts)

    def test_a_mortal_plane_network(self):
        class Fusion:
            kind = "mortal"
            actions = 46
            ours = object()
            mortal = object()

            def planes(self):
                return 1012

            def everything(self, planes, mask):
                raise NotImplementedError

        found = contract.of(Fusion())
        self.assertEqual(found.reads, "mortal")
        self.assertEqual(found.planes, 1012)
        self.assertEqual(found.answers, 46)
        self.assertFalse(found.speaks_our_moves,
                         "Mortal's moves need translating and the contract must say so")
        # The whole checkpoint, not one backbone of it.
        self.assertIn("ours", found.parts)
        self.assertIn("mortal", found.parts)
        self.assertIn("fusion", found.parts)

    def test_the_contract_can_be_written_into_a_log(self):
        class Ours:
            kind = "engine"
            actions = riichi_py.ACTIONS

            def planes(self):
                return riichi_py.PLANES

            def everything(self, planes, mask):
                raise NotImplementedError

        written = contract.of(Ours()).describe()
        self.assertEqual(written["reads"], "engine")
        self.assertIsInstance(written["parts"], list)


class EachKindGetsItsOwnServer(unittest.TestCase):
    def test_engine_planes_are_served_from_the_arena(self):
        class Ours:
            kind = "engine"
            actions = riichi_py.ACTIONS

            def planes(self):
                return riichi_py.PLANES

            def everything(self, planes, mask):
                raise NotImplementedError

        served = contract.serve(Ours())
        self.assertIsInstance(served, contract.EngineServed)

    def test_mortal_planes_are_served_from_the_follower(self):
        class Fusion:
            kind = "mortal"
            actions = 46

            def planes(self):
                return 1012

            def everything(self, planes, mask):
                raise NotImplementedError

        served = contract.serve(Fusion())
        self.assertIsInstance(served, contract.MortalServed)


class UnservableCombinationsAreRefusedByName(unittest.TestCase):
    def test_a_network_with_no_critic_cannot_be_searched(self):
        class Blind:
            kind = "engine"
            actions = riichi_py.ACTIONS

            def planes(self):
                return riichi_py.PLANES

        with self.assertRaises(SystemExit) as caught:
            contract.serve(Blind(), "blind.pt")
        self.assertIn("no value head", str(caught.exception))
        self.assertIn("blind.pt", str(caught.exception))

    def test_the_wrong_number_of_engine_planes_is_refused(self):
        class Odd:
            kind = "engine"
            actions = riichi_py.ACTIONS

            def planes(self):
                return riichi_py.PLANES + 1

            def everything(self, planes, mask):
                raise NotImplementedError

        with self.assertRaises(SystemExit) as caught:
            contract.serve(Odd(), "odd.pt")
        self.assertIn("cannot be searched here", str(caught.exception))

    def test_an_unknown_observation_is_refused_rather_than_approximated(self):
        class Strange:
            kind = "something else"
            actions = 12

            def planes(self):
                return 7

            def everything(self, planes, mask):
                raise NotImplementedError

        with self.assertRaises(SystemExit) as caught:
            contract.serve(Strange(), "strange.pt")
        self.assertIn("no server here builds", str(caught.exception))
        self.assertIn("rather than approximating", str(caught.exception))


class LeavesNeedAFollowerAndSayWhenTheyHaveNone(unittest.TestCase):
    def test_the_refusal_names_what_is_missing(self):
        class Nothing:
            pass

        with self.assertRaises(RuntimeError) as caught:
            contract.arena_follower(Nothing())
        self.assertIn("no follower", str(caught.exception))



class TheContractServesTheSamePositionNormalPlayDoes(unittest.TestCase):
    """The acceptance test that matters: a root served through the contract
    has to produce the policy normal play would, or the search is measuring
    a network nobody trained.

    Needs a real checkpoint, and skips without one.
    """

    CHECKPOINT = "E:/tmp-claude/mahjong/leashed-g35.pt"

    def setUp(self):
        import os

        if not os.path.exists(self.CHECKPOINT):
            self.skipTest(f"no checkpoint at {self.CHECKPOINT}")

    def test_the_root_matches_ordinary_play(self):
        import numpy as np
        import torch

        from neural import zoo
        from neural.observe import Views

        net = zoo.load_player(self.CHECKPOINT, "cpu")
        net.eval()
        served = contract.serve(net, self.CHECKPOINT)
        self.assertIsInstance(served, contract.MortalServed)

        games = 4
        arena = riichi_py.Arena(games=games, seed=1234, bot_places=[])
        arena.strict = True
        views = Views(arena, games, {net.kind})
        contract.remember_follower(arena, views.observer.follower)
        views.advance()

        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        legal = legal.reshape(games, riichi_py.ACTIONS).astype(bool)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
        deciding = players[np.arange(games), np.minimum(seats, 3)].astype(np.int64)
        rows = np.nonzero(live)[0]
        views.prepare(rows, deciding[rows])

        planes, mask = served.root(arena, views, rows, deciding[rows], legal, "cpu")
        self.assertEqual(planes.shape[1], served.contract.planes)
        self.assertEqual(planes.shape[2], riichi_py.POSITIONS)
        self.assertEqual(mask.shape[1], served.contract.answers)

        with torch.no_grad():
            through_contract, _value, _hands = net.everything(planes, mask)

        # What ordinary play would have asked, built its own way.
        own_planes, _own = views.sparse_and_masks(rows, deciding[rows])
        own_mask = zoo.translatable(legal[rows])
        own_mask[~own_mask.any(axis=1), zoo.MORTAL_PASS] = True
        with torch.no_grad():
            ordinarily, _v, _h = net.everything(
                own_planes.dense("cpu"), torch.from_numpy(own_mask)
            )
        self.assertTrue(
            torch.allclose(through_contract, ordinarily, atol=1e-5),
            "the contract served a different position from the one play sees",
        )

    def test_its_answers_translate_to_legal_engine_moves(self):
        import numpy as np

        from neural import zoo

        net = zoo.load_player(self.CHECKPOINT, "cpu")
        served = contract.serve(net, self.CHECKPOINT)

        arena = riichi_py.Arena(games=2, seed=99, bot_places=[])
        legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        legal = legal.reshape(2, riichi_py.ACTIONS).astype(bool)

        # Every one of Mortal's moves either names a legal move of ours or
        # says plainly that it names none. Nothing silently becomes an
        # index in the wrong space.
        for action in range(served.contract.answers):
            got = served.to_engine(action, legal[0])
            self.assertTrue(got == -1 or (0 <= got < riichi_py.ACTIONS), got)
            if got >= 0:
                self.assertTrue(legal[0][got], f"move {action} became an illegal {got}")

    def test_a_declaration_names_a_tile_the_rules_allow(self):
        """Riichi is two decisions, and the second is asked in its own
        space. A contract that forgot it would translate a tile index as
        though it were an ordinary discard."""
        import numpy as np

        from neural import zoo

        net = zoo.load_player(self.CHECKPOINT, "cpu")
        served = contract.serve(net, self.CHECKPOINT)
        legal = np.zeros(riichi_py.ACTIONS, dtype=bool)
        legal[zoo.RIICHI_DISCARD + 5] = True
        self.assertEqual(served.to_engine(5, legal, after_reach=True), zoo.RIICHI_DISCARD + 5)
        self.assertEqual(served.to_engine(6, legal, after_reach=True), -1)

if __name__ == "__main__":
    unittest.main()
