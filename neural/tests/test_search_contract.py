"""What a search may assume about the network it is given, and what it must refuse.

A network of the engine's own lineage is served the engine's planes. One
on Mortal's is served the follower's at the root and a copied, advanced
state at every leaf (see `neural.contract`). What nothing can serve is
refused before an arena is built, by name, and never approximated by
padding or relabelling planes.

    python -m unittest neural.tests.test_search_contract -v
"""
import unittest
from unittest.mock import patch

import numpy as np
import riichi_py
import torch

from neural import contract, searched
from neural.model import MORTAL_PLANES, PolicyValueNet


class SearchContractTests(unittest.TestCase):
    def test_current_network_only_baselines_complete_for_both_action_layouts(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for actions in (78, 46):
                with self.subTest(actions=actions):
                    torch.manual_seed(12)
                    net = PolicyValueNet(8, 1, actions=actions)
                    scores, tally = searched.play(net, 1, 1, None, 1, 1, 0.0, device="cpu")
                    self.assertEqual(scores.shape, (1, 4))
                    self.assertTrue(np.isfinite(scores).all())
                    self.assertEqual(tally, (0, 0))
        finally:
            torch.set_num_threads(previous)

    def test_engine_planes_with_mortal_moves_are_refused_before_any_arena(self):
        """A layout nobody trained: the engine writes ninety-seven planes
        and names seventy-eight moves, and nothing translates forty-six
        back from leaves that carry no Mortal history."""
        net = PolicyValueNet(8, 1, planes=riichi_py.PLANES, actions=46)
        self.assertEqual(net.kind, "engine")
        with patch.object(riichi_py, "Arena", side_effect=AssertionError("too late")):
            with self.assertRaisesRegex(searched.UnsupportedSearchLayout, "Mortal event history"):
                searched.play(net, 1, 1, 0, 1, 1, 0.0, device="cpu")
            with self.assertRaisesRegex(searched.UnsupportedSearchLayout, "Mortal event history"):
                searched.search_with_value_head(
                    net, None, [], [], worlds=1, candidates=1, margin=0.0, hurried=True,
                    device="cpu",
                )

    def test_the_engine_planes_lookahead_needs_the_engines_own_layout(self):
        """`play_lookahead` asks the engine for its own observations at
        every decision inside the search; a network on Mortal's planes is
        served by `MortalServed.play_lookahead` instead, and the layout
        nobody trained is refused either way."""
        net = PolicyValueNet(8, 1, planes=riichi_py.PLANES, actions=46)
        with patch.object(riichi_py, "Arena", side_effect=AssertionError("too late")):
            with self.assertRaisesRegex(searched.UnsupportedSearchLayout, "Mortal event history"):
                searched.play_lookahead(net, None, device="cpu")
            with self.assertRaises(searched.UnsupportedSearchLayout):
                searched.play(net, 1, 1, 0, 1, 1, 0.0, device="cpu", played_by="network")
        mortal = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
        with self.assertRaisesRegex(searched.UnsupportedSearchLayout, "Mortal event history"):
            searched.play_lookahead(mortal, None, device="cpu")

    def test_the_network_moves_every_seat_inside_the_search_on_mortals_planes(self):
        """The strong searchers' way: the seats between the candidate and
        the leaf are played by the network itself, here through a copy of
        each seat's state kept in step with what its world invents; every
        such decision is answered in Mortal's moves and translated back
        under the engine's legality, hand boundaries included."""
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            served = contract.serve(net)
            served.count_crossings = True
            scores, tally = searched.play(
                net, 2, 8, 0, 2, 2, 0.0, pool=1, device="cpu", played_by="network", depth=-1,
                served=served,
            )
        finally:
            torch.set_num_threads(previous)
        self.assertEqual(scores.shape, (2, 4))
        self.assertGreater(tally[0], 0, "nothing was searched")
        self.assertGreater(served.crossed, 0, "no imagined world played into the next hand")

    def test_the_club_played_search_serves_mortals_planes(self):
        """The reason the contract exists: the current lineage reads
        Mortal's planes, and the search that moves the other seats with the
        club heuristic builds them at the root and at every leaf."""
        net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
        self.assertEqual(net.kind, "mortal")
        served = contract.serve(net)
        self.assertIsInstance(served, contract.MortalServed)
        self.assertEqual(served.contract.reads, "mortal")
        self.assertEqual(served.contract.answers, 46)
        self.assertFalse(served.contract.speaks_our_moves)

    def test_a_world_that_deals_on_is_served_across_the_hand_boundary(self):
        """Worlds that end the hand inside the search used to reach the
        network as blank positions, which it valued at +0.49 against a
        real average near zero, and the search preferred whatever ended
        the hand. Now the copy is told how the hand ended and dealt the
        next; the leaf is served, and counted here."""
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            served = contract.serve(net)
            served.count_crossings = True
            scores, tally = searched.play(
                net, 2, 5, 0, 2, 2, 0.0, pool=1, device="cpu", served=served
            )
        finally:
            torch.set_num_threads(previous)
        self.assertEqual(scores.shape, (2, 4))
        self.assertGreater(tally[0], 0, "nothing was searched")
        self.assertGreater(
            served.crossed, 0,
            "no imagined world played into the next hand, so the boundary went untested",
        )

    def test_a_recording_keeps_every_searched_decision_with_its_worlds(self):
        """The root as the network read it, the candidates, what each came
        to in every world, and both choices, one row a searched decision;
        written and read back whole."""
        import tempfile
        from pathlib import Path

        from neural.observe import Planes

        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            served = contract.serve(net)
            recording = searched.Recording()
            scores, tally = searched.play(
                net, 1, 4, 0, 3, 3, 0.0, pool=1, device="cpu", served=served, recording=recording,
            )
        finally:
            torch.set_num_threads(previous)
        self.assertGreater(len(recording), 0, "nothing was recorded")
        # The tally counts the decisions with something to compare, and
        # every one of them is a row: the other seats' single choices and
        # the forced moves cost no worlds and are in neither.
        self.assertEqual(len(recording), tally[0])
        self.assertEqual(len(recording.sure), len(recording))
        self.assertTrue(all(0.0 < sure <= 1.0 for sure in recording.sure), "a probability a row")
        # The engine's mask at each root comes along, so a policy taught
        # from the rows can be asked, and leashed, under it.
        self.assertEqual(len(recording.legal), len(recording))
        for candidates, legal in zip(recording.candidates, recording.legal):
            self.assertEqual(legal.shape, (riichi_py.ACTIONS,))
            self.assertTrue(all(legal[c] for c in candidates), "every candidate was legal")
        for candidates, values, worlds, policy, search in zip(
            recording.candidates, recording.values, recording.per_world, recording.policy, recording.search
        ):
            self.assertGreaterEqual(len(candidates), 2)
            self.assertEqual(len(candidates), len(values))
            self.assertEqual(candidates[0], policy, "the first candidate is the policy's choice")
            self.assertIn(search, candidates, "the search chose among the candidates")
            self.assertTrue(all(len(w) == 3 for w in worlds), "a worth for each of the three worlds")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recording.save(folder, {"note": "test"})
            roots = Planes.load(Path(folder), "root", mmap=False)
            self.assertEqual(len(roots), len(recording))
            self.assertEqual(roots.dense("cpu").shape[1], MORTAL_PLANES)
            per_world = np.load(Path(folder) / "per_world.npy")
            self.assertEqual(per_world.shape[0], len(recording))
            self.assertEqual(per_world.shape[2], 3)
            legal = np.load(Path(folder) / "legal.npy")
            self.assertEqual(legal.shape, (len(recording), riichi_py.ACTIONS))
            self.assertEqual(legal.dtype, np.bool_)

    def test_a_sure_policy_is_taken_at_its_word_and_costs_no_worlds(self):
        """Gated at zero the policy is sure of everything: nothing is
        searched, no world is played, and the games are the ones a search
        that never overrides plays; the tally and the health say so."""
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(9)
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            served = contract.serve(net)
            health: dict = {}
            never, tally_never = searched.play(
                net, 2, 4, 0, 2, 2, 0.0, pool=1, device="cpu", served=served, sure=0.0, health=health,
            )
            always, tally_always = searched.play(
                net, 2, 4, 0, 2, 2, 1e9, pool=1, device="cpu", served=served,
            )
        finally:
            torch.set_num_threads(previous)
        self.assertEqual(tally_never, (0, 0), "a sure policy is never asked")
        self.assertGreater(health["own"], 0)
        self.assertEqual(health["sure"], health["own"], "every own decision was taken sure")
        self.assertGreater(tally_always[0], 0, "ungated, the hesitant decisions were searched")
        self.assertEqual(tally_always[1], 0, "nothing clears a margin that large")
        np.testing.assert_array_equal(never, always)

    def test_the_exception_is_one_class_wherever_it_is_raised(self):
        self.assertIs(searched.UnsupportedSearchLayout, contract.UnsupportedSearchLayout)
        self.assertTrue(issubclass(searched.UnsupportedSearchLayout, ValueError))

    def test_legacy_native_search_contract_is_still_supported(self):
        net = PolicyValueNet(8, 1, planes=riichi_py.PLANES, actions=riichi_py.ACTIONS)
        searched.require_native_search(net)
        self.assertIsInstance(contract.serve(net), contract.EngineServed)


if __name__ == "__main__":
    unittest.main()
