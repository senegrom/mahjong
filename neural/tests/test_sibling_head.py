"""The head that ranks siblings learns from a recording and is measured
the way it will be used.

    python -m unittest neural.tests.test_sibling_head -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from neural import contract, searched, sibling_head
from neural.model import MORTAL_PLANES, PolicyValueNet
from neural.recordings import resolve_recording


class SiblingHeadTests(unittest.TestCase):
    def test_a_fresh_head_ranks_every_move_even(self):
        head = sibling_head.Ranker(8)
        scores = head(torch.randn(3, 8, 34), torch.randn(3, 8))
        self.assertEqual(scores.shape, (3, sibling_head.ACTIONS))
        self.assertTrue(torch.all(scores == 0), "a head that has learned nothing scores every move zero")

    def test_targets_are_differences_from_the_mean_over_the_candidates(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recording = searched.Recording()
            from neural.observe import Planes

            root = Planes(np.array([0, 1], dtype=np.int64), np.zeros(1, dtype=np.uint16), np.ones(1, dtype=np.float16))
            recording.add(root, [(3, 1.0, [1.0, 1.0]), (7, -0.5, [-1.0, 0.0]), (9, 0.4, [0.4, 0.4])], 3, 3, 0, 0, 1)
            recording.add(root, [(1, 0.2, [0.2]), (2, -0.2, [-0.2])], 1, 2, 0, 0, 2)
            recording.save(folder, {"note": "test"})
            recorded = sibling_head.Recorded(Path(folder))
            targets = recorded.targets
            np.testing.assert_allclose(targets[0], [1.0 - 0.3, -0.5 - 0.3, 0.4 - 0.3], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(targets[1][:2], [0.2, -0.2], rtol=1e-5, atol=1e-6)
            self.assertTrue(np.isnan(targets[1][2]), "a missing candidate is no target")
            # The first row's worlds disagree about the second candidate
            # and agree about the first. One world cannot estimate its own
            # uncertainty, so the second row has zero precision, not maximum.
            precision = recorded.precision(floor=0.1)
            self.assertTrue(np.isfinite(precision[0][:3]).all())
            self.assertTrue(np.isnan(precision[0][3:]).all() if precision.shape[1] > 3 else True)
            self.assertGreater(precision[0][0], precision[0][1], "a candidate the worlds agreed on is trusted more")
            np.testing.assert_allclose(precision[1][:2], [0.0, 0.0], rtol=1e-4)
            self.assertTrue(np.isnan(precision[1][2]))

    def test_a_recording_is_kept_as_the_search_runs_and_read_back_whole(self):
        """Written partially every interval while the search runs, marked
        incomplete, and whole at the end; each write is swapped in at once,
        so a reader never sees half of one. The policy's confidence at each
        root comes along, gathered recordings keep it, and a head keeps the
        gate its recordings were made under."""
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(5)
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            served = contract.serve(net)
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                folder = Path(tmp) / "rec"
                meta = {"note": "test", "sure": 0.9, "seed": 4, "games": 1}
                recording = searched.Recording(folder, meta, every=0.0)
                searched.play(net, 1, 4, 0, 2, 2, 0.0, pool=1, device="cpu", served=served, recording=recording)
                self.assertGreater(len(recording), 0)
                partial = json.loads((resolve_recording(folder) / "meta.json").read_text(encoding="utf-8"))
                self.assertFalse(partial["complete"], "written while the search ran")
                self.assertEqual(partial["rows"], len(recording))
                self.assertFalse(folder.with_name("rec.writing").exists())
                self.assertFalse(folder.with_name("rec.previous").exists())
                recording.save(folder, meta)
                recorded = sibling_head.Recorded(folder)
                self.assertTrue(recorded.meta["complete"])
                self.assertEqual(len(recorded), len(recording))
                self.assertEqual(len(recorded.sure), len(recorded))
                self.assertTrue(np.all((recorded.sure > 0) & (recorded.sure <= 1)))
                joined = sibling_head.gathered([recorded, recorded])
                self.assertEqual(len(joined.sure), 2 * len(recorded))
                self.assertEqual(joined.meta["sure"], 0.9)
                # Two chairs of the same deals keep their game numbers, so a
                # held-out deal is held out in both; other deals follow on.
                n = len(recorded)
                np.testing.assert_array_equal(joined.game[:n], joined.game[n:])
                other = sibling_head.Recorded(folder)
                other.meta = {**other.meta, "seed": 999}
                apart = sibling_head.gathered([recorded, other])
                self.assertEqual(int(apart.game[n:].min()), int(recorded.game.max()) + 1)
                training, held = apart.split()
                self.assertTrue(set(apart.game[held]).isdisjoint(set(apart.game[training])))
                head = sibling_head.Ranker(net.channels)
                sibling_head.save(head, Path(tmp) / "head.pt", {"sure": recorded.meta["sure"]})
                _loaded, meta_back = sibling_head.load(Path(tmp) / "head.pt", "cpu")
                self.assertEqual(meta_back["sure"], 0.9)
        finally:
            torch.set_num_threads(previous)

    def test_the_head_can_learn_which_move_the_search_made(self):
        """The other target: not what the worlds averaged to, which
        carries the winner's curse, but which of the candidates the search
        took under its margin. Learned as a choice, and measured against
        naming one at random, which is what it has to beat where the
        search overrode."""
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(4)
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            served = contract.serve(net)
            recording = searched.Recording()
            searched.play(net, 2, 4, 0, 3, 3, 0.0, pool=1, device="cpu", served=served, recording=recording)
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
                recording.save(folder, {"note": "test", "seed": 4, "games": 2})
                recorded = sibling_head.Recorded(Path(folder))
                took = sibling_head.chosen_by_search(recorded)
                self.assertTrue((took >= 0).all(), "the search took one of the candidates it was given")
                np.testing.assert_array_equal(
                    recorded.candidates[np.arange(len(took)), took], recorded.search
                )
                training, held = recorded.split()
                head, history = sibling_head.train(
                    recorded, net, epochs=6, lr=3e-3, batch=32, device="cpu", target="decision"
                )
                on_training = sibling_head.measure(head, net, recorded, training, "cpu")
                del held
        finally:
            torch.set_num_threads(previous)
        self.assertEqual(len(history), 6)
        # It is learning the choice: the cross-entropy over the candidates
        # falls. Whether it generalises is a question for a recording of
        # real size, not for two games.
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
        self.assertIn("keeping_the_policys_move_would_score", on_training)
        overrode = on_training["where_the_search_overrode"]
        self.assertGreater(overrode["rows"], 0)
        self.assertIsNotNone(overrode["by_chance"])
        self.assertLessEqual(overrode["by_chance"], 0.5)

    def test_the_head_learns_a_recording_and_is_measured_against_the_rollouts(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(4)
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            served = contract.serve(net)
            recording = searched.Recording()
            searched.play(net, 2, 4, 0, 3, 3, 0.0, pool=1, device="cpu", served=served, recording=recording)
            self.assertGreater(len(recording), 20)
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
                recording.save(folder, {"note": "test"})
                recorded = sibling_head.Recorded(Path(folder))
                training, held = recorded.split()
                self.assertGreater(len(held), 0, "some game is held out")
                self.assertGreater(len(training), 0, "some game is trained on")
                self.assertEqual(len(training) + len(held), len(recorded))
                head, history = sibling_head.train(recorded, net, epochs=4, lr=3e-3, batch=32, device="cpu")
                self.assertEqual(len(history), 4)
                # The pre-duel reading: how often a player taking the head's
                # favourite past each margin would override, and what the
                # rollouts say it would gain; a wider margin overrides less.
                by_margin = history[-1]["held_out"]["by_margin"]
                self.assertEqual(list(by_margin), ["0", "0.02", "0.05", "0.1", "0.2"])
                self.assertGreaterEqual(by_margin["0"]["overrides"], by_margin["0.2"]["overrides"])
                for reading in by_margin.values():
                    self.assertAlmostEqual(
                        reading["gain_per_decision"], reading["overrides"] * reading["gain_when_overriding"], places=4
                    )
                weighted, weighed = sibling_head.train(
                    recorded, net, epochs=1, lr=3e-3, batch=32, device="cpu", weighted=True
                )
                self.assertEqual(len(weighed), 1)
                self.assertTrue(np.isfinite(weighed[0]["train_loss"]))
                self.assertLess(
                    history[-1]["train_loss"], history[0]["train_loss"],
                    "the head fits the differences it is shown",
                )
                measured = history[-1]["held_out"]
                for key in ("head_agrees_with_rollouts", "policy_agrees_with_rollouts",
                            "worth_of_heads_pick", "worth_of_policys_pick", "worth_of_best"):
                    self.assertIn(key, measured)
                self.assertGreaterEqual(measured["worth_of_best"], measured["worth_of_policys_pick"])
                out = Path(folder) / "head.pt"
                sibling_head.save(head, out, {"note": "test"})
                loaded, meta = sibling_head.load(out)
                self.assertEqual(meta["note"], "test")
                rows = np.arange(min(5, len(recorded)))
                before = sibling_head.scored(head, net, recorded.roots, rows, "cpu")
                after = sibling_head.scored(loaded, net, recorded.roots, rows, "cpu")
                self.assertTrue(torch.allclose(before, after))
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
