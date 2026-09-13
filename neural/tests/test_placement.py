"""The head that judges a position by where it leads in the standings:
self-play labels it, it learns something a constant cannot, and it refuses
to be confused with a head trained on anything else.

    python -m unittest neural.tests.test_placement -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from neural import placement, selfplay
from neural.model import MORTAL_PLANES, PolicyValueNet
from neural.observe import Planes


class SelfPlayLabelsTests(unittest.TestCase):
    def test_a_round_keeps_the_placement_apart_from_the_points(self):
        """The return is the hand's points and the placement together, and
        a boundary needs the second alone; self-play now keeps both."""
        from neural.tests.test_selfplay_contract import FirstLegal

        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            batch = selfplay.play(FirstLegal(), games=2, seed=11, device="cpu")
        finally:
            torch.set_num_threads(previous)
        self.assertEqual(len(batch.placements), batch.decisions)
        # Every value is one of the four the placement bonus can take, or
        # a mean of them where a game was shared.
        allowed = set(float(value) for value in selfplay.PLACEMENT_VALUE)
        seen = set(float(value) for value in batch.placements)
        self.assertTrue(seen)
        self.assertTrue(
            all(min(allowed) <= value <= max(allowed) for value in seen),
            f"placements outside the bonus range: {sorted(seen)}",
        )
        # And they are part of the return, not something else: subtracting
        # them leaves the hand points, which are small.
        points = batch.returns - batch.placements
        self.assertLess(float(points.abs().max()), 25.0)


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous)

    def a_few_positions(self, rows: int = 64, channels: int = 8):
        """Positions whose planes carry the answer in a single row, so a
        head that can read anything at all can read these."""
        drawer = np.random.default_rng(2)
        wanted = drawer.choice([-1.5, -0.5, 0.5, 1.5], size=rows).astype(np.float32)
        indptr = np.arange(rows + 1, dtype=np.int64)
        indices = np.zeros(rows, dtype=np.uint16)
        values = ((wanted + 1.5) / 3.0).astype(np.float16)
        return placement.Positions(Planes(indptr, indices, values), wanted,
                                   games=np.arange(rows) % 8)

    def test_a_fresh_head_says_nothing_and_a_trained_one_reads_the_standings(self):
        torch.manual_seed(7)
        net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
        positions = self.a_few_positions()
        fresh = placement.Judge(net.channels)
        planes = positions.planes.rows(np.arange(4)).dense("cpu")
        features, pooled = placement.features_of(net, planes)
        self.assertTrue(torch.all(fresh(features, pooled) == 0),
                        "a head that has learned nothing judges every position the same")
        head, history = placement.train(positions, net, epochs=30, lr=3e-2, batch=16, device="cpu")
        training, _held = positions.split()
        said = placement.measure(head, net, positions, training, "cpu")
        self.assertEqual(len(history), 30)
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        self.assertGreater(said["explained"], 0.2,
                           "the head reads more of the standings than a constant does")

    def test_a_head_is_refused_when_it_was_trained_against_another_target(self):
        torch.manual_seed(7)
        net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
        head = placement.Judge(net.channels)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            path = Path(folder) / "judge.pt"
            placement.save(head, path, {"note": "test"})
            again, meta = placement.load(path, "cpu")
            self.assertEqual(meta["note"], "test")
            self.assertEqual(
                placement.measure(again, net, self.a_few_positions(), np.arange(4), "cpu")["rows"], 4
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["placement_version"] = placement.PLACEMENT_VERSION + 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "placement version"):
                placement.load(path, "cpu")

    def test_a_round_without_the_labels_is_refused_rather_than_guessed(self):
        class Older:
            observations = Planes.empty()

        with self.assertRaisesRegex(ValueError, "placement part"):
            placement.Positions.of(Older())


if __name__ == "__main__":
    unittest.main()
