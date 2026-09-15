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


def a_round(games: int = 2, seed: int = 11, on_mortal_planes: bool = False):
    """A cheap round: every seat takes the first legal move. On Mortal's
    planes it keeps the observations a head reads; on the engine's it
    keeps none, which is what training does."""
    from neural.tests.test_selfplay_contract import FirstLegal

    class OnMortalPlanes(FirstLegal):
        kind = "mortal"

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return selfplay.play(OnMortalPlanes() if on_mortal_planes else FirstLegal(),
                             games=games, seed=seed, device="cpu")
    finally:
        torch.set_num_threads(previous)


class SelfPlayLabelsTests(unittest.TestCase):
    def test_a_round_keeps_the_placement_apart_from_the_points(self):
        """The return is the hand's points and the placement together, and
        a boundary needs the second alone; self-play now keeps both."""
        batch = a_round()
        self.assertEqual(len(batch.placements), batch.decisions)
        # And each decision knows its game, so a head can hold whole games out.
        self.assertEqual(len(batch.game_of), batch.decisions)
        self.assertEqual(sorted(set(batch.game_of.tolist())), [0, 1])
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

    def test_a_saved_round_reads_back_with_its_games_kept_apart(self):
        """What `selfplay.save_round` writes, `placement.load_rounds` reads:
        the planes, the placements and the games, and a second round's
        games are numbered after the first's so a hold-out by game holds."""
        batch = a_round(on_mortal_planes=True)
        self.assertEqual(len(batch.observations), batch.decisions)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            path = Path(folder) / "round.pt"
            selfplay.save_round(batch, path, {"seed": 11})
            with self.assertRaisesRegex(ValueError, "duplicate or overlapping"):
                placement.load_rounds([path, path])
            second = Path(folder) / "other.pt"
            other = a_round(seed=14, on_mortal_planes=True)
            selfplay.save_round(other, second)
            positions = placement.load_rounds([path, second])
            self.assertEqual(len(positions), batch.decisions + other.decisions)
            self.assertEqual(int(positions.games.max()), 15)
            np.testing.assert_array_equal(positions.placements[: batch.decisions], batch.placements.numpy())
            np.testing.assert_array_equal(positions.games[batch.decisions :], other.game_of.numpy() + 14)
            training, held = positions.split(held_out_every=2)
            self.assertTrue(set(positions.games[training]).isdisjoint(set(positions.games[held])))
            self.assertEqual(len(training) + len(held), len(positions))
            # The planes came through: the same rows, dense, as the round had.
            wanted = batch.observations.rows(np.arange(3)).dense("cpu")
            self.assertTrue(torch.equal(positions.planes.rows(np.arange(3)).dense("cpu"), wanted))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["round_version"] = selfplay.ROUND_VERSION + 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "round version"):
                placement.load_rounds([path])

    def test_a_round_without_planes_cannot_be_saved_for_a_head(self):
        """A round played on the engine's planes kept no observations, and
        a file with placements and no positions would be a trap."""
        batch = a_round()
        with self.assertRaisesRegex(ValueError, "no planes"):
            selfplay.save_round(batch, Path(tempfile.gettempdir()) / "never-written.pt")


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

    def test_a_head_says_whose_features_it_read_and_refuses_another_network(self):
        torch.manual_seed(7)
        net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
        first = placement.fingerprint(net)
        self.assertEqual(first, placement.fingerprint(net), "the same network reads the same")
        other = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
        self.assertNotEqual(first, placement.fingerprint(other))
        placement.require_head_for({"features": first}, net)
        with self.assertRaisesRegex(ValueError, "fitted to a network"):
            placement.require_head_for({"features": first}, other)
        with self.assertRaisesRegex(ValueError, "does not say"):
            placement.require_head_for({}, net)

    def test_a_round_without_the_labels_is_refused_rather_than_guessed(self):
        class Older:
            observations = Planes.empty()

        with self.assertRaisesRegex(ValueError, "placement part"):
            placement.Positions.of(Older())


if __name__ == "__main__":
    unittest.main()
