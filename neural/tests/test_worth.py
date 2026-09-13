"""The estimator that splits the worlds: on moves that are all the same,
it says so, where reading the recording off as it stands does not.

    python -m unittest neural.tests.test_worth -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural import worth


def written(folder: Path, per_world: np.ndarray, seed: int = 1) -> Path:
    """A recording's arrays, as `neural.searched --record` writes them,
    for the fields this estimator reads."""
    folder.mkdir(parents=True, exist_ok=True)
    rows, width, _worlds = per_world.shape
    with np.errstate(invalid="ignore"):
        values = np.nanmean(per_world, axis=2)
    np.save(folder / "per_world.npy", per_world.astype(np.float32))
    np.save(folder / "values.npy", values.astype(np.float32))
    np.save(folder / "candidates.npy", np.tile(np.arange(width), (rows, 1)))
    np.save(folder / "policy.npy", np.zeros(rows, dtype=np.int64))
    np.save(folder / "search.npy", np.zeros(rows, dtype=np.int64))
    np.save(folder / "game.npy", np.arange(rows) % 10)
    np.save(folder / "chair.npy", np.zeros(rows, dtype=np.int64))
    np.save(folder / "step.npy", np.arange(rows))
    (folder / "meta.json").write_text(json.dumps({"worlds": per_world.shape[2], "seed": seed}),
                                      encoding="utf-8")
    return folder


class WorthTests(unittest.TestCase):
    def test_moves_that_are_all_the_same_are_worth_nothing_and_read_as_worth_plenty(self):
        """Four identical moves, valued over sixteen noisy worlds. Read as
        the recording stands, the best of them beats the policy's by a
        third of a unit; decided on half the worlds and scored on the
        other, by nothing."""
        drawer = np.random.default_rng(4)
        per_world = drawer.normal(0.0, 1.0, size=(400, 4, 16))
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            said = worth.measure(written(Path(folder) / "rec", per_world))
        self.assertGreater(said["naive_gap"], 0.2, "the winner's curse, read off as it stands")
        self.assertLess(abs(said["by_best"]["a_decision"]), 4 * said["by_best"]["standard_error"],
                        "taking the best average gains nothing here, and the split says so")
        self.assertLess(abs(said["by_margin"]["a_decision"]), 4 * said["by_margin"]["standard_error"],
                        "nor does the margin")
        # The margin still fires on a tenth of them: three candidates
        # tested at two standard errors, on eight worlds apiece, is about
        # a tenth of decisions by chance alone. It costs nothing -- a
        # move picked out of noise is worth what the policy's was, which
        # is what the line above says -- but it is why the rate the
        # search reports is not a count of improvements found.
        self.assertLess(said["fired"], 0.15)
        self.assertGreater(said["fired"], 0.05)

    def test_a_move_that_really_is_better_is_found_and_credited(self):
        """One candidate a quarter of a unit better than the rest, in the
        same noise: the margin fires on it and the held-back worlds credit
        it with about what it is worth."""
        drawer = np.random.default_rng(5)
        per_world = drawer.normal(0.0, 1.0, size=(400, 4, 32))
        per_world[:, 2, :] += 1.0
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            said = worth.measure(written(Path(folder) / "rec", per_world))
        self.assertGreater(said["fired"], 0.5, "a whole unit over the noise is found")
        self.assertGreater(said["by_margin"]["standard_errors"], 4)
        self.assertGreater(said["by_margin"]["a_decision"], 0.4)
        self.assertLessEqual(said["by_margin"]["a_decision"], 1.2,
                             "and credited with about what it is worth, not more")

    def test_a_world_a_candidate_never_reached_is_not_counted(self):
        """A candidate the search could not try in a world is NaN there,
        and pairs only over the worlds both were tried in."""
        drawer = np.random.default_rng(6)
        per_world = drawer.normal(0.0, 1.0, size=(50, 3, 16))
        per_world[:, 1, 8:] = np.nan
        per_world[:, 2, :] = np.nan
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            said = worth.measure(written(Path(folder) / "rec", per_world))
        self.assertGreater(said["by_margin"]["rows"], 0)
        self.assertIsNotNone(said["fired"])
        edge, error = worth.edge_and_error(per_world[0, 1], per_world[0, 0])
        self.assertTrue(np.isfinite(edge) and np.isfinite(error))
        self.assertEqual(worth.edge_and_error(per_world[0, 2], per_world[0, 0])[1], float("inf"))


if __name__ == "__main__":
    unittest.main()
