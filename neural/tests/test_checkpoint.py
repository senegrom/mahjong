"""A checkpoint that fails to be written must not destroy the last good one.

`torch.save` truncates its destination and then writes into it, so anything
that goes wrong in between leaves a file with the right name that cannot be
loaded — and the previous checkpoint has already gone, because it was that
file. A run can lose hours that way and only discover it at the next
resume.

    python -m unittest neural.tests.test_checkpoint -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from neural import checkpoint

PLACEMENT_VALUE = (1.5, 0.5, -0.5, -1.5)


class WritingIsAllOrNothing(unittest.TestCase):
    def setUp(self):
        self.room = tempfile.TemporaryDirectory()
        self.where = Path(self.room.name) / "latest.pt"
        checkpoint.publish({"generation": 1, "weights": torch.ones(4)}, self.where)

    def tearDown(self):
        self.room.cleanup()

    def test_a_checkpoint_written_normally_reads_back(self):
        self.assertTrue(checkpoint.loadable(self.where))
        self.assertEqual(torch.load(self.where, weights_only=False)["generation"], 1)

    def test_a_failure_while_writing_leaves_the_old_one_intact(self):
        def explode(*_args, **_kwargs):
            raise RuntimeError("the disk filled up")

        with mock.patch("neural.checkpoint.torch.save", side_effect=explode):
            with self.assertRaises(RuntimeError):
                checkpoint.publish({"generation": 2}, self.where)

        self.assertTrue(
            checkpoint.loadable(self.where),
            "the previous checkpoint was destroyed by a save that never finished",
        )
        self.assertEqual(torch.load(self.where, weights_only=False)["generation"], 1)

    def test_the_direct_way_does_destroy_it(self):
        """What the publisher is for, shown rather than asserted."""
        victim = Path(self.room.name) / "direct.pt"
        torch.save({"generation": 1}, victim)

        def explode(*_args, **_kwargs):
            raise RuntimeError("the disk filled up")

        with open(victim, "wb") as handle:
            with self.assertRaises(RuntimeError):
                with mock.patch("torch.save", side_effect=explode):
                    torch.save({"generation": 2}, handle)
        self.assertFalse(
            checkpoint.loadable(victim),
            "truncating and then failing is supposed to ruin the file",
        )

    def test_no_half_written_file_is_left_behind(self):
        def explode(*_args, **_kwargs):
            raise RuntimeError("the disk filled up")

        with mock.patch("neural.checkpoint.torch.save", side_effect=explode):
            with self.assertRaises(RuntimeError):
                checkpoint.publish({"generation": 2}, self.where)
        leftover = list(Path(self.room.name).glob("*.writing"))
        self.assertEqual(len(leftover), 1, "the partial write is kept out of the way")
        self.assertFalse(checkpoint.loadable(leftover[0]))

    def test_the_one_it_replaced_is_kept(self):
        checkpoint.publish({"generation": 2}, self.where)
        previous = self.where.with_name("latest.prev.pt")
        self.assertTrue(previous.exists(), "there is no way back one step")
        self.assertEqual(torch.load(previous, weights_only=False)["generation"], 1)
        self.assertEqual(torch.load(self.where, weights_only=False)["generation"], 2)

    def test_a_first_write_needs_nothing_to_replace(self):
        fresh = Path(self.room.name) / "new" / "latest.pt"
        checkpoint.publish({"generation": 1}, fresh)
        self.assertTrue(checkpoint.loadable(fresh))


class PlacementTiesAgreeAcrossTheLanguages(unittest.TestCase):
    """Python decides what a game paid and Rust's search decides what an
    imagined one would have. If they break ties differently then a searched
    continuation is valued against a different target from the one the
    value head was trained on, and nothing says so."""

    @staticmethod
    def python_places(scores):
        """What `selfplay.play` does."""
        return list((-np.asarray(scores)[None, :]).argsort(axis=1).argsort(axis=1)[0])

    @staticmethod
    def rust_places(scores):
        """`search.rs::placement_value`: behind everyone with more, and
        behind an equal score held by a lower index."""
        return [
            sum(
                1
                for other, their in enumerate(scores)
                if their > mine or (their == mine and other < player)
            )
            for player, mine in enumerate(scores)
        ]

    def test_every_shape_of_tie_agrees(self):
        for scores in (
            [40000, 30000, 20000, 10000],
            [35000, 35000, 20000, 10000],
            [40000, 25000, 25000, 10000],
            [30000, 30000, 30000, 10000],
            [25000, 25000, 25000, 25000],
            [30000, 30000, 20000, 20000],
            [10000, 20000, 30000, 40000],
            [0, 0, 0, 100000],
        ):
            mine = [int(place) for place in self.python_places(scores)]
            theirs = self.rust_places(scores)
            self.assertEqual(mine, theirs, f"{scores} placed differently")
            self.assertEqual(
                [PLACEMENT_VALUE[p] for p in mine],
                [PLACEMENT_VALUE[p] for p in theirs],
            )


if __name__ == "__main__":
    unittest.main()
