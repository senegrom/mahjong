"""Re-heading one of our networks into Mortal's action space
(`neural.rehead`): one command line re-heads one network, its new head's
first weights, the teacher's sampled moves and the minibatches' order all
drawn from `--seed`."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import riichi_py
import torch

from neural import rehead
from neural.model import MORTAL_PLANES, PolicyValueNet


class BoundedArena:
    """A round of a few real decisions, without playing a match."""

    native = riichi_py.Arena

    def __init__(self, games, seed):
        self.arena = self.native(games=games, seed=seed)
        self.games = games
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.arena, name)

    def seats(self):
        return bytes([0xFF] * self.games) if self.steps >= 12 else self.arena.seats()

    def step(self, actions):
        self.arena.step(actions)
        self.steps += 1


class SeedTests(unittest.TestCase):
    def rehead(self, teacher: Path, out: Path, seed: int) -> dict:
        argv = ["rehead", "--teacher", str(teacher), "--out", str(out), "--rounds", "1", "--games", "2",
                "--batch", "4", "--epochs", "1", "--seed", str(seed)]
        with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(rehead.riichi_py, "Arena", BoundedArena), contextlib.redirect_stdout(io.StringIO()):
            rehead.main()
        return torch.load(out / "latest.pt", map_location="cpu", weights_only=True)["model"]

    def test_one_command_line_re_heads_one_network(self):
        """Two processes given the same --seed draw from torch's generators
        alike; each run here draws from them first, as a fresh process
        would, from wherever the one before left them."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            teacher = PolicyValueNet(8, 1, MORTAL_PLANES)
            torch.save({"model": teacher.state_dict(), **teacher.payload_fields()}, root / "teacher.pt")
            first = self.rehead(root / "teacher.pt", root / "first", 11)
            torch.rand(7)
            again = self.rehead(root / "teacher.pt", root / "again", 11)
            other = self.rehead(root / "teacher.pt", root / "other", 12)
        self.assertEqual(first.keys(), again.keys())
        for name in first:
            self.assertTrue(torch.equal(first[name], again[name]), name)
        self.assertFalse(all(torch.equal(first[name], other[name]) for name in first))


if __name__ == "__main__":
    unittest.main()
