"""Re-heading one of our networks into Mortal's action space
(`neural.rehead`): one command line re-heads one network, its new head's
first weights, the teacher's sampled moves and the minibatches' order all
drawn from `--seed`, and a student taught further is dealt rounds it has
not learnt."""

import contextlib
import io
import json
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
    #: The seed of every round dealt, in the order they were.
    dealt: list[int] = []

    def __init__(self, games, seed):
        self.arena = self.native(games=games, seed=seed)
        self.games = games
        self.steps = 0
        self.dealt.append(seed)

    def __getattr__(self, name):
        return getattr(self.arena, name)

    def seats(self):
        return bytes([0xFF] * self.games) if self.steps >= 12 else self.arena.seats()

    def step(self, actions):
        self.arena.step(actions)
        self.steps += 1


def run(teacher: Path, out: Path, *options: str) -> None:
    """One re-heading on the processor, of a few decisions a round."""
    argv = ["rehead", "--teacher", str(teacher), "--out", str(out), "--games", "2", "--batch", "4",
            "--epochs", "1", *options]
    with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False), \
            patch.object(rehead.riichi_py, "Arena", BoundedArena), contextlib.redirect_stdout(io.StringIO()):
        rehead.main()


def a_teacher(root: Path) -> Path:
    teacher = PolicyValueNet(8, 1, MORTAL_PLANES)
    torch.save({"model": teacher.state_dict(), **teacher.payload_fields()}, root / "teacher.pt")
    return root / "teacher.pt"


class SeedTests(unittest.TestCase):
    def rehead(self, teacher: Path, out: Path, seed: int) -> dict:
        run(teacher, out, "--rounds", "1", "--seed", str(seed))
        return torch.load(out / "latest.pt", map_location="cpu", weights_only=True)["model"]

    def test_one_command_line_re_heads_one_network(self):
        """Two processes given the same --seed draw from torch's generators
        alike; each run here draws from them first, as a fresh process
        would, from wherever the one before left them."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            teacher = a_teacher(root)
            first = self.rehead(teacher, root / "first", 11)
            torch.rand(7)
            again = self.rehead(teacher, root / "again", 11)
            other = self.rehead(teacher, root / "other", 12)
        self.assertEqual(first.keys(), again.keys())
        for name in first:
            self.assertTrue(torch.equal(first[name], again[name]), name)
        self.assertFalse(all(torch.equal(first[name], other[name]) for name in first))


class FirstRoundTests(unittest.TestCase):
    """A round's deals come from `--seed` and the round's number. A student
    resumed at round 0 was dealt the rounds it had learnt a second time,
    and its log counted them twice."""

    def setUp(self):
        BoundedArena.dealt.clear()

    def rounds(self, out: Path) -> list[int]:
        return [json.loads(line)["round"] for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_a_student_taught_further_goes_on_from_the_round_it_says(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            teacher = a_teacher(root)
            run(teacher, root / "first", "--rounds", "2", "--seed", "11")
            self.assertEqual(BoundedArena.dealt, [11, 11 + 977])
            self.assertEqual(self.rounds(root / "first"), [0, 1])
            BoundedArena.dealt.clear()
            run(teacher, root / "more", "--rounds", "2", "--seed", "11",
                "--resume", str(root / "first" / "latest.pt"), "--first-round", "2")
            self.assertEqual(BoundedArena.dealt, [11 + 2 * 977, 11 + 3 * 977])
            self.assertEqual(self.rounds(root / "more"), [2, 3])

    def test_a_student_taught_further_without_its_round_is_refused_before_anything_is_dealt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            teacher = a_teacher(root)
            run(teacher, root / "first", "--rounds", "1", "--seed", "11")
            BoundedArena.dealt.clear()
            with self.assertRaisesRegex(ValueError, "--first-round"):
                run(teacher, root / "more", "--rounds", "1", "--seed", "11",
                    "--resume", str(root / "first" / "latest.pt"))
            self.assertEqual(BoundedArena.dealt, [])
            self.assertFalse((root / "more").exists())

    def test_a_round_before_the_first_is_refused_and_a_fresh_student_may_start_later(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            teacher = a_teacher(root)
            with self.assertRaisesRegex(ValueError, "first-round"):
                run(teacher, root / "before", "--rounds", "1", "--seed", "11", "--first-round=-1")
            self.assertEqual(BoundedArena.dealt, [])
            self.assertFalse((root / "before").exists())
            run(teacher, root / "later", "--rounds", "1", "--seed", "11", "--first-round", "5")
            self.assertEqual(BoundedArena.dealt, [11 + 5 * 977])
            self.assertEqual(self.rounds(root / "later"), [5])


if __name__ == "__main__":
    unittest.main()
