"""The placement reward's pre-test: its measures of agreement, its verdict,
and one small run of the whole of it, end to end, on the processor.

    python -m unittest neural.tests.test_placement_pretest -v
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from neural import place_model, placement_pretest, train_combined
from neural.scoreboard import Ledger
from neural.tests.test_resumed_checkpoints import flags, mortal_checkpoint, run


class Measures(unittest.TestCase):
    def test_cosine_overlap_and_correlation(self):
        one = torch.tensor([1.0, 0.0, 1.0])
        self.assertAlmostEqual(placement_pretest.cosine(one, one), 1.0)
        self.assertAlmostEqual(placement_pretest.cosine(one, -one), -1.0)
        self.assertEqual(placement_pretest.cosine(one, torch.zeros(3)), 0.0)
        flips = np.array([True, True, False, False])
        self.assertAlmostEqual(placement_pretest.ochiai(flips, flips), 1.0)
        self.assertAlmostEqual(placement_pretest.ochiai(flips, np.array([True, False, True, True])),
                               1 / math.sqrt(2 * 3))
        self.assertIsNone(placement_pretest.ochiai(flips, np.zeros(4, dtype=bool)))
        self.assertIsNone(placement_pretest.correlation(np.ones(4), np.arange(4.0)))
        self.assertAlmostEqual(placement_pretest.correlation(np.arange(4.0), np.arange(4.0)), 1.0)

    def arms(self, today: tuple, chosen: tuple) -> dict:
        def arm(cosine, flips):
            return {"agreement": {"cosine": {"above_chance": cosine}, "flips": {"above_chance": flips}}}
        return {"today": arm(*today), "alpha 0": arm(*chosen)}

    def test_the_verdict_reads_both_measures_against_today(self):
        verdict = placement_pretest.agreement_verdict
        self.assertEqual(verdict(self.arms((0.02, 0.1), (0.04, 0.2)), "today", "alpha 0")["verdict"], "go")
        self.assertEqual(verdict(self.arms((0.02, 0.1), (0.025, 0.12)), "today", "alpha 0")["verdict"], "kill")
        split = verdict(self.arms((0.02, 0.1), (0.04, 0.1)), "today", "alpha 0")
        self.assertTrue(split["verdict"].startswith("split"))
        self.assertEqual(split["cosine"]["ratio"], 2.0)
        unread = verdict(self.arms((0.02, None), (0.04, None)), "today", "alpha 0")
        self.assertEqual(unread["verdict"], "go")
        self.assertIsNone(unread["flips"]["passed"])
        nothing = verdict(self.arms((-0.01, 0.0), (0.01, 0.05)), "today", "alpha 0")
        self.assertEqual(nothing["verdict"], "go")
        self.assertEqual(nothing["cosine"]["why"], "today's return agrees no better than chance")


class Gradients(unittest.TestCase):
    """The halves' gradients read agreement where there is some and none
    where there is none: advantages that favour the same moves everywhere
    give two halves of the decisions steps that point the same way, and the
    same advantages shuffled among the decisions give steps that do not."""

    def test_a_planted_signal_agrees_with_itself_and_its_shuffle_does_not(self):
        from types import SimpleNamespace

        from neural import combined
        from neural.observe import PLANES, POSITIONS, Planes

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            torch.manual_seed(17)
            options = flags(train_combined, root, resume=False)
            net = combined.build(options[1], options[3], "cpu").eval()
        rng = np.random.default_rng(3)
        n, per_row = 768, 300
        indices = np.sort(rng.choice(PLANES * POSITIONS, size=(n, per_row)), axis=1).astype(np.uint16)
        planes = Planes(np.arange(n + 1, dtype=np.int64) * per_row, indices.reshape(-1),
                        rng.random(n * per_row).astype(np.float16))
        legal = torch.ones(n, 46, dtype=torch.bool)
        with torch.no_grad():
            logits = net.everything(planes.dense("cpu"), legal)[0]
        actions = torch.distributions.Categorical(logits=logits).sample()
        rollout = SimpleNamespace(observations=planes, legal=legal, actions=actions, decisions=n)
        favoured = (actions < 23).float()
        signal = (favoured - favoured.mean()) / favoured.std()
        shuffled = signal[torch.from_numpy(rng.permutation(n))]
        half = torch.from_numpy(rng.permutation(n) < n // 2)
        weights = torch.stack([signal, signal * half, shuffled, shuffled * half])
        sums, how = placement_pretest.head_gradients(
            net, rollout, weights, SimpleNamespace(amp=False, chunk=128), "cpu")
        self.assertIn(how, ("batched", "one at a time"))

        def agreement(whole, first):
            return placement_pretest.cosine(sums[first], sums[whole] - sums[first])

        planted, chance = agreement(0, 1), agreement(2, 3)
        self.assertGreater(planted, 0.3)
        self.assertLess(abs(chance), planted / 3)
        # The sums are linear in the weights: one call of each half alone
        # gives the halves the whole's sum splits into.
        alone, _how = placement_pretest.head_gradients(
            net, rollout, torch.stack([signal * ~half]), SimpleNamespace(amp=False, chunk=128), "cpu")
        torch.testing.assert_close(alone[0], sums[0] - sums[1], rtol=1e-4, atol=1e-6)


class Seeds(unittest.TestCase):
    def test_a_seed_that_sits_a_member_out_is_refused_first(self):
        """The Club is in about three rounds' pools in four, drawn from the
        round's seed; 4321's round leaves it out, so U could not learn its
        seats, and the pre-test refuses before it loads anything (the
        checkpoint named does not exist)."""
        from neural.population import Population

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            out = Path(folder) / "out"
            seated = [str(Path(folder) / "zoo--mortal_298k.pt"), "club"]
            self.assertEqual(placement_pretest.absent_from(Population.from_paths(seated), 4321), ["club"])
            self.assertEqual(placement_pretest.absent_from(Population.from_paths(seated), 4323), [])
            with self.assertRaisesRegex(SystemExit, r"leaves \['club'\] out"):
                placement_pretest.main([str(Path(folder) / "missing.pt"), "--opponents", *seated,
                                        "--seat-share", "0.5", "--seed", "4321", "--device", "cpu",
                                        "--out", str(out)])
            self.assertFalse(out.exists())


class SmallRun(unittest.TestCase):
    """The whole pre-test on tiny networks and a few games: it plays, fits
    U, pays three ways, teaches the critics, takes the gradients and reads
    the flips, and writes what a fork starts from."""

    def test_a_small_pretest_runs_end_to_end(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
                root = Path(folder)
                torch.manual_seed(13)
                # A checkpoint with its optimiser's state, as the trainer writes one.
                run(train_combined, root / "run", flags(train_combined, root, resume=False))
                opponent = mortal_checkpoint(root / "zoo--mortal_298k.pt")
                out = root / "out"
                # A seed whose round keeps the Club in its pool (see
                # `test_a_seed_that_sits_a_member_out_is_refused_first`).
                report = placement_pretest.main([
                    str(root / "run" / "latest.pt"), "--opponents", str(opponent), "club",
                    "--seat-share", "0.5", "--games", "6", "--seed", "4323", "--folds", "2",
                    "--splits", "2", "--batch", "64", "--chunk", "64", "--epochs", "1",
                    "--probes", "40", "--probe-games", "3", "--probe-rounds", "3", "--device", "cpu",
                    "--out", str(out),
                ])
                written = json.loads((out / "report.json").read_text(encoding="utf-8"))
                ledger = Ledger.load(out / "ledger.npz")
                model = place_model.PlaceModel.load(out / "place-model.pt")
        finally:
            torch.set_num_threads(threads)
        self.assertEqual(written["verdict"], json.loads(json.dumps(report["verdict"])))
        self.assertEqual(set(report["arms"]), {"today", "alpha 0", "alpha 0.05"})
        self.assertEqual(report["members"], ["zoo/mortal_298k", "club"])
        self.assertEqual(model.members, ("zoo/mortal_298k", "club"))
        self.assertEqual(len(ledger), report["decisions"])
        self.assertIn(report["verdict"]["verdict"],
                      {"go", "kill", "unread", "split: the two measures disagree; the owner decides"})
        self.assertEqual(report["agreement_method"]["preconditioner"], "the checkpoint's AdamW second moments")
        self.assertGreater(report["agreement_method"]["probes"], 0)
        for name, arm in report["arms"].items():
            with self.subTest(arm=name):
                self.assertEqual(len(arm["agreement"]["cosine"]["real"]), 2)
                self.assertEqual(len(arm["agreement"]["cosine"]["null"]), 2)
                self.assertIn("all", arm["advantage_variance"])
        self.assertIn("East 1", {row["stage"] for row in report["u"]["by_stage"]})
        self.assertEqual(model.fitted["unseen"], report["u"]["unseen"])
        if report["u"]["unseen"]:
            self.assertEqual(report["verdict"]["verdict"], "kill")


if __name__ == "__main__":
    unittest.main()
