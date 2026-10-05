"""Where a generation's time goes is written down: the joined player's own
deciding in the round's play record, and the pass that values a round
before it is learned."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import combined, model, mortal_model, selfplay, train_combined, train_mortal
from neural.observe import Planes


def tiny_fusion():
    net = combined.Combined(model.PolicyValueNet(8, 1, actions=46), mortal_model.build(8, 1))
    net.mortal_config = {"resnet": {"conv_channels": 8, "num_blocks": 1}, "control": {"version": 4}}
    return net


class LearnerDecidingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_the_joined_players_deciding_is_in_the_play_record(self):
        torch.manual_seed(5)
        net = tiny_fusion()
        # What a measurement left behind is not this round's.
        net.timing["network"] = 1e6
        batch = selfplay.play(net, games=1, seed=31, device="cpu")
        self.assertGreater(batch.decisions, 0)
        for name in ("encode", "translate", "network", "riichi"):
            self.assertIn(name, batch.timing)
        self.assertGreater(batch.timing["translate"], 0)
        self.assertGreater(batch.timing["network"], 0)
        self.assertLess(batch.timing["network"], 1e5)
        # Folded into the round's record and cleared for the next.
        self.assertEqual(set(net.timing.values()), {0.0})


class BaselineSecondsTests(unittest.TestCase):
    def test_both_mortal_space_trainers_log_the_baseline_pass(self):
        torch.set_num_threads(1)
        for module in (train_mortal, train_combined):
            with self.subTest(trainer=module.__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                origin, out = root / "source.pt", root / "run"
                if module is train_mortal:
                    net = mortal_model.build(8, 1)
                    payload = {**net.state(), "config": {"resnet": {"conv_channels": 8, "num_blocks": 1},
                                                         "control": {"version": 4}}}
                else:
                    payload = tiny_fusion().state()
                torch.save(payload, origin)

                def collect(learner, **_kwargs):
                    n = 8
                    planes = Planes(np.arange(n + 1, dtype=np.int64), np.zeros(n, dtype=np.uint16),
                                    np.ones(n, dtype=np.float16))
                    return SimpleNamespace(
                        decisions=n, observations=planes, legal=torch.ones(n, 46, dtype=torch.bool),
                        actions=torch.zeros(n, dtype=torch.int64), returns=torch.linspace(-1, 1, n),
                        log_probs=torch.zeros(n), held=torch.full((n, 3, 34), 1 / 34),
                        games=1, hands=1, timing={})

                argv = ["trainer", "--mortal" if module is train_mortal else "--resume", str(origin),
                        "--out", str(out), "--rounds", "1", "--batch", "4", "--epochs", "1",
                        "--games", "1", "--measure-games", "1"]
                if module is train_combined:
                    argv += ["--fixed", "none"]
                with patch.object(sys, "argv", argv), \
                        patch.object(torch.cuda, "is_available", return_value=False), \
                        patch.object(module.selfplay, "play", side_effect=collect), \
                        patch.object(module.selfplay, "measure",
                                     return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                record = json.loads((out / "log.jsonl").read_text().splitlines()[0])
                self.assertIsInstance(record["baseline_seconds"], float)
                self.assertGreaterEqual(record["baseline_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
