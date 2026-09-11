"""Failure recovery, useful gradients, and actual optimizer updates."""
from __future__ import annotations

import contextlib
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import checkpoint, combined, mortal_model, train_combined, train_mortal
from neural.model import MORTAL_PLANES, PolicyValueNet
from neural.observe import Planes
from neural.training_safety import (
    minibatch_indices, require_search_payload, require_legacy_search,
    require_training_engine, validate_training_options,
)


class CheckpointTests(unittest.TestCase):
    def test_success_preserves_previous_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "latest.pt"
            checkpoint.atomic_save({"generation": 1, "x": torch.ones(3)}, path)
            original = path.read_bytes()
            checkpoint.atomic_save({"generation": 2}, path)
            self.assertEqual(path.with_name("latest.pt.previous").read_bytes(), original)
            self.assertEqual(torch.load(path, weights_only=True)["generation"], 2)
            self.assertEqual(list(Path(root).glob("*.tmp")), [])

    def test_partial_serialization_never_truncates_destination(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "latest.pt"
            checkpoint.atomic_save({"generation": 1}, path)
            original = path.read_bytes()
            def broken(payload, stream):
                stream.write(b"incomplete checkpoint")
                raise OSError("disk write failed")
            with patch.object(checkpoint.torch, "save", side_effect=broken):
                with self.assertRaises(OSError):
                    checkpoint.atomic_save({"generation": 2}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(torch.load(path, weights_only=True)["generation"], 1)

    def test_interruptions_on_both_sides_of_each_replace_keep_complete_files(self):
        replace = os.replace
        for boundary in (1, 2):
            for after in (False, True):
                with self.subTest(boundary=boundary, after=after), tempfile.TemporaryDirectory() as root:
                    path = Path(root) / "latest.pt"
                    checkpoint.atomic_save({"generation": 1}, path)
                    calls = 0
                    def interrupted(source, destination):
                        nonlocal calls
                        calls += 1
                        if calls == boundary and not after:
                            raise KeyboardInterrupt()
                        replace(source, destination)
                        if calls == boundary and after:
                            raise KeyboardInterrupt()
                    with patch.object(checkpoint.os, "replace", side_effect=interrupted):
                        with self.assertRaises(KeyboardInterrupt):
                            checkpoint.atomic_save({"generation": 2}, path)
                    expected = 2 if boundary == 2 and after else 1
                    self.assertEqual(torch.load(path, weights_only=True)["generation"], expected)
                    previous = path.with_name("latest.pt.previous")
                    if previous.exists():
                        self.assertEqual(torch.load(previous, weights_only=True)["generation"], 1)

    def test_sync_failure_does_not_delete_published_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "latest.pt"
            checkpoint.atomic_save({"generation": 1}, path)
            with patch.object(checkpoint, "_sync_directory", side_effect=[None, OSError("sync")]):
                with self.assertRaises(OSError):
                    checkpoint.atomic_save({"generation": 2}, path)
            self.assertEqual(torch.load(path, weights_only=True)["generation"], 2)
            self.assertEqual(torch.load(path.with_name("latest.pt.previous"), weights_only=True)["generation"], 1)

    def test_hard_exit_before_and_after_live_publication(self):
        script = '''
import os, sys
from pathlib import Path
from neural import checkpoint
path = Path(sys.argv[1])
original = os.replace
def interrupt(source, destination):
    live = Path(destination) == path
    if live and sys.argv[2] == "before":
        os._exit(71)
    original(source, destination)
    if live:
        os._exit(72)
checkpoint.os.replace = interrupt
checkpoint.atomic_save({"generation": 2}, path)
'''
        for when, generation in (("before", 1), ("after", 2)):
            with self.subTest(when=when), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "latest.pt"
                checkpoint.atomic_save({"generation": 1}, path)
                result = subprocess.run([sys.executable, "-c", script, str(path), when],
                                        capture_output=True, timeout=30)
                self.assertIn(result.returncode, (71, 72), result.stderr.decode())
                self.assertEqual(torch.load(path, weights_only=True)["generation"], generation)
                self.assertEqual(torch.load(path.with_name("latest.pt.previous"), weights_only=True)["generation"], 1)


class LearningSafetyTests(unittest.TestCase):
    def test_eager_epoch_uses_every_row_including_small_rounds(self):
        for count, size in ((1, 8), (3, 8), (19, 8), (16, 8)):
            rows = minibatch_indices(count, size)
            self.assertGreater(len(rows), 0)
            self.assertEqual(sorted(torch.cat(rows).tolist()), list(range(count)))

    def test_compiled_batches_never_silently_train_on_nothing(self):
        with self.assertRaisesRegex(ValueError, "no optimizer updates"):
            minibatch_indices(3, 8, compiled=True)
        rows = minibatch_indices(19, 8, compiled=True)
        self.assertEqual([len(row) for row in rows], [8, 8])
        self.assertEqual(len(set(torch.cat(rows).tolist())), 16)

    def test_invalid_options_fail_before_training(self):
        good = dict(batch=8, epochs=1, games=1, measure_every=1, measure_games=1,
                    rounds=1, generations=1, opponents=[], opponent_share=0.)
        for name in ("batch", "epochs", "games", "measure_every", "measure_games"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_training_options(SimpleNamespace(**{**good, name: 0}))
        with self.assertRaises(FileNotFoundError):
            validate_training_options(SimpleNamespace(**good, resume=Path("/nonexistent/learner.pt")))
        with self.assertRaises(ValueError):
            validate_training_options(SimpleNamespace(**{**good, "opponent_share": 0.5}))

    def test_old_engine_and_unsupported_search_fail_explicitly(self):
        import riichi_py
        with patch.object(riichi_py, "TRAINING_API_VERSION", 0, create=True):
            with self.assertRaisesRegex(RuntimeError, "Rebuild"):
                require_training_engine()
        for payload in ({"combined": {}, "model": {}}, {"mortal": {}}):
            with self.assertRaises(ValueError):
                require_search_payload(payload)
        with self.assertRaises(ValueError):
            require_legacy_search(SimpleNamespace(kind="mortal", actions=46))


class BeliefCorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_supervision_changes_relative_tile_probabilities(self):
        fuse = combined.Fuse(8, phi=16, width=8).double()
        phi = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
        guessed = torch.randn(2, 3, 34, dtype=torch.float64)
        optimizer = torch.optim.SGD(fuse.parameters(), lr=0.2)
        before = fuse.read_hands(phi, guessed).softmax(-1).detach()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = -fuse.read_hands(phi, guessed).log_softmax(-1)[:, :, 0].mean()
            loss.backward()
            self.assertGreater(fuse.hands_out.weight.grad.abs().max().item(), 1e-8)
            self.assertIsNone(phi.grad, "belief supervision must not reshape Mortal")
            optimizer.step()
        after = fuse.read_hands(phi, guessed).softmax(-1).detach()
        self.assertGreater((after - before).abs().max().item(), 1e-5)
        self.assertGreater(fuse.hands_fix[0].weight.grad.abs().max().item(), 1e-8)

    def test_legacy_projection_migration_preserves_predictions(self):
        fuse = combined.Fuse(8, phi=16, width=8).double()
        legacy = deepcopy(fuse.state_dict())
        legacy["hands_out.weight"] = torch.randn(3, 8, 1, dtype=torch.float64)
        legacy["hands_out.bias"] = torch.randn(3, dtype=torch.float64)
        phi = torch.randn(2, 16, dtype=torch.float64)
        guessed = torch.randn(2, 3, 34, dtype=torch.float64)
        spread = fuse.hands_fix(phi).unsqueeze(2).expand(-1, -1, 34)
        expected = guessed + torch.nn.functional.conv1d(
            spread, legacy["hands_out.weight"], legacy["hands_out.bias"])
        fuse.load_state_dict(legacy)
        actual = fuse.read_hands(phi, guessed)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(legacy["hands_out.weight"].ndim, 3)
        restored = combined.Fuse(8, phi=16, width=8).double()
        restored.load_state_dict(fuse.state_dict())
        torch.testing.assert_close(restored.read_hands(phi, guessed), actual)

    def test_only_legacy_projection_optimizer_moments_are_reset(self):
        fuse = combined.Fuse(8, phi=16, width=8)
        optimizer = torch.optim.AdamW(fuse.parameters())
        for parameter, shape in ((fuse.hands_out.weight, (3, 8, 1)),
                                 (fuse.hands_out.bias, (3,))):
            optimizer.state[parameter] = dict(step=torch.tensor(1.),
                exp_avg=torch.ones(shape), exp_avg_sq=torch.ones(shape))
        other = fuse.bias
        optimizer.state[other] = dict(step=torch.tensor(7.),
            exp_avg=torch.ones_like(other), exp_avg_sq=torch.ones_like(other))
        with self.assertWarnsRegex(RuntimeWarning, "legacy belief"):
            reset = combined.Combined.reset_legacy_belief_optimizer_state(
                SimpleNamespace(fuse=fuse), optimizer)
        self.assertEqual(reset, 2)
        self.assertNotIn(fuse.hands_out.weight, optimizer.state)
        self.assertEqual(optimizer.state[other]["step"].item(), 7.)

    def test_auxiliary_losses_do_not_update_policy_backbones(self):
        net = combined.Combined(PolicyValueNet(8, 1, MORTAL_PLANES, actions=46),
                                mortal_model.build(8, 1))
        planes = torch.zeros(2, MORTAL_PLANES, 34)
        allowed = torch.ones(2, 46, dtype=torch.bool)
        _, value, hands = net.everything(planes, allowed)
        loss = (value - 1).square().mean() - hands.log_softmax(-1)[:, :, 0].mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in net.ours.stem.parameters()))
        self.assertTrue(all(p.grad is None for p in net.mortal.parameters()))
        self.assertTrue(any(p.grad is not None for p in net.ours.value.parameters()))


class TrainerSmokeTests(unittest.TestCase):
    """Real networks and training loops; only game collection and padding mocked."""
    def setUp(self):
        torch.manual_seed(53)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def batch(self):
        count = 3
        legal = torch.zeros(count, 46, dtype=torch.bool)
        legal[:, :2] = True
        planes = Planes(np.zeros(count + 1, dtype=np.int64),
                        np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.float16))
        return SimpleNamespace(observations=planes, legal=legal,
            actions=torch.tensor([0, 1, 0]), returns=torch.tensor([1., -1., 0.]),
            log_probs=torch.full((count,), -math.log(2)),
            held=torch.full((count, 3, 34), 1 / 34), decisions=count, hands=1, timing={})

    def run_trainer(self, module, checkpoint_path, out, rounds=1, source="--resume"):
        argv = ["trainer", source, str(checkpoint_path), "--out", str(out),
                "--rounds", str(rounds), "--batch", "8", "--epochs", "1",
                "--measure-every", "1", "--measure-games", "1"]
        if module is train_combined:
            argv += ["--fixed", "none"]
        measured = dict(placement=2.5, score=0., wins=0.25, hands=1)
        with patch.object(sys, "argv", argv), \
                patch.object(module.selfplay, "play", side_effect=lambda *a, **kw: self.batch()), \
                patch.object(module.selfplay, "measure", return_value=measured), \
                patch.object(module, "pad_rows", side_effect=lambda tensor, *a: tensor), \
                patch.object(module, "resident", return_value=None), \
                contextlib.redirect_stdout(io.StringIO()):
            module.main()
        records = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["optimizer_steps"] > 0 for row in records))
        return torch.load(out / "latest.pt", map_location="cpu", weights_only=False)

    def test_combined_small_round_updates_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            net = combined.Combined(PolicyValueNet(8, 1, MORTAL_PLANES, actions=46),
                                    mortal_model.build(8, 1))
            net.mortal_config = {"control": {"version": 4},
                                "resnet": {"conv_channels": 8, "num_blocks": 1}}
            source = root / "source.pt"
            torch.save(net.state(), source)
            uninterrupted = self.run_trainer(train_combined, source, root / "whole", rounds=2)
            first = self.run_trainer(train_combined, source, root / "first")
            resumed = self.run_trainer(train_combined, root / "first/latest.pt", root / "second")
            self.assertEqual(first["generation"], 1)
            self.assertEqual(resumed["generation"], 2)
            self.assertFalse(torch.equal(first["combined"]["weights"], net.fuse.weights.detach()))
            for group in ("combined", "model", "mortal", "current_dqn"):
                for key in uninterrupted[group]:
                    torch.testing.assert_close(uninterrupted[group][key], resumed[group][key], rtol=0, atol=0)
            previous = torch.load(root / "whole/latest.pt.previous", map_location="cpu", weights_only=False)
            self.assertEqual(previous["generation"], 1, "final save must not erase the prior generation")

    def test_mortal_small_round_updates_and_records_rng(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            net = mortal_model.build(8, 1)
            state = net.state()
            state["config"] = {"control": {"version": 4},
                               "resnet": {"conv_channels": 8, "num_blocks": 1}}
            source = root / "mortal.pt"
            torch.save(state, source)
            result = self.run_trainer(train_mortal, source, root / "out", source="--mortal")
            self.assertIn("random_state", result)
            self.assertTrue(any(not torch.equal(result["current_dqn"][key], value)
                                for key, value in state["current_dqn"].items()))


if __name__ == "__main__":
    unittest.main()
