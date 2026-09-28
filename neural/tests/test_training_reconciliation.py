"""Regressions for #41 reconciled with the later #40 safety implementation."""
import contextlib
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

from neural import checkpoints, combined, mortal_model, ppo_loop, train_combined, train_mortal
from neural.model import MORTAL_PLANES, PolicyValueNet
from neural.observe import Planes
from neural.training_safety import (
    TRAINING_API_VERSION, benchmark_history, require_training_engine, validate_training_options,
)


class ReconciledCheckpointTests(unittest.TestCase):
    def test_validated_save_retains_the_exact_previous_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            checkpoints.atomic_save({"generation": 1, "weight": torch.arange(5)}, path)
            original = path.read_bytes()
            checkpoints.atomic_save({"generation": 2, "weight": torch.arange(5) + 1}, path)
            self.assertEqual(path.with_name("latest.pt.previous").read_bytes(), original)
            self.assertEqual(checkpoints.validate_checkpoint(path, require_generation=True), 2)
            self.assertFalse(list(path.parent.glob("*.partial")))

    def test_validation_failure_preserves_live_and_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            for generation in (1, 2):
                checkpoints.atomic_save({"generation": generation}, path)
            before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
            with self.assertRaises(ValueError):
                checkpoints.atomic_save({}, path)
            self.assertEqual({p.name: p.read_bytes() for p in path.parent.iterdir()}, before)

    def test_interruption_around_either_rename_never_deletes_a_live_file(self):
        replace = os.replace
        for target in ("latest.pt.previous", "latest.pt"):
            for after in (False, True):
                with self.subTest(target=target, after=after), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "latest.pt"
                    checkpoints.atomic_save({"generation": 1}, path)
                    def interrupted(source, destination):
                        if Path(destination).name == target:
                            if after:
                                replace(source, destination)
                            raise KeyboardInterrupt("publication boundary")
                        replace(source, destination)
                    with patch.object(checkpoints.os, "replace", side_effect=interrupted):
                        with self.assertRaises(KeyboardInterrupt):
                            checkpoints.atomic_save({"generation": 2}, path)
                    expected = 2 if target == "latest.pt" and after else 1
                    self.assertEqual(checkpoints.validate_checkpoint(path), expected)
                    previous = path.with_name("latest.pt.previous")
                    if previous.exists():
                        self.assertEqual(checkpoints.validate_checkpoint(previous), 1)

    def test_hard_exit_around_live_rename_preserves_recovery(self):
        for after in (False, True):
            with self.subTest(after=after), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "latest.pt"
                checkpoints.atomic_save({"generation": 1}, path)
                script = '''
import os, sys
from pathlib import Path
from neural import checkpoints
path, after = Path(sys.argv[1]), sys.argv[2] == "True"
replace = os.replace
def stop(source, destination):
    if Path(destination) == path:
        if after:
            replace(source, destination)
        os._exit(77)
    replace(source, destination)
checkpoints.os.replace = stop
checkpoints.atomic_save({"generation": 2}, path)
'''
                process = subprocess.run([sys.executable, "-c", script, str(path), str(after)],
                                         timeout=30, capture_output=True, text=True)
                self.assertEqual(process.returncode, 77, process.stderr)
                self.assertEqual(checkpoints.validate_checkpoint(path), 2 if after else 1)
                self.assertEqual(checkpoints.validate_checkpoint(path.with_name("latest.pt.previous")), 1)


class RuntimeContractTests(unittest.TestCase):
    def options(self, **changes):
        fields = dict(batch=2, epochs=1, games=1, measure_every=1, measure_games=1,
                      rounds=1, generations=2, opponents=[], opponent_share=0.)
        return SimpleNamespace(**(fields | changes))

    def test_bad_options_fail_before_starting_an_experiment(self):
        validate_training_options(self.options())
        for fields in ({"games": 0}, {"epochs": True}, {"rounds": -1},
                       {"opponent_share": float("nan")}, {"opponent_share": .2},
                       {"fixed": []}, {"fixed": ["not-a-mode"]},
                       {"freeze_policy": True, "freeze_aux": True}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_training_options(self.options(**fields))
        with tempfile.TemporaryDirectory() as directory:
            for name in ("resume", "ours", "mortal"):
                with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                    validate_training_options(self.options(**{name: Path(directory) / "missing.pt"}))

    def test_legacy_metrics_reset_without_mutating_training_state(self):
        payload = dict(generation=12, smoothed=2.1, best_placement=2.0,
                       optimizer_state={"step": 12}, random_state={"seed": 7})
        before = json.dumps(payload, sort_keys=True)
        with self.assertWarnsRegex(RuntimeWarning, "environment changed"):
            self.assertEqual(benchmark_history(payload), (None, float("inf")))
        self.assertEqual(json.dumps(payload, sort_keys=True), before)
        payload["training_api_version"] = TRAINING_API_VERSION
        self.assertEqual(benchmark_history(payload), (2.1, 2.0))

    def test_old_native_api_is_rejected(self):
        import riichi_py
        with patch.object(riichi_py, "TRAINING_API_VERSION", 1):
            with self.assertRaisesRegex(RuntimeError, "Rebuild and reinstall"):
                require_training_engine()

    def test_auxiliary_gradients_do_not_reach_either_policy_backbone(self):
        net = combined.Combined(PolicyValueNet(8, 1, MORTAL_PLANES, actions=46),
                                mortal_model.build(8, 1))
        _, value, hands = net.everything(torch.zeros(2, MORTAL_PLANES, 34),
                                         torch.ones(2, 46, dtype=torch.bool))
        ((value - 1).square().mean() - hands.log_softmax(-1)[:, :, 0].mean()).backward()
        self.assertTrue(all(p.grad is None for p in net.ours.stem.parameters()))
        self.assertTrue(all(p.grad is None for p in net.mortal.parameters()))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)
                            for p in net.ours.value.parameters()))

    def test_the_mortal_learners_value_loss_leaves_its_encoder_alone(self):
        # The value term's gradient on Mortal's encoder was two and a half
        # to four times the policy term's and nearly orthogonal to it, under
        # one clip on the whole gradient: the encoder the policy reads from
        # was being trained to predict returns.
        from neural import mortal_learner
        torch.manual_seed(3)
        net = mortal_learner.MortalLearner(mortal_model.build(8, 1)).train()
        # A value head that is not zero, so a gradient could reach the
        # encoder through it at all.
        torch.nn.init.normal_(net.value_head.weight)
        planes = torch.rand(4, MORTAL_PLANES, 34)
        legal = torch.ones(4, 46, dtype=torch.bool)
        encoder = list(net.mortal.brain.parameters())
        _logits, value = net.policy(planes, legal)
        torch.nn.functional.mse_loss(value, torch.ones(4)).backward()
        self.assertTrue(all(p.grad is None for p in net.mortal.parameters()))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)
                            for p in net.value_head.parameters()))
        net.zero_grad(set_to_none=True)
        logits, _value = net.policy(planes, legal)
        (-torch.log_softmax(logits, dim=1)[:, 0].mean()).backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in encoder))
        self.assertTrue(all(p.grad is None for p in net.value_head.parameters()))


class ReconciledTrainerTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(53)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def batch(self):
        count = 4  # Keep main's full-minibatch contract; do not silently replace it.
        legal = torch.zeros(count, 46, dtype=torch.bool)
        legal[:, :2] = True
        planes = Planes(np.zeros(count + 1, dtype=np.int64),
                        np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.float16))
        return SimpleNamespace(observations=planes, legal=legal,
            actions=torch.tensor([0, 1, 0, 1]), returns=torch.tensor([1., -1., .5, -.5]),
            log_probs=torch.full((count,), -math.log(2)),
            held=torch.full((count, 3, 34), 1 / 34), decisions=count, hands=1, timing={})

    def run_trainer(self, module, source, out, rounds=1, flag="--resume"):
        argv = ["trainer", flag, str(source), "--out", str(out), "--rounds", str(rounds),
                "--batch", "2", "--epochs", "1", "--measure-every", "1", "--measure-games", "1"]
        if module is train_combined:
            argv += ["--fixed", "none"]
        with patch.object(sys, "argv", argv), \
                patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(module.selfplay, "play", side_effect=lambda *a, **kw: self.batch()), \
                patch.object(module.selfplay, "measure", return_value=dict(placement=2.5, score=0., wins=.25)), \
                patch.object(module, "pad_rows", side_effect=lambda tensor, *a: tensor), \
                patch.object(ppo_loop, "pad_rows", side_effect=lambda tensor, *a: tensor), \
                patch.object(ppo_loop, "resident", return_value=None), \
                contextlib.redirect_stdout(io.StringIO()):
            module.main()
        rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["optimizer_updates"] > 0 for row in rows))
        self.assertTrue(all(row["training_api_version"] == TRAINING_API_VERSION for row in rows))
        return torch.load(out / "latest.pt", map_location="cpu", weights_only=True)

    def test_combined_resume_and_previous_checkpoint_keep_correct_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            net = combined.Combined(PolicyValueNet(8, 1, MORTAL_PLANES, actions=46),
                                    mortal_model.build(8, 1))
            net.mortal_config = {"control": {"version": 4},
                                "resnet": {"conv_channels": 8, "num_blocks": 1}}
            source = root / "source.pt"
            torch.save(net.state(), source)
            whole = self.run_trainer(train_combined, source, root / "whole", rounds=2)
            self.run_trainer(train_combined, source, root / "first")
            split = self.run_trainer(train_combined, root / "first/latest.pt", root / "split")
            self.assertEqual(split["generation"], 2)
            self.assertEqual(split["training_api_version"], TRAINING_API_VERSION)
            for group in ("combined", "model", "mortal", "current_dqn"):
                for key in whole[group]:
                    torch.testing.assert_close(whole[group][key], split[group][key], rtol=0, atol=0)
            self.assertEqual(checkpoints.validate_checkpoint(root / "whole/latest.pt.previous"), 1)

    def test_mortal_retains_main_rng_resume_and_environment_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            net = mortal_model.build(8, 1)
            state = net.state()
            state["config"] = {"control": {"version": 4}, "resnet": {"conv_channels": 8, "num_blocks": 1}}
            source = root / "source.pt"
            torch.save(state, source)
            trained = self.run_trainer(train_mortal, source, root / "out", flag="--mortal")
            self.assertIn("sampling_state", trained)
            self.assertIn("torch", trained["sampling_state"])
            self.assertEqual(trained["training_api_version"], TRAINING_API_VERSION)
            self.assertTrue(any(not torch.equal(trained["current_dqn"][key], value)
                                for key, value in state["current_dqn"].items()))
