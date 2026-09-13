"""Search learner correctness: weighting, deal-level validation and exact resume."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import train_search as training
from neural.checkpoints import atomic_save


class TinyNet(torch.nn.Module):
    """Controlled network with stochastic training to test restored Torch RNG."""
    kind, planes, actions = "mortal", 1012, 46

    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(.2)
        self.policy = torch.nn.Linear(2, 46)
        self.value = torch.nn.Linear(2, 1)

    def everything(self, x, legal=None):
        x = self.drop(x)
        return self.policy(x), self.value(x).squeeze(-1), None

    def state(self):
        return {"model": self.state_dict()}


class Replay:
    """Controlled replay interface; actual shard/engine integration is below."""
    def __init__(self, seed=20, games=None):
        self.metadata = {
            "version": 1, "reward": {"version": 1, "name": "hand_points_over_4000_plus_placement"},
            "observation": {"planes": 1012, "positions": 34, "encoder_version": 4},
            "actions": {"policy": 46, "engine": 78}, "source_revision": "b" * 40,
            "target_kind": "frozen_actor_plus_search_action", "training_api_version": 2,
            "search": {"valued_by": "critic"}, "seed": seed,
        }
        game = np.asarray(games if games is not None else [0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        n = len(game)
        self.metadata.update(games=int(game.max()) + 1, rows=n)
        self.arrays = {"game": game, "features": np.arange(n * 2, dtype=np.float32).reshape(n, 2) / 10,
                       "labels": np.arange(n, dtype=np.int64) % 46,
                       "returns": np.linspace(-1, 1, n, dtype=np.float32)}
        self.seen_training = []
        self.seen_validation = []

    def validate(self):
        if not all(np.isfinite(a).all() for a in self.arrays.values()):
            raise ValueError("invalid controlled replay")

    def loss(self, net, rows, device="cpu", value_weight=.5):
        (self.seen_training if net.training else self.seen_validation).extend(rows.tolist())
        x = torch.from_numpy(self.arrays["features"][rows]).to(device)
        policy, value, _ = net.everything(x)
        labels = torch.from_numpy(self.arrays["labels"][rows]).to(device)
        returns = torch.from_numpy(self.arrays["returns"][rows]).to(device)
        return (torch.nn.functional.cross_entropy(policy, labels)
                + value_weight * torch.nn.functional.mse_loss(value, returns))


def make_learner(replays=None, options=None, saved=None):
    options = options or training.Options(batch=3, lr=1e-3)
    torch.manual_seed(177)
    net = TinyNet()
    if saved is not None:
        net.load_state_dict(saved["model"])
    data = training.Dataset(replays or [Replay()], options.validation_every)
    learner = training.Learner(net, data, options, initial_sha256="a" * 64, initial_generation=4,
                               resume=saved["search_training"] if saved else None)
    return learner


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assertTreeEqual(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for k in a:
                self.assertTreeEqual(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assertTreeEqual(x, y)
        else:
            self.assertEqual(a, b)

    def test_actual_serialized_resume_matches_uninterrupted_including_rng_and_adam(self):
        continuous = make_learner()
        for _ in range(3):
            continuous.epoch()
        expected = continuous.checkpoint()
        split = make_learner()
        split.epoch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            atomic_save(split.checkpoint(), path)
            saved = torch.load(path, weights_only=True)
        torch.manual_seed(9999)
        resumed = make_learner(saved=saved)
        resumed.epoch()
        resumed.epoch()
        self.assertTreeEqual(expected, resumed.checkpoint())

    def test_validation_keeps_every_stage_and_player_of_a_deal_together(self):
        a, b = Replay(seed=20), Replay(seed=19, games=[1, 1, 2, 2])
        data = training.Dataset([a, b], 5)
        training_seeds, validation_seeds = set(), set()
        for replay, train, valid in zip(data.replays, data.training, data.validation):
            training_seeds.update(replay.metadata["seed"] + replay.arrays["game"][train])
            validation_seeds.update(replay.metadata["seed"] + replay.arrays["game"][valid])
        self.assertFalse(training_seeds & validation_seeds)
        self.assertEqual(validation_seeds, {20})

    def test_only_training_rows_enter_optimizer_and_all_tail_rows_are_used(self):
        replay = Replay()
        learner = make_learner([replay], training.Options(batch=5))
        learner.epoch()
        self.assertEqual(sorted(replay.seen_training), [2, 3, 4, 5, 6, 7])
        self.assertEqual(sorted(replay.seen_validation), [0, 1])
        self.assertEqual(learner.updates, 2)  # second minibatch is a singleton

    def test_order_of_shard_paths_does_not_change_training(self):
        a, b = Replay(), Replay(seed=30)
        first = make_learner([a, b]); first.epoch()
        second = make_learner([b, a]); second.epoch()
        self.assertTreeEqual(first.checkpoint(), second.checkpoint())

    def test_each_global_batch_row_appears_once_even_with_empty_training_shards(self):
        a, b = Replay(seed=20, games=[0, 0]), Replay(seed=21)
        data = training.Dataset([a, b], 5)
        seen = []
        indices = np.arange(data.train_rows, dtype=np.int64)[::-1]
        for replay, rows in data.groups(indices):
            seen.extend((id(replay), int(row)) for row in rows)
        expected = [(id(r), int(row)) for r, rows in zip(data.replays, data.training) for row in rows]
        self.assertCountEqual(seen, expected)

    def test_multi_shard_gradients_are_weighted_by_rows_not_groups(self):
        a, b = Replay(seed=21, games=[0, 0]), Replay(seed=31, games=[0, 0, 0, 0, 0, 0])
        options = training.Options(batch=8, validation_every=0)
        learner = make_learner([a, b], options)
        learner.net.drop.p = 0
        combined = Replay(seed=100, games=[0] * 8)
        for key in combined.arrays:
            combined.arrays[key] = np.concatenate([r.arrays[key] for r in learner.data.replays])
        reference = make_learner([combined], options)
        reference.net.drop.p = 0
        learner.epoch(); reference.epoch()
        for key, value in learner.net.state_dict().items():
            torch.testing.assert_close(value, reference.net.state_dict()[key], rtol=1e-6, atol=1e-7)

    def test_resume_rejects_changed_data_settings_runtime_and_parameter_order(self):
        learner = make_learner(); learner.epoch(); saved = learner.checkpoint()
        changed = Replay(); changed.arrays["returns"][0] += .1
        with self.assertRaisesRegex(ValueError, "data changed"):
            make_learner([changed], saved=saved)
        with self.assertRaisesRegex(ValueError, "options changed"):
            make_learner(options=training.Options(batch=2, lr=1e-3), saved=saved)
        for key, value in (("runtime", {}), ("parameters", []), ("version", 2), ("updates", 999)):
            bad = deepcopy(saved); bad["search_training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_learner(saved=bad)

    def test_missing_nonfinite_or_wrong_shaped_optimizer_history_is_rejected(self):
        learner = make_learner(); learner.epoch(); saved = learner.checkpoint()
        for kind in ("empty", "shape", "nan", "lr", "counter"):
            bad = deepcopy(saved)
            optimizer = bad["search_training"]["optimizer"]
            first = next(iter(optimizer["state"].values()))
            if kind == "empty": optimizer["state"] = {}
            elif kind == "shape": first["exp_avg"] = torch.zeros(1)
            elif kind == "nan": first["exp_avg"].fill_(float("nan"))
            elif kind == "lr": optimizer["param_groups"][0]["lr"] *= 2
            else: first["step"].fill_(999)
            with self.subTest(kind=kind), self.assertRaises((ValueError, FloatingPointError)):
                make_learner(saved=bad)

    def test_duplicates_incompatible_sources_heads_and_native_versions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            training.Dataset([Replay(), Replay()], 5)
        for key, value in (("source_revision", "c" * 40), ("reward", {"version": 2}),
                           ("training_api_version", 999), ("training_api_version", True)):
            bad = Replay(seed=30); bad.metadata[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                training.Dataset([Replay(), bad], 5)
        bad = Replay(seed=30); bad.metadata["search"]["valued_by"] = "public"
        with self.assertRaises(ValueError): training.Dataset([Replay(), bad], 5)

    def test_nested_boolean_and_float_schema_versions_are_not_integer_versions(self):
        for value in (True, 1.0):
            bad = Replay(); bad.metadata["reward"]["version"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                training.Dataset([bad], 5)

    def test_missing_validation_is_not_reported_as_zero_loss(self):
        with self.assertRaisesRegex(ValueError, "No validation games"):
            training.Dataset([Replay(seed=21, games=[0, 0])], 5)
        learner = make_learner(options=training.Options(batch=2, validation_every=0))
        self.assertIsNone(learner.epoch()["validation_loss"])
        with self.assertRaisesRegex(ValueError, "two training rows"):
            training.Dataset([Replay(seed=20, games=[0, 0])], 5)

    def test_u64_seed_split_does_not_wrap_through_signed_int64(self):
        replay = Replay(seed=2**64 - 4)
        data = training.Dataset([replay], 5)
        for row in data.validation[0]:
            self.assertEqual((replay.metadata["seed"] + int(replay.arrays["game"][row])) % 5, 0)

    def test_bad_options_fail_without_optimization(self):
        for change in ({"batch": 1}, {"batch": True}, {"lr": 0}, {"lr": float("nan")},
                       {"max_grad_norm": 0}, {"weight_decay": -1}, {"validation_every": 1},
                       {"seed": -1}, {"seed": 2**64}, {"validation_every": 2**64}, {"value_weight": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                training.Options(**change).validate()

    def test_no_checkpoint_before_updates_or_after_partial_epoch_failure(self):
        learner = make_learner()
        with self.assertRaises(RuntimeError): learner.checkpoint()
        learner.epoch()
        before = learner.checkpoint()
        with patch.object(learner.optimizer, "step", side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError): learner.epoch()
        with self.assertRaises(RuntimeError): learner.checkpoint()
        with self.assertRaises(RuntimeError): learner.epoch()
        # Snapshots are copied, not aliases into live network/optimizer tensors.
        restored = make_learner(saved=before)
        self.assertTreeEqual(restored.checkpoint(), before)

    def test_nonfinite_losses_and_gradients_never_reach_optimizer(self):
        for kind in ("loss", "gradient"):
            learner = make_learner()
            replay = learner.data.replays[0]
            if kind == "loss":
                effect = lambda *a, **k: learner.net.policy.weight.sum() * float("nan")
            else:
                # Finite forward value, but an invalid backward gradient.
                class BadGradient(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, x): return x.sum()
                    @staticmethod
                    def backward(ctx, g): return torch.full((46, 2), float("nan"))
                effect = lambda *a, **k: BadGradient.apply(learner.net.policy.weight)
            with patch.object(replay, "loss", side_effect=effect), patch.object(learner.optimizer, "step") as step:
                with self.assertRaises((FloatingPointError, RuntimeError)): learner.epoch()
                step.assert_not_called()
            with self.assertRaises(RuntimeError): learner.checkpoint()

    def test_wrong_network_contract_fails_before_optimizer_creation(self):
        net = TinyNet(); net.actions = 78
        with patch.object(torch.optim, "AdamW") as optimizer, self.assertRaises(ValueError):
            training.Learner(net, training.Dataset([Replay()], 5), training.Options(), initial_sha256="a" * 64)
        optimizer.assert_not_called()

    def controlled_run(self, source, out, *, epochs=1, resume=False, overrides=None):
        def load(path, device):
            state = torch.load(path, weights_only=True)
            net = TinyNet(); net.load_state_dict(state["model"])
            return net, state
        with patch.object(training.Dataset, "load", side_effect=lambda paths, every: training.Dataset([Replay()], every)), \
             patch.object(training, "_load_network", side_effect=load), redirect_stdout(io.StringIO()):
            return training.run(source, [Path("controlled-replay")], out, epochs=epochs, resume=resume,
                                overrides=overrides)

    def test_actual_runner_writes_loadable_checkpoints_and_resumes_to_a_new_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "actor.pt"
            atomic_save({**TinyNet().state(), "generation": 4}, source)
            original = source.read_bytes()
            path = self.controlled_run(source, root / "run-1", overrides={"batch": 3})
            next_path = self.controlled_run(path, root / "run-2", epochs=2, resume=True)
            saved = torch.load(next_path, weights_only=True)
            self.assertEqual(saved["generation"], 7)
            self.assertEqual(saved["search_training"]["epochs"], 3)
            self.assertTrue((root / "run-2/latest.pt.previous").exists())
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse((root / "run-2/champion.pt").exists())
            with self.assertRaises(FileExistsError): self.controlled_run(source, root / "run-1")
            with self.assertRaises(ValueError): self.controlled_run(source, root / "bad", resume=True)
            self.assertFalse((root / "bad").exists())
            with self.assertRaises(ValueError):
                self.controlled_run(next_path, root / "mismatch", resume=True, overrides={"lr": .1})
            self.assertFalse((root / "mismatch").exists())

    def test_interrupted_publication_preserves_previous_completed_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "actor.pt"
            atomic_save({**TinyNet().state(), "generation": 0}, source)
            from neural import checkpoints
            real_replace = checkpoints.os.replace
            saved_bytes = []
            def replace(a, b):
                if Path(b).name == "latest.pt":
                    if saved_bytes:
                        raise OSError("publication failed")
                    saved_bytes.append(Path(a).read_bytes())
                return real_replace(a, b)
            with patch.object(checkpoints.os, "replace", side_effect=replace):
                with self.assertRaises(OSError): self.controlled_run(source, root / "run", epochs=2)
            self.assertEqual((root / "run/latest.pt").read_bytes(), saved_bytes[0])
            saved = torch.load(root / "run/latest.pt", weights_only=True)
            self.assertEqual(saved["search_training"]["epochs"], 1)


_NATIVE = all(importlib.util.find_spec(m) is not None for m in ("riichi_py", "libriichi"))


@unittest.skipUnless(_NATIVE, "requires the real engines installed by repository CI")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_standalone_and_combined_checkpoint_optimization_and_resume(self):
        from neural import combined, model, mortal_model
        from neural.tests.test_search_replay import example
        for joined in (False, True):
            with self.subTest(joined=joined), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                ours = model.PolicyValueNet(8, 1, actions=46)
                net = combined.Combined(ours, mortal_model.build(16, 1)) if joined else ours
                if joined:
                    net.mortal_config = {"resnet": {"conv_channels": 16, "num_blocks": 1}, "control": {"version": 4}}
                atomic_save({**training._model_payload(net), "generation": 4}, root / "actor.pt")
                replay = example(); replay.metadata["training_api_version"] = training.TRAINING_API_VERSION
                replay.save(root / "replay")
                options = {"batch": 2, "validation_every": 0}
                with redirect_stdout(io.StringIO()):
                    all_at_once = training.run(root / "actor.pt", [root / "replay"], root / "continuous",
                                               epochs=2, overrides=options)
                    one = training.run(root / "actor.pt", [root / "replay"], root / "one", epochs=1, overrides=options)
                    two = training.run(one, [root / "replay"], root / "two", epochs=1, resume=True)
                a, b = (torch.load(path, weights_only=True) for path in (all_at_once, two))
                for part in (("combined", "model", "mortal", "current_dqn") if joined else ("model",)):
                    for key in a[part]:
                        torch.testing.assert_close(a[part][key], b[part][key], rtol=0, atol=0)
                self.assertEqual(b["search_training"]["updates"], 4)
                loaded, _ = training._load_network(two, "cpu")
                self.assertEqual(loaded.actions, 46)

    def test_completed_multi_candidate_native_collection_trains_a_loadable_checkpoint(self):
        from neural import model
        from neural.collect_search import collect, SearchSettings
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            net = model.PolicyValueNet(8, 1, actions=46)
            atomic_save({**training._model_payload(net), "generation": 0}, root / "actor.pt")
            replay = collect(net, games=1, seed=7,
                             settings=SearchSettings(worlds=3, candidates=2, played_by="club"),
                             actor_sha256="a" * 64, source_revision="b" * 40)
            replay.save(root / "replay")
            with redirect_stdout(io.StringIO()):
                path = training.run(root / "actor.pt", [root / "replay"], root / "trained", epochs=1,
                                    overrides={"batch": 64, "validation_every": 0})
            saved = torch.load(path, weights_only=True)
            self.assertGreater(saved["search_training"]["updates"], 0)
            self.assertGreater(saved["search_training"]["metrics"]["training_rows"], 0)
            loaded, _ = training._load_network(path, "cpu")
            self.assertTrue(all(torch.isfinite(p).all() for p in loaded.parameters()))


if __name__ == "__main__":
    unittest.main()
