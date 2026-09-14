"""Conservative search fitting: gradients, real replay contracts and Adam resume."""
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
from neural.search_objective import policy_terms, replay_loss
from neural.search_replay import SearchReplay, action_contract, improvement_policy


class FeatureRows:
    """Two-feature adapter for controlled tests; production uses sparse Planes."""
    def __init__(self, values):
        self.values = values

    def rows(self, rows):
        return FeatureRows(self.values[rows])

    def dense(self, device):
        return torch.from_numpy(self.values.copy()).float().to(device)


class ControlledReplay(SearchReplay):
    def observations(self):
        return FeatureRows(self.arrays["root_values"].reshape(-1, 2))


def example(seed=20, games=4):
    n = games * 2
    engine_legal = np.zeros((n, 78), dtype=bool)
    engine_legal[:, :2] = True
    actions = np.arange(n, dtype=np.int64) % 2
    stages = np.zeros(n, dtype=np.int64)
    legal, aliases = action_contract(engine_legal, actions, stages)
    actor = np.zeros((n, 46), dtype=np.float32)
    actor[:, :2] = [.7, .3]
    settings = dict(worlds=8, candidates=2, pool=1, max_steps=4000, depth=0,
                    margin=2.0, temperature=0.0, improve=.5, played_by="network",
                    valued_by="critic", hurried=False)
    metadata = dict(version=1, complete=True, rows=n, games=games, seed=seed,
                    reward={"version": 1, "name": "hand_points_over_4000_plus_placement"},
                    observation={"planes": 1012, "positions": 34, "encoder_version": 4},
                    actions={"policy": 46, "engine": 78}, collection="all_seats_search",
                    target_kind="frozen_actor_plus_search_action", actor_sha256="a" * 64,
                    opponent_sha256=["a" * 64] * 4, source_revision="b" * 40,
                    training_api_version=training.TRAINING_API_VERSION, search=settings)
    arrays = dict(legal=legal, actor_policy=actor,
                  policy_target=improvement_policy(actor, legal, aliases, .5),
                  engine_legal=engine_legal, engine_action=actions, stage=stages,
                  returns=np.linspace(-1, 1, n, dtype=np.float32),
                  game=np.repeat(np.arange(games, dtype=np.int64), 2),
                  player=np.zeros(n, dtype=np.int64), step=np.tile(np.array([1, 2], dtype=np.int64), games),
                  root_indptr=np.arange(0, n * 2 + 1, 2, dtype=np.int64),
                  root_indices=np.tile(np.array([0, 1], dtype=np.uint16), n),
                  root_values=np.linspace(.1, 1, n * 2, dtype=np.float16))
    replay = ControlledReplay(metadata, arrays)
    replay.validate()
    return replay


class TinyNet(torch.nn.Module):
    kind, planes, actions = "mortal", 1012, 46

    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(.2)
        self.policy = torch.nn.Linear(2, 46)
        self.value = torch.nn.Linear(2, 1)

    def everything(self, x, legal):
        x = self.drop(x)
        return self.policy(x), self.value(x).squeeze(-1), None

    def state(self):
        return {"model": self.state_dict()}


def learner(options=None, saved=None):
    options = options or training.Options(batch=3, lr=1e-3, policy_mode="changed", actor_kl=1.0)
    torch.manual_seed(177)
    net = TinyNet()
    if saved is not None:
        net.load_state_dict(saved["model"])
    data = training.Dataset([example()], options.validation_every)
    return training.Learner(net, data, options, initial_sha256="a" * 64,
                            resume=saved["search_training"] if saved else None)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def inputs(self, action=0):
        actor = torch.tensor([[.6, .4]])
        aliases = torch.zeros_like(actor, dtype=torch.bool)
        aliases[:, action] = True
        target = .5 * actor + .5 * aliases.float()
        return actor.log().requires_grad_(), actor, target, torch.ones_like(aliases), aliases

    def assert_tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_agreement_has_zero_policy_gradient_instead_of_sharpening(self):
        values = self.inputs()
        terms = policy_terms(*values)
        terms.loss.backward()
        torch.testing.assert_close(values[0].grad, torch.zeros_like(values[0]), atol=1e-7, rtol=0)
        self.assertEqual(terms.changed_fraction.item(), 0)
        legacy = self.inputs()
        policy_terms(*legacy, policy_mode="all", actor_kl=0).loss.backward()
        self.assertLess(legacy[0].grad[0, 0].item(), -.1)

    def test_changed_move_still_receives_an_improvement_gradient(self):
        values = self.inputs(action=1)
        terms = policy_terms(*values)
        terms.loss.backward()
        self.assertEqual(terms.changed_fraction.item(), 1)
        self.assertLess(values[0].grad[0, 1].item(), 0)

    def test_tied_actor_choices_are_preserved_conservatively(self):
        logits, _, _, legal, aliases = self.inputs(action=1)
        actor = torch.tensor([[.5, .5]])
        logits = actor.log().requires_grad_()
        terms = policy_terms(logits, actor, torch.tensor([[.25, .75]]), legal, aliases)
        terms.loss.backward()
        self.assertEqual(terms.changed_fraction.item(), 0)
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_red_plain_alias_is_not_a_changed_engine_move(self):
        engine = np.zeros((1, 78), dtype=bool); engine[:, [0, 4]] = True
        legal, aliases = action_contract(engine, np.array([4], dtype=np.int64), np.array([0], dtype=np.int64))
        actor = np.zeros((1, 46), dtype=np.float32); actor[:, [0, 4, 34]] = [.2, .3, .5]
        target = improvement_policy(actor, legal, aliases, .5)
        logits = torch.from_numpy(actor).clamp_min(1e-10).log().requires_grad_()
        terms = policy_terms(logits, torch.from_numpy(actor), torch.from_numpy(target),
                             torch.from_numpy(legal), torch.from_numpy(aliases))
        terms.loss.backward()
        self.assertEqual(terms.changed_fraction.item(), 0)
        self.assertLess(logits.grad.abs().max().item(), 1e-7)

    def test_riichi_declaration_and_conditional_discard_are_independent(self):
        engine = np.zeros((2, 78), dtype=bool); engine[:, [0, 34, 35]] = True
        legal, aliases = action_contract(engine, np.array([35, 35], dtype=np.int64),
                                         np.array([0, 1], dtype=np.int64))
        actor = np.zeros((2, 46), dtype=np.float32)
        actor[0, [0, 37]] = [.1, .9]; actor[1, :2] = [.7, .3]
        target = improvement_policy(actor, legal, aliases, .5)
        logits = torch.from_numpy(actor).clamp_min(1e-10).log().requires_grad_()
        terms = policy_terms(logits, torch.from_numpy(actor), torch.from_numpy(target),
                             torch.from_numpy(legal), torch.from_numpy(aliases))
        terms.loss.backward()
        self.assertEqual(terms.changed_fraction.item(), .5)
        self.assertLess(logits.grad[0].abs().max().item(), 1e-7)
        self.assertLess(logits.grad[1, 1].item(), -.1)

    def test_mask_precedes_normalization_and_illegal_logits_have_no_gradient(self):
        for bad in (1e30, float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad=bad):
                logits = torch.tensor([[0., 0., bad]], requires_grad=True)
                actor = torch.tensor([[.5, .5, 0.]])
                legal = torch.tensor([[True, True, False]])
                terms = policy_terms(logits, actor, actor, legal, torch.tensor([[True, False, False]]))
                terms.loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertEqual(logits.grad[0, 2].item(), 0)
                self.assertAlmostEqual(terms.loss.item(), np.log(2), places=6)

    def test_forward_kl_matches_reference_including_zero_actor_mass(self):
        logits = torch.tensor([[1., 2., 3.]], requires_grad=True)
        actor = torch.tensor([[.6, .4, 0.]])
        terms = policy_terms(logits, actor, actor, torch.ones((1, 3), dtype=torch.bool),
                             torch.tensor([[True, False, False]]))
        expected = torch.nn.functional.kl_div(logits.log_softmax(1), actor, reduction="batchmean")
        torch.testing.assert_close(terms.actor_kl, expected)
        terms.loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_frozen_targets_do_not_receive_gradients(self):
        logits, actor, target, legal, aliases = self.inputs(action=1)
        actor.requires_grad_(); target.requires_grad_()
        policy_terms(logits, actor, target, legal, aliases).loss.backward()
        self.assertIsNone(actor.grad); self.assertIsNone(target.grad)
        self.assertIsNotNone(logits.grad)

    def test_leash_pushes_drifted_policy_back_towards_actor(self):
        gradients = []
        for weight in (0., 1.):
            _, actor, target, legal, aliases = self.inputs()
            logits = torch.tensor([[.9, .1]]).log().requires_grad_()
            policy_terms(logits, actor, target, legal, aliases, actor_kl=weight).loss.backward()
            gradients.append(logits.grad.clone())
        self.assertGreater(gradients[0][0, 0].item(), 0)
        torch.testing.assert_close(gradients[1], 2 * gradients[0])

    def test_row_mean_gradient_does_not_reweight_small_shards(self):
        replay = example()
        a = replay.arrays
        _, aliases = action_contract(a["engine_legal"], a["engine_action"], a["stage"])
        inputs = [torch.from_numpy(a[k]) for k in ("actor_policy", "policy_target", "legal")]
        logits = torch.zeros((8, 46), requires_grad=True)
        policy_terms(logits, *inputs, torch.from_numpy(aliases)).loss.backward()
        reference = logits.grad.clone()
        logits.grad = None
        for rows in (slice(0, 1), slice(1, 8)):
            size = len(logits[rows])
            loss = policy_terms(logits[rows], *(v[rows] for v in inputs), torch.from_numpy(aliases[rows])).loss
            (loss * size / 8).backward()
        torch.testing.assert_close(logits.grad, reference)

    def test_forced_action_is_finite_and_has_no_policy_gradient(self):
        logits = torch.tensor([[3., -float("inf")]], requires_grad=True)
        probabilities = torch.tensor([[1., 0.]])
        legal = torch.tensor([[True, False]])
        terms = policy_terms(logits, probabilities, probabilities, legal, legal)
        terms.loss.backward()
        self.assertEqual(terms.loss.item(), 0)
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_invalid_options_and_nonfinite_legal_logits_are_rejected(self):
        for options in ({"policy_mode": "mcts"}, {"actor_kl": True}, {"actor_kl": -1},
                        {"actor_kl": float("nan")}, {"actor_kl": float("inf")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                policy_terms(*self.inputs(), **options)
            with self.assertRaises(ValueError):
                training.Options(**options).validate()
        values = list(self.inputs()); values[0] = torch.tensor([[float("nan"), 0.]])
        with self.assertRaises(FloatingPointError): policy_terms(*values)

    def test_invalid_probabilities_masks_and_shapes_are_rejected(self):
        changes = [(1, torch.tensor([[.8, .4]])), (2, torch.tensor([[1.1, -.1]])),
                   (3, torch.tensor([[False, False]])), (4, torch.tensor([[False, False]])),
                   (3, torch.ones((1, 2))), (0, torch.zeros(2)),
                   (1, torch.tensor([[float("nan"), .4]]))]
        for index, bad in changes:
            values = list(self.inputs()); values[index] = bad
            with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                policy_terms(*values)

    def test_legacy_loss_parity_on_real_replay_contract(self):
        replay = example(); net = TinyNet().eval(); rows = np.arange(8, dtype=np.int64)
        torch.testing.assert_close(replay_loss(replay, net, rows, policy_mode="all", actor_kl=0),
                                   replay.loss(net, rows), rtol=0, atol=0)

    def test_replay_bytes_are_unchanged_after_fitting(self):
        replay = example(); before = training.fingerprint(replay)
        net = TinyNet(); loss = replay_loss(replay, net, np.arange(8, dtype=np.int64))
        loss.backward()
        self.assertEqual(training.fingerprint(replay), before)
        replay.validate()

    def test_value_training_is_retained_when_every_policy_decision_agrees(self):
        replay = example()
        replay.arrays["engine_action"][:] = 0
        legal, aliases = action_contract(replay.arrays["engine_legal"], replay.arrays["engine_action"], replay.arrays["stage"])
        replay.arrays["policy_target"] = improvement_policy(replay.arrays["actor_policy"], legal, aliases, .5)
        replay.validate()
        net = TinyNet(); net.drop.p = 0
        with torch.no_grad():
            net.policy.weight.zero_(); net.policy.bias.zero_()
            net.policy.bias[:2].copy_(torch.tensor([.7, .3]).log())
        replay_loss(replay, net, np.arange(8, dtype=np.int64)).backward()
        self.assertLess(net.policy.bias.grad.abs().max().item(), 1e-7)
        self.assertGreater(net.value.weight.grad.abs().sum().item(), 0)

    def test_selected_value_head_and_shape_contract_are_preserved(self):
        class HeadNet(TinyNet):
            def __init__(self):
                super().__init__(); self.public = torch.nn.Linear(2, 1); self.called = None
            def value_only(self, x, head):
                self.called = head
                return self.public(x).squeeze(-1)
        replay = example(); replay.metadata["search"]["valued_by"] = "public"
        net = HeadNet()
        replay_loss(replay, net, np.arange(8, dtype=np.int64)).backward()
        self.assertEqual(net.called, "public")
        self.assertIsNotNone(net.public.weight.grad); self.assertIsNone(net.value.weight.grad)
        with patch.object(net, "value_only", return_value=torch.zeros((8, 1))):
            with self.assertRaisesRegex(ValueError, "contract"):
                replay_loss(replay, net, np.arange(8, dtype=np.int64))

    def test_invalid_rows_and_mask_mismatch_fail_before_forward(self):
        replay = example(); net = TinyNet()
        for rows in (np.array([], dtype=np.int64), np.array([8], dtype=np.int64),
                     np.array([-1], dtype=np.int64), np.array([0], dtype=np.int32)):
            with self.subTest(rows=rows), patch.object(net, "everything") as forward:
                with self.assertRaises(ValueError): replay_loss(replay, net, rows)
                forward.assert_not_called()
        replay.arrays["legal"][0, 2] = True
        with self.assertRaisesRegex(ValueError, "masks"):
            replay_loss(replay, net, np.array([0], dtype=np.int64))

    def test_serialized_conservative_resume_matches_uninterrupted_adam_and_rng(self):
        continuous = learner()
        for _ in range(3): continuous.epoch()
        interrupted = learner(); interrupted.epoch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            atomic_save(interrupted.checkpoint(), path)
            saved = torch.load(path, weights_only=True)
        torch.manual_seed(999)
        resumed = learner(saved=saved); resumed.epoch(); resumed.epoch()
        self.assert_tree_equal(continuous.checkpoint(), resumed.checkpoint())

    def test_old_checkpoint_migrates_only_to_exact_legacy_objective(self):
        options = training.Options(batch=3, lr=1e-3)
        first = learner(options); first.epoch(); saved = first.checkpoint()
        legacy = deepcopy(saved)
        del legacy["search_training"]["policy_objective_version"]
        for key in ("policy_mode", "actor_kl"): del legacy["search_training"]["options"][key]
        expected = learner(options, saved); expected.epoch()
        migrated = learner(options, legacy); migrated.epoch()
        self.assert_tree_equal(expected.checkpoint(), migrated.checkpoint())
        with self.assertRaisesRegex(ValueError, "options changed"):
            learner(saved=legacy)

    def test_resuming_cannot_silently_change_or_lose_policy_options(self):
        first = learner(); first.epoch(); saved = first.checkpoint()
        for overrides in ({"actor_kl": 2.0}, {"policy_mode": "all"}):
            options = dict(saved["search_training"]["options"]); options.update(overrides)
            with self.assertRaisesRegex(ValueError, "options changed"):
                learner(training.Options(**options), saved)
        for key in ("policy_mode", "actor_kl"):
            bad = deepcopy(saved); del bad["search_training"]["options"][key]
            with self.assertRaisesRegex(ValueError, "objective"):
                learner(saved=bad)
        for version in (True, 2, None):
            bad = deepcopy(saved); bad["search_training"]["policy_objective_version"] = version
            with self.assertRaisesRegex(ValueError, "objective"):
                learner(saved=bad)
        bad = deepcopy(saved); del bad["search_training"]["policy_objective_version"]
        with self.assertRaisesRegex(ValueError, "marker"):
            learner(saved=bad)

    def test_nonfinite_conservative_loss_prevents_step_and_checkpoint(self):
        current = learner()
        with patch.object(training, "replay_loss", return_value=torch.tensor(float("nan"))), \
             patch.object(current.optimizer, "step") as step:
            with self.assertRaises(FloatingPointError): current.epoch()
            step.assert_not_called()
        with self.assertRaises(RuntimeError): current.checkpoint()

    def test_cli_accepts_policy_mode_and_kl_weight(self):
        args = ["train_search", "--checkpoint", "actor.pt", "--replay", "replay", "--out", "run",
                "--policy-mode", "changed", "--actor-kl", "1.5"]
        with patch("sys.argv", args), patch.object(training, "run") as run:
            training.main()
        self.assertEqual(run.call_args.kwargs["overrides"], {"policy_mode": "changed", "actor_kl": 1.5})
        torch.set_num_threads(1)

    def test_real_runner_saves_and_inherits_conservative_options(self):
        def load(path, device):
            state = torch.load(path, weights_only=True)
            net = TinyNet(); net.load_state_dict(state["model"])
            return net, state
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "actor.pt"
            atomic_save({**TinyNet().state(), "generation": 0}, source)
            original = source.read_bytes()
            with patch.object(training.Dataset, "load", side_effect=lambda paths, every: training.Dataset([example()], every)), \
                 patch.object(training, "_load_network", side_effect=load), redirect_stdout(io.StringIO()):
                one = training.run(source, [], root / "one", epochs=1,
                                   overrides={"batch": 3, "policy_mode": "changed", "actor_kl": 1.5})
                two = training.run(one, [], root / "two", epochs=1, resume=True)
                with self.assertRaises(ValueError):
                    training.run(two, [], root / "bad", resume=True, overrides={"actor_kl": 0})
            saved = torch.load(two, weights_only=True)
            self.assertEqual(saved["search_training"]["options"]["policy_mode"], "changed")
            self.assertEqual(saved["search_training"]["options"]["actor_kl"], 1.5)
            self.assertEqual(saved["search_training"]["epochs"], 2)
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse((root / "bad").exists())
            self.assertFalse((root / "two/champion.pt").exists())


_NATIVE = all(importlib.util.find_spec(m) is not None for m in ("riichi_py", "libriichi"))


@unittest.skipUnless(_NATIVE, "requires native engines installed by repository CI")
class NativeTests(unittest.TestCase):
    def test_real_standalone_and_combined_conservative_training_resume(self):
        from neural import combined, model, mortal_model
        from neural.tests.test_search_replay import example as native_example
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            for joined in (False, True):
                with self.subTest(joined=joined), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    ours = model.PolicyValueNet(8, 1, actions=46)
                    net = combined.Combined(ours, mortal_model.build(16, 1)) if joined else ours
                    if joined:
                        net.mortal_config = {"resnet": {"conv_channels": 16, "num_blocks": 1}, "control": {"version": 4}}
                    atomic_save({**training._model_payload(net), "generation": 4}, root / "actor.pt")
                    replay = native_example()
                    replay.metadata["training_api_version"] = training.TRAINING_API_VERSION
                    replay.save(root / "replay")
                    options = dict(batch=2, validation_every=0, policy_mode="changed", actor_kl=1.)
                    with redirect_stdout(io.StringIO()):
                        all_at_once = training.run(root / "actor.pt", [root / "replay"], root / "continuous",
                                                   epochs=2, overrides=options)
                        one = training.run(root / "actor.pt", [root / "replay"], root / "one", epochs=1, overrides=options)
                        two = training.run(one, [root / "replay"], root / "two", epochs=1, resume=True)
                    a, b = (torch.load(path, weights_only=True) for path in (all_at_once, two))
                    for part in (("combined", "model", "mortal", "current_dqn") if joined else ("model",)):
                        for key in a[part]:
                            torch.testing.assert_close(a[part][key], b[part][key], rtol=0, atol=0)
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
