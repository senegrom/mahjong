"""Search training's persisted semantics, including both riichi decisions."""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural.collect_search import SearchSettings, metadata
from neural.search_replay import (
    ACTIONS, MEANINGS, SearchReplay, action_contract, improvement_policy,
)


def example():
    m = metadata(games=1, seed=7, settings=SearchSettings(), actor_sha256="a" * 64,
                 source_revision="b" * 40, training_api_version=1)
    m["rows"] = 3
    engine = np.zeros((3, 78), dtype=bool)
    engine[0, [4, 5]] = True
    engine[1:, [1, 35]] = True
    actions = np.array([4, 35, 35], dtype=np.int64)
    stages = np.array([0, 0, 1], dtype=np.int64)
    legal, aliases = action_contract(engine, actions, stages)
    actor = np.zeros((3, 46), dtype=np.float32)
    actor[0, [4, 34, 5]] = [.6, .2, .2]
    actor[1, [1, 37]] = [.3, .7]
    actor[2, 1] = 1
    a = {
        "root_indptr": np.arange(4, dtype=np.int64) * 2,
        "root_indices": np.tile(np.array([0, 4], dtype=np.uint16), 3),
        "root_values": np.ones(6, dtype=np.float16),
        "legal": legal, "actor_policy": actor,
        "policy_target": improvement_policy(actor, legal, aliases, .5),
        "engine_legal": engine, "engine_action": actions,
        "returns": np.array([1., -.5, -.5], dtype=np.float32),
        "game": np.zeros(3, dtype=np.int64), "player": np.array([0, 1, 1], dtype=np.int64),
        "step": np.array([1, 2, 2], dtype=np.int64), "stage": stages,
    }
    return SearchReplay(m, a)


class ActionTargets(unittest.TestCase):
    def test_red_plain_aliases_share_mass_without_doubling(self):
        replay = example()
        replay.validate()
        np.testing.assert_allclose(replay.arrays["policy_target"][0, [4, 34, 5]], [.675, .225, .1])
        np.testing.assert_allclose(replay.arrays["policy_target"].sum(axis=1), 1)

    def test_zero_actor_mass_uses_uniform_aliases_and_target_is_frozen(self):
        actor = np.array([[0., 0., 1.]], dtype=np.float32)
        legal = np.ones((1, 3), dtype=bool)
        aliases = np.array([[True, True, False]])
        target = improvement_policy(actor, legal, aliases, .5)
        actor[:] = 7
        np.testing.assert_array_equal(target, [[.25, .25, .5]])

    def test_riichi_declares_then_discards_and_uses_different_masks(self):
        replay = example()
        self.assertEqual(np.nonzero(replay.arrays["legal"][1])[0].tolist(), [1, 37])
        self.assertEqual(np.nonzero(replay.arrays["legal"][2])[0].tolist(), [1])
        self.assertAlmostEqual(float(replay.arrays["policy_target"][1, 37]), .85, places=6)
        self.assertEqual(float(replay.arrays["policy_target"][2, 1]), 1.)

    def test_priority_not_just_membership_defines_an_executable_alias(self):
        engine = np.zeros((1, 78), dtype=bool)
        engine[0, [76, 77]] = True
        with self.assertRaisesRegex(ValueError, "cannot be expressed"):
            action_contract(engine, np.array([77]), np.array([0]))
        legal, aliases = action_contract(engine, np.array([76]), np.array([0]))
        self.assertTrue(legal[0, 42] and aliases[0, 42])

    def test_illegal_and_wrong_stage_actions_fail(self):
        engine = np.zeros((1, 78), dtype=bool)
        engine[0, 4] = True
        for action, stage in ((5, 0), (-1, 0), (78, 0), (4, 1), (4, 2)):
            with self.subTest(action=action, stage=stage), self.assertRaises(ValueError):
                action_contract(engine, np.array([action]), np.array([stage]))

    def test_invalid_actor_and_improvement_coefficients_fail(self):
        legal = np.array([[True, False]])
        aliases = legal.copy()
        for actor in ([[float("nan"), 0.]], [[.5, .5]], [[-.1, 0.]], [[.5, 0.]]):
            with self.subTest(actor=actor), self.assertRaises(ValueError):
                improvement_policy(np.array(actor), legal, aliases, .5)
        for coefficient in (-1, 1.1, float("nan"), float("inf")):
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                improvement_policy(np.array([[1., 0.]]), legal, aliases, coefficient)


class ShardValidation(unittest.TestCase):
    def test_roundtrip_is_checksummed_and_memory_mapped(self):
        replay = example()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            folder = Path(temporary) / "round"
            replay.save(folder)
            reopened = SearchReplay.load(folder)
            self.assertEqual(replay.metadata, reopened.metadata)
            for name in replay.arrays:
                self.assertIsInstance(reopened.arrays[name], np.memmap)
                np.testing.assert_array_equal(replay.arrays[name], reopened.arrays[name])
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(len(manifest["sha256"]), len(replay.arrays))

    def test_existing_shard_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            folder = Path(temporary) / "round"
            example().save(folder)
            before = (folder / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                example().save(folder)
            self.assertEqual(before, (folder / "manifest.json").read_bytes())

    def test_publish_failure_leaves_no_complete_manifest(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            folder = Path(temporary) / "round"
            with patch("neural.search_replay.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    example().save(folder)
            self.assertFalse((folder / "manifest.json").exists())
            with self.assertRaises(FileNotFoundError):
                SearchReplay.load(folder)

    def test_checksum_rejects_changed_payload(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            folder = Path(temporary) / "round"
            example().save(folder)
            with (folder / "returns.npy").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                SearchReplay.load(folder)

    def test_legacy_recordings_and_unknown_versions_are_not_relabelled(self):
        for change in ({"version": 999}, {"version": True}, {"complete": False},
                       {"actor_sha256": "latest.pt"}, {"source_revision": "main"},
                       {"reward": {"version": 2, "name": "placement_only"}},
                       {"actions": {"policy": 78, "engine": 78}},
                       {"observation": {"planes": 97, "positions": 34, "encoder_version": 4}}):
            replay = example()
            replay.metadata.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                replay.validate()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            Path(temporary, "meta.json").write_text('{"complete": true}')
            with self.assertRaises(FileNotFoundError):
                SearchReplay.load(Path(temporary))

    def test_invalid_search_settings_are_rejected_before_collection(self):
        for change in ({"worlds": 0}, {"pool": True}, {"candidates": 79}, {"depth": -2},
                       {"margin": float("nan")}, {"temperature": -1}, {"improve": 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                metadata(games=1, seed=7, settings=SearchSettings(**change), actor_sha256="a" * 64,
                         source_revision="b" * 40, training_api_version=1)

    def test_exact_shapes_dtypes_and_finite_values(self):
        alterations = {
            "returns": lambda a: a.reshape(-1, 1),
            "game": lambda a: a.astype(np.float32),
            "root_indices": lambda a: np.full_like(a, 65535),
            "root_values": lambda a: np.full_like(a, np.nan),
            "root_indptr": lambda a: np.array([0, 2, 1, 6], dtype=np.int64),
            "actor_policy": lambda a: np.full_like(a, np.nan),
            "policy_target": lambda a: np.zeros_like(a),
        }
        for name, mutate in alterations.items():
            replay = example()
            replay.arrays[name] = mutate(replay.arrays[name])
            with self.subTest(field=name), self.assertRaises(ValueError):
                replay.validate()

    def test_duplicate_or_unsorted_sparse_columns_are_rejected(self):
        for first_row in ([0, 0], [4, 0]):
            replay = example()
            replay.arrays["root_indices"][:2] = first_row
            with self.subTest(columns=first_row), self.assertRaises(ValueError):
                replay.validate()

    def test_even_tiny_illegal_target_mass_is_rejected(self):
        replay = example()
        replay.arrays["policy_target"][0, 44] = 1e-8
        with self.assertRaises(ValueError):
            replay.validate()

    def test_masks_cannot_be_reconstructed_with_an_incompatible_mapping(self):
        replay = example()
        replay.arrays["legal"][2, 37] = True
        with self.assertRaisesRegex(ValueError, "masks"):
            replay.validate()

    def test_both_riichi_stages_must_identify_the_same_actual_transition(self):
        for name, value in (("game", 1), ("player", 3), ("step", 3), ("returns", 2.), ("stage", 0)):
            replay = example()
            replay.arrays[name][2] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                replay.validate()

    def test_normal_moves_must_not_gain_a_second_stage(self):
        replay = example()
        replay.arrays["engine_action"][1:] = 1
        with self.assertRaises(ValueError):
            replay.validate()


class SupervisedLoss(unittest.TestCase):
    def test_policy_and_selected_evaluator_update_without_ppo(self):
        class Observations:
            def rows(self, rows):
                self.n = len(rows)
                return self
            def dense(self, device):
                return torch.ones((self.n, 2), device=device)
        class Learner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.policy = torch.nn.Linear(2, ACTIONS)
                self.public = torch.nn.Linear(2, 1)
                self.critic = torch.nn.Linear(2, 1)
            def everything(self, x, mask):
                return self.policy(x), self.public(x).squeeze(1), None
            def value_only(self, x, head):
                self.last_head = head
                return self.critic(x).squeeze(1)
        replay = example()
        learner = Learner()
        optimizer = torch.optim.SGD(learner.parameters(), lr=.01)
        before = learner.policy.weight.detach().clone()
        wanted = replay.arrays["policy_target"].copy()
        with patch.object(SearchReplay, "observations", return_value=Observations()):
            loss = replay.loss(learner, np.arange(3, dtype=np.int64))
            optimizer.zero_grad()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(learner.last_head, "critic")
            self.assertGreater(float(learner.critic.weight.grad.abs().sum()), 0.)
            self.assertIsNone(learner.public.weight.grad)
            self.assertEqual(float(learner.policy.weight.grad[44].abs().sum()), 0.)
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in learner.parameters()))
            optimizer.step()
        self.assertFalse(torch.equal(before, learner.policy.weight))
        np.testing.assert_array_equal(replay.arrays["policy_target"], wanted)


class CollectorContract(unittest.TestCase):
    """Run the actual collection loop against a tiny, controlled engine bridge.

    These tests isolate the collector's accounting and adapters; the native
    tests below separately exercise the real rules, follower and networks.
    """
    def run_collection(self, max_steps=3):
        import neural
        from neural.collect_search import collect

        class Sparse:
            def __init__(self, indptr, indices, values):
                self.indptr, self.indices, self.values = indptr, indices, values
            def arrays(self):
                return {name: getattr(self, name) for name in ("indptr", "indices", "values")}
            @staticmethod
            def cat(blocks):
                ptr, columns, values = [0], [], []
                for block in blocks:
                    ptr.extend((block.indptr[1:] + ptr[-1]).tolist())
                    columns.extend(block.indices.tolist())
                    values.extend(block.values.tolist())
                return Sparse(np.asarray(ptr, dtype=np.int64), np.asarray(columns, dtype=np.uint16),
                              np.asarray(values, dtype=np.float16))

        class Arena:
            def __init__(self, **kwargs):
                self.turn = 0
            def all_finished(self):
                return self.turn == 3
            def seats(self):
                return bytes([self.turn if self.turn < 3 else 255])
            def seat_players(self):
                return bytes([0, 1, 2, 3])
            def legal_mask(self):
                legal = np.zeros(78, dtype=np.uint8)
                legal[([4, 5], [1, 35], [0, 1])[self.turn]] = 1
                return legal.tobytes()
            def step(self, actions):
                self.last = actions[0]
                self.turn += 1

        class Views:
            def __init__(self, *args):
                self.observer = SimpleNamespace(follower=object())
            def advance(self):
                pass
            def prepare(self, *args):
                pass

        class Ledger:
            def __init__(self, games):
                self.values = []
                self.pending = []
            def open(self, game, player):
                self.pending.append(len(self.values))
                self.values.append(0.)
            def settle(self, arena):
                for row in self.pending:
                    self.values[row] = float(arena.turn)
                self.pending.clear()
            def close(self, arena):
                self.assert_finished = arena.all_finished()
                if not self.assert_finished or self.pending:
                    raise AssertionError("Account closed before settlement")
                return np.asarray(self.values, dtype=np.float32)

        class Net:
            def eval(self):
                return self
            def everything(self, x, legal):
                logits = torch.zeros_like(legal, dtype=torch.float32).masked_fill(~legal, -torch.inf)
                return logits, torch.zeros(len(x)), torch.zeros(len(x), 3, 34)

        class Served:
            contract = SimpleNamespace(reads="mortal", planes=1012, answers=46)
            def __init__(self, net):
                self.net = net
            def root(self, arena, views, rows, deciding, engine, device):
                x = torch.zeros(len(rows), 1012, 34)
                x[:, 0, 0] = arena.turn + 1
                selected = np.array([np.nonzero(engine[0])[0][0]])
                legal, _ = action_contract(engine, selected, np.zeros(1, dtype=np.int64))
                return x, torch.from_numpy(legal)
            def after_reach(self, arena, rows, deciding, engine, device, **kwargs):
                x = torch.zeros(len(rows), 1012, 34)
                x[:, 0, 0] = 20  # distinguish the actual conditional input
                legal, _ = action_contract(engine, np.array([35]), np.ones(1, dtype=np.int64))
                return x, torch.from_numpy(legal)

        registry = {}
        def remember(arena, follower):
            registry[id(arena)] = follower
        def root_order(served, arena, views, rows, deciding, mask, device):
            x, legal = served.root(arena, views, rows, deciding, mask, device)
            logits, value, hands = served.net.everything(x, legal)
            if mask[:, 34:68].any():
                served.after_reach(arena, rows, deciding, mask, device, follower=views.observer.follower)
            allowed = np.nonzero(mask[0])[0].tolist()
            order = np.array([allowed + [a for a in range(78) if a not in allowed]])
            return order, logits, value, hands
        def search(net, arena, *args, **kwargs):
            self.assertIn(id(arena), registry)
            return [[4], [35], [0]][arena.turn]
        def finished(arena, **kwargs):
            if not arena.all_finished():
                raise RuntimeError("unfinished controlled collection")
        def module(name, **attributes):
            result = ModuleType(name)
            result.__dict__.update(attributes)
            return result
        modules = {
            "riichi_py": module("riichi_py", Arena=Arena, HANDS=102),
            "neural.contract": module("neural.contract", serve=Served, root_order=root_order,
                                     remember_follower=remember, _FOLLOWERS=registry,
                                     forget_follower=lambda arena: registry.pop(id(arena), None)),
            "neural.ledger": module("neural.ledger", Ledger=Ledger, REWARD_VERSION=1, HAND_SCALE=1 / 4000),
            "neural.searched": module("neural.searched", search_with_value_head=search),
            "neural.observe": module("neural.observe", Planes=Sparse, Views=Views),
            "neural.outcomes": module("neural.outcomes", require_finished=finished, validate_budget=lambda *a: None),
            "neural.training_safety": module("neural.training_safety", TRAINING_API_VERSION=1,
                                            require_training_engine=lambda: None),
        }
        self.registry = registry
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            for name, value in modules.items():
                if name.startswith("neural."):
                    stack.enter_context(patch.object(neural, name.split(".")[1], value, create=True))
            return collect(Net(), games=1, seed=7, settings=SearchSettings(max_steps=max_steps),
                           actor_sha256="a" * 64, source_revision="b" * 40)

    def test_actual_collector_links_conditional_input_and_completed_outcome(self):
        replay = self.run_collection()
        replay.validate()
        self.assertFalse(self.registry)
        np.testing.assert_array_equal(replay.arrays["returns"], [1., 2., 2., 3.])
        np.testing.assert_array_equal(replay.arrays["engine_action"], [4, 35, 35, 0])
        np.testing.assert_array_equal(replay.arrays["stage"], [0, 0, 1, 0])
        np.testing.assert_array_equal(replay.arrays["root_values"], [1., 2., 20., 3.])
        np.testing.assert_array_equal(replay.arrays["player"], [0, 1, 1, 2])

    def test_failed_collection_does_not_return_a_partial_training_round(self):
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            self.run_collection(max_steps=1)
        self.assertFalse(self.registry)


_NATIVE = all(importlib.util.find_spec(name) is not None for name in ("riichi_py", "libriichi"))


@unittest.skipUnless(_NATIVE, "requires both native engines; repository CI installs them")
class NativeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_schema_meanings_match_production_translator(self):
        from neural import zoo
        self.assertEqual(MEANINGS, tuple(tuple(zoo.meanings(a)) for a in range(46)))
        # Exercise every engine action separately, not only a happy-path discard.
        for action in range(78):
            engine = np.zeros((1, 78), dtype=bool)
            engine[0, action] = True
            legal, _ = action_contract(engine, np.array([action]), np.array([0]))
            np.testing.assert_array_equal(legal, zoo.translatable(engine))

    def test_actual_combined_policy_and_value_have_finite_gradients(self):
        from neural import combined, model, mortal_model
        net = combined.Combined(model.PolicyValueNet(8, 1, actions=46), mortal_model.build(16, 1))
        net.set_mode("none")
        net.train()
        replay = example()
        loss = replay.loss(net, np.arange(3, dtype=np.int64))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.fuse.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.fuse.value_fix.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_complete_native_game_can_be_reopened_and_trained(self):
        from neural import contract, model
        from neural.collect_search import collect
        net = model.PolicyValueNet(8, 1, actions=46)
        before = len(contract._FOLLOWERS)
        # One candidate keeps this complete-game regression inexpensive. The
        # next test exercises multi-candidate native search and failure cleanup.
        replay = collect(net, games=1, seed=7, settings=SearchSettings(worlds=1, candidates=1),
                         actor_sha256="a" * 64, source_revision="b" * 40)
        self.assertEqual(len(contract._FOLLOWERS), before)
        self.assertGreater(replay.metadata["rows"], 0)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            folder = Path(temporary) / "game"
            replay.save(folder)
            reopened = SearchReplay.load(folder)
            net.train()
            rows = np.arange(min(8, replay.metadata["rows"]), dtype=np.int64)
            loss = reopened.loss(net, rows)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))

    def test_incomplete_multi_candidate_search_returns_no_dataset(self):
        from neural import contract, model
        from neural.collect_search import collect
        from neural.outcomes import IncompleteGamesError
        before = len(contract._FOLLOWERS)
        with self.assertRaises(IncompleteGamesError):
            collect(model.PolicyValueNet(8, 1, actions=46), games=1, seed=7,
                    settings=SearchSettings(worlds=2, candidates=2, max_steps=1),
                    actor_sha256="a" * 64, source_revision="b" * 40)
        self.assertEqual(len(contract._FOLLOWERS), before)


if __name__ == "__main__":
    unittest.main()
