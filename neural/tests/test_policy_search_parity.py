"""Regression contracts for the player the teacher actually simulates."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import contract, policy_inference, searched, zoo
from neural.collect_search import SearchSettings, metadata
from neural.search_replay import validate_metadata
from neural.training_safety import TRAINING_API_VERSION


class TiedNet(torch.nn.Module):
    kind, planes, actions, speaks_mortal = "mortal", 1012, 46, True
    channels, blocks = 8, 1

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.sizes = []

    def everything(self, planes, legal):
        self.sizes.append(len(planes))
        logits = torch.zeros_like(legal, dtype=torch.float32) + self.anchor
        return logits.masked_fill(~legal, -torch.inf), torch.zeros(len(planes), device=planes.device), torch.zeros(len(planes), 3, 34, device=planes.device)

    def forward(self, planes, legal):
        return self.everything(planes, legal)[:2]


class RootServer:
    order = contract.MortalServed.order
    to_engine_rows = contract.MortalServed.to_engine_rows

    def __init__(self, net):
        self.net = net
        self.contract = SimpleNamespace(speaks_our_moves=False)

    def root(self, arena, views, rows, deciding, legal, device):
        return torch.zeros(len(rows), 1012, 34), torch.from_numpy(zoo.translatable(legal[rows]))

    def after_reach(self, arena, rows, deciding, legal, device, follower=None):
        mask = np.zeros((len(rows), 46), dtype=bool)
        mask[:, :34] = legal[rows, 34:68]
        return torch.zeros(len(rows), 1012, 34), torch.from_numpy(mask)


class PolicyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_root_and_conditional_ties_match_ordinary_player(self):
        for moves in ([0, 1], [4, 13], [34, 35], [69, 70], [70, 71, 72, 73, 74], [75, 76, 77]):
            with self.subTest(moves=moves):
                legal = np.zeros((1, 78), dtype=bool)
                legal[0, moves] = True
                net = TiedNet().eval()
                server = RootServer(net)
                views = SimpleNamespace(observer=SimpleNamespace(follower=SimpleNamespace(tell=lambda *args: None)))
                def ask(who, fresh, allowed):
                    logits, _, _ = policy_inference.everything(net, torch.zeros(len(who), 1012, 34), torch.from_numpy(allowed))
                    return logits.detach().numpy(), allowed
                ordinary = zoo.choose_in_mortal_space(ask, views, np.array([0]), np.array([0]), legal)
                order, *_ = contract.root_order(server, None, views, np.array([0]), np.array([0]), legal, "cpu")
                self.assertEqual(int(ordinary[0]), int(order[0, 0]))
                self.assertEqual(len(set(order[0])), 78)

    def test_both_server_orders_preserve_first_policy_index(self):
        logits = torch.zeros(4, 46)
        for method in (contract.EngineServed.order, contract.MortalServed.order):
            np.testing.assert_array_equal(method(None, logits), np.tile(np.arange(46), (4, 1)))

    def test_disabled_search_matches_complete_ordinary_game(self):
        net = TiedNet().eval()
        options = dict(games=1, seed=53, worlds=1, candidates=2, margin=2., device="cpu", max_steps=4000)
        ordinary, _ = searched.play(net, searcher=None, **options)
        bypassed, tally = searched.play(net, searcher=0, sure=0., rollout_batch=1, **options)
        np.testing.assert_array_equal(ordinary, bypassed)
        self.assertEqual(tally, (0, 0))

    def test_precision_resolution_preserves_ordinary_layouts(self):
        self.assertEqual(policy_inference.precision("cpu", 46), "float32")
        self.assertEqual(policy_inference.precision("cuda:0", 46), "bfloat16")
        self.assertEqual(policy_inference.precision("cuda:1", 78), "float32")
        for device, actions in (("mps", 46), ("cpu", 47)):
            with self.assertRaises(ValueError):
                policy_inference.precision(device, actions)
        with patch.object(torch, "autocast") as cast:
            policy_inference.autocast("cuda:0", 46)
            cast.assert_called_once_with("cuda", dtype=torch.bfloat16, enabled=True)

    def test_cpu_policy_explicitly_disables_ambient_autocast(self):
        class Probe(TiedNet):
            def everything(self, planes, legal):
                self.autocast_enabled = torch.is_autocast_enabled("cpu")
                return super().everything(planes, legal)
        net = Probe()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            policy_inference.everything(net, torch.zeros(1, 1012, 34), torch.ones(1, 46, dtype=torch.bool))
            self.assertFalse(net.autocast_enabled)
            self.assertTrue(torch.is_autocast_enabled("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA hardware")
    def test_cuda_full_and_fast_policy_use_same_bfloat16_contract(self):
        from neural import combined, model, mortal_model
        torch.manual_seed(8)
        nets = [model.PolicyValueNet(8, 1, actions=46)]
        nets.append(combined.Combined(model.PolicyValueNet(8, 1, actions=46), mortal_model.build(16, 1)))
        for net in nets:
            net = net.cuda().eval()
            planes = torch.randn(3, 1012, 34, device="cuda")
            legal = torch.zeros(3, 46, dtype=torch.bool, device="cuda")
            legal[:, :8] = True
            with torch.no_grad():
                full = policy_inference.everything(net, planes, legal)[0]
                fast = policy_inference.policy_logits(net, planes, legal)
                torch.testing.assert_close(full, fast, rtol=0, atol=0)
                with policy_inference.autocast("cuda", 46):
                    self.assertTrue(torch.is_autocast_enabled("cuda"))
                    self.assertEqual(torch.get_autocast_dtype("cuda"), torch.bfloat16)


class RolloutBatchTests(unittest.TestCase):
    def run_rollout(self, limit):
        encoded, cloned = [], []
        class Copies:
            def __init__(self, identities):
                self.identities = identities
            @classmethod
            def from_follower(cls, follower, who, version):
                return cls(list(range(len(who))))
            def replace_concealed(self, *args): pass
            def reset_search_private(self, *args): pass
            def synchronize_search_decisions(self, *args): pass
            def feed_some(self, *args): pass
            def feed(self, *args): pass
            def clone_some(self, rows):
                cloned.append(len(rows))
                return Copies(rows)
            def encode_some(self, rows):
                encoded.append(len(rows))
                n = len(rows)
                return np.arange(n + 1, dtype=np.int64), np.zeros(n, dtype=np.uint16), np.asarray([row // 4 for row in rows], dtype=np.float16), None
            def encode(self):
                return self.encode_some(self.identities)
        class Arena:
            def __init__(self): self.applied = None
            def lookahead_slots(self): return [7]
            def lookahead_hands(self): return [[([[]] * 4) for _ in range(7)]]
            def lookahead_initial_state(self): return [[(0, [False] * 4) for _ in range(7)]]
            def lookahead_owed_mjai(self):
                if self.applied is not None: return [], [], [], b"", []
                mask = np.zeros((7, 78), dtype=np.uint8)
                mask[:, 34:36] = 1
                return [0] * 7, list(range(7)), [0] * 7, mask.tobytes(), [[] for _ in range(7)]
            def lookahead_apply(self, actions): self.applied = actions
        class ReachNet(TiedNet):
            def everything(self, planes, legal):
                logits, value, hands = super().everything(planes, legal)
                if not legal[:, 37].any():
                    tile = planes[:, 0, 0].long() % 2
                    logits[torch.arange(len(planes)), tile] = 1
                return logits, value, hands
        arena = Arena()
        net = ReachNet()
        server = object.__new__(contract.MortalServed)
        server.net, server._Imagined = net, Copies
        server.contract = SimpleNamespace(speaks_our_moves=False)
        server.count_crossings = False
        with contract.following(arena, object()):
            server.play_lookahead(arena, device="cpu", batch_size=limit)
        return arena.applied, encoded, cloned, net.sizes

    def test_every_forward_and_riichi_copy_respects_the_limit(self):
        for limit in (1, 2, 3, 8):
            with self.subTest(limit=limit):
                actions, encoded, cloned, sizes = self.run_rollout(limit)
                self.assertEqual(actions, [34 + n % 2 for n in range(7)])
                self.assertTrue(encoded and cloned and sizes)
                self.assertLessEqual(max(encoded + cloned + sizes), limit)
                self.assertEqual(sum(cloned), 7)
                self.assertEqual(sum(sizes), 14)

    def test_invalid_limit_fails_before_reconstructing_states(self):
        server = object.__new__(contract.MortalServed)
        for invalid in (0, -1, True, 1.5, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                server.play_lookahead(None, device="cpu", batch_size=invalid)


class InferenceEvidenceTests(unittest.TestCase):
    def sample(self):
        return metadata(games=2, seed=10, settings=SearchSettings(rollout_batch=3),
                        actor_sha256="a" * 64, source_revision="b" * 40,
                        training_api_version=TRAINING_API_VERSION)

    def test_new_replay_records_precision_order_and_batch(self):
        value = self.sample()
        validate_metadata(value)
        self.assertEqual(value["version"], 3)
        self.assertEqual(value["teacher"]["policy_inference"], policy_inference.describe("cpu"))
        self.assertEqual(value["search"]["rollout_batch"], 3)

    def test_missing_or_downgraded_inference_evidence_is_refused(self):
        for field in ("precision", "tie_break", "version"):
            bad = self.sample()
            del bad["teacher"]["policy_inference"][field]
            with self.assertRaises(ValueError): validate_metadata(bad)
        bad = self.sample(); bad["version"] = 2
        with self.assertRaises(ValueError): validate_metadata(bad)
        for invalid in (True, 0, -1, 1.5):
            bad = self.sample(); bad["search"]["rollout_batch"] = invalid
            with self.assertRaises(ValueError): validate_metadata(bad)

    def test_existing_version_two_replay_keeps_its_contract(self):
        old = deepcopy(self.sample())
        old["version"] = 2
        del old["teacher"]["policy_inference"]
        del old["search"]["rollout_batch"]
        validate_metadata(old)


if __name__ == "__main__":
    unittest.main()
