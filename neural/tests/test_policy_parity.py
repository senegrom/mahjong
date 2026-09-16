"""Ordinary/search choices, explicit arithmetic and bounded rollout contracts."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import riichi_py

from neural import contract, zoo, searched, policy_inference as inference
from neural.observe import Planes
from neural.collect_search import SearchSettings, metadata
from neural.search_replay import validate_metadata


class TiedServed:
    """Controlled positions; actual serving sort, root order and translation."""
    order = contract.MortalServed.order
    to_engine_rows = contract.MortalServed.to_engine_rows
    def __init__(self, reach=False):
        self.net = self
        self.actions = 46
        self.contract = SimpleNamespace(speaks_our_moves=False)
        self.reach = reach
    def root(self, arena, views, rows, deciding, mask, device):
        return torch.zeros((len(rows), 1), device=device), torch.from_numpy(zoo.translatable(mask[rows])).to(device)
    def after_reach(self, arena, rows, deciding, mask, device, follower=None):
        allowed = np.zeros((len(rows), 46), bool); allowed[:, :34] = mask[rows, 34:68]
        return torch.ones((len(rows), 1), device=device), torch.from_numpy(allowed).to(device)
    def everything(self, x, legal):
        logits = torch.zeros_like(legal, dtype=torch.float32)
        if self.reach and not x.any(): logits[:, 37] = 10
        return logits.masked_fill(~legal, -torch.inf), x[:, 0], torch.zeros((len(x), 3, 34), device=x.device)
    def ask(self, who, fresh, allowed):
        x = torch.full((len(who), 1), float(fresh))
        return self.everything(x, torch.from_numpy(allowed))[0].numpy(), allowed


class PolicyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_ties_match_ordinary_moves_and_both_riichi_stages(self):
        cases = [(False, [0, 1], 0), (True, [0, 1, 34, 35], 34),
                 (False, [4, 13], 4), (False, [70, 74], 74), (False, [68], 68)]
        for reach, actions, expected in cases:
            with self.subTest(actions=actions):
                served = TiedServed(reach)
                mask = np.zeros((1, 78), bool); mask[0, actions] = True
                views = SimpleNamespace(observer=SimpleNamespace(follower=SimpleNamespace(tell=lambda *a: None)))
                rows = np.array([0], np.int64)
                ordinary = zoo.choose_in_mortal_space(served.ask, views, rows, rows, mask)[0]
                order, *_ = contract.root_order(served, None, views, rows, rows, mask, 'cpu')
                self.assertEqual(int(ordinary), expected)
                self.assertEqual(int(order[0, 0]), int(ordinary))
                self.assertEqual(len(set(order[0])), 78)

    def test_stable_sort_matches_argmax_in_both_servers(self):
        for width, server in ((46, contract.MortalServed), (78, contract.EngineServed)):
            logits = torch.zeros(7, width)
            for i in range(7): logits[i, :i] = -torch.inf
            ranked = server.order(None, logits)
            np.testing.assert_array_equal(ranked[:, 0], logits.argmax(1).numpy())

    def test_explicit_context_overrides_ambient_autocast_and_restores_it(self):
        net = SimpleNamespace(actions=46)
        self.assertEqual(inference.describe(net, 'cuda:0')['resolved'], 'bfloat16')
        self.assertEqual(inference.describe(net, 'cpu')['resolved'], 'float32')
        self.assertEqual(inference.describe(SimpleNamespace(actions=78), 'cuda')['resolved'], 'float32')
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with inference.context(net, 'cpu'):
                self.assertFalse(torch.is_autocast_enabled('cpu'))
            self.assertTrue(torch.is_autocast_enabled('cpu'))
            with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                with inference.context(net, 'cpu'): raise RuntimeError('sentinel')
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        inference.configure(net, 'bfloat16', 'cpu')
        with inference.context(net, 'cpu'):
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertFalse(torch.is_autocast_enabled('cpu'))

    def test_real_combined_and_standalone_policy_paths_share_arithmetic(self):
        from neural import model, combined, mortal_model
        for joined in (False, True):
            net = model.PolicyValueNet(8, 1, actions=46).eval()
            if joined: net = combined.Combined(net, mortal_model.build(16, 1)).eval()
            x = torch.randn(3, 1012, 34); legal = torch.zeros(3, 46, dtype=torch.bool); legal[:, :9] = True
            for precision in ('auto', 'float32', 'bfloat16'):
                with self.subTest(joined=joined, precision=precision), torch.no_grad():
                    inference.configure(net, precision, 'cpu')
                    with inference.context(net, 'cpu'): full = net.everything(x, legal)[0].float()
                    fast = contract.policy_logits(net, x, legal)
                    torch.testing.assert_close(fast, full, rtol=0, atol=0)
                    rows = np.arange(3, dtype=np.int64)
                    sparse = SimpleNamespace(dense=lambda device: x.to(device))
                    views = SimpleNamespace(sparse_and_masks=lambda *a, **kw: (sparse, legal.numpy()))
                    player = net if joined else zoo.MortalSpacePlayer(net, 'cpu')
                    normal, _ = player._ask(views, list(zip(rows, rows)), False, legal.numpy())
                    torch.testing.assert_close(torch.from_numpy(normal), fast, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA hardware')
    def test_cuda_real_policy_parity(self):
        from neural import model, combined, mortal_model
        for joined in (False, True):
            net = model.PolicyValueNet(8, 1, actions=46)
            if joined: net = combined.Combined(net, mortal_model.build(16, 1))
            net = net.cuda().eval(); x = torch.randn(3, 1012, 34, device='cuda')
            legal = torch.ones((3, 46), dtype=torch.bool, device='cuda')
            for precision in ('auto', 'float32', 'bfloat16'):
                inference.configure(net, precision, 'cuda')
                with torch.no_grad(), inference.context(net, 'cuda'): expected = net.everything(x, legal)[0].float()
                with torch.no_grad(): got = contract.policy_logits(net, x, legal)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
                sparse = SimpleNamespace(dense=lambda device: x)
                views = SimpleNamespace(sparse_and_masks=lambda *a, **kw: (sparse, legal.cpu().numpy()))
                player = net if joined else zoo.MortalSpacePlayer(net, 'cuda')
                normal, _ = player._ask(views, [(0, 0), (1, 1), (2, 2)], False, legal.cpu().numpy())
                torch.testing.assert_close(torch.from_numpy(normal).cuda(), expected, rtol=0, atol=0)

    def test_inference_metadata_is_validated_and_legacy_is_not_relabelled(self):
        settings = SearchSettings(policy_precision='float32', rollout_batch=7)
        m = metadata(games=2, seed=20, settings=settings, actor_sha256='a'*64,
                     source_revision='b'*40, training_api_version=2, device='cpu')
        self.assertEqual(m['teacher']['policy_inference']['resolved'], 'float32')
        validate_metadata(m)
        for key in ('policy_inference',):
            bad = deepcopy(m); del bad['teacher'][key]
            with self.assertRaises(ValueError): validate_metadata(bad)
        for key in ('rollout_batch', 'policy_precision'):
            bad = deepcopy(m); del bad['search'][key]
            with self.assertRaises(ValueError): validate_metadata(bad)
        bad = deepcopy(m); bad['teacher']['policy_inference']['resolved'] = 'bfloat16'
        with self.assertRaises(ValueError): validate_metadata(bad)
        legacy = deepcopy(m); legacy['version'] = 1
        with self.assertRaises(ValueError): validate_metadata(legacy)
        for bad in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError): inference.validate(rollout_batch=bad)
        for bad in (None, True, 'fp16'):
            with self.assertRaises(ValueError): inference.validate(bad)

    def test_search_bypassed_is_exactly_the_ordinary_game_on_tied_logits(self):
        from neural import model
        net = model.PolicyValueNet(8, 1, actions=46).eval()
        with torch.no_grad():
            for module in (net.policy_tiles, net.policy_pooled):
                for parameter in module.parameters(): parameter.zero_()
        options = dict(net=net, games=2, seed=219, worlds=1, candidates=2,
                       margin=2., device='cpu', sure=0., policy_precision='float32')
        baseline, _ = searched.play(searcher=None, **options)
        bypassed, tally = searched.play(searcher=0, **options)
        np.testing.assert_array_equal(baseline, bypassed)
        self.assertEqual(tally, (0, 0))

    def test_cloud_forwards_precision_and_batching_and_binds_experiment_identity(self):
        import json
        from pathlib import Path
        import tempfile
        from neural.tests.test_search_cloud_publication import SearchCloudPublicationTests
        helper = SearchCloudPublicationTests()
        with tempfile.TemporaryDirectory() as temp:
            volume = helper.prepare(Path(temp))
            command, = helper.run_controller(volume, policy_precision='float32', rollout_batch=7)
            self.assertEqual(command[command.index('--policy-precision') + 1], 'float32')
            self.assertEqual(command[command.index('--rollout-batch') + 1], '7')
            identity, = (volume / 'searched-records').glob('*/experiment.json')
            saved = json.loads(identity.read_text())
            self.assertEqual(saved['settings']['policy_precision'], 'float32')
            self.assertEqual(saved['settings']['rollout_batch'], 7)
            for options in ({'policy_precision': 'fp16'}, {'rollout_batch': True}, {'rollout_batch': 0}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    helper.run_controller(volume, **options)

    def test_sibling_evidence_refuses_precision_or_batch_mixing(self):
        from neural.sibling_head import gathered
        info = inference.describe(SimpleNamespace(actions=46), 'cpu')
        first = SimpleNamespace(meta={'policy_inference': info, 'rollout_batch': 7})
        for second in ({}, {'policy_inference': info, 'rollout_batch': 8},
                       {'policy_inference': {**info, 'resolved': 'bfloat16'}, 'rollout_batch': 7}):
            with self.subTest(meta=second), self.assertRaisesRegex(ValueError, 'inference'):
                gathered([first, SimpleNamespace(meta=second)])


class RolloutBatchTests(unittest.TestCase):
    def fake_mortal_rollout(self, batch_size):
        n = 11; calls = []; seen_encode = []; clone_sizes = []
        class Copies:
            @classmethod
            def from_follower(cls, follower, who, version): return cls()
            def replace_concealed(self, *a): pass
            def reset_search_private(self, *a): pass
            def synchronize_search_decisions(self, *a): pass
            def encode_some(self, which):
                seen_encode.append(len(which))
                return self.sparse(which)
            def sparse(self, which):
                count=len(which)
                return np.arange(count+1), np.zeros(count, dtype=np.uint16), np.asarray([x//4+1 for x in which],np.float32), None
            def clone_some(self, which):
                clone_sizes.append(len(which)); other=Copies(); other.which=which; return other
            def feed(self, *args): pass
            def encode(self): return self.sparse(self.which)
        legal = np.zeros((n,78), bool); legal[:, :2] = True; legal[::2,34:36] = True
        class Arena:
            done=False
            def lookahead_slots(self): return [n]
            def lookahead_initial_state(self): return [[(0,[False]*4)]*n]
            def lookahead_hands(self): return [[[[]]*4]*n]
            def lookahead_owed_mjai(self):
                if self.done: return [],[],[],b'',[]
                return [0]*n,list(range(n)),[1]*n,legal.tobytes(),[[]]*n
            def lookahead_decision_state(self): return [False]*n,[None]*n
            def lookahead_apply(self, actions): self.actions=actions; self.done=True
        class Net:
            kind='mortal'; planes=1012; actions=46
            everything=lambda *a: None
            def policy_only(self, planes, mask):
                calls.append(len(planes))
                values=torch.zeros_like(mask,dtype=torch.float32)
                values[torch.arange(len(planes)),planes[:,0,0].long()%2]=2
                values[:,37]=10
                return values.masked_fill(~mask,-torch.inf)
        arena=Arena(); net=Net(); served=contract.serve(net); served._Imagined=Copies
        original=Planes.dense
        def dense(planes,device):
            self.assertLessEqual(len(planes),batch_size)
            return original(planes,device)
        with contract.following(arena,object()), patch.object(Planes,'dense',dense):
            served.play_lookahead(arena,device='cpu',batch_size=batch_size)
        self.assertLessEqual(max(calls),batch_size)
        self.assertLessEqual(max(seen_encode),batch_size)
        self.assertLessEqual(max(clone_sizes),batch_size)
        return arena.actions

    def test_every_mortal_forward_and_riichi_copy_obeys_limit_and_mapping(self):
        expected=[(34 if i%2==0 else 0)+(i+1)%2 for i in range(11)]
        for limit in (1,3,50):
            with self.subTest(limit=limit): self.assertEqual(self.fake_mortal_rollout(limit),expected)

    def test_engine_action_rollouts_are_bounded_too(self):
        n=9; sizes=[]
        x=np.zeros((n,riichi_py.PLANES,34),np.float32)
        legal=np.zeros((n,78),bool); legal[:,:2]=True
        class Arena:
            done=False
            def lookahead_owed(self): return (b'',b'',0) if self.done else (x.tobytes(),legal.tobytes(),n)
            def lookahead_apply(self, actions): self.done=True; self.actions=actions
        class Net:
            kind='engine'; planes=riichi_py.PLANES; actions=78
            everything=lambda *a: None
            def policy_only(self,x,legal):
                sizes.append(len(x)); return torch.zeros_like(legal,dtype=torch.float32).masked_fill(~legal,-torch.inf)
        arena=Arena()
        searched.play_lookahead(Net(),arena,device='cpu',batch_size=2)
        self.assertEqual(sizes,[2,2,2,2,1]); self.assertEqual(arena.actions,[0]*n)

    def test_invalid_limits_fail_before_native_mutation(self):
        for limit in (0,-1,True,2.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                searched.search_with_value_head(None,None,[[0,1]],[],worlds=2,candidates=2,
                    margin=2.,hurried=False,rollout_batch=limit)

    def test_real_mortal_rollout_forwards_limit_without_changing_live_state(self):
        from neural import model
        from neural.observe import Views
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            net = model.PolicyValueNet(8, 1, actions=46).eval()
            arena = riichi_py.Arena(games=1, seed=100, bot_places=[])
            views = Views(arena, 1, {'mortal'}); views.advance()
            served = contract.serve(net)
            mask = np.frombuffer(arena.legal_mask(), np.uint8).reshape(1, 78).astype(bool)
            ranked = [np.flatnonzero(mask[0])[:2].tolist()]
            before = arena.observations(); sizes = []
            original = contract.policy_logits
            def bounded(net, planes, legal):
                sizes.append(len(planes))
                self.assertLessEqual(len(planes), 1)
                return original(net, planes, legal)
            with contract.following(arena, views.observer.follower), patch.object(contract, 'policy_logits', side_effect=bounded):
                chosen = searched.search_with_value_head(net, arena, ranked, [1/34]*102,
                    worlds=3, candidates=2, margin=2., hurried=False, device='cpu', pool=1,
                    played_by='network', served=served, rollout_batch=1, leaf_batch=2)
            self.assertTrue(sizes)
            self.assertIn(chosen[0], ranked[0])
            self.assertEqual(arena.observations(), before)
        finally: torch.set_num_threads(threads)


if __name__=='__main__': unittest.main()
