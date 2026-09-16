"""Ordinary/search tie parity, explicit precision and both rollout batch stages."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import riichi_py

from neural import combined, contract, inference, model, mortal_model, searched, zoo
from neural.collect_search import metadata, SearchSettings
from neural.observe import Planes, Views
from neural.search_replay import validate_metadata
from neural.teacher_options import validate_controls


class TiedServed:
    kind = 'mortal'
    def __init__(self, reach=False):
        self.net = self
        self.contract = SimpleNamespace(speaks_our_moves=False)
        self.reach = reach
    def root(self, arena, views, rows, deciding, legal, device):
        return torch.zeros((len(rows), 1)), torch.from_numpy(zoo.translatable(legal[rows]))
    def after_reach(self, arena, rows, deciding, legal, device, follower=None):
        allowed = np.zeros((len(rows), 46), bool)
        allowed[:, :34] = legal[rows, 34:68]
        return torch.ones((len(rows), 1)), torch.from_numpy(allowed)
    def everything(self, planes, allowed):
        logits = torch.zeros_like(allowed, dtype=torch.float32)
        if self.reach and not planes.any(): logits[:, 37] = 10
        return logits.masked_fill(~allowed, -torch.inf), torch.zeros(len(planes)), torch.zeros(len(planes), 3, 34)
    order = contract.MortalServed.order
    to_engine_rows = contract.MortalServed.to_engine_rows
    def ask(self, who, fresh, allowed):
        return self.everything(torch.full((len(who), 1), float(fresh)), torch.from_numpy(allowed))[0].numpy(), allowed


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_tied_discards_calls_aliases_and_both_riichi_stages_match_ordinary_play(self):
        for actions, reach in [([0, 1], False), ([4, 13, 22], False), ([70, 71, 74], False),
                               ([0, 1, 34, 35], True), ([4, 13, 38, 47], True)]:
            with self.subTest(actions=actions):
                served = TiedServed(reach)
                legal = np.zeros((1, 78), bool); legal[0, actions] = True
                views = SimpleNamespace(observer=SimpleNamespace(follower=SimpleNamespace(tell=lambda *a: None)))
                rows = np.array([0], np.int64)
                ordinary = zoo.choose_in_mortal_space(served.ask, views, rows, rows, legal)
                order, *_ = contract.root_order(served, None, views, rows, rows, legal, 'cpu')
                self.assertEqual(order[0, 0], ordinary[0])
                self.assertEqual(len(set(order[0])), 78)
        scores = torch.zeros((2, 78))
        engine = SimpleNamespace()
        np.testing.assert_array_equal(contract.EngineServed.order(engine, scores)[:, 0], [0, 0])

    def test_real_combined_root_and_ordinary_action_match_on_native_positions(self):
        self.check_native_parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA hardware')
    def test_cuda_real_combined_root_and_ordinary_action_match(self):
        self.check_native_parity('cuda')

    def check_native_parity(self, device):
        torch.manual_seed(71)
        net = combined.Combined(model.PolicyValueNet(8, 1, actions=46), mortal_model.build(16, 1)).to(device).eval()
        served = contract.serve(net)
        arena = riichi_py.Arena(games=2, seed=13, bot_places=[])
        views = Views(arena, 2, {'mortal'})
        saw_call = False
        for _ in range(60):
            views.advance()
            seats = np.frombuffer(arena.seats(), np.uint8)
            rows = np.flatnonzero(seats != 255)
            if not len(rows): break
            players = np.frombuffer(arena.seat_players(), np.uint8).reshape(2, 4)
            deciding = players[rows, seats[rows]].astype(np.int64)
            legal = np.frombuffer(arena.legal_mask(), np.uint8).reshape(2, 78).astype(bool)
            views.prepare(rows, deciding)
            with torch.no_grad():
                order, *_ = contract.root_order(served, arena, views, rows, deciding, legal, device)
                ordinary = net.choose(views, rows, deciding, legal[rows])
            np.testing.assert_array_equal(order[rows, 0], ordinary)
            saw_call |= bool(legal[:, 70].any())
            actions = np.zeros(2, np.int64); actions[rows] = ordinary
            arena.step(actions.tolist())
        self.assertTrue(saw_call)

    def test_precision_is_explicit_and_outer_autocast_cannot_change_cpu_policy(self):
        net = model.PolicyValueNet(8, 1, actions=46).eval()
        x = torch.randn(2, 1012, 34); allowed = torch.ones(2, 46, dtype=torch.bool)
        with torch.no_grad():
            reference = inference.everything(net, x, allowed)[0]
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = inference.everything(net, x, allowed)[0]
                fast = inference.policy_logits(net, x, allowed)
                self.assertTrue(torch.is_autocast_enabled('cpu'))
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(fast, reference, rtol=0, atol=0)
        self.assertEqual(inference.describe('mortal', 'cuda:1')['policy_dtype'], 'bfloat16')
        self.assertEqual(inference.describe('engine', 'cuda')['policy_dtype'], 'float32')
        self.assertEqual(actual.dtype, torch.float32)

    def test_inference_and_batch_provenance_are_validated_and_not_downgraded(self):
        m = metadata(games=2, seed=12, settings=SearchSettings(rollout_batch=7),
                     actor_sha256='a'*64, source_revision='b'*40, training_api_version=2, device='cuda')
        self.assertEqual(m['teacher']['policy_inference']['policy_dtype'], 'bfloat16')
        for field in ('policy_inference',):
            bad = deepcopy(m); del bad['teacher'][field]
            with self.assertRaises(ValueError): validate_metadata(bad)
        for value in (0, True, -1, 1.5):
            bad = deepcopy(m); bad['search']['rollout_batch'] = value
            with self.assertRaises(ValueError): validate_metadata(bad)
            with self.assertRaises(ValueError): validate_controls(rollout_batch=value)
        for value in (True, 2, None):
            bad = deepcopy(m); bad['teacher']['policy_inference']['version'] = value
            with self.assertRaises(ValueError): validate_metadata(bad)
        old = deepcopy(m); del old['search']['rollout_batch']; del old['teacher']['policy_inference']
        validate_metadata(old)  # Historical evidence stays unmarked, never silently upgraded.
        self.assertNotIn('policy_inference', old['teacher'])

    def test_mortal_rollout_encodes_and_runs_both_stages_in_bounded_chunks(self):
        def run(limit):
            seen = []; n = 7
            class Copies:
                def __init__(self, ids=None): self.ids = ids
                @classmethod
                def from_follower(cls, *args): return cls()
                def replace_concealed(self, *args): pass
                def reset_search_private(self, *args): pass
                def encode_some(self, ids):
                    self_outer.assertLessEqual(len(ids), limit)
                    # One feature identifies the slot, so mapping/reordering is checked.
                    return np.arange(len(ids)+1), np.zeros(len(ids), np.uint16), np.array([i//4 for i in ids], np.float32), None
                def clone_some(self, ids):
                    self_outer.assertLessEqual(len(ids), limit)
                    return Copies(ids)
                def feed(self, *args): pass
                def encode(self): return self.encode_some(self.ids)
            class Net:
                kind='mortal'; planes=1012; actions=46
                def everything(self, *args): raise AssertionError('unused auxiliary heads')
                def policy_only(self, x, allowed):
                    seen.append((len(x), bool(allowed[:, 37].any())))
                    self_outer.assertFalse(torch.is_autocast_enabled('cpu'))
                    score = torch.zeros_like(allowed, dtype=torch.float32)
                    score[:, 37] = 20
                    index = x[:, 0, 0].long() % 2
                    score[torch.arange(len(x)), index] = 10
                    return score.masked_fill(~allowed, -torch.inf)
            class Arena:
                def __init__(self): self.actions=None
                def lookahead_slots(self): return [n]
                def lookahead_initial_state(self): return [[(0, [False]*4)]*n]
                def lookahead_hands(self): return [[[[]]*4]*n]
                def lookahead_owed_mjai(self):
                    if self.actions is not None: return [], [], [], b'', []
                    masks=np.zeros((n,78), np.uint8); masks[:, [0, 1, 34, 35]]=1
                    return [0]*n, list(range(n)), [0]*n, masks.tobytes(), [[]]*n
                def lookahead_apply(self, actions): self.actions=actions
            self_outer=self; arena=Arena(); served=contract.serve(Net()); served._Imagined=Copies
            dense = Planes.dense
            def bounded(x, device):
                self.assertLessEqual(len(x),limit)
                return dense(x, device)
            with contract.following(arena, object()), patch.object(Planes, 'dense', bounded):
                served.play_lookahead(arena, device='cpu', batch_size=limit)
            self.assertTrue(any(second for _, second in seen))
            self.assertTrue(any(not second for _, second in seen))
            self.assertLessEqual(max(size for size,_ in seen),limit)
            return arena.actions
        expected = [34+i%2 for i in range(7)]
        self.assertEqual(run(1), expected); self.assertEqual(run(2), expected); self.assertEqual(run(7), expected)

    def test_engine_rollout_obeys_batch_and_preserves_order(self):
        n=7; masks=np.ones((n,78),np.uint8); planes=np.zeros((n,riichi_py.PLANES,34),np.float32)
        planes[:,0,0]=np.arange(n)
        arena=SimpleNamespace(lookahead_owed=lambda:(planes.tobytes(),masks.tobytes(),n),
                              lookahead_apply=lambda actions: self.assertEqual(actions,list(range(n))))
        class Net:
            kind='engine'; planes=riichi_py.PLANES; actions=78
            def __call__(self,x,allowed):
                self_outer.assertLessEqual(len(x),2)
                return -abs(torch.arange(78)[None,:]-x[:,0,:1]), None
        self_outer=self
        searched.play_lookahead(Net(),arena,device='cpu',passes=1,batch_size=2)
        for size in (0,True,-1,1.5):
            with self.assertRaises(ValueError): searched.play_lookahead(Net(),None,batch_size=size)
            served=contract.serve(TiedServed())
            with self.assertRaises(ValueError): served.play_lookahead(None,batch_size=size)

if __name__ == '__main__': unittest.main()
