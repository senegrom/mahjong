"""Regression tests for frozen targets, evaluator selection and continuation inputs."""
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
import riichi_py
from neural import contract, distil, model, searched, worlds
from neural.observe import Planes
from neural.search_learning import improvement_targets, masked_policy_loss


class SearchLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_masked_loss_and_gradients_are_finite_and_target_is_frozen(self):
        legal = torch.tensor([[True, True, False]])
        actor = torch.tensor([[.8, .2, 0.]])
        fixed = improvement_targets(actor, torch.tensor([1]), legal, .5)
        original = fixed.clone()
        actor.fill_(123.)
        raw = torch.tensor([[1., 2., 3.]], requires_grad=True)
        loss = masked_policy_loss(raw.masked_fill(~legal, -torch.inf), fixed, legal)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertEqual(float(raw.grad[0,2]), 0.)
        torch.testing.assert_close(fixed, original)
        torch.testing.assert_close(fixed, torch.tensor([[.4,.6,0.]]))
        # Once the student moves, this is not just scaled hard-label supervision.
        hard = .5 * (torch.softmax(raw.detach()[:,:2],1) - torch.tensor([[0.,1.]]))
        self.assertFalse(torch.allclose(raw.grad[:,:2], hard))

    def test_invalid_targets_fail_instead_of_becoming_nan(self):
        legal = torch.tensor([[True,False]])
        for actor, action in [(torch.tensor([[.5,.5]]),0),
                              (torch.tensor([[float('nan'),0.]]),0),
                              (torch.tensor([[1.,0.]]),1)]:
            with self.assertRaises(ValueError):
                improvement_targets(actor,torch.tensor([action]),legal,.5)

    def test_outcomes_train_the_same_critic_search_reads(self):
        net=model.PolicyValueNet(8,1,riichi_py.PLANES,False)
        planes=torch.randn(2,riichi_py.PLANES,34)
        value=net.value_only(planes,head='critic')
        torch.nn.functional.mse_loss(value,torch.tensor([-1.,1.])).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in net.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in net.value.parameters()))

    def test_real_distillation_updates_on_narrow_masks_without_nan(self):
        from neural.checkpoints import atomic_save
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); checkpoint=root/'origin.pt'; out=root/'out'
            net=model.PolicyValueNet(8,1,riichi_py.PLANES,False)
            before={k:v.detach().clone() for k,v in net.state_dict().items()}
            atomic_save({'model':before,**net.payload_fields(),'generation':0},checkpoint)
            legal=np.zeros((2,78),bool);legal[:,:2]=True
            collected=(np.zeros((2,riichi_py.PLANES,34),np.float32),legal,np.array([1,0]),
                       np.zeros((2,3,34),np.float32),np.array([0,1]),np.array([-1.,1.],np.float32),
                       np.pad(np.array([[.8,.2],[.3,.7]],np.float32),((0,0),(0,76))))
            argv=['distil','--resume',str(checkpoint),'--out',str(out),'--rounds','1',
                  '--games','1','--batch','2','--epochs','2','--valued-by','critic']
            with patch.object(sys,'argv',argv),patch.object(distil,'collect',return_value=collected), \
                 patch.object(distil,'measure',return_value={'placement':2.5,'score':0.,'wins':0.}), \
                 patch.object(torch.cuda,'is_available',return_value=False),contextlib.redirect_stdout(io.StringIO()):
                distil.main()
            saved=torch.load(out/'latest.pt',weights_only=True)
            record=json.loads((out/'log.jsonl').read_text().splitlines()[0])
            self.assertTrue(np.isfinite(record['loss']))
            self.assertEqual(record['optimizer_updates'],2)
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in saved['model'].items() if k.startswith('critic.')))
            self.assertTrue(all(torch.isfinite(v).all() for v in saved['model'].values()))

    def test_selected_head_and_native_wanted_mask_are_forwarded(self):
        for head, result in [('critic',1.),('public',2.),('mean',3.)]:
            class Arena:
                def imagine(self,*a,**kw):return b'',[1]
                def leaves_from(self,*a,**kw):return b'',[1],[0.],[1]
                def decide(self,values,*a):return values
            calls=[]
            def leaves(*args,**kwargs):
                self.assertEqual(kwargs['wanted'],[1])
                return torch.zeros(1,1,1)
            def value(planes,head):
                calls.append(head);return torch.tensor([result])
            served=SimpleNamespace(contract=SimpleNamespace(reads='mortal'),leaves=leaves,value=value)
            got=searched.search_with_value_head(object(),Arena(),[[0]],[],worlds=1,candidates=1,
                margin=2.,hurried=True,device='cpu',pool=1,valued_by=head,served=served)
            self.assertEqual(got,[result]);self.assertEqual(calls,[head])


class ContinuationTests(unittest.TestCase):
    def net(self, actions=46, planes=1012):
        return SimpleNamespace(kind='mortal',planes=planes,actions=actions,everything=lambda *a:None)

    def test_nonterminal_without_history_fails_before_zeros_are_evaluated(self):
        served=contract.serve(self.net())
        arena=SimpleNamespace(leaves_mjai=lambda:([0],[[]]))
        for wanted in ([1], b'\x01', bytearray([1]), memoryview(b'\x01')):
            with self.subTest(wanted=type(wanted).__name__):
                with self.assertRaisesRegex(contract.UnsupportedSearchLayout,'nonterminal'):
                    served.leaves(arena,b'',[1],'cpu',wanted=wanted)
        for wanted in ([0], b'\x00', bytearray([0]), memoryview(b'\x00')):
            terminal=served.leaves(arena,b'',[1],'cpu',wanted=wanted)
            self.assertEqual(tuple(terminal.shape),(1,1012,34))

    def test_mortal_observations_with_engine_actions_keep_their_78_mask(self):
        net=self.net(actions=78); served=contract.serve(net)
        legal=np.zeros((1,78),bool);legal[0,[1,35]]=True
        sparse=Planes.from_follower(np.array([0,0]),[],[])
        views=SimpleNamespace(sparse_and_masks=lambda *a:(sparse,np.ones((1,46),bool)))
        planes,mask=served.root(None,views,np.array([0]),np.array([0]),legal,'cpu')
        self.assertEqual(tuple(mask.shape),(1,78));self.assertTrue(served.contract.speaks_our_moves)
        self.assertEqual(served.to_engine(35,legal[0]),35)
        np.testing.assert_array_equal(served.to_engine_rows(np.array([[35,1,0]]),legal),[[35,1,-1]])

    def test_unknown_mortal_dimensions_are_rejected(self):
        for net in (self.net(45),self.net(46,1000)):
            with self.assertRaises(contract.UnsupportedSearchLayout):contract.serve(net)


class ResamplingTests(unittest.TestCase):
    def test_common_negative_offset_does_not_change_resampling(self):
        for seed in range(30):
            a=worlds.resample(np.array([0.,-2.]),20,np.random.default_rng(seed))
            b=worlds.resample(np.array([-1000.,-1002.]),20,np.random.default_rng(seed))
            self.assertEqual(a.kept,b.kept);self.assertAlmostEqual(a.efficiency,b.efficiency)

    def test_fallback_honours_requested_survivor_count(self):
        got=worlds.resample(np.array([np.nan,np.nan]),20,np.random.default_rng(1))
        self.assertEqual(len(got.kept),20);self.assertEqual(len(got.weights),20)
        self.assertAlmostEqual(sum(got.weights),1.)
        with self.assertRaises(ValueError):worlds.resample(np.zeros((2,2)),1,np.random.default_rng())


if __name__=='__main__':unittest.main()
