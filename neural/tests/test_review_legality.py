"""Teacher distributions retain every core-legal reach and second discard."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


class Views:
    def __init__(self, *args):
        self.observer=SimpleNamespace(follower=self)
        self.told=[]
    def advance(self): pass
    def prepare(self, *args): pass
    def tell(self, *args): self.told.append(args)
    def sparse_and_masks(self, rows, players):
        from neural.observe import Planes
        masks=np.zeros((len(rows),46),bool);masks[:,0]=True
        return Planes.from_follower(np.zeros(len(rows)+1,np.int64),[],[]),masks
    def encode(self, who, after_reach=False):
        masks=np.zeros((len(who),46),bool);masks[:,0]=True
        return np.zeros(len(who)+1,np.int64),[],[],masks


class LegalityReviewTests(unittest.TestCase):
    def test_imitation_preserves_core_reach_even_when_follower_refuses_it(self):
        from neural.imitate import teacher_distribution
        views=Views();seen=[]
        def forward(planes,mask):
            seen.append(mask.cpu().numpy())
            logits=torch.zeros_like(mask,dtype=torch.float32)
            logits[:,37]=20;logits[:,1]=10
            return logits.masked_fill(~mask,-torch.inf),torch.zeros(len(mask))
        teacher=SimpleNamespace(actions=46,kind='mortal',forward=forward)
        legal=np.zeros((1,78),bool);legal[0,[0,34,35]]=True
        odds=teacher_distribution(teacher,views,np.array([0]),np.array([0]),legal)
        self.assertTrue(seen[0][0,37])
        self.assertEqual(np.flatnonzero(seen[1][0]).tolist(),[0,1])
        self.assertGreater(odds[0,35],.99)
        self.assertAlmostEqual(float(odds.sum()),1,places=6)
        self.assertEqual(views.told,[])

    def test_reheading_keeps_core_first_and_second_masks(self):
        from neural import rehead
        legal=np.zeros((1,78),bool);legal[0,[0,34,35]]=True
        class Arena:
            def __init__(self,**kwargs):self.finished=False
            def seats(self):return bytes([255 if self.finished else 0])
            def seat_players(self):return bytes([0,1,2,3])
            def legal_mask(self):return legal.astype(np.uint8).tobytes()
            def step(self,actions):self.finished=True
        def teacher(planes,mask):
            logits=torch.zeros_like(mask,dtype=torch.float32)
            logits[:,35]=30
            return logits.masked_fill(~mask,-torch.inf),torch.zeros(len(mask))
        with patch.object(rehead.riichi_py,'Arena',Arena),patch.object(rehead,'Views',Views), \
             patch.object(torch.distributions.Categorical,'sample',lambda self:self.probs.argmax(-1)):
            _,masks,targets,declared=rehead.collect(teacher,object(),1,1,'cpu',1.)
        self.assertEqual(declared,1)
        self.assertTrue(masks[0,37])
        self.assertEqual(np.flatnonzero(masks[1]).tolist(),[0,1])
        self.assertGreater(targets[0,37],.99)
        self.assertGreater(targets[1,1],.99)

    def test_reheading_does_not_duplicate_red_five_alias_probability(self):
        from neural import rehead, zoo
        legal=np.zeros((1,78),bool);legal[0,[0,4]]=True
        class Arena:
            def __init__(self,**kwargs):self.finished=False
            def seats(self):return bytes([255 if self.finished else 0])
            def seat_players(self):return bytes([0,1,2,3])
            def legal_mask(self):return legal.astype(np.uint8).tobytes()
            def step(self,actions):self.finished=True
        def teacher(planes,mask):
            return torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,-torch.inf),torch.zeros(len(mask))
        with patch.object(rehead.riichi_py,'Arena',Arena),patch.object(rehead,'Views',Views):
            _,_,targets,_=rehead.collect(teacher,object(),1,1,'cpu',1.)
        # 0.5 on tile 0, 0.5 split across ordinary/red aliases of tile 4.
        self.assertAlmostEqual(float(targets[0,0]),.5,places=6)
        self.assertAlmostEqual(float(targets[0,4]+targets[0,34]),.5,places=6)
        self.assertAlmostEqual(float(targets.sum()),1,places=6)
