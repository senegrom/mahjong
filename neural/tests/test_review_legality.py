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

    def test_reheading_puts_a_fives_whole_weight_on_the_plain_five(self):
        # Split across the plain five and Mortal's red one, a five the
        # teacher preferred by less than two to one lost the student's
        # argmax to the runner-up: 0.69% of the teacher's choices, all of
        # them fives. Our tables have no red fives, so the red one means
        # nothing and is never opened.
        from neural import rehead, zoo
        legal=np.zeros((1,78),bool);legal[0,[0,4]]=True
        class Arena:
            def __init__(self,**kwargs):self.finished=False
            def seats(self):return bytes([255 if self.finished else 0])
            def seat_players(self):return bytes([0,1,2,3])
            def legal_mask(self):return legal.astype(np.uint8).tobytes()
            def step(self,actions):self.finished=True
        def teacher(planes,mask):
            logits=torch.full(mask.shape,-torch.inf)
            logits[:,0]=np.log(.4);logits[:,4]=np.log(.6)
            return logits,torch.zeros(len(mask))
        with patch.object(rehead.riichi_py,'Arena',Arena),patch.object(rehead,'Views',Views):
            _,masks,targets,_=rehead.collect(teacher,object(),1,1,'cpu',1.)
        self.assertEqual(np.flatnonzero(masks[0]).tolist(),[0,4])
        self.assertFalse(masks[:,list(zoo.MORTAL_RED_FIVES)].any())
        np.testing.assert_allclose(targets[0,[0,4]],[.4,.6],atol=1e-6)
        self.assertEqual(int(targets[0].argmax()),4)
        self.assertAlmostEqual(float(targets.sum()),1,places=6)

    def test_mortals_red_fives_mean_nothing_at_our_tables(self):
        from neural import mortal_learner, zoo
        from neural.tests.test_evaluation_masks import ConflictingViews
        for red in zoo.MORTAL_RED_FIVES:
            self.assertEqual(zoo.meanings(red),[])
        # Every five and nothing else is ours to discard.
        legal=np.zeros((1,78),bool);legal[0,[4,13,22]]=True
        allowed=zoo.translatable(legal)
        self.assertEqual(np.flatnonzero(allowed[0]).tolist(),[4,13,22])
        self.assertEqual(zoo.first_meaning(np.array(zoo.MORTAL_RED_FIVES),legal.repeat(3,0)).tolist(),[-1]*3)
        # A network that rates a red five highest still plays a plain one:
        # the red one is closed, as the learner records it and as a table
        # asks for a move.
        def score(planes,mask):
            logits=torch.zeros(mask.shape);logits[:,34]=30;logits[:,35]=20;logits[:,13]=10
            return logits.masked_fill(~mask,-torch.inf)
        choice,records=mortal_learner.decide_in_mortal_space(score,ConflictingViews(),
            np.array([0]),np.array([0]),legal,greedy=True,device='cpu')
        self.assertEqual(choice.tolist(),[13]);self.assertEqual(records.actions.tolist(),[13])
        self.assertFalse(records.masks[:,list(zoo.MORTAL_RED_FIVES)].any())
        asked=[]
        def ask(who,fresh,allowed):
            asked.append(allowed.copy())
            return score(None,torch.from_numpy(allowed)).numpy(),np.zeros_like(allowed)
        self.assertEqual(zoo.choose_in_mortal_space(ask,ConflictingViews(),np.array([0]),np.array([0]),legal).tolist(),[13])
        self.assertFalse(asked[0][:,list(zoo.MORTAL_RED_FIVES)].any())
