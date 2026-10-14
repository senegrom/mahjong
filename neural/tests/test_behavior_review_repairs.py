"""PPO compares matching behaviour laws, and diagnostics use successor inputs."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from neural import selfplay, mortal_learner
from neural.tests.test_evaluation_masks import ConflictingViews


class BehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_both_reach_stages_record_the_policys_own_likelihood(self):
        views=ConflictingViews();legal=np.zeros((1,78),bool);legal[0,[0,34,35]]=True
        def score(planes,mask):
            logits=torch.zeros_like(mask,dtype=torch.float32);logits[:,37]=20;logits[:,1]=10
            return logits.masked_fill(~mask,-torch.inf)
        rng=SimpleNamespace(random=lambda n:np.full(n,.9))
        with patch.object(torch.distributions.Categorical,'sample',lambda d:d.probs.argmax(-1)):
            _,records=mortal_learner.decide_in_mortal_space(score,views,np.array([0]),np.array([0]),
                legal,device='cpu',explore_share=.25,wanderer=rng)
        self.assertEqual(records.actions.tolist(),[37,1])
        np.testing.assert_array_equal(records.forced,[False,False])
        logits=score(records.planes.dense('cpu'),torch.from_numpy(records.masks))
        reread=torch.distributions.Categorical(logits=logits).log_prob(torch.from_numpy(records.actions))
        torch.testing.assert_close(reread,torch.from_numpy(records.log_probs),atol=1e-6,rtol=1e-6)

    def test_a_native_rollout_flags_its_forced_moves(self):
        from neural.tests.test_selfplay_contract import FirstLegal
        batch=selfplay.play(FirstLegal(),games=1,seed=808,device='cpu',explore_share=.5)
        self.assertEqual(len(batch.explored),batch.decisions)
        self.assertTrue(bool(batch.explored.any()))
        self.assertFalse(bool(batch.explored.all()))

if __name__=='__main__':unittest.main()
