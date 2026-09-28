"""PPO compares matching behaviour laws, and diagnostics use successor inputs."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from neural import selfplay, mortal_learner
from neural.ppo_control import PolicyDrift
from neural.tests.test_evaluation_masks import ConflictingViews


class BehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_unchanged_exploring_policy_has_unit_ratios_and_zero_drift(self):
        logits=torch.tensor([[.99,.01]]).log().repeat(400,1)
        legal=torch.ones_like(logits,dtype=torch.bool)
        actions,recorded,forced=selfplay.explore(logits,legal,.25,np.random.default_rng(9))
        self.assertTrue(forced.any())
        recalculated=torch.distributions.Categorical(logits=logits).log_prob(actions)
        torch.testing.assert_close((recalculated-recorded).exp(),torch.ones(400),atol=1e-6,rtol=1e-6)
        guard=PolicyDrift(.02);self.assertFalse(guard.check(recorded[~forced],recalculated[~forced]))
        self.assertLess(guard.last,1e-10)

    def test_zero_coefficient_preserves_raw_likelihood_exactly(self):
        logits=torch.randn(3,4);legal=torch.ones_like(logits,dtype=torch.bool);actions=torch.tensor([0,1,2])
        expected=torch.distributions.Categorical(logits=logits).log_prob(actions)
        torch.manual_seed(1)
        chosen,recorded,forced=selfplay.explore(logits,legal,0.,np.random.default_rng(1))
        expected=torch.distributions.Categorical(logits=logits).log_prob(chosen)
        self.assertFalse(forced.any()); self.assertTrue(torch.equal(recorded,expected))

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
