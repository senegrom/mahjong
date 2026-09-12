"""PPO compares matching behaviour laws, and diagnostics use successor inputs."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from neural import selfplay, mortal_learner, counterfactual
from neural.ppo_control import PolicyDrift
from neural.observe import Planes
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

    def test_both_reach_stages_record_their_actual_coefficients(self):
        views=ConflictingViews();legal=np.zeros((1,78),bool);legal[0,[0,34,35]]=True
        def score(planes,mask):
            logits=torch.zeros_like(mask,dtype=torch.float32);logits[:,37]=20;logits[:,1]=10
            return logits.masked_fill(~mask,-torch.inf)
        rng=SimpleNamespace(random=lambda n:np.full(n,.9))
        with patch.object(torch.distributions.Categorical,'sample',lambda d:d.probs.argmax(-1)):
            _,records=mortal_learner.decide_in_mortal_space(score,views,np.array([0]),np.array([0]),
                legal,device='cpu',explore_share=.25,wanderer=rng)
        self.assertEqual(records.actions.tolist(),[37,1])
        np.testing.assert_array_equal(records.epsilon,[.25,0.])
        logits=score(records.planes.dense('cpu'),torch.from_numpy(records.masks))
        reread=torch.distributions.Categorical(logits=logits).log_prob(torch.from_numpy(records.actions))
        torch.testing.assert_close(reread,torch.from_numpy(records.log_probs),atol=1e-6,rtol=1e-6)

    def test_native_rollout_tracks_successors_and_coefficients(self):
        from neural.tests.test_selfplay_contract import FirstLegal
        batch=selfplay.play(FirstLegal(),games=1,seed=808,device='cpu',explore_share=.5)
        self.assertEqual(len(batch.behaviour_epsilon),batch.decisions)
        self.assertEqual(len(batch.after_exploration),batch.decisions)
        self.assertFalse(bool(batch.after_exploration[0]))
        self.assertTrue(bool(batch.after_exploration.any()))
        self.assertTrue(bool((batch.behaviour_epsilon==.5).all()))

    def test_counterfactual_groups_successors_not_the_current_coin(self):
        net=SimpleNamespace(kind='mortal',planes=1012,actions=78,everything=lambda *a:None,
                            value_only=lambda planes,head:torch.zeros(len(planes)))
        batch=SimpleNamespace(after_exploration=torch.tensor([False,True]),
            explored=torch.tensor([True,False]),returns=torch.tensor([10.,20.]),decisions=2,
            observations=Planes.from_follower(np.array([0,0,0]),[],[]),legal=torch.ones(2,78,dtype=torch.bool))
        with patch.object(selfplay,'play',return_value=batch):
            report=counterfactual.measure(net,1,1,.5,'cpu')
        self.assertEqual(report['value_error']['after_forced_action']['rmse'],20.)
        self.assertEqual(report['value_error']['other_decisions']['rmse'],10.)
        self.assertNotIn('coverage_is_the_fault',report)
        self.assertIn('not established',report['coverage_diagnosis'])


if __name__=='__main__':unittest.main()
