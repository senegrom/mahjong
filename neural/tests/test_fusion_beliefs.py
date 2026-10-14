"""A belief head saved with one constant an opponent loads as the tile-wise
head with the same probabilities, and its optimiser's moments follow it.
test_fusion_heads checks that the head can change a distribution, and
what trains it."""
from types import SimpleNamespace
import unittest

import torch

from neural.combined import Fuse, migrate_belief_optimizer


class FusionBeliefTests(unittest.TestCase):
    def test_legacy_model_preserves_probabilities_and_adam_moments_migrate(self):
        torch.manual_seed(7)
        head=Fuse(8,phi=16,width=8)
        old=dict(head.state_dict())
        old['hands_out.weight']=torch.randn(3,8,1)
        old['hands_out.bias']=torch.randn(3)
        phi=torch.randn(2,16); guessed=torch.randn(2,3,34)
        constant=torch.nn.functional.conv1d(head.hands_fix(phi).unsqueeze(2),
                     old['hands_out.weight'],old['hands_out.bias'])
        expected=(guessed+constant).softmax(2)
        head.load_state_dict(old)
        torch.testing.assert_close(head.read_hands(phi,guessed).softmax(2),expected)
        optimiser=torch.optim.AdamW(head.parameters(),lr=.001)
        saved=optimiser.state_dict()
        for key,parameter in zip(saved['param_groups'][0]['params'],head.parameters()):
            if parameter is head.hands_out.weight: shape=(3,8,1)
            elif parameter is head.hands_out.bias: shape=(3,)
            else: continue
            saved['state'][key]={'step':torch.tensor(3.), 'exp_avg':torch.ones(shape)*.1,
                                 'exp_avg_sq':torch.ones(shape)*.2}
        migrated=migrate_belief_optimizer(saved,optimiser,SimpleNamespace(fuse=head))
        optimiser.load_state_dict(migrated)
        for parameter in (head.hands_out.weight,head.hands_out.bias):
            self.assertEqual(optimiser.state[parameter]['exp_avg'].shape,parameter.shape)
            self.assertEqual(optimiser.state[parameter]['step'].item(),3.)
        head.read_hands(phi,guessed).square().mean().backward();optimiser.step()
        clone=Fuse(8,phi=16,width=8);clone.load_state_dict(head.state_dict())
        torch.testing.assert_close(clone.read_hands(phi,guessed),head.read_hands(phi,guessed))
