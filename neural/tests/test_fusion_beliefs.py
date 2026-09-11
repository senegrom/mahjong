"""A fused belief head must alter tile distributions, not their common offset."""
from types import SimpleNamespace
import unittest

import torch

from neural.combined import Fuse, migrate_belief_optimizer


class FusionBeliefTests(unittest.TestCase):
    def test_tile_distribution_changes_and_receives_gradients(self):
        torch.manual_seed(9)
        head=Fuse(8, phi=16, width=8)
        phi=torch.randn(4,16,requires_grad=True)
        guessed=torch.randn(4,3,34)
        initial=head.read_hands(phi,guessed)
        torch.testing.assert_close(initial,guessed)
        optimiser=torch.optim.AdamW(head.parameters(),lr=.01)
        wanted=torch.randint(34,(4,3))
        loss=torch.nn.functional.cross_entropy(initial.reshape(-1,34),wanted.reshape(-1))
        loss.backward()
        self.assertGreater(head.hands_out.weight.grad.abs().max().item(),1e-6)
        self.assertIsNone(phi.grad)
        optimiser.step();optimiser.zero_grad()
        changed=head.read_hands(phi,guessed)
        self.assertGreater((changed.softmax(2)-guessed.softmax(2)).abs().max().item(),1e-5)
        torch.nn.functional.cross_entropy(changed.reshape(-1,34),wanted.reshape(-1)).backward()
        self.assertGreater(head.hands_fix[0].weight.grad.abs().max().item(),1e-6)

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
