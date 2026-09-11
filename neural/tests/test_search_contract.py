"""Native hypothetical snapshots must never be relabelled as Mortal history."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
import riichi_py

from neural import searched
from neural.model import PolicyValueNet


class SearchContractTests(unittest.TestCase):
    def test_current_network_only_baselines_complete_for_both_action_layouts(self):
        previous=torch.get_num_threads();torch.set_num_threads(1)
        try:
            for actions in (78,46):
                with self.subTest(actions=actions):
                    torch.manual_seed(12)
                    net=PolicyValueNet(8,1,actions=actions)
                    scores,tally=searched.play(net,1,1,None,1,1,0.,device='cpu')
                    self.assertEqual(scores.shape,(1,4));self.assertTrue(np.isfinite(scores).all())
                    self.assertEqual(tally,(0,0))
        finally:torch.set_num_threads(previous)

    def test_unsupported_search_rejected_before_touching_any_arena(self):
        for actions in (78,46):
            net=PolicyValueNet(8,1,actions=actions)
            for call in (
                lambda:searched.play(net,1,1,0,1,1,0.,device='cpu'),
                lambda:searched.play_lookahead(net,None,device='cpu'),
                lambda:searched.search_with_value_head(net,None,[],[],worlds=1,candidates=1,
                                                      margin=0.,hurried=True,device='cpu')):
                with self.subTest(actions=actions),patch.object(riichi_py,'Arena',side_effect=AssertionError('too late')):
                    with self.assertRaisesRegex(searched.UnsupportedSearchLayout,'Mortal event history'):call()

    def test_legacy_native_search_contract_is_still_supported(self):
        net=PolicyValueNet(8,1,planes=riichi_py.PLANES,actions=riichi_py.ACTIONS)
        searched.require_native_search(net)
