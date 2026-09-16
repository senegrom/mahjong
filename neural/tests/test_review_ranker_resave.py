"""A validated sibling head can be re-saved without metadata relabelling."""
from pathlib import Path
import tempfile
import unittest

import torch
from neural import model, sibling_head


class RankerResaveTests(unittest.TestCase):
    def test_loaded_head_and_metadata_resave_without_losing_identity(self):
        actor = model.PolicyValueNet(8, 1, actions=46).eval()
        head = sibling_head.new_head(actor)
        with tempfile.TemporaryDirectory() as root:
            source, destination = Path(root) / 'first.pt', Path(root) / 'second.pt'
            meta = {'checkpoint': 'actor.pt', 'sure': 0.8, 'history': [{'epoch': 1}]}
            sibling_head.save(head, source, meta)
            loaded, returned = sibling_head.load(source)
            self.assertEqual(returned, meta)
            self.assertNotIn('feature_contract', returned)
            sibling_head.save(loaded, destination, returned)
            final, final_meta = sibling_head.load(destination)
            self.assertEqual(final_meta, meta)
            self.assertEqual(final.feature_contract, head.feature_contract)
            sibling_head.require_head_for(final, actor)
            for key, value in head.state_dict().items():
                torch.testing.assert_close(final.state_dict()[key], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
