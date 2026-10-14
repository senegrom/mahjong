"""The acting policy's precision contract (`neural.policy_inference`): the
precision every player at a table chooses its moves at."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import mortal_learner, mortal_model, policy_inference
from neural.tests.test_evaluation_masks import ConflictingViews


class PolicyPrecisionTests(unittest.TestCase):
    def test_precision_resolution_preserves_ordinary_layouts(self):
        self.assertEqual(policy_inference.precision("cpu", 46), "float32")
        self.assertEqual(policy_inference.precision("cuda:0", 46), "bfloat16")
        self.assertEqual(policy_inference.precision("cuda:1", 78), "float32")
        for device, actions in (("mps", 46), ("cpu", 47)):
            with self.assertRaises(ValueError):
                policy_inference.precision(device, actions)
        with patch.object(torch, "autocast") as cast:
            policy_inference.autocast("cuda:0", 46)
            cast.assert_called_once_with("cuda", dtype=torch.bfloat16, enabled=True)

    def test_cpu_policy_explicitly_disables_ambient_autocast(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with policy_inference.autocast("cpu", 46):
                self.assertFalse(torch.is_autocast_enabled("cpu"))
            self.assertTrue(torch.is_autocast_enabled("cpu"))

    def test_a_mortal_learner_chooses_at_the_tables_precision(self):
        """`MortalLearner.choose`, by which train_mortal measures its learner
        against the heuristic players and so picks its best checkpoint,
        decided at whatever precision its caller held, float32, while the
        learner's self-play and every table its checkpoint sits at play it
        in bfloat16 on the card. It keeps to the contract now: bfloat16
        where the contract says so, here on a processor taken for the card,
        and float32 on the processor whatever its caller holds."""
        torch.manual_seed(3)
        learner = mortal_learner.MortalLearner(mortal_model.build(8, 1)).eval()
        answered = []

        def inference(planes, mask):
            logits, value = learner.policy(planes, mask)
            answered.append(logits.dtype)
            return logits, value

        learner.inference = inference
        legal = np.zeros((1, 78), dtype=bool)
        legal[0, [0, 1]] = True
        choose = lambda: learner.choose(ConflictingViews(), np.array([0]), np.array([0]), legal)
        with patch.object(policy_inference, "precision", return_value="bfloat16"):
            choose()
        self.assertEqual(answered, [torch.bfloat16])
        answered.clear()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            choose()
        self.assertEqual(answered, [torch.float32])


if __name__ == "__main__":
    unittest.main()
