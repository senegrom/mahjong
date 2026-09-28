"""The acting policy's precision contract (`neural.policy_inference`).

The search whose roots and continuations these tests once also compared
with ordinary play is gone; what remains is the precision every table
player chooses its moves at.
"""
import unittest
from unittest.mock import patch

import torch

from neural import policy_inference


class PolicyParityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
