"""The acting policy's precision contract (`neural.policy_inference`).

The search whose roots and continuations these tests once also compared
with ordinary play is gone; what remains is the precision every table
player chooses its moves at.
"""
import unittest
from unittest.mock import patch

import torch

from neural import policy_inference


class TiedNet(torch.nn.Module):
    kind, planes, actions, speaks_mortal = "mortal", 1012, 46, True
    channels, blocks = 8, 1

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.sizes = []

    def everything(self, planes, legal):
        self.sizes.append(len(planes))
        logits = torch.zeros_like(legal, dtype=torch.float32) + self.anchor
        return logits.masked_fill(~legal, -torch.inf), torch.zeros(len(planes), device=planes.device), torch.zeros(len(planes), 3, 34, device=planes.device)

    def forward(self, planes, legal):
        return self.everything(planes, legal)[:2]


class PolicyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

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
        class Probe(TiedNet):
            def everything(self, planes, legal):
                self.autocast_enabled = torch.is_autocast_enabled("cpu")
                return super().everything(planes, legal)
        net = Probe()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            policy_inference.everything(net, torch.zeros(1, 1012, 34), torch.ones(1, 46, dtype=torch.bool))
            self.assertFalse(net.autocast_enabled)
            self.assertTrue(torch.is_autocast_enabled("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA hardware")
    def test_cuda_full_and_fast_policy_use_same_bfloat16_contract(self):
        from neural import combined, model, mortal_model
        torch.manual_seed(8)
        nets = [model.PolicyValueNet(8, 1, actions=46)]
        nets.append(combined.Combined(model.PolicyValueNet(8, 1, actions=46), mortal_model.build(16, 1)))
        for net in nets:
            net = net.cuda().eval()
            planes = torch.randn(3, 1012, 34, device="cuda")
            legal = torch.zeros(3, 46, dtype=torch.bool, device="cuda")
            legal[:, :8] = True
            with torch.no_grad():
                full = policy_inference.everything(net, planes, legal)[0]
                fast = policy_inference.policy_logits(net, planes, legal)
                torch.testing.assert_close(full, fast, rtol=0, atol=0)
                with policy_inference.autocast("cuda", 46):
                    self.assertTrue(torch.is_autocast_enabled("cuda"))
                    self.assertEqual(torch.get_autocast_dtype("cuda"), torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
