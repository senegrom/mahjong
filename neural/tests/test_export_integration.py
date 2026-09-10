"""Real native encoders and ONNX Runtime; CI must install these dependencies."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import onnx
import onnxruntime
import torch
from libriichi.consts import ACTION_SPACE

from neural import export
from neural.model import MORTAL_PLANES, PolicyValueNet


class ExportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.positions, cls.legal = export.validation_positions()

    def test_native_validation_positions_match_browser_contract(self):
        self.assertEqual(export.BROWSER_PLANES, MORTAL_PLANES)
        self.assertEqual(export.BROWSER_ACTIONS, ACTION_SPACE)
        self.assertGreater(len(self.positions), 4)
        self.assertEqual(tuple(self.positions.shape[1:]), (1012, 34))
        self.assertTrue(torch.isfinite(self.positions).all())
        self.assertTrue(self.positions.ne(0).any())
        self.assertTrue(self.legal.any(dim=1).all())
        self.assertTrue((self.legal.sum(dim=1) > 1).any())

    def test_real_float_and_quantised_graphs_run_on_real_positions(self):
        torch.manual_seed(3)
        net = PolicyValueNet(8, 1, MORTAL_PLANES, attention=False, actions=ACTION_SPACE).eval()
        # Separate policy preferences clearly so quantisation of a random,
        # untrained fixture is not judged on arbitrary near-ties.
        with torch.no_grad():
            net.policy_tiles.weight.zero_()
            net.policy_tiles.bias.zero_()
            net.policy_pooled[-1].weight.zero_()
            net.policy_pooled[-1].bias.copy_(torch.arange(ACTION_SPACE - 34).float() + 1)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = root / "network.pt"
            torch.save({"model": net.state_dict(), **net.payload_fields()}, checkpoint)
            for floating in (True, False):
                with self.subTest(float32=floating):
                    destination = root / f"{floating}.onnx"
                    argv = ["export", str(checkpoint), str(destination)]
                    if floating:
                        argv += ["--float32", "--allow-any-operator"]
                    with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                         patch.object(export, "validation_positions", return_value=(self.positions, self.legal)):
                        export.main()
                    graph = onnx.load(str(destination))
                    onnx.checker.check_model(graph)
                    session = onnxruntime.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
                    result = session.run(None, {"planes": self.positions[:1].numpy()})
                    self.assertEqual([tuple(x.shape) for x in result], [(1, 46), (1, 1), (1, 3, 34)])
                    self.assertTrue(all(torch.isfinite(torch.from_numpy(x)).all() for x in result))


if __name__ == "__main__":
    unittest.main()
