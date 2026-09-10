"""Export boundary tests; fake runtime outputs deliberately exercise failures."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from neural import export


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.trial = torch.zeros(4, export.BROWSER_PLANES, export.POSITIONS)
        self.legal = torch.ones(4, export.BROWSER_ACTIONS, dtype=torch.bool)
        self.expected = [torch.arange(export.BROWSER_ACTIONS).float().repeat(4, 1),
                         torch.arange(4).float().reshape(4, 1), torch.ones(4, 3, 34)]
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def check(self, expected=None, given=None, legal=None, names=export.OUTPUTS):
        expected = self.expected if expected is None else expected
        given = [x.numpy().copy() for x in expected] if given is None else given
        session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="planes")],
            get_outputs=lambda: [SimpleNamespace(name=name) for name in names],
            run=lambda *args: given,
        )
        runtime = SimpleNamespace(InferenceSession=Mock(return_value=session))
        with patch.dict(sys.modules, {"onnxruntime": runtime}):
            export.check_runs(Path("unused.onnx"), lambda _: expected,
                              self.trial, self.legal if legal is None else legal)

    def test_current_browser_contract_is_accepted(self):
        export.validate_browser_model(SimpleNamespace(planes=1012, actions=46), {})

    def test_wrong_observations_actions_and_fusion_are_rejected(self):
        for planes, actions in [(97, 78), (97, 46), (1012, 78), (1, 46)]:
            with self.subTest(planes=planes, actions=actions), self.assertRaises(SystemExit):
                export.validate_browser_model(SimpleNamespace(planes=planes, actions=actions), {})
        with self.assertRaisesRegex(SystemExit, "fusion"):
            export.validate_browser_model(SimpleNamespace(planes=1012, actions=46), {"combined": {}})

    def test_equal_outputs_and_small_finite_errors_pass(self):
        self.check()
        given = [x.numpy().copy() for x in self.expected]
        given[0] += 0.001
        self.check(given=given)

    def test_nan_and_infinity_are_rejected_in_every_head_and_side(self):
        for side in ("reference", "exported"):
            for head in range(3):
                for bad in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(side=side, head=head, bad=bad):
                        expected = [x.clone() for x in self.expected]
                        given = [x.numpy().copy() for x in self.expected]
                        if side == "reference":
                            expected[head].view(-1)[0] = bad
                        else:
                            given[head].flat[0] = bad
                        with self.assertRaisesRegex(SystemExit, f"Non-finite {side}"):
                            self.check(expected=expected, given=given)

    def test_wrong_names_counts_and_shapes_fail(self):
        with self.assertRaisesRegex(SystemExit, "outputs in order"):
            self.check(names=("value", "policy", "hands"))
        with self.assertRaisesRegex(SystemExit, "three playable"):
            self.check(given=[self.expected[0].numpy()])
        given = [x.numpy().copy() for x in self.expected]
        given[1] = given[1].ravel()
        with self.assertRaisesRegex(SystemExit, "shape"):
            self.check(given=given)

    def test_large_finite_errors_fail_in_every_head(self):
        for head in range(3):
            with self.subTest(head=head):
                given = [x.numpy().copy() for x in self.expected]
                given[head] += 100
                with self.assertRaisesRegex(SystemExit, "differs too far"):
                    self.check(given=given)

    def test_legal_choice_disagreement_fails_even_with_small_average_error(self):
        expected = [x.clone() for x in self.expected]
        expected[0][:, 0] = 1.0
        expected[0][:, 1] = 1.001
        given = [x.numpy().copy() for x in expected]
        given[0][:, 0] = 1.002
        legal = torch.zeros_like(self.legal)
        legal[:, :2] = True
        with self.assertRaisesRegex(SystemExit, "best legal moves"):
            self.check(expected=expected, given=given, legal=legal)
        legal[:, 1] = False  # A forced decision is not a preference comparison.
        self.check(expected=expected, given=given, legal=legal)

    def test_empty_legal_mask_fails(self):
        with self.assertRaisesRegex(SystemExit, "validation positions"):
            self.check(legal=torch.zeros_like(self.legal))

    def test_contract_rejection_does_not_touch_existing_destination(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "model.onnx"
            destination.write_bytes(b"previous model")
            with patch.object(export, "load_network", return_value=(SimpleNamespace(planes=97, actions=78), {})), \
                 patch.object(export, "validation_positions") as collect, \
                 patch("sys.argv", ["export", "wrong.pt", str(destination)]):
                with self.assertRaisesRegex(SystemExit, "browser requires"):
                    export.main()
                collect.assert_not_called()
            self.assertEqual(destination.read_bytes(), b"previous model")

    def test_failed_validation_preserves_destination_and_success_replaces_it(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "model.onnx"
            net = SimpleNamespace(planes=1012, actions=46, channels=8, blocks=1)
            def write_graph(_wrapped, _example, filename, **_kwargs):
                Path(filename).write_bytes(b"new model")
            for fail in (True, False):
                with self.subTest(fail=fail):
                    destination.write_bytes(b"previous model")
                    with patch.object(export, "load_network", return_value=(net, {})), \
                         patch.object(export, "Playable", return_value=SimpleNamespace(eval=lambda: None)), \
                         patch.object(export, "validation_positions", return_value=(self.trial, self.legal)), \
                         patch("torch.onnx.export", side_effect=write_graph), \
                         patch.object(export, "check_operators"), \
                         patch.object(export, "check_runs", side_effect=SystemExit("bad graph") if fail else None), \
                         patch("sys.argv", ["export", "network.pt", str(destination), "--float32"]):
                        if fail:
                            with self.assertRaisesRegex(SystemExit, "bad graph"):
                                export.main()
                        else:
                            export.main()
                    self.assertEqual(destination.read_bytes(), b"previous model" if fail else b"new model")
                    self.assertEqual(list(Path(root).iterdir()), [destination])

    def test_missing_operator_config_is_not_silently_accepted(self):
        fake_onnx = SimpleNamespace(load=lambda _: SimpleNamespace(graph=SimpleNamespace(node=[])))
        with patch.dict(sys.modules, {"onnx": fake_onnx}), patch.object(export, "runtime_operators", return_value=None):
            with self.assertRaisesRegex(SystemExit, "no runtime operator list"):
                export.check_operators(Path("unused"))
            export.check_operators(Path("unused"), insist=False)


if __name__ == "__main__":
    unittest.main()
