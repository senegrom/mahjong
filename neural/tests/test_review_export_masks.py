"""Masked fusion logits remain safe at the actual export-validation boundary."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from neural import export


class MaskedPolicyExportTests(unittest.TestCase):
    def setUp(self):
        # Five rows exercise both a browser-sized batch and the final tail.
        self.trial = torch.zeros(5, export.BROWSER_PLANES, export.POSITIONS)
        self.trial[:, 0, 0] = torch.arange(5)
        self.legal = torch.ones(5, export.BROWSER_ACTIONS, dtype=torch.bool)
        self.legal[:, 0] = False
        self.expected = [torch.arange(export.BROWSER_ACTIONS).float().repeat(5, 1),
                         torch.arange(5).float().reshape(5, 1), torch.ones(5, 3, 34)]

    def check(self, expected, given, *, fused=True):
        feeds = []
        names = ['planes', 'legal'] if fused else ['planes']

        def reference(planes, legal=None):
            start = int(planes[0, 0, 0])
            if fused:
                torch.testing.assert_close(legal, self.legal[start:start + len(planes)].float())
            return [value[start:start + len(planes)] for value in expected]

        def run(outputs, feed):
            self.assertEqual(outputs, list(export.OUTPUTS))
            self.assertEqual(set(feed), set(names))
            feeds.append(feed)
            start, size = int(feed['planes'][0, 0, 0]), len(feed['planes'])
            if fused:
                self.assertEqual(feed['legal'].dtype, np.float32)
                np.testing.assert_array_equal(feed['legal'], self.legal[start:start + size].numpy())
            return [value[start:start + size] for value in given]

        session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name=name) for name in names],
            get_outputs=lambda: [SimpleNamespace(name=name) for name in export.OUTPUTS],
            run=run,
        )
        runtime = SimpleNamespace(InferenceSession=Mock(return_value=session))
        with patch.dict('sys.modules', {'onnxruntime': runtime}), contextlib.redirect_stdout(io.StringIO()):
            export.check_runs(Path('unused.onnx'), reference, self.trial, self.legal)
        self.assertEqual([len(feed['planes']) for feed in feeds], [4, 1])

    def test_matching_negative_infinity_is_allowed_only_on_masked_moves(self):
        for fused in (False, True):
            with self.subTest(fused=fused):
                expected = [value.clone() for value in self.expected]
                expected[0][:, 0] = -torch.inf
                self.check(expected, [value.numpy().copy() for value in expected], fused=fused)

    def test_matching_nan_or_positive_infinity_on_illegal_moves_is_rejected(self):
        for bad in (torch.nan, torch.inf):
            with self.subTest(bad=bad):
                expected = [value.clone() for value in self.expected]
                expected[0][:, 0] = bad
                with self.assertRaisesRegex(SystemExit, 'Non-finite reference policy'):
                    self.check(expected, [value.numpy().copy() for value in expected])

    def test_illegal_nan_or_positive_infinity_reports_the_invalid_side(self):
        for side in ('reference', 'exported'):
            for bad in (torch.nan, torch.inf):
                with self.subTest(side=side, bad=bad):
                    expected = [value.clone() for value in self.expected]
                    expected[0][:, 0] = -torch.inf
                    given = [value.numpy().copy() for value in expected]
                    if side == 'reference':
                        expected[0][:, 0] = bad
                    else:
                        given[0][:, 0] = bad
                    with self.assertRaisesRegex(SystemExit, f'Non-finite {side} policy'):
                        self.check(expected, given)

    def test_every_nonfinite_legal_policy_value_reports_the_invalid_side(self):
        for side in ('reference', 'exported'):
            for bad in (torch.nan, torch.inf, -torch.inf):
                with self.subTest(side=side, bad=bad):
                    expected = [value.clone() for value in self.expected]
                    given = [value.numpy().copy() for value in expected]
                    # Last-row corruption must also be checked in the tail batch.
                    if side == 'reference':
                        expected[0][-1, 1] = bad
                    else:
                        given[0][-1, 1] = bad
                    with self.assertRaisesRegex(SystemExit, f'Non-finite {side} policy'):
                        self.check(expected, given)

    def test_one_sided_masked_negative_infinity_still_rejects_different_support(self):
        for side in ('reference', 'exported'):
            with self.subTest(side=side):
                expected = [value.clone() for value in self.expected]
                given = [value.numpy().copy() for value in expected]
                if side == 'reference':
                    expected[0][:, 0] = -torch.inf
                else:
                    given[0][:, 0] = -torch.inf
                with self.assertRaisesRegex(SystemExit, 'different set of moves'):
                    self.check(expected, given)


if __name__ == '__main__':
    unittest.main()
