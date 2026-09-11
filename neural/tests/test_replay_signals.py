"""SIGINT must not erase a manifest's generation or leave a stale live writer."""
import inspect
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from neural import replay
from neural.observe import Planes


def batch(value):
    return SimpleNamespace(
        decisions=3, legal=torch.ones(3, 46, dtype=torch.bool),
        held=torch.full((3, 3, 34), float(value)),
        oracle=torch.full((3, 3, 34), value, dtype=torch.uint8),
        imagined=torch.full((3, 3, 34), value, dtype=torch.uint8),
        returns=torch.full((3,), float(value)),
        observations=Planes(np.arange(4, dtype=np.int64), np.zeros(3, dtype=np.uint16),
                            np.full(3, value, dtype=np.float16)),
    )


class ReplaySignalTests(unittest.TestCase):
    def assert_batch(self, ring, value):
        sample = ring.sample(3, np.random.default_rng(0))
        np.testing.assert_array_equal(sample['observations'].values, value)
        for field in ('held', 'oracle', 'imagined', 'returns'):
            np.testing.assert_array_equal(sample[field].numpy(), value)

    def test_sigint_before_and_after_publication_recovers_disk_and_live_writer(self):
        source, start = inspect.getsourcelines(replay.Ring.push)
        # No filesystem methods are patched. The trace delivers a real signal
        # just before each statement, including the first after atomic rename.
        for statement, expected in [
            ('os.replace(manifest, self.index)', 1),
            ('self.entries = entries', 2),
            ('self.next_slot += 1', 2),
            ('self.maps = {old["slot"]:', 2),
            ('self._dirty = False', 2),
        ]:
            line = start + next(i for i, text in enumerate(source) if text.strip().startswith(statement))
            with self.subTest(statement=statement), tempfile.TemporaryDirectory() as folder:
                ring = replay.Ring(Path(folder), 1)
                ring.push(batch(1))
                delivered = False
                old_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
                old_trace = sys.gettrace()

                def interrupt(frame, event, arg):
                    nonlocal delivered
                    if frame.f_code is replay.Ring.push.__code__ and event == 'line' and frame.f_lineno == line:
                        delivered = True
                        sys.settrace(None)
                        signal.raise_signal(signal.SIGINT)
                    return interrupt

                try:
                    sys.settrace(interrupt)
                    with self.assertRaises(KeyboardInterrupt):
                        ring.push(batch(2))
                finally:
                    sys.settrace(old_trace)
                    signal.signal(signal.SIGINT, old_handler)
                self.assertTrue(delivered)
                manifest = json.loads(ring.index.read_text())
                for entry in manifest['entries']:
                    self.assertTrue((Path(folder) / entry['generation']).is_dir())
                self.assert_batch(ring, expected)
                self.assertEqual(ring.next_slot, expected)
                self.assertEqual(len(ring), 1)
                self.assertEqual(ring.total(), 3)
                self.assert_batch(replay.Ring(Path(folder), 1), expected)
                # A caller catching Ctrl+C can keep using the same Ring safely.
                ring.push(batch(3))
                self.assert_batch(replay.Ring(Path(folder), 1), 3)

    def test_interrupting_initial_publication_does_not_erase_first_batch(self):
        from unittest.mock import patch
        replace = replay.os.replace
        with tempfile.TemporaryDirectory() as folder:
            ring = replay.Ring(Path(folder), 1)

            def commit_then_interrupt(src, dst):
                replace(src, dst)
                if Path(dst) == ring.index:
                    raise KeyboardInterrupt()

            with patch.object(replay.os, 'replace', side_effect=commit_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    ring.push(batch(1))
            self.assert_batch(ring, 1)
            self.assert_batch(replay.Ring(Path(folder), 1), 1)
