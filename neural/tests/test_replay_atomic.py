"""Failure injection must recover an entire old or new replay generation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import replay
from neural.observe import Planes


def batch(value, n=3):
    return SimpleNamespace(
        decisions=n, legal=torch.ones(n, 46, dtype=torch.bool),
        held=torch.full((n, 3, 34), float(value)),
        oracle=torch.full((n, 3, 34), value, dtype=torch.uint8),
        imagined=torch.full((n, 3, 34), value, dtype=torch.uint8),
        returns=torch.full((n,), float(value)),
        observations=Planes(np.arange(n+1, dtype=np.int64), np.zeros(n, dtype=np.uint16),
                            np.full(n, value, dtype=np.float16)))


class ReplayTests(unittest.TestCase):
    def assert_batch(self, ring, value):
        sampled = ring.sample(3, np.random.default_rng(0))
        self.assertTrue((sampled['observations'].values == value).all())
        for field in ('held', 'oracle', 'imagined', 'returns'):
            self.assertTrue((sampled[field].numpy() == value).all(), field)

    def test_every_array_write_failure_preserves_old_data_and_in_memory_state(self):
        save = np.save
        for failure_at in range(1, 9):
            with self.subTest(write=failure_at), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                ring = replay.Ring(root, 1); ring.push(batch(1))
                manifest = ring.index.read_bytes()
                calls = 0
                def fail(path, values, *args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == failure_at:
                        raise OSError('interrupted array write')
                    return save(path, values, *args, **kwargs)
                with patch.object(replay.np, 'save', side_effect=fail):
                    with self.assertRaisesRegex(OSError, 'interrupted'):
                        ring.push(batch(2))
                self.assertEqual(manifest, ring.index.read_bytes())
                self.assertEqual(ring.next_slot, 1)
                self.assert_batch(ring, 1)
                self.assert_batch(replay.Ring(root, 1), 1)
                ring.push(batch(2))
                self.assert_batch(replay.Ring(root, 1), 2)

    def test_failed_batch_or_manifest_publication_keeps_the_old_generation(self):
        replace = os.replace
        for target in ('batch', 'manifest'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); ring = replay.Ring(root, 1); ring.push(batch(1))
                def fail(src, dst):
                    if (target == 'manifest' and Path(dst).name == 'ring.json'
                            or target == 'batch' and Path(dst).name.startswith('batch-')):
                        raise OSError('interrupted publication')
                    return replace(src, dst)
                with patch.object(replay.os, 'replace', side_effect=fail):
                    with self.assertRaisesRegex(OSError, 'interrupted'):
                        ring.push(batch(2))
                self.assert_batch(ring, 1)
                self.assert_batch(replay.Ring(root, 1), 1)

    def test_failure_after_publication_exposes_only_the_complete_new_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); ring = replay.Ring(root, 1); ring.push(batch(1))
            sync = ring._sync_directory
            calls = 0
            def fail(path):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError('post-publication sync failed')
                sync(path)
            with patch.object(ring, '_sync_directory', side_effect=fail):
                with self.assertRaisesRegex(OSError, 'post-publication'):
                    ring.push(batch(2))
            self.assertEqual(ring.next_slot, 2)
            self.assert_batch(ring, 2)
            self.assert_batch(replay.Ring(root, 1), 2)

    def test_hard_process_exit_on_either_side_of_manifest_commit(self):
        # os._exit bypasses finally blocks, unlike an ordinary raised exception.
        code = '''
import os, sys
from pathlib import Path
from neural import replay
from neural.tests.test_replay_atomic import batch
root = Path(sys.argv[1]); phase = sys.argv[2]
ring = replay.Ring(root, 1)
original = os.replace
def interrupted(src, dst):
    if Path(dst).name == 'ring.json' and phase == 'before': os._exit(77)
    original(src, dst)
    if Path(dst).name == 'ring.json' and phase == 'after': os._exit(77)
replay.os.replace = interrupted
ring.push(batch(2))
'''
        for phase, expected in [('before', 1), ('after', 2)]:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as folder:
                ring = replay.Ring(Path(folder), 1); ring.push(batch(1))
                result = subprocess.run([sys.executable, '-c', code, folder, phase],
                                        cwd=Path(__file__).resolve().parents[2], capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 77, result.stderr.decode())
                self.assert_batch(replay.Ring(Path(folder), 1), expected)
                self.assertFalse(list(Path(folder).glob('.staging-*')))

    def test_wraparound_and_legacy_migration_preserve_complete_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = batch(1)
            for field in replay.FIELDS:
                np.save(root / f'{field}-0.npy', getattr(old, field).numpy())
            old.observations.save(root, 'observations-0')
            (root / 'ring.json').write_text(json.dumps({'entries': [{'slot': 0, 'n': 3}], 'next_slot': 1}))
            ring = replay.Ring(root, 2); self.assert_batch(ring, 1)
            for value in range(2, 10):
                ring.push(batch(value))
                ring = replay.Ring(root, 2)
                self.assertEqual(len(ring), 2)
                self.assertEqual(ring.total(), 6)
                for seed in range(10):
                    sample = ring.sample(3, np.random.default_rng(seed))
                    current = float(sample['returns'][0])
                    self.assertIn(current, (value-1, value))
                    self.assertTrue((sample['observations'].values == current).all())
            self.assertFalse((root / 'returns-0.npy').exists())
            self.assertEqual(len(list(root.glob('batch-*'))), 2)

    def test_bad_rows_are_rejected_before_publication_and_missing_data_is_not_sampled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); ring = replay.Ring(root, 1); ring.push(batch(1))
            invalid = batch(2); invalid.returns = torch.zeros(2)
            with self.assertRaisesRegex(ValueError, 'one row'):
                ring.push(invalid)
            self.assert_batch(ring, 1)
            generation = ring.entries[0]['generation']
            # Release local maps before deleting files, including on Windows.
            ring.maps.clear()
            (root / generation / 'observations-values.npy').unlink()
            with self.assertWarnsRegex(RuntimeWarning, 'incomplete replay'):
                restored = replay.Ring(root, 1)
            self.assertEqual(len(restored), 0)
            with self.assertRaises(ValueError):
                restored.sample(1, np.random.default_rng(0))

    def test_invalid_capacity_and_unsafe_manifest_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for rounds in (0, -1, True):
                with self.assertRaises(ValueError):
                    replay.Ring(Path(folder), rounds)
            (Path(folder) / 'ring.json').write_text(json.dumps({
                'version': 2, 'next_slot': 1,
                'entries': [{'slot': 0, 'n': 3, 'generation': '../unrelated'}]}))
            with self.assertRaisesRegex(ValueError, 'generation'):
                replay.Ring(Path(folder), 1)
