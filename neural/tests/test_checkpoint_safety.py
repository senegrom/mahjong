"""No failed save or publication may destroy the previous recovery checkpoint."""
import ast
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from neural import checkpoints


def saved(generation):
    return {"generation": generation, "model": {"weight": torch.arange(8) + generation}}


class CheckpointSafetyTests(unittest.TestCase):
    def test_interrupted_serialization_and_invalid_bytes_preserve_destination(self):
        for interrupt in (False, True):
            with self.subTest(sigint=interrupt), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "latest.pt"
                checkpoints.atomic_save(saved(1), path)
                before = path.read_bytes()
                def fail(_payload, stream):
                    stream.write(b"incomplete checkpoint")
                    stream.flush()
                    if interrupt:
                        signal.raise_signal(signal.SIGINT)
                with patch.object(torch, 'save', side_effect=fail):
                    with self.assertRaises(BaseException):
                        checkpoints.atomic_save(saved(2), path)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(checkpoints.validate_checkpoint(path), 1)
                self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_sigint_after_real_rename_keeps_complete_new_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "latest.pt"
            checkpoints.atomic_save(saved(1), path)
            replace = os.replace
            def commit_then_interrupt(source, target):
                replace(source, target)
                signal.raise_signal(signal.SIGINT)
            with patch.object(checkpoints.os, 'replace', side_effect=commit_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    checkpoints.atomic_save(saved(2), path)
            self.assertEqual(checkpoints.validate_checkpoint(path), 2)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_failed_fsync_or_replace_keeps_a_loadable_checkpoint(self):
        for operation in ('fsync', 'replace'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "latest.pt"
                checkpoints.atomic_save(saved(1), path)
                with patch.object(checkpoints.os, operation, side_effect=OSError('disk failed')):
                    with self.assertRaises(OSError):
                        checkpoints.atomic_save(saved(2), path)
                self.assertEqual(checkpoints.validate_checkpoint(path), 1)

    def test_abrupt_exit_during_serialization_preserves_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'latest.pt'
            checkpoints.atomic_save(saved(1), path)
            script = '''
import os, sys, torch
from pathlib import Path
from neural import checkpoints

def die(payload, stream):
    stream.write(b'partial')
    stream.flush()
    os._exit(19)
torch.save = die
checkpoints.atomic_save({'generation': 2}, Path(sys.argv[1]))
'''
            result = subprocess.run([sys.executable, '-c', script, str(path)], check=False)
            self.assertEqual(result.returncode, 19)
            self.assertEqual(checkpoints.validate_checkpoint(path), 1)

    def test_corrupt_latest_or_best_never_overwrites_volume_models(self):
        for corrupt in ('latest.pt', 'best.pt'):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); source, target = root/'source', root/'target'
                for name in ('latest.pt', 'best.pt'):
                    checkpoints.atomic_save(saved(1), target/name)
                    checkpoints.atomic_save(saved(2), source/name)
                before = {p.name: p.read_bytes() for p in target.iterdir()}
                (source/corrupt).write_bytes(b'incomplete')
                with self.assertRaises(Exception):
                    checkpoints.publish_training_snapshot(source, target, 2)
                self.assertEqual({p.name: p.read_bytes() for p in target.iterdir()}, before)

    def test_generation_and_history_use_the_actual_copied_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source, target = root/'source', root/'target'
            checkpoints.atomic_save(saved(10), source/'latest.pt')
            checkpoints.atomic_save(saved(8), source/'best.pt')
            actual = checkpoints.publish_training_snapshot(source, target, 9)
            self.assertEqual(actual, 10)
            self.assertEqual((target/'generation.txt').read_text(), '10')
            self.assertEqual(checkpoints.validate_checkpoint(target/'history/gen-00010.pt'), 10)
            self.assertEqual(checkpoints.validate_checkpoint(target/'latest.pt'), 10)
            before=(target/'latest.pt').read_bytes()
            with self.assertRaisesRegex(ValueError, 'older'):
                checkpoints.publish_training_snapshot(source, target, 11)
            self.assertEqual(before,(target/'latest.pt').read_bytes())

    def test_every_training_writer_uses_atomic_checkpoint_publication(self):
        root=Path(__file__).resolve().parents[1]
        for name in ('train', 'train_mortal', 'train_combined', 'imitate', 'distil', 'rehead'):
            tree=ast.parse((root/f'{name}.py').read_text())
            calls=[node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
            self.assertTrue(any(isinstance(fn, ast.Name) and fn.id=='atomic_save' for fn in calls),name)
            self.assertFalse(any(isinstance(fn, ast.Attribute) and fn.attr=='save'
                                 and isinstance(fn.value,ast.Name) and fn.value.id=='torch'
                                 for fn in calls),name)
