"""Fault injection for immutable snapshots, provenance and publication."""
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from neural import recordings
from neural.observe import Planes
from neural.searched import Recording
from neural.sibling_head import Recorded


def tiny_recording():
    recording = Recording()
    root = Planes.from_follower([0, 1], [0], [1.])
    recording.add(root, [(0, 1., [1., 1.]), (1, 2., [2., 2.])], 0, 1, 0, 0, 1)
    return recording


class RecordingPublicationTests(unittest.TestCase):
    def test_progress_and_completion_are_distinct_even_without_new_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'record'
            record = tiny_recording()
            record.save(folder, {'note': 'partial'}, complete=False)
            partial = recordings.resolve_recording(folder)
            self.assertFalse(recordings.validate_snapshot(partial, require_complete=False)['complete'])
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                Recorded(folder)
            record.save(folder, {'note': 'complete'})
            final = recordings.resolve_recording(folder)
            self.assertNotEqual(partial, final)
            self.assertTrue(partial.is_dir())
            self.assertEqual(recordings.validate_snapshot(final)['rows'], 1)
            self.assertEqual(len(Recorded(folder)), 1)
            copied = Path(temp) / 'accepted'
            recordings.copy_recording(folder, copied, require_complete=True)
            self.assertEqual(recordings.validate_snapshot(recordings.resolve_recording(copied))['snapshot_id'], final.name)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                recordings.copy_recording(partial, copied, require_complete=True)
            self.assertEqual(recordings.resolve_recording(copied).name, final.name)

    def test_every_array_write_failure_preserves_published_bytes(self):
        for array in recordings.ARRAYS:
            with self.subTest(array=array), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp) / 'record'
                record = tiny_recording()
                record.save(folder, {})
                before = (folder / 'current.json').read_bytes()
                old = recordings.resolve_recording(folder)
                real_save = np.save

                def save(path, *args, **kwargs):
                    if Path(path).stem == array:
                        raise OSError('injected array failure')
                    return real_save(path, *args, **kwargs)

                root = Planes.from_follower([0, 1], [1], [2.])
                record.add(root, [(0, 3., [3.]), (1, 4., [4.])], 0, 1, 1, 1, 2)
                with patch.object(np, 'save', side_effect=save), self.assertRaises(OSError):
                    record.save(folder, {})
                self.assertEqual((folder / 'current.json').read_bytes(), before)
                self.assertEqual(recordings.validate_snapshot(old)['rows'], 1)
                # Compaction during serialization does not consume the recording.
                record.save(folder, {})
                self.assertEqual(recordings.validate_snapshot(recordings.resolve_recording(folder))['rows'], 2)
                self.assertTrue(old.is_dir())

    def test_interruption_on_either_side_of_pointer_rename_leaves_complete_snapshot(self):
        for after in (False, True):
            with self.subTest(after=after), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp) / 'record'
                record = tiny_recording()
                record.save(folder, {'note': 'old'})
                old = recordings.resolve_recording(folder)
                replace = os.replace

                def fail(source, target):
                    if Path(target).name == 'current.json':
                        if after:
                            replace(source, target)
                        raise KeyboardInterrupt('at publication boundary')
                    return replace(source, target)

                with patch.object(recordings.os, 'replace', side_effect=fail), self.assertRaises(KeyboardInterrupt):
                    record.save(folder, {'note': 'new'})
                found = recordings.validate_snapshot(recordings.resolve_recording(folder))
                self.assertEqual(found['note'], 'new' if after else 'old')
                self.assertTrue(old.is_dir())

    def test_partial_copy_never_overwrites_accepted_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / 'source', Path(temp) / 'accepted'
            record = tiny_recording()
            record.save(source, {'note': 'old'})
            recordings.copy_recording(source, destination, require_complete=True)
            old = recordings.resolve_recording(destination)
            before = (destination / 'current.json').read_bytes()
            record.save(source, {'note': 'new'})

            def broken_copy(src, dst, **kwargs):
                (Path(dst) / 'meta.json').write_bytes((Path(src) / 'meta.json').read_bytes())
                raise OSError('copy interrupted')

            with patch.object(recordings.shutil, 'copytree', side_effect=broken_copy), self.assertRaises(OSError):
                recordings.copy_recording(source, destination, require_complete=True)
            self.assertEqual((destination / 'current.json').read_bytes(), before)
            self.assertEqual(recordings.validate_snapshot(old)['note'], 'old')

    def test_manifest_and_pointer_are_validated_before_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'record'
            tiny_recording().save(folder, {})
            snapshot = recordings.resolve_recording(folder)
            with (snapshot / 'policy.npy').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                recordings.validate_snapshot(snapshot)
            for pointer in ({'format': 2, 'snapshot': '../elsewhere'}, [], {'format': 1, 'snapshot': snapshot.name}):
                (folder / 'current.json').write_text(json.dumps(pointer))
                with self.assertRaisesRegex(ValueError, 'pointer'):
                    recordings.resolve_recording(folder)

    def test_experiment_identity_covers_every_setting_and_exact_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'latest.pt'
            checkpoint.write_bytes(b'first generation')
            settings = {'candidates': 4, 'margin': 2., 'pool': 4, 'games': 200, 'seed': 12,
                        'temperature': 0., 'valued_by': 'critic', 'leaf_batch': 256}
            original = recordings.experiment(checkpoint, 1, settings)
            repeated = recordings.experiment(checkpoint, 1, settings)
            self.assertEqual(original['configuration_sha256'], repeated['configuration_sha256'])
            self.assertNotEqual(original['experiment_id'], repeated['experiment_id'])
            for key in settings:
                changed = {**settings, key: 'changed'}
                other = recordings.experiment(checkpoint, 1, changed)
                self.assertNotEqual(original['configuration_sha256'], other['configuration_sha256'])
            checkpoint.write_bytes(b'second generation, same alias')
            other = recordings.experiment(checkpoint, 2, settings)
            self.assertNotEqual(original['checkpoint_sha256'], other['checkpoint_sha256'])
            self.assertNotEqual(original['configuration_sha256'], other['configuration_sha256'])


if __name__ == '__main__':
    unittest.main()
