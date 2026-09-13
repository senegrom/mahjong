"""The actual Modal controller refuses failed or incomplete search experiments."""
from contextlib import redirect_stdout
from functools import partial
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from neural import cloud_runs, recordings
from neural.checkpoints import atomic_save
from neural.tests.test_cloud_isolation import controller
from neural.tests.test_recording_publication import tiny_recording


class SearchCloudPublicationTests(unittest.TestCase):
    def run_controller(self, volume, *, exit_code=0, complete=True, record=True, **options):
        app = controller()
        calls = []

        class Process:
            stdout = None
            calls = 0
            stopped = False

            def __init__(self, command, **kwargs):
                calls.append(command)
                self.command = command
                self.folder = (Path(command[command.index('--record') + 1])
                               if '--record' in command else None)
                self.recording = tiny_recording()

            def wait(self, timeout=None):
                self.calls += 1
                if self.folder is not None:
                    if self.calls == 1:
                        self.recording.save(self.folder, {}, complete=False)
                        raise subprocess.TimeoutExpired(self.command, timeout)
                    if complete:
                        self.recording.save(self.folder, {})
                self.stopped = True
                return exit_code

            def poll(self):
                return exit_code if self.stopped else None

            def terminate(self):
                self.stopped = True

        with patch.object(app, 'VOLUME', volume), \
             patch.object(app, 'workspace', partial(cloud_runs.workspace, root=volume.parent / 'scratch')), \
             patch.object(app, '_environment', return_value={}), \
             patch.object(app.subprocess, 'Popen', side_effect=Process), redirect_stdout(io.StringIO()):
            app.searched(run='run', record=record, **options)
        return calls

    def prepare(self, root):
        volume = root / 'volume'
        atomic_save({'generation': 1}, volume / 'run/latest.pt')
        return volume

    def test_failed_child_keeps_progress_but_publishes_no_accepted_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            volume = self.prepare(Path(temp))
            old = volume / 'searched-records/previous/complete'
            tiny_recording().save(old, {})
            before = (old / 'current.json').read_bytes()
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_controller(volume, exit_code=1)
            attempts = [p for p in (volume / 'searched-records').iterdir() if p.name != 'previous']
            self.assertEqual(len(attempts), 1)
            self.assertFalse((attempts[0] / 'complete').exists())
            progress = recordings.resolve_recording(attempts[0] / 'progress')
            self.assertFalse(recordings.validate_snapshot(progress, require_complete=False)['complete'])
            self.assertEqual((old / 'current.json').read_bytes(), before)
            self.assertEqual(json.loads((attempts[0] / 'result.json').read_text())['returncode'], 1)

    def test_zero_exit_with_only_partial_data_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            volume = self.prepare(Path(temp))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                self.run_controller(volume, complete=False)
            attempt = next((volume / 'searched-records').iterdir())
            self.assertFalse((attempt / 'complete').exists())

    def test_complete_snapshots_and_full_configuration_survive_repeated_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            volume = self.prepare(Path(temp))
            options = dict(games=2, seed=17, candidates=3, margin=1., pool=2,
                           temperature=.2, valued_by='mean', leaf_batch=7)
            calls = self.run_controller(volume, **options)
            self.run_controller(volume, **options)
            attempts = list((volume / 'searched-records').iterdir())
            self.assertEqual(len(attempts), 2)
            configurations = []
            for attempt in attempts:
                progress = recordings.resolve_recording(attempt / 'progress')
                final = recordings.resolve_recording(attempt / 'complete')
                self.assertEqual(recordings.validate_snapshot(final)['rows'], 1)
                self.assertNotEqual(progress.name, final.name)
                identity = json.loads((attempt / 'experiment.json').read_text())
                configurations.append(identity['configuration_sha256'])
                for key, value in options.items():
                    self.assertEqual(identity['settings'][key], value)
                    flag = '--' + key.replace('_', '-')
                    self.assertEqual(calls[0][calls[0].index(flag) + 1], str(value))
            self.assertEqual(configurations[0], configurations[1])

    def test_nonrecording_failures_are_not_returned_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            volume = self.prepare(Path(temp))
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_controller(volume, exit_code=1, record=False)
            self.assertFalse((volume / 'searched-records').exists())

    def test_invalid_configuration_fails_before_starting_any_work(self):
        app = controller()
        for options in ({'games': 0}, {'worlds': True}, {'leaf_batch': 0}, {'margin': float('nan')},
                        {'temperature': -1.}, {'chair': 4}, {'seed': -1}, {'valued_by': 'oracle'}):
            with self.subTest(options=options), patch.object(app, 'workspace', side_effect=AssertionError('late')):
                with self.assertRaises(ValueError):
                    app.searched(**options)


if __name__ == '__main__':
    unittest.main()
