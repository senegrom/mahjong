"""Validate the actual cloud helpers and all three controllers, without Modal."""
from contextlib import ExitStack, redirect_stdout
from functools import partial
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from neural.cloud_requests import stage_opponents, validate_cloud_request
from neural.checkpoints import atomic_save, publish_training_snapshot


class CloudRequestHelpersTests(unittest.TestCase):
    def test_additional_count_rejects_zero_negative_boolean_and_noninteger(self):
        for count in (0, -1, False, True, 1.5, '2', None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_cloud_request(count, [], 0)
        validate_cloud_request(1, [], 0)
        validate_cloud_request(40, ['run/best'], 0.5)

    def test_population_and_share_validation(self):
        for names, share in ((None, .5), ([], .5), (['x'], -1), (['x'], 2),
                             (['x'], float('nan')), (['x'], float('inf')),
                             (['x'], True), ('run/best', .5), ([''], .5), ([None], 0)):
            with self.subTest(names=names, share=share), self.assertRaises(ValueError):
                validate_cloud_request(1, names, share)

    def test_same_basename_keeps_both_exact_checkpoints_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'volume'
            where = root/'scratch'; where.mkdir()
            for name, marker in (('run-a/best', 11), ('run-b/best', 22)):
                atomic_save({'generation': marker, 'model': {'w': torch.tensor([marker])}},
                            source/(name+'.pt'))
            args = stage_opponents(where, ['run-a/best', 'run-b/best'], .5,
                                   lambda name: source/(name+'.pt'))
            paths = args[args.index('--opponents')+1:args.index('--opponent-share')]
            self.assertEqual(len(set(paths)), 2)
            self.assertEqual([torch.load(p, weights_only=True)['generation'] for p in paths], [11, 22])
            saved = json.loads((where/'opponents.json').read_text())
            self.assertEqual(saved['opponent_share'], .5)
            self.assertEqual([e['requested'] for e in saved['opponents']], ['run-a/best', 'run-b/best'])
            self.assertEqual(len({e['sha256'] for e in saved['opponents']}), 2)
            atomic_save({'generation': 1}, where/'latest.pt')
            publish_training_snapshot(where, root/'published', 1)
            self.assertEqual((root/'published/opponents.json').read_bytes(),
                             (where/'opponents.json').read_bytes())

    def test_missing_population_rejected_before_any_staging(self):
        for names in (['absent'], ['present', 'absent']):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); where=root/'scratch'; where.mkdir()
                atomic_save({'generation': 1}, root/'present.pt')
                with self.assertRaises(FileNotFoundError):
                    stage_opponents(where, names, .5, lambda name: root/(name+'.pt'))
                self.assertEqual(list(where.iterdir()), [])

    def test_corrupt_opponent_cannot_produce_population_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); where=root/'scratch';where.mkdir()
            (root/'bad.pt').write_bytes(b'not a checkpoint')
            with self.assertRaises(Exception):
                stage_opponents(where, ['bad'], .5, lambda _: root/'bad.pt')
            self.assertFalse((where/'opponents.json').exists())

    def test_empty_population_does_not_add_scratch_or_disable_requested_share(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertEqual(stage_opponents(root, [], 0, lambda _: None), ['--opponent-share','0.0'])
            self.assertEqual(list(root.iterdir()), [])
            with self.assertRaises(ValueError):
                stage_opponents(root, [], .5, lambda _: None)


class CloudControllerRequestsTests(unittest.TestCase):
    def test_every_controller_rejects_bad_requests_before_workspace_or_popen(self):
        from neural.tests.test_cloud_isolation import controller
        app = controller()
        for trainer in ('train','train_mortal','train_combined'):
            for kwargs in ({'generations':0},{'generations':-1},
                           {'generations':1,'opponent_share':.5,'opponents':[]}):
                with self.subTest(trainer=trainer,kwargs=kwargs), \
                     patch.object(app,'workspace') as work, \
                     patch.object(app.subprocess,'Popen') as popen:
                    with self.assertRaises(ValueError):
                        getattr(app,trainer)(**kwargs)
                    work.assert_not_called();popen.assert_not_called()

    def test_every_controller_preserves_counts_and_all_staged_opponents(self):
        from neural.tests.test_cloud_isolation import controller
        from neural import cloud_runs
        app=controller()
        for trainer in ('train','train_mortal','train_combined'):
            with self.subTest(trainer=trainer),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);volume=root/'volume';volume.mkdir()
                atomic_save({'generation':17},volume/'run/latest.pt')
                for name, generation in [('a/best',11),('b/best',22)]:
                    atomic_save({'generation':generation},volume/(name+'.pt'))
                calls=[]
                def popen(command, **kwargs):
                    calls.append(command)
                    self.assertEqual(command[command.index('--rounds')+1],'3')
                    start=command.index('--opponents')+1
                    paths=command[start:]
                    # All later flags are deliberately excluded from the paths.
                    stop=next((i for i,p in enumerate(paths) if p.startswith('--')),len(paths))
                    paths=paths[:stop]
                    self.assertEqual([torch.load(p,weights_only=True)['generation'] for p in paths],[11,22])
                    return SimpleNamespace(stdout=io.StringIO(),wait=lambda:0)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(app,'VOLUME',volume))
                    stack.enter_context(patch.object(app,'workspace',partial(cloud_runs.workspace,root=root/'scratch')))
                    stack.enter_context(patch.object(app,'_environment',return_value={}))
                    stack.enter_context(patch.object(app,'_save_cache'))
                    stack.enter_context(patch.object(app.subprocess,'Popen',side_effect=popen))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    getattr(app,trainer)(run='run',generations=3,opponents=['a/best','b/best'],opponent_share=.5)
                    self.assertEqual(len(calls),1)
                    calls.clear()
                    for names in (['missing'],['a/best','missing']):
                        with self.assertRaises(FileNotFoundError):
                            getattr(app,trainer)(run='run',generations=3,opponents=names,opponent_share=.5)
                        self.assertEqual(calls,[])
