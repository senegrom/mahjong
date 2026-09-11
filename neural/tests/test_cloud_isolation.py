"""Run the real cloud controllers locally with recording process/volume doubles.

No Modal SDK, external service or GPU job is required. Only top-level deployment
objects are stubbed; scratch allocation and checkpoint IO use their real code.
"""
from contextlib import redirect_stdout
from functools import partial
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from neural import cloud_runs
from neural.checkpoints import atomic_save, validate_checkpoint


class Image:
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class App:
    def __init__(self, *args, **kwargs): pass
    def function(self, **kwargs): return lambda f: f
    def local_entrypoint(self, **kwargs): return lambda f: f


def controller():
    modal=SimpleNamespace(Image=SimpleNamespace(debian_slim=lambda **kw:Image()),App=App,
          Volume=SimpleNamespace(from_name=lambda *a,**kw:SimpleNamespace(reload=lambda:None,commit=lambda:None)))
    path=Path(__file__).resolve().parents[1]/'modal_app.py'
    spec=importlib.util.spec_from_file_location('local_test_modal',path)
    module=importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules',{'modal':modal}):spec.loader.exec_module(module)
    return module


class CloudIsolationTests(unittest.TestCase):
    def test_reused_container_never_supplies_resume_or_other_run_metadata(self):
        app=controller()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);scratch=root/'scratch';volume=root/'volume';volume.mkdir()
            old=scratch/'run';old.mkdir(parents=True)
            for name in ('latest.pt','best.pt','reference.pt'):atomic_save({'generation':71},old/name)
            (old/'replay').mkdir();(old/'log.jsonl').write_text('unrelated history')
            atomic_save({'generation':8},volume/'resumed/latest.pt')
            atomic_save({'generation':3},volume/'resumed/reference.pt')
            calls=[]
            def popen(command,**kwargs):
                where=Path(command[command.index('--out')+1])
                calls.append((command,list(p.name for p in where.iterdir()),
                              json.loads((where/'run.json').read_text())['run']))
                return SimpleNamespace(stdout=io.StringIO(),wait=lambda:0)
            with patch.object(app,'VOLUME',volume),patch.object(app,'workspace',partial(cloud_runs.workspace,root=scratch)), \
                 patch.object(app,'_environment',return_value={}),patch.object(app,'_save_cache'), \
                 patch.object(app.subprocess,'Popen',side_effect=popen),redirect_stdout(io.StringIO()):
                app.train(run='fresh',generations=1)
                app.train(run='resumed',generations=1)
                app.train(run='fresh',generations=1)
            self.assertNotIn('--resume',calls[0][0]);self.assertNotIn('--resume',calls[2][0])
            self.assertIn('--resume',calls[1][0]);self.assertIn('reference.pt',calls[1][1])
            self.assertEqual(calls[0][1],['run.json']);self.assertEqual(calls[2][1],['run.json'])
            self.assertEqual([call[2] for call in calls],['fresh','resumed','fresh'])
            paths=[call[0][call[0].index('--out')+1] for call in calls]
            self.assertEqual(len(set(paths)),3)
            self.assertTrue(all(not Path(path).exists() for path in paths))
            self.assertEqual(validate_checkpoint(old/'latest.pt'),71)

    def test_failed_trainers_keep_only_their_last_completed_publication(self):
        app=controller()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);volume=root/'volume';volume.mkdir();scratch=root/'scratch'
            for trainer in ('train','train_mortal','train_combined'):
                atomic_save({'generation':4},volume/trainer/'latest.pt')
                def popen(command,**kwargs):
                    where=Path(command[command.index('--out')+1])
                    atomic_save({'generation':5},where/'latest.pt')
                    # The controller publishes the completed generation while
                    # reading stdout, then the following round fails.
                    def wait():
                        (where/'latest.pt').write_bytes(b'failed following serialization')
                        return 1
                    return SimpleNamespace(stdout=io.StringIO(json.dumps({'generation':4,'checkpoint_generation':5})+'\n'),wait=wait)
                with self.subTest(trainer=trainer),patch.object(app,'VOLUME',volume), \
                     patch.object(app,'workspace',partial(cloud_runs.workspace,root=scratch)), \
                     patch.object(app,'_environment',return_value={}),patch.object(app,'_save_cache'), \
                     patch.object(app.subprocess,'Popen',side_effect=popen),redirect_stdout(io.StringIO()):
                    result=getattr(app,trainer)(run=trainer,generations=2)
                self.assertIn('exit=1',result)
                self.assertEqual(validate_checkpoint(volume/trainer/'latest.pt'),5)
                self.assertEqual((volume/trainer/'generation.txt').read_text(),'5')

    def test_failed_rehead_and_distillation_cannot_publish_leftover_outputs(self):
        app=controller()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);volume=root/'volume';volume.mkdir();scratch=root/'scratch'
            atomic_save({'generation':3},volume/'run/teacher.pt')
            atomic_save({'generation':2},volume/'run/product.pt')
            before=(volume/'run/product.pt').read_bytes()
            def execute(command,**kwargs):
                out=Path(command[command.index('--out')+1]);atomic_save({'generation':99},out/'latest.pt')
                return SimpleNamespace(stdout='',stderr='failed',returncode=1)
            with patch.object(app,'VOLUME',volume),patch.object(app,'workspace',partial(cloud_runs.workspace,root=scratch)), \
                 patch.object(app,'_environment',return_value={}),patch.object(app.subprocess,'run',side_effect=execute), \
                 redirect_stdout(io.StringIO()):
                app.rehead(teacher='teacher',run='run',name='product')
                app.distil(teacher='teacher',run='run',name='product')
            self.assertEqual(before,(volume/'run/product.pt').read_bytes())

    def test_invalid_run_names_and_paths_fail_before_work_is_started(self):
        app=controller()
        for run in ('../other','/tmp/other','a/b','',r'a\b'):
            with self.subTest(run=run),self.assertRaises(ValueError):
                app._checkpoint(run,'latest')
        for name in ('../other','/tmp/other',r'a\b'):
            with self.subTest(checkpoint=name),self.assertRaises(ValueError):
                app._checkpoint('safe',name)

    def test_controller_failure_terminates_its_owned_child(self):
        events=[]
        process=SimpleNamespace(stdout=io.StringIO(),poll=lambda:None,
                  terminate=lambda:events.append('terminate'),wait=lambda **kw:events.append('wait'))
        with self.assertRaisesRegex(OSError,'publication failed'):
            with cloud_runs.managed_process(process):raise OSError('publication failed')
        self.assertEqual(events,['terminate','wait']);self.assertTrue(process.stdout.closed)
