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
import subprocess
import sys
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
            atomic_save({'generation':3},volume/'resumed/best.pt')
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
            self.assertIn('--resume',calls[1][0]);self.assertIn('best.pt',calls[1][1])
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
                    # Short of the generation it was to reach, the call fails, so
                    # Modal's retries resume it; a returned call simply ended.
                    with self.assertRaises(subprocess.CalledProcessError) as failed:
                        getattr(app,trainer)(run=trainer,generations=2)
                self.assertEqual(failed.exception.returncode,1)
                self.assertIn('generation 5 of 6',failed.exception.output)
                self.assertEqual(validate_checkpoint(volume/trainer/'latest.pt'),5)
                self.assertEqual((volume/trainer/'generation.txt').read_text(),'5')

    def test_a_trainer_that_dies_short_of_its_target_fails_the_call_and_one_past_it_does_not(self):
        # Killed, as at generation 147 on 1 October, the call returned and the
        # run stood until someone noticed. A failure after the last generation
        # has nothing left to resume, so that call still answers.
        app=controller()
        for trainer in ('train','train_mortal','train_combined'):
            for code,generations,raises in ((-9,3,True),(1,1,False),(-9,1,False)):
                with self.subTest(trainer=trainer,code=code,generations=generations), \
                     tempfile.TemporaryDirectory() as folder:
                    root=Path(folder);volume=root/'volume'
                    atomic_save({'generation':4},volume/'run/latest.pt')
                    def popen(command,**kwargs):
                        where=Path(command[command.index('--out')+1])
                        atomic_save({'generation':5},where/'latest.pt')
                        record=json.dumps({'generation':4,'checkpoint_generation':5})
                        return SimpleNamespace(stdout=io.StringIO(record+'\n'),wait=lambda:code)
                    call=lambda:self.launch(app,trainer,volume,root/'scratch',popen,run='run',
                                            generations=generations)
                    if raises:
                        with self.assertRaises(subprocess.CalledProcessError) as failed:call()
                        self.assertEqual(failed.exception.returncode,code)
                    else:
                        self.assertIn(f'exit={code} generation=5 from 4',call())
                    self.assertEqual((volume/'run/generation.txt').read_text(),'5')

    def stop_before_timeout(self, app, trainer, setup=0.0, generation=10_000.0, generations=20):
        """Runs `trainer` from generation 4 on a clock that `setup` seconds
        pass on before the trainer starts and `generation` seconds a
        generation after; answers what was published, what happened in what
        order, and how many generations the trainer was let finish."""
        clock=SimpleNamespace(now=0.0)
        fake_time=SimpleNamespace(time=lambda:clock.now)
        events,finished=[],[]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);volume=root/'volume'
            atomic_save({'generation':4},volume/'run/latest.pt')
            def reload():
                clock.now+=setup
            def popen(command,**kwargs):
                where=Path(command[command.index('--out')+1])
                def lines():
                    for number in range(4,4+generations):
                        clock.now+=generation
                        atomic_save({'generation':number+1},where/'latest.pt')
                        finished.append(number)
                        yield json.dumps({'generation':number,'checkpoint_generation':number+1})+'\n'
                return SimpleNamespace(stdout=lines(),poll=lambda:None,wait=lambda timeout=None:
                                       events.append('wait') or (-15 if 'terminate' in events else 0),
                                       terminate=lambda:events.append('terminate'))
            with patch.object(app,'VOLUME',volume),patch.object(app,'time',fake_time), \
                 patch.object(app,'volume',SimpleNamespace(reload=reload,commit=lambda:None)), \
                 patch.object(app,'workspace',partial(cloud_runs.workspace,root=root/'scratch')), \
                 patch.object(app,'_environment',return_value={}), \
                 patch.object(app,'_save_cache',side_effect=lambda:events.append('cache')), \
                 patch.object(app.subprocess,'Popen',side_effect=popen),redirect_stdout(io.StringIO()):
                try:
                    answer=getattr(app,trainer)(run='run',generations=generations)
                except app.StoppedBeforeTimeout as stopped:
                    answer=stopped
            published=validate_checkpoint(volume/'run/latest.pt')
            self.assertEqual((volume/'run/generation.txt').read_text(),str(published))
        return answer,published,events,finished

    def test_a_trainer_stops_between_generations_before_the_timeout_and_fails_the_call(self):
        # Ten thousand seconds a generation: after the eighth, at 80,000 s,
        # the ninth would end past 86,400, so the trainer is stopped seconds
        # into it, with generation 12 published, the cache saved after every
        # generation and after the trainer is gone, and the call failed so
        # Modal retries it from there.
        app=controller()
        for trainer in ('train','train_mortal','train_combined'):
            with self.subTest(trainer=trainer):
                answer,published,events,finished=self.stop_before_timeout(app,trainer)
                self.assertIsInstance(answer,app.StoppedBeforeTimeout)
                self.assertEqual(published,12)
                self.assertEqual(finished,list(range(4,12)))
                self.assertEqual(events,['cache']*8+['terminate','wait','cache'])

    def test_the_timeout_counts_from_the_call_not_from_the_trainer(self):
        # Copying the run in took 50,000 s of the same day, so only three
        # generations fit: the third ends at 80,000 s and the fourth would not.
        app=controller()
        answer,published,events,finished=self.stop_before_timeout(app,'train_combined',setup=50_000.0)
        self.assertIsInstance(answer,app.StoppedBeforeTimeout)
        self.assertEqual(published,7)
        self.assertEqual(finished,[4,5,6])

    def test_a_trainer_with_time_to_spare_or_at_its_last_generation_is_left_alone(self):
        app=controller()
        # A hundred seconds a generation: twenty of them take a fraction of the day.
        answer,published,events,finished=self.stop_before_timeout(app,'train_combined',generation=100.0)
        self.assertIn('exit=0 generation=24 from 4',answer)
        self.assertEqual(events,['cache']*20+['wait','cache'])
        # The last generation ends 85,000 s in, where another would not fit,
        # but none is due: the trainer finishes on its own and the call answers.
        answer,published,events,finished=self.stop_before_timeout(
            app,'train_combined',setup=5_000.0,generation=80_000.0,generations=1)
        self.assertIn('exit=0 generation=5 from 4',answer)
        self.assertEqual(published,5)
        self.assertNotIn('terminate',events)

    def launch(self, app, trainer, volume, scratch, popen, **kwargs):
        with patch.object(app,'VOLUME',volume),patch.object(app,'workspace',partial(cloud_runs.workspace,root=scratch)), \
             patch.object(app,'_environment',return_value={}),patch.object(app,'_save_cache'), \
             patch.object(app.subprocess,'Popen',side_effect=popen),redirect_stdout(io.StringIO()):
            return getattr(app,trainer)(**kwargs)

    def test_resuming_from_the_runs_own_history_keeps_its_lineage(self):
        # `history/gen-00030` of the run itself used to count as another
        # lineage: the run's log stayed behind and the first publication
        # replaced it with the block's records alone, a leashed run anchored
        # afresh, and the abandoned timeline's gen-00040 stayed in the
        # history beside the new timeline's latest.
        import torch
        app=controller()
        record=lambda generation,timeline:json.dumps({'generation':generation,
            'checkpoint_generation':generation+1,'timeline':timeline})
        timeline=lambda path:torch.load(path,weights_only=True)['timeline']
        for trainer in ('train','train_mortal','train_combined'):
            with self.subTest(trainer=trainer),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);volume=root/'volume';run=volume/'run'
                atomic_save({'generation':40,'timeline':1},run/'latest.pt')
                atomic_save({'generation':35,'timeline':1},run/'best.pt')
                atomic_save({'generation':0,'timeline':0},run/'reference.pt')
                for generation in (30,40):
                    atomic_save({'generation':generation,'timeline':1},run/f'history/gen-{generation:05d}.pt')
                (run/'log.jsonl').write_text(''.join(record(g,1)+'\n' for g in range(40)))
                seen={}
                def popen(command,**kwargs):
                    where=Path(command[command.index('--out')+1])
                    seen['files']={path.name for path in where.iterdir()}
                    seen['carried']=[json.loads(line)['generation']
                                     for line in (where/'log.jsonl').read_text().splitlines()]
                    # Ten generations of a new timeline from 30, each saved and logged.
                    lines=[]
                    for generation in range(30,40):
                        atomic_save({'generation':generation+1,'timeline':2},where/'latest.pt')
                        lines.append(record(generation,2))
                        with (where/'log.jsonl').open('a') as handle:handle.write(lines[-1]+'\n')
                    return SimpleNamespace(stdout=io.StringIO('\n'.join(lines)+'\n'),wait=lambda:0)
                answer=self.launch(app,trainer,volume,root/'scratch',popen,
                                   run='run',resume='history/gen-00030',generations=10)
                self.assertIn('generation=40 from 30',answer)
                # The log came along, cut where the new timeline branches off,
                # and so did the run's best and, for the leashed trainer, the
                # policy the run is leashed to.
                self.assertEqual(seen['carried'],list(range(30)))
                self.assertIn('best.pt',seen['files'])
                if trainer=='train_combined':self.assertIn('reference.pt',seen['files'])
                log=[json.loads(line) for line in (run/'log.jsonl').read_text().splitlines()]
                self.assertEqual([entry['generation'] for entry in log],list(range(40)))
                self.assertEqual([entry['timeline'] for entry in log],[1]*30+[2]*10)
                self.assertEqual((run/'generation.txt').read_text(),'40')
                self.assertEqual([timeline(run/'latest.pt'),timeline(run/'history/gen-00040.pt'),
                                  timeline(run/'history/gen-00030.pt'),timeline(run/'reference.pt')],[2,2,1,0])

    def test_a_carried_log_stops_at_the_checkpoint_it_resumes_from(self):
        # The volume's log can be a generation ahead of its latest.pt when
        # a publication copied the log after the trainer had logged the next
        # generation; that generation is played again, and logged once.
        app=controller()
        for trainer in ('train','train_mortal','train_combined'):
            with self.subTest(trainer=trainer),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);volume=root/'volume';run=volume/'run'
                atomic_save({'generation':40},run/'latest.pt')
                (run/'log.jsonl').write_text(''.join(
                    json.dumps({'generation':g,'checkpoint_generation':g+1})+'\n' for g in range(41))+'{"generation": 41, "chec')
                carried=[]
                def popen(command,**kwargs):
                    where=Path(command[command.index('--out')+1])
                    carried.extend(json.loads(line)['generation'] for line in (where/'log.jsonl').read_text().splitlines())
                    return SimpleNamespace(stdout=io.StringIO(),wait=lambda:0)
                self.launch(app,trainer,volume,root/'scratch',popen,run='run',generations=1)
                self.assertEqual(carried,list(range(40)))

    def test_a_start_from_another_run_inherits_none_of_this_runs_history(self):
        app=controller()
        for trainer in ('train','train_mortal','train_combined'):
            with self.subTest(trainer=trainer),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);volume=root/'volume';run=volume/'run'
                for name in ('latest.pt','best.pt','reference.pt'):atomic_save({'generation':40},run/name)
                (run/'log.jsonl').write_text(json.dumps({'generation':39})+'\n')
                atomic_save({'generation':12},volume/'other/latest.pt')
                seen={}
                def popen(command,**kwargs):
                    seen['files']={path.name for path in Path(command[command.index('--out')+1]).iterdir()}
                    return SimpleNamespace(stdout=io.StringIO(),wait=lambda:0)
                self.launch(app,trainer,volume,root/'scratch',popen,run='run',resume='other/latest',generations=1)
                self.assertIn('latest.pt',seen['files'])
                self.assertFalse(seen['files']&{'log.jsonl','best.pt','reference.pt'})

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
                for function in (app.rehead,app.distil):
                    with self.subTest(function=function.__name__), \
                         self.assertRaises(subprocess.CalledProcessError) as failed:
                        function(teacher='teacher',run='run',name='product')
                    self.assertIn('failed',failed.exception.output)
            self.assertEqual(before,(volume/'run/product.pt').read_bytes())

    def test_every_function_raises_for_a_missing_input_or_a_failed_child(self):
        # A caller reads each answer as a report, so an error returned as
        # text was read as one: a missing checkpoint raises before anything
        # runs, and a child that fails raises with its output.
        app=controller()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);volume=root/'volume';scratch=root/'scratch'
            atomic_save({'generation':3},volume/'run/teacher.pt')
            calls,outcomes=[],[]
            def execute(command,**kwargs):
                calls.append(command)
                return outcomes.pop(0)
            def popen(command,**kwargs):
                calls.append(command)
                return SimpleNamespace(stdout=io.StringIO(),wait=lambda:0)
            with patch.object(app,'VOLUME',volume),patch.object(app,'workspace',partial(cloud_runs.workspace,root=scratch)), \
                 patch.object(app,'_environment',return_value={}),patch.object(app,'_save_cache'), \
                 patch.object(app.subprocess,'run',side_effect=execute), \
                 patch.object(app.subprocess,'Popen',side_effect=popen),redirect_stdout(io.StringIO()):
                for name,call in (
                        ('train_mortal',lambda:app.train_mortal(run='fresh',mortal='zoo/missing',generations=1)),
                        ('train_combined ours',lambda:app.train_combined(run='fresh',ours='missing',mortal='run/teacher',generations=1)),
                        ('train_combined mortal',lambda:app.train_combined(run='fresh',ours='run/teacher',mortal='missing',generations=1)),
                        ('arena',lambda:app.arena(which='missing',run='run')),
                        ('distil teacher',lambda:app.distil(teacher='missing',run='run')),
                        ('distil student',lambda:app.distil(teacher='teacher',student='missing',run='run')),
                        ('rehead teacher',lambda:app.rehead(teacher='missing',run='run')),
                        ('rehead student',lambda:app.rehead(teacher='teacher',resume='missing',run='run'))):
                    with self.subTest(name=name),self.assertRaises(FileNotFoundError):
                        call()
                self.assertEqual(calls,[])
                outcomes.append(SimpleNamespace(stdout='{"placement": 2.4',stderr='Traceback: boom',returncode=1))
                with self.assertRaises(subprocess.CalledProcessError) as failed:
                    app.arena(which='teacher',run='run')
                self.assertEqual(failed.exception.returncode,1)
                self.assertIn('boom',failed.exception.output)
                outcomes.append(SimpleNamespace(stdout='{"placement": 2.4}',stderr='a warning',returncode=0))
                self.assertEqual(app.arena(which='teacher',run='run'),'generation 3\n{"placement": 2.4}')
            self.assertEqual([command[:3] for command in calls],[[sys.executable,'-m','neural.arena']]*2)
            self.assertEqual(list(scratch.iterdir()),[])

    def test_failed_duels_raise_instead_of_returning_text(self):
        # A caller pooling many duels reads each answer as the table's JSON
        # report, so a missing checkpoint or a failed table must raise: one
        # failed call, not a message that ends the whole pool unparsed.
        app=controller()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);volume=root/'volume';scratch=root/'scratch'
            atomic_save({'generation':5},volume/'run/latest.pt')
            atomic_save({'generation':4},volume/'run/older.pt')
            calls,outcomes=[],[]
            def execute(command,**kwargs):
                calls.append(command)
                self.assertTrue(Path(command[3]).exists() and Path(command[4]).exists())
                return outcomes.pop(0)
            with patch.object(app,'VOLUME',volume),patch.object(app,'workspace',partial(cloud_runs.workspace,root=scratch)), \
                 patch.object(app,'_environment',return_value={}),patch.object(app.subprocess,'run',side_effect=execute), \
                 redirect_stdout(io.StringIO()):
                for challenger,incumbent in (('missing','older'),('latest','missing')):
                    with self.subTest(challenger=challenger,incumbent=incumbent),self.assertRaises(FileNotFoundError):
                        app.duel(challenger=challenger,incumbent=incumbent,run='run')
                self.assertEqual(calls,[])
                outcomes.append(SimpleNamespace(stdout='{"placement": 2.4',stderr='Traceback: boom',returncode=1))
                with self.assertRaises(subprocess.CalledProcessError) as failed:
                    app.duel(challenger='latest',incumbent='older',run='run')
                self.assertEqual(failed.exception.returncode,1)
                self.assertIn('boom',failed.exception.output)
                report={'placement':2.4,'by_deal':[2.5,2.3]}
                outcomes.append(SimpleNamespace(stdout=json.dumps(report),stderr='a warning',returncode=0))
                self.assertEqual(json.loads(app.duel(challenger='latest',incumbent='older',run='run')),report)
            self.assertEqual(len(calls),2)
            self.assertEqual(calls[0][:3],[sys.executable,'-m','neural.duel'])
            self.assertEqual([Path(path).name for path in calls[0][3:5]],['latest.pt','older.pt'])
            # Each call copied the checkpoints into its own scratch and removed it.
            self.assertEqual(list(scratch.iterdir()),[])

    def test_no_function_defaults_to_the_retired_engine_plane_lineage(self):
        # w320-run's networks read the engine's planes as version 1 wrote
        # them and are refused on loading, so a default naming one would
        # fail every call that took it.
        import inspect
        app=controller()
        for name,function in vars(app).items():
            if not inspect.isfunction(function) or function.__module__!=app.__name__:continue
            for parameter in inspect.signature(function).parameters.values():
                if isinstance(parameter.default,str):
                    self.assertNotIn('w320-run',parameter.default,f'{name}({parameter.name})')
        self.assertEqual(inspect.signature(app.duel).parameters['incumbent'].default,'zoo/mortal_298k')
        # A teacher is named by the caller: a network, or '' for the heuristic player.
        self.assertIs(inspect.signature(app.distil).parameters['teacher'].default,inspect.Parameter.empty)

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

    def test_deployment_metadata_does_not_require_local_torch(self):
        # Modal builds Torch into the remote image; its deployment client must
        # still import the application on a machine with only the Modal SDK.
        script = """
import sys
sys.modules['torch'] = None
from neural.tests.test_cloud_isolation import controller
module = controller()
assert module.app is not None
"""
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[2], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
