"""Policy guard mathematics, early-stop ownership, precision and memory bounds."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import model, train, train_combined, train_mortal, combined, mortal_model, mortal_learner
from neural.observe import Planes
from neural.ppo_control import PolicyDrift, baseline_batch_size
from neural.tests.test_evaluation_masks import ConflictingViews


class ControlTests(unittest.TestCase):
    def test_kl_is_nonnegative_and_matches_exact_enumeration(self):
        old = torch.tensor([.2, .8])
        new = torch.tensor([.8, .2])
        # Five recorded actions have exactly the old policy's frequencies.
        indices = torch.tensor([0, 1, 1, 1, 1])
        guard = PolicyDrift(.1)
        self.assertTrue(guard.check(old.log()[indices], new.log()[indices]))
        self.assertAlmostEqual(guard.last, float((old * (old.log()-new.log())).sum()), places=6)
        self.assertTrue(guard.stopped)
        guard = PolicyDrift(0)
        self.assertFalse(guard.check(old.log()[indices], new.log()[indices]))
        self.assertGreater(guard.maximum, 0)
        self.assertFalse(PolicyDrift(.02).check(old.log(), old.log()))
        for delta in [-10., -.01, .01, 10.]:
            guard = PolicyDrift()
            guard.check(torch.zeros(2), torch.full((2,), delta))
            self.assertGreaterEqual(guard.last, 0)

    def test_nonfinite_drift_never_enters_optimizer(self):
        # The guard that is on reads every minibatch's drift and refuses a
        # nonfinite one before the step; off, it reads nothing from the
        # card until the round's end (the gradient clip refuses nonfinite
        # values there), so the deferred figure is nonfinite rather than
        # an error.
        for value in [float('nan'), float('inf'), -float('inf'), 1000.]:
            with self.assertRaises(FloatingPointError):
                PolicyDrift(.02).check(torch.zeros(2), torch.full((2,), value))
            off = PolicyDrift()
            self.assertFalse(off.check(torch.zeros(2), torch.full((2,), value)))
            self.assertFalse(off.stopped)
        for threshold in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError): PolicyDrift(threshold)
        with self.assertRaises(ValueError): PolicyDrift().check(torch.zeros(2), torch.zeros(3))

    def test_baseline_size_never_exceeds_rollout_or_budget(self):
        self.assertEqual(baseline_batch_size(32, 2048), 32)
        self.assertEqual(baseline_batch_size(10000, 32), 32)
        self.assertEqual(baseline_batch_size(10000, 2048, 16), 16)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): baseline_batch_size(32, 16, value)

    def test_invalid_controls_rejected_by_every_trainer(self):
        for module in (train, train_mortal, train_combined):
            for option, value in [('--target-kl','nan'),('--target-kl','-1'),('--baseline-batch','0'),
                                  ('--lr','nan'),('--clip','1'),('--entropy','inf')]:
                with self.subTest(trainer=module.__name__,option=option), \
                        patch.object(sys,'argv',['trainer',option,value]):
                    with self.assertRaises(ValueError): module.main()

    def test_both_riichi_stages_inherit_the_callers_precision(self):
        linear = torch.nn.Linear(1, 46)
        legal = np.zeros((1,78),dtype=bool);legal[0,[0,34,35]]=True
        for enabled in (False, True):
            seen=[]
            def score(planes, allowed):
                logits=linear(planes[:,0,:1]); seen.append(logits.dtype)
                logits=logits*0;logits[:,37]=20;logits[:,1]=10
                return logits.masked_fill(~allowed,-torch.inf)
            with torch.no_grad(), torch.autocast('cpu',dtype=torch.bfloat16,enabled=enabled):
                _,records=mortal_learner.decide_in_mortal_space(score,ConflictingViews(),
                    np.array([0]),np.array([0]),legal,greedy=True,device='cpu')
            self.assertEqual(records.actions.tolist(),[37,1])
            self.assertEqual(seen,[torch.bfloat16 if enabled else torch.float32]*2)

    def test_every_trainer_stops_remaining_epochs_and_closes_prefetch(self):
        torch.set_num_threads(1)
        for module in (train, train_mortal, train_combined):
            with self.subTest(trainer=module.__name__),tempfile.TemporaryDirectory() as folder:
                root=Path(folder); origin=root/'source.pt';out=root/'run'
                if module is train:
                    net=model.PolicyValueNet(8,1)
                    payload={'model':net.state_dict(),**net.payload_fields()}
                elif module is train_mortal:
                    net=mortal_model.build(8,1)
                    payload={**net.state(),'config':{'resnet':{'conv_channels':8,'num_blocks':1},'control':{'version':4}}}
                else:
                    net=combined.Combined(model.PolicyValueNet(8,1,actions=46),mortal_model.build(8,1))
                    net.mortal_config={'resnet':{'conv_channels':8,'num_blocks':1},'control':{'version':4}}
                    payload=net.state()
                torch.save(payload,origin)
                def collect(learner, **_kwargs):
                    n=8; actions=78 if module is train else 46
                    planes=Planes(np.arange(n+1,dtype=np.int64),np.zeros(n,dtype=np.uint16),np.ones(n,dtype=np.float16))
                    return SimpleNamespace(decisions=n,observations=planes,legal=torch.ones(n,actions,dtype=torch.bool),
                        actions=torch.zeros(n,dtype=torch.int64),returns=torch.linspace(-1,1,n),log_probs=torch.zeros(n),
                        held=torch.full((n,3,34),1/34),oracle=torch.zeros(n,model.ORACLE_PLANES,34,dtype=torch.uint8),
                        imagined=torch.zeros(n,model.HIDDEN_HANDS_PLANES,34,dtype=torch.uint8),games=1,hands=1,timing={})
                checks=[]
                def check(guard,*_args):
                    checks.append(1);guard.stopped=len(checks)==2;return guard.stopped
                argv=['trainer','--mortal' if module is train_mortal else '--resume',str(origin),
                      '--out',str(out),'--rounds','1','--batch','2','--epochs','3','--games','1',
                      '--measure-games','1','--target-kl','.02','--baseline-batch','2']
                if module is train: argv+=['--replay-steps','0']
                if module is train_combined: argv+=['--fixed','none']
                before={t.ident for t in threading.enumerate() if t.name=='mahjong-prefetch'}
                with patch.object(sys,'argv',argv),patch.object(module.selfplay,'play',side_effect=collect), \
                     patch.object(module.selfplay,'measure',return_value={'placement':2.5,'score':0.,'wins':.25}), \
                     patch.object(PolicyDrift,'check',check),contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                rows=[json.loads(line) for line in (out/'log.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['optimizer_updates'],1)
                self.assertTrue(rows[0]['kl_early_stop']);self.assertEqual(rows[0]['baseline_batch'],2)
                self.assertEqual(len(checks),2) # no subsequent epoch silently continues
                after={t.ident for t in threading.enumerate() if t.name=='mahjong-prefetch'}
                self.assertEqual(before,after)

    def test_cloud_forwards_controls_and_rejects_missing_opponents(self):
        from functools import partial
        from neural import cloud_runs
        from neural.checkpoints import atomic_save
        from neural.tests.test_cloud_isolation import controller
        app=controller()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);volume=root/'volume';scratch=root/'scratch';volume.mkdir()
            for name in ('train','train_mortal','train_combined'):
                atomic_save({'generation':1},volume/name/'latest.pt')
            commands=[]
            def popen(command,**kwargs):
                commands.append(command)
                return SimpleNamespace(stdout=io.StringIO(),wait=lambda:0)
            with patch.object(app,'VOLUME',volume), \
                 patch.object(app,'workspace',partial(cloud_runs.workspace,root=scratch)), \
                 patch.object(app,'_environment',return_value={}),patch.object(app,'_save_cache'), \
                 patch.object(app.subprocess,'Popen',side_effect=popen),contextlib.redirect_stdout(io.StringIO()):
                for name in ('train','train_mortal','train_combined'):
                    getattr(app,name)(run=name,generations=1,target_kl=.02,baseline_batch=16)
                    with self.assertRaises(FileNotFoundError):
                        getattr(app,name)(run=name,generations=1,opponents=['missing'],opponent_share=.5)
            self.assertEqual(len(commands),3)
            for command in commands:
                self.assertEqual(command[command.index('--target-kl')+1],'0.02')
                self.assertEqual(command[command.index('--baseline-batch')+1],'16')

    def test_real_main_training_step_and_checkpoint_without_padding_stubs(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'run'
            run=subprocess.run([sys.executable,'-m','neural.train','--rounds','1','--games','1',
                '--channels','8','--blocks','1','--batch','32','--epochs','2','--replay-steps','1',
                '--measure-games','1','--baseline-batch','16','--target-kl','.000001','--out',str(out)],
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                text=True,capture_output=True,timeout=90)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            record=json.loads((out/'log.jsonl').read_text().splitlines()[0])
            self.assertGreater(record['optimizer_updates'],0)
            self.assertLess(record['optimizer_updates'],record['decisions']//32*2)
            self.assertTrue(record['kl_early_stop'])
            self.assertEqual(record['replay_optimizer_updates'],1)
            checkpoint=torch.load(out/'latest.pt',map_location='cpu',weights_only=True)
            self.assertEqual(checkpoint['generation'],1)
            self.assertEqual(checkpoint['training_controls']['baseline_batch'],16)

    def test_cloud_opponents_with_identical_basenames_keep_distinct_weights(self):
        from functools import partial
        from neural import cloud_runs
        from neural.checkpoints import atomic_save, validate_checkpoint
        from neural.tests.test_cloud_isolation import controller
        app=controller()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);volume=root/'volume';scratch=root/'scratch'
            atomic_save({'generation':11},volume/'family-a/latest.pt')
            atomic_save({'generation':22},volume/'family-b/latest.pt')
            observed=[]
            def popen(command,**kwargs):
                at=command.index('--opponents');end=command.index('--opponent-share')
                paths=[Path(path) for path in command[at+1:end]]
                observed.append(([validate_checkpoint(path) for path in paths],len(set(paths))))
                return SimpleNamespace(stdout=io.StringIO(),wait=lambda:0)
            for name in ('train','train_mortal','train_combined'):
                atomic_save({'generation':1},volume/name/'latest.pt')
            with patch.object(app,'VOLUME',volume), \
                 patch.object(app,'workspace',partial(cloud_runs.workspace,root=scratch)), \
                 patch.object(app,'_environment',return_value={}),patch.object(app,'_save_cache'), \
                 patch.object(app.subprocess,'Popen',side_effect=popen),contextlib.redirect_stdout(io.StringIO()):
                for name in ('train','train_mortal','train_combined'):
                    getattr(app,name)(run=name,generations=1,
                        opponents=['family-a/latest','family-b/latest'],opponent_share=.5)
            self.assertEqual(observed,[([11,22],2)]*3)

    def test_recording_helper_does_not_override_caller_autocast(self):
        views=ConflictingViews()
        legal=np.zeros((1,78),dtype=bool);legal[0,[0,34,35]]=True
        def score(planes,allowed):
            logits=torch.zeros_like(allowed,dtype=torch.float32)
            logits[:,37]=20;logits[:,1]=10
            return logits.masked_fill(~allowed,-torch.inf)
        with patch.object(torch,'autocast',side_effect=AssertionError('helper overrides caller precision')):
            _,records=mortal_learner.decide_in_mortal_space(score,views,
                np.array([0]),np.array([0]),legal,greedy=True,device='cpu')
        self.assertEqual(records.actions.tolist(),[37,1])
