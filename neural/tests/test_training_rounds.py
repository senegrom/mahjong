"""Exercise actual trainer entry points around restart and zero-update rounds."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import combined, model, mortal_model, train, train_combined, train_mortal
from neural.observe import Planes
from neural.checkpoints import atomic_save


def config():
    return {'resnet':{'conv_channels':16,'num_blocks':1},'control':{'version':4}}


class TrainingRoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def simulate(self, net, games, seed, device, **kwargs):
        net.eval(); n=32
        observations=Planes(np.arange(n+1,dtype=np.int64),np.zeros(n,dtype=np.uint16),
                            np.ones(n,dtype=np.float16))
        legal=torch.ones(n,46,dtype=torch.bool)
        with torch.no_grad():
            logits,_=net.policy(observations.dense('cpu'),legal)
            distribution=torch.distributions.Categorical(logits=logits)
            actions=distribution.sample();log_probs=distribution.log_prob(actions)
        self.actions.append(actions.clone())
        return SimpleNamespace(observations=observations,legal=legal,actions=actions,
            returns=torch.linspace(-1.,1.,n), log_probs=log_probs,games=games,hands=1,
            decisions=n,timing={})

    def mortal_run(self, out, source, rounds, resume=False, batch=16):
        args=['trainer','--resume' if resume else '--mortal',str(source),'--rounds',str(rounds),
              '--out',str(out),'--games','1','--batch',str(batch),'--epochs','1','--seed','42',
              '--measure-every','100']
        log=io.StringIO()
        with patch.object(sys,'argv',args),patch.object(train_mortal.selfplay,'play',side_effect=self.simulate), \
             patch.object(train_mortal.selfplay,'measure',return_value={'placement':2.5,'score':30000.,'wins':.25}), \
             patch.object(train_mortal,'pad_rows',side_effect=lambda x,*a,**kw:x),redirect_stdout(log):
            train_mortal.main()
        return torch.load(out/'latest.pt',map_location='cpu',weights_only=True),log.getvalue()

    def test_actual_mortal_trainer_serialized_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);initial=root/'initial.pt';torch.manual_seed(7)
            net=mortal_model.build(16,1);atomic_save({**net.state(),'config':config()},initial)
            self.actions=[];continuous,log=self.mortal_run(root/'continuous',initial,2)
            actions=[a.clone() for a in self.actions]
            self.actions=[];self.mortal_run(root/'split',initial,1)
            resumed,_=self.mortal_run(root/'split',root/'split/latest.pt',1,resume=True)
            for first,second in zip(actions,self.actions): self.assertTrue(torch.equal(first,second))
            for part in ('mortal','current_dqn','value_head'):
                for key,tensor in continuous[part].items():
                    self.assertTrue(torch.equal(tensor,resumed[part][key]), (part,key))
            self.assertTrue(torch.equal(continuous['sampling_state']['torch'],resumed['sampling_state']['torch']))
            self.assertEqual(resumed['generation'],2)
            records=[json.loads(line) for line in log.splitlines() if line.startswith('{')]
            self.assertEqual([record['optimizer_updates'] for record in records],[2,2])
            self.assertEqual([record['checkpoint_generation'] for record in records],[1,2])

    def test_all_three_trainers_refuse_underfilled_round_before_checkpoint_change(self):
        for module in (train,train_mortal,train_combined):
            with self.subTest(trainer=module.__name__),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);source=root/'latest.pt'
                if module is train:
                    net=model.PolicyValueNet(8,1)
                    payload={'model':net.state_dict(),**net.payload_fields()}
                elif module is train_mortal:
                    # The published Mortal format is enough for from_mortal,
                    # but a resume checkpoint also needs its learned value head.
                    mortal=mortal_model.build(16,1)
                    from neural import mortal_learner
                    initial=root/'origin.pt';atomic_save({**mortal.state(),'config':config()},initial)
                    net=mortal_learner.from_mortal(initial,'cpu',1.)
                    payload={**net.state(),'config':config(),'learner':'mortal'}
                else:
                    net=combined.Combined(model.PolicyValueNet(8,1,actions=46),mortal_model.build(16,1))
                    net.mortal_config=config();payload=net.state()
                atomic_save({**payload,'generation':4},source); before=source.read_bytes()
                args=['trainer','--resume',str(source),'--rounds','1','--out',str(root),'--games','1',
                      '--batch','64','--epochs','1']
                log=io.StringIO()
                with patch.object(sys,'argv',args),patch.object(module.selfplay,'play',return_value=SimpleNamespace(decisions=32)),redirect_stdout(log):
                    with self.assertRaisesRegex(ValueError,'No optimizer updates possible'):
                        module.main()
                self.assertEqual(source.read_bytes(),before)
                self.assertFalse((root/'log.jsonl').exists())

    def test_invalid_learning_options_fail_before_model_construction(self):
        for module in (train,train_mortal,train_combined):
            for name,value in (('--epochs','0'),('--batch','0'),('--epochs','-1')):
                with self.subTest(module=module.__name__,name=name,value=value),patch.object(sys,'argv',['trainer',name,value]):
                    with self.assertRaisesRegex(ValueError,'positive integer'):module.main()
