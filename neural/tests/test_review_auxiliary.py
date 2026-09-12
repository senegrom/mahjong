"""No auxiliary entry point may publish untrained output after a bad request."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from neural.auxiliary_training import validate_auxiliary_options, require_supervised_rows


def options(**changes):
    values=dict(batch=2,epochs=1,rounds=1,games=1,lr=.001,temperature=1.,
                resume=None,teacher=None,measure_every=1,measure_games=1)
    values.update(changes)
    return SimpleNamespace(**values)


class AuxiliaryHelperTests(unittest.TestCase):
    def test_bad_options_and_missing_resume_fail(self):
        for kwargs in ({'batch':1},{'epochs':0},{'rounds':0},{'rounds':-1},{'games':0},
                       {'lr':float('nan')},{'temperature':0},{'max_steps':0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_auxiliary_options(options(**kwargs))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                validate_auxiliary_options(options(resume=Path(folder)/'missing.pt'))
        validate_auxiliary_options(options(batch=2048))

    def test_partial_batches_are_allowed_but_singleton_rounds_are_not(self):
        for rows in (0,1,-1,True):
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                require_supervised_rows(rows)
        require_supervised_rows(2)
        require_supervised_rows(2049)


class AuxiliaryEntryTests(unittest.TestCase):
    def test_real_entrypoints_reject_before_loading_models_collecting_or_publishing(self):
        from neural import imitate, rehead, distil
        for module in (imitate,rehead,distil):
            with self.subTest(module=module.__name__),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);destination=root/'latest.pt';destination.write_bytes(b'prior checkpoint')
                for bad in ({'batch':1},{'epochs':0},{'rounds':0},{'resume':root/'missing.pt'}):
                    args=options(out=root,**bad)
                    with patch.object(module,'parse_args',return_value=args), \
                         patch.object(module,'collect') as collect, \
                         patch.object(module.torch,'load') as load, \
                         patch.object(module,'atomic_save') as save, \
                         redirect_stdout(io.StringIO()):
                        with self.assertRaises((ValueError,FileNotFoundError)):
                            module.main()
                        load.assert_not_called();collect.assert_not_called();save.assert_not_called()
                        self.assertEqual(destination.read_bytes(),b'prior checkpoint')

    def test_real_trainers_reject_singleton_rounds_and_update_valid_small_rounds(self):
        import sys
        import numpy as np
        import torch
        import riichi_py
        from neural import imitate,rehead,distil
        from neural.model import PolicyValueNet,MORTAL_PLANES
        from neural.observe import Planes
        from neural.checkpoints import atomic_save
        for module in (imitate,rehead,distil):
            for n in (1,2):
                with self.subTest(module=module.__name__,rows=n),tempfile.TemporaryDirectory() as folder:
                    root=Path(folder);out=root/'out';out.mkdir()
                    destination=out/'latest.pt';destination.write_bytes(b'last good output')
                    width=46 if module is rehead else 78
                    planes=riichi_py.PLANES if module is distil else MORTAL_PLANES
                    student=PolicyValueNet(8,1,planes,attention=False,actions=width)
                    before={key:value.detach().clone() for key,value in student.state_dict().items()}
                    saved=root/'student.pt'
                    atomic_save({'model':before,'generation':0,**student.payload_fields()},saved)
                    command=[module.__name__,'--resume',str(saved),'--rounds','1','--games','1',
                             '--batch','2','--epochs','1','--out',str(out)]
                    if module is rehead:
                        teacher=PolicyValueNet(8,1,MORTAL_PLANES,attention=False,actions=78)
                        teacher_path=root/'teacher.pt'
                        atomic_save({'model':teacher.state_dict(),'generation':0,**teacher.payload_fields()},teacher_path)
                        command+=['--teacher',str(teacher_path)]
                    sparse=Planes.from_follower(np.zeros(n+1,np.int64),[],[])
                    masks=np.ones((n,width),bool)
                    labels=np.arange(n,dtype=np.int64)%2
                    truth=np.zeros((n,3,34),np.float32)
                    if module is imitate:
                        collected=(sparse,masks,labels,truth,None)
                    elif module is rehead:
                        targets=np.zeros((n,width),np.float32);targets[np.arange(n),labels]=1
                        collected=(sparse,masks,targets,0)
                    else:
                        collected=(np.zeros((n,planes,34),np.float32),masks,labels,truth,labels.copy(),
                                   np.linspace(-1,1,n,dtype=np.float32),
                                   np.full((n,width),1/width,dtype=np.float32))
                    with patch.object(sys,'argv',command),patch.object(module,'collect',return_value=collected), \
                         patch.object(torch.cuda,'is_available',return_value=False),redirect_stdout(io.StringIO()):
                        from contextlib import ExitStack
                        with ExitStack() as stack:
                            measurement=module.selfplay if module is imitate else module
                            if module is not rehead:
                                stack.enter_context(patch.object(measurement,'measure',return_value={
                                    'placement':2.5,'score':0.,'wins':0.}))
                            if n==1:
                                with self.assertRaisesRegex(ValueError,'two training rows'):
                                    module.main()
                                self.assertEqual(destination.read_bytes(),b'last good output')
                            else:
                                module.main()
                                import json
                                record=json.loads((out/'log.jsonl').read_text().splitlines()[-1])
                                self.assertGreater(record['optimizer_updates'],0)
                                learned=torch.load(destination,weights_only=True)['model']
                                self.assertTrue(any(not torch.equal(before[key],learned[key]) for key in before))
