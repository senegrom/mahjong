"""Reject broadcastable/wrong dense targets before they reach any loss."""
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import torch

from neural.replay_schema import validate_dense_replay


def arrays(n=2):
    import riichi_py as r
    return dict(legal=np.ones((n,r.ACTIONS),bool),
                held=np.full((n,r.OPPONENTS,r.POSITIONS),1/r.POSITIONS,np.float32),
                oracle=np.zeros((n,r.ORACLE_PLANES,r.POSITIONS),np.uint8),
                imagined=np.zeros((n,r.HIDDEN_HANDS_PLANES,r.POSITIONS),np.uint8),
                returns=np.arange(n,dtype=np.float32))


def bad_cases():
    yield 'returns',np.array([[0.],[1.]],np.float32)
    yield 'returns',np.array([0.,float('nan')],np.float32)
    yield 'returns',np.array([0.,float('inf')],np.float32)
    yield 'returns',np.array([0.,1.],np.float64)
    good=arrays()
    yield 'legal',good['legal'].astype(np.uint8)
    yield 'legal',np.zeros_like(good['legal'])
    yield 'held',np.full_like(good['held'],2.)
    yield 'held',np.full_like(good['held'],float('nan'))
    yield 'held',np.full_like(good['held'],.001)
    yield 'held',good['held'][:,:,:1]
    yield 'oracle',np.full_like(good['oracle'],2)
    yield 'imagined',good['imagined'].astype(np.float32)


class DenseReplayHelpersTests(unittest.TestCase):
    def test_canonical_arrays_and_empty_hands_are_allowed(self):
        good=arrays();validate_dense_replay(good,2)
        good['held'][0,0]=0;validate_dense_replay(good,2)

    def test_extra_dimensions_dtypes_and_bad_values_are_rejected(self):
        for field,value in bad_cases():
            with self.subTest(field=field,shape=value.shape,dtype=value.dtype):
                data=arrays();data[field]=value
                with self.assertRaises(ValueError):validate_dense_replay(data,2)

    def test_chunk_boundaries_are_checked(self):
        data=arrays(4097);data['returns'][-1]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):validate_dense_replay(data,4097)


class DenseReplayBoundaryTests(unittest.TestCase):
    def test_push_and_reopen_reject_every_bad_dense_schema(self):
        from neural.replay import Ring
        from neural.tests.test_replay_atomic import batch
        for field,value in bad_cases():
            with self.subTest(field=field),tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
                root=Path(folder);ring=Ring(root,1);ring.push(batch(1,n=2))
                before=ring.index.read_bytes()
                bad=batch(2,n=2);setattr(bad,field,torch.from_numpy(value.copy()))
                with self.assertRaises(ValueError):ring.push(bad)
                self.assertEqual(ring.index.read_bytes(),before)
                np.testing.assert_array_equal(ring.sample(2,np.random.default_rng(0))['returns'],[1,1])
                generation=json.loads(before)['entries'][0]['generation']
                del ring  # its maps of the batch must close before the file is rewritten
                np.save(root/generation/(field+'.npy'),value)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');reopened=Ring(root,1)
                self.assertEqual(len(reopened),0)
                self.assertTrue(any('Ignoring incomplete' in str(w.message) for w in caught))
