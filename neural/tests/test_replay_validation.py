"""Sparse columns must never escape their own observation."""
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from neural.observe import Planes, WIDTH
from neural.replay import Ring
from neural.tests.test_replay_atomic import batch


class ReplayValidationTests(unittest.TestCase):
    def test_cross_row_column_is_rejected_on_push_dense_and_reopen(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root=Path(folder); ring=Ring(root, 1); ring.push(batch(1, 2))
            old=ring.index.read_bytes()
            bad=batch(7,2)
            bad.observations=Planes(np.array([0,1,1],dtype=np.int64),
                                    np.array([WIDTH+5],dtype=np.uint16),
                                    np.array([7],dtype=np.float16))
            with self.assertRaisesRegex(ValueError,'bounds'):
                ring.push(bad)
            with self.assertRaisesRegex(ValueError,'bounds'):
                bad.observations.dense('cpu')
            self.assertEqual(old, ring.index.read_bytes())
            generation=json.loads(old)['entries'][0]['generation']
            np.save(root/generation/'observations-indices.npy', np.array([WIDTH+5,0],dtype=np.uint16))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); reopened=Ring(root,1)
            self.assertEqual(len(reopened),0)
            self.assertTrue(any('Ignoring incomplete' in str(w.message) for w in caught))

    def test_valid_edge_columns_preserve_rows(self):
        values=Planes(np.array([0,1,2],dtype=np.int64),
                      np.array([WIDTH-1,0],dtype=np.uint16), np.array([3,7],dtype=np.float16))
        dense=values.dense('cpu').reshape(2,WIDTH).numpy()
        self.assertEqual(dense[0,-1],3);self.assertEqual(dense[1,0],7)
        self.assertEqual(np.count_nonzero(dense),2)
        Planes.empty().validate()

    def test_wrong_dtypes_offsets_and_nonfinite_values_fail(self):
        good=(np.array([0,1,1],dtype=np.int64), np.array([0],dtype=np.uint16),
              np.array([1],dtype=np.float16))
        for component, replacement in (
                (0,np.array([],dtype=np.int64)),(0,np.array([0,2,1],dtype=np.int64)),
                (0,np.array([0,1,1],dtype=np.uint64)),(0,np.array([0,1,2],dtype=np.int64)),
                (1,np.array([0],dtype=np.int32)),(1,np.array([-1],dtype=np.int16)),
                (2,np.array([np.nan],dtype=np.float16)),(2,np.array([np.inf],dtype=np.float32)),
                (2,np.array([1],dtype=object))):
            arrays=list(good);arrays[component]=replacement
            with self.subTest(component=component, array=str(replacement)):
                with self.assertRaises(ValueError): Planes(*arrays).validate(expected_rows=2)
