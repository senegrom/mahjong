"""Reconstruction and densification, not only inference, obey the leaf budget."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import riichi_py

from neural import contract, searched
from neural.observe import Planes


class LeafBatchTests(unittest.TestCase):
    def test_mortal_reconstruction_encoding_and_densification_are_bounded(self):
        sizes = []

        class Copies:
            @classmethod
            def from_follower(cls, follower, who, version):
                sizes.append(len(who))
                copy = cls()
                copy.who = who
                return copy

            def feed(self, lines):
                self.lines = lines

            def encode(self):
                n = len(self.who)
                values = [float(line[0]) for line in self.lines]
                return np.arange(n + 1), np.zeros(n, dtype=np.uint16), values, None

        count = 77
        arena = SimpleNamespace(leaves_mjai=lambda: ([0] * count, [[str(i + 1)] for i in range(count)]))
        contract.remember_follower(arena, object())
        net = SimpleNamespace(kind='mortal', planes=1012, actions=46, everything=lambda *a: None)
        served = contract.serve(net)
        served._Imagined = Copies
        wanted = np.array([i % 3 != 0 for i in range(count)], dtype=np.uint8)
        dense = Planes.dense

        def bounded(planes, device):
            self.assertLessEqual(len(planes), 7)
            return dense(planes, device)

        values = np.zeros(count)
        with patch.object(Planes, 'dense', bounded):
            for slots, planes in served.leaf_batches(arena, b'', [3, 0, 74], 'cpu', wanted=wanted, batch_size=7):
                self.assertLessEqual(len(slots), 7)
                values[slots] = planes[:, 0, 0].numpy()
        np.testing.assert_array_equal(values, np.where(wanted, np.arange(1, count + 1), 0))
        self.assertGreater(len(sizes), 1)
        self.assertLessEqual(max(sizes), 7)

    def test_native_leaf_transfers_respect_the_same_budget(self):
        shape = (13, riichi_py.PLANES, 34)
        array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        net = SimpleNamespace(kind='engine', planes=riichi_py.PLANES, actions=riichi_py.ACTIONS,
                              everything=lambda *a: None)
        served = contract.serve(net)
        wanted = bytes([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
        seen = []
        for slots, planes in served.leaf_batches(None, array.tobytes(), [6, 7], 'cpu', wanted=wanted, batch_size=2):
            self.assertLessEqual(len(slots), 2)
            np.testing.assert_array_equal(planes.numpy(), array[slots])
            seen.extend(slots)
        self.assertEqual(seen, np.flatnonzero(list(wanted)).tolist())

    def test_terminal_slots_never_reconstruct_or_call_a_critic(self):
        net = SimpleNamespace(kind='mortal', planes=1012, actions=46, everything=lambda *a: None)
        served = contract.serve(net)
        served._Imagined = SimpleNamespace(from_follower=lambda *a: self.fail('terminal encoding'))
        arena = SimpleNamespace(leaves_mjai=lambda: ([0, 0], [[], []]))
        self.assertEqual(list(served.leaf_batches(arena, b'', [2], 'cpu', wanted=b'\0\0', batch_size=1)), [])

    def test_real_search_uses_the_batched_contract_not_the_dense_helper(self):
        class Arena:
            def imagine(self, *args, **kwargs): return b'', [1]
            def leaves_from(self, *args, **kwargs): return b'', [3], [0., 0., 0.], [1, 0, 1]
            def decide(self, values, *args): return values

        def batches(*args, **kwargs):
            self.assertEqual(kwargs['batch_size'], 1)
            for slot in (0, 2):
                yield np.array([slot]), torch.tensor([[[slot + 1.]]])

        served = SimpleNamespace(contract=SimpleNamespace(reads='mortal'), leaf_batches=batches,
                                 leaves=lambda *a, **kw: self.fail('unbounded leaf helper'),
                                 value=lambda x, head: x[:, 0, 0])
        values = searched.search_with_value_head(object(), Arena(), [[0, 1]], [], worlds=1,
            candidates=2, margin=0., hurried=True, device='cpu', pool=1, served=served, leaf_batch=1)
        self.assertEqual(values, [1., 0., 3.])

    def test_bad_metadata_and_batch_limits_fail_before_reconstruction(self):
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                contract.leaf_slots([1], [1], size)
        for wanted in ([2], [0, 1], [float('nan')]):
            with self.assertRaises(ValueError):
                contract.leaf_slots([1], wanted, 1)


if __name__ == '__main__':
    unittest.main()
