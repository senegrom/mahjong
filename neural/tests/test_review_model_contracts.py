"""Real checkpoint/serving boundaries, not just metadata helper round trips."""
from pathlib import Path
import tempfile
import unittest

import torch
import riichi_py

from neural import contract, model, zoo
from neural.checkpoints import atomic_save


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_reader_declaration_roundtrips(self):
        for planes, actions in ((riichi_py.PLANES, riichi_py.ACTIONS), (1012, 46)):
            for version in (None, 3, 4, 999):
                with self.subTest(planes=planes, version=version), tempfile.TemporaryDirectory() as temp:
                    torch.manual_seed(5)
                    net = model.PolicyValueNet(8, 1, planes=planes, actions=actions).eval()
                    if version is not None:
                        # Synthetic declaration exercises the contract, NOT a calibration claim.
                        net.reader_proposal_version = version
                    path = Path(temp) / 'reader.pt'
                    atomic_save({'model': net.state_dict(), **net.payload_fields()}, path)
                    restored = contract.unwrap(zoo.load_player(path, 'cpu'))
                    self.assertEqual(getattr(restored, 'reader_proposal_version', None), version)
                    if version is None:
                        self.assertNotIn('reader_proposal_version', restored.payload_fields())
    def test_explicit_file_marker_loads_but_malformed_declarations_fail(self):
        net = model.PolicyValueNet(8, 1, actions=46)
        payload = dict(model=net.state_dict(), **net.payload_fields())
        # Producer-authored old files with an explicit marker need not have
        # called the new metadata writer first.
        loaded = model.from_payload({**payload, 'reader_proposal_version': 4}, 'cpu')
        self.assertEqual(loaded.reader_proposal_version, 4)
        for value in (None, True, 4., '4', 0, -1, 2**32):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'reader_proposal_version'):
                model.from_payload({**payload, 'reader_proposal_version': value}, 'cpu')
            net.reader_proposal_version = value
            with self.assertRaises(ValueError): net.payload_fields()

if __name__ == '__main__': unittest.main()
