"""Real checkpoint/serving boundaries, not just metadata helper round trips."""
from pathlib import Path
import tempfile
import unittest

import torch
import riichi_py

from neural import model, zoo
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
                    restored = zoo.unwrap(zoo.load_player(path, 'cpu'))
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


class EngineObservationContractTests(unittest.TestCase):
    """A network of the engine's kind is only loaded onto the planes it was
    trained on. Version 1 of those planes miscounted the unseen tiles, and
    every checkpoint saved before the version was recorded is version 1."""

    def test_the_running_engine_says_which_planes_it_writes(self):
        self.assertEqual(model.ENGINE_OBSERVATION, riichi_py.OBSERVATION_VERSION)
        self.assertGreaterEqual(model.ENGINE_OBSERVATION, 2)

    def test_an_engine_network_records_the_planes_it_was_trained_on(self):
        with tempfile.TemporaryDirectory() as temp:
            net = model.PolicyValueNet(8, 1, planes=riichi_py.PLANES).eval()
            payload = {'model': net.state_dict(), **net.payload_fields()}
            self.assertEqual(payload['engine_observation'], model.ENGINE_OBSERVATION)
            path = Path(temp) / 'engine.pt'
            atomic_save(payload, path)
            self.assertEqual(zoo.load_player(path, 'cpu').planes, riichi_py.PLANES)

    def test_an_engine_network_from_older_planes_is_refused(self):
        net = model.PolicyValueNet(8, 1, planes=riichi_py.PLANES)
        fields = net.payload_fields()
        del fields['engine_observation']
        old = {'model': net.state_dict(), **fields}
        # A checkpoint from before the shape was recorded says nothing at all.
        oldest = {'model': net.state_dict()}
        for payload in (old, oldest, {**old, 'engine_observation': 1}):
            with self.subTest(keys=sorted(payload)):
                with self.assertRaisesRegex(ValueError, 'version 1'):
                    model.from_payload(payload, 'cpu')
                with self.assertRaisesRegex(ValueError, 'version 1'):
                    model.shape_of(payload)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'w320.pt'
            atomic_save({**old, 'generation': 290}, path)
            with self.assertRaisesRegex(ValueError, 'engine writes version'):
                zoo.load_player(path, 'cpu')
            # The same half inside a fusion.
            fused = Path(temp) / 'fused.pt'
            torch.save({'combined': {}, **old}, fused)
            from neural import combined
            with self.assertRaisesRegex(ValueError, 'engine writes version'):
                combined.load(fused, 'cpu')

    def test_mortal_planes_are_not_the_engines_to_version(self):
        net = model.PolicyValueNet(8, 1, actions=46)
        payload = {'model': net.state_dict(), **net.payload_fields()}
        self.assertNotIn('engine_observation', payload)
        self.assertEqual(model.from_payload(payload, 'cpu').kind, 'mortal')

if __name__ == '__main__': unittest.main()
