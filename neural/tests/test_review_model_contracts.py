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

    def test_a_checkpoint_saved_with_the_reader_of_hidden_hands_still_loads(self):
        # The reader served the search removed in September 2026. A
        # checkpoint from before then carries its weights and its
        # declaration, and loads with both left behind, alone or in a fusion.
        from neural import combined, mortal_model
        reader = {'reader_stem.0.weight': torch.zeros(128, 1048, 3),
                  'reader_tower.0.conv1.weight': torch.zeros(128, 128, 3),
                  'reader_tail.0.weight': torch.ones(128),
                  'reader.2.bias': torch.zeros(1)}
        with tempfile.TemporaryDirectory() as temp:
            torch.manual_seed(5)
            net = model.PolicyValueNet(8, 1, actions=46).eval()
            old = {'model': {**net.state_dict(), **reader}, **net.payload_fields(),
                   'reader_proposal_version': 4}
            path = Path(temp) / 'reader.pt'
            atomic_save(old, path)
            restored = zoo.unwrap(zoo.load_player(path, 'cpu'))
            for key, value in net.state_dict().items():
                self.assertTrue(torch.equal(restored.state_dict()[key], value), key)
            self.assertFalse(any(key.startswith('reader') for key in restored.state_dict()))
            self.assertNotIn('reader_proposal_version', restored.payload_fields())
            fused = combined.Combined(net, mortal_model.build(8, 1))
            fused.mortal_config = {'resnet': {'conv_channels': 8, 'num_blocks': 1}, 'control': {'version': 4}}
            state = fused.state()
            state['model'] = {**state['model'], **reader}
            torch.save({**state, 'reader_proposal_version': 4}, Path(temp) / 'fused.pt')
            loaded, _state = combined.load(Path(temp) / 'fused.pt', 'cpu')
            self.assertFalse(any(key.startswith('reader') for key in loaded.ours.state_dict()))



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
