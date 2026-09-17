"""Every public placement architecture survives the real artifact/replay boundary."""
from pathlib import Path
import hashlib
import tempfile
import unittest

import torch

from neural import placement, contract
from neural.model import PolicyValueNet
from neural.checkpoints import atomic_save
from neural.collect_search import collect, SearchSettings, metadata
from neural.search_replay import SearchReplay, validate_metadata
from neural.train_search import run


class PlacementVersions(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.net = PolicyValueNet(8, 1, actions=46).eval()

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_tower_depth_and_predictions_survive_training_save_load_and_resave(self):
        planes = torch.randn(2, 1012, 34)
        for version in (1, 2, 3, 4):
            for looks in (1, 2, 4, 6):
                with self.subTest(version=version, looks=looks), tempfile.TemporaryDirectory() as tmp:
                    head = placement.new_head(self.net, feature_version=version, looks=looks)
                    opt = torch.optim.AdamW(head.parameters())
                    (head.judge(self.net, planes) - 1).square().mean().backward()
                    opt.step(); head.eval()
                    path = Path(tmp) / 'head.pt'
                    meta = dict(features=placement.fingerprint(self.net, version), note='a real update')
                    placement.save(head, path, meta)
                    loaded, meta = placement.load(path)
                    self.assertEqual(loaded.looks, looks if version in (3, 4) else 0)
                    placement.require_head_for(meta, self.net)
                    torch.testing.assert_close(loaded.judge(self.net, planes), head.judge(self.net, planes), rtol=0, atol=0)
                    served = contract.serve(self.net, placement_head=loaded)
                    torch.testing.assert_close(contract.placement_value(served, planes), head.judge(self.net, planes))
                    placement.save(loaded, Path(tmp) / 'again.pt', meta)
                    with self.assertRaises(ValueError):
                        placement.save(loaded, path, {**meta, 'looks': looks + 10})
                    self.assertEqual(placement.load(path)[0].looks, loaded.looks)

    def test_legacy_zero_depth_is_reconstructed_from_contiguous_saved_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'head.pt'
            head = placement.new_head(self.net, feature_version=4, looks=2)
            placement.save(head, path, dict(features='planes-1012'))
            payload = torch.load(path, weights_only=True); payload['looks'] = 0
            torch.save(payload, path)
            loaded, meta = placement.load(path)
            self.assertEqual(loaded.looks, 2)
            placement.save(loaded, path, meta)
            self.assertEqual(torch.load(path, weights_only=True)['looks'], 2)

    def test_replay_keeps_strict_version_specific_provenance(self):
        for version in (1, 2, 3, 4):
            head = dict(sha256='a' * 64, features=placement.fingerprint(self.net, version), feature_version=version)
            m = metadata(games=1, seed=2, settings=SearchSettings(valued_by='placement', depth=-1),
                         actor_sha256='b' * 64, source_revision='c' * 40, training_api_version=2,
                         head_provenance=head)
            for bad in ({**head, 'feature_version': True}, {**head, 'feature_version': 5},
                        {**head, 'sha256': ''}, {**head, 'features': 'unknown'},
                        {**head, 'feature_version': 4, 'features': 'planes-97'}):
                m['teacher']['placement_head'] = bad
                with self.assertRaises(ValueError): validate_metadata(m)

    def test_every_version_collects_reopens_trains_and_resumes_native_teacher_replay(self):
        for version in (1, 2, 3, 4):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                head = placement.new_head(self.net, feature_version=version, looks=2)
                placement.save(head, root / 'head.pt', dict(features=placement.fingerprint(self.net, version)))
                atomic_save({**self.net.payload_fields(), 'model': self.net.state_dict(), 'generation': 1}, root / 'actor.pt')
                actor_hash = hashlib.sha256((root / 'actor.pt').read_bytes()).hexdigest()
                replay = collect(self.net, games=1, seed=77,
                    settings=SearchSettings(valued_by='placement', objective='placement', depth=-1, sure=0),
                    actor_sha256=actor_hash, source_revision='b' * 40,
                    placement_head=root / 'head.pt', device='cpu')
                replay.save(root / 'replay')
                loaded = SearchReplay.load(root / 'replay')
                self.assertEqual(loaded.metadata['teacher']['placement_head']['feature_version'], version)
                first = run(root / 'actor.pt', [root / 'replay'], root / 'first', epochs=1,
                            overrides=dict(batch=32, validation_every=0, policy_mode='changed', actor_kl=1.))
                second = run(first, [root / 'replay'], root / 'second', epochs=1, resume=True)
                self.assertEqual(torch.load(second, weights_only=True)['search_training']['epochs'], 2)


if __name__ == '__main__': unittest.main()
