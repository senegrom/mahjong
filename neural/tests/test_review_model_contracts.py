"""Real checkpoint/serving boundaries, not just metadata helper round trips."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import riichi_py

from neural import contract, duel, model, ranked, searched, sibling_head, zoo
from neural.checkpoints import atomic_save
from neural.observe import Views


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_reader_declaration_roundtrips_and_controls_actual_search(self):
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
                    arena = riichi_py.Arena(games=1, seed=11, bot_places=[])
                    views = Views(arena, 1, {restored.kind})
                    views.advance()
                    legal = np.frombuffer(arena.legal_mask(), np.uint8).reshape(1, 78).astype(bool)
                    ranking = [np.flatnonzero(legal[0])[:2].tolist()]
                    health = {}
                    served = contract.serve(restored)
                    follower = views.observer.follower if restored.kind == 'mortal' else None
                    with contract.following(arena, follower), patch.object(
                        restored, 'read_plausibility', wraps=restored.read_plausibility
                    ) as reader:
                        searched.search_with_value_head(restored, arena, ranking, [1/34]*102,
                            worlds=3, candidates=2, margin=2., hurried=True, device='cpu',
                            pool=2, played_by='club', served=served, leaf_batch=2, health=health)
                    self.assertEqual(reader.call_count, 3 if version == 4 else 0)
                    expected = restored.kind if version == 4 else 'uniform_unverified_proposal'
                    self.assertEqual(health['world_reader'], expected)

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

    def test_ranker_roundtrip_accepts_only_its_supporting_features(self):
        torch.manual_seed(1)
        actor = model.PolicyValueNet(8, 1, actions=46).eval()
        head = sibling_head.new_head(actor)
        with torch.no_grad():
            head.tiles.weight.normal_()
            head.rest.weight.normal_()
        planes = torch.randn(2, 1012, 34)
        with torch.no_grad(): expected = head(*sibling_head.features_of(actor, planes))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'actor.pt'
            saved_head = Path(temp) / 'head.pt'
            atomic_save({'model': actor.state_dict(), **actor.payload_fields()}, path)
            sibling_head.save(head, saved_head, {'checkpoint': str(path)})
            reloaded_actor = contract.unwrap(zoo.load_player(path, 'cpu'))
            reloaded_head, _ = sibling_head.load(saved_head)
            player = ranked.RankedPlayer(reloaded_actor, reloaded_head, device='cpu')
            with torch.no_grad(): actual = player.head(*sibling_head.features_of(player.net, planes))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            other = model.PolicyValueNet(8, 1, actions=46).eval()
            with self.assertRaisesRegex(ValueError, 'different supporting features'):
                ranked.RankedPlayer(other, reloaded_head, device='cpu')
            # The real CLI/cloud entry point must also refuse a mutable path
            # that now holds another same-shaped actor, before starting a duel.
            atomic_save({'model': other.state_dict(), **other.payload_fields()}, path)
            with patch('sys.argv', ['ranked', str(path), str(saved_head), '--device', 'cpu']), \
                    patch.object(duel, 'duel') as play, self.assertRaises(ValueError):
                ranked.main()
            play.assert_not_called()

    def test_ranker_refuses_legacy_and_relabelled_contracts(self):
        actor = model.PolicyValueNet(8, 1, actions=46).eval()
        head = sibling_head.new_head(actor)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'head.pt'
            sibling_head.save(head, path, {})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                sibling_head.save(head, path, {'feature_contract': head.feature_contract})
            with self.assertRaises(ValueError):
                sibling_head.save(sibling_head.Ranker(8), path, {})
            self.assertEqual(before, path.read_bytes())
            torch.save({'ranker': head.state_dict(), 'channels': 8,
                        'checkpoint': 'actor.pt'}, path)
            legacy, _ = sibling_head.load(path)
            with self.assertRaisesRegex(ValueError, 'supporting feature contract'):
                ranked.RankedPlayer(actor, legacy, device='cpu')
            payload = torch.load(Path(temp) / 'head.pt', weights_only=True)
            for changes in ({'version': 2}, {'version': True}, {'sha256': 'wrong'}, {'channels': 16}):
                torch.save({**payload, 'feature_contract': {**head.feature_contract, **changes}}, path)
                with self.subTest(changes=changes), self.assertRaises(ValueError): sibling_head.load(path)

    def test_ranker_contract_includes_observation_and_action_meanings(self):
        actor = model.PolicyValueNet(8, 1, actions=46).eval()
        head = sibling_head.new_head(actor)
        # Identical feature tensors are not sufficient when the actor's action
        # contract changed from Mortal's conditional riichi to engine actions.
        other = model.PolicyValueNet(8, 1, actions=78).eval()
        for name in ('stem', 'tower', 'tail'):
            getattr(other, name).load_state_dict(getattr(actor, name).state_dict())
        with self.assertRaises(ValueError): ranked.RankedPlayer(other, head, device='cpu')


if __name__ == '__main__': unittest.main()
