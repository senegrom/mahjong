"""Teacher contracts, feature compatibility, fresh confirmation and fast inference."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import placement, model, combined, mortal_model, searched
from neural.teacher_options import candidate_set, validate_controls
from neural.collect_search import SearchSettings, metadata
from neural.search_replay import validate_metadata, student_value_head


class TeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_new_placement_features_are_distinct_and_include_mortal(self):
        ours = model.PolicyValueNet(8, 1, actions=46).eval()
        net = combined.Combined(ours, mortal_model.build(16, 1)).eval()
        x = torch.randn(2, ours.planes, 34)
        features, context = placement.features_of(net, x)
        torch.testing.assert_close(context[:, :8], features.amax(2))
        torch.testing.assert_close(context[:, 8:], net.mortal.features(x).float())
        head = placement.new_head(net)
        self.assertEqual(head(features, context).shape, (2,))
        self.assertFalse(torch.allclose(features.mean(2), context[:, :8]))
        before = placement.fingerprint(net, 2)
        with torch.no_grad(): next(net.mortal.parameters()).add_(1)
        self.assertNotEqual(placement.fingerprint(net, 2), before)

    def test_legacy_placement_head_loads_with_exact_original_features(self):
        net = model.PolicyValueNet(8, 1, actions=46).eval()
        x = torch.randn(2, net.planes, 34)
        head = placement.Judge(8, feature_version=1)
        with torch.no_grad(): head.body[2].weight.normal_()
        features, context = placement.features_of(net, x, 1)
        expected = head(features, context)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            torch.save(dict(judge=head.state_dict(), channels=8, hidden=64,
                            placement_version=1, features=placement.fingerprint(net)), path)
            loaded, meta = placement.load(path)
            placement.require_head_for(meta, net)
            actual = loaded(*placement.features_of(net, x, loaded.feature_version))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(loaded.feature_version, 1)

    def test_fast_policy_logits_match_and_skip_auxiliary_towers(self):
        ours = model.PolicyValueNet(8, 1, actions=46).eval()
        joined = combined.Combined(ours, mortal_model.build(16, 1)).eval()
        x = torch.randn(2, ours.planes, 34)
        legal = torch.ones(2, 46, dtype=torch.bool); legal[:, 3] = False
        for net in (ours, joined):
            reference = net(x, legal)[0]
            with patch.object(ours, 'hands_from', side_effect=AssertionError('unused belief tower')):
                torch.testing.assert_close(net.policy_only(x, legal), reference, rtol=0, atol=0)

    def test_candidate_expansion_preserves_incumbent_and_legality(self):
        legal = np.zeros(78, bool); legal[[1, 2, 3, 35, 70, 74]] = True
        order = [1, 1, 2, 3, 35, 74, 70, -1, 90]
        chosen = candidate_set(order, legal, 2, 3, np.random.default_rng(1))
        self.assertEqual(chosen, [1, 2, 35, 74, 70])
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertTrue(legal[chosen].all())

    def test_bad_teacher_combinations_fail_before_search(self):
        for options in (dict(objective='placement'), dict(search_calls=True, played_by='club'),
                        dict(confirm_worlds=2), dict(confirm_worlds=True), dict(extra_candidates=-1),
                        dict(audit_share=float('nan')), dict(valued_by='placement', depth=0)):
            with self.subTest(options=options), self.assertRaises(ValueError): validate_controls(**options)

    def test_teacher_and_student_value_contracts_cannot_be_relabelled(self):
        settings = SearchSettings(objective='placement', valued_by='placement', depth=-1,
                                  search_calls=True, confirm_worlds=16)
        m = metadata(games=2, seed=12, settings=settings, actor_sha256='a'*64,
                     source_revision='b'*40, training_api_version=2,
                     head_provenance=dict(sha256='c'*64, features='d'*16, feature_version=2))
        validate_metadata(m)
        self.assertEqual(student_value_head(m), 'critic')
        for field in ('placement_head', 'student_value_head', 'search_api_version'):
            bad=deepcopy(m); del bad['teacher'][field]
            with self.subTest(field=field), self.assertRaises(ValueError): validate_metadata(bad)
        legacy=deepcopy(m); legacy['version']=1
        with self.assertRaises(ValueError): validate_metadata(legacy)

    def test_confirmation_uses_fresh_batch_and_keeps_discovery_out_of_test(self):
        class Arena:
            def __init__(self): self.stage=0; self.tally=None
            def judgements(self):
                return [[(1, 0., [0.,0.,0.]), (2, 1., [1.,1.,1.])]]
            def judgement_weights(self): return [[1.,1.,1.]]
            def record_search_tally(self, asked, changed): self.tally=(asked,changed)
        arena=Arena(); calls=[]
        def once(net, arena, ranked, belief, **options):
            calls.append((ranked, options['worlds']))
            return [2] if len(calls)==1 else [1]
        evidence={}
        with patch.object(searched, '_search_once', side_effect=once):
            got=searched.search_with_value_head(None, arena, [[1,2,3]], [], worlds=8,
                    candidates=3, margin=2., hurried=False, confirm_worlds=64,
                    played_by='network', evidence=evidence)
        self.assertEqual(calls, [([[1,2,3]],8), ([[1,2]],64)])
        self.assertEqual(got,[1]); self.assertEqual(arena.tally,(1,0))
        self.assertEqual(evidence['confirmed'],[False])


if __name__ == '__main__': unittest.main()
