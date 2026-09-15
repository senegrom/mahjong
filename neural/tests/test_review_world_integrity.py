"""Independent proposal evidence, reader layouts, and native/diagnostic parity."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import riichi_py

from neural import contract, worth
from neural.model import PolicyValueNet


class ReaderTests(unittest.TestCase):
    def test_reader_uses_its_own_planes_and_relative_hands_in_bounded_batches(self):
        torch.set_num_threads(1)
        counts = [3, 2]
        native = np.arange(2 * riichi_py.PLANES * 34, dtype=np.float32).reshape(2, riichi_py.PLANES, 34)
        hands = np.arange(5 * 12 * 34, dtype=np.float32).reshape(5, 12, 34)
        arena = SimpleNamespace(observations=lambda: native.tobytes(),
                                seats=lambda: bytes([0, 1]),
                                seat_players=lambda: bytes([2, 3, 0, 1, 3, 0, 1, 2]))
        seen_who = []
        def encode(who):
            seen_who.extend(who)
            return np.arange(len(who) + 1), np.zeros(len(who)), [g + p + 1 for g, p in who], []
        follower = SimpleNamespace(encode=encode)
        for planes in (riichi_py.PLANES, 1012):
            net = PolicyValueNet(8, 1, planes=planes, actions=78)
            calls = []
            def read(x, h):
                calls.append((x.clone(), h.clone()))
                return h[:, 0, 0]
            net.read_plausibility = read
            served = contract.serve(net)
            self.assertEqual(served.contract.describe()['reader_planes'], planes)
            with contract.following(arena, follower):
                said = contract.proposal_scores(served, arena, hands.tobytes(), counts, 'cpu', batch_size=2)
            self.assertEqual([len(x) for x, _ in calls], [2, 2, 1])
            np.testing.assert_array_equal(said, hands[:, 0, 0])
            torch.testing.assert_close(torch.cat([h for _, h in calls]), torch.from_numpy(hands))
            x = torch.cat([x for x, _ in calls])
            self.assertEqual(tuple(x.shape), (5, planes, 34))
            if planes == 1012:
                np.testing.assert_array_equal(x[:, 0, 0], [3, 3, 3, 2, 2])
            else:
                np.testing.assert_array_equal(x, native[[0, 0, 0, 1, 1]])
        self.assertTrue(seen_who)

    def test_absent_reader_is_an_explicit_uniform_capability(self):
        net = SimpleNamespace(kind='mortal', planes=1012, actions=46, everything=lambda *a: None)
        served = contract.serve(net)
        self.assertEqual(served.contract.describe()['world_reader'], 'uniform')
        np.testing.assert_array_equal(contract.proposal_scores(served, None, b'', [3], 'cpu'), [0, 0, 0])


class IndependentWorldTests(unittest.TestCase):
    def prepared(self):
        arena = riichi_py.Arena(games=1, seed=19, bot_places=[])
        legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(1, 78)
        ranked = [np.flatnonzero(legal[0])[:2].tolist()]
        arena.imagine([1.] * riichi_py.HANDS, worlds=2)
        return arena, ranked

    def test_both_rollout_paths_collapse_repeats_and_keep_their_mass(self):
        for network in (False, True):
            arena, ranked = self.prepared()
            kept, weights = [[0] * 16 + [1] * 16], [[1 / 32] * 32]
            if network:
                arena.lookahead_begin(ranked, kept, weights, candidates=2)
                self.assertEqual(arena.lookahead_slots(), [4])
                for _ in range(8000):
                    _, masks, count = arena.lookahead_owed()
                    if not count: break
                    actions = np.frombuffer(masks, dtype=np.uint8).reshape(count, 78).argmax(1)
                    arena.lookahead_apply(actions.tolist())
                _, counts, _, _ = arena.lookahead_leaves()
            else:
                _, counts, _, _ = arena.leaves_from(ranked, kept, weights, candidates=2)
            self.assertEqual(counts, [4], 'two candidates times TWO independent proposals')
            chosen = arena.decide([0.] * 4, 0., ranked)
            self.assertEqual(chosen, [ranked[0][0]], 'two proposals cannot clear the evidence minimum')
            np.testing.assert_allclose(arena.judgement_weights(), [[.5, .5]])

    def test_invalid_weights_do_not_consume_the_proposal_pool(self):
        arena, ranked = self.prepared()
        with self.assertRaises(ValueError):
            arena.lookahead_begin(ranked, [[0, 1]], [[1., float('nan')]])
        arena.lookahead_begin(ranked, [[0, 1]], [[.5, .5]])
        self.assertEqual(arena.lookahead_slots(), [4])

    def test_python_margin_matches_the_shared_native_fixtures(self):
        path = Path(__file__).resolve().parents[2] / 'engine/riichi-core/tests/fixtures/search_margin.tsv'
        for row in path.read_text().splitlines():
            name, margin, differences, weights, expected = row.split('\t')
            diff, mass = np.array(list(map(float, differences.split(',')))), np.array(list(map(float, weights.split(','))))
            with self.subTest(name=name):
                got = worth.picked_by_margin(np.stack([np.zeros(len(diff)), diff]),
                                              np.array([True, True]), np.arange(len(diff)), float(margin), mass)
                self.assertEqual(got, int(expected))


if __name__ == '__main__': unittest.main()

class RecordedWorldTests(unittest.TestCase):
    def test_nonuniform_independent_mass_survives_recording_and_loading(self):
        import tempfile
        from neural.searched import Recording
        from neural.sibling_head import Recorded
        from neural.observe import Planes
        from neural.recordings import resolve_recording, validate_snapshot
        root = Planes.from_follower([0, 1], [0], [1.])
        record = Recording()
        record.add(root, [(0, .75, [0., 1.]), (1, 2.25, [0., 3.])],
                   0, 0, 0, 0, 1, weights=[.25, .75])
        with tempfile.TemporaryDirectory() as tmp:
            record.save(tmp, {})
            self.assertEqual(validate_snapshot(resolve_recording(tmp))['recording_format'], 3)
            loaded = Recorded(Path(tmp))
            np.testing.assert_array_equal(loaded.world_weights, [[.25, .75]])
            np.testing.assert_allclose(worth.weighted_means(loaded.per_world[0], loaded.world_weights[0]), loaded.values[0])
            self.assertTrue(np.isfinite(loaded.precision()).all())
