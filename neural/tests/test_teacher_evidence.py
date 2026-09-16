"""End-to-end evidence and boundary-label integration, not only helper maths."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from neural import placement, selfplay, worth
from neural.observe import Planes


class EvidenceTests(unittest.TestCase):
    def write_evidence(self, folder, repeats):
        # Every alternative beats the incumbent by the same amount in every
        # world. Only two independent games, regardless of repeated decisions.
        (folder / "meta.json").write_text("{}")
        edge = np.repeat([1., 3.], repeats)
        table = np.zeros((len(edge), 2, 8), dtype=np.float64)
        table[:, 1, :] = edge[:, None]
        for name, values in {
            'per_world': table, 'values': table.mean(axis=2),
            'game': np.repeat([7, 8], repeats),
            'world_weights': np.ones((len(edge), 8)),
        }.items():
            np.save(folder / (name + '.npy'), values)

    def test_worth_cluster_error_does_not_shrink_with_duplicated_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_evidence(root, 1)
            one = worth.measure(root)['by_margin']
            self.write_evidence(root, 20)
            many = worth.measure(root)['by_margin']
            self.assertEqual(one['independent_deals'], 2)
            self.assertEqual(many['rows'], 40)
            self.assertEqual(one['a_decision'], many['a_decision'])
            self.assertAlmostEqual(one['standard_error'], many['standard_error'])
            self.assertAlmostEqual(one['standard_error'], 1.)

    def test_boundary_only_fit_uses_real_labels_and_rejects_missing_or_bad_labels(self):
        batch = SimpleNamespace(
            seed=19, game_of=torch.tensor([0, 0, 1]),
            observations=Planes.from_follower([0, 1, 2, 3], [0, 1, 2], [1., 1., 1.]),
            decisions=3, placements=torch.tensor([.5, .5, -1.5]),
            returns=torch.tensor([1., 2., 0.]), games=2, hands=2,
            reward_version=1, final_scores=np.zeros((2, 4), dtype=np.int32),
            boundary=torch.tensor([True, False, True]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'round.pt'
            selfplay.save_round(batch, path)
            positions = placement.load_rounds([path], boundary_only=True)
            np.testing.assert_array_equal(positions.games, [19, 20])
            np.testing.assert_array_equal(positions.placements, [.5, -1.5])
            original = torch.load(path, weights_only=False)
            for bad in (np.array([1, 0, 1]), np.array([True])):
                torch.save({**original, 'boundary': bad}, path)
                with self.assertRaisesRegex(ValueError, 'boundary'):
                    placement.load_rounds([path], boundary_only=True)
            del original['boundary']
            torch.save(original, path)
            self.assertEqual(len(placement.load_rounds([path])), 3)
            with self.assertRaisesRegex(ValueError, 'boundary'):
                placement.load_rounds([path], boundary_only=True)


if __name__ == '__main__':
    unittest.main()
