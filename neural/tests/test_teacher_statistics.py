import unittest
import numpy as np
from neural.teacher_statistics import clustered_mean, validation_summary


class ClusterTests(unittest.TestCase):
    def test_duplicates_within_games_do_not_invent_precision(self):
        a = clustered_mean([1., -1., 2., -2.], [0, 1, 2, 3])
        b = clustered_mean(np.repeat([1., -1., 2., -2.], 100),
                           np.repeat([0, 1, 2, 3], 100))
        self.assertAlmostEqual(a['standard_error'], b['standard_error'])
        self.assertEqual(b['independent_deals'], 4)

    def test_one_game_is_not_zero_uncertainty(self):
        result = clustered_mean(np.ones(100), np.zeros(100, dtype=np.int64))
        self.assertIsNone(result['standard_error'])
        self.assertIsNone(result['standard_errors'])

    def test_unequal_clusters_preserve_decision_weighting(self):
        got = clustered_mean([1., 1., -1.], [2, 2, 3])
        self.assertAlmostEqual(got['mean'], 1 / 3)
        self.assertAlmostEqual(got['standard_error'], 8 / 9)

    def test_singleton_clusters_match_sample_mean_error(self):
        x = np.array([1., 0., 2., -2., 3.])
        got = clustered_mean(x, np.arange(len(x)))
        self.assertAlmostEqual(got['standard_error'], x.std(ddof=1) / np.sqrt(len(x)))

    def test_uint64_seed_identity_and_reordering(self):
        seeds = np.array([2**64 - 1, 2**64 - 2, 2**64 - 1], dtype=np.uint64)
        values = np.array([1., -1., 0.])
        self.assertEqual(clustered_mean(values, seeds), clustered_mean(values[::-1], seeds[::-1]))

    def test_invalid_evidence_fails(self):
        for x, g in (([np.nan], [0]), ([1], [-1]), ([1], [.5]), ([1, 2], [1])):
            with self.assertRaises(ValueError):
                clustered_mean(x, g)
        with self.assertRaises(ValueError):
            validation_summary([1, 2], [0, 1], total_deals=1)

    def test_empty_evidence_and_coverage(self):
        empty = validation_summary([], np.array([], dtype=np.uint64), total_deals=10)
        self.assertEqual(empty['rows'], 0)
        self.assertIsNone(empty['a_decision'])
        got = validation_summary([1., 2.], [0, 1], total_deals=4)
        self.assertEqual(got['a_deal_so_far'], .75)


if __name__ == '__main__':
    unittest.main()
