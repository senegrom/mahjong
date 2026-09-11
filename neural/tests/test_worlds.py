"""Whether the search's choice of worlds estimates what it claims to.

The old rule kept the highest-scoring worlds and renormalised them. These
tests are the ones it would have failed.

    python -m unittest neural.tests.test_worlds -v
"""

from __future__ import annotations

import unittest

import numpy as np

from neural.worlds import resample


def top_k(scores, keep):
    """What was done before, for comparison: the highest scores, weighted
    by those scores and renormalised."""
    order = np.argsort(-np.asarray(scores))[:keep]
    top = np.asarray(scores)[order]
    weight = np.exp(top - top.max())
    return list(order), list(weight / weight.sum())


class ItEstimatesTheRightThing(unittest.TestCase):
    def test_a_known_expectation_is_recovered(self):
        """Proposals carry a value; the weights say how much each counts.
        The resampled average must find the weighted mean."""
        rng = np.random.default_rng(11)
        count = 400
        scores = rng.normal(size=count)
        value = rng.normal(size=count)
        weight = np.exp(scores - scores.max())
        wanted = float((weight * value).sum() / weight.sum())

        got = []
        for trial in range(400):
            chosen = resample(scores, keep=16, rng=np.random.default_rng(trial))
            got.append(float(np.mean(value[chosen.kept])))
        # The estimator is unbiased, so the average over many draws lands
        # on the weighted mean; one draw of sixteen does not have to.
        self.assertAlmostEqual(float(np.mean(got)), wanted, delta=0.05)

    def test_uniform_weights_give_the_unweighted_estimator(self):
        """A reader that has learned nothing must not bias the search."""
        rng = np.random.default_rng(3)
        value = rng.normal(size=200)
        got = []
        for trial in range(400):
            chosen = resample(np.zeros(200), keep=16, rng=np.random.default_rng(trial))
            got.append(float(np.mean(value[chosen.kept])))
        self.assertAlmostEqual(float(np.mean(got)), float(value.mean()), delta=0.05)

    def test_efficiency_reports_weight_concentration(self):
        flat = resample(np.zeros(100), keep=8, rng=np.random.default_rng(0))
        self.assertAlmostEqual(flat.efficiency, 1.0, places=6)

        spiked = np.full(100, -50.0)
        spiked[7] = 0.0
        concentrated = resample(spiked, keep=8, rng=np.random.default_rng(0))
        self.assertLess(concentrated.efficiency, 0.02)
        self.assertEqual(set(concentrated.kept), {7}, "all the mass was on one world")


class TheRareWorldSurvives(unittest.TestCase):
    """The failure that matters in this game: the unlikely hand that
    decides whether a discard was a mistake."""

    def test_a_rare_world_is_reachable_where_top_k_removes_it(self):
        # Ninety-nine worlds the reader likes slightly, and one it likes
        # much less, which top-k therefore never keeps.
        scores = np.zeros(100)
        scores[0] = -3.0  # about five per cent of the mass of one of the rest

        kept, _weights = top_k(scores, keep=8)
        self.assertNotIn(0, kept, "top-k is supposed to drop it; that is the point")

        seen = any(
            0 in resample(scores, keep=8, rng=np.random.default_rng(trial)).kept
            for trial in range(600)
        )
        self.assertTrue(seen, "the rare world was never reachable")

    def test_it_is_rare_rather_than_common(self):
        """Reachable is not the same as over-weighted."""
        scores = np.zeros(100)
        scores[0] = -3.0
        drawn = sum(
            resample(scores, keep=8, rng=np.random.default_rng(trial)).kept.count(0)
            for trial in range(2000)
        )
        share = drawn / (2000 * 8)
        expected = np.exp(-3.0) / (99 + np.exp(-3.0))
        self.assertAlmostEqual(share, expected, delta=0.01)


class DegenerateInputFailsSafely(unittest.TestCase):
    def test_no_proposals(self):
        chosen = resample(np.zeros(0), keep=8, rng=np.random.default_rng(0))
        self.assertEqual(chosen.kept, [])
        self.assertEqual(chosen.efficiency, 0.0)

    def test_weights_that_are_not_numbers(self):
        scores = np.array([np.nan, np.inf, -np.inf, 0.0])
        chosen = resample(scores, keep=4, rng=np.random.default_rng(0))
        self.assertEqual(len(chosen.kept), 4)
        self.assertTrue(all(0 <= index < 4 for index in chosen.kept))

    def test_every_weight_impossible(self):
        chosen = resample(np.full(10, -np.inf), keep=4, rng=np.random.default_rng(0))
        self.assertEqual(len(chosen.kept), 4)
        self.assertEqual(chosen.efficiency, 0.0, "it must say the weights were no use")

    def test_the_weights_returned_are_uniform(self):
        """Resampling has already spent the importance weights; applying
        them again would correct for the same thing twice."""
        rng = np.random.default_rng(5)
        chosen = resample(rng.normal(size=50), keep=10, rng=rng)
        self.assertEqual(chosen.weights, [0.1] * 10)



class UnsupportedCheckpointsAreRefused(unittest.TestCase):
    """The mismatch that used to pass in silence."""

    def test_a_mortal_plane_network_is_refused(self):
        from neural.worlds import assert_searchable

        class Fusion:
            kind = "mortal"

            def planes(self):
                return 1012

        with self.assertRaises(SystemExit) as caught:
            assert_searchable(Fusion(), "leashed-run/latest.pt")
        self.assertIn("cannot be searched here", str(caught.exception))
        self.assertIn("leashed-run/latest.pt", str(caught.exception))

    def test_an_engine_plane_network_is_allowed(self):
        import riichi_py

        from neural.worlds import assert_searchable

        class Ours:
            kind = "engine"

            def planes(self):
                return riichi_py.PLANES

        assert_searchable(Ours())  # does not raise


if __name__ == "__main__":
    unittest.main()
