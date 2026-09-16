"""Expanded supervised proposals must remain executable by the student."""
import unittest
import numpy as np
from neural.teacher_actions import representable_moves
from neural.teacher_options import candidate_set
from neural.search_replay import action_contract


class TeacherActionTests(unittest.TestCase):
    def test_expansion_preserves_calls_and_riichi_but_excludes_aliased_extra_kan(self):
        legal = np.zeros((3, 78), bool)
        legal[0, [1, 76, 77]] = True
        legal[1, [70, 75]] = True
        legal[2, [1, 34, 35]] = True
        before = legal.copy()
        coverage = representable_moves(legal)
        np.testing.assert_array_equal(legal, before)
        self.assertFalse(coverage[0, 77])
        self.assertTrue(coverage[0, 76])
        self.assertTrue(coverage[1, 75])
        self.assertTrue(coverage[2, 34:36].all())
        expanded = candidate_set([1, 76, 77], coverage[0], 1, 5, np.random.default_rng(1))
        self.assertEqual(expanded, [1, 76])
        for row, choices in enumerate(coverage):
            for action in np.nonzero(choices)[0]:
                action_contract(legal[row:row+1], np.array([action], dtype=np.int64),
                                np.array([0], dtype=np.int64))

    def test_bad_shape_and_dtype_fail_before_proposing(self):
        for legal in (np.zeros(78, bool), np.zeros((1, 77), bool), np.zeros((1, 78))):
            with self.subTest(shape=legal.shape), self.assertRaises(ValueError):
                representable_moves(legal)
        self.assertEqual(representable_moves(np.zeros((0, 78), bool)).shape, (0, 78))
        self.assertFalse(representable_moves(np.zeros((1, 78), bool)).any())


if __name__ == '__main__': unittest.main()
