"""Invalid search batches cannot mutate a prefix of games or consume cursors."""
import unittest

import numpy as np
import riichi_py


class AtomicLookaheadTests(unittest.TestCase):
    def prepared(self):
        games = 2
        arena = riichi_py.Arena(games=games, seed=19, bot_places=[])
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, riichi_py.ACTIONS)
        ranked = [np.flatnonzero(row)[:2].tolist() for row in mask]
        arena.imagine([1.] * (games * riichi_py.HANDS), worlds=2)
        arena.lookahead_begin(ranked, [[0, 1]] * games, [[.5, .5]] * games, candidates=2, depth=1)
        return arena

    def test_short_long_and_illegal_batches_are_atomic_across_games(self):
        arena, control = self.prepared(), self.prepared()
        before = arena.lookahead_owed()
        self.assertGreater(before[2], 1)
        masks = np.frombuffer(before[1], dtype=np.uint8).reshape(-1, riichi_py.ACTIONS)
        good = masks.argmax(1).tolist()
        # Consume initial history on both copies, so any accidental cursor
        # advancement during a failed application is visible too.
        initial = arena.lookahead_owed_mjai()
        self.assertEqual(initial, control.lookahead_owed_mjai())
        invalid = good.copy()
        invalid[-1] = int(np.flatnonzero(~masks[-1].astype(bool))[0])
        for bad in (good[:-1], good + [good[0]], invalid, good[:-1] + [riichi_py.ACTIONS]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    arena.lookahead_apply(bad)
                self.assertEqual(arena.lookahead_owed(), before)
                events = arena.lookahead_owed_mjai()
                self.assertEqual(events[:4], initial[:4])
                self.assertTrue(all(not lines for lines in events[4]))
        arena.lookahead_apply(good)
        control.lookahead_apply(good)
        self.assertEqual(arena.lookahead_owed(), control.lookahead_owed())
        self.assertEqual(arena.lookahead_owed_mjai(), control.lookahead_owed_mjai())

    def test_unfinished_worlds_are_not_published_or_consumed(self):
        arena = self.prepared()
        before = arena.lookahead_owed()
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            arena.lookahead_leaves()
        self.assertEqual(arena.lookahead_owed(), before)


if __name__ == '__main__':
    unittest.main()
