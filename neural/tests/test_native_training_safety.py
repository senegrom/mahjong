"""Real native tests for strict actions and simulation/dealing RNG isolation."""
import unittest

import numpy as np
import riichi_py


class NativeTrainingSafetyTests(unittest.TestCase):
    def test_invalid_row_rejects_whole_batch_before_any_game_moves(self):
        arena = riichi_py.Arena(games=2, seed=81)
        actions = np.frombuffer(arena.teacher(), dtype=np.uint8).astype(int).tolist()
        observations = arena.observations()
        seats = arena.seats()
        before = [arena.debug(game) for game in range(2)]
        arena.mjai_all()
        bad = actions.copy()
        bad[1] = riichi_py.ACTIONS + 7
        with self.assertRaisesRegex(ValueError, "illegal action.*game 1"):
            arena.step(bad)
        self.assertEqual(arena.observations(), observations)
        self.assertEqual(arena.seats(), seats)
        self.assertEqual([arena.debug(game) for game in range(2)], before)
        self.assertEqual(arena.mjai_all(), [[], []])
        arena.step(actions)
        self.assertNotEqual(arena.observations(), observations)

    def test_invalid_call_is_rejected_without_consuming_claim_window(self):
        arena = riichi_py.Arena(games=1, seed=1)
        found = False
        for _ in range(8000):
            if arena.all_finished():
                break
            mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8).astype(bool)
            if mask[riichi_py.PASS]:
                found = True
                before = arena.debug(0)
                with self.assertRaises(ValueError):
                    arena.step([0])  # a discard is not an answer to a claim
                self.assertEqual(arena.debug(0), before)
                break
            arena.step(np.frombuffer(arena.teacher(), dtype=np.uint8).astype(int).tolist())
        self.assertTrue(found, "fixture must exercise a real claim window")

    def test_imagining_worlds_does_not_change_real_deals_or_outcomes(self):
        plain = riichi_py.Arena(games=2, seed=17)
        imagined = riichi_py.Arena(games=2, seed=17)
        beliefs = np.ones(2 * riichi_py.HANDS, dtype=np.float32).tobytes()
        hands = 0
        for _ in range(8000):
            self.assertEqual(plain.seats(), imagined.seats())
            self.assertEqual(plain.mjai_all(), imagined.mjai_all())
            if plain.all_finished():
                break
            self.assertEqual(plain.observations(), imagined.observations())
            for _extra in range(2):
                imagined.imagined_hands_bytes(beliefs)
            actions = np.frombuffer(plain.teacher(), dtype=np.uint8).astype(int).tolist()
            other = np.frombuffer(imagined.teacher(), dtype=np.uint8).astype(int).tolist()
            self.assertEqual(actions, other)
            plain.step(actions)
            imagined.step(other)
            hands += int(np.frombuffer(plain.hand_ended(), dtype=np.uint8).sum())
        self.assertTrue(plain.all_finished() and imagined.all_finished())
        self.assertGreaterEqual(hands, 16, "must compare subsequent hands, not just the initial deal")
        self.assertEqual(plain.final_scores(), imagined.final_scores())


if __name__ == "__main__":
    unittest.main()
