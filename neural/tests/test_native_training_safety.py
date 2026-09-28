"""Real native tests for strict actions, refused inputs and reported hand endings."""
import json
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

    def test_wrong_shaped_inputs_are_refused_before_anything_moves(self):
        # Each of these used to panic inside the extension, which Python
        # sees as a PanicException, a BaseException that `except Exception`
        # does not catch.
        with self.assertRaisesRegex(ValueError, 'bot place 4'):
            riichi_py.Arena(games=1, seed=3, bot_places=[1, 4])
        arena = riichi_py.Arena(games=2, seed=17)
        # One game's events are only reached through mjai_all, which cannot
        # name a game that does not exist.
        self.assertFalse(hasattr(arena, 'mjai'))
        self.assertEqual(len(arena.mjai_all()), 2)

    def test_every_hand_the_native_bots_finish_is_reported(self):
        # Four heuristic players play whole games while the arena is made,
        # before any step: the endings are reported until the first step,
        # every one of them, and not as a single flag.
        arena = riichi_py.Arena(games=3, seed=5, bot_places=[0, 1, 2, 3])
        self.assertTrue(arena.all_finished())
        ended = np.frombuffer(arena.hand_ended(), dtype=np.uint8)
        np.testing.assert_array_equal(ended, np.asarray(arena.hands_done()))
        self.assertTrue((ended >= 8).all(), ended)
        # What the first of them moved, by person, as the game's own log
        # has it: the scores after its last win or draw less those it was
        # dealt with.
        results = np.frombuffer(arena.hand_result(), dtype=np.int32).reshape(3, 4)
        for game, lines in enumerate(arena.mjai_all()):
            events = [json.loads(line) for line in lines]
            first = events.index(next(e for e in events if e['type'] == 'end_kyoku'))
            dealt = next(e for e in events if e['type'] == 'start_kyoku')['scores']
            after = [e for e in events[:first] if e['type'] in ('hora', 'ryukyoku')][-1]['scores']
            np.testing.assert_array_equal(results[game], np.subtract(after, dealt))
        arena.step([0, 0, 0])
        self.assertFalse(np.frombuffer(arena.hand_ended(), dtype=np.uint8).any())


if __name__ == "__main__":
    unittest.main()
