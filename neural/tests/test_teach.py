"""The search teaching the policy: what the recordings say becomes a
question in the network's own moves, and the leash decides how far the
answer may move.

    python -m unittest neural.tests.test_teach -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import riichi_py
import torch

from neural import contract, searched, sibling_head, teach, zoo
from neural.model import MORTAL_PLANES, PolicyValueNet
from neural.observe import Planes


def a_recording(folder: Path, games: int = 2, seed: int = 4) -> sibling_head.Recorded:
    """A real recording of a searched game, small enough for a test."""
    torch.manual_seed(3)
    net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
    served = contract.serve(net)
    recording = searched.Recording()
    searched.play(net, games, seed, 0, 3, 3, 0.0, pool=1, device="cpu", served=served, recording=recording)
    recording.save(folder, {"note": "test", "seed": seed, "games": games, "sure": 1.0})
    return sibling_head.Recorded(folder)


class MoveNamesTests(unittest.TestCase):
    def test_our_moves_are_named_in_mortals(self):
        table = teach.OURS_TO_MORTAL
        self.assertEqual(table[5], 5, "a discard is its tile")
        self.assertTrue(
            (table[zoo.RIICHI_DISCARD : zoo.TSUMO] == zoo.MORTAL_RIICHI).all(),
            "every riichi discard is the one reach, whose tile is a second question",
        )
        self.assertEqual(table[zoo.PON], zoo.MORTAL_PON)
        self.assertEqual(table[zoo.PASS], zoo.MORTAL_PASS)
        self.assertEqual(table[zoo.TSUMO], zoo.MORTAL_AGARI)
        self.assertEqual(table[zoo.RON], zoo.MORTAL_AGARI)
        self.assertEqual(len(table), riichi_py.ACTIONS)


class LessonTests(unittest.TestCase):
    def test_two_reaches_are_one_move_worth_the_better_of_them(self):
        """A reach throwing this tile and a reach throwing that one are one
        move to the network; the table asks which tile afterwards, so the
        move is worth what the better of them was worth."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recording = searched.Recording()
            root = Planes(np.array([0, 1], dtype=np.int64), np.zeros(1, dtype=np.uint16),
                          np.ones(1, dtype=np.float16))
            legal = np.zeros(riichi_py.ACTIONS, dtype=bool)
            legal[[3, zoo.RIICHI_DISCARD + 3, zoo.RIICHI_DISCARD + 8]] = True
            recording.add(
                root,
                [(3, 0.0, [0.0]), (zoo.RIICHI_DISCARD + 3, 0.2, [0.2]), (zoo.RIICHI_DISCARD + 8, 0.5, [0.5])],
                3, zoo.RIICHI_DISCARD + 8, 0, 0, 1, sure=0.4, legal=legal,
            )
            recording.save(folder, {"note": "test", "seed": 1, "games": 1})
            lesson = teach.Lesson(sibling_head.Recorded(Path(folder)), temperature=0.1, target="values")
        self.assertEqual(len(lesson), 1, "the row has two distinct moves to compare")
        moves = lesson.moves[0]
        self.assertEqual(list(moves), [3, -1, zoo.MORTAL_RIICHI],
                         "the weaker reach lost its group to the better one")
        np.testing.assert_allclose(lesson.worth[0][2], 0.5)
        self.assertAlmostEqual(float(lesson.targets[0].sum()), 1.0, places=5)
        self.assertGreater(lesson.targets[0][2], lesson.targets[0][0], "the better move is taught")
        self.assertEqual(lesson.targets[0][1], 0.0, "a move that lost its group teaches nothing")
        self.assertTrue(lesson.whole_mask, "the recording kept the table's mask")
        self.assertTrue(lesson.allowed[0][zoo.MORTAL_RIICHI] and lesson.allowed[0][3])

    def test_a_recording_without_the_tables_mask_is_refused(self):
        """The fusion's correction reads the mask itself, so a network
        asked under the compared moves alone answers a question the table
        never asked; such a recording is refused unless the caller says
        plainly that it wants the narrower question."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recorded = a_recording(Path(folder))
            recorded.legal = None
            with self.assertRaisesRegex(ValueError, "did not keep the table's mask"):
                teach.Lesson(recorded)
            lesson = teach.Lesson(recorded, without_mask=True)
            self.assertFalse(lesson.whole_mask)
            self.assertGreater(len(lesson), 0)

    def test_the_mask_the_table_had_is_the_question_asked(self):
        """Under the table's own mask the fusion answers differently from
        under the compared moves alone: the mask is an input to it, which
        is why the recording keeps it."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recorded = a_recording(Path(folder))
            whole = teach.Lesson(recorded)
            narrow = teach.Lesson(recorded, without_mask=True)
            narrow.allowed = np.zeros_like(narrow.allowed)
            for row in range(len(narrow.moves)):
                narrow.allowed[row, narrow.moves[row][narrow.valid[row]]] = True
            self.assertTrue((whole.allowed.sum(axis=1) >= narrow.allowed.sum(axis=1)).all())
            self.assertGreater(whole.allowed.sum(), narrow.allowed.sum(),
                               "the table allowed more than the search compared")

    def test_a_colder_lesson_is_a_sharper_one(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recorded = a_recording(Path(folder))
            cold = teach.Lesson(recorded, temperature=0.02, target="values")
            warm = teach.Lesson(recorded, temperature=1.0, target="values")
        self.assertGreater(len(cold), 0)
        self.assertGreater(cold.targets.max(axis=1).mean(), warm.targets.max(axis=1).mean())

    def test_the_lesson_is_the_move_the_search_made_under_its_margin(self):
        """The default lesson is the search's own decision, not the best
        average: four candidates over a handful of worlds have a best by
        luck, and the margin is what tells the two apart. Where the search
        kept the policy's move, that is what is taught."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recorded = a_recording(Path(folder))
            lesson = teach.Lesson(recorded)
            values = teach.Lesson(recorded, target="values")
        self.assertGreater(len(lesson), 0)
        taught = lesson.moves[lesson.rows][
            np.arange(len(lesson.rows)), lesson.targets[lesson.rows].argmax(axis=1)
        ]
        made = teach.OURS_TO_MORTAL[recorded.search[lesson.rows]]
        np.testing.assert_array_equal(taught, made, "the move taught is the move the search made")
        np.testing.assert_allclose(lesson.targets[lesson.rows].sum(axis=1), 1.0)
        self.assertTrue(
            ((lesson.targets[lesson.rows] == 0) | (lesson.targets[lesson.rows] == 1)).all(),
            "one move a row, not a spread over the candidates",
        )
        # Where the search kept the policy's move, that is the move
        # taught; this recording was searched without a margin, so it
        # kept fewer than it would in play.
        kept = ~lesson.changed[lesson.rows]
        self.assertTrue(kept.any() and lesson.changed[lesson.rows].any())
        policys = teach.OURS_TO_MORTAL[recorded.policy[lesson.rows]]
        np.testing.assert_array_equal(taught[kept], policys[kept])
        # The raw ordering can disagree with the search's choice: two of
        # our moves that are one of Mortal's group to the better of them.
        self.assertEqual(values.targets.shape, lesson.targets.shape)


class TeachingTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous)

    def test_the_policy_learns_what_the_rollouts_preferred(self):
        """Taught without a leash, the policy comes to pick the move the
        rollouts liked on the rows it was taught, and its worth on the
        search's own scale rises."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recorded = a_recording(Path(folder))
            lesson = teach.Lesson(recorded, temperature=0.05)
            torch.manual_seed(8)
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            rows = lesson.split()[0]
            before = teach.measure(net, lesson, rows, "cpu")
            teach.teach(net, lesson, epochs=6, lr=3e-3, batch=32, device="cpu", leash=0.0, hold="none")
            after = teach.measure(net, lesson, rows, "cpu")
        self.assertGreater(after["agrees_with_rollouts"], before["agrees_with_rollouts"])
        self.assertGreater(after["worth_of_its_pick"], before["worth_of_its_pick"])
        self.assertLessEqual(after["worth_of_its_pick"], after["worth_of_best"] + 1e-6)

    def test_a_hard_leash_keeps_the_policy_where_it_was(self):
        """The lesson pulls and the leash holds: at the same learning rate
        a heavy leash leaves the policy nearer where it started and further
        from what the rollouts wanted, which is the trade it is for.

        The rein is the learning rate as much as the weight. Gradients are
        clipped to a norm of one, so every step moves the weights by about
        the learning rate however heavy the leash, and the policy circles
        its starting point rather than sitting on it; the drift floor is
        the learning rate's, not the leash's.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recorded = a_recording(Path(folder))
            lesson = teach.Lesson(recorded, temperature=0.05)
            rows = lesson.split()[0]
            torch.manual_seed(8)
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            loose = teach.teach(net, lesson, epochs=3, lr=3e-4, batch=32, device="cpu",
                                leash=0.0, hold="none")
            learned = teach.measure(net, lesson, rows, "cpu")["agrees_with_rollouts"]
            torch.manual_seed(8)
            tight_net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            tight = teach.teach(tight_net, lesson, epochs=3, lr=3e-4, batch=32, device="cpu",
                                leash=30.0, hold="none")
            held_back = teach.measure(tight_net, lesson, rows, "cpu")["agrees_with_rollouts"]
        self.assertLess(tight[-1]["held_out"]["leash_kl"], loose[-1]["held_out"]["leash_kl"] / 4,
                        "a heavy leash holds the policy nearer where it started")
        self.assertLess(held_back, learned, "and it learns less of the lesson for it")

    def test_the_rows_that_change_something_can_be_made_to_count_for_more(self):
        """Nine decisions in ten teach the move the policy already makes.
        Emphasis counts the others for more, and the reading says how the
        two kinds went apart."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recorded = a_recording(Path(folder))
            plain = teach.Lesson(recorded)
            loud = teach.Lesson(recorded, emphasis=5.0)
            changed = plain.changed[plain.rows]
            self.assertTrue(changed.any() and (~changed).any())
            np.testing.assert_allclose(plain.weights[plain.rows], 1.0)
            np.testing.assert_allclose(loud.weights[plain.rows][changed], 5.0)
            np.testing.assert_allclose(loud.weights[plain.rows][~changed], 1.0)
            torch.manual_seed(8)
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            said = teach.measure(net, plain, plain.rows, "cpu")
        self.assertIn("where_it_changed", said)
        self.assertIn("where_it_agreed", said)
        self.assertEqual(said["where_it_changed"]["rows"] + said["where_it_agreed"]["rows"],
                         said["rows"])
        self.assertNotIn("where_it_changed", said["where_it_changed"], "read apart only once")

    def test_only_the_decisions_the_search_overrode_are_taught(self):
        """Where the search agreed with the policy, a one-hot on the
        policy's own move is a demand to sharpen rather than to keep, and
        sharpening is what erodes this lineage. Teaching only what the
        search overrode leaves the logits of a row it agreed on where the
        leash puts them; teaching every row moves them more.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recorded = a_recording(Path(folder))
            lesson = teach.Lesson(recorded)
            training = lesson.split()[0]
            agreed = training[~lesson.changed[training]]
            self.assertGreater(len(agreed), 0)
            planes, allowed, moves, valid, _targets, _weights = teach.lesson_batch(
                lesson, agreed[:16], "cpu"
            )
            torch.manual_seed(8)
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            with torch.no_grad():
                before = net.everything(planes, allowed)[0].float().clone()
            torch.manual_seed(8)
            only = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            teach.teach(only, lesson, epochs=2, lr=1e-3, batch=32, device="cpu", leash=0.0,
                        hold="none", rows="changed")
            torch.manual_seed(8)
            every = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            teach.teach(every, lesson, epochs=2, lr=1e-3, batch=32, device="cpu", leash=0.0,
                        hold="none", rows="all")
            with torch.no_grad():
                moved_only = (only.everything(planes, allowed)[0].float() - before)
                moved_every = (every.everything(planes, allowed)[0].float() - before)
            picked = moves.clamp(min=0)
            # How much the move the policy already made was pushed up on
            # the rows the search agreed with.
            pushed_only = float(moved_only.gather(1, picked)[valid][::3].mean())
            pushed_every = float(moved_every.gather(1, picked)[valid][::3].mean())
        self.assertGreater(pushed_every, pushed_only,
                           "teaching every row sharpens the moves it already made")

    def test_nothing_to_teach_is_refused_rather_than_pretended(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recorded = a_recording(Path(folder))
            lesson = teach.Lesson(recorded, temperature=0.05)
            net = PolicyValueNet(8, 1, planes=MORTAL_PLANES, actions=46)
            for parameter in net.parameters():
                parameter.requires_grad_(False)
            with self.assertRaisesRegex(ValueError, "nothing to teach"):
                teach.teach(net, lesson, epochs=1, device="cpu", leash=0.0, hold="none")


if __name__ == "__main__":
    unittest.main()
