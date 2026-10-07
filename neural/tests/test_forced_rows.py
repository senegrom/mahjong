"""A seated player asked to (`--skip-forced`) is not asked about a row with
a single move open in Mortal's moves: the best of one move is that move
whatever the network says, so every choice stays what asking would have
made it, and the forwards shrink by the rows that had no choice to make.
The rows still asked are then scored in a smaller batch, which on the card
can change their bits, so it is a switch, off unless a trainer is told,
and off it asks every row in the very batches it always did."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import riichi_py
import torch

from neural import combined, mortal_model, ppo_loop, train_combined, train_mortal, zoo
from neural.observe import Views
from neural.tests import test_mixed_tables
from neural.tests.test_bfloat16_players import checkpoints
from neural.tests.test_step_planes import deciding


def choose_by_asking_every_row(ask, views, rows, players, legal, stats=None):
    """`zoo.choose_in_mortal_space` as it was before it left the rows with
    one move open unasked, kept as the reference."""
    who = list(zip(np.asarray(rows).tolist(), np.asarray(players).tolist()))
    legal = np.atleast_2d(legal)
    allowed = zoo.translatable(legal)
    orphan = ~allowed.any(axis=1)
    allowed[orphan, zoo.MORTAL_PASS] = True
    values, own = ask(who, False, allowed)
    ranked = np.where(allowed, values, -np.inf)
    best = ranked.argmax(axis=1)
    if stats is not None:
        stats["orphans"] = stats.get("orphans", 0) + int(orphan.sum())
        stats["fallbacks"] = stats.get("fallbacks", 0) + int(
            ((np.where(own, values, -np.inf).argmax(axis=1) != best) & ~orphan).sum()
        )
    choice = zoo.first_meaning(best, legal)
    choice = np.where(orphan | (choice < 0), legal.argmax(axis=1), choice)
    second = np.nonzero((best == zoo.MORTAL_RIICHI) & ~orphan)[0]
    if len(second):
        follower = views.observer.follower
        for i in second:
            game, player = who[i]
            follower.tell(game, player, json.dumps({"type": "reach", "actor": player}))
        tiles = legal[second, zoo.RIICHI_DISCARD:zoo.TSUMO]
        allowed_after = np.zeros((len(second), zoo.MORTAL_ACTIONS), dtype=bool)
        allowed_after[:, :34] = tiles
        after, _own_after = ask([who[i] for i in second], True, allowed_after)
        ranked_tiles = np.where(tiles, after[:, :riichi_py.POSITIONS], -np.inf)
        choice[second] = zoo.RIICHI_DISCARD + ranked_tiles.argmax(axis=1)
    return choice.astype(np.int64)


class Telling:
    """Views whose follower only writes down what it is told."""

    def __init__(self):
        self.told = []
        self.observer = SimpleNamespace(follower=self)

    def tell(self, game, player, event):
        self.told.append((game, player, json.loads(event)["type"]))


def row_values(who, fresh):
    """Values a row at a time, the same whatever else is asked with it."""
    values = np.empty((len(who), zoo.MORTAL_ACTIONS), dtype=np.float32)
    own = np.empty((len(who), zoo.MORTAL_ACTIONS), dtype=bool)
    for row, (game, player) in enumerate(who):
        rng = np.random.default_rng([game, player, int(fresh)])
        values[row] = rng.normal(size=zoo.MORTAL_ACTIONS)
        own[row] = rng.random(zoo.MORTAL_ACTIONS) < 0.3
    return values, own


def random_rows(rng):
    """A step's rows of every kind that has one move open or several:
    discards, reaches by one tile or several, calls, wins, passes, and the
    orphan with nothing Mortal can name."""
    n = int(rng.integers(1, 12))
    legal = np.zeros((n, riichi_py.ACTIONS), dtype=bool)
    for row in range(n):
        kind = rng.integers(0, 5)
        if kind == 0:  # a declared riichi's draw: throw it
            legal[row, rng.integers(0, 34)] = True
        elif kind == 1:  # a reach, by one tile or several
            tiles = rng.choice(34, size=int(rng.integers(1, 4)), replace=False)
            legal[row, tiles] = True
            legal[row, zoo.RIICHI_DISCARD + tiles] = True
        elif kind == 2:  # a call or a pass
            legal[row, rng.choice([68, 69, 71, 72, 73, 74, 75])] = True
            legal[row, zoo.PASS] = True
        elif kind == 3:  # two kans, or a win either way: Mortal's one move
            legal[row, [[75, 76], [68, 69], [zoo.PASS]][rng.integers(0, 3)]] = True
        # kind 4: nothing at all, the orphan
    return rng.integers(0, 50, size=n), rng.integers(0, 4, size=n), legal


class ForcedRowsTests(unittest.TestCase):
    def test_by_default_every_row_is_asked_in_the_same_batches_as_before(self):
        """Without the switch nothing changes: the same questions, about
        the same rows in the same order with the same moves allowed, so the
        values every row is scored with are what they were, and so are the
        choices, the reaches told and the counts."""
        rng = np.random.default_rng(23)
        questions = 0
        for trial in range(300):
            rows, players, legal = random_rows(rng)
            asked = {"old": [], "new": []}

            def asking(which):
                def ask(who, fresh, allowed):
                    asked[which].append((list(who), fresh, allowed.copy()))
                    return row_values(who, fresh)
                return ask

            old_views, new_views = Telling(), Telling()
            old_stats, new_stats = {}, {}
            want = choose_by_asking_every_row(asking("old"), old_views, rows, players, legal, old_stats)
            got = zoo.choose_in_mortal_space(asking("new"), new_views, rows, players, legal, new_stats)
            np.testing.assert_array_equal(got, want, err_msg=f"trial {trial}")
            self.assertEqual(new_views.told, old_views.told)
            self.assertEqual(new_stats, old_stats)
            self.assertEqual(len(asked["new"]), len(asked["old"]))
            for (who, fresh, allowed), (want_who, want_fresh, want_allowed) in zip(asked["new"], asked["old"]):
                self.assertEqual((who, fresh), (want_who, want_fresh))
                np.testing.assert_array_equal(allowed, want_allowed)
            questions += len(asked["new"])
        self.assertGreater(questions, 300)

    def test_the_same_choices_without_asking_about_rows_with_one_move(self):
        rng = np.random.default_rng(17)
        asked_rows = forced = 0
        for trial in range(300):
            rows, players, legal = random_rows(rng)
            asked = []

            def ask(who, fresh, allowed):
                asked.append(allowed.copy())
                return row_values(who, fresh)

            old_views, new_views = Telling(), Telling()
            old_stats, new_stats = {}, {}
            want = choose_by_asking_every_row(lambda who, fresh, allowed: row_values(who, fresh),
                                              old_views, rows, players, legal, old_stats)
            got = zoo.choose_in_mortal_space(ask, new_views, rows, players, legal, new_stats,
                                             skip_forced=True)
            np.testing.assert_array_equal(got, want, err_msg=f"trial {trial}")
            self.assertEqual(new_views.told, old_views.told)
            self.assertEqual(new_stats["orphans"], old_stats["orphans"])
            # Only rows with a choice to make were asked, at either step.
            for allowed in asked:
                self.assertTrue((allowed.sum(axis=1) > 1).all())
                asked_rows += len(allowed)
            forced += int((zoo.translatable(legal).sum(axis=1) <= 1).sum())
        self.assertGreater(asked_rows, 300)
        self.assertGreater(forced, 300)

    def test_a_step_of_rows_with_one_move_each_asks_nothing(self):
        legal = np.zeros((3, riichi_py.ACTIONS), dtype=bool)
        legal[0, 5] = True
        legal[1, [68, 69]] = True
        legal[2, [75, 76, 77]] = True

        def ask(who, fresh, allowed):
            raise AssertionError("asked about a row with one move")

        choice = zoo.choose_in_mortal_space(ask, Telling(), np.arange(3), np.zeros(3), legal,
                                            skip_forced=True)
        self.assertEqual(choice.tolist(), [5, 68, 75])


class SeatedMortalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_a_seated_mortal_plays_whole_tables_the_same(self):
        """Two tables dealt alike, one played by asking every row and one
        the new way, stay in step move for move. A small Mortal is asked
        through its own encoding and forward, and its values are steered
        a game at a time, the same for both tables, towards the engine's
        heuristic move and towards a reach whenever one is allowed: a
        random network alone hardly ever has a ready hand, and here the
        draws a declared reach must throw, and the reaches only one tile
        keeps ready, are many."""
        torch.manual_seed(3)
        net = mortal_model.build(8, 1).eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        with patch.object(zoo.mortal_model, "load", return_value=net):
            player = zoo.MortalPlayer("unused", "cpu")
        games = 8
        tables = [riichi_py.Arena(games=games, seed=20261005) for _ in range(2)]
        views = [Views(table, games) for table in tables]
        counts = {"rows": 0, "asked": 0, "reaches": 0, "tiles asked": 0}
        old_stats, new_stats = {}, {}
        # Mortal's move for each of ours.
        mortal_of = np.isfinite(zoo.PRIORITY).argmax(axis=0)

        def steered(table, teacher):
            def ask(who, fresh, allowed):
                values, own = player._ask(views[table], who, fresh, allowed)
                for row, (game, _player) in enumerate(who):
                    move = int(teacher[game])
                    if fresh and zoo.RIICHI_DISCARD <= move < zoo.TSUMO:
                        values[row, move - zoo.RIICHI_DISCARD] += 100.0
                    elif not fresh and move != 0xFF:
                        values[row, mortal_of[move]] += 100.0
                        values[row, zoo.MORTAL_RIICHI] += 200.0 * allowed[row, zoo.MORTAL_RIICHI]
                if table == 1:
                    counts["tiles asked" if fresh else "asked"] += len(who)
                return values, own
            return ask

        for _step in range(1500):
            if tables[0].all_finished():
                break
            for view in views:
                view.advance()
            live, players = deciding(tables[0], games)
            np.testing.assert_array_equal(deciding(tables[1], games)[0], live)
            mask = np.frombuffer(tables[0].legal_mask(), dtype=np.uint8).reshape(games, -1)
            legal = mask.astype(bool)[live]
            teacher = np.frombuffer(tables[0].teacher(), dtype=np.uint8)
            for view in views:
                view.prepare(live, players)
            want = choose_by_asking_every_row(steered(0, teacher), views[0], live, players,
                                              legal, old_stats)
            got = zoo.choose_in_mortal_space(steered(1, teacher), views[1], live, players,
                                             legal, new_stats, skip_forced=True)
            np.testing.assert_array_equal(got, want)
            counts["rows"] += len(live)
            counts["reaches"] += int(((got >= zoo.RIICHI_DISCARD) & (got < zoo.TSUMO)).sum())
            choice = np.zeros(games, dtype=np.int64)
            choice[live] = got
            for table in tables:
                table.step(choice.tolist())
        np.testing.assert_array_equal(np.frombuffer(tables[0].seats(), dtype=np.uint8),
                                      np.frombuffer(tables[1].seats(), dtype=np.uint8))
        # A good share of the rows went unasked, among them reaches whose
        # tile was the only one, and reaches with a choice were still asked.
        self.assertGreater(counts["rows"] - counts["asked"], counts["rows"] // 20, counts)
        self.assertGreater(counts["reaches"], counts["tiles asked"], counts)
        self.assertGreater(counts["tiles asked"], 0, counts)
        self.assertEqual(new_stats.get("orphans", 0), old_stats.get("orphans", 0))
        self.assertLessEqual(new_stats.get("fallbacks", 0), old_stats.get("fallbacks", 0))


class SwitchTests(unittest.TestCase):
    def test_seated_players_ask_every_row_unless_the_trainer_says_otherwise(self):
        """Every player that chooses in Mortal's moves asks every row unless
        the trainer seating it was given `--skip-forced`; a trainer that has
        no such option, as `train.py`, seats them asking every row too."""
        legal = np.zeros((1, riichi_py.ACTIONS), dtype=bool)
        legal[0, 0] = True
        with tempfile.TemporaryDirectory() as folder:
            paths = checkpoints(Path(folder))
            names = ("mortal", "reheaded", "fused")
            for options in ({"skip_forced": False}, {"skip_forced": True}, {}):
                args = SimpleNamespace(opponents=[*(paths[name] for name in names), "club"],
                                       compile=False, **options)
                seated = ppo_loop.load_others(args, "cpu")
                self.assertEqual(seated[-1].kind, "club")
                for name, player in zip(names, seated):
                    with self.subTest(player=name, **options):
                        passed = {}

                        def choose(*_args, **kwargs):
                            passed.update(kwargs)
                            return np.zeros(1, dtype=np.int64)

                        with patch.object(zoo, "choose_in_mortal_space", side_effect=choose):
                            player.choose(None, np.zeros(1), np.zeros(1), legal)
                        self.assertIs(passed["skip_forced"], options.get("skip_forced", False))
        # Players nobody seated, as a duel or the measurement loads them.
        self.assertFalse(zoo.MortalPlayer.skip_forced)
        self.assertFalse(zoo.MortalSpacePlayer.skip_forced)
        self.assertFalse(combined.Combined.skip_forced)

    def test_the_trainers_and_the_cloud_leave_it_off_unless_asked(self):
        for module in (train_combined, train_mortal):
            with self.subTest(trainer=module.__name__):
                with patch.object(sys, "argv", ["trainer"]):
                    self.assertFalse(module.parse_args().skip_forced)
                with patch.object(sys, "argv", ["trainer", "--skip-forced"]):
                    self.assertTrue(module.parse_args().skip_forced)
        launch = test_mixed_tables.CloudLaunchTests.launch
        for trainer in ("train_mortal", "train_combined"):
            with self.subTest(cloud=trainer):
                self.assertNotIn("--skip-forced", launch(self, trainer))
                self.assertIn("--skip-forced", launch(self, trainer, skip_forced=True))


if __name__ == "__main__":
    unittest.main()
