"""Others seated by player: one table can hold the learner, published Mortal
and an old checkpoint at once, in self-play, in both trainers' checks and in
both cloud launches."""
from contextlib import ExitStack, redirect_stdout
from functools import partial
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from neural import population
from neural.checkpoints import atomic_save
from neural.cloud_requests import validate_cloud_request
from neural.training_safety import validate_training_options


class MixedTablesTests(unittest.TestCase):
    def test_the_learner_always_keeps_a_seat(self):
        owner = population.mixed_tables(20_000, 0.9, [1.0, 1.0], np.random.default_rng(3))
        self.assertEqual(owner.shape, (20_000, 4))
        self.assertTrue(((owner < 0).sum(axis=1) >= 1).all())

    def test_each_player_is_somebody_else_at_about_the_share(self):
        owner = population.mixed_tables(20_000, 0.5, [1.0, 1.0], np.random.default_rng(5))
        # A table of four others gives one player back, so a little under half.
        self.assertAlmostEqual(float((owner >= 0).mean()), 0.5 - 0.25 / 16, delta=0.01)

    def test_one_table_can_hold_every_member_and_the_learner(self):
        owner = population.mixed_tables(5_000, 0.5, [1.0, 1.0], np.random.default_rng(11))
        both = ((owner == 0).any(axis=1) & (owner == 1).any(axis=1) & (owner < 0).any(axis=1))
        self.assertGreater(int(both.sum()), 500)

    def test_members_are_drawn_by_weight(self):
        owner = population.mixed_tables(20_000, 0.5, [3.0, 1.0], np.random.default_rng(13))
        seated = owner[owner >= 0]
        self.assertAlmostEqual(float((seated == 0).mean()), 0.75, delta=0.02)

    def test_bad_shares_and_weights_are_refused(self):
        rng = np.random.default_rng(0)
        for share in (0.0, 1.0, -0.1, float("nan")):
            with self.subTest(share=share), self.assertRaises(ValueError):
                population.mixed_tables(4, share, [1.0], rng)
        for weights in ([], [0.0, 0.0], [-1.0, 2.0]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                population.mixed_tables(4, 0.5, weights, rng)


class MatchupsOfMixedTablesTests(unittest.TestCase):
    def test_a_member_counts_every_game_it_sat_in(self):
        members = [population.Member("mortal", "reference", "published", 1.0),
                   population.Member("gen31", "reference", "old", 1.0)]
        owner = np.array([[0, 1, -1, -1], [0, -1, -1, -1], [-1, -1, -1, -1]])
        rows = population.matchups(owner, np.array([2.0, 3.0, 2.5]), members)
        by_name = {row["name"]: row for row in rows}
        self.assertEqual(by_name["mortal"]["games"], 2)
        self.assertEqual(by_name["mortal"]["placement"], 2.5)
        self.assertEqual(by_name["gen31"]["games"], 1)
        self.assertEqual(by_name["itself"]["games"], 1)


class Counting:
    """The collector's first-legal stand-in, counting the decisions it is asked for."""

    kind = "engine"

    def __init__(self):
        self.asked = 0

    def eval(self):
        return self

    def choose(self, views, rows, players, legal):
        self.asked += len(rows)
        return np.asarray(legal, dtype=bool).argmax(axis=1).astype(np.int64)


class SelfPlayOnMixedTablesTests(unittest.TestCase):
    def test_several_others_play_and_the_report_names_them(self):
        from neural import selfplay
        from neural.tests.test_selfplay_contract import FirstLegal

        others = [Counting(), Counting()]
        roster = population.Population(members=[
            population.Member("mortal", "reference", "a stand-in", 1.0),
            population.Member("gen31", "reference", "a stand-in", 1.0),
        ])
        batch = selfplay.play(FirstLegal(), games=16, seed=7, device="cpu",
                              opponents=others, seat_share=0.5, population=roster)
        self.assertEqual(batch.seated.shape, (16, 4))
        self.assertTrue(((batch.seated < 0).sum(axis=1) >= 1).all())
        self.assertTrue(all(other.asked > 0 for other in others))
        names = {row["name"] for row in batch.matchups}
        self.assertTrue({"mortal", "gen31"} <= names, batch.matchups)

    def test_by_game_and_by_player_together_are_refused(self):
        from neural import selfplay
        from neural.tests.test_selfplay_contract import FirstLegal

        with self.assertRaises(ValueError):
            selfplay.play(FirstLegal(), games=2, seed=1, device="cpu", opponents=[Counting()],
                          opponent_share=0.25, seat_share=0.5)
        with self.assertRaises(ValueError):
            selfplay.play(FirstLegal(), games=2, seed=1, device="cpu", opponents=[Counting()],
                          seat_share=1.0)


class TrainerChecksTests(unittest.TestCase):
    def options(self, **kwargs):
        base = dict(batch=8, epochs=1, games=4, measure_every=1, measure_games=4)
        return SimpleNamespace(**{**base, **kwargs})

    def test_a_share_by_player_needs_others_and_excludes_a_share_by_game(self):
        with tempfile.TemporaryDirectory() as folder:
            other = Path(folder) / "other.pt"
            other.write_bytes(b"x")
            validate_training_options(self.options(opponents=[other], seat_share=0.5))
            for kwargs in ({"opponents": [], "seat_share": 0.5},
                           {"opponents": [other], "seat_share": 1.0},
                           {"opponents": [other], "seat_share": 0.5, "opponent_share": 0.25}):
                with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                    validate_training_options(self.options(**kwargs))

    def test_the_cloud_check_says_the_same(self):
        validate_cloud_request(1, ["zoo/mortal_298k"], 0.0, 0.5)
        for opponents, by_game, by_player in ((None, 0.0, 0.5), (["a"], 0.25, 0.5),
                                               (["a"], 0.0, 1.0), (["a"], 0.0, True)):
            with self.subTest(by_game=by_game, by_player=by_player), self.assertRaises(ValueError):
                validate_cloud_request(1, opponents, by_game, by_player)


class CloudLaunchTests(unittest.TestCase):
    def launch(self, trainer, **kwargs):
        from neural.tests.test_cloud_isolation import controller
        from neural import cloud_runs
        app = controller()
        calls = []
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            volume = root / "volume"
            atomic_save({"generation": 31}, volume / "run/latest.pt")
            atomic_save({"generation": 30}, volume / "zoo/mortal_298k.pt")

            def popen(command, **_kwargs):
                calls.append(command)
                return SimpleNamespace(stdout=io.StringIO(), wait=lambda: 0)

            stack.enter_context(patch.object(app, "VOLUME", volume))
            stack.enter_context(patch.object(app, "workspace", partial(cloud_runs.workspace, root=root / "scratch")))
            stack.enter_context(patch.object(app, "_environment", return_value={}))
            stack.enter_context(patch.object(app, "_save_cache"))
            stack.enter_context(patch.object(app.subprocess, "Popen", side_effect=popen))
            stack.enter_context(redirect_stdout(io.StringIO()))
            getattr(app, trainer)(run="run", generations=3, opponents=["zoo/mortal_298k"], **kwargs)
        (command,) = calls
        return command

    def test_both_trainers_are_told_to_seat_by_player(self):
        for trainer in ("train_mortal", "train_combined"):
            with self.subTest(trainer=trainer):
                command = self.launch(trainer, seat_share=0.5)
                self.assertEqual(command[command.index("--seat-share") + 1], "0.5")
                self.assertEqual(command[command.index("--opponent-share") + 1], "0.0")

    def test_without_it_nothing_changes(self):
        for trainer in ("train_mortal", "train_combined"):
            with self.subTest(trainer=trainer):
                command = self.launch(trainer, opponent_share=0.25)
                self.assertNotIn("--seat-share", command)


if __name__ == "__main__":
    unittest.main()
