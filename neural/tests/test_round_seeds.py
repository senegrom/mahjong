"""A round's deals are never the last round's, and a stalled cloud run stops
at its generation rather than playing its count again from the top."""
from contextlib import ExitStack, redirect_stdout
from functools import partial
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from neural.checkpoints import atomic_save
from neural.training_state import ROUND_SEED_STRIDE, round_seed


class RoundSeedTests(unittest.TestCase):
    def test_generation_range_does_not_depend_on_round_size(self):
        for games in (1, 128, 512, 1000, 1024, 4096, 8192):
            with self.subTest(games=games):
                self.assertEqual(round_seed(20260907, 30, games),
                                 20260907 + 31 * ROUND_SEED_STRIDE)

    def test_large_rounds_never_replay_a_deal(self):
        for games in (1024, 4096, 8192):
            with self.subTest(games=games):
                dealt = set()
                for generation in range(30, 36):
                    first = round_seed(20260907, generation, games)
                    deals = set(range(first, first + games))
                    self.assertFalse(dealt & deals)
                    dealt |= deals

    def test_resume_with_smaller_or_larger_rounds_never_replays(self):
        for sizes in ([4096] * 10 + [2048, 128, 8192, 1, 4096],
                      [128, 8192, 512, 4096, 1000, 2048]):
            dealt = set()
            for generation, games in enumerate(sizes):
                first = round_seed(20260907, generation, games)
                deals = set(range(first, first + games))
                self.assertFalse(dealt & deals)
                dealt |= deals
                # A restart at the same absolute generation is deterministic.
                self.assertEqual(first, round_seed(20260907, generation, games))

    def test_legacy_resume_clears_both_previous_seed_formulas(self):
        seed, resumed_at = 20260907, 10
        for old_games in (128, 1024, 4096, 8192, ROUND_SEED_STRIDE):
            for old_stride in (1000, max(1000, old_games)):
                old_end = seed + (resumed_at - 1) * old_stride + old_games
                for new_games in (128, 2048, 8192):
                    self.assertGreaterEqual(round_seed(seed, resumed_at, new_games), old_end)

    def test_full_ranges_are_disjoint_even_at_their_boundaries(self):
        first = round_seed(0, 0, ROUND_SEED_STRIDE)
        second = round_seed(0, 1, ROUND_SEED_STRIDE)
        self.assertEqual(first + ROUND_SEED_STRIDE, second)
        self.assertEqual(round_seed(0, (1 << 32) - 2, ROUND_SEED_STRIDE)
                         + ROUND_SEED_STRIDE, 1 << 64)

    def test_bad_inputs_and_u64_overflow_are_refused(self):
        for args in ((-1, 0, 1), (1 << 64, 0, 1), (True, 0, 1),
                     (0, -1, 1), (0, True, 1), (0, 1.5, 1),
                     (0, 0, 0), (0, 0, True), (0, 0, 2.5),
                     (0, 0, ROUND_SEED_STRIDE + 1),
                     (0, (1 << 32) - 1, 1),
                     (1, (1 << 32) - 2, ROUND_SEED_STRIDE)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                round_seed(*args)


class MeasureSeedTests(unittest.TestCase):
    """The measurement against the heuristic players, taken every few
    generations, plays deals of its own each time. Seeded one apart, two
    measurements 5 generations apart shared 507 of their 512 games, and the
    smoothed figure best.pt is chosen on kept the luck of those deals."""

    def test_every_trainer_measures_on_deals_no_other_measurement_played(self):
        import sys
        from neural import train, train_combined, train_mortal
        from neural.tests.test_resumed_checkpoints import faked, flags
        for module in (train, train_mortal, train_combined):
            with self.subTest(trainer=module.__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                asked = []

                def measured(_net, *, games, seed, **_options):
                    asked.append(range(seed, seed + games))
                    return {"placement": 2.5, "score": 0.0, "wins": 0.25}

                argv = ["trainer", "--out", str(root / "run"), "--rounds", "3", "--batch", "4",
                        "--epochs", "1", "--games", "1", "--measure-every", "1",
                        "--measure-games", "16", *flags(module, root, resume=False)]
                with patch.object(sys, "argv", argv), faked(module), \
                        patch.object(module.selfplay, "measure", side_effect=measured):
                    module.main()
                self.assertEqual(len(asked), 3)
                dealt = set()
                for deals in asked:
                    self.assertEqual(len(deals), 16)
                    self.assertFalse(dealt & set(deals), asked)
                    dealt |= set(deals)

    def test_measurements_stay_clear_of_the_rounds(self):
        from neural.training_state import MEASURE_SEED_STRIDE, measure_seed
        self.assertEqual(measure_seed(1, 512) - measure_seed(0, 512), MEASURE_SEED_STRIDE)
        last = measure_seed(65_000, MEASURE_SEED_STRIDE) + MEASURE_SEED_STRIDE
        self.assertLessEqual(last, round_seed(0, 0, 1))
        for args in ((-1, 1), (0, 0), (0, MEASURE_SEED_STRIDE + 1), (1 << 20, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                measure_seed(*args)


TRAINERS = ('train', 'train_mortal', 'train_combined')


class StopGenerationTests(unittest.TestCase):
    def launch(self, trainer, **kwargs):
        from neural.tests.test_cloud_isolation import controller
        from neural import cloud_runs
        app = controller()
        calls = []
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            volume = root / "volume"
            atomic_save({"generation": 30}, volume / "run/latest.pt")

            def popen(command, **_kwargs):
                calls.append(command)
                return SimpleNamespace(stdout=io.StringIO(), wait=lambda: 0)

            stack.enter_context(patch.object(app, "VOLUME", volume))
            stack.enter_context(patch.object(app, "workspace", partial(cloud_runs.workspace, root=root / "scratch")))
            stack.enter_context(patch.object(app, "_environment", return_value={}))
            stack.enter_context(patch.object(app, "_save_cache"))
            stack.enter_context(patch.object(app.subprocess, "Popen", side_effect=popen))
            stack.enter_context(redirect_stdout(io.StringIO()))
            getattr(app, trainer)(run="run", **kwargs)
        (command,) = calls
        return command[command.index("--rounds") + 1], command[command.index("--generations") + 1]

    def test_a_stop_generation_replaces_the_count(self):
        for trainer in TRAINERS:
            with self.subTest(trainer=trainer):
                self.assertEqual(self.launch(trainer, generations=30, until=60), ("0", "60"))

    def test_without_one_the_count_runs_from_the_resume(self):
        for trainer in TRAINERS:
            with self.subTest(trainer=trainer):
                self.assertEqual(self.launch(trainer, generations=30), ("30", "1000000"))

    def test_a_call_started_again_finishes_its_target(self):
        # Resumed at 30 with a target of 40, every trainer stops at 40;
        # `train`, which took no target, went on to 70.
        from neural.tests.test_cloud_isolation import controller
        app = controller()
        for trainer in TRAINERS:
            with self.subTest(trainer=trainer):
                rounds, generations = self.launch(trainer, generations=40, until=40)
                self.assertEqual(app._target(["--rounds", rounds, "--generations", generations], 30), 40)

    def test_a_bad_stop_generation_is_refused_before_any_work(self):
        from neural.tests.test_cloud_isolation import controller
        app = controller()
        for trainer in TRAINERS:
            for until in (-1, True, 2.5, "60"):
                with self.subTest(trainer=trainer, until=until), patch.object(app, "workspace") as work:
                    with self.assertRaises(ValueError):
                        getattr(app, trainer)(run="run", generations=30, until=until)
                    work.assert_not_called()


if __name__ == "__main__":
    unittest.main()
