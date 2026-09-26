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


TRAINERS = ('train_mortal', 'train_combined')


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
