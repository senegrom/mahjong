"""A round's deals are never the last round's, and a stalled Mortal run stops
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
from neural.training_state import round_seed


class RoundSeedTests(unittest.TestCase):
    def test_rounds_of_a_thousand_or_fewer_keep_their_seeds(self):
        for games in (1, 128, 512, 1000):
            with self.subTest(games=games):
                self.assertEqual(round_seed(20260907, 30, games), 20260907 + 30 * 1000)

    def test_large_rounds_never_replay_a_deal(self):
        for games in (1024, 4096, 8192):
            with self.subTest(games=games):
                dealt = set()
                for generation in range(30, 36):
                    first = round_seed(20260907, generation, games)
                    deals = set(range(first, first + games))
                    self.assertFalse(dealt & deals)
                    dealt |= deals


class MortalStopGenerationTests(unittest.TestCase):
    def launch(self, **kwargs):
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
            app.train_mortal(run="run", **kwargs)
        (command,) = calls
        return command[command.index("--rounds") + 1], command[command.index("--generations") + 1]

    def test_a_stop_generation_replaces_the_count(self):
        self.assertEqual(self.launch(generations=30, until=60), ("0", "60"))

    def test_without_one_the_count_runs_from_the_resume(self):
        self.assertEqual(self.launch(generations=30), ("30", "1000000"))

    def test_a_bad_stop_generation_is_refused_before_any_work(self):
        from neural.tests.test_cloud_isolation import controller
        app = controller()
        for until in (-1, True, 2.5, "60"):
            with self.subTest(until=until), patch.object(app, "workspace") as work:
                with self.assertRaises(ValueError):
                    app.train_mortal(run="run", generations=30, until=until)
                work.assert_not_called()


if __name__ == "__main__":
    unittest.main()
