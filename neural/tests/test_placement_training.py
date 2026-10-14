"""train_combined's placement reward: its options, refused where they would
be ignored; the critic's warm-up, which moves the critic and nothing else;
a run resumed as it was trained; U refitted on the run's own rounds; and
the default run, which none of it touches.

    python -m unittest neural.tests.test_placement_training -v
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from neural import cloud_requests, place_model, train_combined
from neural.observe import Planes
from neural.tests.test_placement_reward import ledger_of
from neural.tests.test_resumed_checkpoints import flags


def ledgered_round():
    """A round of eight decisions over Mortal's moves, two hands of one
    game, with the ledger a placement run settles it from."""
    n = 8
    planes = Planes(np.arange(n + 1, dtype=np.int64), np.zeros(n, dtype=np.uint16),
                    np.ones(n, dtype=np.float16))
    ledger = ledger_of(
        [[(0, 1, 0, 0, 0, 30000, 30000, 30000, 30000), (0, 2, 0, 0, 1, 38000, 26000, 30000, 26000)]],
        [(0, hand, player, 0.25 * (player - 1.5)) for hand in (0, 1) for player in range(4)],
        [[30000, -5000, 10000, -35000]])
    return SimpleNamespace(
        decisions=n, observations=planes, legal=torch.ones(n, 46, dtype=torch.bool),
        actions=torch.arange(n, dtype=torch.int64), returns=torch.linspace(-1, 1, n),
        log_probs=torch.full((n,), -3.8), held=torch.full((n, 3, 34), 1 / 34),
        games=1, hands=2, timing={}, ledger=ledger)


@contextlib.contextmanager
def faked(asked: list):
    """train_combined on the processor, every round `ledgered_round`, the
    options each round was asked for kept in `asked`."""

    def played(player, *_args, **kwargs):
        asked.append(kwargs)
        return ledgered_round()

    printed = io.StringIO()
    with patch.object(torch.cuda, "is_available", return_value=False), \
            patch.object(train_combined.selfplay, "play", side_effect=played), \
            patch.object(train_combined.selfplay, "measure",
                         return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
            contextlib.redirect_stdout(printed):
        yield printed


def run(out: Path, options: list[str], asked: list | None = None) -> str:
    argv = ["trainer", "--out", str(out), "--rounds", "1", "--batch", "4", "--epochs", "1",
            "--games", "1", "--measure-games", "1", *options]
    with patch.object(sys, "argv", argv), faked([] if asked is None else asked) as printed:
        train_combined.main()
    return printed.getvalue()


def saved(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def write_u(path: Path, members=()) -> Path:
    model = place_model.PlaceModel(members, hidden=8)
    model.fitted = {"made": "for a test"}
    model.save(path)
    return path


class PlacementOptions(unittest.TestCase):
    def parse(self, *options: str):
        with patch.object(sys, "argv", ["trainer", "--out", "x", *options]), \
                contextlib.redirect_stderr(io.StringIO()):
            return train_combined.parse_args()

    def test_the_default_asks_for_nothing_new(self):
        args = self.parse()
        self.assertEqual((args.reward, args.alpha, args.place_model, args.place_refit_every,
                          args.value_warm_up), ("hybrid", 0.0, None, 0, 0))

    def test_an_option_of_the_placement_reward_alone_is_refused(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            u = write_u(Path(folder) / "u.pt")
            for options in (["--alpha", "0.05"], ["--value-warm-up", "1"], ["--place-refit-every", "50"],
                            ["--place-model", str(u)]):
                with self.subTest(options=options), self.assertRaises(SystemExit):
                    self.parse(*options)
            for options in (["--reward", "placement", "--alpha", "-1"],
                            ["--reward", "placement", "--alpha", "nan"],
                            ["--reward", "placement", "--value-warm-up", "-1"],
                            ["--reward", "placement", "--place-refit-rounds", "0"],
                            ["--reward", "placement", "--value-warm-up-lr", "0"],
                            ["--reward", "placement", "--place-model", str(Path(folder) / "missing.pt")]):
                with self.subTest(options=options), self.assertRaises(SystemExit):
                    self.parse(*options)
            args = self.parse("--reward", "placement", "--alpha", "0.05", "--place-model", str(u),
                              "--value-warm-up", "1", "--place-refit-every", "50")
            self.assertEqual((args.reward, args.alpha, args.value_warm_up), ("placement", 0.05, 1))

    def test_a_launch_passes_nothing_new_unless_it_asks_for_the_reward(self):
        self.assertEqual(cloud_requests.placement_arguments(), [])
        self.assertEqual(
            cloud_requests.placement_arguments("placement", 0.05, "x", 50, 2, 1, 1e-4),
            ["--reward", "placement", "--alpha", "0.05", "--place-refit-every", "50",
             "--place-refit-rounds", "2", "--value-warm-up", "1", "--value-warm-up-lr", "0.0001"])
        for asked in ({"alpha": 0.05}, {"place_model": "u"}, {"refit_every": 5}, {"warm_up": 1},
                      {"reward": "grp"}, {"reward": "placement", "alpha": -1.0},
                      {"reward": "placement", "refit_rounds": 0}, {"reward": "placement", "warm_up_lr": 0.0}):
            with self.subTest(asked=asked), self.assertRaises(ValueError):
                cloud_requests.placement_arguments(**asked)


class PlacementRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.folder.name)
        torch.manual_seed(11)
        # A default generation first, as fusion-long's checkpoints were made.
        self.asked: list = []
        run(self.root / "run", flags(train_combined, self.root, resume=False), self.asked)
        self.start = saved(self.root / "run" / "latest.pt")
        self.u = write_u(self.root / "u.pt")

    def tearDown(self):
        self.folder.cleanup()

    def resume(self, *options: str, asked: list | None = None) -> str:
        return run(self.root / "run", [*flags(train_combined, self.root, resume=True), *options], asked)

    def test_a_default_run_keeps_no_ledger_and_carries_nothing_new(self):
        self.assertEqual(len(self.asked), 1)
        self.assertNotIn("ledger", self.asked[0])
        self.assertNotIn("reward", self.start)
        self.assertNotIn("place_model", self.start)
        self.assertEqual(len(self.start["optimizer_state"]["param_groups"]), 3)

    def test_the_warm_up_moves_the_critic_and_nothing_else(self):
        """A warm-up generation, resumed from a default run's checkpoint as
        a fork from fusion-long would be: every weight but the critic's is
        the same after it, bit for bit, and so is every move's probability;
        the main optimiser's moments are untouched; the critic has moved."""
        asked: list = []
        printed = self.resume("--reward", "placement", "--place-model", str(self.u), "--value-warm-up", "1",
                              asked=asked)
        self.assertEqual([options.get("ledger") for options in asked], [True])
        self.assertIn("the placement reward begins at generation 1", printed)
        after = saved(self.root / "run" / "latest.pt")
        critic = {("model", name) for name in after["model"] if name.startswith("value.")}
        critic |= {("combined", name) for name in after["combined"] if name.startswith("value_fix.")}
        self.assertTrue(critic)
        moved = set()
        for part in ("model", "combined", "mortal", "current_dqn"):
            for name, tensor in after[part].items():
                if not torch.equal(tensor, self.start[part][name]):
                    moved.add((part, name))
        self.assertEqual(moved, critic)
        for key, state in after["optimizer_state"]["state"].items():
            for name, tensor in state.items():
                self.assertTrue(torch.equal(tensor, self.start["optimizer_state"]["state"][key][name]),
                                (key, name))
        self.assertEqual(after["reward"]["warmed"], 1)
        self.assertEqual(after["reward"]["since"], 1)
        self.assertEqual(after["reward"]["alpha"], 0.0)
        self.assertEqual(after["place_model"]["fitted"], {"made": "for a test"})

        # The same probabilities for every move, read through the networks
        # as they stand before and after.
        from neural import combined

        def logits(payload: dict) -> torch.Tensor:
            path = self.root / "probe.pt"
            torch.save(payload, path)
            net, _state = combined.load(path, "cpu")
            torch.manual_seed(3)
            planes = torch.rand(6, 1012, 34)
            legal = torch.rand(6, 46) < 0.5
            legal[:, 0] = True
            return net.eval().everything(planes, legal)[0]

        self.assertTrue(torch.equal(logits(self.start), logits(after)))

    def test_a_warm_up_is_had_once_and_the_run_then_learns(self):
        self.resume("--reward", "placement", "--place-model", str(self.u), "--value-warm-up", "1")
        warmed = saved(self.root / "run" / "latest.pt")
        self.resume("--reward", "placement", "--place-model", str(self.u), "--value-warm-up", "1")
        learned = saved(self.root / "run" / "latest.pt")
        self.assertEqual(learned["reward"]["warmed"], 1)
        self.assertEqual(learned["generation"], 3)
        moved = [name for name, tensor in learned["combined"].items()
                 if not name.startswith("value_fix.") and not torch.equal(tensor, warmed["combined"][name])]
        self.assertTrue(moved, "a generation after the warm-up should move the head")
        log = (self.root / "run" / "log.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertIn('"value_warm_up": true', log[1])
        self.assertIn('"value_warm_up": false', log[2])
        self.assertIn('"reward": "placement"', log[2])
        self.assertIn('"value_error_east"', log[2])
        self.assertNotIn('"reward"', log[0])

    def test_a_placement_run_resumes_only_as_it_was_trained(self):
        self.resume("--reward", "placement", "--place-model", str(self.u))
        with self.assertRaisesRegex(SystemExit, "placement reward at alpha 0.0"):
            self.resume()
        with self.assertRaisesRegex(SystemExit, "start a new run"):
            self.resume("--reward", "placement", "--alpha", "0.05")

    def test_a_resumed_run_settles_by_the_u_its_checkpoint_carries(self):
        self.resume("--reward", "placement", "--place-model", str(self.u))
        carried = saved(self.root / "run" / "latest.pt")["place_model"]
        other = write_u(self.root / "other.pt", ())
        printed = self.resume("--reward", "placement", "--place-model", str(other))
        self.assertIn("is not read", printed)
        again = saved(self.root / "run" / "latest.pt")["place_model"]
        for name, tensor in carried["weights"].items():
            self.assertTrue(torch.equal(tensor, again["weights"][name]), name)

    def test_a_run_starting_the_reward_needs_a_u_that_knows_its_others(self):
        with self.assertRaisesRegex(SystemExit, "give --place-model"):
            self.resume("--reward", "placement")
        with self.assertRaisesRegex(SystemExit, "fitted without"):
            self.resume("--reward", "placement", "--place-model", str(self.u), "--opponents", "club",
                        "--seat-share", "0.5")


class CloudLaunches(unittest.TestCase):
    """The cloud's side, run locally with Modal stood in for: a placement
    launch stages U and names every option; a default launch names none;
    the pre-test publishes what a fork starts from."""

    def launch(self, app, volume: Path, scratch: Path, popen, function: str, **kwargs):
        import functools

        from neural import cloud_runs

        with patch.object(app, "VOLUME", volume), \
                patch.object(app, "workspace", functools.partial(cloud_runs.workspace, root=scratch)), \
                patch.object(app, "_environment", return_value={}), patch.object(app, "_save_cache"), \
                patch.object(app.subprocess, "Popen", side_effect=popen), \
                contextlib.redirect_stdout(io.StringIO()):
            return getattr(app, function)(**kwargs)

    def test_a_placement_launch_stages_u_and_names_its_options(self):
        import json

        from neural.checkpoints import atomic_save
        from neural.tests.test_cloud_isolation import controller

        app = controller()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            volume = root / "volume"
            atomic_save({"generation": 7}, volume / "base" / "latest.pt")
            write_u(volume / "base" / "u.pt")
            commands = []

            def popen(command, **_kwargs):
                where = Path(command[command.index("--out") + 1])
                commands.append((command, sorted(path.name for path in where.iterdir())))
                atomic_save({"generation": 8}, where / "latest.pt")
                record = json.dumps({"generation": 7, "checkpoint_generation": 8})
                return SimpleNamespace(stdout=io.StringIO(record + "\n"), wait=lambda: 0)

            self.launch(app, volume, root / "scratch", popen, "train_combined", run="fork",
                        resume="base/latest", generations=1, reward="placement", alpha=0.05,
                        place_model="base/u", value_warm_up=1, place_refit_every=50)
            self.launch(app, volume, root / "scratch", popen, "train_combined", run="plain",
                        resume="base/latest", generations=1)
            with self.assertRaises(ValueError):
                self.launch(app, volume, root / "scratch", popen, "train_combined", run="plain",
                            resume="base/latest", generations=1, alpha=0.05)
        (placed, files), (plain, _files) = commands
        self.assertIn("place-model.pt", files)
        at = placed.index("--reward")
        self.assertEqual(placed[at:at + 4], ["--reward", "placement", "--alpha", "0.05"])
        for flag, value in (("--place-refit-every", "50"), ("--value-warm-up", "1")):
            self.assertEqual(placed[placed.index(flag) + 1], value)
        self.assertTrue(placed[placed.index("--place-model") + 1].endswith("place-model.pt"))
        for flag in ("--reward", "--alpha", "--place-model", "--value-warm-up", "--place-refit-every"):
            self.assertNotIn(flag, plain)

    def test_the_pretest_publishes_what_a_fork_starts_from(self):
        import json

        from neural.checkpoints import atomic_save
        from neural.tests.test_cloud_isolation import child, controller

        app = controller()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            volume = root / "volume"
            atomic_save({"generation": 1010}, volume / "fusion-long" / "latest.pt")
            atomic_save({"generation": 30}, volume / "zoo" / "mortal_298k.pt")
            asked = []

            def popen(command, **_kwargs):
                out = Path(command[command.index("--out") + 1])
                asked.append(command)
                out.mkdir(parents=True)
                (out / "report.json").write_text('{"verdict": "go"}', encoding="utf-8")
                (out / "ledger.npz").write_bytes(b"ledger")
                write_u(out / "place-model.pt")
                return child(stdout='{"phase": "play"}\nVERDICT {"verdict": {"verdict": "go"}}\n')

            answer = self.launch(app, volume, root / "scratch", popen, "placement_pretest",
                                 opponents=["zoo/mortal_298k", "club"], games=64, probes=100)
            target = volume / "fusion-long" / "pretest-g1010"
            self.assertTrue(answer.rstrip().endswith(f"wrote {target}"))
            self.assertIn("VERDICT", answer)
            self.assertEqual(json.loads((target / "report.json").read_text(encoding="utf-8")), {"verdict": "go"})
            self.assertEqual(place_model.PlaceModel.load(target / "place-model.pt").fitted, {"made": "for a test"})
            self.assertTrue((target / "ledger.npz").exists())
        command, = asked
        self.assertIn("neural.placement_pretest", command)
        self.assertEqual(command[command.index("--seat-share") + 1], "0.5")
        self.assertEqual(command[command.index("--games") + 1], "64")


class Refits(unittest.TestCase):
    def ledger(self, rng, games: int):
        starts, finals = [], []
        for _game in range(games):
            lead = int(rng.integers(0, 4))
            scores = [25000] * 4
            scores[lead] = 45000
            scores[(lead + 1) % 4] = 25000 - 0
            final = [10000, 0, -10000, -20000]
            final = [final[(player - lead) % 4] for player in range(4)]
            starts.append([(0, 1, 0, 0, 0, 30000, 30000, 30000, 30000), (1, 4, 0, 0, 3, *scores)])
            finals.append(final)
        return ledger_of(starts, [(0, 0, 0, 0.0)], finals)

    def test_u_is_refitted_when_due_and_kept_only_when_it_reads_better(self):
        rng = np.random.default_rng(2)
        args = SimpleNamespace(alpha=0.0, place_model=None, value_warm_up=0, place_refit_every=3,
                               place_refit_rounds=2)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            args.place_model = write_u(Path(folder) / "u.pt")
            placing = place_model.PlacementRun.begin(args, {}, 10, [])
        self.assertEqual([g for g in range(10, 20) if placing.refit_due(g)], [12, 15, 18])
        before = placing.reward
        for generation in (10, 11, 12):
            placing.remember(self.ledger(rng, 300), generation)
        self.assertEqual(len(placing.recent), 2)
        report = placing.refit(12)
        self.assertTrue(report["adopted"])
        self.assertLess(report["new_cross_entropy"], report["old_cross_entropy"])
        self.assertIsNot(placing.reward, before)
        self.assertEqual(report["generation"], 13)
        self.assertEqual(placing.fields()["reward"]["refits"][-1], report)
        # A U already as good as the data allows is kept against a worse one.
        kept = placing.reward
        with patch.object(place_model, "fit", return_value=place_model.PlaceModel((), hidden=8)):
            worse = placing.refit(15)
        self.assertFalse(worse["adopted"])
        self.assertIs(placing.reward, kept)


if __name__ == "__main__":
    unittest.main()
