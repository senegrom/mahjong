"""`--compile-learning` compiles the learning step's forward and nothing
else: play and the seated others stay eager, with their weights cast once
and the step's planes in bfloat16. The compiled step learns what the eager
one learns, as nearly as bfloat16 lets either; it is compiled once for each
mode a run draws, with gradients and without, and never again however many
generations follow, within a recompile limit raised to hold them all; and
the cloud keeps whatever a generation compiled."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import combined, train_combined
from neural.observe import Planes
from neural.tests import test_mixed_tables
from neural.tests.test_bfloat16_players import fusion
from neural.tests.test_cloud_isolation import controller
from neural.tests.test_learning_planes import encoded_like

CUDA = torch.cuda.is_available()
TRITON = importlib.util.find_spec("triton") is not None


def fake_round(phi: bool = False, values: bool = False):
    n = 8
    planes = Planes(np.arange(n + 1, dtype=np.int64), np.zeros(n, dtype=np.uint16),
                    np.ones(n, dtype=np.float16))
    return SimpleNamespace(
        decisions=n, observations=planes, legal=torch.ones(n, 46, dtype=torch.bool),
        actions=torch.zeros(n, dtype=torch.int64), returns=torch.linspace(-1, 1, n),
        log_probs=torch.zeros(n), held=torch.full((n, 3, 34), 1 / 34),
        games=1, hands=1, timing={},
        values=torch.linspace(-0.5, 0.5, n) if values else None,
        phi=torch.randn(n, 1024) if phi else None)


def forget_whether_triton_runs():
    """The compiler asks once whether a card can run its kernels and keeps
    the answer; asked while a test had told the trainer there was no card,
    it keeps a wrong one, and a later test that compiles for the card is
    refused. Forgotten here."""
    from torch.utils import _triton

    getattr(getattr(_triton, "has_triton", None), "cache_clear", lambda: None)()


def train(argv_tail, play, compile_with=None):
    """`train_combined.main` for some generations on the processor from a
    small joined player, every round `play(player)`'s, with
    `torch.compile` given to `compile_with` when that is set. Answers the
    records."""
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        origin, out = root / "source.pt", root / "run"
        torch.manual_seed(3)
        torch.save(fusion(8, 1).state(), origin)
        argv = ["trainer", "--resume", str(origin), "--out", str(out), "--batch", "4",
                "--epochs", "1", "--games", "1", "--measure-games", "1", "--measure-every", "1000",
                *argv_tail]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            stack.enter_context(patch.object(train_combined.selfplay, "play",
                                             side_effect=lambda player, **_kwargs: play(player)))
            stack.enter_context(patch.object(train_combined.selfplay, "measure",
                                             return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}))
            if compile_with is not None:
                stack.enter_context(patch.object(torch, "compile", compile_with))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            train_combined.main()
        return [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]


class SwitchTests(unittest.TestCase):
    def test_off_by_default_and_not_with_compile(self):
        with patch.object(sys, "argv", ["trainer"]):
            args = train_combined.parse_args()
        self.assertFalse(args.compile_learning)
        self.assertFalse(args.compile)
        with patch.object(sys, "argv", ["trainer", "--compile-learning"]):
            self.assertTrue(train_combined.parse_args().compile_learning)
        with patch.object(sys, "argv", ["trainer", "--compile", "--compile-learning"]), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                train_combined.parse_args()

    def test_the_cloud_passes_one_compile_or_the_other(self):
        launch = test_mixed_tables.CloudLaunchTests.launch
        command = launch(self, "train_combined")
        self.assertIn("--compile", command)
        self.assertNotIn("--compile-learning", command)
        command = launch(self, "train_combined", compile_learning=True)
        self.assertIn("--compile-learning", command)
        self.assertNotIn("--compile", command)
        command = launch(self, "train_combined", compile=False)
        self.assertNotIn("--compile", command)
        self.assertNotIn("--compile-learning", command)

    def test_only_the_learning_forward_is_compiled(self):
        """Play stays eager, the learner's forward for deciding untouched,
        and the learning forward is compiled for its one shape."""
        compiled, seen = [], []

        def recording(function, **options):
            compiled.append((getattr(function, "__name__", None), options))
            return function

        def play(player):
            seen.append(player.backbones_forward == player.backbones)
            return fake_round()

        train(["--rounds", "1", "--fixed", "none", "--compile-learning"], play, recording)
        self.assertEqual(compiled, [("everything", {"dynamic": False})])
        self.assertEqual(seen, [True])


class CacheTests(unittest.TestCase):
    def test_a_save_copies_only_what_the_volume_lacks(self):
        app = controller()
        commits = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            local, shared = root / "local", root / "shared"
            (shared / "a").mkdir(parents=True)
            (shared / "a" / "graph.py").write_text("seeded")
            (shared / "kernel.bin").write_bytes(b"seeded")
            with patch.object(app, "LOCAL_CACHE", local), patch.object(app, "SHARED_CACHE", shared), \
                    patch.object(app, "_ON_VOLUME", {}), \
                    patch.object(app, "volume", SimpleNamespace(commit=lambda: commits.append(1))), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app._seed_cache(), local)
                self.assertEqual((local / "a" / "graph.py").read_text(), "seeded")
                # Nothing compiled: nothing copied, nothing committed.
                app._save_cache()
                self.assertEqual(commits, [])
                # A generation compiled two files: those two, one commit.
                (local / "b").mkdir()
                (local / "b" / "new.py").write_text("compiled")
                (local / "fresh.bin").write_bytes(b"compiled")
                (shared / "kernel.bin").write_bytes(b"left alone")
                app._save_cache()
                self.assertEqual(commits, [1])
                self.assertEqual((shared / "b" / "new.py").read_text(), "compiled")
                self.assertEqual((shared / "fresh.bin").read_bytes(), b"compiled")
                self.assertEqual((shared / "kernel.bin").read_bytes(), b"left alone")
                app._save_cache()
                self.assertEqual(commits, [1])
                # A file the compiler rewrote goes again.
                (local / "fresh.bin").write_bytes(b"compiled again")
                app._save_cache()
                self.assertEqual(commits, [1, 1])
                self.assertEqual((shared / "fresh.bin").read_bytes(), b"compiled again")

    def test_the_compilers_workers_are_the_containers_share(self):
        app = controller()
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(app, "_seed_cache", return_value=Path(folder)), \
                contextlib.redirect_stdout(io.StringIO()):
            environment = app._environment(app.TRAINER_CPUS)
            self.assertEqual(environment["TORCHINDUCTOR_COMPILE_THREADS"], str(app.TRAINER_CPUS))
            self.assertEqual(app._environment()["TORCHINDUCTOR_COMPILE_THREADS"],
                             app._environment()["RAYON_NUM_THREADS"])


class RecompileTests(unittest.TestCase):
    """The trainer's own compile, with a backend that only counts the graphs
    it is handed: the compiler's guards decide when a graph is made anew,
    whatever compiles it, and this is quick."""

    def setUp(self):
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        forget_whether_triton_runs()

    def run_modes(self, flags, play):
        graphs, by_generation = [], []
        real_compile = torch.compile

        def counting(gm, _inputs):
            graphs.append(torch.is_grad_enabled())
            return gm.forward

        def compile_with(function, **options):
            return real_compile(function, backend=counting, **options)

        def playing(player):
            by_generation.append((player.mode, len(graphs)))
            return play(player)

        failing = {"fail_on_recompile_limit_hit": True} if hasattr(
            torch._dynamo.config, "fail_on_recompile_limit_hit") else {}
        with torch._dynamo.config.patch(recompile_limit=8, **failing):
            records = train(["--rounds", "30", "--seed", "11", "--compile-learning",
                             "--fixed", *combined.Combined.MODES, *flags], playing, compile_with)
            self.assertGreaterEqual(torch._dynamo.config.recompile_limit, 16)
        self.assertEqual([record["fixed"] for record in records], [mode for mode, _ in by_generation])
        counts = [count for _mode, count in by_generation] + [len(graphs)]
        made = {}
        for (mode, _count), before, after in zip(by_generation, counts, counts[1:]):
            if mode in made:
                # A mode seen before compiles nothing more, however many
                # generations follow.
                self.assertEqual(after - before, 0, mode)
            else:
                made[mode] = after - before
        self.assertEqual(set(made), set(combined.Combined.MODES))
        return made, graphs

    def test_each_mode_compiles_once_with_gradients_and_once_without(self):
        made, graphs = self.run_modes([], lambda player: fake_round())
        self.assertEqual(made, {mode: 2 for mode in combined.Combined.MODES})
        self.assertEqual(sorted(graphs), [False] * 6 + [True] * 6)

    def test_from_plays_values_and_vectors_each_mode_compiles_once(self):
        """With the baseline from play no pass is compiled; the modes that
        reuse Mortal's vectors compile their step with them."""
        made, graphs = self.run_modes(
            ["--baseline-from-play", "--reuse-phi"],
            lambda player: fake_round(phi=player.keep_phi, values=True))
        self.assertEqual(made, {mode: 1 for mode in combined.Combined.MODES})
        self.assertEqual(graphs, [True] * 6)


@unittest.skipUnless(CUDA and TRITON, "the compiler's kernels are the card's")
class SameLearningTests(unittest.TestCase):
    """The step compiled as the trainer compiles it, against the eager one,
    in two modes: the same loss within bfloat16's rounding, and gradients
    as close to a float32 step's as the eager step's are."""

    def setUp(self):
        torch._dynamo.reset()
        forget_whether_triton_runs()

    def tearDown(self):
        torch._dynamo.reset()

    def test_compiled_and_eager_learn_alike(self):
        def step(forward, net, planes, legal, actions, amp=True):
            net.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, value, guessed = forward(planes if amp else planes.float(), legal)
            loss = (-torch.log_softmax(logits.float(), dim=1).gather(1, actions[:, None]).mean()
                    + value.float().square().mean() + torch.log_softmax(guessed.float(), dim=2).mean())
            loss.backward()
            grads = torch.cat([p.grad.reshape(-1) for p in net.parameters() if p.grad is not None])
            return float(loss.detach()), grads

        for mode in ("none", "mortal"):
            torch.manual_seed(3)
            net = fusion().cuda()
            net.set_mode(mode)
            net.train()
            planes = encoded_like(256, seed=4).dense("cuda", torch.bfloat16)
            legal = torch.rand(256, 46, device="cuda") < 0.4
            legal[:, 0] = True
            actions = torch.multinomial(legal.float(), 1).squeeze(1)
            exact_loss, exact = step(net.everything, net, planes, legal, actions, amp=False)
            eager_loss, eager = step(net.everything, net, planes, legal, actions)
            compiled = torch.compile(net.everything, dynamic=False)
            loss, grads = step(compiled, net, planes, legal, actions)
            with self.subTest(mode=mode):
                self.assertLess(abs(loss - eager_loss), 2.0 ** -8 * abs(eager_loss))
                # As near the float32 step as the eager one comes, give or
                # take; bfloat16 rounds in other places compiled.
                self.assertLess(abs(loss - exact_loss),
                                2 * abs(eager_loss - exact_loss) + 2.0 ** -10 * abs(exact_loss))
                eager_off = float((eager - exact).norm() / exact.norm())
                compiled_off = float((grads - exact).norm() / exact.norm())
                self.assertLess(compiled_off, 2 * eager_off + 1e-3)
                self.assertGreater(float(torch.nn.functional.cosine_similarity(grads, eager, dim=0)), 0.99)


if __name__ == "__main__":
    unittest.main()
