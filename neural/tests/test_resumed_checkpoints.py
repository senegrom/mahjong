"""What a trainer does with the checkpoint it resumes from. It restores
Adam's moments as the checkpoint has them and each group's rate, weight
decay and kernel as its own code builds them, and lets the checkpoint go:
every trainer, fresh or resumed, leashed or not. The fusion's relaunch with
the arguments its watcher gives it restores the optimiser it always did."""

import contextlib
from functools import partial
import gc
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from neural import (
    cloud_runs, combined, model, mortal_learner, mortal_model, train, train_combined, train_mortal,
)
from neural.checkpoints import atomic_save
from neural.observe import Planes
from neural.tests.test_cloud_isolation import child, controller
from neural.tests.test_compiled_learning import compiler_settings_kept

MORTAL_CONFIG = {"resnet": {"conv_channels": 8, "num_blocks": 1}, "control": {"version": 4}}

#: What the watcher that relaunches fusion-long spawns `train_combined`
#: with: the recipe of 12 October 2026, every rate halved from generation
#: 873 on, the efficiency switches on, and its others seated by player.
LIVE = {
    "games": 4096, "batch": 2048, "epochs": 2, "lr": 1e-6, "lr_ours": 4e-7, "lr_mortal": 3e-7,
    "entropy": 0.0, "leash": 0.0, "entropy_target": 0.0, "target_kl": 0.01,
    "measure_every": 50, "measure_games": 512, "compile": False, "play_ahead": True,
    "play_graphs": True, "compile_learning": True, "preview_reach": True, "skip_forced": True,
    "reuse_phi": True, "generations": 900, "until": 1010, "resume": "latest",
    "opponents": ["zoo/mortal_298k", "mortal-run/history/gen-00030", "club"],
    "opponent_share": 0.0, "seat_share": 0.5, "run": "fusion-long",
}


def fake_round(actions: int):
    """A round of eight decisions over `actions` moves, with everything
    each trainer reads of one."""
    n = 8
    planes = Planes(np.arange(n + 1, dtype=np.int64), np.zeros(n, dtype=np.uint16),
                    np.ones(n, dtype=np.float16))
    return SimpleNamespace(
        decisions=n, observations=planes, legal=torch.ones(n, actions, dtype=torch.bool),
        actions=torch.zeros(n, dtype=torch.int64), returns=torch.linspace(-1, 1, n),
        log_probs=torch.full((n,), -1.0), held=torch.full((n, 3, 34), 1 / 34),
        oracle=torch.zeros(n, model.ORACLE_PLANES, 34, dtype=torch.uint8),
        games=1, hands=1, timing={})


def mortal_checkpoint(path: Path) -> Path:
    """A small Mortal in the layout a published one has."""
    torch.save({**mortal_model.build(8, 1).state(), "config": MORTAL_CONFIG}, path)
    return path


def flags(module, root: Path, resume: bool) -> list[str]:
    """The command line that starts `module` from a small network written
    under `root`, or resumes it from the run it wrote at `root / "run"`."""
    if resume:
        start = ["--resume", str(root / "run" / "latest.pt")]
    elif module is train:
        start = ["--channels", "8", "--blocks", "1"]
    else:
        ours, mortal = root / "ours.pt", mortal_checkpoint(root / "mortal.pt")
        net = model.PolicyValueNet(8, 1, actions=46)
        torch.save({"model": net.state_dict(), **net.payload_fields()}, ours)
        start = ["--mortal", str(mortal)] if module is train_mortal else [
            "--ours", str(ours), "--mortal", str(mortal)]
    own = {train: ["--replay-steps", "0"], train_mortal: [], train_combined: ["--fixed", "none"]}
    return [*start, *own[module]]


@contextlib.contextmanager
def faked(module, play=None, round_of=None):
    """`module` on the processor, every round a fake one, measured as even;
    `play`, when given, is called as each round is played, and
    `round_of(player)`, when given, makes the round. What the trainer
    prints is kept in the stream this yields."""
    actions = 78 if module is train else 46

    def played(player, *_args, **_kwargs):
        if play is not None:
            play()
        return fake_round(actions) if round_of is None else round_of(player)

    printed = io.StringIO()
    with patch.object(torch.cuda, "is_available", return_value=False), \
            patch.object(module.selfplay, "play", side_effect=played), \
            patch.object(module.selfplay, "measure",
                         return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
            contextlib.redirect_stdout(printed):
        yield printed


def run(module, out: Path, options: list[str], play=None) -> str:
    """One generation of `module` on a fake round, into `out` (see
    `faked`). Answers what the trainer printed."""
    argv = ["trainer", "--out", str(out), "--rounds", "1", "--batch", "4", "--epochs", "1",
            "--games", "1", "--measure-games", "1", *options]
    with patch.object(sys, "argv", argv), faked(module, play) as printed:
        module.main()
    return printed.getvalue()


class Held(dict):
    """A checkpoint, or the optimiser's state inside one, that can be
    watched for being let go of."""


class Recorded(torch.optim.AdamW):
    """AdamW, every one built kept, so that a test can read what a
    trainer's optimiser holds as its first round is played."""

    built: list = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        Recorded.built.append(self)


def held_by(optimiser) -> tuple[list[dict], list]:
    """Every group's settings, and every parameter's state in order, copied."""
    groups = [{key: value for key, value in group.items() if key != "params"}
              for group in optimiser.param_groups]
    states = []
    for group in optimiser.param_groups:
        for parameter in group["params"]:
            state = optimiser.state.get(parameter)
            states.append({key: value.clone() for key, value in state.items()} if state else None)
    return groups, states


def saved_as(saved: dict) -> tuple[list[dict], list]:
    """The same of an optimiser's state as a checkpoint saved it."""
    groups = [{key: value for key, value in group.items() if key != "params"}
              for group in saved["param_groups"]]
    states = [saved["state"].get(index) for group in saved["param_groups"] for index in group["params"]]
    return groups, states


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assertSameStates(self, got: list, want: list) -> None:
        self.assertEqual(len(got), len(want))
        self.assertGreater(sum(state is not None for state in want), 0)
        for index, (one, two) in enumerate(zip(got, want)):
            if two is None:
                self.assertIsNone(one, index)
                continue
            self.assertEqual(sorted(one), sorted(two), index)
            for key in two:
                self.assertEqual(one[key].dtype, two[key].dtype, (index, key))
                self.assertTrue(torch.equal(one[key], two[key]), (index, key))

    def test_no_trainer_keeps_the_checkpoint_it_resumed_from(self):
        """Each trainer kept the whole checkpoint it resumed from, its
        weights and Adam's moments, bound until it returned: about a third
        of a gigabyte of the host's memory through every generation at the
        fusion's shapes, and with the leash a copy of the reference besides.
        Looked for as the first round is played, nothing it read is held."""
        watched: list[weakref.ref] = []
        held: list[int] = []

        def watch(payload: dict) -> Held:
            payload = Held(payload)
            for key in ("optimizer_state", "optimizer"):
                if isinstance(payload.get(key), dict):
                    payload[key] = Held(payload[key])
                    watched.append(weakref.ref(payload[key]))
            watched.append(weakref.ref(payload))
            return payload

        def watching(load):
            """`load`, a loader that answers a network and its checkpoint,
            with the checkpoint watched."""
            def loaded(*args, **kwargs):
                net, payload = load(*args, **kwargs)
                return net, watch(payload)
            return loaded

        def read_whole(load):
            """`torch.load`, watching what it reads from a `latest.pt`, the
            checkpoint train resumes from."""
            def loaded(path, *args, **kwargs):
                payload = load(path, *args, **kwargs)
                return watch(payload) if Path(path).name == "latest.pt" else payload
            return loaded

        def look():
            gc.collect()
            held.append(sum(ref() is not None for ref in watched))

        loaders = {
            train: (torch, "load", read_whole(torch.load)),
            train_mortal: (mortal_learner, "load", watching(mortal_learner.load)),
            train_combined: (combined, "load", watching(combined.load)),
        }
        for module in (train, train_mortal, train_combined):
            # Leashed, the fusion reads its reference too, which a fresh
            # start writes first.
            leash = ["--leash", "0.1"] if module is train_combined else []
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                torch.manual_seed(5)
                for resume in (False, True):
                    with self.subTest(trainer=module.__name__, resume=resume):
                        watched.clear()
                        held.clear()
                        where, name, loader = loaders[module]
                        with patch.object(where, name, side_effect=loader):
                            run(module, root / "run", [*flags(module, root, resume), *leash], look)
                        # Read: the checkpoint resumed from and the optimiser's
                        # state in it, and the leash's reference.
                        self.assertEqual(len(watched), (2 if resume else 0) + (1 if leash else 0))
                        self.assertEqual(held, [0])

    def test_a_resumed_optimiser_takes_its_settings_from_the_code_and_its_moments_from_the_checkpoint(self):
        """Loading an optimiser's state takes every setting from the saved
        groups. train.py set its rate, weight decay and kernel again after,
        the Mortal-space trainers only their rates, so a change to their
        weight decay would never have reached a resumed run. Resumed from a
        checkpoint whose groups say otherwise, every trainer's optimiser has
        its own code's settings and the checkpoint's moments and step counts,
        and everything else of each group as the checkpoint saved it."""
        for module in (train, train_mortal, train_combined):
            with self.subTest(trainer=module.__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                torch.manual_seed(7)
                taken = []
                play = lambda: taken.append(held_by(Recorded.built[-1]))  # noqa: E731
                with patch.object(torch.optim, "AdamW", Recorded):
                    run(module, root / "run", flags(module, root, resume=False), play)
                    (built, _states), = taken
                    # A checkpoint from other code: another decay and rate
                    # in every group, and the kernel of the card.
                    latest = root / "run" / "latest.pt"
                    payload = torch.load(latest, map_location="cpu", weights_only=False)
                    key = "optimizer" if module is train else "optimizer_state"
                    for group in payload[key]["param_groups"]:
                        group.update(weight_decay=0.5, lr=0.123, fused=True)
                    saved = saved_as(payload[key])
                    atomic_save(payload, latest)
                    taken.clear()
                    printed = run(module, root / "run", flags(module, root, resume=True), play)
                self.assertIn("restored AdamW state", printed)
                (groups, states), = taken
                self.assertEqual(len(groups), len(built))
                for group, own, kept in zip(groups, built, saved[0]):
                    self.assertEqual({name: group[name] for name in ("lr", "weight_decay", "fused")},
                                     {name: own[name] for name in ("lr", "weight_decay", "fused")})
                    self.assertEqual({name: value for name, value in group.items()
                                      if name not in ("lr", "weight_decay", "fused")},
                                     {name: value for name, value in kept.items()
                                      if name not in ("lr", "weight_decay", "fused")})
                self.assertSameStates(states, saved[1])

    def test_the_fusion_relaunched_as_its_watcher_does_restores_the_optimiser_it_always_did(self):
        """fusion-long relaunched as its watcher relaunches it: the cloud's
        `train_combined` given the watcher's arguments, over a volume whose
        run holds a fusion this trainer wrote, and the trainer run on the
        command line and the workspace the cloud hands it, here on the
        processor with its rounds faked, the compiler left out and the batch
        cut to the round. What its optimiser holds as the first round is
        played is what the code before `ppo_loop.restore_optimiser`
        restored: every setting of every group as the checkpoint saved it
        but the rates, which are the watcher's, and every moment and step
        count as saved. The checkpoints the run writes on the card are
        fused and decay as the code builds them, so restoring the kernel
        and the decay from the code changes nothing there either."""
        app = controller()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            volume = root / "volume"
            torch.manual_seed(9)
            # The run as this trainer leaves it, and the others it seats.
            run(train_combined, root / "start", flags(train_combined, root, resume=False))
            (volume / "fusion-long").mkdir(parents=True)
            (root / "start" / "latest.pt").replace(volume / "fusion-long" / "latest.pt")
            (volume / "zoo").mkdir()
            mortal_checkpoint(volume / "zoo" / "mortal_298k.pt")
            (volume / "mortal-run" / "history").mkdir(parents=True)
            mortal_checkpoint(volume / "mortal-run" / "history" / "gen-00030.pt")
            saved = saved_as(torch.load(volume / "fusion-long" / "latest.pt", map_location="cpu",
                                        weights_only=False)["optimizer_state"])

            taken, commands = [], []

            def round_of(player):
                # Recorded at the player's own probabilities, as play records
                # them, so that the drift guard the watcher turns on lets the
                # generation learn.
                batch = fake_round(46)
                with torch.no_grad():
                    logits, _value, _hands = player.everything(batch.observations.dense("cpu"),
                                                               batch.legal)
                batch.log_probs = torch.log_softmax(logits.float(), dim=1)[
                    torch.arange(batch.decisions), batch.actions]
                return batch

            def popen(command, **_kwargs):
                commands.append(command)
                argv = ["trainer", *command[3:]]
                # One generation, of whole minibatches of the fake round.
                argv[argv.index("--rounds") + 1] = "1"
                argv[argv.index("--batch") + 1] = "4"
                with patch.object(sys, "argv", argv), patch.object(torch.optim, "AdamW", Recorded), \
                        patch.object(torch, "compile", lambda forward, **_options: forward), \
                        compiler_settings_kept(), \
                        faked(train_combined, lambda: taken.append(held_by(Recorded.built[-1])),
                              round_of) as printed:
                    train_combined.main()
                return child(printed.getvalue())

            with patch.object(app, "VOLUME", volume), \
                    patch.object(app, "workspace", partial(cloud_runs.workspace, root=root / "scratch")), \
                    patch.object(app, "_environment", return_value={}), \
                    patch.object(app, "_save_cache"), \
                    patch.object(app.subprocess, "Popen", side_effect=popen), \
                    contextlib.redirect_stdout(io.StringIO()):
                answer = app.train_combined(**LIVE)
        (command,) = commands
        for flag in ("--play-ahead", "--play-graphs", "--compile-learning", "--preview-reach",
                     "--skip-forced", "--reuse-phi", "--resume"):
            self.assertIn(flag, command)
        self.assertIn("generation=2 from 1", answer)
        (groups, states), = taken
        rates = (LIVE["lr"], LIVE["lr_ours"], LIVE["lr_mortal"])
        self.assertEqual(groups, [{**group, "lr": rate} for group, rate in zip(saved[0], rates)])
        self.assertEqual([group["weight_decay"] for group in groups], [1e-4, 1e-4, 0.01])
        self.assertSameStates(states, saved[1])


if __name__ == "__main__":
    unittest.main()
