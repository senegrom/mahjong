"""What a trainer does with the checkpoint it resumes from: it takes the
weights and Adam's moments into the network and the optimiser and lets the
checkpoint go, every trainer, fresh or resumed, leashed or not."""

import contextlib
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

from neural import combined, model, mortal_learner, mortal_model, train, train_combined, train_mortal
from neural.observe import Planes

MORTAL_CONFIG = {"resnet": {"conv_channels": 8, "num_blocks": 1}, "control": {"version": 4}}


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


def flags(module, root: Path, resume: bool) -> list[str]:
    """The command line that starts `module` from a small network written
    under `root`, or resumes it from the run it wrote at `root / "run"`."""
    if resume:
        start = ["--resume", str(root / "run" / "latest.pt")]
    elif module is train:
        start = ["--channels", "8", "--blocks", "1"]
    else:
        ours, mortal = root / "ours.pt", root / "mortal.pt"
        net = model.PolicyValueNet(8, 1, actions=46)
        torch.save({"model": net.state_dict(), **net.payload_fields()}, ours)
        torch.save({**mortal_model.build(8, 1).state(), "config": MORTAL_CONFIG}, mortal)
        start = ["--mortal", str(mortal)] if module is train_mortal else [
            "--ours", str(ours), "--mortal", str(mortal)]
    own = {train: ["--replay-steps", "0"], train_mortal: [], train_combined: ["--fixed", "none"]}
    return [*start, *own[module]]


def run(module, out: Path, options: list[str], play=None) -> str:
    """One generation of `module` on a fake round, on the processor, into
    `out`; `play`, when given, is called as the round is played. Answers
    what the trainer printed."""
    actions = 78 if module is train else 46

    def played(*_args, **_kwargs):
        if play is not None:
            play()
        return fake_round(actions)

    argv = ["trainer", "--out", str(out), "--rounds", "1", "--batch", "4", "--epochs", "1",
            "--games", "1", "--measure-games", "1", *options]
    printed = io.StringIO()
    with patch.object(sys, "argv", argv), \
            patch.object(torch.cuda, "is_available", return_value=False), \
            patch.object(module.selfplay, "play", side_effect=played), \
            patch.object(module.selfplay, "measure",
                         return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
            contextlib.redirect_stdout(printed):
        module.main()
    return printed.getvalue()


class Held(dict):
    """A checkpoint, or the optimiser's state inside one, that can be
    watched for being let go of."""


class ReleasedCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

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


if __name__ == "__main__":
    unittest.main()
