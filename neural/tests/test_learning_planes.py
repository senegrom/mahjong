"""The learning step's planes reach the card sooner and smaller without a
bit of what it learns changing: made dense in bfloat16 where only autocast
reads them, copied from page-locked memory on a stream of their own, and
checked in full once a round rather than once a minibatch."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from contextlib import closing
from unittest.mock import patch

import numpy as np
import torch

from neural import combined, mortal_model, ppo_loop, train_combined
from neural.model import PolicyValueNet
from neural.observe import WIDTH, DevicePlanes, Planes
from neural.prefetch import Prefetcher

CUDA = torch.cuda.is_available()


def every_finite_half() -> Planes:
    """Every finite float16, signed zeros and subnormals among them, one
    entry each, in rows of 256 at distinct columns."""
    bits = np.arange(1 << 16).astype(np.uint16)
    values = bits.view(np.float16)
    values = values[np.isfinite(values)]
    rows = -(-len(values) // 256)
    indptr = np.minimum(np.arange(rows + 1, dtype=np.int64) * 256, len(values))
    columns = (np.arange(len(values)) % 256 * 131 % WIDTH).astype(np.uint16)
    return Planes(indptr, columns, values)


def encoded_like(rows: int, seed: int) -> Planes:
    """Planes of a step's shape whose rows, as the encoder writes them,
    never name a column twice, so a dense copy says one value a column."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(1400, 2500, size=rows)
    counts[rng.choice(rows, size=rows // 50, replace=False)] = 0
    indptr = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.concatenate([np.sort(rng.choice(WIDTH, size=int(count), replace=False))
                              for count in counts]).astype(np.uint16)
    values = rng.random(int(indptr[-1]), dtype=np.float32).astype(np.float16)
    return Planes(indptr, indices, values)


def minibatches_as_they_were(rollout, size):
    """`ppo_loop.minibatches` for planes on the host as it was before the
    staging: pageable copies on the step's own stream. The reference."""
    order = torch.randperm(rollout.decisions)
    slices = [order[start : start + size] for start in range(0, rollout.decisions, size)]
    slices = [drawn for drawn in slices if drawn.numel() == size]
    for drawn in slices:
        yield drawn.to(rollout.device), rollout.observations.rows(drawn.numpy()).dense(rollout.device)


def host_round(planes: Planes, device: str):
    n = len(planes)
    return ppo_loop.Round(
        decisions=n, device=device, observations=planes, legal=torch.ones(n, 46, dtype=torch.bool),
        actions=torch.zeros(n, dtype=torch.int64), returns=torch.zeros(n),
        old_log_probs=torch.zeros(n), on_card=None,
    )


def same_bits(test, got: torch.Tensor, want: torch.Tensor):
    test.assertEqual(got.dtype, want.dtype)
    test.assertEqual(got.shape, want.shape)
    width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[got.element_size()]
    test.assertTrue(torch.equal(got.view(width), want.view(width)))


class BfloatPlanesTests(unittest.TestCase):
    def test_bfloat16_planes_are_the_cast_of_float32_planes_bit_for_bit(self):
        planes = every_finite_half()
        for device in ("cpu", "cuda") if CUDA else ("cpu",):
            with self.subTest(device=device):
                single = planes.dense(device)
                self.assertEqual(single.dtype, torch.float32)
                half = planes.dense(device, torch.bfloat16)
                same_bits(self, half, single.to(torch.bfloat16))
                # float32 still holds every value exactly.
                written = single.reshape(len(planes), -1)
                kept = written[torch.repeat_interleave(torch.arange(len(planes)),
                                                       torch.from_numpy(np.diff(planes.indptr))).to(device),
                               torch.from_numpy(planes.indices.astype(np.int64)).to(device)]
                same_bits(self, kept.cpu().half(), torch.from_numpy(planes.values))
                if device == "cuda":
                    on_card = DevicePlanes(planes, device)
                    picks = torch.randperm(len(planes), device=device)
                    same_bits(self, on_card.rows(picks, torch.bfloat16),
                              on_card.rows(picks).to(torch.bfloat16))
                    same_bits(self, on_card.rows(picks), planes.rows(picks.cpu().numpy()).dense(device))
                    same_bits(self, planes.dense(device, torch.bfloat16, pinned=True), half)

    @unittest.skipUnless(CUDA, "autocast's bfloat16 is the card's")
    def test_learning_from_bfloat16_planes_is_bit_for_bit_the_same(self):
        """The joined player's whole forward and backward under autocast,
        from the same rows made dense in float32 and in bfloat16: the
        three stems are handed the same bits, the answers are the same, and
        so is every gradient."""
        torch.manual_seed(41)
        net = combined.Combined(PolicyValueNet(32, 2, actions=46), mortal_model.build(16, 2)).cuda()
        net.train()
        planes = encoded_like(96, seed=8)
        legal = torch.rand(96, 46, device="cuda") < 0.5
        legal[:, 0] = True
        stems = [net.ours.stem[0], net.ours.belief_stem[0], net.mortal.brain.encoder.net[0]]
        deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True
        try:
            runs = []
            for dtype in (torch.float32, torch.bfloat16):
                handed = []
                hooks = [stem.register_forward_pre_hook(
                    lambda _module, inputs: handed.append(inputs[0].to(torch.bfloat16)))
                    for stem in stems]
                net.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, value, guessed = net.everything(planes.dense("cuda", dtype), legal)
                loss = (torch.log_softmax(logits.float(), dim=1).masked_fill(~legal, 0).sum()
                        + value.float().square().sum() + guessed.float().square().sum())
                loss.backward()
                for hook in hooks:
                    hook.remove()
                runs.append((handed, (logits, value, guessed),
                             [p.grad.clone() for p in net.parameters() if p.grad is not None]))
        finally:
            torch.backends.cudnn.deterministic = deterministic
        (handed, answers, grads), (handed_half, answers_half, grads_half) = runs
        self.assertEqual(len(handed), 3)
        for got, want in zip(handed_half, handed):
            same_bits(self, got, want)
        for got, want in zip(answers_half, answers):
            same_bits(self, got, want)
        self.assertEqual(len(grads_half), len(grads))
        self.assertGreater(len(grads), 50)
        for got, want in zip(grads_half, grads):
            same_bits(self, got, want)


class TrainerPlanesTests(unittest.TestCase):
    @unittest.skipUnless(CUDA, "mixed precision is the card's")
    def test_bfloat16_planes_only_for_eager_learning_in_mixed_precision(self):
        """The joined player's trainer makes its planes dense in bfloat16
        with --amp, and in float32, as it always did, without it or with
        --compile, whose graphs were not shown to give the same bits under
        the cloud's torch. The compiler itself is not run here."""
        torch.set_num_threads(1)
        real_baseline, real_minibatches = ppo_loop.baseline, ppo_loop.minibatches

        def round_of_eight(_learner, **_kwargs):
            n = 8
            planes = Planes(np.arange(n + 1, dtype=np.int64), np.zeros(n, dtype=np.uint16),
                            np.ones(n, dtype=np.float16))
            return SimpleNamespace(
                decisions=n, observations=planes, legal=torch.ones(n, 46, dtype=torch.bool),
                actions=torch.zeros(n, dtype=torch.int64), returns=torch.linspace(-1, 1, n),
                log_probs=torch.zeros(n), held=torch.full((n, 3, 34), 1 / 34),
                games=1, hands=1, timing={})

        for flags, wanted in ((["--amp"], torch.bfloat16), (["--amp", "--compile"], torch.float32),
                              ([], torch.float32)):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                origin = root / "source.pt"
                torch.manual_seed(3)
                net = combined.Combined(PolicyValueNet(8, 1, actions=46), mortal_model.build(8, 1))
                net.mortal_config = {"resnet": {"conv_channels": 8, "num_blocks": 1},
                                     "control": {"version": 4}}
                torch.save(net.state(), origin)
                handed = []

                def baseline(*args, dtype=torch.float32, **kwargs):
                    handed.append(("baseline", dtype))
                    return real_baseline(*args, dtype=dtype, **kwargs)

                def minibatches(*args, dtype=torch.float32, **kwargs):
                    handed.append(("minibatches", dtype))
                    return real_minibatches(*args, dtype=dtype, **kwargs)

                argv = ["trainer", "--resume", str(origin), "--out", str(root / "run"), "--rounds", "1",
                        "--batch", "4", "--epochs", "1", "--games", "1", "--measure-games", "1",
                        "--fixed", "none", *flags]
                with patch.object(sys, "argv", argv), \
                        patch.object(torch, "compile", lambda function, **_kwargs: function), \
                        patch.object(train_combined.selfplay, "play", side_effect=round_of_eight), \
                        patch.object(train_combined.selfplay, "measure",
                                     return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
                        patch.object(ppo_loop, "baseline", side_effect=baseline), \
                        patch.object(ppo_loop, "minibatches", side_effect=minibatches), \
                        contextlib.redirect_stdout(io.StringIO()):
                    train_combined.main()
                self.assertEqual({name for name, _dtype in handed}, {"baseline", "minibatches"})
                self.assertEqual({dtype for _name, dtype in handed}, {wanted})


class StagedMinibatchTests(unittest.TestCase):
    @unittest.skipUnless(CUDA, "the staging is the card's")
    def test_staged_minibatches_are_the_old_ones_however_busy_the_step(self):
        planes = encoded_like(700, seed=12)
        rollout = host_round(planes, "cuda")
        busy = torch.randn(1024, 1024, device="cuda")
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(5)
            reference = list(minibatches_as_they_were(rollout, 64))
            torch.manual_seed(5)
            kept = []
            with closing(ppo_loop.minibatches(rollout, 64, dtype=dtype)) as staged:
                for picks, dense in staged:
                    # The step's stream falls behind, then reads the minibatch,
                    # which is let go before it has: the staging stream must
                    # neither be read early nor reuse that memory too soon.
                    for _ in range(20):
                        busy = torch.tanh(busy @ busy)
                    kept.append((picks.clone(), dense.clone()))
                    del picks, dense
            self.assertEqual(len(kept), len(reference))
            self.assertEqual(len(kept), 700 // 64)
            for (picks, dense), (want_picks, want_dense) in zip(kept, reference):
                same_bits(self, picks, want_picks)
                same_bits(self, dense, want_dense.to(dtype))

    @unittest.skipUnless(CUDA, "the staging is the card's")
    def test_a_minibatch_still_being_staged_is_waited_for(self):
        """The staging stream is slowed, a spin on the card before each
        minibatch, and the step reads every minibatch the moment it has
        it: what it reads is still the whole minibatch."""
        planes = encoded_like(300, seed=17)
        rollout = host_round(planes, "cuda")
        dense = Planes.dense

        def slowly(self, device, dtype=torch.float32, pinned=False):
            torch.cuda._sleep(20_000_000)
            return dense(self, device, dtype, pinned)

        torch.manual_seed(7)
        reference = list(minibatches_as_they_were(rollout, 50))
        torch.manual_seed(7)
        kept = []
        with patch.object(Planes, "dense", slowly):
            with closing(ppo_loop.minibatches(rollout, 50, dtype=torch.bfloat16)) as staged:
                for picks, planes_now in staged:
                    kept.append((picks.clone(), planes_now.clone()))
        self.assertEqual(len(kept), len(reference))
        for (picks, planes_now), (want_picks, want_planes) in zip(kept, reference):
            same_bits(self, picks, want_picks)
            same_bits(self, planes_now, want_planes.to(torch.bfloat16))

    @unittest.skipUnless(CUDA, "the staging is the card's")
    def test_every_epoch_stages_on_the_same_stream_and_memory(self):
        """A stream of its own each epoch stranded the last epoch's memory,
        which the allocator keeps a stream at a time: the card filled up and
        every allocation stalled to free it. One stream a device keeps the
        memory the card holds flat from epoch to epoch."""
        self.assertIs(ppo_loop.staging_stream("cuda"),
                      ppo_loop.staging_stream(torch.device("cuda", torch.cuda.current_device())))
        rollout = host_round(encoded_like(256, seed=18), "cuda")
        held = []
        for _epoch in range(6):
            with closing(ppo_loop.minibatches(rollout, 64, dtype=torch.bfloat16)) as staged:
                for _picks, _planes in staged:
                    pass
            torch.cuda.synchronize()
            held.append(torch.cuda.memory_reserved())
        self.assertEqual(held[-1], held[1])

    @unittest.skipUnless(CUDA, "the staging is the card's")
    def test_stopping_early_leaves_no_worker_behind(self):
        rollout = host_round(encoded_like(600, seed=13), "cuda")
        before = {thread.ident for thread in threading.enumerate() if thread.name == "mahjong-prefetch"}
        with closing(ppo_loop.minibatches(rollout, 32, dtype=torch.bfloat16)) as staged:
            for _picks, _dense in staged:
                break
        torch.cuda.synchronize()
        after = {thread.ident for thread in threading.enumerate() if thread.name == "mahjong-prefetch"}
        self.assertEqual(after, before)

    def test_off_the_card_the_minibatches_are_the_old_ones(self):
        rollout = host_round(encoded_like(300, seed=14), "cpu")
        torch.manual_seed(6)
        reference = list(minibatches_as_they_were(rollout, 50))
        torch.manual_seed(6)
        with closing(ppo_loop.minibatches(rollout, 50)) as staged:
            got = list(staged)
        self.assertEqual(len(got), len(reference))
        for (picks, dense), (want_picks, want_dense) in zip(got, reference):
            same_bits(self, picks, want_picks)
            same_bits(self, dense, want_dense)


class CheckedOnceTests(unittest.TestCase):
    def test_a_round_is_trusted_once_its_value_pass_has_checked_it(self):
        planes = encoded_like(40, seed=15)
        rollout = host_round(planes, "cpu")
        seen = []

        def value(chunk, dense):
            seen.append(chunk)
            return (dense.sum(dim=(1, 2)),)

        self.assertFalse(rollout.observations.trusted)
        ppo_loop.baseline(rollout, 16, value)
        self.assertEqual([chunk.start for chunk in seen], [0, 16, 32])
        self.assertTrue(rollout.observations.trusted)
        # The same arrays, not a copy; the round as self-play left it is untouched.
        self.assertIs(rollout.observations.values, planes.values)
        self.assertFalse(planes.trusted)
        # The minibatches after it skip the value check, and only that.
        self.assertTrue(rollout.observations.rows([3, 1]).trusted)

    def test_a_value_that_is_not_a_number_is_still_refused_in_the_pass(self):
        planes = encoded_like(40, seed=16)
        planes.values[planes.indptr[37]] = np.nan
        rollout = host_round(planes, "cpu")
        with self.assertRaisesRegex(ValueError, "finite"):
            ppo_loop.baseline(rollout, 16, lambda chunk, dense: (dense.sum(dim=(1, 2)),))
        self.assertFalse(rollout.observations.trusted)


class ReceiveTests(unittest.TestCase):
    def test_each_item_is_received_in_the_consuming_thread_in_order(self):
        received = []

        def receive(item):
            received.append((item, threading.get_ident()))
            return item * 10

        with Prefetcher(range(6), lambda x: x + 1, depth=2, receive=receive) as stream:
            time.sleep(0.05)
            self.assertEqual(received, [])  # prepared ahead, not yet handed over
            self.assertEqual(list(stream), [10, 20, 30, 40, 50, 60])
        self.assertEqual([item for item, _thread in received], [1, 2, 3, 4, 5, 6])
        self.assertEqual({thread for _item, thread in received}, {threading.get_ident()})
        self.assertFalse(stream.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
