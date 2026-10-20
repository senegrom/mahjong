"""In a generation that holds Mortal still, Mortal's vector of every
decision is the same function of the same planes all generation, so a
trainer asked to (`--reuse-phi`) keeps it from the pass that values the
round before it is learned and learns from it instead of running Mortal on
every minibatch. Given the vector the forward answers as it would have; the
pass keeps each decision's own, in record order, beside the values and the
drift guard's probabilities it always gave; learned from them the loss is
the recomputed one but for the batches they were worked out in; they are
kept only on a card with room for them, and only where a pass is made; and
whenever Mortal trains they are neither kept nor read."""

import contextlib
import copy
import functools
import io
import json
from pathlib import Path
import sys
import tempfile
from contextlib import closing
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import combined, policy_inference, ppo_loop, selfplay, train_combined, zoo
from neural.observe import Planes, pad_rows
from neural.tests import test_mixed_tables
from neural.tests.test_baseline_from_play import Taught, varied
from neural.tests.test_bfloat16_players import fusion
from neural.tests.test_deciding_forward import positions, same_bits

CUDA = torch.cuda.is_available()


class KnownVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.positions = positions(steps=12)

    def test_given_its_vector_the_forward_answers_as_without_and_runs_no_mortal(self):
        for device in ("cpu", "cuda") if CUDA else ("cpu",):
            torch.manual_seed(5)
            net = fusion().to(device).eval()
            ran = []
            net.mortal.brain.register_forward_hook(lambda *_args: ran.append(1))
            for sparse, allowed in self.positions[::3]:
                planes, legal = sparse.dense(device), torch.from_numpy(allowed).to(device)
                with self.subTest(device=device, rows=len(allowed)), torch.no_grad(), \
                        torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    want = net.everything(planes, legal)
                    phi = net.mortal.features(planes)
                    ran.clear()
                    got = net.everything(planes, legal, phi)
                    self.assertEqual(ran, [])
                    for name, one, two in zip(("logits", "value", "hands"), got, want):
                        same_bits(self, one, two, name)

    def test_given_its_vector_a_frozen_mortal_learns_the_same(self):
        """A step in a generation that holds Mortal still: the loss and
        every gradient are the same, given Mortal's vector or not."""
        for device in ("cpu", "cuda") if CUDA else ("cpu",):
            for mode in ("mortal", "mortal+head"):
                torch.manual_seed(7)
                net = fusion().to(device)
                net.set_mode(mode)
                net.train()
                sparse, allowed = self.positions[4]
                planes, legal = sparse.dense(device), torch.from_numpy(allowed).to(device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    phi = net.mortal.features(planes)
                runs = []
                deterministic = torch.backends.cudnn.deterministic
                torch.backends.cudnn.deterministic = True
                try:
                    for given in (None, phi):
                        net.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                            logits, value, guessed = (net.everything(planes, legal) if given is None
                                                      else net.everything(planes, legal, given))
                        loss = (torch.log_softmax(logits.float(), dim=1).masked_fill(~legal, 0).sum()
                                + value.float().square().sum() + guessed.float().square().sum())
                        loss.backward()
                        runs.append((loss.detach(), {name: p.grad.clone() for name, p in net.named_parameters()
                                                     if p.grad is not None}))
                finally:
                    torch.backends.cudnn.deterministic = deterministic
                (loss, grads), (loss_given, grads_given) = runs
                with self.subTest(device=device, mode=mode):
                    same_bits(self, loss_given, loss)
                    self.assertEqual(set(grads_given), set(grads))
                    self.assertFalse(any(name.startswith("mortal.") for name in grads))
                    self.assertGreater(len(grads), 20)
                    for name in grads:
                        same_bits(self, grads_given[name], grads[name], name)


def switches(**asked) -> SimpleNamespace:
    """The trainer's switches the pass reads, as the command line leaves
    them unless `asked` says otherwise."""
    return SimpleNamespace(**{"baseline_from_play": False, "check_baseline": False, **asked})


def keeper(device: str, spare: int | None = None):
    """What the trainer has the pass keep Mortal's vectors in (see
    `train_combined.value_round`)."""
    return functools.partial(ppo_loop.kept_on_card, device=device, spare=spare)


class PassTests(unittest.TestCase):
    """The pass that values a round before it is learned, as the trainer
    makes it (`train_combined.value_round`), on a round with many reaches,
    each recorded twice: once for the reach and once for its tile."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        cls.net = varied(fusion()).eval()
        cls.batch = selfplay.play(Taught(cls.net), games=2, seed=53, device="cpu")

    def test_the_pass_keeps_each_decisions_vector_in_record_order(self):
        """Each row Mortal's own vector of its planes, as Mortal works it
        out in the pass's chunk, bit for bit; and beside them the values and
        the drift guard's probabilities the pass gave when it asked
        `everything`, bit for bit too."""
        batch, net, rows = self.batch, self.net, 64
        self.assertGreater(int((batch.actions == zoo.MORTAL_RIICHI).sum()), 5)
        rollout = ppo_loop.on_device(batch, "cpu")
        guess, _seconds, said, begun, vectors = train_combined.value_round(
            net.decision, rollout, batch, rows, switches(), torch.float32, False, lagged=True,
            keep=keeper("cpu"))
        self.assertEqual(said, {})
        self.assertEqual(vectors.shape, (batch.decisions, 1024))
        self.assertEqual(vectors.dtype, torch.float32)
        before = torch.empty(batch.decisions)

        def asked_everything(chunk, planes):
            mask = pad_rows(rollout.legal[chunk], rows, True)
            logits, value, _guessed = net.everything(planes, mask)
            taken = rollout.actions[chunk]
            before[chunk] = torch.distributions.Categorical(
                logits=logits[: len(taken)].float(), validate_args=False).log_prob(taken)
            return (value,)

        (want,) = ppo_loop.baseline(ppo_loop.on_device(batch, "cpu"), rows, asked_everything)
        same_bits(self, guess, want, "values")
        same_bits(self, begun, before, "the guard's probabilities")
        with torch.no_grad():
            for start in range(0, batch.decisions, rows):
                count = min(rows, batch.decisions - start)
                planes = pad_rows(batch.observations.slice(start, start + rows).dense("cpu"), rows)
                same_bits(self, vectors[start:start + count], net.mortal.features(planes)[:count], f"row {start}")

    def test_none_are_kept_without_room_or_without_a_pass(self):
        batch, net = self.batch, self.net
        asked = []

        def no_room(shape, dtype):
            asked.append((shape, dtype))
            return None

        guess, _seconds, _said, begun, vectors = train_combined.value_round(
            net.decision, ppo_loop.on_device(batch, "cpu"), batch, 64, switches(), torch.float32, False,
            lagged=False, keep=no_room)
        # Asked once, in the shape and the precision the pass gives them.
        self.assertEqual(asked, [((batch.decisions, 1024), torch.float32)])
        self.assertIsNone(vectors)
        self.assertIsNone(begun)
        self.assertEqual(guess.shape, (batch.decisions,))
        # Valued from play, as --baseline-from-play values a round the
        # learner played itself: no pass, and nothing asked to keep them.
        asked.clear()
        guess, _seconds, said, _begun, vectors = train_combined.value_round(
            net.decision, ppo_loop.on_device(batch, "cpu"), batch, 64, switches(baseline_from_play=True),
            torch.float32, False, lagged=False, keep=no_room)
        self.assertIs(said["baseline_from_play"], True)
        self.assertTrue(torch.equal(guess, batch.values))
        self.assertEqual((asked, vectors), ([], None))
        # Checked against the pass, the pass keeps them.
        _guess, _seconds, said, _begun, vectors = train_combined.value_round(
            net.decision, ppo_loop.on_device(batch, "cpu"), batch, 64,
            switches(baseline_from_play=True, check_baseline=True), torch.float32, False, lagged=False,
            keep=keeper("cpu"))
        self.assertIn("baseline_difference", said)
        self.assertEqual(vectors.shape, (batch.decisions, 1024))

    @unittest.skipUnless(CUDA, "mixed precision is the card's")
    def test_on_the_card_the_vectors_are_the_steps_in_bfloat16(self):
        """The pass's vectors, worked out in its chunks in record order,
        against Mortal's as the learning step works them out on minibatches
        drawn as the trainer draws them, in a generation that holds Mortal
        still: most of them bit for bit."""
        torch.manual_seed(13)
        net = varied(fusion(32, 3)).cuda().eval()
        net.set_mode("mortal")
        batch = selfplay.play(Taught(net), games=1, seed=59, device="cuda", amp=True)
        rollout = ppo_loop.on_device(batch, "cuda", card=False)
        _guess, _seconds, _said, _begun, vectors = train_combined.value_round(
            net.decision, rollout, batch, 64, switches(), torch.bfloat16, True, lagged=False,
            keep=keeper("cuda", spare=0))
        self.assertEqual((vectors.dtype, vectors.device.type), (torch.bfloat16, "cuda"))
        net.train()
        entries = identical = 0
        with closing(ppo_loop.minibatches(rollout, 64, dtype=torch.bfloat16)) as staged:
            for picks, planes in staged:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    want = net.mortal.features(planes)
                got = vectors[picks]
                self.assertEqual(want.dtype, torch.bfloat16)
                # Where the batches differ the sums beneath may be added in
                # another order: a step or two of bfloat16 at an entry's
                # own size, or at the vector's usual size near nought.
                usual = float(want.float().abs().mean())
                torch.testing.assert_close(got.float(), want.float(), rtol=2.0 ** -6, atol=usual * 2.0 ** -6)
                identical += int((got.view(torch.int16) == want.view(torch.int16)).sum())
                entries += want.numel()
        self.assertGreater(entries, 0)
        self.assertGreater(identical, entries // 2)


def ppo_loss(net, planes, legal, rollout, picks, held, advantages, phi=None):
    """The learning step's loss as `train_combined` makes it, without the
    leash or exploration."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=planes.is_cuda):
        logits, value, guessed = (net.everything(planes, legal) if phi is None
                                  else net.everything(planes, legal, phi))
    logits, value = logits.float(), value.float()
    hands_loss, _covered = ppo_loop.hands_loss_of(guessed.float(), held[picks])
    distribution = torch.distributions.Categorical(logits=logits, validate_args=False)
    log_prob = distribution.log_prob(rollout.actions[picks])
    policy_loss, _ratio, _clipped = ppo_loop.clipped_policy_loss(
        log_prob, rollout.old_log_probs[picks], advantages[picks], 0.2)
    value_loss = torch.nn.functional.mse_loss(value, rollout.returns[picks])
    return policy_loss + 0.5 * value_loss + hands_loss - 0.0005 * distribution.entropy().mean()


class SameLossTests(unittest.TestCase):
    def check(self, device):
        """A round played and valued as `train_combined` plays and values
        one in a generation that holds Mortal still, the pass keeping
        Mortal's vectors, then the step's loss and gradients on its
        minibatches, from the pass's vectors and recomputed."""
        torch.manual_seed(17)
        net = varied(fusion(32, 3)).to(device)
        net.set_mode("mortal")
        net.eval()
        amp = device == "cuda"
        dtype = torch.bfloat16 if amp else torch.float32
        player = Taught(policy_inference.precast(copy.deepcopy(net)).requires_grad_(False) if amp else net)
        batch = selfplay.play(player, games=1, seed=61, device=device, amp=amp)
        rollout = ppo_loop.on_device(batch, device)
        held = batch.held.to(device)
        _guess, _seconds, _said, _begun, phi = train_combined.value_round(
            net.decision, rollout, batch, 128, switches(), dtype, amp, lagged=False,
            # Kept on the card whatever this desktop's card has to spare.
            keep=keeper(device, spare=0))
        self.assertIsNotNone(phi)
        advantages, _spread = ppo_loop.standardised(rollout.returns, torch.zeros_like(rollout.returns))
        net.train()
        gaps, angles = [], []
        torch.manual_seed(19)
        with closing(ppo_loop.minibatches(rollout, 128, dtype=dtype)) as staged:
            for picks, planes in staged:
                losses, grads = [], []
                for given in (None, phi[picks]):
                    net.zero_grad(set_to_none=True)
                    loss = ppo_loss(net, planes, rollout.legal[picks], rollout, picks, held, advantages, given)
                    loss.backward()
                    losses.append(float(loss.detach()))
                    grads.append(torch.cat([p.grad.reshape(-1) for p in net.parameters() if p.grad is not None]))
                gaps.append(abs(losses[1] - losses[0]) / max(abs(losses[0]), 1e-3))
                angles.append(float(torch.nn.functional.cosine_similarity(grads[0], grads[1], dim=0)))
        return gaps, angles

    def test_on_the_processor_the_loss_is_the_recomputed_one(self):
        torch.set_num_threads(2)
        gaps, angles = self.check("cpu")
        self.assertGreater(len(gaps), 3)
        self.assertLess(max(gaps), 1e-5)
        self.assertGreater(min(angles), 1 - 1e-6)

    @unittest.skipUnless(CUDA, "mixed precision is the card's")
    def test_on_the_card_the_loss_is_the_recomputed_one_within_bfloat16(self):
        gaps, angles = self.check("cuda")
        self.assertGreater(len(gaps), 3)
        # bfloat16 keeps eight significant bits: a loss within a few of
        # its steps, and gradients pointing the same way.
        self.assertLess(max(gaps), 2.0 ** -6)
        self.assertGreater(min(angles), 0.999)


def fake_round(values: bool = False):
    """Eight decisions, each with planes of its own, so that a vector out
    of its row shows; and the values play kept, when asked."""
    n = 8
    planes = Planes(np.arange(n + 1, dtype=np.int64), np.arange(n, dtype=np.uint16) * 101,
                    np.ones(n, dtype=np.float16))
    return SimpleNamespace(
        decisions=n, observations=planes, legal=torch.ones(n, 46, dtype=torch.bool),
        actions=torch.zeros(n, dtype=torch.int64), returns=torch.linspace(-1, 1, n),
        log_probs=torch.zeros(n), held=torch.full((n, 3, 34), 1 / 34),
        games=1, hands=1, timing={}, values=torch.linspace(-0.5, 0.5, n) if values else None)


class TrainerTests(unittest.TestCase):
    def train(self, flags, fixed, room=True, values=False):
        """One generation of `train_combined` on a fake round, on the
        processor: which learning forwards were given Mortal's vectors,
        each checked against Mortal's own of the minibatch's rows; what the
        pass asked to keep them in; and the record."""
        torch.set_num_threads(1)
        given, asked = [], []
        everything = combined.Combined.everything

        def watched(self, planes, legal, phi=None):
            if torch.is_grad_enabled():
                given.append(phi is not None)
                if phi is not None:
                    # Each row's own, in the order the minibatch drew them.
                    with torch.no_grad():
                        torch.testing.assert_close(phi, self.mortal.features(planes), rtol=1e-5, atol=1e-6)
            return everything(self, planes, legal, phi)

        def kept_on_card(shape, dtype, device, spare=None):
            asked.append((shape, dtype, spare))
            return torch.empty(shape, dtype=dtype) if room else None

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            origin, out = root / "source.pt", root / "run"
            torch.save(fusion(8, 1).state(), origin)
            argv = ["trainer", "--resume", str(origin), "--out", str(out), "--rounds", "1",
                    "--batch", "4", "--epochs", "1", "--games", "1", "--measure-games", "1",
                    "--fixed", fixed, *flags]
            with patch.object(sys, "argv", argv), \
                    patch.object(torch.cuda, "is_available", return_value=False), \
                    patch.object(train_combined.selfplay, "play",
                                 side_effect=lambda _player, **_kwargs: fake_round(values)), \
                    patch.object(train_combined.selfplay, "measure",
                                 return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
                    patch.object(combined.Combined, "everything", watched), \
                    patch.object(ppo_loop, "kept_on_card", side_effect=kept_on_card), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_combined.main()
            record = json.loads((out / "log.jsonl").read_text().splitlines()[0])
        return given, asked, record

    def test_reused_only_in_generations_that_hold_mortal_still(self):
        for fixed in combined.Combined.MODES:
            with self.subTest(fixed=fixed):
                given, asked, record = self.train(["--reuse-phi"], fixed)
                still = "mortal" in fixed.split("+")
                self.assertEqual(given, [still, still])
                # Asked for once, by the pass, in the shape and precision it
                # gives them, with the learning step's room to spare.
                self.assertEqual(asked, [((8, 1024), torch.float32, None)] if still else [])
                self.assertIs(record["reused_phi"], still)

    def test_not_without_the_switch_nor_room_on_the_card_nor_a_pass(self):
        given, asked, record = self.train([], "mortal")
        self.assertEqual((given, asked), ([False, False], []))
        self.assertNotIn("reused_phi", record)
        given, asked, record = self.train(["--reuse-phi"], "mortal", room=False)
        self.assertEqual((given, len(asked)), ([False, False], 1))
        self.assertIs(record["reused_phi"], False)
        # Valued from play's values, the round gets no pass to keep them.
        given, asked, record = self.train(["--reuse-phi", "--baseline-from-play"], "mortal", values=True)
        self.assertEqual((given, asked), ([False, False], []))
        self.assertIs(record["baseline_from_play"], True)
        self.assertIs(record["reused_phi"], False)


class RoomTests(unittest.TestCase):
    """Room on the card for Mortal's vectors (`ppo_loop.room_on_card`): what
    the driver has free and what this process's allocator holds without
    using it, past what the learning step, and beside a round played ahead
    the round, keep to spare."""

    GB = 1 << 30
    VECTORS = int(3.2 * GB)
    SPARE = 20 * GB

    def room(self, free: int, reserved: int, allocated: int, wanted: int = VECTORS) -> bool:
        with patch.object(torch.cuda, "mem_get_info", return_value=(free, 46 * self.GB)), \
                patch.object(torch.cuda, "memory_reserved", return_value=reserved), \
                patch.object(torch.cuda, "memory_allocated", return_value=allocated):
            return ppo_loop.room_on_card(wanted, "cuda", self.SPARE)

    def test_what_the_allocator_holds_unused_is_room(self):
        """As a005's card was beside a round played ahead: the driver has
        22 GB free, and counts as used 23 GB the allocator holds, of which
        it is using 7."""
        self.assertTrue(self.room(22 * self.GB, 23 * self.GB, 7 * self.GB))
        # Read from the driver alone, as the rule was, there was no room.
        self.assertFalse(self.room(22 * self.GB, 0, 0))

    def test_what_the_card_is_using_is_not(self):
        self.assertFalse(self.room(22 * self.GB, 23 * self.GB, 23 * self.GB))

    def test_room_is_the_spare_left_after_and_not_a_byte_less(self):
        free, unused = 10 * self.GB, 13 * self.GB + 7
        left = free + unused - self.SPARE
        self.assertTrue(self.room(free, unused + self.GB, self.GB, wanted=left))
        self.assertFalse(self.room(free, unused + self.GB, self.GB, wanted=left + 1))

    def test_on_the_processor_a_tensor_is_kept_without_asking_the_card(self):
        with patch.object(torch.cuda, "mem_get_info", side_effect=AssertionError("asked the card")):
            kept = ppo_loop.kept_on_card((5, 3), torch.bfloat16, "cpu", spare=1 << 60)
        self.assertEqual((kept.shape, kept.dtype, kept.device.type), ((5, 3), torch.bfloat16, "cpu"))


class EndToEndTests(unittest.TestCase):
    def test_a_generation_learns_the_same_from_the_vectors_the_pass_kept(self):
        """One real generation that holds Mortal still, on the processor,
        with and without the switch: the same play, the same losses to the
        record's last digit, Mortal untouched, and the weights moved the
        same way. Not asked to be the same to the bit: the pass works each
        vector out in a batch of other rows than the step's, and where that
        moves one in float32's last bit, Adam, dividing each gradient by its
        own size, makes it visible wherever a gradient is nearly nought."""
        torch.set_num_threads(2)
        runs = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            origin = root / "source.pt"
            torch.manual_seed(23)
            torch.save(varied(fusion(16, 2)).state(), origin)
            start = torch.load(origin, map_location="cpu", weights_only=False)
            for flags in ([], ["--reuse-phi"]):
                out = root / ("reused" if flags else "recomputed")
                argv = ["trainer", "--resume", str(origin), "--out", str(out), "--rounds", "1",
                        "--batch", "64", "--epochs", "1", "--games", "1", "--measure-games", "1",
                        "--measure-every", "5", "--fixed", "mortal", "--seed", "5", *flags]
                with patch.object(sys, "argv", argv), \
                        patch.object(torch.cuda, "is_available", return_value=False), \
                        patch.object(train_combined.selfplay, "measure",
                                     return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    train_combined.main()
                record = json.loads((out / "log.jsonl").read_text().splitlines()[0])
                state = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
                runs.append((record, state))
        (plain, plain_state), (reused, reused_state) = runs
        self.assertIs(reused["reused_phi"], True)
        self.assertGreater(plain["optimizer_updates"], 3)
        for key in ("decisions", "hands", "optimizer_updates", "mean_return", "return_variance"):
            self.assertEqual(reused[key], plain[key], key)
        for key in ("policy_loss", "value_loss", "entropy", "hands_loss", "approx_kl", "value_error"):
            self.assertAlmostEqual(reused[key], plain[key], places=4, msg=key)
        for part in ("mortal", "current_dqn"):
            for name, want in start[part].items():
                # Held still: every weight and statistic as it started.
                self.assertTrue(torch.equal(reused_state[part][name], want), name)
                self.assertTrue(torch.equal(plain_state[part][name], want), name)
        for part in ("combined", "model"):
            moves = [torch.cat([(state[part][name] - want).reshape(-1)
                                for name, want in start[part].items() if want.is_floating_point()])
                     for state in (plain_state, reused_state)]
            self.assertGreater(float(moves[0].norm()), 0.0, part)
            self.assertGreater(float(torch.nn.functional.cosine_similarity(*moves, dim=0)), 0.99, part)


class SwitchTests(unittest.TestCase):
    def test_the_trainer_and_the_cloud_leave_it_off_unless_asked(self):
        with patch.object(sys, "argv", ["trainer"]):
            self.assertFalse(train_combined.parse_args().reuse_phi)
        with patch.object(sys, "argv", ["trainer", "--reuse-phi"]):
            self.assertTrue(train_combined.parse_args().reuse_phi)
        launch = test_mixed_tables.CloudLaunchTests.launch
        self.assertNotIn("--reuse-phi", launch(self, "train_combined"))
        self.assertIn("--reuse-phi", launch(self, "train_combined", reuse_phi=True))


if __name__ == "__main__":
    unittest.main()
