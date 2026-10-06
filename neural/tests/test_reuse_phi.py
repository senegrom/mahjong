"""In a generation that holds Mortal still, Mortal's vector of every
decision is the same function of the same planes all generation, so a
trainer asked to (`--reuse-phi`) keeps it from play and learns from it
instead of running Mortal on every minibatch. Given the vector the forward
answers as it would have; learned from play's vectors the loss is the
recomputed one but for the batches they were worked out in; and whenever
Mortal trains the vectors are neither kept nor read."""

import contextlib
import copy
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
from neural.observe import Planes
from neural.tests import test_mixed_tables
from neural.tests.test_baseline_from_play import Taught, varied
from neural.tests.test_bfloat16_players import fusion
from neural.tests.test_deciding_forward import positions, same_bits

CUDA = torch.cuda.is_available()


def chunks(planes: Planes, device: str, dtype=torch.float32, size: int = 256):
    for start in range(0, len(planes), size):
        yield start, planes.slice(start, start + size).dense(device, dtype)


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


class RoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_a_round_keeps_each_decisions_vector_when_asked(self):
        """In record order, a riichi's tile included. A round played
        without being asked keeps none (see test_baseline_from_play)."""
        torch.manual_seed(11)
        net = varied(fusion())
        player = Taught(net)
        player.keep_phi = True
        batch = selfplay.play(player, games=2, seed=53, device="cpu")
        self.assertEqual(batch.phi.shape, (batch.decisions, 1024))
        self.assertEqual(batch.phi.dtype, torch.float32)
        self.assertGreater(int((batch.actions == zoo.MORTAL_RIICHI).sum()), 5)
        with torch.no_grad():
            for start, planes in chunks(batch.observations, "cpu"):
                want = net.mortal.features(planes)
                torch.testing.assert_close(batch.phi[start:start + len(want)], want, rtol=1e-5, atol=1e-5)
        # The joined player keeps them in its own self-play too.
        own = selfplay.play(net, games=1, seed=55, device="cpu")
        self.assertEqual(own.phi.shape, (own.decisions, 1024))

    @unittest.skipUnless(CUDA, "mixed precision is the card's")
    def test_on_the_card_the_copys_vectors_are_the_learners_in_bfloat16(self):
        torch.manual_seed(13)
        net = varied(fusion(32, 3)).cuda().eval()
        actor = policy_inference.precast(copy.deepcopy(net)).requires_grad_(False)
        player = Taught(actor)
        player.keep_phi = True
        batch = selfplay.play(player, games=1, seed=59, device="cuda", amp=True)
        self.assertEqual(batch.phi.dtype, torch.bfloat16)
        self.assertEqual(batch.phi.device.type, "cpu")
        self.assertEqual(len(batch.phi), batch.decisions)
        entries = identical = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start, planes in chunks(batch.observations, "cuda", torch.bfloat16):
                want = net.mortal.features(planes)
                got = batch.phi[start:start + len(want)].cuda()
                self.assertEqual(want.dtype, torch.bfloat16)
                # Where the batches differ the sums beneath may be added in
                # another order: a step or two of bfloat16 at an entry's
                # own size, or at the vector's usual size near nought.
                usual = float(want.float().abs().mean())
                torch.testing.assert_close(got.float(), want.float(), rtol=2.0 ** -6, atol=usual * 2.0 ** -6)
                identical += int((got.view(torch.int16) == want.view(torch.int16)).sum())
                entries += want.numel()
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
        """A round played as `train_combined` plays it in a generation that
        holds Mortal still, then the step's loss and gradients on its
        minibatches, from play's vectors and recomputed."""
        torch.manual_seed(17)
        net = varied(fusion(32, 3)).to(device)
        net.set_mode("mortal")
        net.eval()
        amp = device == "cuda"
        player = Taught(policy_inference.precast(copy.deepcopy(net)).requires_grad_(False) if amp else net)
        player.keep_phi = True
        batch = selfplay.play(player, games=1, seed=61, device=device, amp=amp)
        rollout = ppo_loop.on_device(batch, device)
        held = batch.held.to(device)
        # Kept on the card whatever this desktop's card has to spare.
        phi = ppo_loop.kept_on_card(batch.phi, device, spare=0)
        self.assertIsNotNone(phi)
        advantages, _spread = ppo_loop.standardised(rollout.returns, torch.zeros_like(rollout.returns))
        net.train()
        gaps, angles = [], []
        torch.manual_seed(19)
        with closing(ppo_loop.minibatches(rollout, 128, dtype=torch.bfloat16 if amp else torch.float32)) as staged:
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


def fake_round(phi: bool):
    n = 8
    planes = Planes(np.arange(n + 1, dtype=np.int64), np.zeros(n, dtype=np.uint16),
                    np.ones(n, dtype=np.float16))
    return SimpleNamespace(
        decisions=n, observations=planes, legal=torch.ones(n, 46, dtype=torch.bool),
        actions=torch.zeros(n, dtype=torch.int64), returns=torch.linspace(-1, 1, n),
        log_probs=torch.zeros(n), held=torch.full((n, 3, 34), 1 / 34),
        games=1, hands=1, timing={}, values=None,
        phi=torch.randn(n, 1024) if phi else None)


class TrainerTests(unittest.TestCase):
    def train(self, flags, fixed, room=True):
        """One generation of `train_combined` on a fake round, on the
        processor: whether the player was asked to keep the vectors, which
        learning forwards were given them, and the record."""
        torch.set_num_threads(1)
        kept, given = [], []
        everything = combined.Combined.everything

        def play(player, **_kwargs):
            kept.append(player.keep_phi)
            return fake_round(player.keep_phi)

        def watched(self, planes, legal, phi=None):
            if torch.is_grad_enabled():
                given.append(phi is not None)
            return everything(self, planes, legal, phi)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            origin, out = root / "source.pt", root / "run"
            torch.save(fusion(8, 1).state(), origin)
            argv = ["trainer", "--resume", str(origin), "--out", str(out), "--rounds", "1",
                    "--batch", "4", "--epochs", "1", "--games", "1", "--measure-games", "1",
                    "--fixed", fixed, *flags]
            with patch.object(sys, "argv", argv), \
                    patch.object(torch.cuda, "is_available", return_value=False), \
                    patch.object(train_combined.selfplay, "play", side_effect=play), \
                    patch.object(train_combined.selfplay, "measure",
                                 return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
                    patch.object(combined.Combined, "everything", watched), \
                    patch.object(ppo_loop, "kept_on_card",
                                 side_effect=lambda tensor, device: tensor if room else None), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_combined.main()
            record = json.loads((out / "log.jsonl").read_text().splitlines()[0])
        return kept, given, record

    def test_reused_only_in_generations_that_hold_mortal_still(self):
        for fixed in combined.Combined.MODES:
            with self.subTest(fixed=fixed):
                kept, given, record = self.train(["--reuse-phi"], fixed)
                still = "mortal" in fixed.split("+")
                self.assertEqual(kept, [still])
                self.assertEqual(given, [still, still])
                self.assertIs(record["reused_phi"], still)

    def test_not_without_the_switch_nor_without_room_on_the_card(self):
        kept, given, record = self.train([], "mortal")
        self.assertEqual((kept, given), ([False], [False, False]))
        self.assertNotIn("reused_phi", record)
        kept, given, record = self.train(["--reuse-phi"], "mortal", room=False)
        self.assertEqual((kept, given), ([True], [False, False]))
        self.assertIs(record["reused_phi"], False)


class EndToEndTests(unittest.TestCase):
    def test_a_generation_learns_the_same_from_plays_vectors(self):
        """One real generation that holds Mortal still, on the processor,
        with and without the switch: the same play, the same losses to the
        record's last digit, Mortal untouched, and the weights moved the
        same way. Not to the bit: play worked the vectors out a row at a
        time and the step 64 at a time, a difference in float32's last bit
        that Adam, dividing each gradient by its own size, makes visible
        wherever a gradient is nearly nought."""
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
        self.assertFalse(combined.Combined.keep_phi)
        launch = test_mixed_tables.CloudLaunchTests.launch
        self.assertNotIn("--reuse-phi", launch(self, "train_combined"))
        self.assertIn("--reuse-phi", launch(self, "train_combined", reuse_phi=True))


if __name__ == "__main__":
    unittest.main()
