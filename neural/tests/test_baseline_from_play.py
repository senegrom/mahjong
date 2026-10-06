"""The values a learner's head gave its decisions as it played are kept,
one a decision in record order, a riichi's tile included, and a trainer
asked to (`--baseline-from-play`) measures the advantages against them
instead of a pass over the round. They are the pass's values: the same
head on the same weights and planes, with only the batches different.
Without the switch the pass is the baseline as it always was, and with
`--check-baseline` both are made and their distance recorded."""

import contextlib
import copy
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

from neural import (
    mortal_learner, mortal_model, policy_inference, ppo_loop, selfplay, train_combined, train_mortal, zoo,
)
from neural.observe import Planes, pad_rows
from neural.tests import test_mixed_tables
from neural.tests.test_bfloat16_players import fusion
from neural.tests.test_evaluation_masks import ConflictingViews

CUDA = torch.cuda.is_available()


#: Mortal's move for each of ours: the one it is a meaning of.
TO_MORTAL = np.array([int(np.flatnonzero(zoo.MEANS[:, ours])[0]) for ours in range(zoo.ACTIONS)])


def varied(net):
    """`net` with values that differ from position to position, so a value
    out of its place in the round shows."""
    with torch.no_grad():
        for parameter in net.ours.value.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    return net


class Taught:
    """A joined player that plays the heuristic player's moves: its forward
    is asked, by the very path a learner decides by, and its values and
    vectors recorded, but the logits it chooses from are the heuristic
    player's. A random network seldom holds a closed ready hand, and the
    heuristic player declares riichi often, so a round has many second
    steps. `keep_phi` is passed on to the network."""

    kind = "mortal"

    def __init__(self, net) -> None:
        self.net = net
        self.timing: dict = {}

    @property
    def keep_phi(self) -> bool:
        return getattr(self.net, "keep_phi", False)

    @keep_phi.setter
    def keep_phi(self, value: bool) -> None:
        self.net.keep_phi = value

    def eval(self):
        self.net.eval()
        return self

    @torch.no_grad()
    def decide(self, views, rows, players, legal, greedy=False, explore_share=0.0, wanderer=None):
        taught = np.frombuffer(views.arena.teacher(), dtype=np.uint8)[rows].astype(np.int64)
        # The rows asked a second time, for a riichi's tile, in their order.
        reaching = [i for i, move in enumerate(taught) if zoo.RIICHI_DISCARD <= move < zoo.TSUMO]
        asked = []

        def score(planes, mask):
            # What `Combined.decide` records of the forward: the logits and
            # the value, and Mortal's vector when it keeps it.
            answer = self.net.decision(planes, mask)
            answer = answer if self.keep_phi else answer[:2]
            if not asked:
                wanted = TO_MORTAL[taught]
            else:
                wanted = taught[reaching] - zoo.RIICHI_DISCARD
            asked.append(len(mask))
            logits = torch.full(mask.shape, -20.0, device=mask.device)
            logits[torch.arange(len(mask)), torch.from_numpy(wanted).to(mask.device)] = 20.0
            return (logits.masked_fill(~mask, -torch.inf), *answer[1:])

        return mortal_learner.decide_in_mortal_space(
            score, views, rows, players, legal, greedy, str(next(self.net.parameters()).device),
            self.timing, explore_share=explore_share, wanderer=wanderer)


def passed(net, batch, device, rows=64, amp=False, dtype=torch.float32):
    """The baseline pass over a round as `train_combined` makes it."""
    rollout = ppo_loop.on_device(batch, device)
    net.eval()

    def values_of(chunk, planes):
        mask = pad_rows(rollout.legal[chunk], rows, True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            return (net.everything(planes, mask)[1],)

    (guess,) = ppo_loop.baseline(rollout, rows, values_of, dtype=dtype)
    return guess


class BlankViews:
    """Views whose follower serves empty planes, a row for every player
    asked about, and takes whatever it is told."""

    def __init__(self):
        self.observer = SimpleNamespace(follower=self)

    def tell(self, game, player, event):
        pass

    def sparse_and_masks(self, rows, players, fresh=False):
        n = len(rows)
        return (Planes(np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.uint16),
                       np.zeros(0, dtype=np.float16)), np.ones((n, 46), dtype=bool))

    def encode(self, who):
        n = len(who)
        return (np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.uint16),
                np.zeros(0, dtype=np.float32), np.ones((n, 46), dtype=bool))


class RecordsTests(unittest.TestCase):
    def test_each_decision_keeps_its_value_in_record_order(self):
        """A reach and its tile are two records, each with the value of the
        position it was decided in; a row nothing translates is not
        recorded, nor is its value."""
        legal = np.zeros((3, 78), dtype=bool)
        legal[0, [0, 34, 35]] = True  # a discard or a reach
        legal[2, [5]] = True  # a discard
        calls = []

        def score(planes, mask):
            calls.append(len(mask))
            logits = torch.zeros(mask.shape)
            logits[:, zoo.MORTAL_RIICHI] = 20
            logits[:, 1] = 10
            # Row r of the n-th question is worth 100 n + r.
            value = 100.0 * len(calls) + torch.arange(len(mask), dtype=torch.float32)
            return logits.masked_fill(~mask, -torch.inf), value

        choice, records = mortal_learner.decide_in_mortal_space(
            score, BlankViews(), np.array([0, 1, 2]), np.array([0, 1, 2]), legal,
            greedy=True, device="cpu")
        self.assertEqual(calls, [3, 1])
        self.assertEqual(choice.tolist(), [35, 0, 5])
        self.assertEqual(records.slots.tolist(), [0, 2, 0])
        self.assertEqual(records.actions.tolist(), [zoo.MORTAL_RIICHI, 5, 1])
        self.assertEqual(records.values.dtype, np.float32)
        self.assertEqual(records.values.tolist(), [100.0, 102.0, 200.0])
        self.assertEqual(len(records.values), len(records.actions))

    def test_a_score_of_logits_alone_records_no_values(self):
        legal = np.zeros((1, 78), dtype=bool)
        legal[0, [0, 34, 35]] = True

        def score(planes, mask):
            logits = torch.zeros(mask.shape)
            logits[:, zoo.MORTAL_RIICHI] = 20
            return logits.masked_fill(~mask, -torch.inf)

        _choice, records = mortal_learner.decide_in_mortal_space(
            score, ConflictingViews(), np.array([0]), np.array([0]), legal, greedy=True, device="cpu")
        self.assertEqual(len(records.actions), 2)
        self.assertIsNone(records.values)

    def test_the_mortal_learner_records_its_value_heads_values(self):
        torch.manual_seed(3)
        learner = mortal_learner.MortalLearner(mortal_model.build(8, 1))
        with torch.no_grad():
            learner.value_head.weight.normal_()
        batch = selfplay.play(learner, games=1, seed=12, device="cpu", want_held=False)
        self.assertEqual(batch.values.shape, (batch.decisions,))
        rows = np.arange(0, batch.decisions, 7)
        planes = batch.observations.rows(rows).dense("cpu")
        with torch.no_grad():
            _logits, want = learner.policy(planes, batch.legal[rows])
        torch.testing.assert_close(batch.values[rows], want, rtol=1e-5, atol=1e-5)


class RoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_a_rounds_values_are_the_pass_values_on_the_processor(self):
        """In record order, a riichi's tile included: the pass, on the
        processor in float32, gives each decision the value play kept."""
        torch.manual_seed(31)
        net = varied(fusion(16, 2))
        batch = selfplay.play(Taught(net), games=2, seed=41, device="cpu")
        self.assertEqual(batch.values.shape, (batch.decisions,))
        self.assertEqual(batch.values.dtype, torch.float32)
        # Reaches, each with a second record of its own.
        self.assertGreater(int((batch.actions == zoo.MORTAL_RIICHI).sum()), 5)
        guess = passed(net, batch, "cpu")
        # Values that differ row from row, so one out of its place shows.
        self.assertGreater(float(guess.std()), 0.05)
        torch.testing.assert_close(batch.values, guess, rtol=1e-5, atol=1e-5)
        # The joined player's own self-play keeps them too.
        own = selfplay.play(net, games=1, seed=42, device="cpu")
        self.assertEqual(own.values.shape, (own.decisions,))
        torch.testing.assert_close(own.values, passed(net, own, "cpu"), rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(CUDA, "mixed precision is the card's")
    def test_on_the_card_the_copy_values_as_the_learners_pass(self):
        """Played as `train_combined` plays, by the learner's copy with its
        weights cast once, and valued as it values, by the learner itself
        on bfloat16 planes: the same batch gives the same bits, and a
        round differs only where its batches differ from the pass's."""
        torch.manual_seed(37)
        net = varied(fusion(32, 3)).cuda().eval()
        actor = policy_inference.precast(copy.deepcopy(net)).requires_grad_(False)
        torch.manual_seed(43)
        batch = selfplay.play(Taught(actor), games=2, seed=47, device="cuda", amp=True)
        self.assertGreater(int((batch.actions == zoo.MORTAL_RIICHI).sum()), 5)
        rows = torch.arange(0, batch.decisions, 3)
        planes = batch.observations.rows(rows.numpy())
        legal = batch.legal[rows].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _logits, played, _phi = actor.decision(planes.dense("cuda"), legal)
            _logits, valued, _guessed = net.everything(planes.dense("cuda", torch.bfloat16), legal)
        self.assertTrue(torch.equal(played.view(torch.int32), valued.view(torch.int32)))
        guess = passed(net, batch, "cuda", rows=256, amp=True, dtype=torch.bfloat16)
        gap = (batch.values.cuda() - guess).abs()
        self.assertGreater(float(guess.std()), 0.05)
        # Where the batches differ the kernels may add in another order, and
        # bfloat16 keeps eight significant bits: a value can move by a step
        # or two at its own size, never by the spread between positions.
        scale = guess.abs().clamp(min=1.0)
        self.assertTrue(bool((gap <= scale * 2.0 ** -6).all()), float((gap / scale).max()))
        self.assertLess(float(gap.mean()), float(scale.mean()) * 2.0 ** -6)


def fake_round(values=None):
    """A round of eight decisions with known returns and, when given, the
    values play recorded."""
    n = 8
    planes = Planes(np.arange(n + 1, dtype=np.int64), np.zeros(n, dtype=np.uint16),
                    np.ones(n, dtype=np.float16))
    return SimpleNamespace(
        decisions=n, observations=planes, legal=torch.ones(n, 46, dtype=torch.bool),
        actions=torch.zeros(n, dtype=torch.int64), returns=torch.linspace(-1, 1, n),
        log_probs=torch.zeros(n), held=torch.full((n, 3, 34), 1 / 34),
        games=1, hands=1, timing={}, values=values)


class TrainerTests(unittest.TestCase):
    def train(self, module, flags, values):
        """One generation of `module` on a fake round, on the processor:
        its record, and how many passes over the round it made."""
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            origin, out = root / "source.pt", root / "run"
            if module is train_mortal:
                net = mortal_model.build(8, 1)
                torch.save({**net.state(), "config": {"resnet": {"conv_channels": 8, "num_blocks": 1},
                                                      "control": {"version": 4}}}, origin)
            else:
                torch.save(fusion(8, 1).state(), origin)
            argv = ["trainer", "--mortal" if module is train_mortal else "--resume", str(origin),
                    "--out", str(out), "--rounds", "1", "--batch", "4", "--epochs", "1",
                    "--games", "1", "--measure-games", "1", *flags]
            if module is train_combined:
                argv += ["--fixed", "none"]
            passes = []
            real = ppo_loop.baseline

            def baseline(*args, **kwargs):
                guesses = real(*args, **kwargs)
                passes.append(guesses[0].clone())
                return guesses

            with patch.object(sys, "argv", argv), \
                    patch.object(torch.cuda, "is_available", return_value=False), \
                    patch.object(module.selfplay, "play", side_effect=lambda *_a, **_k: fake_round(values)), \
                    patch.object(module.selfplay, "measure",
                                 return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
                    patch.object(ppo_loop, "baseline", side_effect=baseline), \
                    contextlib.redirect_stdout(io.StringIO()):
                module.main()
            record = json.loads((out / "log.jsonl").read_text().splitlines()[0])
        return record, passes

    def test_without_the_switch_the_pass_is_the_baseline(self):
        values = torch.full((8,), 0.25)
        for module in (train_combined, train_mortal):
            with self.subTest(trainer=module.__name__):
                record, passes = self.train(module, [], values)
                self.assertEqual(len(passes), 1)
                returns = torch.linspace(-1, 1, 8)
                self.assertAlmostEqual(record["value_error"],
                                       round(float(((returns - passes[0]) ** 2).mean()), 4))
                for key in ("baseline_from_play", "baseline_difference", "baseline_difference_mean"):
                    self.assertNotIn(key, record)

    def test_with_it_plays_values_are_the_baseline_and_no_pass_is_made(self):
        values = torch.linspace(-0.5, 0.5, 8)
        returns = torch.linspace(-1, 1, 8)
        for module in (train_combined, train_mortal):
            with self.subTest(trainer=module.__name__):
                record, passes = self.train(module, ["--baseline-from-play"], values)
                self.assertEqual(passes, [])
                self.assertIs(record["baseline_from_play"], True)
                self.assertAlmostEqual(record["value_error"],
                                       round(float(((returns - values) ** 2).mean()), 4))
                self.assertNotIn("baseline_difference", record)

    def test_checked_both_are_made_and_their_distance_recorded(self):
        values = torch.linspace(-0.5, 0.5, 8)
        returns = torch.linspace(-1, 1, 8)
        for module in (train_combined, train_mortal):
            for flags in (["--baseline-from-play", "--check-baseline"], ["--check-baseline"]):
                with self.subTest(trainer=module.__name__, flags=flags):
                    record, passes = self.train(module, flags, values)
                    self.assertEqual(len(passes), 1)
                    gap = (values - passes[0]).abs()
                    self.assertAlmostEqual(record["baseline_difference"], round(float(gap.max()), 6))
                    self.assertAlmostEqual(record["baseline_difference_mean"], round(float(gap.mean()), 6))
                    guess = values if "--baseline-from-play" in flags else passes[0]
                    self.assertAlmostEqual(record["value_error"],
                                           round(float(((returns - guess) ** 2).mean()), 4))
                    self.assertEqual("baseline_from_play" in record, "--baseline-from-play" in flags)

    def test_a_round_without_values_falls_back_to_the_pass(self):
        for module in (train_combined, train_mortal):
            with self.subTest(trainer=module.__name__):
                record, passes = self.train(module, ["--baseline-from-play", "--check-baseline"], None)
                self.assertEqual(len(passes), 1)
                for key in ("baseline_from_play", "baseline_difference", "baseline_difference_mean"):
                    self.assertNotIn(key, record)


class SwitchTests(unittest.TestCase):
    def test_the_trainers_and_the_cloud_leave_them_off_unless_asked(self):
        for module in (train_combined, train_mortal):
            with self.subTest(trainer=module.__name__):
                with patch.object(sys, "argv", ["trainer"]):
                    args = module.parse_args()
                self.assertFalse(args.baseline_from_play)
                self.assertFalse(args.check_baseline)
                with patch.object(sys, "argv", ["trainer", "--baseline-from-play", "--check-baseline"]):
                    args = module.parse_args()
                self.assertTrue(args.baseline_from_play)
                self.assertTrue(args.check_baseline)
        launch = test_mixed_tables.CloudLaunchTests.launch
        for trainer in ("train_mortal", "train_combined"):
            with self.subTest(cloud=trainer):
                command = launch(self, trainer)
                self.assertNotIn("--baseline-from-play", command)
                self.assertNotIn("--check-baseline", command)
                command = launch(self, trainer, baseline_from_play=True)
                self.assertIn("--baseline-from-play", command)
                self.assertNotIn("--check-baseline", command)
                command = launch(self, trainer, check_baseline=True)
                self.assertIn("--check-baseline", command)
                self.assertNotIn("--baseline-from-play", command)


if __name__ == "__main__":
    unittest.main()
