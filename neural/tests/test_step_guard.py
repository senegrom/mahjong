"""The drift guard and the gradient clip decided on the card
(`ppo_control.StepGuard`): every update is launched without waiting, fused
AdamW refuses on the card, through `found_inf`, the first update either
refuses and every one after it, and the host reads each verdict a couple of
updates later. What a generation learns is bit for bit what deciding on the
host at once learns: the weights, Adam's moments and step counts, torch's
generator, the record's sums and the drift's figures, the updates counted
and the minibatch the head's shift is read on; through a stop by the guard
wherever it falls, and through the errors a drift or a gradient that is not
a number raises, with nothing applied in between. Here on the processor,
whose fused AdamW takes `found_inf` as the card's does, and on a card when
there is one."""

import contextlib
import copy
import dataclasses
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import combined, model, mortal_model, ppo_loop, train_combined
from neural.observe import Planes
from neural.ppo_control import NONFINITE_DRIFT, PolicyDrift, StepGuard
from neural.tests.test_bfloat16_players import fusion
from neural.tests.test_learning_planes import encoded_like

CUDA = torch.cuda.is_available()
SEED = 20261020
# Eight minibatches of four an epoch, and two rows left over, which differ
# from one epoch to the next.
ROWS, BATCH, MINIBATCHES = 34, 4, 8
# Far above what the learning moves the policy by in two epochs here, and
# far below what one poisoned row (see `poisoned`) makes of a minibatch.
TARGET = 1.0


def lesson(device: str = "cpu") -> SimpleNamespace:
    """A small joined player and its fused AdamW, one update in so that
    every parameter has Adam's state for a refused update to leave alone,
    and a round for it to learn: planes of a step's shape, the moves made
    and their probabilities as the player now stands, returns, and what the
    opponents held."""
    torch.manual_seed(SEED)
    net = fusion(8, 1).to(device)
    net.set_mode("none")
    args = SimpleNamespace(lr=1e-4, lr_ours=1e-4, lr_mortal=1e-4, epochs=2, batch=BATCH, explore=0.0,
                           clip=0.2, value_weight=0.5, hands_weight=1.0, leash=0.0, entropy=0.0005,
                           entropy_target=0.0, entropy_rate=0.01, entropy_max=0.05)
    optimiser = torch.optim.AdamW(train_combined.optimiser_groups(net, args), lr=args.lr,
                                  weight_decay=1e-4, fused=True)
    planes = encoded_like(ROWS, SEED)
    legal = torch.rand(ROWS, 46) < 0.4
    legal[:, 0] = True
    dense = planes.dense(device)
    logits, value, guessed = net.everything(dense, legal.to(device))
    (logits.float().logsumexp(1).mean() + value.float().mean() + guessed.float().mean()).backward()
    optimiser.step()
    optimiser.zero_grad(set_to_none=True)
    actions = torch.multinomial(legal.float(), 1).squeeze(1)
    with torch.no_grad():
        logits, _value, _guessed = net.everything(dense, legal.to(device))
        log_probs = torch.distributions.Categorical(
            logits=logits.float(), validate_args=False).log_prob(actions.to(device)).cpu()
    batch = SimpleNamespace(decisions=ROWS, observations=planes, legal=legal, actions=actions,
                            returns=torch.randn(ROWS), log_probs=log_probs)
    rollout = ppo_loop.on_device(batch, device, card=False)
    advantages, _spread = ppo_loop.standardised(rollout.returns, torch.zeros_like(rollout.returns))
    return SimpleNamespace(net=net, optimiser=optimiser, args=args, rollout=rollout, device=device,
                           held=torch.softmax(torch.randn(ROWS, 3, 34), dim=2).to(device),
                           advantages=advantages, begun=rollout.old_log_probs.clone())


def orders() -> list[torch.Tensor]:
    """The order each epoch of `learned` draws its minibatches in."""
    torch.manual_seed(SEED + 1)
    return [torch.randperm(ROWS) for _ in range(2)]


def poisoned(values: torch.Tensor, row: int, by: float) -> torch.Tensor:
    """`values` with `by` added at `row`: thirty nats on a probability the
    guard counts from makes the minibatch holding it drift by about seven,
    and a NaN return or probability is not a number wherever it goes."""
    values = values.clone()
    values[row] += by
    return values


def at_once(*args, **kwargs) -> StepGuard:
    """A guard that has every verdict on the host at once, as the trainer
    always did, whatever the optimiser could do on the card."""
    return StepGuard(*args, **{**kwargs, "late": False})


def learned(lesson: SimpleNamespace, late: bool, target: float = TARGET, args=None, begun=None,
            returns=None, explored=None) -> dict:
    """The trainer's epochs (`train_combined.ppo_epochs`) over the lesson's
    round, from its player and optimiser as they stand, with the guard's
    verdicts read `late` or had at once, and everything they leave."""
    net, optimiser = copy.deepcopy((lesson.net, lesson.optimiser))
    net.train()
    rollout = lesson.rollout if returns is None else dataclasses.replace(lesson.rollout, returns=returns)
    drift = PolicyDrift(target)
    totals = ppo_loop.Totals(lesson.device, "leash", "hands", "covered")
    clock = ppo_loop.LearningClock()
    guards = []

    def guard(*guard_args, **guard_kwargs):
        guards.append((StepGuard if late else at_once)(*guard_args, **guard_kwargs))
        return guards[-1]

    torch.manual_seed(SEED + 1)
    raised = updates = picks = planes = None
    with patch.object(train_combined, "StepGuard", guard):
        try:
            updates, picks, planes = train_combined.ppo_epochs(
                net, net.everything, rollout, lesson.held, optimiser, totals, drift, args or lesson.args,
                torch.float32, False, None, advantages=lesson.advantages,
                begun=lesson.begun if begun is None else begun,
                explored=(torch.zeros(ROWS, dtype=torch.bool, device=lesson.device)
                          if explored is None else explored),
                floor=train_combined.EntropyFloor(args or lesson.args), clock=clock)
        except (RuntimeError, FloatingPointError) as error:
            raised = error
    if picks is not None and planes is None:
        # As the trainer reads the head's shift: made again from the picks.
        planes = ppo_loop.planes_of(rollout, picks)
    [made] = guards
    return {
        "weights": dict(net.state_dict()),
        "adam": [optimiser.state[parameter] for group in optimiser.param_groups
                 for parameter in group["params"]],
        "generator": torch.get_rng_state(),
        "sums": totals.sums,
        "figures": drift.metrics(),
        "updates": updates,
        "noted": len(clock.updates),
        "picks": picks,
        "planes": planes,
        "raised": None if raised is None else (type(raised), str(raised)),
        # Not compared: how the verdicts were had.
        "how": {"late": made.late, "stepped": getattr(made, "_stepped", None)},
    }


def differences(one, other, where: str = "") -> list[str]:
    """Where `one` and `other` differ, bit for bit, apart from `how`."""
    if torch.is_tensor(one) or torch.is_tensor(other):
        same = (torch.is_tensor(one) and torch.is_tensor(other) and one.dtype == other.dtype
                and one.shape == other.shape
                and torch.equal(one.detach().cpu().reshape(-1).view(torch.uint8),
                                other.detach().cpu().reshape(-1).view(torch.uint8)))
        return [] if same else [where]
    if isinstance(one, dict):
        if not isinstance(other, dict) or one.keys() != other.keys():
            return [where]
        return [found for key in one if key != "how"
                for found in differences(one[key], other[key], f"{where}.{key}")]
    if isinstance(one, (list, tuple)):
        if type(one) is not type(other) or len(one) != len(other):
            return [where]
        return [found for index, (a, b) in enumerate(zip(one, other))
                for found in differences(a, b, f"{where}[{index}]")]
    if isinstance(one, float):
        same = isinstance(other, float) and (one.hex() == other.hex() or math.isnan(one) and math.isnan(other))
        return [] if same else [where]
    return [] if one == other else [where]


class LateVerdictTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.lesson = lesson()
        self.first, self.second = orders()

    def assertLearnedAlike(self, one: dict, other: dict):
        self.assertEqual(differences(one, other), [])

    def test_with_nothing_refused_late_verdicts_learn_what_deciding_at_once_learns(self):
        for target in (0.0, TARGET):
            with self.subTest(target=target):
                now, late = learned(self.lesson, False, target), learned(self.lesson, True, target)
                self.assertEqual((now["how"]["late"], late["how"]["late"]), (False, True))
                self.assertEqual(now["updates"], 2 * MINIBATCHES)
                self.assertFalse(now["figures"]["kl_early_stop"])
                self.assertGreater(now["figures"]["sampled_kl_max"], 0)
                self.assertLearnedAlike(now, late)

    def test_a_stop_read_late_ends_the_learning_where_deciding_at_once_ends_it(self):
        """Wherever the guard stops it: at the first minibatch, early in an
        epoch, two and one before its end, whose verdicts are read only
        once the epoch is done and before the next draws its order, on its
        last, and in the second epoch."""
        cases = {k: self.first[k * BATCH] for k in (0, 3, MINIBATCHES - 3, MINIBATCHES - 2, MINIBATCHES - 1)}
        # A row the first epoch left over is first met in the second.
        for row in self.first[MINIBATCHES * BATCH:]:
            k = int((self.second == row).nonzero()) // BATCH
            if k < MINIBATCHES:
                cases[MINIBATCHES + k] = row
        self.assertGreaterEqual(max(cases), MINIBATCHES)
        for k, row in cases.items():
            with self.subTest(minibatch=k):
                begun = poisoned(self.lesson.begun, row, 30.0)
                now, late = learned(self.lesson, False, begun=begun), learned(self.lesson, True, begun=begun)
                self.assertTrue(now["figures"]["kl_early_stop"])
                self.assertEqual(now["updates"], k)
                self.assertGreater(now["figures"]["sampled_kl_last"], TARGET)
                # Read two updates late, but within its epoch: the updates
                # launched after it were refused on the card.
                within = MINIBATCHES - 1 - k % MINIBATCHES
                self.assertEqual(late["how"]["stepped"], k + 1 + min(StepGuard.LAG, within))
                self.assertLearnedAlike(now, late)

    def test_a_gradient_that_is_not_a_number_raises_the_clips_own_error_late(self):
        for k in (0, 3, MINIBATCHES - 1):
            with self.subTest(minibatch=k):
                returns = poisoned(self.lesson.rollout.returns, self.first[k * BATCH], math.nan)
                now = learned(self.lesson, False, returns=returns)
                late = learned(self.lesson, True, returns=returns)
                self.assertEqual(now["raised"][0], RuntimeError)
                self.assertIn("non-finite, so it cannot be clipped", now["raised"][1])
                self.assertEqual(now["noted"], k)
                self.assertLearnedAlike(now, late)

    def test_a_drift_that_is_not_a_number_raises_the_guards_error_late(self):
        for k in (3, MINIBATCHES - 2):
            with self.subTest(minibatch=k):
                begun = poisoned(self.lesson.begun, self.first[k * BATCH], math.nan)
                now, late = learned(self.lesson, False, begun=begun), learned(self.lesson, True, begun=begun)
                self.assertEqual(now["raised"], (FloatingPointError, NONFINITE_DRIFT))
                self.assertLearnedAlike(now, late)

    def test_rows_exploration_forced_are_kept_from_the_late_guard_as_from_the_guard_at_once(self):
        torch.manual_seed(SEED + 2)
        explored = torch.rand(ROWS) < 0.3
        # Every row of one minibatch forced, which the guard does not ask about.
        explored[self.first[:BATCH]] = True
        args = SimpleNamespace(**{**vars(self.lesson.args), "explore": 0.1})
        stop = self.first[5 * BATCH + 1]
        explored[stop] = False
        for begun in (None, poisoned(self.lesson.begun, stop, 30.0)):
            with self.subTest(stopped=begun is not None):
                now = learned(self.lesson, False, args=args, begun=begun, explored=explored)
                late = learned(self.lesson, True, args=args, begun=begun, explored=explored)
                self.assertEqual(now["figures"]["kl_early_stop"], begun is not None)
                self.assertLearnedAlike(now, late)

    def test_the_entropy_floor_and_an_optimiser_that_cannot_refuse_have_every_verdict_at_once(self):
        parameter = torch.nn.Parameter(torch.zeros(3))
        drift = PolicyDrift(TARGET)
        self.assertFalse(StepGuard(drift, torch.optim.AdamW([parameter]), "cpu").late)
        fused = torch.optim.AdamW([parameter], fused=True)
        self.assertTrue(StepGuard(drift, fused, "cpu").late)
        self.assertFalse(StepGuard(drift, fused, "cpu", late=False).late)
        made = []

        def recorded(*args, **kwargs):
            made.append(StepGuard(*args, **kwargs))
            return made[-1]

        floored = SimpleNamespace(**{**vars(self.lesson.args), "entropy_target": 0.5})
        net, optimiser = copy.deepcopy((self.lesson.net, self.lesson.optimiser))
        with patch.object(train_combined, "StepGuard", recorded):
            train_combined.ppo_epochs(
                net, net.everything, self.lesson.rollout, self.lesson.held, optimiser,
                ppo_loop.Totals("cpu", "leash", "hands", "covered"), drift, floored, torch.float32, False,
                None, advantages=self.lesson.advantages, begun=self.lesson.begun,
                explored=torch.zeros(ROWS, dtype=torch.bool), floor=train_combined.EntropyFloor(floored))
        self.assertEqual([guard.late for guard in made], [False])


class StepGuardTests(unittest.TestCase):
    def test_fused_adamw_refuses_an_update_on_found_inf_and_takes_back_its_step(self):
        """What the late guard relies on: `found_inf` of one leaves the
        weights, the moments and the step count as they were, bit for bit,
        and of nought changes nothing of the update."""
        torch.manual_seed(SEED)
        parameter = torch.nn.Parameter(torch.randn(64))
        optimiser = torch.optim.AdamW([parameter], lr=1e-2, weight_decay=0.1, fused=True)
        for _ in range(3):
            parameter.grad = torch.randn(64)
            optimiser.step()
        parameter.grad = torch.randn(64)
        before = copy.deepcopy((parameter.detach(), optimiser.state[parameter]))
        optimiser.found_inf = torch.ones(())
        optimiser.step()
        del optimiser.found_inf
        self.assertEqual(differences(before, (parameter.detach(), optimiser.state[parameter])), [])
        other, plain = copy.deepcopy((parameter, optimiser))
        # A parameter's copy leaves its gradient behind.
        other.grad = parameter.grad.clone()
        optimiser.found_inf = torch.zeros(())
        optimiser.step()
        del optimiser.found_inf
        plain.step()
        self.assertEqual(differences((parameter.detach(), optimiser.state[parameter]),
                                     (other.detach(), plain.state[other])), [])
        self.assertFalse(torch.equal(parameter.detach(), before[0]))

    def test_a_stop_is_read_two_updates_late_with_nothing_applied_after_it(self):
        parameter = torch.nn.Parameter(torch.zeros(4))
        optimiser = torch.optim.AdamW([parameter], lr=0.1, fused=True)
        drift = PolicyDrift(TARGET)
        clock = ppo_loop.LearningClock()
        guard = StepGuard(drift, optimiser, "cpu", clock=clock)
        zero = torch.zeros(4)
        weights = []
        for update, gap in enumerate((0.0, 30.0, 0.0, 0.0)):
            parameter.grad = torch.ones(4)
            self.assertFalse(guard.check(zero, zero - gap))
            guard.clip([parameter], 1.0)
            guard.step()
            weights.append(parameter.detach().clone())
            guard.launched(torch.tensor([update]))
            self.assertEqual(drift.stopped, update == 3)
        self.assertEqual(guard.updates, 1)
        self.assertEqual(len(clock.updates), 1)
        self.assertEqual(int(guard.ended), 1)
        for later in weights[1:]:
            self.assertTrue(torch.equal(later, weights[0]))
        self.assertEqual(float(optimiser.state[parameter]["step"]), 1.0)
        self.assertGreater(drift.maximum, TARGET)
        self.assertEqual(drift.last, drift.maximum)

    def test_the_late_guard_compares_its_target_as_the_host_does(self):
        """A target just under a drift, one that single precision rounds up
        to it: the host, comparing the figure it reads in double precision,
        stops; so does the late verdict."""
        torch.manual_seed(SEED)
        for _ in range(20):
            old, new = torch.randn(8) * 0.1, torch.randn(8) * 0.1
            estimate = float(PolicyDrift().estimate(old, new))
            for target in (estimate, math.nextafter(estimate, 0.0)):
                stop = PolicyDrift(target).check(old, new)
                _bad, over = PolicyDrift(target).verdict(old, new, torch.zeros((), dtype=torch.bool))
                self.assertEqual(bool(over), stop)
            self.assertTrue(PolicyDrift(math.nextafter(estimate, 0.0)).check(old, new))

    def test_an_update_refused_adds_nothing_to_the_record(self):
        totals = ppo_loop.Totals("cpu")
        totals.add(policy=torch.tensor(0.25), value=torch.tensor(-1.5))
        before = {name: value.clone() for name, value in totals.sums.items()}
        totals.add(unless=torch.tensor(True), policy=torch.tensor(math.nan), value=torch.tensor(7.0))
        self.assertEqual(differences(before, totals.sums), [])
        totals.add(unless=torch.tensor(False), policy=torch.tensor(0.5), value=torch.tensor(1.0))
        self.assertEqual((float(totals["policy"]), float(totals["value"])), (0.75, -0.5))


class TrainerTests(unittest.TestCase):
    def test_the_trainer_stops_its_remaining_epochs_on_a_stop_read_late_and_closes_prefetch(self):
        """The joined player's trainer with its AdamW fused, as on a card,
        here on the processor: the guard refuses the second update, whose
        verdict is read two updates later, at the epoch's last; the
        generation records one update and the stop, no later epoch begins,
        and the prefetcher's threads are gone (see test_training_controls,
        whose trainers decide at once)."""
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            origin, out = root / "source.pt", root / "run"
            net = combined.Combined(model.PolicyValueNet(8, 1, actions=46), mortal_model.build(8, 1))
            net.mortal_config = {"resnet": {"conv_channels": 8, "num_blocks": 1}, "control": {"version": 4}}
            torch.save(net.state(), origin)

            def collect(_learner, **_kwargs):
                n = 8
                planes = Planes(np.arange(n + 1, dtype=np.int64), np.zeros(n, dtype=np.uint16),
                                np.ones(n, dtype=np.float16))
                return SimpleNamespace(decisions=n, observations=planes, legal=torch.ones(n, 46, dtype=torch.bool),
                                       actions=torch.zeros(n, dtype=torch.int64), returns=torch.linspace(-1, 1, n),
                                       log_probs=torch.zeros(n), held=torch.full((n, 3, 34), 1 / 34),
                                       games=1, hands=1, timing={})

            verdicts = []

            def verdict(_drift, _old_log_prob, _new_log_prob, _refused):
                verdicts.append(1)
                return torch.tensor(False), torch.tensor(len(verdicts) == 2)

            adamw = torch.optim.AdamW

            def fused(*args, **kwargs):
                return adamw(*args, **{**kwargs, "fused": True})

            argv = ["trainer", "--resume", str(origin), "--out", str(out), "--rounds", "1", "--batch", "2",
                    "--epochs", "3", "--games", "1", "--measure-games", "1", "--target-kl", ".02",
                    "--baseline-batch", "2", "--fixed", "none"]
            before = {t.ident for t in threading.enumerate() if t.name == "mahjong-prefetch"}
            with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False), \
                    patch.object(train_combined.selfplay, "play", side_effect=collect), \
                    patch.object(train_combined.selfplay, "measure",
                                 return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
                    patch.object(torch.optim, "AdamW", fused), patch.object(PolicyDrift, "verdict", verdict), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_combined.main()
            rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["optimizer_updates"], 1)
            self.assertTrue(rows[0]["kl_early_stop"])
            # Asked of the two updates launched after the one refused, the
            # epoch's last, and of none after: no later epoch began.
            self.assertEqual(len(verdicts), 2 + StepGuard.LAG)
            self.assertEqual(before, {t.ident for t in threading.enumerate() if t.name == "mahjong-prefetch"})


@unittest.skipUnless(CUDA, "the card's verdicts through page-locked memory")
class OnTheCardTests(unittest.TestCase):
    def setUp(self):
        deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic", deterministic)
        torch.set_num_threads(1)
        self.lesson = lesson("cuda")
        self.first, _second = orders()

    def test_on_the_card_late_verdicts_learn_what_deciding_at_once_learns(self):
        stop = poisoned(self.lesson.begun, self.first[3 * BATCH], 30.0)
        broken = poisoned(self.lesson.rollout.returns, self.first[(MINIBATCHES - 1) * BATCH], math.nan)
        for name, options in (("nothing refused", {}), ("a stop", {"begun": stop}),
                              ("a gradient that is not a number", {"returns": broken})):
            with self.subTest(name):
                now, late = learned(self.lesson, False, **options), learned(self.lesson, True, **options)
                self.assertTrue(late["how"]["late"])
                self.assertEqual(differences(now, late), [])
