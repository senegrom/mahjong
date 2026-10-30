"""The noise pre-test: the oracle's term and its control, its readings of
the residual and of the gradient's noise, the twin ceiling, the verdicts,
the reward it refuses, its cloud launch, and one small run of the whole of
it, end to end, on the processor.

    python -m unittest neural.tests.test_noise_pretest -v
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from neural import noise_pretest, place_model, train_combined
from neural.observe import PLANES, POSITIONS, Planes
from neural.scoreboard import STAGE_NAMES, Ledger
from neural.tests.test_resumed_checkpoints import flags, mortal_checkpoint, run


class Terms(unittest.TestCase):
    """The oracle's term and its control start as the checkpoint's judge,
    from the same weights, and the control never reads the oracle's
    planes."""

    def test_both_start_at_nought_and_only_the_oracles_reads_what_is_hidden(self):
        torch.manual_seed(5)
        net = SimpleNamespace(planes=PLANES, channels=8, ours=SimpleNamespace(attention=True))
        heads = noise_pretest.correction_pair(net, SimpleNamespace(tower_channels=16, tower_blocks=1), 3, "cpu")
        seeing, blind = heads["oracle"], heads["control"]
        self.assertFalse(seeing.blind)
        self.assertTrue(blind.blind)
        for (name, one), (_name, two) in zip(seeing.state_dict().items(), blind.state_dict().items()):
            self.assertTrue(torch.equal(one, two), name)
        n = 12
        pooled = torch.randn(n, 8)
        planes = (torch.rand(n, PLANES, POSITIONS) < 0.05).float()
        oracle = (torch.rand(n, 32, POSITIONS) < 0.1).float()
        other = (torch.rand(n, 32, POSITIONS) < 0.1).float()
        own = torch.randn(n)
        with torch.no_grad():
            for head in (seeing, blind):
                self.assertTrue(torch.equal(own + head(pooled, planes, oracle), own))
        target = torch.randn(n)
        for head in (seeing, blind):
            optimiser = torch.optim.AdamW(head.parameters(), lr=1e-2)
            loss = ((head(pooled, planes, oracle) - target) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            self.assertFalse(torch.equal(blind(pooled, planes, oracle), torch.zeros(n)))
            self.assertTrue(torch.equal(blind(pooled, planes, oracle), blind(pooled, planes, other)))
            self.assertFalse(torch.equal(seeing(pooled, planes, oracle), seeing(pooled, planes, other)))


class Noise(unittest.TestCase):
    """N read from the Gram matrices alone, and its jackknife, against the
    same read from the sums themselves."""

    @staticmethod
    def spread(sums: np.ndarray) -> float:
        return float(((sums - sums.mean(axis=0)) ** 2).sum() / (len(sums) - 1))

    def test_n_is_the_spread_of_the_group_sums(self):
        sums = np.random.default_rng(1).standard_normal((6, 40))
        self.assertAlmostEqual(noise_pretest.noise(sums @ sums.T), self.spread(sums), places=10)

    def test_the_ratios_and_their_jackknife(self):
        groups = 5
        rng = np.random.default_rng(2)
        own = rng.standard_normal((groups, 30))
        control = 0.8 * own + 0.3 * rng.standard_normal((groups, 30))
        # Every group's sum halved: a quarter of own's noise, read the same
        # with any group left out, so with no error at all.
        oracle = 0.5 * own
        stacked = np.concatenate([own, control, oracle])
        readings = noise_pretest.noise_readings(stacked @ stacked.T, groups)
        self.assertEqual(noise_pretest.ARMS, ("own", "control", "oracle"))
        self.assertAlmostEqual(readings["oracle_over_own"]["value"], 0.25)
        self.assertAlmostEqual(readings["oracle_over_own"]["se"], 0.0)
        whole = self.spread(control) / self.spread(own)
        left = np.array([self.spread(np.delete(control, g, axis=0)) / self.spread(np.delete(own, g, axis=0))
                         for g in range(groups)])
        error = math.sqrt((groups - 1) / groups * ((left - left.mean()) ** 2).sum())
        self.assertGreater(error, 0.0)
        # The report keeps six significant figures.
        for read, wanted in ((readings["N"]["own"]["value"], self.spread(own)),
                             (readings["control_over_own"]["value"], whole),
                             (readings["control_over_own"]["se"], error),
                             (readings["oracle_over_control"]["value"],
                              0.25 * self.spread(own) / self.spread(control))):
            self.assertAlmostEqual(read / wanted, 1.0, places=5)

    def test_the_preconditioned_products_divide_every_sum_by_the_moments(self):
        sums = torch.randn(4, 10)
        denominator = torch.rand(10) + 0.5
        plain, preconditioned = noise_pretest.grams(sums, denominator)
        wide = sums.double()
        np.testing.assert_allclose(plain, (wide @ wide.T).numpy())
        scaled = wide / denominator.double()
        np.testing.assert_allclose(preconditioned, (scaled @ scaled.T).numpy())
        self.assertIsNone(noise_pretest.grams(sums, None)[1])


class Residuals(unittest.TestCase):
    def test_a_baseline_that_halves_every_residual_cuts_three_quarters_everywhere(self):
        """The cuts read 1 - V(top)/V(bottom) cell by cell, plain and
        weighted, with no error where the ratio is the same in every
        group; and V is the weighted variance, read directly."""
        rng = np.random.default_rng(3)
        n, groups = 400, 4
        returns = rng.standard_normal(n)
        residual = rng.standard_normal(n)
        baselines = {"own": returns - 2 * residual, "control": returns - residual,
                     "oracle": returns - 0.5 * residual}
        weights = rng.random(n)
        cells = noise_pretest.cells_of(rng.integers(0, 8, n), rng.integers(-1, 70, n))
        group = rng.integers(0, groups, n)
        readings, sums = noise_pretest.residual_readings(returns, baselines, weights, cells, group, groups)
        for top, bottom in (("oracle", "control"), ("control", "own")):
            cut = noise_pretest.cut(sums, top, bottom, cells)
            for kind in ("plain", "weighted"):
                for split, (_labels, names) in cells.items():
                    for name in names:
                        with self.subTest(top=top, kind=kind, cell=name):
                            self.assertAlmostEqual(cut[kind][split][name]["value"], 0.75)
                            self.assertAlmostEqual(cut[kind][split][name]["se"], 0.0)
        mean = (weights * residual).sum() / weights.sum()
        direct = (weights * (residual - mean) ** 2).sum() / weights.sum()
        self.assertAlmostEqual(noise_pretest.overall(sums, "control"), direct)
        # The report keeps six significant figures.
        self.assertAlmostEqual(readings["control"]["weighted"]["all"]["all"]["value"] / direct, 1.0, places=5)
        self.assertAlmostEqual(readings["control"]["plain"]["all"]["all"]["value"] / residual.var(), 1.0,
                               places=5)
        self.assertEqual(set(readings["own"]["plain"]["by_tiles_left"]), set(noise_pretest.LEFT_NAMES))

    def test_the_information_cut_reads_against_the_better_of_the_control_and_own(self):
        """Given two baselines to read against, the cut is read against
        whichever has the less V, cell by cell and with each group left out
        alike: here the control in East's hands and own in South's, and own
        over every decision."""
        rng = np.random.default_rng(4)
        n, groups = 800, 4
        returns = rng.standard_normal(n)
        residual = rng.standard_normal(n)
        stages = rng.integers(0, 8, n)
        baselines = {"own": returns - residual, "control": returns - np.where(stages < 4, 0.8, 1.25) * residual,
                     "oracle": returns - 0.5 * residual}
        cells = noise_pretest.cells_of(stages, rng.integers(0, 70, n))
        _readings, sums = noise_pretest.residual_readings(returns, baselines, rng.random(n), cells,
                                                          rng.integers(0, groups, n), groups)
        both = noise_pretest.cut(sums, "oracle", ("control", "own"), cells)
        control = noise_pretest.cut(sums, "oracle", "control", cells)
        own = noise_pretest.cut(sums, "oracle", "own", cells)
        for kind in ("plain", "weighted"):
            for stage, name in enumerate(STAGE_NAMES):
                better = control if stage < 4 else own
                with self.subTest(kind=kind, stage=name):
                    self.assertEqual(both[kind]["by_stage"][name], better[kind]["by_stage"][name])
            self.assertEqual(both[kind]["all"]["all"], own[kind]["all"]["all"])
            self.assertAlmostEqual(both[kind]["all"]["all"]["value"], 0.75)

    def test_deals_are_drawn_whole_and_balanced(self):
        group, half, check = noise_pretest.deal_draws(41, 20, 0.25, 7)
        self.assertEqual(set(np.bincount(group).tolist()), {2, 3})
        self.assertEqual(sorted(np.bincount(half).tolist()), [20, 21])
        for side in (0, 1):
            mine = half == side
            self.assertGreaterEqual(int(check[mine].sum()), 1)
            self.assertGreaterEqual(int((~check[mine]).sum()), 1)
            self.assertEqual(int(check[mine].sum()), round(0.25 * int(mine.sum())))
        _group, half, check = noise_pretest.deal_draws(4, 3, 0.25, 7)
        for side in (0, 1):
            self.assertEqual(int(check[half == side].sum()), 1)

    def test_a_fit_still_falling_at_its_last_epoch_has_not_finished(self):
        falling = noise_pretest.still_falling
        self.assertTrue(falling([0.2, 0.15, 0.14, 0.13], 3, 0.002))
        self.assertFalse(falling([0.2, 0.15, 0.14, 0.1399999], 3, 0.002))
        self.assertFalse(falling([0.2, 0.15, 0.13, 0.14], 3, 0.002))
        self.assertFalse(falling([0.2, 0.15, 0.14], 3, 0.002))
        self.assertTrue(falling([0.2, 0.15], 1, 0.002))
        self.assertFalse(falling([0.2, 0.25], 1, 0.002))


class Ceiling(unittest.TestCase):
    def test_the_ceiling_of_a_hand_built_round(self):
        """Game 0: player 1 decides with chances of differing 0, 0.5 and
        0.5, player 3 once with 0.2, in the order 1, 3, 1, 1; game 1:
        player 2 once with 0.4, recorded between game 0's decisions."""
        differ = np.array([0.0, 0.2, 0.5, 0.4, 0.5])
        games = np.array([0, 0, 0, 1, 0])
        players = np.array([1, 3, 1, 2, 1], dtype=np.int8)
        ceiling = noise_pretest.twin_ceiling(differ, games, players)
        # Seat by seat: 0 + 1 x 0.5 + 0.5 x 0.5 for player 1, 0.2, 0.4.
        by_seat = 0.75 + 0.2 + 0.4
        # Game by game: game 0 in the order made, 0, 0.2, 0.5, 0.5, parts at
        # the second with 0.2, the third with 0.8 x 0.5, the fourth with
        # 0.8 x 0.5 x 0.5; game 1 at its only decision.
        by_game = 0.2 + 0.4 + 0.2 + 0.4
        self.assertEqual(ceiling["decisions"], 5)
        self.assertAlmostEqual(ceiling["differing_draws"], 1.6)
        self.assertEqual(ceiling["by_seat"]["sequences"], 3)
        self.assertEqual(ceiling["by_game"]["sequences"], 2)
        self.assertAlmostEqual(ceiling["by_seat"]["expected_partings"], by_seat)
        self.assertAlmostEqual(ceiling["by_game"]["expected_partings"], by_game)
        self.assertAlmostEqual(ceiling["by_seat"]["share"], round(by_seat / 1.6, 6))
        self.assertAlmostEqual(ceiling["by_game"]["share"], round(by_game / 1.6, 6))

    def test_the_chance_two_draws_differ_keeps_a_near_certain_rows_millionths(self):
        logits = torch.tensor([[0.0, 0.0, float("-inf")], [0.0, -30.0, float("-inf")]])
        differ = noise_pretest.disagreement(logits)
        self.assertAlmostEqual(float(differ[0]), 0.5)
        rare = math.exp(-30) / (1 + math.exp(-30))
        self.assertAlmostEqual(float(differ[1]) / (2 * rare * (1 - rare)), 1.0, places=9)


class Verdicts(unittest.TestCase):
    def test_the_registered_readings(self):
        verdict = noise_pretest.oracle_verdict
        passed = {"passed": True}
        cut = {"value": 0.35, "se": 0.02}
        ratio = {"value": 0.7, "se": 0.04}
        self.assertEqual(verdict(cut, passed, [], ratio)["verdict"], "go")
        self.assertEqual(verdict({"value": 0.29, "se": 0.02}, passed, [], ratio)["verdict"], "kill")
        self.assertEqual(verdict(cut, passed, [], {"value": 0.81, "se": 0.01})["verdict"], "kill")
        # At most 0.8, but 0.78 + 2 x 0.07 is not below 0.9.
        self.assertEqual(verdict(cut, passed, [], {"value": 0.78, "se": 0.07})["verdict"], "kill")
        self.assertEqual(verdict(cut, {"passed": False}, [], ratio)["verdict"], "unread")
        self.assertEqual(verdict(cut, passed, ["the oracle fit of half 0"], ratio)["verdict"], "unread")
        self.assertEqual(verdict(cut, passed, [], None)["verdict"], "unread")
        # The twin ceiling is read game by game; seat by seat is only said.
        twins = noise_pretest.duplicates_verdict
        built = twins({"by_game": {"share": 0.2}, "by_seat": {"share": 0.21}})
        self.assertEqual(built["verdict"], "build a twin round next")
        self.assertEqual((built["ceiling"], built["by_seat"], built["needed"]), (0.2, 0.21, 0.2))
        self.assertEqual(twins({"by_game": {"share": 0.19}, "by_seat": {"share": 0.38}})["verdict"], "stop")
        self.assertEqual(twins({"by_game": {"share": None}, "by_seat": {"share": None}})["verdict"], "stop")


class Reward(unittest.TestCase):
    def test_only_a_placement_runs_checkpoint_is_read(self):
        u = place_model.PlaceModel(("zoo/mortal_298k", "club"), hidden=8).state()
        for payload in ({}, {"reward": {"kind": "hybrid"}, "place_model": u},
                        {"reward": {"kind": "placement", "alpha": 0.05}}):
            with self.subTest(payload=sorted(payload)), self.assertRaises(SystemExit):
                noise_pretest.carried_reward(payload)
        reward = noise_pretest.carried_reward({"reward": {"kind": "placement", "alpha": 0.05, "since": 1},
                                               "place_model": u})
        self.assertEqual(reward.alpha, 0.05)
        self.assertEqual(reward.model.members, ("zoo/mortal_298k", "club"))


#: How the card says it has no room: PyTorch's own error, and the plain
#: RuntimeErrors of cuDNN's and cuBLAS's allocations.
NO_ROOM = (
    lambda: torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    lambda: RuntimeError("cuDNN error: CUDNN_STATUS_EXECUTION_FAILED_CUDART"),
    lambda: RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"),
    lambda: RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"),
    lambda: RuntimeError("CUDA error: out of memory"),
)


class Halving(unittest.TestCase):
    def test_a_chunk_without_room_is_halved_for_the_rest_of_the_pass(self):
        """A chunk that runs out of the card's memory, however the card says
        so, is read again at half its rows, as often as it takes, and every
        chunk after it at that size; the sums are the whole round's all the
        same. Any other failure raises."""
        n = 20
        rollout = SimpleNamespace(
            observations=Planes(np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.uint16),
                                np.zeros(0, dtype=np.float16)),
            legal=torch.ones(n, 46, dtype=torch.bool), actions=torch.arange(n), decisions=n)
        weights = torch.randn(3, n)

        def summed(failure):
            sizes = []

            def gradients(_net, part, asked, _args, _device, fallback=True):
                sizes.append(part.decisions)
                if part.decisions > 3:
                    raise failure()
                return (asked * part.actions.float()).sum(dim=1, keepdim=True), "batched"

            args = SimpleNamespace(gradient_chunk=8, amp=False)
            with patch.object(noise_pretest.placement_pretest, "head_gradients", side_effect=gradients):
                try:
                    return noise_pretest.gradient_sums(None, rollout, weights, args, "cpu"), sizes, args
                except RuntimeError as error:
                    return error, sizes, args

        for failure in NO_ROOM:
            with self.subTest(failure=repr(failure())):
                (sums, how, halved), sizes, args = summed(failure)
                torch.testing.assert_close(sums, (weights * torch.arange(n).float()).sum(dim=1, keepdim=True))
                self.assertEqual(how, "batched")
                # No room at eight rows, nor at four: two at a time from
                # there on, nothing larger tried again.
                self.assertEqual(sizes, [8, 4] + [2] * 10)
                self.assertEqual(halved, 2)
                self.assertEqual(args.gradient_chunk, 2)
        error, sizes, args = summed(lambda: RuntimeError("vmap: no batching rule for this graph"))
        self.assertIn("no batching rule", str(error))
        self.assertEqual(sizes, [8])
        self.assertEqual(args.gradient_chunk, 8)


class Preflight(unittest.TestCase):
    """The gradient's preflight with the products stood in for: a chunk
    without room, however the card says so, or whose peak holds more than
    four fifths of the card, is halved until it goes through, and its time
    is projected; any other failure, or no room even at the fewest rows,
    ends the run before anything is played."""

    CARD = 48 * 2**30

    def fly(self, failure, peak=None):
        """The preflight at 512 rows a chunk and four groups, `failure(rows)`
        raised by the products of a chunk of that many rows when it answers
        one, and `peak(rows)` the share of the card such a chunk peaked
        at. Answers the flight, or the SystemExit that ended it, and the
        rows of every chunk the products were asked about."""
        tried = []

        def gradients(_net, made, weights, args, _device, fallback=True):
            tried.append(made.decisions)
            self.assertEqual(tuple(weights.shape), (12, made.decisions))
            self.assertEqual(args.chunk, made.decisions)
            error = failure(made.decisions)
            if error is not None:
                raise error
            return torch.zeros(len(weights), 1), "batched"

        def card(_device):
            return None if peak is None else (int(peak(tried[-1]) * self.CARD), self.CARD)

        args = SimpleNamespace(gradient_chunk=512, groups=4, seed=3, amp=False, games=8)
        net = SimpleNamespace(planes=PLANES)
        with patch.object(noise_pretest.placement_pretest, "head_gradients", side_effect=gradients), \
                patch.object(noise_pretest, "card_peak", side_effect=card), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                return noise_pretest.preflight(net, args, "cpu"), tried, args
            except SystemExit as stopped:
                return stopped, tried, args

    def test_a_chunk_without_room_is_halved_until_it_goes_through(self):
        for failure in NO_ROOM:
            with self.subTest(failure=repr(failure())):
                flight, tried, args = self.fly(lambda rows, failure=failure: failure() if rows > 100 else None)
                # Twice at the size that went through: the first try warms
                # the card and the second is timed.
                self.assertEqual(tried, [512, 256, 128, 64, 64])
                self.assertEqual((flight["rows_a_chunk"], args.gradient_chunk), (64, 64))
                self.assertEqual((flight["how"], flight["vectors"]), ("batched", 12))
                self.assertEqual(flight["projected_rows"], 8 * noise_pretest.ROWS_A_GAME)
                self.assertGreaterEqual(flight["projected_minutes"], 0.0)
                self.assertIsNone(flight["peak_bytes"])

    def test_a_chunk_whose_peak_holds_too_much_of_the_card_is_halved(self):
        flight, tried, args = self.fly(lambda rows: None, peak=lambda rows: 0.9 if rows > 100 else 0.5)
        self.assertEqual(tried, [512, 512, 256, 256, 128, 128, 64, 64])
        self.assertEqual((flight["rows_a_chunk"], args.gradient_chunk), (64, 64))
        self.assertEqual((flight["peak_bytes"], flight["card_bytes"]), (int(0.5 * self.CARD), self.CARD))
        # Four fifths of the card is still room.
        flight, tried, _args = self.fly(lambda rows: None, peak=lambda rows: noise_pretest.CARD_SHARE)
        self.assertEqual((tried, flight["rows_a_chunk"]), ([512, 512], 512))

    def test_any_other_failure_or_no_room_at_the_fewest_rows_ends_the_run(self):
        for failure in (RuntimeError("vmap: no batching rule for this graph"), NotImplementedError("no rule")):
            with self.subTest(failure=repr(failure)):
                stopped, tried, args = self.fly(lambda rows, failure=failure: failure)
                self.assertIsInstance(stopped, SystemExit)
                self.assertIn("cannot be taken at once", str(stopped))
                self.assertEqual((tried, args.gradient_chunk), ([512], 512))
        stopped, tried, _args = self.fly(lambda rows: NO_ROOM[1]())
        self.assertIsInstance(stopped, SystemExit)
        self.assertIn(f"even at {noise_pretest.FEWEST_ROWS} rows", str(stopped))
        self.assertEqual(tried, [512, 256, 128, 64, 32])
        stopped, tried, _args = self.fly(lambda rows: None, peak=lambda rows: 0.95)
        self.assertIsInstance(stopped, SystemExit)
        self.assertEqual(tried, [512, 512, 256, 256, 128, 128, 64, 64, 32, 32])

    def test_the_budget_adds_the_spent_the_known_and_the_projected(self):
        flight = {"projected_minutes": 30.0}
        known = sum(noise_pretest.PHASE_MINUTES.values())
        self.assertEqual(noise_pretest.budget(flight, 4.04, SimpleNamespace(games=4096, max_minutes=75.0)),
                         {"spent": 4.0, "other_phases": known, "gradient": 30.0, "minutes": round(34.04 + known, 1),
                          "most": 75.0})
        halved = noise_pretest.budget(flight, 0.0, SimpleNamespace(games=2048, max_minutes=75.0))
        self.assertEqual(halved["other_phases"], round(known / 2, 1))


class Outputs(unittest.TestCase):
    def test_a_folder_that_holds_anything_is_refused_before_any_work(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            (root / "used").mkdir()
            (root / "used" / "report.json").write_text("an earlier run's", encoding="utf-8")
            (root / "a-file").write_text("not a folder", encoding="utf-8")
            for out in (root / "used", root / "a-file"):
                with self.subTest(out=out.name), self.assertRaisesRegex(SystemExit, "not an empty folder"):
                    # The checkpoint is never looked for.
                    noise_pretest.main([str(root / "missing.pt"), "--out", str(out)])
            self.assertEqual((root / "used" / "report.json").read_text(encoding="utf-8"), "an earlier run's")
            self.assertEqual(sorted(path.name for path in (root / "used").iterdir()), ["report.json"])
            # An empty folder is taken.
            (root / "empty").mkdir()
            noise_pretest.refuse_kept(root / "empty")
            noise_pretest.refuse_kept(root / "not-there")

    def test_a_file_is_written_whole_or_not_at_all(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            path = Path(folder) / "report.json"
            noise_pretest.kept(path, lambda stream: stream.write(b"first"))
            self.assertEqual(path.read_bytes(), b"first")

            def cut_short(stream):
                stream.write(b"half of a sec")
                raise OSError("no space left on the device")

            with self.assertRaisesRegex(OSError, "no space left"):
                noise_pretest.kept(path, cut_short)
            self.assertEqual(path.read_bytes(), b"first")
            self.assertEqual([entry.name for entry in Path(folder).iterdir()], ["report.json"])


class Advantages(unittest.TestCase):
    def test_each_arms_advantage_is_centred_over_the_round_and_kept_to_its_groups_rows(self):
        """The rows of weights: arm after arm, in `ARMS`' order, and in each
        arm group after group; the return less the arm's baseline, centred
        over the whole round and not standardised, on the group's decisions
        and nought on every other."""
        rng = np.random.default_rng(6)
        n, groups = 60, 4
        returns = torch.from_numpy(3.0 * rng.standard_normal(n) + 1.0)
        baselines = {arm: torch.from_numpy(spread * rng.standard_normal(n) + shift)
                     for arm, spread, shift in zip(noise_pretest.ARMS, (2.0, 1.0, 0.5), (0.5, -1.0, 2.0))}
        group = torch.from_numpy(rng.permutation(n) % groups)
        weights = noise_pretest.advantage_weights(returns, baselines, group, groups).numpy()
        self.assertEqual(weights.shape, (len(noise_pretest.ARMS) * groups, n))
        mine = group.numpy()
        for index, arm in enumerate(noise_pretest.ARMS):
            advantage = (returns - baselines[arm]).numpy()
            centred = advantage - advantage.mean()
            rows = weights[index * groups:(index + 1) * groups]
            for g in range(groups):
                with self.subTest(arm=arm, group=g):
                    np.testing.assert_allclose(rows[g][mine == g], centred[mine == g])
                    self.assertTrue((rows[g][mine != g] == 0).all())
            # Every decision in one group: together the arm's rows are its
            # advantage centred over the round, its own spread kept, and no
            # group's centred on its own.
            np.testing.assert_allclose(rows.sum(axis=0), centred)
            self.assertAlmostEqual(float(rows.sum()), 0.0, places=9)
            self.assertAlmostEqual(float(rows.sum(axis=0).std()), float(advantage.std()), places=9)
            self.assertGreater(abs(float(advantage.std()) - 1.0), 0.5)
            self.assertGreater(max(abs(float(centred[mine == g].mean())) for g in range(groups)), 1e-3)


class CrossFitting(unittest.TestCase):
    def test_each_halfs_terms_are_taught_and_checked_on_it_and_asked_only_about_the_other(self):
        """The real fits, with every row they read recorded: what a half's
        oracle's term and control are taught, the rows their minibatches
        and their checks read, come only from that half (taught from its
        fit deals, checked on its check deals), and what they are asked
        about is the other half, all of it."""
        rng = np.random.default_rng(8)
        deals, each, lit = 12, 4, 30
        n = deals * each
        game = np.repeat(np.arange(deals), each)
        _group, half_of, check_of = noise_pretest.deal_draws(deals, 3, 0.25, 9)
        half, check = half_of[game], check_of[game]
        indices = np.sort(rng.choice(PLANES * POSITIONS, size=(n, lit)), axis=1).astype(np.uint16)
        rollout = SimpleNamespace(observations=Planes(np.arange(n + 1, dtype=np.int64) * lit, indices.reshape(-1),
                                                      rng.random(n * lit).astype(np.float16)))
        inputs = {"own": torch.randn(n), "pooled": torch.randn(n, 8)}
        oracle = (rng.random((n, 32, POSITIONS)) < 0.1).astype(np.uint8)
        returns = torch.randn(n)
        net = SimpleNamespace(planes=PLANES, channels=8, ours=SimpleNamespace(attention=True))
        args = SimpleNamespace(amp=False, chunk=16, batch=8, critic_lr=1e-3, critic_epochs=2, value_weight=0.5,
                               still_falling=0.002, tower_channels=16, tower_blocks=1, seed=5)
        seen = []
        real_fit, real_ask = noise_pretest.fit_corrections, noise_pretest.corrections_of
        real_read = noise_pretest.staged

        def fitting(net, rollout, pooled, oracle, target, taught, checked, *rest):
            seen.append(("fit", np.array(taught), np.array(checked)))
            answer = real_fit(net, rollout, pooled, oracle, target, taught, checked, *rest)
            seen.append(("fitted",))
            return answer

        def asking(heads, rollout, pooled, oracle, rows, *rest):
            seen.append(("asked", np.array(rows)))
            return real_ask(heads, rollout, pooled, oracle, rows, *rest)

        def reading(rollout, oracle, batches, *rest):
            seen.append(("read", np.concatenate(batches)))
            return real_read(rollout, oracle, batches, *rest)

        with patch.object(noise_pretest, "fit_corrections", side_effect=fitting), \
                patch.object(noise_pretest, "corrections_of", side_effect=asking), \
                patch.object(noise_pretest, "staged", side_effect=reading):
            baselines, fits = noise_pretest.cross_fitted_corrections(net, rollout, inputs, oracle, returns, half,
                                                                     check, args, "cpu")
        self.assertEqual([entry[0] for entry in seen].count("fit"), 2)
        sides, answered, fitting_now = [], [], False
        for entry in seen:
            if entry[0] == "fit":
                side = int(half[entry[1][0]])
                sides.append(side)
                taught = set(np.flatnonzero((half == side) & ~check).tolist())
                checked = set(np.flatnonzero((half == side) & check).tolist())
                self.assertEqual(sorted(entry[1].tolist()), sorted(taught))
                self.assertEqual(sorted(entry[2].tolist()), sorted(checked))
                self.assertTrue(taught and checked)
                fitting_now = True
                continue
            if entry[0] == "fitted":
                fitting_now = False
                continue
            rows = set(entry[1].tolist())
            if fitting_now:
                # Minibatches of the fit deals, an epoch a read; checks of
                # the check deals.
                self.assertIn(rows, (taught, checked) if entry[0] == "read" else (checked,))
            else:
                self.assertEqual(rows, set(np.flatnonzero(half != side).tolist()))
                self.assertEqual(len(entry[1]), int((half != side).sum()))
                answered.append(side)
        self.assertEqual(sides, [0, 1])
        # Each half's terms asked about the other half once, and it read
        # once for them.
        self.assertEqual(answered, [0, 0, 1, 1])
        self.assertEqual([(fit["half"], fit["taught_rows"], fit["checked_rows"]) for fit in fits],
                         [(side, int(((half == side) & ~check).sum()), int(((half == side) & check).sum()))
                          for side in (0, 1)])
        for name in ("oracle", "control"):
            self.assertTrue(torch.isfinite(baselines[name]).all())


class CloudLaunch(unittest.TestCase):
    """The cloud's side, run locally with Modal stood in for: the launch
    names the pre-test's options, writes only its own folder of the
    volume, and keeps what the child wrote whether or not it succeeded."""

    WRITTEN = ("report.json", "ledger.npz", "baselines.npz", "gram.npz")

    def launch(self, app, volume: Path, scratch: Path, popen, **kwargs):
        from neural import cloud_runs

        with patch.object(app, "VOLUME", volume), \
                patch.object(app, "workspace", functools.partial(cloud_runs.workspace, root=scratch)), \
                patch.object(app, "_environment", return_value={}), \
                patch.object(app.subprocess, "Popen", side_effect=popen), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return app.noise_pretest(**kwargs)

    @staticmethod
    def volume(root: Path) -> Path:
        from neural.checkpoints import atomic_save

        volume = root / "volume"
        atomic_save({"generation": 2820}, volume / "fusion-place-a005" / "history" / "gen-02820.pt")
        atomic_save({"generation": 30}, volume / "zoo" / "mortal_298k.pt")
        return volume

    @staticmethod
    def files(volume: Path) -> set[str]:
        return {path.relative_to(volume).as_posix() for path in volume.rglob("*") if path.is_file()}

    def test_the_launch_names_its_options_and_keeps_what_the_child_wrote(self):
        from neural.tests.test_cloud_isolation import child, controller

        app = controller()
        asked = []
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            volume = self.volume(root)
            before = self.files(volume)

            def popen(command, **_kwargs):
                asked.append(command)
                out = Path(command[command.index("--out") + 1])
                out.mkdir(parents=True)
                for name in self.WRITTEN:
                    (out / name).write_text(name, encoding="utf-8")
                return child(stdout='{"phase": "play"}\nVERDICT {"oracle": {"verdict": "kill"}}\n')

            answer = self.launch(app, volume, root / "scratch", popen,
                                 checkpoint="fusion-place-a005/history/gen-02820",
                                 opponents=["zoo/mortal_298k", "club"], games=64, groups=4,
                                 gradient_chunk=128, critic_epochs=2, preview_reach=False, max_minutes=40.0)
            target = volume / "noise-pretest" / "g2820"
            self.assertTrue(answer.rstrip().endswith(f"wrote {target}"))
            self.assertIn("VERDICT", answer)
            for name in self.WRITTEN:
                self.assertEqual((target / name).read_text(encoding="utf-8"), name)
            self.assertEqual(self.files(volume) - before, {f"noise-pretest/g2820/{name}" for name in self.WRITTEN})
            self.assertEqual(before - self.files(volume), set())
            self.assertEqual(list((root / "scratch").iterdir()), [])
        command, = asked
        self.assertEqual(command[1:3], ["-m", "neural.noise_pretest"])
        for flag, value in (("--games", "64"), ("--groups", "4"), ("--gradient-chunk", "128"),
                            ("--critic-epochs", "2"), ("--seed", "9700000"), ("--seat-share", "0.5"),
                            ("--opponent-share", "0.0"), ("--max-minutes", "40.0")):
            self.assertEqual(command[command.index(flag) + 1], value, flag)
        self.assertIn("--amp", command)
        self.assertIn("--no-preview-reach", command)
        # What the cloud passes is what the pre-test reads.
        args = noise_pretest.parse_args(command[3:])
        self.assertEqual((args.games, args.groups, args.gradient_chunk, args.critic_epochs, args.seed),
                         (64, 4, 128, 2, 9_700_000))
        self.assertEqual((args.seat_share, args.opponent_share, args.amp, args.preview_reach), (0.5, 0.0, True, False))
        self.assertEqual(args.max_minutes, 40.0)
        self.assertEqual([path.name for path in args.opponents], ["zoo--mortal_298k.pt", "club"])

    def test_a_child_that_fails_still_leaves_what_it_wrote(self):
        """And only what it wrote whole: a hidden copy it was writing when
        it stopped stays behind. The registered epochs and minutes are what
        the cloud passes unless told otherwise."""
        from neural.tests.test_cloud_isolation import child, controller

        app = controller()
        asked = []
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            volume = self.volume(root)

            def popen(command, **_kwargs):
                asked.append(command)
                out = Path(command[command.index("--out") + 1])
                out.mkdir(parents=True)
                (out / "report.json").write_text('{"verdict": {"oracle": {"verdict": "pending"}}}',
                                                 encoding="utf-8")
                (out / "baselines.npz").write_bytes(b"baselines")
                (out / ".gram.npz.x1y2z3.partial").write_bytes(b"gr")
                return child(stdout='{"phase": "readings"}\n', stderr="Traceback: out of memory", code=1)

            with self.assertRaises(subprocess.CalledProcessError) as failed:
                self.launch(app, volume, root / "scratch", popen, checkpoint="fusion-place-a005/history/gen-02820",
                            name="g2820-try", opponents=["zoo/mortal_298k", "club"])
            self.assertIn("out of memory", failed.exception.output)
            target = volume / "noise-pretest" / "g2820-try"
            self.assertEqual(sorted(path.name for path in target.iterdir()), ["baselines.npz", "report.json"])
            self.assertEqual(list((root / "scratch").iterdir()), [])
        command, = asked
        args = noise_pretest.parse_args(command[3:])
        self.assertEqual((args.critic_epochs, args.max_minutes, args.games, args.groups, args.gradient_chunk),
                         (10, 75.0, 4096, 20, 512))

    def test_a_folder_that_already_holds_a_calls_files_is_refused_before_the_child(self):
        from neural.tests.test_cloud_isolation import controller

        app = controller()
        asked = []
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            volume = self.volume(root)
            earlier = volume / "noise-pretest" / "g2820"
            earlier.mkdir(parents=True)
            (earlier / "report.json").write_text("an earlier call's", encoding="utf-8")
            # Named by the generation, or by the caller.
            for name in ("", "g2820"):
                with self.subTest(name=name), self.assertRaisesRegex(FileExistsError, "afresh"):
                    self.launch(app, volume, root / "scratch", asked.append,
                                checkpoint="fusion-place-a005/history/gen-02820", name=name,
                                opponents=["zoo/mortal_298k", "club"])
            self.assertEqual((earlier / "report.json").read_text(encoding="utf-8"), "an earlier call's")
            self.assertEqual(sorted(path.name for path in earlier.iterdir()), ["report.json"])
            self.assertEqual(list((root / "scratch").iterdir()), [])
        self.assertEqual(asked, [])

    def test_a_missing_checkpoint_or_a_bad_name_fails_before_any_work(self):
        from neural.tests.test_cloud_isolation import controller

        app = controller()
        asked = []
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            volume = self.volume(root)
            with self.assertRaises(FileNotFoundError):
                self.launch(app, volume, root / "scratch", asked.append, checkpoint="fusion-place-a005/missing",
                            opponents=["zoo/mortal_298k", "club"])
            for checkpoint, name in (("fusion-place-a005/history/gen-02820", "../elsewhere"),
                                     ("/fusion-place-a005/history/gen-02820", "g2820")):
                with self.subTest(checkpoint=checkpoint, name=name), self.assertRaises(ValueError):
                    self.launch(app, volume, root / "scratch", asked.append, checkpoint=checkpoint, name=name,
                                opponents=["zoo/mortal_298k", "club"])
            self.assertFalse((volume / "noise-pretest").exists())
        self.assertEqual(asked, [])


class SmallRun(unittest.TestCase):
    """The whole pre-test on tiny networks and a few games: it plays,
    keeping the oracle's planes, pays the reward the checkpoint carries,
    fits the four baselines, reads the residuals and the twin ceiling,
    takes the gradient, and writes everything it says it writes; and the
    verdict's numbers are the ones its files hold. A run whose preflight
    is over its minutes stops before it plays."""

    GROUPS = 4

    def close(self, read, wanted, what: str) -> None:
        """`read` from the report, kept to six significant figures, against
        `wanted` read again from the files."""
        self.assertLessEqual(abs(read - wanted), 1e-5 * abs(wanted) + 1e-12, what)

    def jackknifed(self, statistic, group: np.ndarray) -> tuple[float, float]:
        """`statistic(rows)` over every decision, and its delete-a-group
        jackknife error over the groups."""
        left = np.array([statistic(group != g) for g in range(self.GROUPS)])
        error = math.sqrt((self.GROUPS - 1) / self.GROUPS * ((left - left.mean()) ** 2).sum())
        return statistic(np.ones(len(group), dtype=bool)), error

    def test_a_small_pretest_runs_end_to_end(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
                root = Path(folder)
                torch.manual_seed(13)
                # A checkpoint with its optimiser's state, as the trainer
                # writes one, then paid the placement reward as a placement
                # run's checkpoint carries it.
                run(train_combined, root / "run", flags(train_combined, root, resume=False))
                payload = torch.load(root / "run" / "latest.pt", map_location="cpu", weights_only=False)
                with self.assertRaisesRegex(SystemExit, "no placement reward"):
                    noise_pretest.carried_reward(payload)
                payload["reward"] = {"kind": "placement", "alpha": 0.05, "since": 1, "warmed": 0, "refits": []}
                payload["place_model"] = place_model.PlaceModel(("zoo/mortal_298k", "club"), hidden=8).state()
                checkpoint = root / "placed.pt"
                torch.save(payload, checkpoint)
                opponent = mortal_checkpoint(root / "zoo--mortal_298k.pt")
                # A seed whose round keeps the Club in its pool (see
                # test_placement_pretest's Seeds).
                options = [
                    str(checkpoint), "--opponents", str(opponent), "club", "--seat-share", "0.5",
                    "--games", "8", "--seed", "4323", "--groups", str(self.GROUPS), "--critic-epochs", "2",
                    "--batch", "64", "--chunk", "64", "--gradient-chunk", "64", "--tower-channels", "16",
                    "--tower-blocks", "1", "--device", "cpu",
                ]
                over = root / "over"
                with self.assertRaisesRegex(SystemExit, "over --max-minutes 0.001"), \
                        contextlib.redirect_stdout(io.StringIO()) as printed:
                    noise_pretest.main([*options, "--max-minutes", "0.001", "--out", str(over)])
                self.assertIn('"budget"', printed.getvalue())
                self.assertNotIn('"phase": "play"', printed.getvalue())
                self.assertEqual(list(over.iterdir()), [])
                out = root / "out"
                report = noise_pretest.main([*options, "--out", str(out)])
                written = json.loads((out / "report.json").read_text(encoding="utf-8"))
                ledger = Ledger.load(out / "ledger.npz")
                with np.load(out / "baselines.npz") as saved:
                    arrays = {name: saved[name] for name in saved.files}
                with np.load(out / "gram.npz") as saved:
                    grams = {name: saved[name] for name in saved.files}
                files = sorted(path.name for path in out.iterdir())
        finally:
            torch.set_num_threads(threads)
        # Every file renamed into place, none left half written.
        self.assertEqual(files, ["baselines.npz", "gram.npz", "ledger.npz", "report.json"])
        self.assertEqual(written["verdict"], json.loads(json.dumps(report["verdict"])))
        self.assertIn(report["verdict"]["oracle"]["verdict"], {"go", "kill", "unread"})
        self.assertIn(report["verdict"]["duplicates"]["verdict"], {"build a twin round next", "stop"})
        decisions = report["round"]["decisions"]
        self.assertEqual(len(ledger), decisions)
        self.assertGreater(report["round"]["steps"], 0)
        self.assertGreater(report["round"]["oracle_seconds"], 0.0)
        self.assertEqual(report["members"], ["zoo/mortal_298k", "club"])
        self.assertEqual(report["reward"], {"kind": "placement", "alpha": 0.05})
        for name in ("returns", *noise_pretest.BASELINES, "log_probs", "differ", "stage", "tiles_left", "group"):
            with self.subTest(array=name):
                self.assertEqual(len(arrays[name]), decisions)
        for name in noise_pretest.BASELINES:
            self.assertTrue(np.isfinite(arrays[name]).all(), name)
        self.assertTrue(((arrays["differ"] >= 0) & (arrays["differ"] < 1)).all())
        # Mortal's plane of the tiles left is the one counted: one number a
        # row, agreeing with the oracle's draws wherever fewer than sixteen
        # are left.
        self.assertEqual(report["tiles_left_check"]["unread_rows"], 0)
        self.assertGreater(report["tiles_left_check"]["rows"], 0)
        self.assertEqual(report["tiles_left_check"]["agreeing"], 1.0)
        self.assertTrue(((arrays["tiles_left"] >= 0) & (arrays["tiles_left"] <= 69)).all())
        self.assertEqual(set(report["residual"]), set(noise_pretest.BASELINES))
        self.assertIsNotNone(report["residual"]["own"]["weighted"]["all"]["all"]["value"])
        self.assertEqual(len(report["fits"]), 2)
        for fit in report["fits"]:
            for name in ("oracle", "control"):
                self.assertGreaterEqual(len(fit[name]["check_errors"]), 2)
        self.assertEqual(grams["plain"].shape, (12, 12))
        self.assertEqual(grams["preconditioned"].shape, (12, 12))
        self.assertEqual(report["gradient"]["preconditioner"], "the checkpoint's AdamW second moments")
        self.assertEqual(set(report["gradient"]["plain"]["N"]), set(noise_pretest.ARMS))
        self.assertIn(report["preflight"]["how"], ("batched", "one at a time"))
        self.assertEqual(report["preflight"]["budget"]["most"], 75.0)
        self.assertLessEqual(report["preflight"]["budget"]["minutes"], 75.0)
        # The twin ceiling the verdict reads is the one game by game.
        twins = report["twins"]
        self.assertTrue(0 <= twins["by_game"]["share"] <= twins["by_seat"]["share"] <= 1)
        self.assertEqual(report["verdict"]["duplicates"]["ceiling"], twins["by_game"]["share"])
        self.assertEqual(report["verdict"]["duplicates"]["by_seat"], twins["by_seat"]["share"])

        # Every move play took, against the chance two draws differ that the
        # ceiling is read from: 1 - Σπ² lies between 2π(1 - π) and 1 - π²
        # for the move's own π, if both came from one policy.
        chance = np.exp(arrays["log_probs"].astype(np.float64))
        differ = arrays["differ"].astype(np.float64)
        self.assertTrue((differ >= 2 * chance * (1 - chance) - 1e-6).all())
        self.assertTrue((differ <= 1 - chance ** 2 + 1e-6).all())

        # The oracle's verdict read again from the files: the weighted
        # information cut against the better of the control and own, and the
        # gate, from the baselines; N(oracle)/N(own) from the Gram matrix.
        oracle = report["verdict"]["oracle"]
        returns = arrays["returns"].astype(np.float64)
        weights = np.expm1(arrays["log_probs"].astype(np.float64)) ** 2
        group = arrays["group"].astype(np.int64)

        def spread(name: str, rows: np.ndarray) -> float:
            residual, weight = (returns - arrays[name].astype(np.float64))[rows], weights[rows]
            mean = (weight * residual).sum() / weight.sum()
            return float((weight * (residual - mean) ** 2).sum() / weight.sum())

        cut, error = self.jackknifed(
            lambda rows: 1.0 - spread("oracle", rows) / min(spread("control", rows), spread("own", rows)), group)
        self.close(oracle["information_cut"]["value"], cut, "information cut")
        self.close(oracle["information_cut"]["se"], error, "its error")
        everyone = np.ones(len(returns), dtype=bool)
        control, own = spread("control", everyone), spread("own", everyone)
        self.assertEqual(oracle["information_cut"]["against"], "control" if control <= own else "own")
        self.close(oracle["gate"]["control"], control, "the gate's control")
        self.close(oracle["gate"]["own"], own, "the gate's own")
        self.assertEqual(oracle["gate"]["passed"], control <= 1.02 * own)
        self.assertEqual(report["cuts"]["information"]["weighted"]["all"]["all"],
                         {name: oracle["information_cut"][name] for name in ("value", "se")})

        def noise(arm: str, kept: np.ndarray) -> float:
            rows = noise_pretest.ARMS.index(arm) * self.GROUPS + kept
            block = grams["plain"][np.ix_(rows, rows)]
            return float((np.trace(block) - block.sum() / len(rows)) / (len(rows) - 1))

        everyone = np.arange(self.GROUPS)
        ratio = noise("oracle", everyone) / noise("own", everyone)
        left = np.array([noise("oracle", np.delete(everyone, g)) / noise("own", np.delete(everyone, g))
                         for g in range(self.GROUPS)])
        error = math.sqrt((self.GROUPS - 1) / self.GROUPS * ((left - left.mean()) ** 2).sum())
        self.close(oracle["noise_ratio"]["value"], ratio, "N(oracle)/N(own)")
        self.close(oracle["noise_ratio"]["se"], error, "its error")


if __name__ == "__main__":
    unittest.main()
