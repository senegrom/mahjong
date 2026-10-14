"""Reading a ruler look by look (`neural.looks`): Pocock's bounds as published,
an error of 5% over all four looks with no difference, a rule fixed before
the first seed and read in its own order, a look that waits for every seed
it covers, and the difference oriented as the reports' players are named."""

import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from neural import looks


def report(seed: int, by_deal: list[float], **extra) -> dict:
    """A single ruler's report as `neural.duel` writes one."""
    return {"seed": seed, "by_deal": by_deal, "placement": sum(by_deal) / len(by_deal), **extra}


def paired(seed: int, first: list[float], second: list[float]) -> dict:
    """A paired call's report as `neural.ruler` writes one."""
    return {"seed": seed, "arms": [{"challenger": "a", "by_deal": first}, {"challenger": "b", "by_deal": second}]}


class BoundTests(unittest.TestCase):
    def test_the_bounds_are_pococks(self):
        # Pocock (1977), two-sided 5%, equally spaced looks.
        for count, published in ((1, 1.960), (2, 2.178), (3, 2.289), (4, 2.361)):
            with self.subTest(looks=count):
                self.assertAlmostEqual(looks.pocock_bound(count), published, delta=0.0015)

    def test_with_no_difference_any_look_crosses_one_time_in_twenty(self):
        bound = looks.pocock_bound(4)
        stops = looks.stopping(bound, 4)
        self.assertAlmostEqual(sum(up + down for up, down in stops), 0.05, delta=0.0005)
        # Symmetric, and each look stops fewer than the first, which alone is 1.8%.
        for up, down in stops:
            self.assertAlmostEqual(up, down, places=9)
        self.assertAlmostEqual(sum(stops[0]), 0.0182, delta=0.0005)
        # One look at 1.96 crosses 5% of the time, and four at 1.96 would cross far more.
        self.assertAlmostEqual(sum(looks.stopping(1.96, 1)[0]), 0.05, delta=0.0005)
        self.assertGreater(sum(up + down for up, down in looks.stopping(1.96, 4)), 0.12)

    def test_a_real_difference_stops_on_its_own_side_and_more_often_with_more_seeds(self):
        bound = looks.pocock_bound(4)
        power = {}
        for drift in (0.5, 1.0, 2.0):
            stops = looks.stopping(bound, 4, drift=drift)
            power[drift] = sum(up for up, _ in stops)
            self.assertLess(sum(down for _, down in stops), 0.01)
        self.assertLess(power[0.5], power[1.0])
        self.assertLess(power[1.0], power[2.0])
        # Two standard errors a look's worth of deals is four by the last look.
        self.assertGreater(power[2.0], 0.95)

    def test_the_whole_rule_holds_its_error_with_no_difference(self):
        """Series simulated end to end through `read`, with the spread
        estimated from the deals as a real series has it: about one in
        twenty stops, on either side alike."""
        rule = looks.design()
        rng = random.Random(20261014)
        stopped = {"first": 0, "second": 0, "level": 0}
        trials = 1500
        for _ in range(trials):
            by_seed = {seed: [rng.gauss(0.0, 0.5) for _ in range(24)] for seed in rule.seeds}
            _taken, state = looks.read(rule, by_seed)
            stopped[state] += 1
        rate = (stopped["first"] + stopped["second"]) / trials
        self.assertGreater(rate, 0.03)
        self.assertLess(rate, 0.075)


class DesignTests(unittest.TestCase):
    def test_twenty_seeds_by_default_and_forty_for_a_small_effect(self):
        usual = looks.design()
        self.assertEqual((usual.per_look, usual.looks, len(usual.seeds)), (5, 4, 20))
        self.assertAlmostEqual(usual.bound, 2.3613, places=3)
        self.assertEqual(usual.seeds[:3], [9_100_000, 9_200_003, 9_300_006])
        for expected in (0.006, -0.004, 0.0):
            small = looks.design(expected)
            self.assertEqual((small.per_look, small.looks, len(small.seeds)), (10, 4, 40))
            self.assertEqual(small.bound, usual.bound)
        self.assertEqual(looks.design(0.01).per_look, 5)

    def test_a_design_is_kept_as_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "design.json"
            rule = looks.design(0.005, "910 against 810", first_seed=1_000)
            rule.save(path)
            again = looks.Design.load(path)
            self.assertEqual(again, rule)
            self.assertEqual(again.seeds[0], 1_000)
            self.assertEqual(json.loads(path.read_text())["what"], "910 against 810")

    def test_nonsense_is_refused(self):
        for bad in ({"per_look": 0}, {"looks": -1}, {"step": 0}, {"alpha": 1.5}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                looks.Design(**bad)


class ReadTests(unittest.TestCase):
    def rule(self) -> looks.Design:
        return looks.Design(first_seed=100, step=10, per_look=2, looks=3)

    def test_a_look_waits_for_every_seed_it_covers(self):
        rule = self.rule()
        # Seeds 100 and 120 are in, 110 is not: the first look covers 100
        # and 110, so nothing is read, however clear 120 is.
        taken, state = looks.read(rule, {100: [1.0, 1.1], 120: [5.0, 5.0]})
        self.assertEqual((taken, state), ([], "waiting"))

    def test_the_first_look_past_the_bound_stops_it_on_its_side(self):
        rule = self.rule()
        rng = random.Random(3)
        strong = {seed: [0.5 + rng.gauss(0, 0.1) for _ in range(50)] for seed in rule.seeds}
        taken, state = looks.read(rule, strong)
        self.assertEqual((len(taken), state), (1, "first"))
        self.assertEqual((taken[0].seeds, taken[0].deals), (2, 100))
        weak = {seed: [-value for value in values] for seed, values in strong.items()}
        taken, state = looks.read(rule, weak)
        self.assertEqual((len(taken), state, taken[0].decision), (1, "second", "second"))

    def test_no_crossing_reads_level_at_the_last_look(self):
        rule = self.rule()
        rng = random.Random(4)
        level = {seed: [rng.gauss(0, 0.5) for _ in range(40)] for seed in rule.seeds}
        # Seeds outside the rule are left out of every look.
        level[999] = [100.0] * 40
        taken, state = looks.read(rule, level)
        self.assertEqual(state, "level")
        self.assertEqual([look.decision for look in taken], ["go on", "go on", "level"])
        self.assertEqual(taken[-1].deals, 6 * 40)
        # Two players that played every deal alike read level with no error.
        same = {seed: [0.0] * 40 for seed in rule.seeds}
        taken, state = looks.read(rule, same)
        self.assertEqual(state, "level")
        self.assertEqual((taken[-1].edge, taken[-1].error, taken[-1].z), (0.0, 0.0, 0.0))

    def test_a_look_reads_every_deal_so_far(self):
        rule = self.rule()
        by_seed = {100: [0.4, -0.4], 110: [0.3, -0.1], 120: [0.2, -0.2], 130: [0.6, -0.2]}
        taken, state = looks.read(rule, by_seed)
        self.assertEqual(state, "waiting")
        self.assertAlmostEqual(taken[0].edge, 0.05)
        self.assertAlmostEqual(taken[1].edge, 0.075)
        self.assertEqual((taken[0].deals, taken[1].deals), (4, 8))
        spread = (sum((value - 0.075) ** 2 for values in by_seed.values() for value in values) / 7) ** 0.5
        self.assertAlmostEqual(taken[1].error, spread / 8 ** 0.5)


class DifferenceTests(unittest.TestCase):
    def test_positive_when_the_first_player_is_the_stronger(self):
        # A placement below 2.5 is the challenger ahead.
        self.assertEqual(looks.differences(report(1, [2.0, 3.0])), [0.5, -0.5])
        # The first arm placed 2.0 where the second placed 2.5: the first ahead.
        self.assertEqual(looks.differences(paired(1, [2.0, 3.0], [2.5, 2.5])), [0.5, -0.5])
        self.assertEqual(looks.against(report(1, [2.0, 3.0]), report(1, [2.5, 2.5])), [0.5, -0.5])

    def test_reports_of_other_deals_are_not_paired(self):
        with self.assertRaises(ValueError):
            looks.against(report(1, [2.0, 3.0]), report(2, [2.5, 2.5]))
        with self.assertRaises(ValueError):
            looks.against(report(1, [2.0, 3.0]), report(1, [2.5]))
        with self.assertRaises(ValueError):
            looks.differences(paired(1, [2.0, 3.0], [2.5]))


class CommandLineTests(unittest.TestCase):
    def run_main(self, *arguments) -> str:
        out = io.StringIO()
        with patch.object(sys, "argv", ["looks", *map(str, arguments)]), contextlib.redirect_stdout(out):
            looks.main()
        return out.getvalue()

    def test_a_series_is_read_by_its_registered_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rule = looks.Design(first_seed=7, step=1, per_look=1, looks=2, what="a against b")
            rule.save(root / "design.json")
            for seed in (7, 8):
                (root / f"seed-{seed}.json").write_text(json.dumps(paired(seed, [2.0] * 30, [3.0] * 29 + [2.9])))
            text = self.run_main(root)
            self.assertIn("comparing: a against b", text)
            self.assertIn("stopped at look 1: a is the stronger", text)
            answer = json.loads(self.run_main(root, "--json"))
            self.assertEqual(answer["state"], "first")
            self.assertEqual(answer["design"]["per_look"], 1)

    def test_two_series_of_single_rulers_are_read_deal_by_deal(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "910", Path(folder) / "810"
            first.mkdir()
            second.mkdir()
            rng = random.Random(9)
            for seed in looks.design().seeds:
                deals = [rng.choice([1.0, 2.0, 3.0, 4.0]) for _ in range(20)]
                # The same deals, placed the same: the two read level with no error.
                (first / f"seed-{seed}.json").write_text(json.dumps(report(seed, deals, challenger="910")))
                (second / f"seed-{seed}.json").write_text(json.dumps(report(seed, deals, challenger="810")))
            text = self.run_main(first, "--against", second)
            self.assertIn("not registered", text)
            self.assertIn("level: no look crossed", text)


if __name__ == "__main__":
    unittest.main()
