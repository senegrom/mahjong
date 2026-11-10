"""Reading a ruler look by look (`neural.looks`): Pocock's bounds as published,
an error of 5% over all four looks with no difference, a rule fixed before
the first seed and read in its own order, another rule registered as asked
and read at its own bound, a look that waits for every seed it covers, the
difference oriented as the reports' players are named, and no look read of
a series a report of which fails the all-last dama rule's check or a late
rule's, a late rule held to its own floor (fired for the decline, judged for
a rule not named), each ruled arm named by its rule."""

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


#: What a challenger that follows the all-last dama rule says of it (see
#: `neural.all_last`).
ACCOUNT = {"riichi": 3, "fired": 2, "tables": [0, 5], "hands": 2, "kept": {"no_yaku": 1, "short": 0, "dead": 0},
           "riichi_after_firing": 0}


def ruled(seed: int) -> dict:
    """A paired call of 1410 with the all-last dama rule, the first, and
    plain, as `neural.ruler` writes one: the first a little the stronger,
    its account of the rule, and the pair's check, passed."""
    kept = paired(seed, [2.0] * 30, [3.0] * 29 + [2.9])
    for arm in kept["arms"]:
        arm["challenger"] = "history/gen-01410"
    kept["arms"][0]["all_last"] = dict(ACCOUNT)
    kept["all_last_check"] = {"differing": 2, "violations": []}
    return kept


#: What a challenger that follows a late rule says of it (see
#: `neural.late_rules`).
LATE = {"rule": "decline-last", "judged": 4, "fired": 1, "tables": [3], "hands": 1, "kept": {"not_last": 3},
        "kept_after_firing": 0}


def late(seed: int, rule: str = "decline-last") -> dict:
    """A paired call of 1410 with a late rule, the first, and plain, as
    `neural.ruler` writes one: the first a little the stronger, its account
    of the rule, and the pair's check, passed."""
    kept = paired(seed, [2.0] * 30, [3.0] * 29 + [2.9])
    for arm in kept["arms"]:
        arm["challenger"] = "history/gen-01410"
    kept["arms"][0]["late_rule"] = {**LATE, "rule": rule}
    kept["late_rule_check"] = {"differing": 1, "violations": []}
    return kept


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

    def test_another_rule_is_registered_as_asked_and_the_usual_one_is_unchanged(self):
        usual = looks.design(0.004, "a against b")
        self.assertEqual(looks.design(0.004, "a against b", looks=None, per_look=None, bound=0.0), usual)
        self.assertEqual((usual.per_look, usual.looks, usual.bound), (10, 4, looks.pocock_bound(4)))
        one = looks.design(0.001, "a against b", looks=1, per_look=20, bound=2.0)
        self.assertEqual((one.per_look, one.looks, one.bound, one.expected), (20, 1, 2.0, 0.001))
        self.assertEqual(one.seeds, looks.design().seeds)
        # Pocock's for one look, where only the looks are named.
        self.assertAlmostEqual(looks.design(looks=1, per_look=20).bound, 1.96, delta=0.0015)

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

    def test_a_rule_of_one_look_reads_first_only_above_its_bound(self):
        """Read once, at 2.0: z just above it reads first, z just below
        level; the usual rule's first look reads a z it would stop at as
        going on. Each seed's fifty deals sit one either side of a centre,
        so z is the centre times the root of one less than the deals read."""
        rule = looks.design(looks=1, per_look=2, bound=2.0)
        usual = looks.design()

        def by_seed(z: float, seeds: list[int]) -> dict[int, list[float]]:
            centre = z / (50 * len(seeds) - 1) ** 0.5
            return {seed: [centre + 1.0, centre - 1.0] * 25 for seed in seeds}

        for z, want in ((2.01, "first"), (1.99, "level"), (-2.01, "second")):
            with self.subTest(z=z):
                taken, state = looks.read(rule, by_seed(z, rule.seeds))
                self.assertAlmostEqual(taken[0].z, z, places=6)
                self.assertEqual((len(taken), state), (1, want))
        taken, state = looks.read(usual, by_seed(2.2, usual.seeds[:5]))
        self.assertAlmostEqual(taken[0].z, 2.2, places=6)
        self.assertEqual((taken[0].decision, state), ("go on", "waiting"))

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

    def test_a_series_with_the_rules_at_every_seat_names_its_incumbent_with_them(self):
        rules = {"all_last": dict(ACCOUNT), "late_rule": {"rule": "decline-last", "judged": 2, "fired": 1}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            looks.Design(first_seed=7, step=1, per_look=1, looks=1).save(root / "design.json")
            (root / "seed-7.json").write_text(json.dumps(report(7, [3.0] * 29 + [2.9], challenger="a",
                                                                incumbent="b", **rules, incumbent_rules=rules)))
            self.assertIn("stopped at look 1: b with the rules is the stronger", self.run_main(root))

    def test_one_checkpoint_averaging_over_the_suits_and_plain_reads_as_two(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            looks.Design(first_seed=7, step=1, per_look=1, looks=2).save(root / "design.json")
            for seed in (7, 8):
                kept = paired(seed, [2.0] * 30, [3.0] * 29 + [2.9])
                for arm in kept["arms"]:
                    arm["challenger"] = "history/gen-00810"
                kept["arms"][0]["suits"] = {"gate": 0.9, "asked": 100, "again": 18, "changed": 2}
                (root / f"seed-{seed}.json").write_text(json.dumps(kept))
            text = self.run_main(root)
            self.assertIn("stopped at look 1: history/gen-00810 averaged over the suits is the stronger", text)

    def test_one_checkpoint_with_the_all_last_dama_rule_and_plain_reads_as_two(self):
        """The ruled arm is named apart, and a rule registered with a bound
        of its own says so, where the usual one says Pocock's."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            looks.design(looks=1, per_look=2, bound=2.0).save(root / "design.json")
            for seed in looks.design().seeds[:2]:
                (root / f"seed-{seed}.json").write_text(json.dumps(ruled(seed)))
            text = self.run_main(root)
            self.assertIn("stop when |z| > 2.0 (the bound registered)", text)
            self.assertIn("stopped at look 1: history/gen-01410 with all-last dama is the stronger", text)
            (root / "design.json").unlink()
            self.assertIn("(Pocock, 5% two-sided over all looks)", self.run_main(root))

    def test_a_series_with_a_report_that_fails_the_all_last_check_is_refused(self):
        """By the arms the series was registered to rule, if it was, or else
        by what each report says: the arms played otherwise where the rule
        did not fire, the ruled arm never fired, the plain arm followed the
        rule, or the report says nothing of the check. A folder read against
        one with such a report is refused alike."""
        violation = {"deal": 7, "place": 2, "tables": [2007, 6007]}
        never = {**ACCOUNT, "fired": 0}
        seeds = looks.design().seeds[:2]
        for name, broken, registered, at, why in (
            ("violations", {"all_last_check": {"differing": 2, "violations": [violation]}}, "first", 1,
             "at 1 of the 2 deals and seats whose tables ended differently in the two arms, the rule fired at "
             "neither table"),
            ("never fired", {"first": never}, "first", 1, "the first challenger's rule never fired (3 riichi judged)"),
            ("never fired, never registered", {"first": never}, None, 1, "the first challenger's rule never fired"),
            ("plain arm ruled", {"second": ACCOUNT}, "first", 1, "the second challenger was to play plain"),
            ("unchecked", {"all_last_check": None}, "first", 1, "it says nothing of the check"),
            # Every report fails a registration that rules another arm, or none.
            ("registered for the second", {}, "second", 0, "the first challenger was to play plain"),
            ("registered plain", {}, "", 0, "the first challenger was to play plain"),
        ):
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                if registered is not None:
                    rule = {"all_last_dama": registered} if registered else {}
                    (root / "series.json").write_text(json.dumps({"also": "history/gen-01410", **rule}))
                for seed in seeds:
                    kept = ruled(seed)
                    if seed == seeds[1]:
                        for which, arm in zip(("first", "second"), kept["arms"]):
                            if which in broken:
                                arm["all_last"] = broken[which]
                        kept["all_last_check"] = broken.get("all_last_check", kept["all_last_check"])
                    (root / f"seed-{seed}.json").write_text(
                        json.dumps({key: value for key, value in kept.items() if value is not None}))
                with self.assertRaises(SystemExit) as stop:
                    self.run_main(root)
                self.assertIn(f"seed {seeds[at]}'s report fails the all-last dama check: {why}",
                              str(stop.exception.code))
                self.assertIn("The series is invalid, and none of its looks is read", str(stop.exception.code))
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "910", Path(folder) / "810"
            for root in (first, second):
                root.mkdir()
            for seed in looks.design().seeds:
                (first / f"seed-{seed}.json").write_text(json.dumps(report(seed, [2.0, 3.0], challenger="910")))
                (second / f"seed-{seed}.json").write_text(
                    json.dumps(report(seed, [2.0, 3.0], challenger="810", all_last=ACCOUNT)))
            with self.assertRaisesRegex(SystemExit, "810: seed 9100000's report fails the all-last dama check: "
                                                    "the rule is checked only in a paired call"):
                self.run_main(first, "--against", second)
            self.assertIn("level: no look crossed", self.run_main(first, "--against", first))

    def test_one_checkpoint_with_a_late_rule_and_plain_reads_as_two(self):
        """Each ruled arm is named by its rule, the plain arm as the
        checkpoint, and a series of one dama rule still reads as it did."""
        names = {"decline-last": "declining last-place wins"}
        self.assertEqual(set(looks.LATE), set(names))
        for rule, name in names.items():
            with self.subTest(rule=rule), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                looks.design(looks=1, per_look=2, bound=2.0).save(root / "design.json")
                (root / "series.json").write_text(json.dumps({"also": "history/gen-01410", "late_rule": rule}))
                for seed in looks.design().seeds[:2]:
                    (root / f"seed-{seed}.json").write_text(json.dumps(late(seed, rule)))
                text = self.run_main(root)
                self.assertIn(f"stopped at look 1: history/gen-01410 {name} is the stronger", text)
                self.assertEqual(looks.ruled(late(1, rule)["arms"][1]), "")
        self.assertEqual(looks.ruled(ruled(1)["arms"][0]), looks.RULED)
        self.assertEqual(looks.late_named("blind-dama"), " with the late rule blind-dama")

    def test_a_series_with_a_report_that_fails_a_late_rules_check_is_refused(self):
        """By the rule the series was registered to follow, if it was, or
        else by what each report says: the arms played otherwise where the
        rule did not fire, the ruled arm fell short of its rule's floor (the
        decline never fired, or judged nothing), it followed another rule,
        the plain arm followed one, or the report says nothing of the check.
        A rule not named in the floors is held only to having judged, so
        judging but never firing is no fault there."""
        violation = {"deal": 7, "place": 2, "tables": [2007, 6007]}
        never = {"fired": 0, "tables": [], "hands": 0}
        seeds = looks.design().seeds[:2]
        for name, broken, registered, at, why in (
            ("violations", {"late_rule_check": {"differing": 2, "violations": [violation]}}, "decline-last", 1,
             "at 1 of the 2 deals and seats whose tables ended differently in the two arms, the rule fired at "
             "neither table"),
            ("judged nothing", {"first": {**LATE, "judged": 0, "fired": 0}}, "decline-last", 1,
             "the first challenger's decline-last rule judged nothing"),
            ("judged nothing, never registered", {"first": {**LATE, "judged": 0, "fired": 0}}, None, 1,
             "the first challenger's decline-last rule judged nothing"),
            ("declined nothing", {"first": {**LATE, **never}}, "decline-last", 1,
             "the first challenger's decline-last rule never fired (4 decisions judged)"),
            ("declined nothing, never registered", {"first": {**LATE, **never}}, None, 1,
             "the first challenger's decline-last rule never fired"),
            ("another rule", {"first": {**LATE, "rule": "blind-dama"}}, "decline-last", 1,
             "the first challenger was to follow decline-last, and its report says it followed blind-dama"),
            ("plain arm ruled", {"second": LATE}, "decline-last", 1, "the second challenger was to play plain"),
            ("unchecked", {"late_rule_check": None}, "decline-last", 1, "it says nothing of the late rule's check"),
            # Every report fails a registration that follows no late rule.
            ("registered plain", {}, "", 0, "the first challenger was to play plain"),
        ):
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                if registered is not None:
                    rule = {"late_rule": registered} if registered else {}
                    (root / "series.json").write_text(json.dumps({"also": "history/gen-01410", **rule}))
                for seed in seeds:
                    kept = late(seed, registered or "decline-last")
                    if seed == seeds[1]:
                        for which, arm in zip(("first", "second"), kept["arms"]):
                            if which in broken:
                                arm["late_rule"] = broken[which]
                        kept["late_rule_check"] = broken.get("late_rule_check", kept["late_rule_check"])
                    (root / f"seed-{seed}.json").write_text(
                        json.dumps({key: value for key, value in kept.items() if value is not None}))
                with self.assertRaises(SystemExit) as stop:
                    self.run_main(root)
                self.assertIn(f"seed {seeds[at]}'s report fails the late rule's check: {why}",
                              str(stop.exception.code))
                self.assertIn("The series is invalid, and none of its looks is read", str(stop.exception.code))
        # Judged without firing, the arms the same: no fault in a rule not
        # named in the floors alone, by registration or by what the report
        # says.
        for rule, problem in (("blind-dama", None), ("decline-last", "rule never fired")):
            quiet = late(seeds[0], rule)
            quiet["arms"][0]["late_rule"] = {**LATE, "rule": rule, **never}
            quiet["late_rule_check"] = {"differing": 0, "violations": []}
            for registered in ({"late_rule": rule}, None):
                with self.subTest(rule=rule, registered=registered):
                    found = looks.failed_check({seeds[0]: quiet}, registered)
                    if problem is None:
                        self.assertIsNone(found)
                    else:
                        self.assertIn(problem, found)
        self.assertEqual(set(looks.LATE_FLOOR), set(looks.LATE))
        # A late rule's series is no dama series, and the other way round.
        self.assertIsNone(looks.failed_check({seeds[0]: late(seeds[0])}, {"late_rule": "decline-last"}))
        self.assertIsNone(looks.failed_check({seeds[0]: ruled(seeds[0])}, {"all_last_dama": "first"}))
        self.assertIn("the first challenger was to play plain",
                      looks.failed_check({seeds[0]: ruled(seeds[0])}, {"late_rule": "decline-last"}))

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

    def write(self, folder: Path, by_deal, settings: dict | None = None, **extra) -> Path:
        """A series of single rulers kept in `folder`, on the usual rule's
        seeds, each placing `by_deal(seed)`, registered as `settings` when
        they are given."""
        folder.mkdir()
        if settings is not None:
            (folder / "series.json").write_text(json.dumps(settings))
            looks.design().save(folder / "design.json")
        for seed in looks.design().seeds:
            (folder / f"seed-{seed}.json").write_text(json.dumps(
                report(seed, by_deal(seed), games_per_seat=20, **extra)))
        return folder

    def test_two_series_are_read_against_each_other_only_at_one_table(self):
        """`--against` reads two series deal by deal only once they are seen
        to have played at one table, from the registration of each that has
        one and from the reports of a folder never registered: the same
        incumbent, of the same run where it is one of a run's own, asked the
        same way, the rules of the last hand at both or neither, as many
        deals a seed. Read, the comparison says what was read."""
        rules = {"all_last": dict(ACCOUNT), "late_rule": {"rule": "decline-last", "judged": 2, "fired": 1}}
        mortal = {"challenger": "a", "incumbent": "zoo/mortal_298k", "run": "fusion-long", "games": 20,
                  "questions": "fixed", "also": "", "against": ""}
        level = lambda seed: [2.0, 3.0] * 10  # noqa: E731
        for name, second, settings, extra, why in (
            ("another incumbent", {**mortal, "incumbent": "history/gen-03000"}, mortal, {},
             "its incumbent is history/gen-03000, not zoo/mortal_298k"),
            ("asked otherwise", {**mortal, "questions": "seating"}, mortal, {},
             "it was asked seating and this series fixed"),
            ("another run's archive", {**mortal, "incumbent": "history/gen-03000", "run": "other"},
             {**mortal, "incumbent": "history/gen-03000"}, {},
             "its incumbent is other's history/gen-03000, not fusion-long's"),
            ("never registered, asked by seating", None, mortal, {"incumbent": "zoo/mortal_298k"},
             "it was asked seating and this series fixed"),
            ("never registered, ruled", None, {**mortal, "questions": "seating"},
             {"incumbent": "zoo/mortal_298k", **rules, "incumbent_rules": rules},
             "its players follow the rules of the last hand and this series' do not"),
        ):
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                first = self.write(Path(folder) / "first", level, settings)
                other = self.write(Path(folder) / "other", level, second, **extra)
                with self.assertRaises(SystemExit) as stop:
                    self.run_main(first, "--against", other)
                self.assertIn(f"cannot be read against {other}: {why}", str(stop.exception.code))
        with tempfile.TemporaryDirectory() as folder:
            first = self.write(Path(folder) / "first", level, mortal)
            other = self.write(Path(folder) / "other", level, {**mortal, "challenger": "b"})
            text = self.run_main(first, "--against", other)
            self.assertIn("a comparison first's rule was not registered for", text)
            self.assertIn("comparing: first against other, deal by deal, both at zoo/mortal_298k's table, "
                          "questions fixed", text)
            self.assertIn("level: no look crossed", text)


if __name__ == "__main__":
    unittest.main()
