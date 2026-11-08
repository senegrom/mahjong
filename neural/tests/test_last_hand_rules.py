"""Both kept rules of the last hand at every seat of a ruler (`neural.ruler
--last-hand-rules`), as the site's Trained AI plays any network:
decline-last asked first and the all-last dama rule around it.

The order is the site's and self-play's: a declined self-draw can leave a
riichi as the move, which dama must judge, where asked the other way round
the riichi would stand. A ruler seats every player so, the incumbent too,
only asked in fixed rows with the reach's tile previewed and never beside a
rule measured alone against plain play; its report says what each player's
rules did, the incumbent's under `incumbent_rules`, and checks no pair; a
rule that never fires leaves every game as it was, and one that fires does so
at the incumbent's seats as at the challenger's. The command seats it when
run as a module, and a series' reports are read only while each shows that
every player followed both rules (`looks.rules_problem`)."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

from neural import all_last, duel, late_rules, looks, ruler, zoo
from neural.tests.test_all_last import Answers, Coded, Told, steer_reaches
from neural.tests.test_bfloat16_players import checkpoints
from neural.tests.test_late_rules import LateCoded, moves, turn_row
from neural.tests.test_ruler import SEED, Taught, steer

KINDS = all_last.KINDS
FIXED = {"questions": "fixed", "rows": 4, "preview_reach": True}
#: As `test_late_rules.RulerTests` and `test_all_last.RulerTests` seat their players.
OPTIONS = {"questions": "fixed", "rows": 4, "skip_forced": True, "preview_reach": True}


class BothCodes:
    """An arena as both rules read it: the decline code of each game and
    player, and the dama codes of each game, a code a tile."""

    def __init__(self, decline: dict, dama: dict) -> None:
        self.decline = decline
        self.dama = dama
        self.asked: list[str] = []

    def all_last_decline(self, rows):
        self.asked.append("decline")
        return bytes(self.decline.get(tuple(row), 0) for row in rows)

    def all_last_dama(self, games):
        self.asked.append("dama")
        codes = np.zeros((len(games), KINDS), dtype=np.uint8)
        for row, game in enumerate(games):
            for tile, code in self.dama.get(game, {}).items():
                codes[row, tile] = code
        return codes.tobytes()

    def hands_done(self):
        return [0] * 8


#: A turn in game 5 of player 1, as `test_late_rules` has it: the self-draw
#: is best and leaves the player alone in last, riichi comes next, and its
#: preview throws the 3, which keeps the player first undeclared.
WHO = [(5, 1)]
LEGAL = np.stack([turn_row([3, 9, 20], [3, 9], tsumo=True)])
ROWS = {(5, 1): moves(agari=5.0, reach=4.0, d9=3.0)}
PREVIEWS = {(5, 1): moves(d3=2.0, d9=1.0)}


class OrderTests(unittest.TestCase):
    def asked(self, rule):
        """The turn's question, its reach previewed, through `rule`, a
        wrapper made of `Answers`; what it answered for the turn."""
        allowed = zoo.translatable(LEGAL)
        allowed_ahead = np.zeros((1, zoo.MORTAL_ACTIONS), dtype=bool)
        allowed_ahead[:, :KINDS] = LEGAL[[0], zoo.RIICHI_DISCARD:zoo.TSUMO]
        question = np.concatenate([allowed, allowed_ahead])
        arena = BothCodes({(5, 1): late_rules.LAST}, {5: {3: all_last.KEEPS_FIRST, 9: all_last.SHORT}})
        got, _own = rule(Answers(ROWS, PREVIEWS))(Told(arena), WHO, False, question, WHO)
        return got[0], arena.asked

    def test_decline_first_and_dama_then_judges_the_riichi_it_left(self):
        dama, decline = all_last.Account(), late_rules.Account("decline-last")
        got, asked = self.asked(lambda ask: all_last.ruled(late_rules.decline_last(ask, decline), dama))
        self.assertEqual(asked, ["decline", "dama"])
        self.assertEqual(np.flatnonzero(np.isfinite(got)).tolist(), [3])
        self.assertEqual((decline.fired, dama.riichi, dama.fired), (1, 1, 1))

    def test_asked_the_other_way_round_the_riichi_would_stand(self):
        dama, decline = all_last.Account(), late_rules.Account("decline-last")
        got, asked = self.asked(lambda ask: late_rules.decline_last(all_last.ruled(ask, dama), decline))
        self.assertEqual(asked, ["decline"])
        self.assertEqual(int(np.argmax(got)), zoo.MORTAL_RIICHI)
        self.assertEqual((decline.fired, dama.riichi), (1, 0))


class SeatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.paths = checkpoints(Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_every_kind_of_player_follows_decline_and_then_dama_around_it(self):
        for name in ("fused", "reheaded", "mortal"):
            with self.subTest(player=name):
                player = ruler.seat(self.paths[name], "cpu", **FIXED, last_hand_rules=True)
                self.assertEqual(late_rules.account_of(player), late_rules.Account("decline-last"))
                self.assertEqual(all_last.account_of(player), all_last.Account())
                # Dama outermost, decline the question it wraps.
                self.assertEqual(player._ask.__name__, "dama_where_riichi_buys_nothing")
                inner = [cell.cell_contents for cell in player._ask.__closure__ if callable(cell.cell_contents)]
                self.assertEqual([ask.__name__ for ask in inner], ["decline_a_win_that_leaves_it_last"])
                self.assertEqual(ruler.rules_of(player), {"all_last": all_last.Account().report(),
                                                          "late_rule": late_rules.Account("decline-last").report()})
        self.assertEqual(ruler.rules_of(ruler.seat(self.paths["fused"], "cpu", **FIXED)), {})

    def test_it_needs_its_switches_and_no_rule_apart(self):
        for options, why in (({**FIXED, "all_last_dama": True}, "takes neither"),
                             ({**FIXED, "late_rule": "decline-last"}, "takes neither"),
                             ({}, "fixed questions and the reach's tile previewed"),
                             ({"questions": "fixed", "rows": 4}, "fixed questions and the reach's tile previewed")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, why):
                ruler.seat(self.paths["fused"], "cpu", **options, last_hand_rules=True)
        # Dama goes around decline only once decline is followed, and decline
        # never inside dama.
        with self.assertRaisesRegex(ValueError, "follow it first"):
            all_last.dama(ruler.seat(self.paths["fused"], "cpu", **FIXED), fixed=True, around_decline=True)
        both = ruler.seat(self.paths["fused"], "cpu", **FIXED, last_hand_rules=True)
        with self.assertRaisesRegex(ValueError, "already follows a rule of the last hand"):
            late_rules.follow(both, "decline-last", fixed=True)
        with self.assertRaisesRegex(ValueError, "already leaves the last hand's riichi undeclared"):
            all_last.dama(both, fixed=True, around_decline=True)


@contextlib.contextmanager
def both_coded(dama: int):
    """Every ruler's views served from an arena whose dama codes are `dama`
    on every tile a reach may throw (`test_all_last.Coded`) and whose wins
    never leave a player last (`test_late_rules.LateCoded`), asking the
    heuristic player once a step; yields the views made."""
    made = []

    class CodedViews(Taught):
        def __init__(self, arena, *args, **kwargs):
            super().__init__(Coded(LateCoded(arena, False), dama), *args, **kwargs)
            made.append(self)

    with patch.object(duel, "Views", CodedViews):
        yield made


class RulerTests(unittest.TestCase):
    """Whole rulers of small networks on the processor, one deal a seating,
    every player pushed towards the heuristic player's moves and towards a
    reach wherever one is open, as the rules' own ruler tests push them."""

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.folder = tempfile.TemporaryDirectory()
        cls.paths = checkpoints(Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        cls.folder.cleanup()

    def seated(self, name: str, rules: bool = True):
        """As `ruler.seat(last_hand_rules=True)` seats a player, the pushes
        inside the rules, which stay its outermost question."""
        player = steer(steer_reaches(ruler.seat(self.paths[name], "cpu", **OPTIONS)))
        if rules:
            late_rules.follow(player, "decline-last", fixed=True)
            all_last.dama(player, fixed=True, around_decline=True)
        return player

    def play(self, dama: int, challengers=("fused",), rules: bool = True):
        with both_coded(dama) as made:
            got = ruler.ruler([self.seated(name, rules) for name in challengers], self.seated("mortal", rules), 1,
                              SEED, "cpu", questions="fixed")
        (views,) = made
        # The events of the last step, which no later step handed out.
        views.arena.mjai_all()
        return got, views.arena.events

    def test_rules_that_never_fire_judge_at_every_seat_and_change_no_game(self):
        got, _events = self.play(all_last.SHORT)
        plain, _events = self.play(all_last.SHORT, rules=False)
        self.assertEqual(got["by_deal"], plain["by_deal"])
        for who, account in (("challenger", got), ("incumbent", got["incumbent_rules"])):
            with self.subTest(who=who):
                self.assertGreater(account["all_last"]["riichi"], 0)
                self.assertGreater(account["late_rule"]["judged"], 0)
                self.assertEqual((account["all_last"]["fired"], account["late_rule"]["fired"]), (0, 0))
        self.assertIsNone(looks.rules_problem(got))
        self.assertEqual(looks.ruled(got), looks.RULES)
        self.assertNotIn("incumbent_rules", plain)
        self.assertNotIn("all_last_check", got)
        self.assertNotIn("late_rule_check", got)

    def test_dama_fires_at_the_incumbents_seats_as_at_the_challengers(self):
        got, events = self.play(all_last.KEEPS_FIRST)
        self.assertGreater(got["all_last"]["fired"], 0)
        self.assertGreater(got["incumbent_rules"]["all_last"]["fired"], 0)
        self.assertEqual(sum(json.loads(line)["type"] == "reach" for table in events for line in table), 0)

    def test_a_pair_with_every_seat_ruled_is_not_checked(self):
        got, _events = self.play(all_last.SHORT, challengers=("fused", "fused"))
        self.assertEqual([sorted(set(arm) & {"all_last", "late_rule"}) for arm in got["arms"]],
                         [["all_last", "late_rule"]] * 2)
        self.assertIn("incumbent_rules", got)
        self.assertNotIn("all_last_check", got)
        self.assertNotIn("late_rule_check", got)
        self.assertEqual(got["difference"]["same_games"], 4)
        self.assertIsNone(looks.rules_problem(got))


class CommandTests(unittest.TestCase):
    """`python -m neural.ruler --last-hand-rules`, as every call on Modal
    runs it, the module run as `__main__`."""

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.folder = tempfile.TemporaryDirectory()
        cls.paths = checkpoints(Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        cls.folder.cleanup()

    def test_the_command_seats_every_player_with_the_rules_and_reports_them(self):
        out = io.StringIO()
        argv = ["ruler", str(self.paths["fused"]), str(self.paths["mortal"]), "--questions", "fixed", "--rows", "4",
                "--preview-reach", "--skip-forced", "--games", "1", "--seed", str(SEED), "--device", "cpu",
                "--last-hand-rules", "--timing"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), warnings.catch_warnings():
            # Said by `runpy` of a module the test has imported already, which is the point.
            warnings.filterwarnings("ignore", "'neural.ruler' found in sys.modules", RuntimeWarning)
            try:
                runpy.run_module("neural.ruler", run_name="__main__")
            except SystemExit as stop:
                self.assertIn(stop.code, (None, 0))
        report = json.loads(out.getvalue())
        self.assertEqual(report["late_rule"]["rule"], "decline-last")
        self.assertIn("riichi", report["all_last"])
        self.assertEqual(sorted(report["incumbent_rules"]), ["all_last", "late_rule"])
        self.assertIs(report["play"]["last_hand_rules"], True)
        self.assertNotIn("all_last_check", report)

    def test_the_command_refuses_it_without_its_switches_or_beside_a_rule_apart(self):
        fixed = ["--questions", "fixed", "--preview-reach"]
        for argv, why in ((["--last-hand-rules"], "--last-hand-rules needs --questions fixed and --preview-reach"),
                          (["--last-hand-rules", "--questions", "fixed"], "--preview-reach"),
                          (["--last-hand-rules", *fixed, "--also", "a.pt", "--late-rule", "decline-last"],
                           "takes neither"),
                          (["--last-hand-rules", *fixed, "--also", "a.pt", "--all-last-dama", "first"],
                           "takes neither")):
            with self.subTest(argv=argv), patch.object(sys, "argv", ["ruler", "a.pt", "b.pt", *argv]), \
                    contextlib.redirect_stderr(io.StringIO()) as said, self.assertRaises(SystemExit) as stop:
                ruler.main()
            self.assertEqual(stop.exception.code, 2)
            self.assertIn(why, said.getvalue())


def account(riichi: int = 3, judged: int = 4) -> dict:
    return {"all_last": {"riichi": riichi, "fired": 1, "tables": [2], "hands": 1,
                         "kept": {"no_yaku": 0, "short": riichi - 1, "dead": 0}, "riichi_after_firing": 0},
            "late_rule": {"rule": "decline-last", "judged": judged, "fired": 1, "tables": [3], "hands": 1,
                          "kept": {"not_last": judged - 1}, "kept_after_firing": 0}}


class LooksTests(unittest.TestCase):
    def report(self, **changes) -> dict:
        report = {"seed": SEED, "by_deal": [2.4, 2.5], **account(), "incumbent_rules": account()}
        report.update(changes)
        return {name: value for name, value in report.items() if value is not None}

    def test_a_report_must_show_every_player_followed_both_rules(self):
        self.assertIsNone(looks.rules_problem(self.report()))
        paired = {"seed": SEED, "arms": [{"by_deal": [2.4], **account()}, {"by_deal": [2.5], **account()}],
                  "incumbent_rules": account()}
        self.assertIsNone(looks.rules_problem(paired))
        plain_dama = {**account(), "late_rule": None}
        for changes, why in (({"incumbent_rules": None}, "the incumbent was to follow the rules of the last hand"),
                             ({"incumbent_rules": {"all_last": account()["all_last"]}}, "the incumbent was to follow "
                                                                                         "decline-last"),
                             ({"all_last": None}, "the first challenger was to follow the all-last dama rule"),
                             ({"late_rule": {**account()["late_rule"], "rule": "blind-dama"}},
                              "the first challenger was to follow decline-last"),
                             ({"incumbent_rules": account(riichi=0)}, "the incumbent's all-last dama rule judged "
                                                                      "nothing"),
                             ({"late_rule": account(judged=0)["late_rule"]}, "the first challenger's decline-last "
                                                                             "rule judged nothing")):
            with self.subTest(changes=changes):
                self.assertIn(why, looks.rules_problem(self.report(**changes)))
        paired["arms"][1] = {"by_deal": [2.5], **{k: v for k, v in plain_dama.items() if v is not None}}
        self.assertIn("the second challenger was to follow decline-last", looks.rules_problem(paired))

    def test_a_series_is_held_to_it_when_registered_so_or_when_its_reports_say_so(self):
        broken = self.report(incumbent_rules=None)
        registered = {"last_hand_rules": True}
        self.assertIsNone(looks.failed_check({SEED: self.report()}, registered))
        self.assertIsNone(looks.failed_check({SEED: self.report()}))
        self.assertIn(f"seed {SEED}'s report fails the rules' check: the incumbent was to follow",
                      looks.failed_check({SEED: broken}, registered))
        # Never registered: the incumbent's account says the seats were all
        # ruled, and the rules' check, not a paired rule's, is asked.
        self.assertIn("fails the rules' check: the first challenger was to follow the all-last dama rule",
                      looks.failed_check({SEED: self.report(all_last=None)}))
        self.assertEqual(looks.ruled(self.report()), " with the rules")
        self.assertEqual(looks.ruled({"all_last": {}}), looks.RULED)


if __name__ == "__main__":
    unittest.main()
