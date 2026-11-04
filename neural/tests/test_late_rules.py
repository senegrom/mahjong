"""The late rule of the last hand (`neural.late_rules`) and the arena's codes
it reads.

The decline rule asks the arena only about the rows of a first question
whose best move is a win, each row a game and a player by place, and edits
only the rows where the engine says it fires: it passes a ron, or strikes a
self-draw from the row and leaves the best other move of the same answer,
where the win leaves the player alone in last. Nothing else is edited,
nobody is told anything the table does not do, a question asked afresh is
left alone, a code that says nothing of a win is refused, and the account
adds up. A rule is seated only asked in fixed rows with the reach's tile
previewed, outermost, one rule of the last hand a player. The arena's codes
agree with its legal moves and the hand. A ruler whose challenger follows a
rule that never fires plays the plain arm's games; one that fires wherever
it is asked changes a game only where it fired, as its check finds; and the
arena's own codes, played through the ruler, judge and fire the rule at a
deal found ahead of time. The command plays a rule only on one checkpoint
against itself, asked alike, never with the all-last dama rule, and seats it
when run as a module."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import riichi_py
import torch

from neural import all_last, duel, late_rules, ruler, suits, zoo
from neural.tests.test_all_last import Answers, Told, legal_row, steer_reaches
from neural.tests.test_bfloat16_players import checkpoints
from neural.tests.test_ruler import SEED, Taught, steer, taught

KINDS = all_last.KINDS


def moves(**named: float) -> np.ndarray:
    """A row of values over Mortal's moves: nought, and the moves named
    (`d<tile>` a discard, `reach`, `agari` a win, `pass`) as given."""
    row = np.zeros(zoo.MORTAL_ACTIONS, dtype=np.float32)
    special = {"reach": zoo.MORTAL_RIICHI, "agari": zoo.MORTAL_AGARI, "pass": zoo.MORTAL_PASS}
    for name, value in named.items():
        row[special[name] if name in special else int(name[1:])] = value
    return row


def claim_row(*more: int) -> np.ndarray:
    """Our engine's moves open in a claim window: the ron, the pass, and
    `more`."""
    row = np.zeros(riichi_py.ACTIONS, dtype=bool)
    row[[zoo.RON, zoo.PASS, *more]] = True
    return row


def turn_row(discards, reaches=(), tsumo: bool = False) -> np.ndarray:
    """Our engine's moves open at a turn: the discards of `discards`, riichi
    with `reaches`, and a self-draw if `tsumo`."""
    row = legal_row(discards, list(reaches))
    row[zoo.TSUMO] = tsumo
    return row


class LateCodes:
    """An arena as the late rule reads it: each row's decline code, by game
    and player, as the engine hands it over, and how many hands each game
    has finished. Every question it is asked is written down, and every row
    it is asked about must be a pair of plain integers, as the arena takes
    them."""

    def __init__(self, decline=None, hands=None) -> None:
        self.decline = decline or {}
        self.hands = dict(hands or {})
        self.asked: list[tuple[str, list]] = []

    def note(self, name: str, rows) -> None:
        for row in rows:
            assert type(row) is tuple and len(row) == 2 and all(type(part) is int for part in row), row
        self.asked.append((name, list(rows)))

    def all_last_decline(self, rows):
        self.note("decline", rows)
        return bytes(self.decline.get(row, 0) for row in rows)

    def hands_done(self):
        games = [game for game, _player in self.decline]
        return [self.hands.get(game, 0) for game in range(1 + max([0, *games, *self.hands]))]


def ask_through(rule: str, account, answers, arena, who, legal, fresh=False, preview=()):
    """`who`'s first question as `zoo.choose_in_mortal_space` asks it with
    every switch, the rows of `preview` asked ahead, through `rule`'s
    wrapper: what the rule answered, and what `answers` answered plain."""
    allowed = zoo.translatable(legal)
    allowed_ahead = np.zeros((len(preview), zoo.MORTAL_ACTIONS), dtype=bool)
    allowed_ahead[:, :KINDS] = legal[list(preview), zoo.RIICHI_DISCARD:zoo.TSUMO]
    question = np.concatenate([allowed, allowed_ahead])
    ahead = [who[i] for i in preview]
    got, _own = late_rules.WRAPPERS[rule](answers, account)(Told(arena), who, fresh, question.copy(), ahead)
    plain, _own = answers(None, who, fresh, question.copy(), ahead)
    return got, plain


def only(tile: int) -> np.ndarray:
    row = np.full(zoo.MORTAL_ACTIONS, -np.inf, dtype=np.float32)
    row[tile] = 0.0
    return row


#: Six decisions for the decline rule. Game 0: a claim window, the win best,
#: which leaves the player last. Game 1: a turn, the win best, which leaves
#: the player last, the 9 next. Game 2: a claim window, the win best, which
#: does not. Game 3: a turn, a discard best. Game 4: a turn, the win best,
#: not the last hand. Game 5: a turn, the win best, which leaves the player
#: last, and riichi next, its preview throwing the 3.
DECLINE_WHO = [(0, 2), (1, 1), (2, 3), (3, 0), (4, 2), (5, 1)]
DECLINE_LEGAL = np.stack([
    claim_row(zoo.PON),
    turn_row([3, 9, 20], tsumo=True),
    claim_row(),
    turn_row([3, 9], tsumo=True),
    turn_row([3, 9], tsumo=True),
    turn_row([3, 9, 20], [3, 9], tsumo=True),
])
DECLINE_ROWS = {
    (0, 2): moves(agari=5.0, **{"pass": 1.0}),
    (1, 1): moves(agari=5.0, d9=4.0, d3=1.0),
    (2, 3): moves(agari=5.0, **{"pass": 4.0}),
    (3, 0): moves(agari=2.0, d3=3.0),
    (4, 2): moves(agari=5.0, d3=1.0),
    (5, 1): moves(agari=5.0, reach=4.0, d9=3.0),
}
DECLINE_PREVIEWS = {(5, 1): moves(d3=2.0, d9=1.0)}
DECLINE_CODES = {(0, 2): late_rules.LAST, (1, 1): late_rules.LAST, (2, 3): late_rules.NOT_LAST,
                 (5, 1): late_rules.LAST}


class DeclineTests(unittest.TestCase):
    def test_it_declines_only_a_win_that_leaves_the_player_last(self):
        account = late_rules.Account("decline-last")
        arena = LateCodes(decline=DECLINE_CODES, hands={0: 7, 1: 7, 5: 3})
        got, plain = ask_through("decline-last", account, Answers(DECLINE_ROWS, DECLINE_PREVIEWS), arena,
                                 DECLINE_WHO, DECLINE_LEGAL, preview=[5])
        self.assertEqual(arena.asked, [("decline", [(0, 2), (1, 1), (2, 3), (4, 2), (5, 1)])])
        # A ron declined is the pass, and nothing else stays open.
        np.testing.assert_array_equal(got[0], only(zoo.MORTAL_PASS))
        # A self-draw declined is struck from its row, and the rest stand.
        for row in (1, 5):
            want = plain[row].copy()
            want[zoo.MORTAL_AGARI] = -np.inf
            np.testing.assert_array_equal(got[row], want)
        for row in (2, 3, 4, 6):
            np.testing.assert_array_equal(got[row], plain[row])
        self.assertEqual(account.report(), {"rule": "decline-last", "judged": 4, "fired": 3, "tables": [0, 1, 5],
                                            "hands": 3, "kept": {"not_last": 1}, "kept_after_firing": 0})

    def test_the_pass_or_the_best_other_move_is_played_and_nobody_is_told_a_reach(self):
        """Through the very choice every player of Mortal's moves makes,
        with every switch: a declined ron passes, a declined self-draw
        plays the best other move of its answer, a reach among them, its
        tile the preview's, and nobody is told anything."""
        for codes, want in (
            (DECLINE_CODES, [zoo.PASS, zoo.DISCARD + 9, zoo.RON, zoo.DISCARD + 3, zoo.TSUMO,
                             zoo.RIICHI_DISCARD + 3]),
            ({}, [zoo.RON, zoo.TSUMO, zoo.RON, zoo.DISCARD + 3, zoo.TSUMO, zoo.TSUMO]),
        ):
            account = late_rules.Account("decline-last")
            views = Told(LateCodes(decline=codes))
            ask = late_rules.decline_last(Answers(DECLINE_ROWS, DECLINE_PREVIEWS), account)
            rows = np.array([game for game, _ in DECLINE_WHO])
            players = np.array([player for _, player in DECLINE_WHO])
            with self.subTest(codes=bool(codes)):
                choice = zoo.choose_in_mortal_space(
                    lambda who, fresh, allowed, ahead=(): ask(views, who, fresh, allowed, ahead),
                    views, rows, players, DECLINE_LEGAL, skip_forced=True, preview_reach=True)
                self.assertEqual(choice.tolist(), want)
                self.assertEqual(views.told, [])
                self.assertEqual(account.fired, 3 if codes else 0)

    def test_the_account_counts_a_win_taken_after_one_declined(self):
        account = late_rules.Account("decline-last")
        arena = LateCodes(decline={(0, 1): late_rules.LAST}, hands={0: 5})
        ask = late_rules.decline_last(Answers({(0, 1): moves(agari=5.0, **{"pass": 1.0})}, {}), account)
        allowed = zoo.translatable(claim_row()[None])
        for code, hand in ((late_rules.LAST, 5), (late_rules.NOT_LAST, 5), (late_rules.NOT_LAST, 6)):
            arena.decline[(0, 1)] = code
            arena.hands[0] = hand
            ask(Told(arena), [(0, 1)], False, allowed.copy())
        self.assertEqual(account.report(), {"rule": "decline-last", "judged": 3, "fired": 1, "tables": [0],
                                            "hands": 1, "kept": {"not_last": 2}, "kept_after_firing": 1})

    def test_a_question_asked_afresh_or_without_a_win_best_is_left_alone(self):
        account = late_rules.Account("decline-last")
        arena = LateCodes(decline=DECLINE_CODES)
        got, plain = ask_through("decline-last", account, Answers(DECLINE_ROWS, DECLINE_PREVIEWS), arena,
                                 DECLINE_WHO, DECLINE_LEGAL, fresh=True, preview=[5])
        np.testing.assert_array_equal(got, plain)
        got, plain = ask_through("decline-last", account, Answers(DECLINE_ROWS, {}), arena, [DECLINE_WHO[3]],
                                 DECLINE_LEGAL[[3]])
        np.testing.assert_array_equal(got, plain)
        self.assertEqual((arena.asked, account.judged), ([], 0))

    def test_a_code_that_says_nothing_of_a_win_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "says nothing of a win"):
            ask_through("decline-last", late_rules.Account("decline-last"), Answers(DECLINE_ROWS, {}),
                        LateCodes(decline={(0, 2): 7}), [DECLINE_WHO[0]], DECLINE_LEGAL[[0]])
        with self.assertRaisesRegex(RuntimeError, "outermost"):
            late_rules.decline_last(Answers(DECLINE_ROWS, {}), late_rules.Account("decline-last"))(
                SimpleNamespace(), [DECLINE_WHO[0]], False, zoo.translatable(DECLINE_LEGAL[[0]]))


class AccountTests(unittest.TestCase):
    def test_each_rule_keeps_its_own_reasons_and_no_other_rule_is_known(self):
        for rule, kept in late_rules.KEPT.items():
            with self.subTest(rule=rule):
                self.assertEqual(late_rules.Account(rule).report()["kept"], dict.fromkeys(kept, 0))
        with self.assertRaisesRegex(ValueError, "no late rule"):
            late_rules.Account("all-last-dama")
        self.assertEqual(late_rules.RULES, tuple(late_rules.WRAPPERS))
        self.assertEqual(late_rules.RULES, tuple(late_rules.KEPT))


class SeatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.paths = checkpoints(Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_only_a_player_asked_in_fixed_rows_with_the_reach_previewed_follows_one(self):
        fixed = {"questions": "fixed", "rows": 4, "preview_reach": True}
        wrapped = {"decline-last": "decline_a_win_that_leaves_it_last"}
        for rule in late_rules.RULES:
            for name in ("fused", "reheaded", "mortal"):
                with self.subTest(rule=rule, player=name):
                    player = ruler.seat(self.paths[name], "cpu", **fixed, late_rule=rule)
                    self.assertEqual(late_rules.account_of(player), late_rules.Account(rule))
                    self.assertEqual(player._ask.__name__, wrapped[rule])
                    self.assertIsNone(all_last.account_of(player))
                    with self.assertRaisesRegex(ValueError, "already follows a rule of the last hand"):
                        late_rules.follow(player, rule, fixed=True)
        # Outside an average over the suits, whose renamed game names no arena.
        both = ruler.seat(self.paths["fused"], "cpu", **fixed, average_suits=True, late_rule="decline-last")
        self.assertIsNotNone(suits.averaging(both))
        self.assertEqual(both._ask.__name__, "decline_a_win_that_leaves_it_last")
        for options, why in (({}, "fixed questions"), ({"questions": "fixed"}, "fixed questions"),
                             ({"questions": "fixed", "skip_forced": True}, "previewed"),
                             ({**fixed, "all_last_dama": True}, "one rule of the last hand")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, why):
                ruler.seat(self.paths["fused"], "cpu", **options, late_rule="decline-last")
        # Asked of the rule itself, without the seating's own refusals.
        dama = ruler.seat(self.paths["fused"], "cpu", **fixed, all_last_dama=True)
        with self.assertRaisesRegex(ValueError, "already follows a rule of the last hand"):
            late_rules.follow(dama, "decline-last", fixed=True)
        with self.assertRaisesRegex(ValueError, "no late rule"):
            late_rules.follow(ruler.seat(self.paths["fused"], "cpu", **fixed), "dama", fixed=True)
        with self.assertRaisesRegex(ValueError, "questions fixed"):
            late_rules.follow(ruler.seat(self.paths["fused"], "cpu", **fixed), "decline-last")
        with self.assertRaisesRegex(ValueError, "preview-reach"):
            late_rules.follow(ruler.seat(self.paths["fused"], "cpu", questions="fixed", rows=4), "decline-last",
                              fixed=True)
        with self.assertRaisesRegex(ValueError, "Mortal's moves"):
            late_rules.follow(ruler.seat(self.paths["ours, our moves"], "cpu", questions="fixed", rows=4),
                              "decline-last", fixed=True)
        # An engine before the decline codes, or before a riichi win's
        # decline code was read without its hidden ura dora.
        for engine in (SimpleNamespace(), SimpleNamespace(ALL_LAST_API_VERSION=1),
                       SimpleNamespace(ALL_LAST_API_VERSION=2)):
            with self.subTest(engine=engine), patch.object(late_rules, "riichi_py", engine), \
                    self.assertRaisesRegex(RuntimeError, "Rebuild"):
                late_rules.follow(ruler.seat(self.paths["fused"], "cpu", **fixed), "decline-last", fixed=True)


class BindingTests(unittest.TestCase):
    """`riichi_py.Arena.all_last_decline`, the engine's own codes."""

    def test_the_engine_answers_the_codes(self):
        self.assertGreaterEqual(riichi_py.ALL_LAST_API_VERSION, 3)
        arena = riichi_py.Arena(games=3, seed=SEED)
        rows = [(2, 0), (0, 0), (0, 1)]
        # The first hand is East 1's, so nothing is asked.
        codes = arena.all_last_decline(rows)
        self.assertIsInstance(codes, bytes)
        self.assertEqual(codes, bytes(len(rows)))
        self.assertEqual(arena.all_last_decline([]), b"")
        with self.assertRaisesRegex(ValueError, "no game 3"):
            arena.all_last_decline([(0, 0), (3, 0)])
        with self.assertRaisesRegex(ValueError, "no player 4"):
            arena.all_last_decline([(0, 4)])

    def test_the_codes_agree_with_the_legal_moves_and_the_hand(self):
        """Games the heuristic player plays at every seat, asked at every
        step about the player who owes each game's decision: a decline code
        exactly where a non-dealer of South 4 may win, the hand read off the
        arena's own events. The engine's Club test plays each code through
        to the table (see `club_games_agree_with_the_real_flow` in
        all_last.rs)."""
        games = 48
        arena = riichi_py.Arena(games=games, seed=SEED + 103)
        last = [None] * games
        asked = 0
        while not arena.all_finished():
            for game, lines in enumerate(arena.mjai_all()):
                for line in lines:
                    if '"start_kyoku"' in line:
                        last[game] = json.loads(line)
            seats = np.frombuffer(arena.seats(), dtype=np.uint8)
            legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, riichi_py.ACTIONS).astype(bool)
            players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
            live = np.flatnonzero(seats != 0xFF)
            rows = [(int(game), int(players[game, seats[game]])) for game in live]
            decline = np.frombuffer(arena.all_last_decline(rows), dtype=np.uint8)
            for at, (game, player) in enumerate(rows):
                hand = last[game]
                south_four = hand["bakaze"] == "S" and hand["kyoku"] == 4
                dealer = player == hand["oya"]
                win = legal[game, zoo.TSUMO] or legal[game, zoo.RON]
                self.assertEqual(bool(decline[at]), bool(south_four and not dealer and win), (game, hand, decline[at]))
                asked += int(decline[at] != 0)
            taught = np.frombuffer(arena.teacher(), dtype=np.uint8).astype(np.int64)
            taught[taught == 0xFF] = 0
            arena.step(taught.tolist())
        self.assertGreater(asked, 5)


class LateCoded:
    """An arena whose decline codes are stand-ins, at any hand, wherever its
    legal moves allow a win: the win leaves the player last, or does not. A
    stand-in for the engine's, which fires everywhere or nowhere, as the
    engine's own codes seldom do in a ruler of one deal a seating (see
    `RulerTests.test_the_engines_own_codes_judge_and_fire_each_rule` for
    those). Everything else is the arena's own; `riichi_py.Arena` cannot be
    subclassed, so this hands it on. Each table's events are kept as they
    are handed out."""

    def __init__(self, arena, fires: bool) -> None:
        self.arena = arena
        self.fires = fires
        self.events: list[list[str]] = [[] for _ in range(arena.games)]

    def legal(self, rows) -> np.ndarray:
        legal = np.frombuffer(self.arena.legal_mask(), dtype=np.uint8).reshape(-1, riichi_py.ACTIONS).astype(bool)
        return legal[[game for game, _player in rows]]

    def all_last_decline(self, rows):
        legal = self.legal(rows)
        win = legal[:, zoo.TSUMO] | legal[:, zoo.RON]
        return np.where(win, late_rules.LAST if self.fires else late_rules.NOT_LAST, 0).astype(np.uint8).tobytes()

    def mjai_all(self):
        lines = self.arena.mjai_all()
        for kept, more in zip(self.events, lines):
            kept.extend(more)
        return lines

    def __getattr__(self, name):
        return getattr(self.arena, name)


@contextlib.contextmanager
def late_coded(fires: bool):
    """Every ruler's views served from a `LateCoded` arena, asking the
    heuristic player once a step (see `test_ruler.Taught`); yields the views
    made."""
    made = []

    class CodedViews(Taught):
        def __init__(self, arena, *args, **kwargs):
            super().__init__(LateCoded(arena, fires), *args, **kwargs)
            made.append(self)

    with patch.object(duel, "Views", CodedViews):
        yield made


def steer_every_reach(player):
    """`player` pushed, in each reach's preview, towards one tile: the one the
    heuristic player throws with a reach of its own, or else the one it
    throws undeclared if a reach may throw it, or else the reach's
    lowest-numbered tile. With `test_ruler.steer`, which settles every first
    question, no move before a rule fires then turns on a near tie of a small
    random network, whose last bits can differ from one processor to another
    (`test_all_last.steer_reaches` leaves a reach the heuristic player would
    not make to the network's own preview)."""
    asked = player._ask

    def steered(views, who, fresh, allowed, ahead=()):
        values, own = asked(views, who, fresh, allowed, ahead)
        for at, (game, _player) in enumerate(ahead):
            row = len(who) + at
            taught = int(views.taught[game])
            if zoo.RIICHI_DISCARD <= taught < zoo.TSUMO:
                tile = taught - zoo.RIICHI_DISCARD
            elif zoo.DISCARD <= taught < zoo.DISCARD + KINDS and allowed[row, taught - zoo.DISCARD]:
                tile = taught - zoo.DISCARD
            else:
                tile = int(np.flatnonzero(allowed[row, :KINDS])[0])
            values[row, tile] += 100.0
        return values, own

    player._ask = steered
    return player


def first_difference(events: list[list[str]], table: int) -> tuple[dict, dict]:
    """Where the ruled arm's table `table` and the plain arm's table of the
    same deal and seat first part, as the events each was handed: the ruled
    arm's event there, and the plain arm's."""
    ruled, plain = ([json.loads(line) for line in events[at]] for at in (table, table + 4))
    part = next(at for at, (mine, theirs) in enumerate(zip(ruled, plain)) if mine != theirs)
    return ruled[part], plain[part]


class RulerTests(unittest.TestCase):
    """Whole paired rulers of small networks on the processor, one deal a
    seating, in fixed rows of four with every switch, every player pushed
    towards the heuristic player's moves and towards a reach wherever one
    is open (see `test_ruler.steer`), and the first challenger following
    a late rule."""

    OPTIONS = {"questions": "fixed", "rows": 4, "skip_forced": True, "preview_reach": True}

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

    def seated(self, name: str):
        return steer(steer_reaches(ruler.seat(self.paths[name], "cpu", **self.OPTIONS)))

    def paired(self, rule: str, fires: bool):
        with late_coded(fires) as made:
            first = late_rules.follow(self.seated("fused"), rule, fixed=True)
            got = ruler.ruler([first, self.seated("fused")], self.seated("mortal"), 1, SEED, "cpu",
                              questions="fixed")
        (views,) = made
        # The events of the last step, which no later step handed out.
        views.arena.mjai_all()
        return got, views.arena.events

    def test_a_rule_that_never_fires_plays_the_plain_arms_games(self):
        for rule in late_rules.RULES:
            with self.subTest(rule=rule):
                got, _events = self.paired(rule, fires=False)
                ruled, plain = got["arms"]
                account = ruled["late_rule"]
                self.assertEqual(account["rule"], rule)
                self.assertGreater(account["judged"], 0)
                self.assertEqual((account["fired"], account["tables"], account["hands"]), (0, [], 0))
                self.assertEqual(sum(account["kept"].values()), account["judged"])
                self.assertNotIn("late_rule", plain)
                self.assertNotIn("all_last", ruled)
                self.assertEqual(ruled["by_deal"], plain["by_deal"])
                self.assertEqual(got["difference"]["same_games"], 4)
                self.assertEqual(got["late_rule_check"], {"differing": 0, "violations": []})
                self.assertNotIn("all_last_check", got)

    def test_a_rule_that_fires_wherever_it_is_asked_changes_games_only_where_it_fired(self):
        """And where it fired first, it played its move where the plain arm
        played the network's: a pass or another move where the plain arm
        won."""
        for rule in late_rules.RULES:
            with self.subTest(rule=rule):
                got, events = self.paired(rule, fires=True)
                account = got["arms"][0]["late_rule"]
                self.assertGreater(account["fired"], 0)
                self.assertLessEqual(set(account["tables"]), {0, 1, 2, 3})
                check = got["late_rule_check"]
                self.assertGreater(check["differing"], 0)
                self.assertEqual(check["violations"], [])
                for table in account["tables"]:
                    if events[table] == events[table + 4]:
                        continue
                    ruled, plain = first_difference(events, table)
                    self.assertEqual((plain["type"], plain["actor"]), ("hora", table))
                    self.assertFalse(ruled["type"] == "hora" and ruled["actor"] == table, ruled)

    #: For each rule, a seed and a number of deals at which the arena's own
    #: codes, played through the ruler, judge the rule and fire it: found
    #: once ahead of time, as the engine's Club test finds its games, by
    #: playing the deals from `SEED` on with the first challenger alone.
    #: The decline fired at one of the first 16 deals, `SEED` + 5. About
    #: fifteen seconds a deal.
    REAL = {"decline-last": (SEED + 5, 1)}

    def surely(self, name: str):
        """A player seated as `seated` seats one, every move it makes before
        a rule fires settled by the heuristic player (see
        `steer_every_reach`)."""
        return steer(steer_every_reach(ruler.seat(self.paths[name], "cpu", **self.OPTIONS)))

    def test_the_engines_own_codes_judge_and_fire_each_rule(self):
        """The arena's codes, not a stand-in's, through the ruler's play and
        each rule's question: at the deals found for it (`REAL`), the rule
        judged and fired, the arms ended some table differently, and every
        table they ended differently at is one it fired at."""
        for rule, (seed, deals) in self.REAL.items():
            with self.subTest(rule=rule), taught():
                first = late_rules.follow(self.surely("fused"), rule, fixed=True)
                got = ruler.ruler([first, self.surely("fused")], self.surely("mortal"), deals, seed, "cpu",
                                  questions="fixed")
                account = got["arms"][0]["late_rule"]
                self.assertEqual(account["rule"], rule)
                self.assertGreater(account["judged"], 0, account)
                self.assertGreater(account["fired"], 0, account)
                self.assertEqual(sum(account["kept"].values()), account["judged"] - account["fired"])
                # Else the check below would hold of arms that never parted.
                self.assertGreater(got["late_rule_check"]["differing"], 0, got["late_rule_check"])
                self.assertEqual(got["late_rule_check"]["violations"], [])


class CheckTests(unittest.TestCase):
    def test_every_table_that_differs_must_be_one_the_rule_fired_at(self):
        layout = ruler.Layout(2, 3)
        scores = np.arange(layout.games * 4, dtype=np.int32).reshape(layout.games, 4) % 7
        scores[12:] = scores[:12]
        scores[19, 0] += 1000
        scores[21, 3] -= 1000
        fired = late_rules.Account("decline-last", tables={7, 2})
        self.assertEqual(ruler.dama_check(scores, layout, [fired, None]),
                         {"differing": 2, "violations": [{"deal": 0, "place": 3, "tables": [9, 21]}]})

    def run_main(self, check: dict | None, *extra: str, also: str = "a.pt", rule: str = "decline-last"):
        """`neural.ruler`'s command with a late rule, its players and its
        play stood in for: what it wrote, what it said, and how it ended."""
        arm = {"placement": 2.5, "standard_error": 0.1, "by_deal": [2.5]}
        result = {"seed": 5, "games_per_seat": 1, "arms": [dict(arm), dict(arm)]}
        if check is not None:
            result["late_rule_check"] = check
        out, said = io.StringIO(), io.StringIO()
        seated = []

        def seat(path, device, **options):
            seated.append(options)

        with patch.object(ruler, "seat", side_effect=seat), patch.object(ruler, "ruler", return_value=result), \
                patch.object(sys, "argv", ["ruler", "a.pt", "b.pt", "--also", also, "--questions", "fixed",
                                           "--preview-reach", "--late-rule", rule, *extra]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(said):
            try:
                ruler.main()
            except SystemExit as stop:
                return out.getvalue(), said.getvalue(), stop.code, seated
        return out.getvalue(), said.getvalue(), None, seated

    def test_the_first_challenger_follows_the_rule_and_a_failed_check_is_kept(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.chdir(folder):
            Path("a.pt").write_bytes(b"one network")
            for rule in late_rules.RULES:
                with self.subTest(rule=rule):
                    out, said, code, seated = self.run_main({"differing": 3, "violations": []}, rule=rule)
                    self.assertEqual((said, code), ("", None))
                    self.assertEqual([options["late_rule"] for options in seated[:2]], [rule, None])
                    self.assertNotIn("late_rule", seated[2])
                    self.assertEqual(json.loads(out)["late_rule_check"]["violations"], [])
            violation = {"deal": 0, "place": 1, "tables": [1, 5]}
            out, said, code, _seated = self.run_main({"differing": 3, "violations": [violation]}, rule="decline-last")
            self.assertIsNone(code)
            self.assertEqual(json.loads(out)["late_rule_check"]["violations"], [violation])
            self.assertIn("the decline-last rule's check failed: at 1 of the 3 deals and seats", said)

    def test_the_rule_compares_one_checkpoint_with_itself_asked_alike_and_alone(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.chdir(folder):
            for name, data in (("a.pt", b"one network"), ("copy.pt", b"one network"), ("other.pt", b"two network")):
                Path(name).write_bytes(data)
            for also, extra in (("a.pt", []), ("copy.pt", []), ("a.pt", ["--average-suits", "both"])):
                with self.subTest(also=also, extra=extra):
                    _out, _said, code, _seated = self.run_main({"differing": 0, "violations": []}, *extra, also=also)
                    self.assertIsNone(code)
            for also, extra, why in (("other.pt", [], "--also must name the challenger's own checkpoint"),
                                     ("a.pt", ["--average-suits", "first"], "--average-suits first"),
                                     ("a.pt", ["--all-last-dama", "first"], "two rules of the last hand")):
                with self.subTest(also=also, extra=extra):
                    _out, said, code, _seated = self.run_main(None, *extra, also=also)
                    self.assertEqual(code, 2)
                    self.assertIn(why, said)
        for argv, why in ((["--late-rule", "decline-last"], "--questions fixed and --preview-reach"),
                          (["--late-rule", "decline-last", "--questions", "fixed"], "--preview-reach"),
                          (["--late-rule", "dama"], "invalid choice"),
                          (["--late-rule", "decline-last", "--questions", "fixed", "--preview-reach"], "--also")):
            with self.subTest(argv=argv), patch.object(sys, "argv", ["ruler", "a.pt", "b.pt", *argv]), \
                    contextlib.redirect_stderr(io.StringIO()) as said, self.assertRaises(SystemExit):
                ruler.main()
            self.assertIn(why, said.getvalue())


class CommandTests(unittest.TestCase):
    """`python -m neural.ruler` with each late rule, as the desktop's pilot
    and every call on Modal run it, the module run as `__main__`."""

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

    def test_the_command_seats_each_rule_and_reports_it(self):
        """On tiny networks, one deal a seating, the second challenger a
        copy of the first's checkpoint, as `modal_app.ruler` makes one."""
        fused, mortal = self.paths["fused"], self.paths["mortal"]
        copy = fused.with_name("2-fused.pt")
        shutil.copyfile(fused, copy)
        for rule in late_rules.RULES:
            out = io.StringIO()
            argv = ["ruler", str(fused), str(mortal), "--also", str(copy), "--questions", "fixed", "--rows", "4",
                    "--preview-reach", "--skip-forced", "--games", "1", "--seed", str(SEED), "--device", "cpu",
                    "--late-rule", rule, "--timing"]
            with self.subTest(rule=rule), patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), \
                    warnings.catch_warnings():
                # Said by `runpy` of a module the test has imported already, which is the point.
                warnings.filterwarnings("ignore", "'neural.ruler' found in sys.modules", RuntimeWarning)
                try:
                    runpy.run_module("neural.ruler", run_name="__main__")
                except SystemExit as stop:
                    self.assertIn(stop.code, (None, 0))
                report = json.loads(out.getvalue())
                ruled, plain = report["arms"]
                self.assertEqual(set(ruled["late_rule"]), {"rule", "judged", "fired", "tables", "hands", "kept",
                                                           "kept_after_firing"})
                self.assertEqual(ruled["late_rule"]["rule"], rule)
                self.assertNotIn("late_rule", plain)
                self.assertEqual(set(report["late_rule_check"]), {"differing", "violations"})
                self.assertEqual(report["late_rule_check"]["violations"], [])
                self.assertEqual(report["play"]["late_rule"], rule)


if __name__ == "__main__":
    unittest.main()
