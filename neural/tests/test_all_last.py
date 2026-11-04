"""The all-last dama rule (`neural.all_last`) and the arena's codes it reads.

The rule fires only where a row's best move is riichi in the last hand and
the engine says the reach's own tile keeps the player first: the tile the
reach's preview throws, or the one tile a reach unpreviewed can throw, and
not the best move once riichi is put aside. It then makes the discard of
that tile the row's only finite answer, so the tile is played undeclared
and nobody is told a reach; it leaves every other row, the previews and a
question asked afresh as they came, asks the arena about nothing else, and
its account adds up. It is seated only asked in fixed rows, as its seating
says, with the reach's tile previewed, outermost, and never on a player that
follows a late rule of the last hand. The arena's codes agree
with its legal moves and with the hand being the last, a non-dealer's. A
ruler whose challenger follows the rule and never fires plays the plain
arm's games; one that fires wherever riichi is best declares no riichi,
throws undeclared the very tile the plain arm's reach throws, and every
table its games differ at is one it fired at, which the ruler checks and
keeps in its report. The command plays the rule only on one checkpoint
against itself, asked alike, and seats it when run as a module."""

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

from neural import all_last, duel, ruler, suits, zoo
from neural.tests.test_bfloat16_players import checkpoints
from neural.tests.test_ruler import SEED, Taught, steer

KINDS = all_last.KINDS
REACH = zoo.MORTAL_RIICHI


class Codes:
    """An arena as the rule reads it: each game's codes, a code a tile,
    as the engine hands them over, and how many hands each game has
    finished. Every question it is asked is written down."""

    def __init__(self, codes: dict[int, dict[int, int]], hands: dict[int, int] | None = None) -> None:
        self.codes = codes
        self.hands = dict(hands or {})
        self.asked: list[list[int]] = []

    def all_last_dama(self, games):
        self.asked.append(list(games))
        rows = np.zeros((len(games), KINDS), dtype=np.uint8)
        for row, game in enumerate(games):
            for tile, code in self.codes.get(game, {}).items():
                rows[row, tile] = code
        return rows.tobytes()

    def hands_done(self):
        return [self.hands.get(game, 0) for game in range(1 + max([0, *self.codes, *self.hands]))]


class Told:
    """Views whose follower only writes down what it is told, of `arena`."""

    def __init__(self, arena) -> None:
        self.arena = arena
        self.told: list[tuple] = []

    def tell(self, game, player, line):
        self.told.append((game, player, json.loads(line)["type"]))


def values(**moves: float) -> np.ndarray:
    """A row of values over Mortal's moves: nought, and the moves named
    (`d<tile>` a discard, `reach` riichi) as given."""
    row = np.zeros(zoo.MORTAL_ACTIONS, dtype=np.float32)
    for name, value in moves.items():
        row[REACH if name == "reach" else int(name[1:])] = value
    return row


class Answers:
    """A player's question answered with the values set for each row: the
    first question's by (game, player), the reaches' previews by the same
    in `previews`; a move not allowed at minus infinity, as a network
    answers. Every question is written down."""

    def __init__(self, rows: dict, previews: dict) -> None:
        self.rows = rows
        self.previews = previews
        self.calls: list[tuple] = []

    def __call__(self, views, who, fresh, allowed, ahead=()):
        self.calls.append((list(who), fresh, list(ahead)))
        found = [self.rows[tuple(entry)] for entry in who] + [self.previews[tuple(entry)] for entry in ahead]
        answer = np.stack(found) if found else np.zeros((0, zoo.MORTAL_ACTIONS), dtype=np.float32)
        return np.where(allowed, answer, -np.inf).astype(np.float32), np.zeros(answer.shape, dtype=bool)


def legal_row(discards, reaches, legal=None) -> np.ndarray:
    """Our engine's moves open at a turn: the discards of `discards` and
    riichi with `reaches`."""
    row = np.zeros(riichi_py.ACTIONS, dtype=bool) if legal is None else legal
    row[zoo.DISCARD + np.asarray(discards, dtype=int)] = True
    row[zoo.RIICHI_DISCARD + np.asarray(reaches, dtype=int)] = True
    return row


#: Five decisions at once. Game 0: riichi is best, its preview throws the
#: 5 (2 above the 9), and the 5 keeps first, though the best discard is the
#: 9. Game 1: riichi is best and its preview throws the 9, which is short,
#: though the 5 would keep first. Game 2: a discard is best. Game 3: riichi
#: is best but it is not the last hand. Game 4: riichi is best with the 20
#: alone, so its reach is left unasked with `skip_forced`, and the 20 keeps
#: first.
WHO = [(0, 1), (1, 2), (2, 0), (3, 3), (4, 1)]
LEGAL = np.stack([
    legal_row([3, 5, 9, 20], [5, 9]),
    legal_row([3, 5, 9, 20], [5, 9]),
    legal_row([3, 5, 9, 20], [3]),
    legal_row([3, 5, 9, 20], [3, 5]),
    legal_row([3, 20], [20]),
])
ROWS = {
    (0, 1): values(reach=5.0, d9=4.0, d5=1.0),
    (1, 2): values(reach=5.0, d9=4.0, d5=1.0),
    (2, 0): values(reach=2.0, d3=3.0),
    (3, 3): values(reach=5.0, d3=1.0),
    (4, 1): values(reach=5.0, d3=4.0),
}
PREVIEWS = {(0, 1): values(d5=3.0, d9=1.0), (1, 2): values(d5=1.0, d9=3.0), (3, 3): values(d3=1.0, d5=2.0)}
CODES = {
    0: {3: all_last.NOT_WAITING, 5: all_last.KEEPS_FIRST, 9: all_last.SHORT, 20: all_last.NOT_WAITING},
    1: {3: all_last.NOT_WAITING, 5: all_last.KEEPS_FIRST, 9: all_last.SHORT, 20: all_last.NOT_WAITING},
    2: {3: all_last.KEEPS_FIRST, 5: all_last.NOT_WAITING},
    4: {3: all_last.NOT_WAITING, 20: all_last.KEEPS_FIRST},
}


class RuleTests(unittest.TestCase):
    def asked(self, account: all_last.Account, arena: Codes, fresh: bool = False):
        """The five decisions' question, as `zoo.choose_in_mortal_space`
        asks it with every switch: the reaches with more than one tile
        previewed after the first question's rows."""
        allowed = zoo.translatable(LEGAL)
        ahead = [0, 1, 3]
        allowed_ahead = np.zeros((len(ahead), zoo.MORTAL_ACTIONS), dtype=bool)
        allowed_ahead[:, :KINDS] = LEGAL[ahead, zoo.RIICHI_DISCARD:zoo.TSUMO]
        answers = Answers(ROWS, PREVIEWS)
        question = np.concatenate([allowed, allowed_ahead])
        got, _own = all_last.ruled(answers, account)(Told(arena), WHO, fresh, question, [WHO[i] for i in ahead])
        plain, _own = answers(None, WHO, fresh, question, [WHO[i] for i in ahead])
        return got, plain

    def test_it_fires_only_where_the_reach_tile_keeps_first(self):
        account = all_last.Account()
        arena = Codes(CODES, hands={0: 7, 4: 7})
        got, plain = self.asked(account, arena)
        # Asked once, about the rows whose best move is riichi.
        self.assertEqual(arena.asked, [[0, 1, 3, 4]])
        # Game 0 throws the 5, the reach's tile, not the 9 the network
        # rates above it; game 4 the 20; nothing else is left open there.
        for row, tile in ((0, 5), (4, 20)):
            want = np.full(zoo.MORTAL_ACTIONS, -np.inf, dtype=np.float32)
            want[tile] = 0.0
            np.testing.assert_array_equal(got[row], want)
        # The rest as they came: the short reach, the discard, the reach
        # that is not the last hand's, and the previews.
        for row in (1, 2, 3, 5, 6, 7):
            np.testing.assert_array_equal(got[row], plain[row])
        self.assertEqual(account.report(), {"riichi": 3, "fired": 2, "tables": [0, 4], "hands": 2,
                                            "kept": {"no_yaku": 0, "short": 1, "dead": 0},
                                            "riichi_after_firing": 0})
        self.assertEqual(account.hands, {(0, 7), (4, 7)})

    def test_the_discard_is_played_and_nobody_is_told_a_reach(self):
        """Through the very choice every player of Mortal's moves makes,
        with every switch: where the rule fires, the reach's tile without
        the declaration; elsewhere the plain choice, a reach included."""
        for codes, want in (
            (CODES, [zoo.DISCARD + 5, zoo.RIICHI_DISCARD + 9, zoo.DISCARD + 3, zoo.RIICHI_DISCARD + 5,
                     zoo.DISCARD + 20]),
            ({}, [zoo.RIICHI_DISCARD + 5, zoo.RIICHI_DISCARD + 9, zoo.DISCARD + 3, zoo.RIICHI_DISCARD + 5,
                  zoo.RIICHI_DISCARD + 20]),
        ):
            account = all_last.Account()
            views = Told(Codes(codes))
            ask = all_last.ruled(Answers(ROWS, PREVIEWS), account)
            rows, players = np.array([game for game, _ in WHO]), np.array([player for _, player in WHO])
            with self.subTest(codes=bool(codes)):
                choice = zoo.choose_in_mortal_space(
                    lambda who, fresh, allowed, ahead=(): ask(views, who, fresh, allowed, ahead),
                    views, rows, players, LEGAL, skip_forced=True, preview_reach=True)
                self.assertEqual(choice.tolist(), want)
                self.assertEqual(views.told, [])
                self.assertEqual(account.fired, 2 if codes else 0)

    def test_a_question_asked_afresh_is_left_alone(self):
        account = all_last.Account()
        arena = Codes(CODES)
        got, plain = self.asked(account, arena, fresh=True)
        np.testing.assert_array_equal(got, plain)
        self.assertEqual((arena.asked, account.riichi), ([], 0))

    def test_without_a_reach_best_the_arena_is_not_asked(self):
        account = all_last.Account()
        arena = Codes(CODES)
        ask = all_last.ruled(Answers({(2, 0): ROWS[(2, 0)]}, {}), account)
        got, _own = ask(Told(arena), [(2, 0)], False, zoo.translatable(LEGAL[[2]]))
        self.assertEqual(got[0].argmax(), 3)
        self.assertEqual((arena.asked, account.riichi), ([], 0))

    def test_the_account_adds_up_over_hands(self):
        """A reach declared later in a hand the rule fired in is counted;
        one in the hand after it is not. Every reach the rule judged is
        fired or kept, for its reason."""
        account = all_last.Account()
        arena = Codes({0: {20: all_last.KEEPS_FIRST}}, hands={0: 5})
        ask = all_last.ruled(Answers({(0, 1): values(reach=5.0)}, {}), account)
        allowed = zoo.translatable(legal_row([3, 20], [20])[None])
        for code, hand in ((all_last.KEEPS_FIRST, 5), (all_last.SHORT, 5), (all_last.NO_YAKU, 6),
                           (all_last.DEAD, 6), (all_last.KEEPS_FIRST, 6), (all_last.DEAD, 6)):
            arena.codes[0][20] = code
            arena.hands[0] = hand
            ask(Told(arena), [(0, 1)], False, allowed.copy())
        report = account.report()
        self.assertEqual(report, {"riichi": 6, "fired": 2, "tables": [0], "hands": 2,
                                  "kept": {"no_yaku": 1, "short": 1, "dead": 2}, "riichi_after_firing": 2})
        self.assertEqual(report["riichi"], report["fired"] + sum(report["kept"].values()))

    def test_codes_that_contradict_the_legal_moves_are_refused(self):
        allowed = zoo.translatable(LEGAL)
        for codes, why in (
            # The reach's tile with no riichi code at all, or a riichi that is not legal.
            ({0: {5: all_last.NOT_WAITING}}, "which no legal riichi has"),
            ({0: {9: all_last.SHORT}}, "which no legal riichi has"),
            # Unpreviewed, a reach must name its one tile.
            ({4: {3: all_last.SHORT, 20: all_last.KEEPS_FIRST}}, "not one"),
        ):
            arena = Codes(codes)
            who = [WHO[0]] if 0 in codes else [WHO[4]]
            rows = [0] if 0 in codes else [4]
            ahead = [WHO[0]] if 0 in codes else []
            question = allowed[rows]
            if ahead:
                preview = np.zeros((1, zoo.MORTAL_ACTIONS), dtype=bool)
                preview[0, :KINDS] = LEGAL[0, zoo.RIICHI_DISCARD:zoo.TSUMO]
                question = np.concatenate([question, preview])
            with self.subTest(codes=codes), self.assertRaisesRegex(RuntimeError, why):
                all_last.ruled(Answers(ROWS, PREVIEWS), all_last.Account())(Told(arena), who, False, question, ahead)
        # Asked of views that name no arena, as a renamed game's.
        with self.assertRaisesRegex(RuntimeError, "outermost"):
            all_last.ruled(Answers(ROWS, PREVIEWS), all_last.Account())(
                SimpleNamespace(), [WHO[4]], False, allowed[[4]])


class SeatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.paths = checkpoints(Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_only_a_player_asked_in_fixed_rows_with_the_reach_previewed_follows_it(self):
        fixed = {"questions": "fixed", "rows": 4, "preview_reach": True}
        for name in ("fused", "reheaded", "mortal"):
            with self.subTest(player=name):
                player = ruler.seat(self.paths[name], "cpu", **fixed, all_last_dama=True)
                self.assertEqual(all_last.account_of(player), all_last.Account())
                self.assertEqual(player._ask.__name__, "dama_where_riichi_buys_nothing")
                with self.assertRaisesRegex(ValueError, "already"):
                    all_last.dama(player)
                plain = ruler.seat(self.paths[name], "cpu", **fixed)
                self.assertIsNone(all_last.account_of(plain))
                self.assertNotIn("_ask", vars(plain))
        # The rule outside an average over the suits, whose renamed game
        # names no arena.
        both = ruler.seat(self.paths["fused"], "cpu", **fixed, average_suits=True, all_last_dama=True)
        self.assertIsNotNone(suits.averaging(both))
        self.assertEqual(both._ask.__name__, "dama_where_riichi_buys_nothing")
        for options, why in (({}, "fixed questions"), ({"questions": "fixed"}, "fixed questions"),
                             ({"questions": "fixed", "skip_forced": True}, "previewed")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, why):
                ruler.seat(self.paths["fused"], "cpu", **options, all_last_dama=True)
        # Asked of the rule itself, without the seating's own refusal: a
        # seating that does not say the questions are fixed is refused,
        # whatever forwards the player holds.
        with self.assertRaisesRegex(ValueError, "questions fixed"):
            all_last.dama(ruler.seat(self.paths["fused"], "cpu"))
        with self.assertRaisesRegex(ValueError, "questions fixed"):
            all_last.dama(ruler.seat(self.paths["fused"], "cpu", **fixed))
        with self.assertRaisesRegex(ValueError, "preview-reach"):
            all_last.dama(ruler.seat(self.paths["fused"], "cpu", questions="fixed", rows=4), fixed=True)
        with self.assertRaisesRegex(ValueError, "Mortal's moves"):
            all_last.dama(ruler.seat(self.paths["ours, our moves"], "cpu", questions="fixed", rows=4), fixed=True)
        with patch.object(all_last, "riichi_py", SimpleNamespace()), self.assertRaisesRegex(RuntimeError, "Rebuild"):
            all_last.dama(ruler.seat(self.paths["fused"], "cpu", **fixed), fixed=True)
        # Nor a player that follows a late rule of the last hand (see
        # `neural.late_rules`): one rule of the last hand a player, refused
        # by the rule itself as by the late rule's seating.
        late = ruler.seat(self.paths["fused"], "cpu", **fixed, late_rule="decline-last")
        with self.assertRaisesRegex(ValueError, "already follows a late rule of the last hand"):
            all_last.dama(late, fixed=True)
        self.assertIsNone(all_last.account_of(late))
        self.assertEqual(late._ask.__name__, "decline_a_win_that_leaves_it_last")


class BindingTests(unittest.TestCase):
    """`riichi_py.Arena.all_last_dama`, the engine's own codes."""

    def test_the_engine_answers_the_codes(self):
        self.assertGreaterEqual(riichi_py.ALL_LAST_API_VERSION, 1)
        arena = riichi_py.Arena(games=3, seed=SEED)
        codes = arena.all_last_dama([2, 0, 0])
        self.assertIsInstance(codes, bytes)
        # The first hand is East 1's, so nothing is asked.
        self.assertEqual(codes, bytes(3 * KINDS))
        self.assertEqual(arena.all_last_dama([]), b"")
        with self.assertRaisesRegex(ValueError, "no game 3"):
            arena.all_last_dama([0, 3])

    def test_the_codes_agree_with_the_legal_moves_and_the_hand(self):
        """Games the heuristic player plays at every seat, asked at every
        step: codes exactly where a non-dealer owes a turn's move in South 4
        with riichi open (the hand read off the arena's own events), and
        there a riichi code exactly on the tiles riichi may throw, the other
        tiles held not waiting, and nothing for the tiles not held."""
        games = 48
        arena = riichi_py.Arena(games=games, seed=SEED + 7)
        last = [None] * games
        asked = kept_first = 0
        while not arena.all_finished():
            for game, lines in enumerate(arena.mjai_all()):
                for line in lines:
                    if '"start_kyoku"' in line:
                        last[game] = json.loads(line)
            seats = np.frombuffer(arena.seats(), dtype=np.uint8)
            legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, riichi_py.ACTIONS).astype(bool)
            players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
            codes = np.frombuffer(arena.all_last_dama(list(range(games))), dtype=np.uint8).reshape(games, KINDS)
            for game in range(games):
                live = seats[game] != 0xFF
                turn = live and legal[game, :KINDS].any()
                reaches = legal[game, zoo.RIICHI_DISCARD:zoo.TSUMO]
                want = (turn and last[game]["bakaze"] == "S" and last[game]["kyoku"] == 4
                        and int(players[game, seats[game]]) != last[game]["oya"] and reaches.any())
                self.assertEqual(bool(codes[game].any()), bool(want), (game, last[game], codes[game]))
                if not want:
                    continue
                asked += 1
                kept_first += int((codes[game] == all_last.KEEPS_FIRST).any())
                code = codes[game]
                np.testing.assert_array_equal((code >= all_last.KEEPS_FIRST) & (code <= all_last.DEAD), reaches)
                np.testing.assert_array_equal(code == all_last.NOT_WAITING, legal[game, :KINDS] & ~reaches)
                np.testing.assert_array_equal(code == all_last.ABSENT, ~legal[game, :KINDS])
            moves = np.frombuffer(arena.teacher(), dtype=np.uint8).astype(np.int64)
            moves[moves == 0xFF] = 0
            arena.step(moves.tolist())
        self.assertGreater(asked, 5)
        self.assertGreater(kept_first, 0)


class Coded:
    """An arena whose last-hand codes are `code` on every tile its legal
    moves let a reach throw, at any hand: a stand-in for the engine's, which
    a ruler of one deal a seating seldom puts to the test. Everything else
    is the arena's own; `riichi_py.Arena` cannot be subclassed, so this
    hands it on. Each table's events are kept as they are handed out."""

    def __init__(self, arena, code: int) -> None:
        self.arena = arena
        self.code = code
        self.events: list[list[str]] = [[] for _ in range(arena.games)]

    def all_last_dama(self, games):
        legal = np.frombuffer(self.arena.legal_mask(), dtype=np.uint8).reshape(-1, riichi_py.ACTIONS).astype(bool)
        codes = np.zeros((len(games), KINDS), dtype=np.uint8)
        codes[legal[list(games), zoo.RIICHI_DISCARD:zoo.TSUMO]] = self.code
        return codes.tobytes()

    def mjai_all(self):
        lines = self.arena.mjai_all()
        for kept, more in zip(self.events, lines):
            kept.extend(more)
        return lines

    def __getattr__(self, name):
        return getattr(self.arena, name)


@contextlib.contextmanager
def coded(code: int):
    """Every ruler's views served from a `Coded` arena of `code`, asking
    the heuristic player once a step (see `test_ruler.Taught`); yields the
    views made."""
    made = []

    class CodedViews(Taught):
        def __init__(self, arena, *args, **kwargs):
            super().__init__(Coded(arena, code), *args, **kwargs)
            made.append(self)

    with patch.object(duel, "Views", CodedViews):
        yield made


def steer_reaches(player):
    """`player` pushed, in each reach's preview, towards the tile the
    heuristic player would throw with it: which tile a reach throws then
    turns on no near tie of a small random network, whose last bits on the
    processor can depend on where a row sits in its question."""
    asked = player._ask

    def steered(views, who, fresh, allowed, ahead=()):
        values, own = asked(views, who, fresh, allowed, ahead)
        for at, (game, _player) in enumerate(ahead):
            taught = int(views.taught[game])
            if zoo.RIICHI_DISCARD <= taught < zoo.TSUMO:
                values[len(who) + at, taught - zoo.RIICHI_DISCARD] += 100.0
        return values, own

    player._ask = steered
    return player


def reaches(events: list[str], player: int) -> int:
    return sum(1 for line in events if json.loads(line) == {"type": "reach", "actor": player})


class RulerTests(unittest.TestCase):
    """Whole paired rulers of small networks on the processor, one deal a
    seating, in fixed rows of four with every switch, every player pushed
    towards the heuristic player's moves and towards a reach wherever one
    is open (see `test_ruler.steer`), and the first challenger following
    the rule."""

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

    def paired(self, code: int):
        with coded(code) as made:
            first = all_last.dama(self.seated("fused"), fixed=True)
            got = ruler.ruler([first, self.seated("fused")], self.seated("mortal"), 1, SEED, "cpu",
                              questions="fixed")
        (views,) = made
        # The events of the last step, which no later step handed out.
        views.arena.mjai_all()
        return got, views.arena.events

    def test_a_rule_that_never_fires_plays_the_plain_arms_games(self):
        got, _events = self.paired(all_last.SHORT)
        ruled, plain = got["arms"]
        rule = ruled["all_last"]
        self.assertGreater(rule["riichi"], 0)
        self.assertEqual((rule["fired"], rule["tables"], rule["hands"]), (0, [], 0))
        self.assertEqual(rule["kept"]["short"], rule["riichi"])
        self.assertNotIn("all_last", plain)
        self.assertEqual(ruled["by_deal"], plain["by_deal"])
        self.assertEqual(got["difference"]["same_games"], 4)
        self.assertEqual(got["all_last_check"], {"differing": 0, "violations": []})

    def test_a_rule_that_fires_wherever_riichi_is_best_declares_none(self):
        got, events = self.paired(all_last.KEEPS_FIRST)
        rule = got["arms"][0]["all_last"]
        self.assertGreater(rule["fired"], 0)
        self.assertEqual(rule["riichi"], rule["fired"])
        self.assertLessEqual(set(rule["tables"]), {0, 1, 2, 3})
        # The challenger sits at place p of table p of the first arm, and of
        # table 4 + p of the second.
        self.assertEqual(sum(reaches(events[place], place) for place in range(4)), 0)
        self.assertGreater(sum(reaches(events[4 + place], place) for place in range(4)), 0)
        check = got["all_last_check"]
        self.assertGreater(check["differing"], 0)
        self.assertEqual(check["violations"], [])
        # Each table it fired at plays as the plain arm's table of its deal
        # and seat until the plain challenger declares, and there the ruled
        # one throws, undeclared, the very tile the reach throws.
        for table in rule["tables"]:
            ruled, plain = ([json.loads(line) for line in events[at]] for at in (table, table + 4))
            part = next(at for at, (mine, theirs) in enumerate(zip(ruled, plain)) if mine != theirs)
            with self.subTest(table=table):
                self.assertEqual(plain[part], {"type": "reach", "actor": table})
                self.assertEqual(ruled[part]["type"], "dahai")
                self.assertEqual(ruled[part], plain[part + 1])


class CheckTests(unittest.TestCase):
    def test_every_table_that_differs_must_be_one_the_rule_fired_at(self):
        layout = ruler.Layout(2, 3)
        scores = np.arange(layout.games * 4, dtype=np.int32).reshape(layout.games, 4) % 7
        # The second arm the first's, but at deal 1 of place 2 (tables 7
        # and 19) and deal 0 of place 3 (tables 9 and 21).
        scores[12:] = scores[:12]
        scores[19, 0] += 1000
        scores[21, 3] -= 1000
        fired = all_last.Account(tables={7, 2})
        self.assertEqual(ruler.dama_check(scores, layout, [fired, None]),
                         {"differing": 2, "violations": [{"deal": 0, "place": 3, "tables": [9, 21]}]})
        # A firing at either arm's table answers for it, and a table the
        # rule fired at may still end the same.
        self.assertEqual(ruler.dama_check(scores, layout, [fired, all_last.Account(tables={21})])["violations"], [])
        self.assertEqual(ruler.dama_check(scores, layout, [None, all_last.Account(tables={19, 21})])["violations"],
                         [])

    def run_main(self, check: dict | None, *extra: str, also: str = "a.pt"):
        """`neural.ruler`'s command, with its players and its play stood
        in for: what it wrote, what it said, and how it ended."""
        arm = {"placement": 2.5, "standard_error": 0.1, "by_deal": [2.5]}
        result = {"seed": 5, "games_per_seat": 1, "arms": [dict(arm), dict(arm)]}
        if check is not None:
            result["all_last_check"] = check
        out, said = io.StringIO(), io.StringIO()
        with patch.object(ruler, "seat", return_value=None), patch.object(ruler, "ruler", return_value=result), \
                patch.object(sys, "argv", ["ruler", "a.pt", "b.pt", "--also", also, "--questions", "fixed",
                                           "--preview-reach", "--all-last-dama", "first", *extra]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(said):
            try:
                ruler.main()
            except SystemExit as stop:
                return out.getvalue(), said.getvalue(), stop.code
        return out.getvalue(), said.getvalue(), None

    def test_a_failed_check_is_kept_in_the_report_and_the_ruler_ends_as_usual(self):
        """So that the cloud keeps the report and the launcher, not a
        respawn, deals with it (see `test_ruler_cloud`)."""
        violation = {"deal": 0, "place": 1, "tables": [1, 5]}
        out, said, code = self.run_main({"differing": 3, "violations": [violation]})
        self.assertIsNone(code)
        self.assertEqual(json.loads(out)["all_last_check"]["violations"], [violation])
        self.assertIn("the all-last dama check failed: at 1 of the 3 deals and seats", said)
        out, said, code = self.run_main({"differing": 3, "violations": []})
        self.assertEqual((json.loads(out)["all_last_check"]["violations"], said, code), ([], "", None))

    def test_the_rule_compares_one_checkpoint_with_itself_asked_alike(self):
        """`--also` must name the challenger's checkpoint, as itself or as a
        copy of the same bytes, as `modal_app.ruler` names it, and the two
        arms must average over the suits alike."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, data in (("a.pt", b"one network"), ("copy.pt", b"one network"), ("other.pt", b"two network")):
                (root / name).write_bytes(data)
            with contextlib.chdir(root):
                for also, extra in (("a.pt", []), ("copy.pt", []), ("a.pt", ["--average-suits", "both"])):
                    with self.subTest(also=also, extra=extra):
                        _out, _said, code = self.run_main({"differing": 0, "violations": []}, *extra, also=also)
                        self.assertIsNone(code)
                for also, extra, why in (("other.pt", [], "--also must name the challenger's own checkpoint"),
                                         ("missing.pt", [], "--also must name the challenger's own checkpoint"),
                                         ("a.pt", ["--average-suits", "first"], "--average-suits first"),
                                         ("copy.pt", ["--average-suits", "second"], "--average-suits second")):
                    with self.subTest(also=also, extra=extra):
                        _out, said, code = self.run_main(None, *extra, also=also)
                        self.assertEqual(code, 2)
                        self.assertIn(why, said)
            with patch.object(sys, "argv", ["ruler", "a.pt", "b.pt", "--questions", "fixed", "--preview-reach",
                                            "--all-last-dama", "first"]), \
                    contextlib.redirect_stderr(io.StringIO()) as said, self.assertRaises(SystemExit):
                ruler.main()
            self.assertIn("--also must name the challenger's own checkpoint", said.getvalue())

    def test_the_rule_needs_its_switches_and_a_second_challenger_named(self):
        for argv, why in ((["--all-last-dama", "second"], "--also"), (["--all-last-dama", "both"], "--also"),
                          (["--all-last-dama", "first"], "--questions fixed and --preview-reach"),
                          (["--all-last-dama", "first", "--questions", "fixed"], "--preview-reach")):
            with self.subTest(argv=argv), patch.object(sys, "argv", ["ruler", "a.pt", "b.pt", *argv]), \
                    contextlib.redirect_stderr(io.StringIO()) as said, self.assertRaises(SystemExit):
                ruler.main()
            self.assertIn(why, said.getvalue())


class CommandTests(unittest.TestCase):
    """`python -m neural.ruler` with the rule, as the desktop's pilot and
    every call on Modal run it. Run so, the module is `__main__`, a module
    apart from the `neural.ruler` that `all_last` imports, with classes of
    its own, and the rule was once refused there however it was asked."""

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

    def test_the_command_seats_the_rule_and_reports_it(self):
        """On tiny networks, one deal a seating, the second challenger a
        copy of the first's checkpoint, as `modal_app.ruler` makes one."""
        fused, mortal = self.paths["fused"], self.paths["mortal"]
        copy = fused.with_name("2-fused.pt")
        shutil.copyfile(fused, copy)
        out = io.StringIO()
        argv = ["ruler", str(fused), str(mortal), "--also", str(copy), "--questions", "fixed", "--rows", "4",
                "--preview-reach", "--skip-forced", "--games", "1", "--seed", str(SEED), "--device", "cpu",
                "--all-last-dama", "first"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), warnings.catch_warnings():
            # Said by `runpy` of a module the test has imported already, which is the point.
            warnings.filterwarnings("ignore", "'neural.ruler' found in sys.modules", RuntimeWarning)
            try:
                runpy.run_module("neural.ruler", run_name="__main__")
            except SystemExit as stop:
                self.assertIn(stop.code, (None, 0))
        report = json.loads(out.getvalue())
        ruled, plain = report["arms"]
        self.assertEqual(set(ruled["all_last"]), {"riichi", "fired", "tables", "hands", "kept", "riichi_after_firing"})
        self.assertNotIn("all_last", plain)
        self.assertEqual(set(report["all_last_check"]), {"differing", "violations"})


if __name__ == "__main__":
    unittest.main()
