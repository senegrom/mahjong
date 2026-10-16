"""Characters and circles renamed (`neural.suits`). The renaming is its own
inverse on tiles, on Mortal's moves and on mjai lines, and renames a whole
game exactly as Mortal's own augmentation does. The second follower, fed
every event renamed and told every reach the first is told, encodes each
position as the renamed game replayed from its start encodes it, and
everything but the efficiency lookahead as the game's own planes renamed.
A player that averages asks only the questions it hesitates over of the
renamed game, in a question of their own, and where its two views
disagree it plays the mean of the two. With the option off a ruler
follows nothing more, and a player that averages but never hesitates
plays the plain player's games."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import riichi_py
import torch
from libriichi.dataset import GameplayLoader
from libriichi.state import PlayerState

from neural import duel, observe, ruler, suits, zoo
from neural.tests.test_bfloat16_players import checkpoints, fusion
from neural.tests.test_ruler import SEED, Taught, steer

CUDA = torch.cuda.is_available()
REACH = zoo.RIICHI_DISCARD
TSUMO = zoo.TSUMO
#: Rows of Mortal's v4 planes whose row names a tile as well as their
#: columns (engine/libriichi/src/state/obs_repr.rs): the tiles each discard
#: needs, keeping shanten (891-924) or lowering it (925-958), a row a
#: discard; and the red fives, a row a suit, which our tables never deal.
TILE_ROWS = np.arange(1012)
for _start in (891, 925):
    TILE_ROWS[_start:_start + 34] = _start + suits.KINDS
for _a, _b in [(4, 5), (32, 33)] + [(723 + 7 * k + 4, 723 + 7 * k + 5) for k in range(4)]:
    TILE_ROWS[_a], TILE_ROWS[_b] = _b, _a
#: The rows the efficiency lookahead breaks ties in by tile index,
#: characters before circles, so that the renamed game's planes may differ
#: there from the game's planes renamed: its best expected points, the
#: discard with the most tiles needed, and its tables by imagined turn.
LOOKAHEAD = np.r_[889:891, 959:1012]
EXACT = np.setdiff1d(np.arange(1012), LOOKAHEAD)


def renamed_planes(planes: np.ndarray) -> np.ndarray:
    """The game's planes, (..., 1012, 34), with characters and circles
    renamed in place: every tile column, and every row that names a tile."""
    return planes[..., TILE_ROWS, :][..., suits.KINDS]


def dense(planes: observe.Planes) -> np.ndarray:
    return planes.dense("cpu").numpy()


def replayed(lines: list[str], player: int) -> tuple[np.ndarray, np.ndarray]:
    """A player's planes and Mortal's own mask after `lines`, from a state
    fed them from the start, the values halved as the follower halves them."""
    state = PlayerState(player)
    for line in lines:
        state.update(line)
    planes, mask = state.encode_obs(4, False)
    return np.asarray(planes, dtype=np.float32).astype(np.float16).astype(np.float32), np.asarray(mask, dtype=bool)


def heuristic(arena) -> np.ndarray:
    """The heuristic player's move at every table, nought at a finished one."""
    moves = np.frombuffer(arena.teacher(), dtype=np.uint8).astype(np.int64)
    moves[moves == 0xFF] = 0
    return moves


class Logged:
    """An arena whose mjai events are kept, a log a game, as they are
    handed out to whoever follows them."""

    def __init__(self, arena, games: int) -> None:
        self.arena = arena
        self.logs: list[list[str]] = [[] for _ in range(games)]

    def mjai_all(self):
        lines = self.arena.mjai_all()
        for log, more in zip(self.logs, lines):
            log.extend(more)
        return lines

    def __getattr__(self, name):
        return getattr(self.arena, name)


def whole_game(seed: int) -> list[str]:
    """Every event of one game the heuristic player plays at every seat."""
    arena = Logged(riichi_py.Arena(games=1, seed=seed), 1)
    while not arena.all_finished():
        arena.mjai_all()
        arena.step(heuristic(arena).tolist())
    arena.mjai_all()
    return arena.logs[0]


class RenamingTests(unittest.TestCase):
    def test_the_swap_is_its_own_inverse(self):
        self.assertEqual(suits.KINDS.tolist()[:18], [*range(9, 18), *range(9)])
        self.assertEqual(suits.KINDS.tolist()[18:], list(range(18, 34)))
        np.testing.assert_array_equal(suits.KINDS[suits.KINDS], np.arange(34))
        # The red fives' discards trade places too; riichi to pass name no tile.
        self.assertEqual(suits.MOVES.tolist()[34:], [35, 34, *range(36, 46)])
        np.testing.assert_array_equal(suits.MOVES[suits.MOVES], np.arange(46))
        values = np.random.default_rng(3).normal(size=(5, 46))
        np.testing.assert_array_equal(suits.swap_moves(suits.swap_moves(values)), values)
        self.assertEqual(suits.swap_moves(values)[:, 9].tolist(), values[:, 0].tolist())
        lines = {
            '{"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}':
                '{"type":"dahai","actor":0,"pai":"5p","tsumogiri":false}',
            '{"type":"pon","actor":1,"target":3,"pai":"9p","consumed":["9p","9p"]}':
                '{"type":"pon","actor":1,"target":3,"pai":"9m","consumed":["9m","9m"]}',
            '{"type":"chi","actor":2,"target":1,"pai":"5mr","consumed":["4m","6m"]}':
                '{"type":"chi","actor":2,"target":1,"pai":"5pr","consumed":["4p","6p"]}',
            '{"type":"dora","dora_marker":"7s"}': '{"type":"dora","dora_marker":"7s"}',
            '{"type":"hora","actor":0,"target":0,"pai":"1m","uradora_markers":["E","2p"],"ura_markers":["E","2p"]}':
                '{"type":"hora","actor":0,"target":0,"pai":"1p","uradora_markers":["E","2m"],"ura_markers":["E","2m"]}',
            # Words that hold an m or a p, and numbers, stay as they are.
            '{"type":"start_game","names":["player 0","player 1","player 2","player 3"]}':
                '{"type":"start_game","names":["player 0","player 1","player 2","player 3"]}',
            '{"type":"ryukyoku","reason":"exhaustive","tenpais":[true,false,false,true],"deltas":[1500,-1500,-1500,1500]}':
                '{"type":"ryukyoku","reason":"exhaustive","tenpais":[true,false,false,true],"deltas":[1500,-1500,-1500,1500]}',
        }
        for line, renamed in lines.items():
            with self.subTest(line=line):
                self.assertEqual(suits.rename_line(line), renamed)
                self.assertEqual(suits.rename_line(renamed), line)
        game = whole_game(SEED)
        renamed = [suits.rename_line(line) for line in game]
        self.assertEqual([suits.rename_line(line) for line in renamed], game)
        self.assertGreater(sum(a != b for a, b in zip(game, renamed)), len(game) // 4)
        self.assertEqual(suits.rename_games([game[:3], [], game[3:5]], [2, 0]), [renamed[3:5], renamed[:3]])

    def test_a_game_is_renamed_as_mortals_own_augmentation_renames_it(self):
        """Mortal's loader renames a game's events itself (`Event::augment`)
        before encoding every decision of every player in it; ours, renamed
        as lines and loaded plain, gives the same planes, masks and moves
        to the bit. A renaming that renamed nothing would fail here, since
        Mortal's does rename."""
        game = whole_game(SEED + 1)
        mortal = GameplayLoader(4, oracle=False, augmented=True).load_log("\n".join(game))
        ours = GameplayLoader(4, oracle=False).load_log("\n".join(suits.rename_line(line) for line in game))
        self.assertEqual(len(mortal), 4)
        for theirs, mine in zip(mortal, ours):
            with self.subTest(player=theirs.take_player_id()):
                planes, masks = theirs.take_obs(), theirs.take_masks()
                self.assertGreater(len(planes), 100)
                mine_planes = mine.take_obs()
                self.assertEqual(len(mine_planes), len(planes))
                for a, b in zip(planes, mine_planes):
                    np.testing.assert_array_equal(a, b)
                for a, b in zip(masks, mine.take_masks()):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(theirs.take_actions(), mine.take_actions())


class RenamedFollowerTests(unittest.TestCase):
    """Two games, the second followed renamed as well. At every few steps
    its deciding player is encoded from both followers; at its first few
    chances to declare riichi a reach is previewed, told, encoded afresh
    and played, as a player of Mortal's moves tells one."""

    @classmethod
    def setUpClass(cls):
        arena = Logged(riichi_py.Arena(games=2, seed=SEED + 2), 2)
        views = observe.Views(arena, 2, renamed=[1])
        cls.asked: list[dict] = []
        cls.told = 0
        cls.refused = None
        step = 0
        while not arena.all_finished():
            step += 1
            views.advance()
            seats = np.frombuffer(arena.seats(), dtype=np.uint8)
            players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(2, 4)
            legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(2, zoo.ACTIONS).astype(bool)
            moves = heuristic(arena)
            if seats[1] != 0xFF:
                player = int(players[1, seats[1]])
                log = list(arena.logs[1])
                reach = json.dumps({"type": "reach", "actor": player})
                if legal[1, REACH:TSUMO].any() and cls.told < 3:
                    preview = views.renamed.sparse_and_masks([1], [player], after_reach=True)
                    views.tell(1, player, reach)
                    cls.told += 1
                    cls.asked.append({"log": log + [reach], "player": player, "told": True,
                                      "plain": views.sparse_and_masks([1], [player], fresh=True),
                                      "renamed": views.renamed.sparse_and_masks([1], [player], fresh=True),
                                      "preview": preview})
                    moves[1] = REACH + int(np.flatnonzero(legal[1, REACH:TSUMO])[0])
                elif step % 10 == 0:
                    cls.asked.append({"log": log, "player": player, "told": False,
                                      "plain": views.sparse_and_masks([1], [player]),
                                      "renamed": views.renamed.sparse_and_masks([1], [player])})
            if seats[0] != 0xFF and cls.refused is None:
                try:
                    views.renamed.sparse_and_masks([0], [int(players[0, seats[0]])])
                    cls.refused = False
                except ValueError:
                    cls.refused = True
            arena.step(moves.tolist())

    def test_each_position_is_the_renamed_game_replayed(self):
        self.assertGreater(len(self.asked), 40)
        self.assertEqual(self.told, 3)
        for asked in self.asked:
            renamed_log = [suits.rename_line(line) for line in asked["log"]]
            for name, lines in (("plain", asked["log"]), ("renamed", renamed_log)):
                planes, masks = asked[name]
                want, own = replayed(lines, asked["player"])
                np.testing.assert_array_equal(dense(planes)[0], want, err_msg=name)
                np.testing.assert_array_equal(masks[0], own, err_msg=name)
            if asked["told"]:
                # The reach previewed before it was told is the reach told.
                np.testing.assert_array_equal(dense(asked["preview"][0]), dense(asked["renamed"][0]))
                np.testing.assert_array_equal(asked["preview"][1], asked["renamed"][1])
        self.assertTrue(self.refused, "a table the views do not follow renamed is refused")

    def test_the_game_state_is_the_games_own_renamed(self):
        """Every row but the efficiency lookahead's is the game's own row
        renamed, bit for bit, and Mortal's own mask of the renamed game is
        the game's mask renamed; the lookahead's tie-breaks may differ."""
        for asked in self.asked:
            plain = renamed_planes(dense(asked["plain"][0])[0])
            renamed = dense(asked["renamed"][0])[0]
            np.testing.assert_array_equal(renamed[EXACT], plain[EXACT])
            np.testing.assert_array_equal(asked["renamed"][1][0], suits.swap_moves(asked["plain"][1][0]))

    def test_views_follow_nothing_renamed_unless_asked(self):
        arena = riichi_py.Arena(games=1, seed=SEED)
        views = observe.Views(arena, 1)
        self.assertIsNone(views.renamed)
        self.assertIsNone(views.observer.renamed)
        with self.assertRaisesRegex(ValueError, "Mortal's planes"):
            observe.Views(arena, 1, {"engine"}, renamed=[0])
        with self.assertRaisesRegex(ValueError, "games of the arena"):
            observe.Views(arena, 1, renamed=[1])


class Asked:
    """A player's question, as `zoo.choose_in_mortal_space` asks it, with
    answers made up by game, player, whether the game is renamed and
    whether a reach is previewed; every call is written down."""

    def __init__(self, renamed) -> None:
        self.renamed = renamed
        self.calls: list[tuple] = []

    @staticmethod
    def answer(game: int, player: int, renamed: bool, preview: bool) -> np.ndarray:
        rng = np.random.default_rng([game, player, int(renamed), int(preview)])
        return rng.normal(scale=2.0, size=46).astype(np.float32)

    def __call__(self, views, who, fresh, allowed, ahead=()):
        self.calls.append((views, list(who), fresh, np.array(allowed), list(ahead)))
        rows = [self.answer(game, player, views is self.renamed, False) for game, player in who]
        rows += [self.answer(game, player, views is self.renamed, True) for game, player in ahead]
        values = np.stack(rows) if rows else np.zeros((0, 46), dtype=np.float32)
        return np.where(allowed, values, -np.inf).astype(np.float32), np.zeros((len(rows), 46), dtype=bool)


class AveragingTests(unittest.TestCase):
    def setUp(self):
        self.views = SimpleNamespace(renamed=object())

    def allowed(self, rows: int) -> np.ndarray:
        allowed = np.random.default_rng(rows).random((rows, 46)) < 0.4
        allowed[:, 0] = allowed[:, 9] = True
        # One row with a single move open, which never hesitates.
        allowed[0] = False
        allowed[0, 45] = True
        return allowed

    def test_only_hesitant_rows_are_asked_again_and_answered_by_the_mean(self):
        """Rows and previews of a reach alike: the plain question is asked
        as it came; the hesitant ones go to the renamed game in a question
        of their own, as fresh as the first, their moves renamed, and come
        back as the mean of the two answers, the confident rows untouched."""
        asked = Asked(self.views.renamed)
        account = suits.Averaging(gate=0.9)
        ask = suits.averaged(asked, account)
        who = [(game, game % 4) for game in range(6)]
        ahead = [(10, 1), (11, 2)]
        allowed = self.allowed(8)
        for fresh in (False, True):
            asked.calls.clear()
            values, _own = ask(self.views, who, fresh, allowed, ahead)
            plain, again = asked.calls
            self.assertIs(plain[0], self.views)
            self.assertEqual((plain[1], plain[2], plain[4]), (who, fresh, ahead))
            np.testing.assert_array_equal(plain[3], allowed)
            first, _ = asked(self.views, who, fresh, allowed, ahead)
            hesitant = np.flatnonzero(suits.top_weight(first) < 0.9)
            self.assertNotIn(0, hesitant)
            self.assertTrue(len(hesitant) and any(row >= len(who) for row in hesitant))
            self.assertIs(again[0], self.views.renamed)
            self.assertEqual(again[1], [who[row] for row in hesitant if row < len(who)])
            self.assertEqual(again[4], [ahead[row - len(who)] for row in hesitant if row >= len(who)])
            self.assertEqual(again[2], fresh)
            np.testing.assert_array_equal(again[3], suits.swap_moves(allowed[hesitant]))
            second, _ = asked(self.views.renamed, again[1], fresh, again[3], again[4])
            want = first.copy()
            want[hesitant] = (first[hesitant] + suits.swap_moves(second)) / 2
            np.testing.assert_array_equal(values, want)
            # The moves not allowed stay out of reach.
            self.assertTrue(np.isneginf(values[~allowed]).all())
        self.assertEqual(account.asked, 16)
        self.assertEqual(account.again, 2 * len(hesitant))
        changed = (want[hesitant].argmax(axis=1) != first[hesitant].argmax(axis=1)).sum()
        self.assertEqual(account.changed, 2 * changed)

    def test_a_confident_player_never_asks_the_renamed_game(self):
        asked = Asked(None)
        account = suits.Averaging(gate=0.0)
        views = SimpleNamespace(renamed=None)
        values, _own = suits.averaged(asked, account)(views, [(0, 0), (1, 1)], False, self.allowed(2))
        self.assertEqual(len(asked.calls), 1)
        self.assertEqual((account.asked, account.again, account.changed), (2, 0, 0))
        np.testing.assert_array_equal(values, asked(views, [(0, 0), (1, 1)], False, self.allowed(2))[0])
        # Hesitant at a table that follows no renamed game, it says so.
        with self.assertRaisesRegex(RuntimeError, "follows no renamed game"):
            suits.averaged(asked, suits.Averaging())(views, [(0, 0), (1, 1)], False, self.allowed(2))

    def test_only_a_policy_in_mortals_moves_averages(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = checkpoints(Path(folder))
            for name in ("fused", "reheaded"):
                with self.subTest(player=name):
                    player = ruler.seat(paths[name], "cpu", average_suits=True)
                    self.assertEqual(suits.averaging(player), suits.Averaging(suits.GATE))
                    self.assertIn("_ask", vars(player))
                    with self.assertRaisesRegex(ValueError, "already"):
                        suits.average(player)
                    plain = ruler.seat(paths[name], "cpu")
                    self.assertIsNone(suits.averaging(plain))
                    self.assertNotIn("_ask", vars(plain))
            for name in ("mortal", "ours, our moves"):
                with self.subTest(player=name), self.assertRaisesRegex(ValueError, "no policy in Mortal's moves"):
                    ruler.seat(paths[name], "cpu", average_suits=True)
        for gate in (-0.1, 1.5):
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, "gate"):
                suits.average(fusion(), gate)

    def test_a_position_whose_two_views_disagree_gets_the_averaged_move(self):
        """A small joined player at a real table, played by the heuristic
        player until a position where it hesitates and the mean of its two
        views picks another move than the plain view; there it plays the
        mean's move, and the plain player, the plain view's."""
        torch.manual_seed(5)
        net = fusion().eval()
        plain_ask = net._ask
        arena = riichi_py.Arena(games=1, seed=SEED + 3)
        views = observe.Views(arena, 1, renamed=[0])
        found = None
        while found is None and not arena.all_finished():
            views.advance()
            seat = int(np.frombuffer(arena.seats(), dtype=np.uint8)[0])
            player = int(np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(1, 4)[0, seat])
            legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(1, zoo.ACTIONS).astype(bool)
            allowed = zoo.translatable(legal)
            if allowed.sum() > 1:
                plain, _ = plain_ask(views, [(0, player)], False, allowed)
                renamed, _ = plain_ask(views.renamed, [(0, player)], False, suits.swap_moves(allowed))
                mean = (plain + suits.swap_moves(renamed)) / 2
                best, averaged = int(plain.argmax()), int(mean.argmax())
                if (suits.top_weight(plain)[0] < suits.GATE and best != averaged
                        and zoo.MORTAL_RIICHI not in (best, averaged)):
                    found = player, legal, averaged
                    break
            arena.step(heuristic(arena).tolist())
        self.assertIsNotNone(found, "no position where the two views disagree")
        player, legal, averaged = found
        rows, players = np.array([0]), np.array([player])
        before = net.choose(views, rows, players, legal)
        suits.average(net)
        after = net.choose(views, rows, players, legal)
        self.assertEqual(after.tolist(), zoo.first_meaning(np.array([averaged]), legal).tolist())
        self.assertNotEqual(after.tolist(), before.tolist())
        self.assertEqual(net.suit_average, suits.Averaging(suits.GATE, asked=1, again=1, changed=1))


class Stop(Exception):
    """Ends a ruler once its views are made."""


class CountedTells:
    """A follower, counting what it is told."""

    def __init__(self, follower) -> None:
        self.follower = follower
        self.told = 0

    def tell(self, *told):
        self.told += 1
        return self.follower.tell(*told)

    def __getattr__(self, name):
        return getattr(self.follower, name)


class Recorded(Taught):
    """`Taught` views that keep themselves, their renamed game's follower
    counting what it is told."""

    made: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.observer.renamed is not None:
            self.observer.renamed = CountedTells(self.observer.renamed)
        Recorded.made.append(self)


@contextlib.contextmanager
def recorded():
    Recorded.made = []
    with patch.object(duel, "Views", Recorded):
        yield Recorded.made


class RulerTests(unittest.TestCase):
    """Whole rulers of small networks on the processor, one deal a seating,
    pushed towards reaches (see `test_ruler.steer`), so reaches are told
    and their tiles asked again."""

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

    def test_with_the_option_off_a_ruler_follows_nothing_more(self):
        """Seated without the option, a player asks as it always did, and a
        ruler of such players makes its views as it always did, following
        nothing renamed; that such a ruler plays the old duel's games bit
        for bit is `test_ruler.GamesTests`'. A player that averages has its
        own tables followed renamed, or every table as the incumbent."""
        made = []

        class Made(observe.Views):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                made.append(self)
                raise Stop

        plain, incumbent = (ruler.seat(self.paths[name], "cpu") for name in ("fused", "mortal"))
        self.assertNotIn("_ask", vars(plain))
        for challengers in ([plain], [plain, plain]):
            with patch.object(duel, "Views", Made), self.assertRaises(Stop):
                ruler.ruler(challengers, incumbent, 1, SEED, "cpu")
            self.assertIsNone(made[-1].renamed)
            self.assertIsNone(made[-1].observer.renamed)
        averaging = ruler.seat(self.paths["fused"], "cpu", average_suits=True)
        self.assertIsNone(ruler.renamed_tables([plain, plain], incumbent, ruler.Layout(2, 3)))
        np.testing.assert_array_equal(ruler.renamed_tables([plain, averaging], incumbent, ruler.Layout(2, 3)),
                                      np.arange(12, 24))
        np.testing.assert_array_equal(ruler.renamed_tables([plain], averaging, ruler.Layout(1, 2)), np.arange(8))

    def test_averaging_plays_the_plain_players_games_where_it_never_hesitates(self):
        """One network three times: averaging at the ruler's gate, averaging
        with a gate nothing passes, and plain. The tables of the first two
        are followed renamed and told every reach. The first asks most of
        its questions of the renamed game too, reaches' tiles among them,
        and the mean changes some of its moves; the second never asks, and
        plays the plain arm's games to the end."""
        with recorded() as made:
            arms = [steer(ruler.seat(self.paths["fused"], "cpu", average_suits=True)),
                    steer(suits.average(ruler.seat(self.paths["fused"], "cpu"), gate=0.0)),
                    steer(ruler.seat(self.paths["fused"], "cpu"))]
            incumbent = steer(ruler.seat(self.paths["mortal"], "cpu"))
            got = ruler.ruler(arms, incumbent, 1, SEED, "cpu")
        (views,) = made
        self.assertEqual(views.observer.renamed_games, list(range(8)))
        self.assertGreater(views.observer.renamed.told, 0)
        hesitant, never, plain = got["arms"]
        self.assertGreater(hesitant["suits"]["again"], hesitant["suits"]["asked"] // 2)
        self.assertGreater(hesitant["suits"]["changed"], 0)
        self.assertGreater(arms[0].asked_again, 0)
        self.assertGreater(never["suits"]["asked"], 100)
        self.assertEqual((never["suits"]["gate"], never["suits"]["again"], never["suits"]["changed"]), (0.0, 0, 0))
        self.assertNotIn("suits", plain)
        self.assertEqual(never["by_deal"], plain["by_deal"])

    def test_a_second_challenger_that_averages_needs_naming(self):
        for argv in (["--average-suits", "second"], ["--average-suits", "both"]):
            with self.subTest(argv=argv), patch.object(sys, "argv", ["ruler", "a.pt", "b.pt", *argv]), \
                    contextlib.redirect_stderr(io.StringIO()) as said, self.assertRaises(SystemExit):
                ruler.main()
            self.assertIn("--also", said.getvalue())


@unittest.skipUnless(CUDA, "the card is where fixed rows hold every answer to its own row")
class CardTests(unittest.TestCase):
    def test_in_fixed_rows_the_renamed_questions_are_asked_in_pieces_too(self):
        """On the card in bfloat16, in pieces of sixteen rows from a graph,
        with every switch: the questions asked again are put to the network
        in pieces of the same size as every other."""
        with tempfile.TemporaryDirectory() as folder:
            paths = checkpoints(Path(folder))
            options = {"questions": "fixed", "rows": 16, "graphs": True, "skip_forced": True, "preview_reach": True}
            with patch.object(duel, "Views", Taught):
                first = steer(ruler.seat(paths["fused"], "cuda", **options, average_suits=True))
                incumbent = steer(ruler.seat(paths["mortal"], "cuda", **options))
                sizes = []
                for forward in ruler.forwards_of(first, ruler.FixedRows):
                    inner = forward.forward
                    forward.forward = lambda planes, mask, inner=inner: sizes.append(len(planes)) or inner(planes, mask)
                got = ruler.ruler([first], incumbent, 2, SEED, "cuda", questions="fixed")
        self.assertEqual(set(sizes), {16})
        self.assertGreater(got["suits"]["again"], 0)


if __name__ == "__main__":
    unittest.main()
