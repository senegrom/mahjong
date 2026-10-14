"""The ruler (`neural.ruler`): every table of a duel in one arena, each dealt
from its deal's own seed. Asked as the old duel asked, it plays the old
duel's games bit for bit, on the processor and on the card, where its
players' weights are cast once and graphs answer at the very sizes asked;
asked in fixed rows on the card, a row's answer is its own, so the switches
that change a question's size change no game and two challengers that are
one network play the same games; a second challenger is read deal by deal
against the first; a timed duel plays the same games; and the cloud keeps
every report on the volume."""

import contextlib
from functools import partial
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import riichi_py
import torch

from neural import cloud_runs, duel, observe, ruler, zoo
from neural.checkpoints import atomic_save
from neural.tests.test_bfloat16_players import checkpoints
from neural.tests.test_cloud_isolation import child, controller

CUDA = torch.cuda.is_available()
#: Mortal's move for each of ours: the one it is a meaning of.
MORTAL_OF = np.isfinite(zoo.PRIORITY).argmax(axis=0)
SEED = 9_100_000


class Taught(observe.Views):
    """Views that ask the heuristic player once a step, as they advance,
    what it would do at every table, for the players `steer` pushes. The
    heuristic player draws from a generator of its own at every table, so
    asking it at each question, as often as a path asks questions, would
    have two paths push towards different moves."""

    def advance(self) -> None:
        super().advance()
        self.taught = np.frombuffer(self.arena.teacher(), dtype=np.uint8).copy()


def steer(player):
    """`player` pushed, in each step's first question, towards the
    heuristic player's move and towards a reach wherever one is open, as
    `test_encode_aside.SteeredMortal` is: a random network seldom holds a
    ready hand, so without the push a reach would hardly ever be told or
    asked ahead. Each reach's tile is its network's own. Counts the reaches
    it was pushed to and the rows it was asked again after telling one."""
    asked = player._ask
    player.reaches = player.asked_again = 0

    def steered(views, who, fresh, allowed, ahead=()):
        values, own = asked(views, who, fresh, allowed, ahead)
        if fresh:
            player.asked_again += len(who)
            return values, own
        for row, (game, _player) in enumerate(who):
            values[row, MORTAL_OF[int(views.taught[game])]] += 100.0
            values[row, zoo.MORTAL_RIICHI] += 200.0 * allowed[row, zoo.MORTAL_RIICHI]
        player.reaches += int(allowed[: len(who), zoo.MORTAL_RIICHI].sum())
        return values, own

    player._ask = steered
    return player


def same_reading(test, got: dict, want: dict) -> None:
    """Two reports of a duel the same in every figure, to the bit."""
    for name in ("games_per_seat", "games_total", "placement", "standard_error", "by_seat", "by_deal"):
        test.assertEqual(got[name], want[name], name)


class ArenaSeedsTests(unittest.TestCase):
    """`riichi_py.Arena(seeds=...)`: a table a seed."""

    def play(self, arena, games: int) -> tuple[list, np.ndarray]:
        """Every table played to the end by the heuristic player's moves:
        each table's events and the final scores."""
        events = [[] for _ in range(games)]
        steps = 0
        while not arena.all_finished():
            for table, lines in enumerate(arena.mjai_all()):
                events[table].extend(lines)
            choice = np.frombuffer(arena.teacher(), dtype=np.uint8).astype(np.int64)
            choice[choice == 0xFF] = 0
            arena.step(choice.tolist())
            steps += 1
            self.assertLess(steps, 4000)
        for table, lines in enumerate(arena.mjai_all()):
            events[table].extend(lines)
        return events, np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, 4).copy()

    def test_consecutive_seeds_deal_the_arena_of_the_first(self):
        want = self.play(riichi_py.Arena(games=3, seed=SEED), 3)
        got = self.play(riichi_py.Arena(games=3, seeds=[SEED, SEED + 1, SEED + 2]), 3)
        self.assertEqual(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
        self.assertGreater(len(want[0][0]), 200)

    def test_tables_dealt_one_seed_play_one_game(self):
        events, scores = self.play(riichi_py.Arena(games=3, seeds=[SEED + 1, SEED + 1, SEED]), 3)
        self.assertEqual(events[0], events[1])
        np.testing.assert_array_equal(scores[0], scores[1])
        self.assertNotEqual(events[0], events[2])
        alone = self.play(riichi_py.Arena(games=1, seed=SEED + 1), 1)
        self.assertEqual(events[0], alone[0][0])

    def test_the_seeds_are_checked(self):
        self.assertGreaterEqual(riichi_py.ARENA_SEEDS_API_VERSION, 1)
        for bad in ({"seeds": [1, 2]}, {"seeds": [1, 2, 3, 4]}, {"seed": 5, "seeds": [1, 2, 3]}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                riichi_py.Arena(games=3, **bad)
        # The heuristic player still takes the places it is named for.
        arena = riichi_py.Arena(games=2, seeds=[7, 7], bot_places=[1, 2, 3])
        self.assertEqual(arena.games, 2)

    def test_an_engine_without_them_is_refused_by_name(self):
        with patch.object(ruler, "riichi_py", SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "Arena\\(seeds"):
                ruler.require_seeded_arena()


class LayoutTests(unittest.TestCase):
    def test_every_deal_at_four_tables_a_challenger(self):
        layout = ruler.Layout(arms=2, deals=3)
        self.assertEqual(layout.games, 24)
        self.assertEqual(layout.seeds(10), [10, 11, 12] * 8)
        self.assertEqual(layout.block(1, 2), slice(18, 21))
        np.testing.assert_array_equal(layout.arm_of, [0] * 12 + [1] * 12)
        np.testing.assert_array_equal(layout.place_of, ([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3) * 2)
        np.testing.assert_array_equal(layout.block_of, np.repeat(np.arange(8), 3))
        # As the engine wraps its own seed + i.
        self.assertEqual(ruler.Layout(1, 2).seeds(2**64 - 1), [2**64 - 1, 0, 2**64 - 1, 0, 2**64 - 1, 0,
                                                              2**64 - 1, 0])

    def test_rows_are_asked_a_seating_at_a_time_or_all_at_once(self):
        block_of = ruler.Layout(1, 3).block_of
        rows = np.array([0, 2, 3, 7, 8, 11])
        self.assertEqual([part.tolist() for part in ruler.groups(rows, block_of, apart=True)],
                         [[0, 2], [3], [7, 8], [11]])
        self.assertEqual([part.tolist() for part in ruler.groups(rows, block_of, apart=False)], [rows.tolist()])
        self.assertEqual(ruler.groups(rows[:0], block_of, apart=True), [])


class FixedRowsTests(unittest.TestCase):
    def test_every_question_is_put_in_pieces_of_one_size(self):
        seen = []

        def forward(planes, mask):
            seen.append(len(planes))
            return planes.sum(dim=(1, 2)) + mask.sum(dim=1), None

        fixed = ruler.FixedRows(forward, 4)
        planes = torch.arange(10 * 2 * 3, dtype=torch.float32).reshape(10, 2, 3)
        mask = torch.zeros(10, 5, dtype=torch.bool)
        answer, nothing = fixed(planes, mask)
        self.assertIsNone(nothing)
        self.assertEqual(seen, [4, 4, 4])
        torch.testing.assert_close(answer, planes.sum(dim=(1, 2)), rtol=0, atol=0)
        self.assertEqual((fixed.pieces, fixed.padded), (3, 2))
        # The padding is planes of nought and every move open, as a graph's is.
        padded = []
        ruler.FixedRows(lambda planes, mask: padded.append((planes, mask)) or planes[:, 0, 0], 4)(
            planes[:1], mask[:1])
        self.assertEqual(padded[0][0][1:].abs().sum().item(), 0.0)
        self.assertTrue(padded[0][1][1:].all())
        self.assertFalse(padded[0][1][0].any())
        with self.assertRaises(ValueError):
            ruler.FixedRows(forward, 0)

    def test_each_kind_of_player_is_asked_in_fixed_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = checkpoints(Path(folder))
            for name, attribute in (("mortal", "forward"), ("reheaded", "forward"), ("fused", "deciding"),
                                    ("ours, our moves", "forward")):
                with self.subTest(player=name):
                    player = ruler.seat(paths[name], "cpu", questions="fixed", rows=8)
                    self.assertIsInstance(getattr(player, attribute), ruler.FixedRows)
                    self.assertEqual(ruler.forwards_of(player), [getattr(player, attribute)])
                    network = getattr(player, "net", player)
                    self.assertFalse(any(p.requires_grad for p in network.parameters()))
                    # Asked as the old duel asks it, it is left alone.
                    plain = ruler.seat(paths[name], "cpu")
                    self.assertEqual(ruler.forwards_of(plain), [])


class RefusalTests(unittest.TestCase):
    def test_the_switches_that_change_a_questions_size_need_fixed_questions(self):
        for flag in ("skip_forced", "preview_reach"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "unless the questions are fixed"):
                ruler.validate_questions("seating", 256, **{name: name == flag for name in
                                                             ("graphs", "skip_forced", "preview_reach")})
        # Graphs at the very sizes asked change no answer, so the old duel's
        # questions may have them.
        ruler.validate_questions("seating", 256, graphs=True, skip_forced=False, preview_reach=False)
        for bad in (("joined", 256), ("fixed", 0), ("fixed", 2.5)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ruler.validate_questions(*bad, False, False, False)

    def test_the_heuristic_player_and_an_empty_ruler_are_refused(self):
        with self.assertRaisesRegex(ValueError, "heuristic player"):
            ruler.seat("club", "cpu")
        with self.assertRaisesRegex(ValueError, "challenger"):
            ruler.play([], None, 1, SEED, "cpu")


class ReportTests(unittest.TestCase):
    def scores(self, layout: ruler.Layout, rng) -> np.ndarray:
        scores = rng.integers(0, 60, size=(layout.games, 4)) * 1000
        return scores.astype(np.int32)

    def test_a_ruler_of_one_reports_as_the_duel_reports(self):
        """The same arithmetic on the same scores: the old duel's tables
        are the ruler's blocks."""
        rng = np.random.default_rng(1)
        layout = ruler.Layout(1, 5)
        scores = self.scores(layout, rng)
        tables = [scores[layout.block(0, place)] for place in range(4)]
        with patch.object(duel, "table", side_effect=lambda one, other, games, seed, place, **kw: tables[place]):
            want = duel.duel(None, None, 5, SEED, "cpu")
        with patch.object(ruler, "play", return_value=scores):
            got = ruler.ruler([None], None, 5, SEED, "cpu")
        self.assertEqual(got, want)
        self.assertEqual(list(got), list(want))

    def test_two_challengers_are_read_deal_by_deal(self):
        rng = np.random.default_rng(2)
        layout = ruler.Layout(2, 40)
        scores = self.scores(layout, rng)
        with patch.object(ruler, "play", return_value=scores):
            got = ruler.ruler([None, None], None, 40, SEED, "cpu")
        first, second = got["arms"]
        difference = got["difference"]
        self.assertEqual(first, ruler.arm_report(scores, layout, 0))
        self.assertEqual(second, ruler.arm_report(scores, layout, 1))
        by_deal = np.array(second["by_deal"]) - np.array(first["by_deal"])
        self.assertAlmostEqual(difference["edge"], by_deal.mean())
        self.assertAlmostEqual(difference["edge"], (2.5 - first["placement"]) - (2.5 - second["placement"]))
        self.assertAlmostEqual(difference["standard_error"], by_deal.std(ddof=1) / 40 ** 0.5)
        self.assertAlmostEqual(difference["unpaired_error"], np.hypot(first["standard_error"],
                                                                       second["standard_error"]))
        self.assertAlmostEqual(difference["correlation"], np.corrcoef(first["by_deal"], second["by_deal"])[0, 1])
        # Two arms that placed the same on every table differ by nothing, exactly.
        scores[layout.block(1, 0).start:] = scores[: layout.block(1, 0).start]
        with patch.object(ruler, "play", return_value=scores):
            same = ruler.ruler([None, None], None, 40, SEED, "cpu")["difference"]
        self.assertEqual((same["edge"], same["standard_error"], same["same_deals"], same["same_games"]),
                         (0.0, 0.0, 40, 160))
        self.assertEqual(same["verdict"], "the same placements on every deal")


class GamesTests(unittest.TestCase):
    """Whole duels of small networks on the processor, the challenger a
    joined player and the incumbent a Mortal, both pushed towards reaches
    (see `steer`), so a reach is told and asked again, or asked ahead,
    hundreds of times."""

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

    def seated(self, name: str, device: str = "cpu", **options):
        return steer(ruler.seat(self.paths[name], device, **options))

    def test_asked_by_seating_it_plays_the_old_duels_games(self):
        with patch.object(duel, "Views", Taught):
            old = [steer(zoo.load_player(self.paths[name], "cpu")) for name in ("fused", "mortal")]
            want = duel.duel(*old, 2, SEED, "cpu")
            new = [self.seated(name) for name in ("fused", "mortal")]
            got = ruler.ruler([new[0]], new[1], 2, SEED, "cpu", questions="seating")
        same_reading(self, got, want)
        for before, after in zip(old, new):
            self.assertGreater(after.reaches, 5)
            # Told and asked again, as the old duel asked: the same rows.
            self.assertEqual((after.reaches, after.asked_again), (before.reaches, before.asked_again))

    def test_in_fixed_rows_every_question_is_asked_in_pieces(self):
        """Whole duels in pieces of four rows with the switches on: every
        forward is asked at four rows, the rows with one move go unasked,
        and a reach's tile is asked with the reach, so nothing is told. That
        no switch changes a game is the card's property, tested there: on
        the processor a row's answer can depend on where it sits in the
        batch."""
        with patch.object(duel, "Views", Taught):
            players = [self.seated(name, questions="fixed", rows=4, skip_forced=True, preview_reach=True)
                       for name in ("fused", "mortal")]
            sizes = []
            for player in players:
                for forward in ruler.forwards_of(player, ruler.FixedRows):
                    inner = forward.forward
                    forward.forward = lambda planes, mask, inner=inner: sizes.append(len(planes)) or inner(
                        planes, mask)
            got = ruler.ruler([players[0]], players[1], 2, SEED, "cpu", questions="fixed")
        self.assertEqual(got["games_total"], 8)
        self.assertEqual(set(sizes), {4})
        self.assertGreater(players[1].reaches, 5)
        self.assertEqual(players[1].asked_again, 0)
        self.assertGreater(sum(forward.padded for forward in ruler.forwards_of(players[1])), 0)

    def test_two_challengers_that_are_one_network_play_the_same_games(self):
        """Asked by seating, each arm's seatings are asked in batches of
        their own, the same rows in the same places, so on any device the
        two arms of one network play the same games."""
        with patch.object(duel, "Views", Taught):
            arms = [self.seated("fused") for _ in range(2)]
            got = ruler.ruler(arms, self.seated("mortal"), 2, SEED, "cpu", questions="seating")
        self.assertEqual(got["arms"][0]["by_deal"], got["arms"][1]["by_deal"])
        self.assertEqual((got["difference"]["edge"], got["difference"]["standard_error"],
                          got["difference"]["same_games"]), (0.0, 0.0, 8))

    def test_a_timed_duel_plays_the_same_games_and_says_where_the_time_went(self):
        players = [zoo.load_player(self.paths[name], "cpu") for name in ("reheaded", "mortal")]
        plain = duel.duel(*players, 1, SEED, "cpu")
        timed = duel.duel(*players, 1, SEED, "cpu", timing={})
        old = timed.pop("play")
        self.assertEqual(timed, plain)
        self.assertEqual(old["path"], "duel")
        self.assertEqual(set(old["parts"]), {*duel.Split.PARTS, "other"})
        self.assertAlmostEqual(sum(old["parts"].values()), old["seconds"], delta=0.01)
        self.assertGreater(old["parts"]["encode"], 0)
        self.assertGreater(old["decisions"]["incumbent"], 2 * old["decisions"]["challengers"])
        self.assertEqual(old["aside"], 0.0)
        # The ruler encodes the incumbent's views aside, says how much, and
        # asks the same rows of each side.
        seated = [ruler.seat(self.paths[name], "cpu") for name in ("reheaded", "mortal")]
        got = ruler.ruler([seated[0]], seated[1], 1, SEED, "cpu", timing={}, how={"questions": "seating"})
        new = got.pop("play")
        self.assertEqual(got, plain)
        self.assertEqual((new["path"], new["questions"]), ("ruler", "seating"))
        self.assertGreater(new["aside"], 0)
        self.assertEqual(new["decisions"], old["decisions"])
        # The old duel's four tables take their steps one after another; the
        # ruler's, together, as many as the longest.
        self.assertLess(new["steps"], old["steps"])
        self.assertGreaterEqual(4 * new["steps"], old["steps"])


@unittest.skipUnless(CUDA, "the card's bfloat16 is what moves bits")
class CardTests(unittest.TestCase):
    """The same on the card, where the ruler's players hold their weights
    in bfloat16, cast once, and the old duel's are cast by autocast on every
    question; and in fixed rows answered from a CUDA graph."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.paths = checkpoints(Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_asked_by_seating_it_plays_the_old_duels_games(self):
        """With and without graphs at the sizes asked, recorded here after
        a size's first question so that most questions come from them."""
        with patch.object(duel, "Views", Taught):
            old = [steer(zoo.load_player(self.paths[name], "cuda")) for name in ("fused", "mortal")]
            want = duel.duel(*old, 3, SEED, "cuda")
            for graphs in (False, True):
                new = [ruler.seat(self.paths[name], "cuda", graphs=graphs) for name in ("fused", "mortal")]
                self.assertEqual(new[0].fuse.tiles[0].weight.dtype, torch.bfloat16)
                exact = [forward for player in new for forward in ruler.forwards_of(player, ruler.ExactGraphs)]
                self.assertEqual(len(exact), 2 if graphs else 0)
                for forward in exact:
                    forward.after = 2
                new = [steer(player) for player in new]
                got = ruler.ruler([new[0]], new[1], 3, SEED, "cuda", questions="seating")
                with self.subTest(graphs=graphs):
                    same_reading(self, got, want)
                    if graphs:
                        replays = sum(forward.graphed.replays for forward in exact)
                        asked = sum(sum(forward.asked.values()) for forward in exact)
                        self.assertGreater(replays, asked // 2)

    def test_in_fixed_rows_graphs_and_switches_change_no_game(self):
        """Pieces of sixteen rows, so most questions are cut and padded; the
        rows left unasked and the tiles asked ahead change which rows sit
        beside which, and no answer."""
        results = []
        with patch.object(duel, "Views", Taught):
            for switches in ({}, {"graphs": True}, {"graphs": True, "skip_forced": True, "preview_reach": True}):
                players = [steer(ruler.seat(self.paths[name], "cuda", questions="fixed", rows=16, **switches))
                           for name in ("fused", "mortal")]
                results.append(ruler.ruler([players[0]], players[1], 3, SEED, "cuda", questions="fixed"))
                self.assertGreater(players[1].reaches, 5)
        same_reading(self, results[1], results[0])
        same_reading(self, results[2], results[0])

    def test_in_fixed_rows_two_challengers_that_are_one_network_play_the_same_games(self):
        """The incumbent is asked about both arms' tables at once, where a
        table of the second arm sits elsewhere in the question than its
        twin of the first: its answers are the same all the same."""
        with patch.object(duel, "Views", Taught):
            options = {"questions": "fixed", "rows": 16, "graphs": True, "skip_forced": True, "preview_reach": True}
            arms = [steer(ruler.seat(self.paths["fused"], "cuda", **options)) for _ in range(2)]
            incumbent = steer(ruler.seat(self.paths["mortal"], "cuda", **options))
            got = ruler.ruler(arms, incumbent, 3, SEED, "cuda", questions="fixed")
        self.assertEqual(got["arms"][0]["by_deal"], got["arms"][1]["by_deal"])
        self.assertEqual((got["difference"]["edge"], got["difference"]["same_games"]), (0.0, 12))


class CloudTests(unittest.TestCase):
    """`modal_app.ruler`, with the volume and the child stood in for."""

    def call(self, report: dict, *, code: int = 0, **arguments):
        app = controller()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            volume, scratch = root / "volume", root / "scratch"
            atomic_save({"generation": 910}, volume / "fusion-long/history/gen-00910.pt")
            atomic_save({"generation": 810}, volume / "fusion-long/history/gen-00810.pt")
            atomic_save({"mortal": {}}, volume / "zoo/mortal_298k.pt")
            calls = []

            def popen(command, **kwargs):
                calls.append(command)
                return child(json.dumps(report), "a warning", code)

            with patch.object(app, "VOLUME", volume), \
                    patch.object(app, "workspace", partial(cloud_runs.workspace, root=scratch)), \
                    patch.object(app, "_environment", return_value={}), \
                    patch.object(app.subprocess, "Popen", side_effect=popen), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                answer = app.ruler(run="fusion-long", **arguments)
            kept = sorted(str(path.relative_to(volume)).replace("\\", "/") for path in (volume / "rulers").rglob("*")) \
                if (volume / "rulers").exists() else []
            texts = {name: (volume / name).read_text(encoding="utf-8") for name in kept if name.endswith(".json")}
            self.assertEqual(list(scratch.iterdir()), [])
        return answer, calls, texts

    def test_a_paired_call_is_kept_on_the_volume_with_the_volumes_names(self):
        report = {"seed": 9_100_000, "arms": [{"challenger": "/scratch/x/0.pt", "by_deal": [2.4]},
                                              {"challenger": "/scratch/x/2.pt", "by_deal": [2.5]}],
                  "incumbent": "/scratch/x/1.pt"}
        answer, calls, texts = self.call(
            report, challenger="history/gen-00910", also="history/gen-00810", incumbent="zoo/mortal_298k",
            seed=9_100_000, tag="g910-g810", questions="fixed", graphs=True, skip_forced=True)
        got = json.loads(answer)
        self.assertEqual([arm["challenger"] for arm in got["arms"]], ["history/gen-00910", "history/gen-00810"])
        self.assertEqual((got["incumbent"], got["run"]), ("zoo/mortal_298k", "fusion-long"))
        self.assertEqual(got["generations"]["history/gen-00910"], "910")
        self.assertEqual(texts, {"rulers/g910-g810/seed-9100000.json": answer})
        (command,) = calls
        self.assertEqual(command[1:3], ["-m", "neural.ruler"])
        self.assertEqual([Path(path).name for path in (command[3], command[4], command[command.index("--also") + 1])],
                         ["0-history--gen-00910.pt", "1-zoo--mortal_298k.pt", "2-history--gen-00810.pt"])
        for flag in ("--graphs", "--skip-forced", "--timing"):
            self.assertIn(flag, command)
        self.assertNotIn("--preview-reach", command)
        self.assertEqual(command[command.index("--questions") + 1], "fixed")
        self.assertEqual(command[command.index("--rows") + 1], "256")

    def test_without_a_tag_nothing_is_kept_and_bad_requests_fail_before_work(self):
        answer, calls, texts = self.call({"seed": 5, "by_deal": [2.5], "challenger": "x", "incumbent": "y"},
                                         challenger="history/gen-00910", seed=5)
        self.assertEqual(texts, {})
        self.assertEqual(json.loads(answer)["challenger"], "history/gen-00910")
        self.assertNotIn("--also", calls[0])
        self.assertNotIn("--rows", calls[0])
        app = controller()
        for bad in ({"questions": "joined"}, {"skip_forced": True}, {"preview_reach": True}, {"tag": "../up"},
                    {"tag": "a/b"}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                app.ruler(challenger="x", **bad)

    def test_a_failed_child_raises(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.call({}, code=1, challenger="history/gen-00910", tag="t")

    def test_the_duel_asks_for_timing_only_when_told(self):
        app = controller()
        for timing in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                atomic_save({"generation": 5}, root / "volume/run/latest.pt")
                atomic_save({"generation": 4}, root / "volume/run/older.pt")
                calls = []
                with patch.object(app, "VOLUME", root / "volume"), \
                        patch.object(app, "workspace", partial(cloud_runs.workspace, root=root / "scratch")), \
                        patch.object(app, "_environment", return_value={}), \
                        patch.object(app.subprocess, "Popen",
                                     side_effect=lambda command, **kw: calls.append(command) or child("{}")), \
                        contextlib.redirect_stdout(io.StringIO()):
                    app.duel(challenger="latest", incumbent="older", run="run", timing=timing)
                with self.subTest(timing=timing):
                    self.assertEqual("--timing" in calls[0], timing)
                    self.assertEqual(calls[0][:3], [sys.executable, "-m", "neural.duel"])


if __name__ == "__main__":
    unittest.main()
