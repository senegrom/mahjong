"""A seated player is not asked about a row with a single move open in
Mortal's moves: the best of one move is that move whatever the network
says, so every choice stays what asking would have made it, and the
forwards shrink by the rows that had no choice to make."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import riichi_py
import torch

from neural import mortal_model, zoo
from neural.observe import Views
from neural.tests.test_step_planes import deciding


def choose_by_asking_every_row(ask, views, rows, players, legal, stats=None):
    """`zoo.choose_in_mortal_space` as it was before it left the rows with
    one move open unasked, kept as the reference."""
    who = list(zip(np.asarray(rows).tolist(), np.asarray(players).tolist()))
    legal = np.atleast_2d(legal)
    allowed = zoo.translatable(legal)
    orphan = ~allowed.any(axis=1)
    allowed[orphan, zoo.MORTAL_PASS] = True
    values, own = ask(who, False, allowed)
    ranked = np.where(allowed, values, -np.inf)
    best = ranked.argmax(axis=1)
    if stats is not None:
        stats["orphans"] = stats.get("orphans", 0) + int(orphan.sum())
        stats["fallbacks"] = stats.get("fallbacks", 0) + int(
            ((np.where(own, values, -np.inf).argmax(axis=1) != best) & ~orphan).sum()
        )
    choice = zoo.first_meaning(best, legal)
    choice = np.where(orphan | (choice < 0), legal.argmax(axis=1), choice)
    second = np.nonzero((best == zoo.MORTAL_RIICHI) & ~orphan)[0]
    if len(second):
        follower = views.observer.follower
        for i in second:
            game, player = who[i]
            follower.tell(game, player, json.dumps({"type": "reach", "actor": player}))
        tiles = legal[second, zoo.RIICHI_DISCARD:zoo.TSUMO]
        allowed_after = np.zeros((len(second), zoo.MORTAL_ACTIONS), dtype=bool)
        allowed_after[:, :34] = tiles
        after, _own_after = ask([who[i] for i in second], True, allowed_after)
        ranked_tiles = np.where(tiles, after[:, :riichi_py.POSITIONS], -np.inf)
        choice[second] = zoo.RIICHI_DISCARD + ranked_tiles.argmax(axis=1)
    return choice.astype(np.int64)


class Telling:
    """Views whose follower only writes down what it is told."""

    def __init__(self):
        self.told = []
        self.observer = SimpleNamespace(follower=self)

    def tell(self, game, player, event):
        self.told.append((game, player, json.loads(event)["type"]))


def row_values(who, fresh):
    """Values a row at a time, the same whatever else is asked with it."""
    values = np.empty((len(who), zoo.MORTAL_ACTIONS), dtype=np.float32)
    own = np.empty((len(who), zoo.MORTAL_ACTIONS), dtype=bool)
    for row, (game, player) in enumerate(who):
        rng = np.random.default_rng([game, player, int(fresh)])
        values[row] = rng.normal(size=zoo.MORTAL_ACTIONS)
        own[row] = rng.random(zoo.MORTAL_ACTIONS) < 0.3
    return values, own


class ForcedRowsTests(unittest.TestCase):
    def test_the_same_choices_without_asking_about_rows_with_one_move(self):
        rng = np.random.default_rng(17)
        asked_rows = forced = 0
        for trial in range(300):
            n = int(rng.integers(1, 12))
            legal = np.zeros((n, riichi_py.ACTIONS), dtype=bool)
            for row in range(n):
                kind = rng.integers(0, 5)
                if kind == 0:  # a declared riichi's draw: throw it
                    legal[row, rng.integers(0, 34)] = True
                elif kind == 1:  # a reach, by one tile or several
                    tiles = rng.choice(34, size=int(rng.integers(1, 4)), replace=False)
                    legal[row, tiles] = True
                    legal[row, zoo.RIICHI_DISCARD + tiles] = True
                elif kind == 2:  # a call or a pass
                    legal[row, rng.choice([68, 69, 71, 72, 73, 74, 75])] = True
                    legal[row, zoo.PASS] = True
                elif kind == 3:  # two kans, or a win either way: Mortal's one move
                    legal[row, [[75, 76], [68, 69], [zoo.PASS]][rng.integers(0, 3)]] = True
                # kind 4: nothing at all, the orphan
            rows = rng.integers(0, 50, size=n)
            players = rng.integers(0, 4, size=n)
            asked = []

            def ask(who, fresh, allowed):
                asked.append(allowed.copy())
                return row_values(who, fresh)

            old_views, new_views = Telling(), Telling()
            old_stats, new_stats = {}, {}
            want = choose_by_asking_every_row(lambda who, fresh, allowed: row_values(who, fresh),
                                              old_views, rows, players, legal, old_stats)
            got = zoo.choose_in_mortal_space(ask, new_views, rows, players, legal, new_stats)
            np.testing.assert_array_equal(got, want, err_msg=f"trial {trial}")
            self.assertEqual(new_views.told, old_views.told)
            self.assertEqual(new_stats["orphans"], old_stats["orphans"])
            # Only rows with a choice to make were asked, at either step.
            for allowed in asked:
                self.assertTrue((allowed.sum(axis=1) > 1).all())
                asked_rows += len(allowed)
            forced += int((zoo.translatable(legal).sum(axis=1) <= 1).sum())
        self.assertGreater(asked_rows, 300)
        self.assertGreater(forced, 300)

    def test_a_step_of_rows_with_one_move_each_asks_nothing(self):
        legal = np.zeros((3, riichi_py.ACTIONS), dtype=bool)
        legal[0, 5] = True
        legal[1, [68, 69]] = True
        legal[2, [75, 76, 77]] = True

        def ask(who, fresh, allowed):
            raise AssertionError("asked about a row with one move")

        choice = zoo.choose_in_mortal_space(ask, Telling(), np.arange(3), np.zeros(3), legal)
        self.assertEqual(choice.tolist(), [5, 68, 75])


class SeatedMortalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_a_seated_mortal_plays_whole_tables_the_same(self):
        """Two tables dealt alike, one played by asking every row and one
        the new way, stay in step move for move. A small Mortal is asked
        through its own encoding and forward, and its values are steered
        a game at a time, the same for both tables, towards the engine's
        heuristic move and towards a reach whenever one is allowed: a
        random network alone hardly ever has a ready hand, and here the
        draws a declared reach must throw, and the reaches only one tile
        keeps ready, are many."""
        torch.manual_seed(3)
        net = mortal_model.build(8, 1).eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        with patch.object(zoo.mortal_model, "load", return_value=net):
            player = zoo.MortalPlayer("unused", "cpu")
        games = 8
        tables = [riichi_py.Arena(games=games, seed=20261005) for _ in range(2)]
        views = [Views(table, games) for table in tables]
        counts = {"rows": 0, "asked": 0, "reaches": 0, "tiles asked": 0}
        old_stats, new_stats = {}, {}
        # Mortal's move for each of ours.
        mortal_of = np.isfinite(zoo.PRIORITY).argmax(axis=0)

        def steered(table, teacher):
            def ask(who, fresh, allowed):
                values, own = player._ask(views[table], who, fresh, allowed)
                for row, (game, _player) in enumerate(who):
                    move = int(teacher[game])
                    if fresh and zoo.RIICHI_DISCARD <= move < zoo.TSUMO:
                        values[row, move - zoo.RIICHI_DISCARD] += 100.0
                    elif not fresh and move != 0xFF:
                        values[row, mortal_of[move]] += 100.0
                        values[row, zoo.MORTAL_RIICHI] += 200.0 * allowed[row, zoo.MORTAL_RIICHI]
                if table == 1:
                    counts["tiles asked" if fresh else "asked"] += len(who)
                return values, own
            return ask

        for _step in range(1500):
            if tables[0].all_finished():
                break
            for view in views:
                view.advance()
            live, players = deciding(tables[0], games)
            np.testing.assert_array_equal(deciding(tables[1], games)[0], live)
            mask = np.frombuffer(tables[0].legal_mask(), dtype=np.uint8).reshape(games, -1)
            legal = mask.astype(bool)[live]
            teacher = np.frombuffer(tables[0].teacher(), dtype=np.uint8)
            for view in views:
                view.prepare(live, players)
            want = choose_by_asking_every_row(steered(0, teacher), views[0], live, players,
                                              legal, old_stats)
            got = zoo.choose_in_mortal_space(steered(1, teacher), views[1], live, players,
                                             legal, new_stats)
            np.testing.assert_array_equal(got, want)
            counts["rows"] += len(live)
            counts["reaches"] += int(((got >= zoo.RIICHI_DISCARD) & (got < zoo.TSUMO)).sum())
            choice = np.zeros(games, dtype=np.int64)
            choice[live] = got
            for table in tables:
                table.step(choice.tolist())
        np.testing.assert_array_equal(np.frombuffer(tables[0].seats(), dtype=np.uint8),
                                      np.frombuffer(tables[1].seats(), dtype=np.uint8))
        # A good share of the rows went unasked, among them reaches whose
        # tile was the only one, and reaches with a choice were still asked.
        self.assertGreater(counts["rows"] - counts["asked"], counts["rows"] // 20, counts)
        self.assertGreater(counts["reaches"], counts["tiles asked"], counts)
        self.assertGreater(counts["tiles asked"], 0, counts)
        self.assertEqual(new_stats.get("orphans", 0), old_stats.get("orphans", 0))
        self.assertLessEqual(new_stats.get("fallbacks", 0), old_stats.get("fallbacks", 0))


if __name__ == "__main__":
    unittest.main()
