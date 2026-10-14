"""The placement reward: where every hand began, U, and the returns settled
from them; and the promise that a round asked for none of it plays and is
paid exactly as before.

    python -m unittest neural.tests.test_placement_reward -v
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import riichi_py

from neural import place_model, scoreboard, selfplay, zoo
from neural.observe import Planes
from neural.outcomes import placement_rewards
from neural.population import Member, Population


class FirstLegal:
    """Plays the first legal move, by the network's path through
    `everything`: deterministic, so a round is the same on every machine."""

    kind = "engine"

    def eval(self):
        return self

    def choose(self, views, rows, players, legal):
        return np.asarray(legal, dtype=bool).argmax(axis=1).astype(np.int64)

    def everything(self, planes, mask):
        rows = mask.shape[0]
        first = mask.float().argmax(dim=1)
        logits = torch.full(mask.shape, float("-inf"))
        logits[torch.arange(rows), first] = 0.0
        return logits, torch.zeros(rows), torch.zeros(rows, 3, 34)


class Teacher:
    """Decides as the engine's heuristic player would and records it, by
    the path of a learner that decides for itself; of the kind named, so a
    table of the mortal kind is followed by Mortal's engine as a trainer's
    is."""

    def __init__(self, kind: str = "engine") -> None:
        self.kind = kind

    def eval(self):
        return self

    def choose(self, views, rows, players, legal):
        return np.frombuffer(views.arena.teacher(), dtype=np.uint8)[rows].astype(np.int64)

    def decide(self, views, rows, players, legal, greedy, **_options):
        chosen = self.choose(views, rows, players, legal)
        records = SimpleNamespace(
            planes=Planes(np.zeros(len(rows) + 1, dtype=np.int64), np.zeros(0, dtype=np.uint16),
                          np.zeros(0, dtype=np.float16)),
            masks=legal.copy(), actions=chosen, log_probs=np.zeros(len(rows)),
            slots=np.arange(len(rows)))
        return chosen, records


def rounds():
    """Three rounds that cover self-play's paths: a learner alone through
    `everything`; one that decides for itself with others seated by player,
    the engine's Club among them; and one whose table Mortal's engine
    follows, with an other seated in every game."""
    roster = Population([Member("zoo/mortal_298k", "reference", "", 1.5),
                         Member("club", "reference", "", 0.2)])
    return {
        "alone": lambda **extra: selfplay.play(FirstLegal(), games=4, seed=4242, device="cpu", **extra),
        "mixed": lambda **extra: selfplay.play(
            Teacher(), games=6, seed=777, device="cpu", opponents=[Teacher(), zoo.ClubPlayer()],
            seat_share=0.5, population=roster, **extra),
        "followed": lambda **extra: selfplay.play(
            Teacher("mortal"), games=3, seed=31337, device="cpu", opponents=[Teacher("mortal")],
            opponent_share=1.0, **extra),
    }


def digest(batch) -> dict:
    return {
        "decisions": int(batch.decisions),
        "hands": int(batch.hands),
        "returns": hashlib.sha256(batch.returns.numpy().tobytes()).hexdigest()[:16],
        "actions": hashlib.sha256(batch.actions.numpy().tobytes()).hexdigest()[:16],
        "final": hashlib.sha256(np.ascontiguousarray(batch.final_scores).tobytes()).hexdigest()[:16],
    }


#: What self-play answered for `rounds()` at 7a0c405, before the placement
#: reward existed. The engine deals from its seed alone and the players
#: here choose without chance, and the returns are integers' sums in double
#: precision rounded once to single, so the digests are the same on every
#: machine: a change to how the default reward is paid changes them.
TODAY = {
    "alone": {"decisions": 2715, "hands": 32, "returns": "0f664d51d8469f0d",
              "actions": "b85053772aca5449", "final": "89a47aa4f97dd617"},
    "mixed": {"decisions": 3507, "hands": 84, "returns": "00f1dffd5e44a3f4",
              "actions": "7b941e83f0efce6d", "final": "40da2a67a3352b77"},
    "followed": {"decisions": 1823, "hands": 38, "returns": "33533a4bed2eca96",
                 "actions": "db46f8acb0bd089d", "final": "b555dc6c0261a27d"},
}


def ledger_of(starts_by_game: list[list[tuple]], decisions: list[tuple], final_scores, owners=None,
              members=()) -> scoreboard.Ledger:
    """A ledger written by hand: each game's hands as (round, kyoku, honba,
    sticks, dealer, s0, s1, s2, s3), and the decisions as (game, hand,
    player, points)."""
    begun = scoreboard.HandStarts(len(starts_by_game))
    for game, starts in enumerate(starts_by_game):
        for wind, kyoku, honba, sticks, dealer, *scores in starts:
            begun.begin(game, {"bakaze": "ES"[wind], "kyoku": kyoku, "honba": honba,
                               "kyotaku": sticks, "oya": dealer, "scores": scores})
    games, hands, players, points = (np.array(column) for column in zip(*decisions))
    owners = np.full((len(starts_by_game), 4), -1) if owners is None else np.asarray(owners)
    return scoreboard.Ledger.of_round(begun, games, hands, players, points, owners, members,
                                      np.asarray(final_scores))


class DefaultRoundsUnchanged(unittest.TestCase):
    """With nothing asked of it, self-play plays and pays as it did."""

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.played = {name: make() for name, make in rounds().items()}
        cls.kept = {name: make(ledger=True) for name, make in rounds().items()}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_the_default_pays_as_it_did_at_fixed_seeds(self):
        for name, batch in self.played.items():
            with self.subTest(round=name):
                self.assertEqual(digest(batch), TODAY[name])
                self.assertIsNone(batch.ledger)

    def test_keeping_a_ledger_changes_nothing_the_round_records(self):
        """The ledger reads the events and nothing else: every field of the
        round, its returns included, is what the round without it holds."""
        for name, batch in self.kept.items():
            with self.subTest(round=name):
                plain = self.played[name]
                self.assertEqual(digest(batch), TODAY[name])
                for field in ("returns", "actions", "log_probs", "legal", "held", "oracle", "explored"):
                    one, two = getattr(batch, field), getattr(plain, field)
                    if one is None or two is None:
                        self.assertIs(one, two, field)
                        continue
                    self.assertTrue(torch.equal(one, two), field)
                for array in Planes.ARRAYS:
                    np.testing.assert_array_equal(getattr(batch.observations, array),
                                                  getattr(plain.observations, array))
                np.testing.assert_array_equal(batch.final_scores, plain.final_scores)
                self.assertEqual(batch.matchups, plain.matchups)

    def test_the_ledger_places_every_decision_in_its_hand(self):
        for name, batch in self.kept.items():
            with self.subTest(round=name):
                ledger = batch.ledger
                self.assertEqual(len(ledger), batch.decisions)
                self.assertEqual(int(ledger.count.sum()), batch.hands)
                starts = ledger.starts
                # Every game opens at East 1, the first player dealing, with
                # everyone on the starting score and nothing on the table.
                opening = starts[ledger.first]
                np.testing.assert_array_equal(opening[:, scoreboard.ROUND:scoreboard.DEALER + 1],
                                              np.tile([0, 1, 0, 0, 0], (len(opening), 1)))
                np.testing.assert_array_equal(opening[:, scoreboard.SCORES], 30_000)
                # No point is made or lost between hands: what the players
                # hold and what lies on the table always add up to the start.
                held = starts[:, scoreboard.SCORES].sum(axis=1) + 1000 * starts[:, scoreboard.STICKS]
                np.testing.assert_array_equal(held, 120_000)
                # A game's stages never go back.
                for game in range(len(ledger.first)):
                    stages = ledger.stages[ledger.first[game]:ledger.first[game] + ledger.count[game]]
                    self.assertTrue(np.all(np.diff(stages) >= 0), (game, stages))
                # The decisions are the learner's alone.
                self.assertTrue(np.all(ledger.owners[ledger.games, ledger.players] == -1))
                # Each decision's points are what its own hand moved its
                # player's score: the next hand's start less its own.
                rows = ledger.rows_of(ledger.games, ledger.hands)
                following = ledger.hands < ledger.count[ledger.games] - 1
                players = ledger.players.astype(np.int64)
                moved = (starts[rows[following] + 1, 7 + players[following]]
                         - starts[rows[following], 7 + players[following]])
                np.testing.assert_allclose(ledger.points[following] * 4000, moved, atol=1e-6)
                # And a hand's every decision is paid as the default pays.
                bonus = placement_rewards(ledger.final_scores, selfplay.PLACEMENT_VALUE)
                np.testing.assert_allclose(
                    batch.returns.numpy(),
                    (ledger.points + bonus[ledger.games, players]).astype(np.float32))

    def test_a_ledger_round_trips_through_a_file(self):
        ledger = self.kept["mixed"].ledger
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            path = Path(folder) / "round.npz"
            ledger.save(path)
            loaded = scoreboard.Ledger.load(path)
        for name in scoreboard.Ledger.ARRAYS:
            np.testing.assert_array_equal(getattr(loaded, name), getattr(ledger, name), name)
        self.assertEqual(loaded.members, ledger.members)
        self.assertEqual(loaded.members, ("zoo/mortal_298k", "club"))


class ScoreboardRefusals(unittest.TestCase):
    def test_a_negative_score_is_kept_as_the_table_holds_it(self):
        begun = scoreboard.HandStarts(1)
        begun.read([['{"type":"start_game"}',
                     '{"type":"start_kyoku","bakaze":"S","kyoku":3,"honba":2,"kyotaku":1,"oya":2,'
                     '"dora_marker":"5m","scores":[51000,-2300,40300,30000],"tehais":[]}',
                     '{"type":"tsumo","actor":2,"pai":"5m"}']])
        self.assertEqual(begun.rows, [[0, 0, 1, 3, 2, 1, 2, 51000, -2300, 40300, 30000]])
        self.assertEqual(begun.current(np.array([0])).tolist(), [0])
        seen = place_model.view(np.array(begun.rows), np.array([1]))
        np.testing.assert_allclose(seen[0, -7:-3], [-0.23, 4.03, 3.0, 5.1], rtol=1e-6)

    def test_a_round_outside_east_and_south_is_refused(self):
        begun = scoreboard.HandStarts(1)
        with self.assertRaisesRegex(ValueError, "East and South only"):
            begun.begin(0, {"bakaze": "W", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
                            "scores": [30000] * 4})

    def test_a_hand_whose_start_went_unread_is_refused(self):
        ledger = ledger_of([[(0, 1, 0, 0, 0, 30000, 30000, 30000, 30000)]], [(0, 0, 0, 0.0)],
                           [[1, 2, 3, 4]])
        ledger.check([1])
        with self.assertRaisesRegex(RuntimeError, "went unread"):
            ledger.check([2])
        ledger.hands[0] = 1
        with self.assertRaisesRegex(RuntimeError, "outside the hands"):
            ledger.check([1])


class PlaceModelReads(unittest.TestCase):
    def test_tied_places_are_shared_as_the_rewards_share_them(self):
        rng = np.random.default_rng(5)
        scores = rng.integers(-2, 3, size=(500, 4)) * 1000
        soft = place_model.soft_places(scores)
        np.testing.assert_allclose(soft.sum(axis=2), 1.0)
        np.testing.assert_allclose(soft @ np.array(place_model.VALUES),
                                   placement_rewards(scores, place_model.VALUES), atol=1e-6)
        self.assertEqual(place_model.VALUES, tuple(riichi_py.PLACEMENT_VALUE))

    def test_seats_are_read_from_the_players_own(self):
        starts = np.array([[0, 0, 0, 2, 1, 2, 3, 10000, 20000, 30000, 60000]])
        for player in range(4):
            seen = place_model.view(starts, np.array([player]))[0]
            self.assertEqual(seen[:8].tolist(), [0, 1, 0, 0, 0, 0, 0, 0])
            self.assertAlmostEqual(seen[8], 0.25)
            self.assertAlmostEqual(seen[9], 0.5)
            dealer = np.zeros(4)
            dealer[(3 - player) % 4] = 1
            self.assertEqual(seen[10:14].tolist(), dealer.tolist())
            mine = np.roll(np.array([1.0, 2.0, 3.0, 6.0]), -player)
            np.testing.assert_allclose(seen[14:18], mine, rtol=1e-6)
            np.testing.assert_allclose(seen[18:21], mine[0] - mine[1:], rtol=1e-6)

    def test_who_sits_where_is_read_and_a_stranger_refused(self):
        ledger = ledger_of([[(0, 1, 0, 0, 0, 30000, 30000, 30000, 30000)]], [(0, 0, 0, 0.0)],
                           [[1, 2, 3, 4]], owners=[[-1, 1, -1, 0]], members=("club", "zoo/mortal_298k"))
        members = ("zoo/mortal_298k", "club", "mortal-run/history/gen-00030")
        # Player 1 is published Mortal, player 3 the Club, the rest the learner.
        np.testing.assert_array_equal(place_model.kinds_of(ledger, members), [[0, 1, 0, 2]])
        x = place_model.inputs(ledger, members, np.array([0]), np.array([1]))
        self.assertEqual(x.shape, (1, place_model.width(members)))
        seats = x[0, place_model.BASE:].reshape(4, 4)
        # From player 1's chair: itself, then players 2, 3 and 0.
        self.assertEqual(seats.argmax(axis=1).tolist(), [1, 0, 2, 0])
        with self.assertRaisesRegex(ValueError, "not fitted"):
            place_model.inputs(ledger, ("club",), np.array([0]), np.array([0]))

    def test_a_model_round_trips_and_rides_in_a_checkpoint(self):
        model = place_model.PlaceModel(("club",), hidden=8)
        model.fitted = {"games": 3}
        x = np.random.default_rng(1).normal(size=(5, place_model.width(("club",)))).astype(np.float32)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            path = Path(folder) / "u.pt"
            model.save(path)
            again = place_model.PlaceModel.load(path)
            from neural.checkpoints import atomic_save

            atomic_save({"generation": 3, "place_model": model.state()}, Path(folder) / "latest.pt")
            carried = torch.load(Path(folder) / "latest.pt", map_location="cpu", weights_only=True)
        np.testing.assert_array_equal(again.expected(x), model.expected(x))
        np.testing.assert_array_equal(
            place_model.PlaceModel.from_state(carried["place_model"]).expected(x), model.expected(x))
        self.assertEqual(again.fitted, {"games": 3})
        self.assertEqual(again.members, ("club",))

    def test_a_fit_learns_what_the_scoreboard_says_and_reads_as_calibrated(self):
        """Synthetic boundaries where the leader at South 4 wins and East 1
        says nothing: U learns both, its answers on held-out games match
        the outcomes, and a U that leans one way is refused."""
        rng = np.random.default_rng(7)
        games = 3000
        starts, finals = [], []
        for game in range(games):
            lead = rng.integers(0, 4)
            scores = np.full(4, 25000)
            scores[lead] = 45000
            scores[(lead + 1) % 4] = 30000
            final = np.zeros(4, dtype=np.int64)
            final[lead] = 40000 if rng.random() < 0.9 else 1000
            final[(lead + 1) % 4] = 20000
            final[(lead + 2) % 4] = 5000 + rng.integers(0, 2) * 10000
            final[(lead + 3) % 4] = 10000
            starts.append([(0, 1, 0, 0, 0, 30000, 30000, 30000, 30000),
                           (1, 4, 0, 0, 3, *(int(s) for s in scores))])
            finals.append(final)
        ledger = ledger_of(starts, [(0, 0, 0, 0.0)], finals)
        table = place_model.Boundaries.of(ledger, ())
        answers, _fold = place_model.cross_fitted(table, (), folds=4, epochs=30, batch=512)
        report = place_model.calibration(answers, table.bonus, table.stages, table.games)
        self.assertTrue(report["passed"], report["worst"])
        by_stage = {row["stage"]: row for row in report["by_stage"]}
        self.assertLess(abs(by_stage["East 1"]["explained"]), 0.01)
        self.assertGreater(by_stage["South 4"]["explained"], 0.5)
        leaning = place_model.calibration(answers + 0.15, table.bonus, table.stages, table.games)
        self.assertFalse(leaning["passed"])
        self.assertGreater(leaning["cells_missing_beyond_noise"], 0)


class PlacementReturns(unittest.TestCase):
    def ledger(self):
        # Two games: three hands and two. Player 2 of the first game is a
        # seated other and makes no decision.
        return ledger_of(
            [[(0, 1, 0, 0, 0, 30000, 30000, 30000, 30000),
              (0, 2, 0, 1, 1, 37000, 25000, 30000, 27000),
              (0, 2, 1, 0, 1, 33000, 31000, 30000, 26000)],
             [(0, 1, 0, 0, 0, 30000, 30000, 30000, 30000),
              (0, 1, 1, 0, 0, 34000, 28000, 30000, 28000)]],
            [(0, 0, 0, 0.25), (0, 0, 1, -0.25), (0, 1, 3, 0.0), (0, 2, 0, 1.5), (0, 2, 1, -0.5),
             (1, 0, 3, -0.5), (1, 1, 0, 2.0), (1, 1, 2, 0.0)],
            [[20000, -5000, 0, -15000], [-3000, 5000, 5000, -7000]],
            owners=[[-1, -1, 0, -1], [-1, -1, -1, -1]], members=("club",))

    def test_a_hand_is_worth_the_next_boundary_and_the_last_its_bonus(self):
        ledger = self.ledger()
        at_starts = np.arange(20, dtype=np.float64).reshape(5, 4) / 10
        worth = place_model.hand_returns(ledger, at_starts)
        np.testing.assert_allclose(worth[0], at_starts[1])
        np.testing.assert_allclose(worth[1], at_starts[2])
        np.testing.assert_allclose(worth[3], at_starts[4])
        # Last hands: the bonus, ties shared (second and third tie in game 2).
        np.testing.assert_allclose(worth[2], [1.5, -0.5, 0.5, -1.5])
        np.testing.assert_allclose(worth[4], [-0.5, 1.0, 1.0, -1.5])
        returns = place_model.placement_returns(ledger, at_starts, alpha=0.0)
        np.testing.assert_allclose(returns, [0.4, 0.5, 1.1, 1.5, -0.5, 1.9, -0.5, 1.0], rtol=1e-6)
        points = place_model.placement_returns(ledger, at_starts, alpha=0.05) - returns
        np.testing.assert_allclose(points, 0.05 * ledger.points, atol=1e-6)
        with self.assertRaises(ValueError):
            place_model.placement_returns(ledger, at_starts, alpha=-1)

    def test_hand_credits_add_up_to_the_bonus(self):
        """At alpha nought a game's credits, U at the next boundary less U at
        the hand's own start, add up to its bonus less U at its first hand,
        for every player: nothing is credited twice and nothing missed."""
        batch = rounds()["mixed"](ledger=True)
        ledger = batch.ledger
        model = place_model.PlaceModel(ledger.members, hidden=16)
        at_starts = place_model.expected_at_starts(ledger, model)
        credits = place_model.hand_credits(ledger, at_starts)
        games = ledger.starts[:, scoreboard.GAME]
        summed = np.zeros((len(ledger.first), 4))
        np.add.at(summed, games, credits)
        bonus = placement_rewards(ledger.final_scores, place_model.VALUES)
        np.testing.assert_allclose(summed, bonus - at_starts[ledger.first], atol=1e-9)
        # Every decision is paid its own hand's worth.
        reward = place_model.PlacementReward(model, alpha=0.0)
        returns = reward.returns(ledger)
        rows = ledger.rows_of(ledger.games, ledger.hands)
        worth = place_model.hand_returns(ledger, at_starts)
        np.testing.assert_allclose(returns, worth[rows, ledger.players.astype(np.int64)], rtol=1e-6)
        last = ledger.hands == ledger.count[ledger.games] - 1
        np.testing.assert_allclose(returns[last], bonus[ledger.games[last], ledger.players[last]], rtol=1e-6)

    def test_settling_by_placement_replaces_the_returns_alone(self):
        batch = rounds()["alone"](ledger=True)
        before = {name: getattr(batch, name) for name in ("actions", "log_probs", "legal")}
        model = place_model.PlaceModel((), hidden=8)
        selfplay.settle_placement(batch, place_model.PlacementReward(model, alpha=0.05))
        self.assertEqual(batch.returns.dtype, torch.float32)
        self.assertEqual(len(batch.returns), batch.decisions)
        np.testing.assert_allclose(
            batch.returns.numpy(),
            place_model.placement_returns(batch.ledger, place_model.expected_at_starts(batch.ledger, model), 0.05))
        for name, value in before.items():
            self.assertIs(getattr(batch, name), value)
        plain = rounds()["alone"]()
        with self.assertRaisesRegex(ValueError, "no ledger"):
            selfplay.settle_placement(plain, place_model.PlacementReward(model))


if __name__ == "__main__":
    unittest.main()
