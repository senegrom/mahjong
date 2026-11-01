"""The last hand's rules in the learner's own self-play (`neural.selfplay_rules`,
`train_combined --last-hand-rules`).

Inside the learner's deciding (`mortal_learner.decide_in_mortal_space`), each
rule reads the move drawn: decline-last every win drawn, then the all-last
dama rule every riichi still played, at the tile its preview drew. A
declined ron passes, a declined self-draw plays the best other move of the
same answer, and a reach whose drawn tile keeps first throws it undeclared;
each such move is recorded alone, flagged, at a log-probability of nought,
and a reach left undeclared loses its tile record. Codes that contradict the
legal moves are refused, rules that never fire change nothing, and under
argmax each rule plays what the ruler's player plays. The policy's learning
leaves a move recorded alone out exactly: the weights, Adam's state, the
entropy, the clipped fraction and the drift are bit for bit those of the
same round with that move's advantage flipped. Self-play hands the rules to
the learner only when asked, needs the preview and a learner that decides
for itself, and reconciles its records with the rules' account at the
round's end; the trainer and the cloud leave the rules off unless asked,
check every round before learning it, and stop with a code of their own,
which the cloud answers for rather than retry. On the processor with the
real engine, a round of a learner that declares wherever it can fires each
rule, and one of a learner that declares only as a declined self-draw's
best other move leaves reaches that dama judges; at mixed tables the rules
play only the learner's seats; each flagged record is the decision a rule
played and nothing else, the table's own events witness every firing, a
round with the rules off differs only at tables a rule fired at, the
trainer's epochs learn nothing from the flagged rows, and the measurement
and the placement reward's refit run on as before.
"""

from __future__ import annotations

import collections
import contextlib
import copy
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

from neural import (
    all_last, cloud_requests, late_rules, mortal_learner, place_model, ppo_loop, scoreboard, selfplay,
    selfplay_rules, train_combined, zoo,
)
from neural.checkpoints import validate_checkpoint
from neural.observe import Planes
from neural.ppo_control import PolicyDrift
from neural.tests import test_mixed_tables
from neural.tests.test_baseline_from_play import TO_MORTAL, fake_round
from neural.tests.test_bfloat16_players import fusion
from neural.tests.test_late_rules import claim_row, moves, turn_row

DAMA, DECLINE = cloud_requests.LAST_HAND_RULES
BOTH = (DAMA, DECLINE)
KEEPS_FIRST, NO_YAKU, SHORT, DEAD, NOT_WAITING = (all_last.KEEPS_FIRST, all_last.NO_YAKU, all_last.SHORT,
                                                  all_last.DEAD, all_last.NOT_WAITING)
LAST, NOT_LAST = late_rules.LAST, late_rules.NOT_LAST
#: Big enough that softmax over a small network's logits is exactly one-hot:
#: every other move's probability underflows to nought in float32, so play
#: depends on nothing drawn at random.
PUSH = 1e4
#: The mjai names of the tiles by kind, as the engine writes them.
TILE_NAMES = [f"{number}{suit}" for suit in "mps" for number in range(1, 10)] + list("ESWNPFC")
NATIVE_ARENA = riichi_py.Arena


class Hands:
    """How many hands each game has finished, as `Arena.hands_done` answers,
    read by game: `hands`' count, or nought."""

    def __init__(self, hands: dict[int, int]) -> None:
        self.hands = hands

    def __getitem__(self, game: int) -> int:
        return self.hands.get(int(game), 0)


class RuleCodes:
    """An arena as the rules read it in self-play: each game's dama codes, a
    code a tile kind, each row's decline code by game and player by place,
    as the engine hands them over, and how many hands each game has
    finished. Every question is written down, and every row asked about
    must be a pair of plain integers, as the arena takes them."""

    def __init__(self, dama=None, decline=None, hands=None) -> None:
        self.dama = dama or {}
        self.decline = decline or {}
        self.hands = dict(hands or {})
        self.asked: list[tuple[str, list]] = []

    def all_last_dama(self, games):
        games = list(games)
        assert all(type(game) is int for game in games), games
        self.asked.append(("dama", games))
        codes = np.zeros((len(games), all_last.KINDS), dtype=np.uint8)
        for at, game in enumerate(games):
            for tile, code in self.dama.get(game, {}).items():
                codes[at, tile] = code
        return codes.tobytes()

    def all_last_decline(self, rows):
        rows = list(rows)
        for row in rows:
            assert type(row) is tuple and len(row) == 2 and all(type(part) is int for part in row), row
        self.asked.append(("decline", rows))
        return bytes(self.decline.get(row, 0) for row in rows)

    def hands_done(self):
        return Hands(self.hands)


class RuleViews:
    """Views of a table whose arena is `arena`: the follower serves empty
    planes for whoever is asked, as they stand or with a reach previewed,
    and writes down whatever it is told, which nothing here may need."""

    def __init__(self, arena) -> None:
        self.arena = arena
        self.observer = SimpleNamespace(follower=self)
        self.told: list[tuple] = []

    def tell(self, game, player, line):
        self.told.append((game, player, line))

    def sparse_and_masks(self, rows, players, fresh=False, after_reach=False):
        n = len(rows)
        return (Planes(np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.uint16),
                       np.zeros(0, dtype=np.float16)), np.ones((n, zoo.MORTAL_ACTIONS), dtype=bool))


class Scored:
    """A forward's answers set by hand, as `score(planes, mask)` gives them:
    the first question's rows `first`, then each reach's preview,
    `previews[row]` or nought, a move outside the mask at minus infinity;
    and as each row's value 100 plus its row for the first question, and
    200 plus its row for a preview. Every question's size is written down."""

    def __init__(self, first, previews=None) -> None:
        self.first = np.asarray(first, dtype=np.float32)
        self.previews = previews or {}
        self.sizes: list[int] = []

    def __call__(self, planes, mask):
        allowed = mask.numpy()
        n = len(self.first)
        self.sizes.append(len(allowed))
        ahead = np.flatnonzero(allowed[:n, zoo.MORTAL_RIICHI])
        assert len(allowed) == n + len(ahead), "one question, every reach that may be declared previewed"
        zero = np.zeros(zoo.MORTAL_ACTIONS, dtype=np.float32)
        logits = np.concatenate([self.first, *[self.previews.get(int(row), zero)[None] for row in ahead]])
        values = np.concatenate([100.0 + np.arange(n), 200.0 + ahead]).astype(np.float32)
        return torch.from_numpy(np.where(allowed, logits, -np.inf).astype(np.float32)), torch.from_numpy(values)


def decide(score, arena, games, players, legal, rules=None, seed=0, greedy=False, explore_share=0.0, starts=None):
    """`decide_in_mortal_space` on the processor with the reach previewed, its
    moves drawn from a generator seeded by `seed`, the views given the
    round's record of its hands `starts` when there is one; also the views
    asked."""
    views = RuleViews(arena)
    if starts is not None:
        views.starts = starts
    choice, records = mortal_learner.decide_in_mortal_space(
        score, views, np.asarray(games), np.asarray(players), np.atleast_2d(legal), greedy, "cpu", {},
        explore_share=explore_share, wanderer=np.random.default_rng(seed), preview_reach=True,
        generator=torch.Generator().manual_seed(seed), rules=rules)
    return choice, records, views


def reconciled(played, choice, records) -> None:
    """The round's end (`selfplay_rules.reconcile`) after one step, its
    reaches sent and tile records gathered counted as `selfplay.play` counts
    them, from the moves sent and the records' rows."""
    choice, slots = np.asarray(choice), np.asarray(records.slots)
    sent = int(np.count_nonzero((choice >= zoo.RIICHI_DISCARD) & (choice < zoo.TSUMO)))
    selfplay_rules.reconcile(played, int(records.ruled.sum()), sent, len(slots) - len(np.unique(slots)))


def one_hot(move: int) -> np.ndarray:
    row = np.zeros(zoo.MORTAL_ACTIONS, dtype=bool)
    row[move] = True
    return row


class DamaTests(unittest.TestCase):
    def test_it_fires_only_where_the_drawn_tile_keeps_first(self):
        """The preview rates the 5 and the 9 alike, and the draw decides:
        drawn, the 5, which keeps first, is thrown undeclared and recorded
        alone at a log-probability of nought, with no tile record and
        nothing told; the 9, which is short, leaves the reach and its tile
        record as they were."""
        legal = turn_row([3, 5, 9], [5, 9])
        drawn = {}
        for seed in range(64):
            played = selfplay_rules.Played(BOTH)
            arena = RuleCodes(dama={0: {3: NOT_WAITING, 5: KEEPS_FIRST, 9: SHORT}}, hands={0: 7})
            score = Scored([moves(reach=30.0, d3=1.0)], {0: moves(d5=0.0, d9=0.0)})
            choice, records, views = decide(score, arena, [0], [1], legal, played, seed)
            drawn.setdefault(int(choice[0]), (records, views, arena, played, score))
            if len(drawn) == 2:
                break
        self.assertEqual(set(drawn), {zoo.DISCARD + 5, zoo.RIICHI_DISCARD + 9})

        records, views, arena, played, score = drawn[zoo.DISCARD + 5]
        self.assertEqual(score.sizes, [2])
        self.assertEqual(arena.asked, [("dama", [0])])
        np.testing.assert_array_equal(records.masks, one_hot(5)[None])
        self.assertEqual(records.actions.tolist(), [5])
        self.assertEqual(records.log_probs.tolist(), [0.0])
        self.assertEqual(records.slots.tolist(), [0])
        self.assertEqual(records.ruled.tolist(), [True])
        self.assertEqual(records.forced.tolist(), [False])
        self.assertEqual(records.values.tolist(), [100.0])
        self.assertEqual(len(records.planes.indptr) - 1, 1)
        self.assertEqual(views.told, [])
        self.assertEqual(played.report()[DAMA], {"judged": 1, "fired": 1, "tables": 1, "hands": 1,
                                                 "kept": {"no_yaku": 0, "short": 0, "dead": 0},
                                                 "riichi_after_firing": 0, "calls": 1})
        self.assertEqual(played.events, [(0, 7, 1, DAMA, 5)])
        self.assertEqual((played.records, played.tile_records, played.reaches_sent, played.both), (1, 0, 0, 0))

        records, views, arena, played, _score = drawn[zoo.RIICHI_DISCARD + 9]
        self.assertEqual(records.actions.tolist(), [zoo.MORTAL_RIICHI, 9])
        self.assertEqual(records.slots.tolist(), [0, 0])
        self.assertEqual(records.ruled.tolist(), [False, False])
        np.testing.assert_array_equal(records.masks[0], zoo.translatable(legal)[0])
        np.testing.assert_array_equal(np.flatnonzero(records.masks[1]), [5, 9])
        self.assertAlmostEqual(float(records.log_probs[1]), float(np.log(0.5)), places=5)
        self.assertEqual(records.values.tolist(), [100.0, 200.0])
        self.assertEqual(views.told, [])
        self.assertEqual(played.report()[DAMA]["kept"], {"no_yaku": 0, "short": 1, "dead": 0})
        self.assertEqual((played.dama.fired, played.records, played.tile_records, played.reaches_sent), (0, 0, 1, 1))

    def test_codes_two_to_four_leave_the_reach_counted_by_reason(self):
        legal = np.stack([turn_row([3, 5], [5])] * 3)
        played = selfplay_rules.Played(BOTH)
        arena = RuleCodes(dama={0: {5: NO_YAKU}, 1: {5: SHORT}, 2: {5: DEAD, 3: NOT_WAITING}}, hands={2: 4})
        score = Scored([moves(reach=30.0, d3=1.0)] * 3)
        choice, records, _views = decide(score, arena, [0, 1, 2], [1, 2, 3], legal, played)
        self.assertEqual(choice.tolist(), [zoo.RIICHI_DISCARD + 5] * 3)
        self.assertEqual(records.actions.tolist(), [zoo.MORTAL_RIICHI] * 3 + [5] * 3)
        self.assertFalse(records.ruled.any())
        self.assertEqual(played.report()[DAMA], {"judged": 3, "fired": 0, "tables": 0, "hands": 0,
                                                 "kept": {"no_yaku": 1, "short": 1, "dead": 1},
                                                 "riichi_after_firing": 0, "calls": 1})

    def test_codes_that_contradict_the_legal_moves_are_refused(self):
        cases = (
            # The drawn tile coded nought on a row the arena judged.
            ("which no legal riichi has", turn_row([3, 5], [5]), {0: {3: KEEPS_FIRST}}),
            # The drawn tile coded as one whose riichi is not legal.
            ("which no legal riichi has", turn_row([3, 5], [5]), {0: {5: NOT_WAITING}}),
            # A tile that keeps first, whose discard is not open.
            ("keeps first, is not open", turn_row([3], [5]), {0: {5: KEEPS_FIRST}}),
        )
        for message, legal, codes in cases:
            with self.subTest(message=message, codes=codes), \
                    self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, message):
                decide(Scored([moves(reach=30.0, d3=1.0)]), RuleCodes(dama=codes), [0], [1], legal,
                       selfplay_rules.Played(BOTH))
        with self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, "says nothing of a win"):
            decide(Scored([moves(agari=30.0)]), RuleCodes(decline={(0, 1): 3}), [0], [1], claim_row(),
                   selfplay_rules.Played(BOTH))
        self.assertTrue(issubclass(selfplay_rules.RuleCheckFailed, RuntimeError))


class DeclineTests(unittest.TestCase):
    def test_a_ron_declined_in_a_claim_window_is_the_pass_recorded_alone(self):
        played = selfplay_rules.Played(BOTH)
        arena = RuleCodes(decline={(0, 2): LAST}, hands={0: 9})
        choice, records, views = decide(Scored([moves(agari=30.0, **{"pass": 1.0})]), arena, [0], [2],
                                        claim_row(zoo.PON), played)
        self.assertEqual(choice.tolist(), [zoo.PASS])
        np.testing.assert_array_equal(records.masks, one_hot(zoo.MORTAL_PASS)[None])
        self.assertEqual((records.actions.tolist(), records.log_probs.tolist(), records.ruled.tolist()),
                         ([zoo.MORTAL_PASS], [0.0], [True]))
        self.assertEqual(arena.asked, [("decline", [(0, 2)])])
        self.assertEqual(played.report()[DECLINE], {
            "judged": 1, "fired": 1, "in_window": 1, "at_turn": 0, "to_riichi": 0, "tables": 1, "hands": 1,
            "kept": {"not_last": 0}, "kept_after_firing": 0, "calls": 1})
        self.assertEqual(played.events, [(0, 9, 2, DECLINE, zoo.MORTAL_PASS)])
        self.assertEqual(views.told, [])

    def test_a_self_draw_declined_at_the_turn_plays_the_best_other_move_recorded_alone(self):
        played = selfplay_rules.Played(BOTH)
        arena = RuleCodes(decline={(0, 1): LAST})
        choice, records, _views = decide(Scored([moves(agari=30.0, d9=4.0, d3=1.0)]), arena, [0], [1],
                                         turn_row([3, 9], tsumo=True), played)
        self.assertEqual(choice.tolist(), [zoo.DISCARD + 9])
        np.testing.assert_array_equal(records.masks, one_hot(9)[None])
        self.assertEqual((records.actions.tolist(), records.log_probs.tolist(), records.ruled.tolist()),
                         ([9], [0.0], [True]))
        self.assertEqual((played.in_window, played.at_turn, played.to_riichi), (0, 1, 0))

    def test_a_declined_self_draw_whose_best_other_move_is_riichi(self):
        """Its tile is drawn from its preview, then judged by dama: thrown
        undeclared where it keeps first, one record of the discard alone and
        none of the tile; and where it is short, the reach recorded alone
        and the tile as the policy drew it, at its real log-probability."""
        legal = turn_row([3, 5, 9], [5, 9], tsumo=True)
        found = {}
        for seed in range(64):
            played = selfplay_rules.Played(BOTH)
            arena = RuleCodes(dama={0: {3: NOT_WAITING, 5: KEEPS_FIRST, 9: SHORT}}, decline={(0, 1): LAST},
                              hands={0: 2})
            score = Scored([moves(agari=30.0, reach=20.0, d9=4.0)], {0: moves(d5=0.0, d9=0.0)})
            choice, records, views = decide(score, arena, [0], [1], legal, played, seed)
            found.setdefault(int(choice[0]), (records, views, arena, played))
            if len(found) == 2:
                break
        self.assertEqual(set(found), {zoo.DISCARD + 5, zoo.RIICHI_DISCARD + 9})

        records, views, arena, played = found[zoo.DISCARD + 5]
        self.assertEqual(arena.asked, [("decline", [(0, 1)]), ("dama", [0])])
        np.testing.assert_array_equal(records.masks, one_hot(5)[None])
        self.assertEqual((records.actions.tolist(), records.log_probs.tolist(), records.ruled.tolist()),
                         ([5], [0.0], [True]))
        self.assertEqual((played.at_turn, played.to_riichi, played.both, played.dama.fired, played.decline.fired,
                          played.records), (1, 1, 1, 1, 1, 1))
        self.assertEqual(played.events, [(0, 2, 1, DECLINE, zoo.MORTAL_RIICHI), (0, 2, 1, DAMA, 5)])
        self.assertEqual(views.told, [])
        reconciled(played, [zoo.DISCARD + 5], records)

        records, views, arena, played = found[zoo.RIICHI_DISCARD + 9]
        self.assertEqual(records.actions.tolist(), [zoo.MORTAL_RIICHI, 9])
        np.testing.assert_array_equal(records.masks[0], one_hot(zoo.MORTAL_RIICHI))
        np.testing.assert_array_equal(np.flatnonzero(records.masks[1]), [5, 9])
        self.assertEqual(records.ruled.tolist(), [True, False])
        self.assertEqual(float(records.log_probs[0]), 0.0)
        self.assertAlmostEqual(float(records.log_probs[1]), float(np.log(0.5)), places=5)
        self.assertEqual((played.to_riichi, played.both, played.records, played.tile_records, played.reaches_sent),
                         (1, 0, 1, 1, 1))
        self.assertEqual(played.dama.kept["short"], 1)
        reconciled(played, [zoo.RIICHI_DISCARD + 9], records)


class DrawnMovesTests(unittest.TestCase):
    def test_only_the_moves_drawn_are_judged(self):
        """A win or a riichi the policy rated best but did not draw is not
        asked about; one it drew but did not rate best is."""
        legal = np.stack([claim_row(), claim_row(), turn_row([3, 5], [5]), turn_row([3, 5], [5])])
        first = [moves(agari=1.0, **{"pass": 0.999}), moves(agari=0.999, **{"pass": 1.0}),
                 moves(reach=1.0, d3=0.999), moves(reach=0.999, d3=1.0)]
        wanted = [zoo.PASS, zoo.RON, zoo.DISCARD + 3, zoo.DISCARD + 5]
        for seed in range(400):
            arena = RuleCodes(dama={2: {5: KEEPS_FIRST, 3: NOT_WAITING}, 3: {5: KEEPS_FIRST, 3: NOT_WAITING}},
                              decline={(0, 1): NOT_LAST, (1, 2): NOT_LAST})
            played = selfplay_rules.Played(BOTH)
            choice, _records, _views = decide(Scored(first), arena, [0, 1, 2, 3], [1, 2, 3, 0], legal, played, seed)
            if choice.tolist() == wanted:
                break
        self.assertEqual(choice.tolist(), wanted)
        self.assertEqual(arena.asked, [("decline", [(1, 2)]), ("dama", [3])])
        self.assertEqual((played.decline.judged, played.dama.riichi, played.dama.fired), (1, 1, 1))


class LastHandTests(unittest.TestCase):
    def test_with_the_hands_kept_only_rows_the_engine_can_answer_are_asked_about(self):
        """Given the round's record of its hands (`views.starts`), the engine
        is asked only about a non-dealer's decisions in South 4, and about a
        game whose hand is not known; without it, about every row drawn."""
        legal = np.stack([claim_row()] * 4 + [turn_row([3, 5], [5])] * 2)
        first = [moves(agari=30.0)] * 4 + [moves(reach=30.0)] * 2
        games, players = [0, 1, 2, 3, 4, 5], [1, 2, 3, 0, 1, 2]
        # Game 0 South 4, its dealer another; 1 East 4; 2 South 4 with the
        # player dealing; 3 not begun; 4 South 3; 5 South 4, another dealing.
        starts = SimpleNamespace(stage=np.array([7, 3, 7, -1, 6, 7], dtype=np.int8),
                                 dealer=np.array([0, 0, 3, -1, 0, 0], dtype=np.int8))
        for kept, decline, dama in ((starts, [(0, 1), (3, 0)], [5]),
                                    (None, [(0, 1), (1, 2), (2, 3), (3, 0)], [4, 5])):
            arena = RuleCodes()
            decide(Scored(first), arena, games, players, legal, selfplay_rules.Played(BOTH), starts=kept)
            with self.subTest(kept=kept is not None):
                self.assertEqual(arena.asked, [("decline", decline), ("dama", dama)])
        # Before any game has begun South 4, nothing at all.
        arena = RuleCodes()
        early = SimpleNamespace(stage=np.full(6, 6, dtype=np.int8), dealer=np.zeros(6, dtype=np.int8),
                                reached_last=False)
        _choice, records, _views = decide(Scored(first), arena, games, players, legal,
                                          selfplay_rules.Played(BOTH), starts=early)
        self.assertEqual(arena.asked, [])
        self.assertFalse(records.ruled.any())

    def test_the_hands_begun_keep_each_games_stage_and_dealer(self):
        """As `Ledger.stages` numbers them, South 4 the last."""
        begun = scoreboard.HandStarts(3)
        self.assertEqual((begun.stage.tolist(), begun.dealer.tolist()), ([-1] * 3, [-1] * 3))
        begun.begin(2, {"bakaze": "E", "kyoku": 2, "honba": 0, "kyotaku": 0, "oya": 1, "scores": [25000] * 4})
        views = SimpleNamespace(starts=begun)
        self.assertFalse(selfplay_rules.may_ask(views))
        begun.begin(1, {"bakaze": "S", "kyoku": 4, "honba": 1, "kyotaku": 0, "oya": 2, "scores": [25000] * 4})
        self.assertTrue(selfplay_rules.may_ask(views))
        self.assertTrue(selfplay_rules.may_ask(SimpleNamespace()))
        self.assertEqual((begun.stage.tolist(), begun.dealer.tolist()), ([-1, 7, 1], [-1, 2, 1]))
        self.assertEqual(selfplay_rules.LAST_STAGE, 7)
        np.testing.assert_array_equal(
            selfplay_rules.last_hand(views, np.array([0, 1, 1, 2]), np.array([0, 1, 2, 0])), [True, True, False, False])


def random_rows(rng, n: int) -> np.ndarray:
    """`n` rows of our engine's moves, each a claim window with a ron or a
    turn, with discards, often reaches among them, often a self-draw and
    now and then a quad."""
    legal = np.zeros((n, riichi_py.ACTIONS), dtype=bool)
    for row in range(n):
        if rng.random() < 0.35:
            legal[row, [zoo.RON, zoo.PASS]] = True
            legal[row, zoo.PON] = rng.random() < 0.5
            continue
        discards = rng.choice(34, size=int(rng.integers(2, 8)), replace=False)
        legal[row, discards] = True
        if rng.random() < 0.7:
            reaches = rng.choice(discards, size=int(rng.integers(1, len(discards) + 1)), replace=False)
            legal[row, zoo.RIICHI_DISCARD + reaches] = True
        legal[row, zoo.TSUMO] = rng.random() < 0.5
        legal[row, zoo.CONCEALED_KAN] = rng.random() < 0.2
    return legal


def random_question(rng, legal: np.ndarray):
    """Values for each row and each possible preview, a win and a riichi
    often best, and codes for every reach's tiles and every win."""
    n = len(legal)
    first = rng.normal(size=(n, zoo.MORTAL_ACTIONS)).astype(np.float32)
    first[:, zoo.MORTAL_AGARI] += 2.0 * (rng.random(n) < 0.6)
    first[:, zoo.MORTAL_RIICHI] += 2.0 * (rng.random(n) < 0.6)
    previews = {row: rng.normal(size=zoo.MORTAL_ACTIONS).astype(np.float32) for row in range(n)}
    players = rng.integers(0, 4, size=n)
    dama = {row: {int(tile): int(rng.choice([KEEPS_FIRST, NO_YAKU, SHORT, DEAD]))
                  for tile in np.flatnonzero(legal[row, zoo.RIICHI_DISCARD:zoo.TSUMO])} for row in range(n)}
    decline = {(row, int(players[row])): int(rng.choice([0, LAST, NOT_LAST])) for row in range(n)}
    return first, previews, players, dama, decline


class ParityTests(unittest.TestCase):
    def test_under_argmax_each_rule_plays_what_the_rulers_player_plays(self):
        """Greedy, on the same codes and answers, each rule alone: the moves
        `decide_in_mortal_space` sends are those `zoo.choose_in_mortal_space`
        plays through the ruler's wrapper, and the firings are the same."""
        rng = np.random.default_rng(20261101)
        fired = {DAMA: 0, DECLINE: 0}
        for trial in range(300):
            n = int(rng.integers(1, 7))
            legal = random_rows(rng, n)
            first, previews, players, dama, decline = random_question(rng, legal)
            games = np.arange(n)

            def answers(_views, who, fresh, allowed, ahead=()):
                assert not fresh
                found = [first[game] for game, _player in who] + [previews[game] for game, _player in ahead]
                answer = np.stack(found)
                return np.where(allowed, answer, -np.inf).astype(np.float32), np.zeros(answer.shape, dtype=bool)

            for rule in BOTH:
                arena = RuleCodes(dama=dama, decline=decline)
                views = RuleViews(arena)
                if rule == DAMA:
                    account = all_last.Account()
                    ask = all_last.ruled(answers, account)
                else:
                    account = late_rules.Account(DECLINE)
                    ask = late_rules.decline_last(answers, account)
                want = zoo.choose_in_mortal_space(
                    lambda who, fresh, allowed, ahead=(): ask(views, who, fresh, allowed, ahead),
                    views, games, players, legal, preview_reach=True)
                played = selfplay_rules.Played((rule,))
                got, records, ours = decide(Scored(first, previews), RuleCodes(dama=dama, decline=decline), games,
                                            players, legal, played, greedy=True)
                with self.subTest(trial=trial, rule=rule):
                    np.testing.assert_array_equal(got, want)
                    self.assertEqual(played.dama.fired if rule == DAMA else played.decline.fired, account.fired)
                    self.assertEqual(views.told, [])
                    self.assertEqual(ours.told, [])
                    reconciled(played, got, records)
                fired[rule] += account.fired
        self.assertGreater(fired[DAMA], 20)
        self.assertGreater(fired[DECLINE], 20)


class NeverFiringTests(unittest.TestCase):
    def test_rules_that_never_fire_change_nothing(self):
        """Asked and never firing, because the arena judges nothing or keeps
        every move it judges: the moves and every record are bit for bit
        those of deciding without the rules, exploration included, but for
        the flags, all false, which without the rules are None."""
        rng = np.random.default_rng(31)
        for trial in range(120):
            n = int(rng.integers(1, 7))
            legal = random_rows(rng, n)
            first, previews, players, dama, _decline = random_question(rng, legal)
            kept = {game: {tile: NO_YAKU + (tile % 3) for tile in codes} for game, codes in dama.items()}
            stayed = {(row, int(players[row])): NOT_LAST for row in range(n)}
            for name, arena in (("judged nothing", RuleCodes()),
                                ("kept every move", RuleCodes(dama=kept, decline=stayed))):
                seed = 1000 + trial
                want, plain, _views = decide(Scored(first, previews), RuleCodes(), np.arange(n), players, legal,
                                             None, seed, explore_share=0.1)
                played = selfplay_rules.Played(BOTH)
                got, ruled, views = decide(Scored(first, previews), arena, np.arange(n), players, legal, played,
                                           seed, explore_share=0.1)
                with self.subTest(trial=trial, arena=name):
                    np.testing.assert_array_equal(got, want)
                    for field in ("masks", "actions", "log_probs", "slots", "forced", "values"):
                        a, b = getattr(ruled, field), getattr(plain, field)
                        self.assertEqual(a.dtype, b.dtype, field)
                        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8), field)
                    for part in ("indptr", "indices", "values"):
                        np.testing.assert_array_equal(getattr(ruled.planes, part), getattr(plain.planes, part))
                    self.assertIsNone(plain.ruled)
                    self.assertEqual(ruled.ruled.dtype, bool)
                    self.assertEqual(len(ruled.ruled), len(ruled.actions))
                    self.assertFalse(ruled.ruled.any())
                    self.assertEqual((played.dama.fired, played.decline.fired, played.records), (0, 0, 0))
                    self.assertEqual(views.told, [])


class RecordsTests(unittest.TestCase):
    def test_the_records_line_up(self):
        """Slots, values and flags, one a record: a reach thrown undeclared
        takes its tile record's value with it, a reach kept keeps its own,
        and a row nothing translates is never asked about or recorded."""
        legal = np.stack([turn_row([3, 5], [5]), np.zeros(riichi_py.ACTIONS, dtype=bool), turn_row([3, 9], [9]),
                          claim_row()])
        arena = RuleCodes(dama={0: {5: KEEPS_FIRST, 3: NOT_WAITING}, 2: {9: SHORT, 3: NOT_WAITING}},
                          decline={(3, 0): LAST})
        played = selfplay_rules.Played(BOTH)
        first = [moves(reach=30.0), moves(), moves(reach=30.0), moves(agari=30.0)]
        choice, records, _views = decide(Scored(first), arena, [0, 1, 2, 3], [1, 2, 3, 0], legal, played)
        self.assertEqual(choice.tolist(), [zoo.DISCARD + 5, 0, zoo.RIICHI_DISCARD + 9, zoo.PASS])
        self.assertEqual(arena.asked, [("decline", [(3, 0)]), ("dama", [0, 2])])
        self.assertEqual(records.slots.tolist(), [0, 2, 3, 2])
        self.assertEqual(records.actions.tolist(), [5, zoo.MORTAL_RIICHI, zoo.MORTAL_PASS, 9])
        # Row 2's preview is the second asked ahead: row 0's went with it.
        self.assertEqual(records.values.tolist(), [100.0, 102.0, 103.0, 202.0])
        self.assertEqual(records.ruled.tolist(), [True, False, True, False])
        self.assertEqual(records.forced.tolist(), [False] * 4)
        self.assertEqual(len(records.planes.indptr) - 1, 4)
        self.assertEqual((played.records, played.tile_records, played.reaches_sent), (2, 1, 1))
        # Counted from the moves sent and the records' rows, as the round
        # counts them, they agree with the account.
        reconciled(played, choice, records)


class CheckTests(unittest.TestCase):
    """The checks that keep a rule's move from the policy, given records and
    moves that do not agree, as a slip in the code would leave them."""

    def setUp(self):
        self.legal = np.stack([turn_row([3, 5], [5]), turn_row([3, 9], [9]), claim_row()])
        self.decidable = np.ones(3, dtype=bool)

    def check(self, choice, picked, ruled, tiles, thrown=None):
        """`check_step` on these rows, the tile records of rows `tiles`
        holding the tiles `thrown`, by default those their reaches throw."""
        choice, tiles = np.asarray(choice), np.asarray(tiles, dtype=np.int64)
        if thrown is None:
            thrown = np.clip(choice[tiles] - zoo.RIICHI_DISCARD, 0, 33)
        selfplay_rules.check_step(choice, np.asarray(picked), self.legal, self.decidable,
                                  np.asarray(ruled, dtype=bool), tiles, np.asarray(thrown, dtype=np.int64))

    def test_records_that_agree_with_the_moves_pass(self):
        self.check([5, zoo.RIICHI_DISCARD + 9, zoo.PASS], [5, zoo.MORTAL_RIICHI, zoo.MORTAL_PASS],
                   [True, False, True], [1])
        # A reach left by a declined self-draw, flagged, with its tile record.
        self.check([3, zoo.RIICHI_DISCARD + 9, zoo.RON], [3, zoo.MORTAL_RIICHI, zoo.MORTAL_AGARI],
                   [False, True, False], [1])

    def test_a_tile_record_of_a_reach_left_undeclared_is_refused(self):
        with self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, "tile record and the table disagree"):
            self.check([5, zoo.RIICHI_DISCARD + 9, zoo.PASS], [5, zoo.MORTAL_RIICHI, zoo.MORTAL_PASS],
                       [True, False, True], [0, 1])

    def test_a_reach_without_its_tile_record_is_refused(self):
        with self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, "tile record and the table disagree"):
            self.check([5, zoo.RIICHI_DISCARD + 9, zoo.PASS], [5, zoo.MORTAL_RIICHI, zoo.MORTAL_PASS],
                       [True, True, True], [])

    def test_a_tile_record_of_another_tile_than_the_reach_throws_is_refused(self):
        """The reach's row has its tile record, but the record holds a tile
        the table is not sent, as filtering the records' tiles and their
        rows apart would leave them; a reach a declined self-draw left too."""
        for picked, ruled in (([5, zoo.MORTAL_RIICHI, zoo.MORTAL_PASS], [True, False, True]),
                              ([3, zoo.MORTAL_RIICHI, zoo.MORTAL_AGARI], [False, True, False])):
            with self.subTest(ruled=ruled), self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, "throw tiles"):
                self.check([picked[0], zoo.RIICHI_DISCARD + 9, zoo.PASS if ruled[2] else zoo.RON], picked, ruled,
                           [1], thrown=[3])
        with self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, "throw tiles"):
            self.check([5, zoo.RIICHI_DISCARD + 9, zoo.PASS], [5, zoo.MORTAL_RIICHI, zoo.MORTAL_PASS],
                       [True, False, True], [1], thrown=[9, 9])

    def test_a_rules_move_that_is_not_what_was_sent_is_refused(self):
        with self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, "not their recorded moves"):
            self.check([3, zoo.RIICHI_DISCARD + 9, zoo.PASS], [5, zoo.MORTAL_RIICHI, zoo.MORTAL_PASS],
                       [True, False, True], [1])
        with self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, "not their recorded moves"):
            self.check([5, zoo.RIICHI_DISCARD + 9, zoo.RON], [5, zoo.MORTAL_RIICHI, zoo.MORTAL_PASS],
                       [True, False, True], [1])

    def test_a_round_whose_account_does_not_add_up_is_refused(self):
        played = selfplay_rules.Played(BOTH)
        played.dama.fired, played.decline.fired, played.in_window, played.records = 2, 1, 1, 3
        played.tile_records = played.reaches_sent = 4
        selfplay_rules.reconcile(played, 3, 4, 4)
        for change, flagged, counted, message in (
            (None, 2, (4, 4), "2 decisions were recorded as a rule's, the account counts 3"),
            (("tile_records", 5), 3, (4, 4), "4 reaches were sent to the table and 4 tile records gathered, and "
                                             "the account counts 4 reaches and 5 tile records"),
            # The round's own counts against each other, and against the
            # account's, which agree with each other.
            (None, 3, (5, 4), "5 reaches were sent to the table and 4 tile records gathered"),
            (None, 3, (5, 5), "5 reaches were sent to the table and 5 tile records gathered, and the account "
                              "counts 4 reaches"),
            (("in_window", 0), 3, (4, 4), "decline fired 1 times, 0 in a claim window"),
            (("both", 1), 3, (4, 4), "1 reaches left undeclared after a declined self-draw, 0 declined self-draws"),
        ):
            changed = copy.deepcopy(played)
            if change is not None:
                setattr(changed, *change)
            with self.subTest(message=message), self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, message):
                selfplay_rules.reconcile(changed, flagged, *counted)


class LearningTests(unittest.TestCase):
    def test_a_move_recorded_alone_teaches_the_policy_nothing(self):
        """The trainer's epochs (test_step_guard's harness) over a round some
        of whose rows hold one move alone at a log-probability of nought, as
        a rule's moves are recorded: learned with those rows' advantages at
        +5 and at -5, the weights, Adam's state, torch's generator, the
        entropy, clipped, kl and every other sum but the policy's, and the
        drift's figures, are bit for bit the same."""
        from neural.tests.test_step_guard import differences, learned, lesson, orders

        torch.set_num_threads(1)
        base = lesson()
        first, _second = orders()
        alone = first[[1, 6, 13, 22, 30]]
        legal = base.rollout.legal.clone()
        legal[alone] = False
        legal[alone, base.rollout.actions[alone]] = True
        old = base.rollout.old_log_probs.clone()
        old[alone] = 0.0
        begun = base.begun.clone()
        begun[alone] = 0.0
        rollout = SimpleNamespace(**{**vars(base.rollout), "legal": legal, "old_log_probs": old})
        runs = []
        for sign in (5.0, -5.0):
            advantages = base.advantages.clone()
            advantages[alone] = sign
            runs.append(learned(SimpleNamespace(**{**vars(base), "rollout": ppo_loop.Round(**vars(rollout)),
                                                   "advantages": advantages, "begun": begun}), late=True))
        self.assertGreater(runs[0]["updates"], 0)
        self.assertEqual(differences(runs[0], runs[1]), [".sums.policy"])


class PlayTests(unittest.TestCase):
    def learner(self, told: list, signature: str = "options"):
        from neural.tests.test_selfplay_contract import FirstLegalDecider

        class Learner(test_mixed_tables.Seated):
            timing: dict = {}

            if signature == "options":
                def decide(self, views, rows, players, legal, greedy=False, **options):
                    told.append(options.get("rules", "not handed"))
                    return FirstLegalDecider().decide(views, rows, players, legal, greedy)
            else:
                def decide(self, views, rows, players, legal, greedy=False, explore_share=0.0, wanderer=None,
                           preview_reach=False):
                    told.append("not handed")
                    return FirstLegalDecider().decide(views, rows, players, legal, greedy)

        return Learner()

    def test_play_hands_the_learner_the_rules_only_when_asked(self):
        for rules in ((), BOTH, (DECLINE,), (DECLINE, DAMA)):
            told: list = []
            batch = selfplay.play(self.learner(told), games=1, seed=3, device="cpu", preview_reach=True,
                                  last_hand_rules=rules)
            with self.subTest(rules=rules):
                self.assertTrue(told)
                if not rules:
                    self.assertEqual(set(told), {"not handed"})
                    self.assertIsNone(batch.ruled)
                    self.assertIsNone(batch.rules)
                    continue
                # One account for the round, handed with every question.
                self.assertTrue(all(account is told[0] for account in told))
                self.assertEqual(told[0].rules, selfplay_rules.canonical(rules))
                self.assertEqual(batch.ruled.dtype, torch.bool)
                self.assertEqual(len(batch.ruled), batch.decisions)
                self.assertEqual(batch.rules, told[0].report())
                self.assertEqual(batch.rules["rules"], [name for name in BOTH if name in rules])
        # A learner that knows nothing of the rules still plays without them.
        told = []
        selfplay.play(self.learner(told, "old"), games=1, seed=3, device="cpu", preview_reach=True)
        self.assertEqual(set(told), {"not handed"})

    def test_at_mixed_tables_the_seated_others_play_plain(self):
        """The learner is handed the round's account with every question and
        the seated others are asked as ever, with the four arguments a seated
        other takes and nothing of the rules (this one would fail if handed
        more), the Club bot's places played by the arena."""
        told: list = []
        other = test_mixed_tables.Counting()
        batch = selfplay.play(self.learner(told), games=8, seed=7, device="cpu", preview_reach=True,
                              last_hand_rules=BOTH, opponents=[other, zoo.ClubPlayer()], seat_share=0.5,
                              ledger=True)
        self.assertGreater(other.asked, 0)
        self.assertTrue((batch.ledger.owners >= 0).any())
        self.assertTrue(told and all(isinstance(account, selfplay_rules.Played) and account is told[0]
                                     for account in told))
        self.assertEqual(len(batch.ruled), batch.decisions)
        self.assertEqual(batch.rules, told[0].report())

    def test_play_refuses_the_rules_without_the_preview_or_a_learner_of_its_own_moves(self):
        with self.assertRaisesRegex(ValueError, "preview_reach"):
            selfplay.play(self.learner([]), games=1, seed=3, device="cpu", last_hand_rules=BOTH)
        with self.assertRaisesRegex(ValueError, "decides for itself"):
            selfplay.play(test_mixed_tables.Seated(), games=1, seed=3, device="cpu", preview_reach=True,
                          last_hand_rules=BOTH)
        with self.assertRaisesRegex(ValueError, "no rule of the last hand"):
            selfplay.play(self.learner([]), games=1, seed=3, device="cpu", preview_reach=True,
                          last_hand_rules=("wide-dama",))
        with self.assertRaisesRegex(ValueError, "preview_reach"):
            mortal_learner.decide_in_mortal_space(
                Scored([moves(d3=1.0)]), RuleViews(RuleCodes()), np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int64), turn_row([3])[None], device="cpu", rules=selfplay_rules.Played(BOTH))

    def test_a_round_whose_flags_do_not_add_up_is_refused(self):
        """A learner whose records flag a decision the rules' account does
        not know: the round is refused at its end."""
        from neural.tests.test_selfplay_contract import FirstLegalDecider

        class Flagging(test_mixed_tables.Seated):
            timing: dict = {}

            def decide(self, views, rows, players, legal, greedy=False, **options):
                choice, records = FirstLegalDecider().decide(views, rows, players, legal, greedy)
                records.ruled = np.ones(len(records.actions), dtype=bool)
                return choice, records

        with self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, "does not add up"):
            selfplay.play(Flagging(), games=1, seed=3, device="cpu", preview_reach=True, last_hand_rules=BOTH)


def ruled_round(flagged=(3,), begun_rows=None, log_prob=0.0, legal_moves=1, rules=BOTH):
    """`fake_round`'s eight decisions, those of `flagged` a rule's: one move
    open, the action, at `log_prob`; and its account as the round says it."""
    batch = fake_round()
    ruled = torch.zeros(batch.decisions, dtype=torch.bool)
    for row in flagged:
        ruled[row] = True
        batch.legal[row] = False
        batch.legal[row, : legal_moves] = True
        batch.legal[row, batch.actions[row]] = True
        batch.log_probs[row] = log_prob
    played = selfplay_rules.Played(rules)
    played.decline.fired = played.decline.judged = played.in_window = played.records = len(flagged)
    batch.ruled, batch.rules = ruled, played.report()
    return batch


class CheckRuledTests(unittest.TestCase):
    def test_flagged_rows_read_as_certain_pass_and_the_largest_distance_is_answered(self):
        batch = ruled_round(flagged=(1, 3))
        rollout = ppo_loop.on_device(batch, "cpu", card=False)
        self.assertEqual(train_combined.check_ruled(batch, rollout, None), 0.0)
        begun = torch.full((batch.decisions,), -2.0)
        begun[[1, 3]] = torch.tensor([0.0, -1e-7])
        self.assertEqual(train_combined.check_ruled(batch, rollout, begun), float(torch.tensor(1e-7)))
        self.assertIsNone(train_combined.check_ruled(fake_round(), rollout, None))
        unflagged = ruled_round(flagged=())
        self.assertEqual(train_combined.check_ruled(unflagged, ppo_loop.on_device(unflagged, "cpu", card=False),
                                                    None), 0.0)

    def test_a_flagged_row_the_policy_could_learn_from_is_refused(self):
        for name, batch, begun in (
            ("other moves", ruled_round(legal_moves=2), None),
            ("other than nought", ruled_round(log_prob=-1e-3), None),
            ("from certain", ruled_round(), torch.full((8,), 2e-6)),
            ("from certain", ruled_round(), torch.full((8,), float("nan"))),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, name):
                train_combined.check_ruled(batch, ppo_loop.on_device(batch, "cpu", card=False), begun)

    def test_the_pass_reads_the_rules_moves_whether_the_round_was_played_ahead_or_not(self):
        """The trainer's baseline pass (`train_combined.value_round`) reads
        the probability of every move made in a round played by the rules,
        played ahead or not, so `ruled_begun_max` is measured on every round
        that gets the pass: here a forward that ignores the mask, which a
        flagged row read so would be learned from, is caught either way. A
        round valued from play gets no pass and reads nothing; one played
        without the rules reads nothing unless played ahead, as before."""
        def passing(planes, mask):
            rows = len(mask)
            return torch.zeros(rows, zoo.MORTAL_ACTIONS), torch.zeros(rows), torch.zeros(rows, 4)

        switches = SimpleNamespace(baseline_from_play=False, check_baseline=False)
        batch = ruled_round(flagged=(1, 3))
        rollout = ppo_loop.on_device(batch, "cpu", card=False)
        for lagged in (False, True):
            with self.subTest(lagged=lagged):
                _guess, _seconds, _said, read, _kept = train_combined.value_round(
                    passing, rollout, batch, 4, switches, torch.float32, False, lagged=lagged)
                self.assertTrue(torch.allclose(read, torch.full((8,), -float(np.log(zoo.MORTAL_ACTIONS)))))
                with self.assertRaisesRegex(selfplay_rules.RuleCheckFailed, "from certain"):
                    train_combined.check_ruled(batch, rollout, read)
        valued = ruled_round(flagged=(1, 3))
        valued.values = torch.zeros(valued.decisions)
        _guess, _seconds, said, read, _kept = train_combined.value_round(
            passing, ppo_loop.on_device(valued, "cpu", card=False), valued, 4,
            SimpleNamespace(baseline_from_play=True, check_baseline=False), torch.float32, False, lagged=False)
        self.assertIs(said["baseline_from_play"], True)
        self.assertIsNone(read)
        plain = fake_round()
        for lagged in (False, True):
            _guess, _seconds, _said, read, _kept = train_combined.value_round(
                passing, ppo_loop.on_device(plain, "cpu", card=False), plain, 4, switches, torch.float32, False,
                lagged=lagged)
            self.assertEqual(read is None, not lagged)


def trainer_run(folder: Path, flags: list[str], round_of, argv_extra=()):
    """train_combined on the processor for one generation from a small joined
    player, each round `round_of(kwargs)`: the options each round was
    played with, the exit code (None when it ended cleanly) and what it
    printed."""
    origin, out = folder / "source.pt", folder / "run"
    torch.save(fusion(8, 1).state(), origin)
    argv = ["trainer", "--resume", str(origin), "--out", str(out), "--rounds", "1", "--batch", "4", "--epochs",
            "1", "--games", "1", "--measure-games", "1", "--fixed", "none", *flags, *argv_extra]
    asked: list[dict] = []

    def play(*_args, **kwargs):
        asked.append(kwargs)
        return round_of(kwargs)

    printed = io.StringIO()
    code = None
    torch.set_num_threads(1)
    with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False), \
            patch.object(train_combined.selfplay, "play", side_effect=play), \
            patch.object(train_combined.selfplay, "measure",
                         return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
            contextlib.redirect_stdout(printed):
        try:
            train_combined.main()
        except SystemExit as exit:
            code = exit.code
    return asked, code, printed.getvalue(), out


class TrainerTests(unittest.TestCase):
    def test_the_trainer_leaves_the_rules_off_unless_asked_and_refuses_them_without_the_preview(self):
        with patch.object(sys, "argv", ["trainer"]):
            self.assertEqual(train_combined.parse_args().last_hand_rules, [])
        with patch.object(sys, "argv", ["trainer", "--preview-reach", "--last-hand-rules", DECLINE, DAMA]):
            self.assertEqual(train_combined.parse_args().last_hand_rules, list(BOTH))
        for argv in (["--last-hand-rules", DAMA, DECLINE], ["--preview-reach", "--last-hand-rules", "wide-dama"],
                     ["--preview-reach", "--last-hand-rules", DAMA, DAMA]):
            with self.subTest(argv=argv), patch.object(sys, "argv", ["trainer", *argv]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train_combined.parse_args()

    def test_the_trainer_plays_by_the_rules_it_was_told_and_says_so(self):
        """Its rounds are played by the rules, the record carries each
        round's account with the largest distance the pass read, and the
        checkpoint names the rules, and still loads where every series
        loads its players (`zoo.load_player`); without the rules, none of
        it."""
        for flags in ([], ["--preview-reach", "--last-hand-rules", DAMA, DECLINE]):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as folder:
                ruled = bool(flags)
                asked, code, _printed, out = trainer_run(
                    Path(folder), flags, lambda _kwargs: ruled_round(flagged=(2, 5)) if ruled else fake_round())
                self.assertIsNone(code)
                self.assertEqual([kwargs.get("last_hand_rules") for kwargs in asked], [BOTH if ruled else None])
                record = json.loads((out / "log.jsonl").read_text().splitlines()[0])
                saved = torch.load(out / "latest.pt", map_location="cpu", weights_only=True)
                if not ruled:
                    self.assertNotIn("last_hand_rules", record)
                    self.assertNotIn("last_hand_rules", saved["training_controls"])
                    continue
                report = ruled_round(flagged=(2, 5)).rules
                self.assertEqual(record["last_hand_rules"], selfplay_rules.recorded(report, 0.0))
                self.assertEqual(list(record["last_hand_rules"]),
                                 ["rules", DAMA, DECLINE, "ruled_records", "both", "tile_records", "reaches_sent",
                                  "ruled_begun_max", "seconds"])
                self.assertEqual(saved["training_controls"]["last_hand_rules"], list(BOTH))
                # Loaded as a series and the cloud's copies load it.
                self.assertEqual(validate_checkpoint(out / "latest.pt"), 1)
                player = zoo.load_player(out / "latest.pt", "cpu")
                self.assertIsInstance(player, train_combined.combined.Combined)

    def test_a_round_played_without_the_rules_it_was_told_is_never_learned(self):
        for name, round_of in (
            ("without the last hand's rules", lambda _kwargs: fake_round()),
            ("played by the last hand's rules", lambda _kwargs: ruled_round(rules=(DAMA,))),
            ("a rule of the last hand made", lambda _kwargs: ruled_round(log_prob=-0.5)),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                _asked, code, printed, out = trainer_run(
                    Path(folder), ["--preview-reach", "--last-hand-rules", DAMA, DECLINE], round_of)
                self.assertEqual(code, cloud_requests.RULE_CHECK_EXIT)
                self.assertIn(name, (out / cloud_requests.RULE_CHECK_FILE).read_text(encoding="utf-8"))
                self.assertIn("stopped: rule check failed", printed)
                self.assertFalse((out / "latest.pt").exists())
                self.assertFalse((out / "log.jsonl").exists() and (out / "log.jsonl").read_text().strip())

    def test_a_check_that_fails_on_the_round_played_ahead_stops_the_trainer_as_itself(self):
        """With --play-ahead the round is played on the worker beside the
        learning: the check that fails there reaches the trainer as itself,
        after the round before it was learned and saved, and stops it
        without learning or saving its own."""
        failure = selfplay_rules.RuleCheckFailed("the round's account of the last hand's rules does not add up")
        seen = []
        real = train_combined.stop_on_rule_check

        def stop(out, raised):
            seen.append(raised)
            real(out, raised)

        def round_of(kwargs):
            if kwargs.get("own_draws") and len(rounds) == 1:
                raise failure
            rounds.append(kwargs)
            return ruled_round()

        rounds: list = []
        with tempfile.TemporaryDirectory() as folder, patch.object(train_combined, "stop_on_rule_check", stop):
            asked, code, _printed, out = trainer_run(
                Path(folder), ["--preview-reach", "--last-hand-rules", DAMA, DECLINE, "--play-ahead"], round_of,
                argv_extra=["--rounds", "2"])
            self.assertEqual(code, cloud_requests.RULE_CHECK_EXIT)
            self.assertEqual(len(asked), 2)
            self.assertIs(seen[0], failure)
            records = (out / "log.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["generation"] for line in records], [0])
            self.assertEqual(torch.load(out / "latest.pt", map_location="cpu", weights_only=True)["generation"], 1)
            self.assertEqual((out / cloud_requests.RULE_CHECK_FILE).read_text(encoding="utf-8").strip(),
                             str(failure))


class CloudTests(unittest.TestCase):
    def test_rule_arguments_ask_for_nothing_new_unless_told_and_refuse_without_the_preview(self):
        self.assertEqual(cloud_requests.rule_arguments(None, False), [])
        self.assertEqual(cloud_requests.rule_arguments([], False), [])
        self.assertEqual(cloud_requests.rule_arguments([DECLINE, DAMA], True), ["--last-hand-rules", *BOTH])
        self.assertEqual(cloud_requests.rule_arguments((DECLINE,), True), ["--last-hand-rules", DECLINE])
        for rules, preview, message in (([DAMA], False, "preview_reach"), (["wide-dama"], True, "no rule"),
                                        ([DAMA, DAMA], True, "twice"), (DAMA, True, "a list of names"),
                                        ([DAMA, 3], True, "a list of names")):
            with self.subTest(rules=rules, preview=preview), self.assertRaisesRegex(ValueError, message):
                cloud_requests.rule_arguments(rules, preview)

    def test_a_trainer_stopped_by_a_rule_check_ends_the_call_and_leaves_its_marker(self):
        """Exit `RULE_CHECK_EXIT`: the call answers rather than fail for
        Modal to retry the same round, writes rule-check-failed.json beside
        the run's checkpoints, publishes nothing new, and any other exit
        short of the target still fails the call as it did."""
        import subprocess
        from functools import partial

        from neural import cloud_runs
        from neural.checkpoints import atomic_save
        from neural.tests.test_cloud_isolation import controller

        app = controller()
        for code in (cloud_requests.RULE_CHECK_EXIT, 1):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                volume = root / "volume"
                atomic_save({"generation": 2610}, volume / "run/latest.pt")
                (volume / "run" / "generation.txt").write_text("2610")
                commits = []

                def popen(command, **_kwargs):
                    where = Path(command[command.index("--out") + 1])
                    message = "the round's account of the last hand's rules does not add up: " + "x" * 400
                    (where / cloud_requests.RULE_CHECK_FILE).write_text(message + "\n", encoding="utf-8")
                    return SimpleNamespace(stdout=io.StringIO("stopped: rule check failed\n"), wait=lambda: code)

                with patch.object(app, "VOLUME", volume), \
                        patch.object(app, "workspace", partial(cloud_runs.workspace, root=root / "scratch")), \
                        patch.object(app, "_environment", return_value={}), patch.object(app, "_save_cache"), \
                        patch.object(app, "volume", SimpleNamespace(commit=lambda: commits.append(1),
                                                                    reload=lambda: None)), \
                        patch.object(app.subprocess, "Popen", side_effect=popen), \
                        contextlib.redirect_stdout(io.StringIO()):
                    call = partial(app.train_combined, run="run", generations=3, preview_reach=True,
                                   last_hand_rules=list(BOTH), compile=False)
                    if code == 1:
                        with self.assertRaises(subprocess.CalledProcessError):
                            call()
                        self.assertFalse((volume / "run" / "rule-check-failed.json").exists())
                        continue
                    answer = call()
                self.assertTrue(answer.startswith("stopped: rule check failed at generation 2610: the round's"))
                self.assertEqual(len(answer), len("stopped: rule check failed at generation 2610: ") + 300)
                marker = json.loads((volume / "run" / "rule-check-failed.json").read_text(encoding="utf-8"))
                self.assertEqual(marker["generation"], 2610)
                self.assertTrue(marker["message"].startswith("the round's account"))
                self.assertIn("T", marker["time"])
                self.assertEqual(commits, [1])
                self.assertEqual(validate_checkpoint(volume / "run/latest.pt"), 2610)

    def test_a_rule_check_stop_after_a_generation_publishes_that_generation_as_a_clean_exit_does(self):
        """The trainer prints a generation's record before it appends it to
        its log, so the publication made as the record arrives can miss it;
        a clean exit publishes once more, and so does a rule check's stop,
        which refused the round after that generation without learning or
        saving anything of it, before it leaves its marker."""
        from functools import partial

        from neural import cloud_runs
        from neural.checkpoints import atomic_save
        from neural.tests.test_cloud_isolation import controller

        app = controller()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            volume = root / "volume"
            atomic_save({"generation": 2610}, volume / "run/latest.pt")
            (volume / "run" / "generation.txt").write_text("2610")
            record = json.dumps({"generation": 2610, "checkpoint_generation": 2611})
            logs: list[str] = []

            def popen(command, **_kwargs):
                where = Path(command[command.index("--out") + 1])

                def printed():
                    atomic_save({"generation": 2611}, where / "latest.pt")
                    yield record + "\n"
                    (where / "log.jsonl").write_text(record + "\n", encoding="utf-8")
                    (where / cloud_requests.RULE_CHECK_FILE).write_text("the round's account does not add up\n",
                                                                         encoding="utf-8")
                    yield "stopped: rule check failed: the round's account does not add up\n"

                return SimpleNamespace(stdout=printed(), wait=lambda: cloud_requests.RULE_CHECK_EXIT)

            def publish(where, seen, *_args):
                log = where / "log.jsonl"
                logs.append(log.read_text(encoding="utf-8") if log.exists() else "")
                return seen

            with patch.object(app, "VOLUME", volume), \
                    patch.object(app, "workspace", partial(cloud_runs.workspace, root=root / "scratch")), \
                    patch.object(app, "_environment", return_value={}), patch.object(app, "_save_cache"), \
                    patch.object(app, "_publish", side_effect=publish), \
                    patch.object(app, "volume", SimpleNamespace(commit=lambda: None, reload=lambda: None)), \
                    patch.object(app.subprocess, "Popen", side_effect=popen), \
                    contextlib.redirect_stdout(io.StringIO()):
                answer = app.train_combined(run="run", generations=3, preview_reach=True,
                                            last_hand_rules=list(BOTH), compile=False)
            self.assertEqual(logs, ["", record + "\n"])
            self.assertTrue(answer.startswith("stopped: rule check failed at generation"))
            self.assertTrue((volume / "run" / "rule-check-failed.json").exists())


# ---------------------------------------------------------------------------
# End to end, on the processor, with the real engine.

#: Tables, and the deals of a round of `Reaching` that fires the all-last
#: dama rule (4 times), decline-last in a claim window (3) and decline-last at
#: a player's own turn (4), found among 200 rounds' seeds by simulating
#: `Reaching` on the engine alone. `Reaching` reaches wherever it can, so a
#: hand ready to win by self-draw has declared already, and no declined
#: self-draw leaves it a reach; the deals of a round of `Damaish` do (twice,
#: each judged by dama and kept, short), found by simulating `Damaish` the
#: same way, the second ruled round the placement reward's refit is run over
#: too. No round found had dama throw undeclared a reach a declined self-draw
#: left (`both`): a self-draw that would leave the player alone in last and
#: then a tile that keeps them first through every win nearly contradict each
#: other, and `DeclineTests` covers that case. And the deals of a round of
#: `Reaching` seated by player at a share of a half, against a
#: `TeacherOpponent` and the Club bot, that fires the all-last dama rule
#: (once) and decline-last in a claim window (once) and at the turn (3
#: times), all at the learner's seats, found by playing seeds 4100 to 4111.
E2E_GAMES = 16
E2E_SEED = 3480
E2E_DAMAISH = 3030
E2E_MIXED = 4101


class Recording:
    """The real arena, keeping every game's mjai events as they are read."""

    def __init__(self, *args, **kwargs):
        self.native = NATIVE_ARENA(*args, **kwargs)
        self.lines: list[list[str]] | None = None

    def mjai_all(self):
        got = self.native.mjai_all()
        if self.lines is None:
            self.lines = [[] for _ in got]
        for game, lines in enumerate(got):
            self.lines[game].extend(lines)
        return got

    def __getattr__(self, name):
        return getattr(self.native, name)


class Reaching:
    """A learner that plays the heuristic player's moves, but a reach
    wherever one is open, and a win, above all, wherever the heuristic
    player takes one; every reach's tile is the heuristic player's own where
    it declares and otherwise the first riichi tile open. Each is pushed by
    `PUSH`, so play is the same whatever is drawn, and the rules are asked as
    often as a learner can make them: every last-hand riichi a non-dealer can
    declare, and every win the heuristic player takes. Its network gives the
    values recorded, and the measurement plays the network's own plain
    choice (`choose`). `witnessed` keeps each event a rule added, with how
    many of its game's events had been read when it was decided (its arena
    a `Recording`); `reach_tiles`, by (game, events read, player), the tile
    the record of each reach a declined self-draw left holds, where the
    reach stood; `accounts`, each round's account it was handed; and
    `tile_records_on_dama_rows`, how many tile records a step kept for a
    row the all-last dama rule had thrown undeclared, of `dama_rows`."""

    kind = "mortal"

    def __init__(self, net) -> None:
        self.net = net
        self.timing: dict = {}
        self.witnessed: list[tuple] = []
        self.reach_tiles: dict[tuple, int] = {}
        self.accounts: list = []
        self.tile_records_on_dama_rows = 0
        self.dama_rows = 0

    def eval(self):
        self.net.eval()
        return self

    def choose(self, views, rows, players, legal):
        return self.net.choose(views, rows, players, legal)

    def pushes(self, taught: np.ndarray, allowed: np.ndarray, legal: np.ndarray):
        """What is added to the network's logits: for each row, by the
        heuristic player's move `taught`, and for each reach's preview, the
        rows that may declare (`ahead`) and the tile each is pushed to."""
        n = len(taught)
        win = (taught == zoo.TSUMO) | (taught == zoo.RON)
        first = np.zeros((n, zoo.MORTAL_ACTIONS), dtype=np.float32)
        first[np.arange(n), TO_MORTAL[taught]] = PUSH
        first[allowed[:, zoo.MORTAL_RIICHI], zoo.MORTAL_RIICHI] += 2 * PUSH
        first[win, zoo.MORTAL_AGARI] += 4 * PUSH
        ahead = np.flatnonzero(allowed[:, zoo.MORTAL_RIICHI])
        declares = (taught[ahead] >= zoo.RIICHI_DISCARD) & (taught[ahead] < zoo.TSUMO)
        tiles = np.where(declares, taught[ahead] - zoo.RIICHI_DISCARD,
                         legal[ahead, zoo.RIICHI_DISCARD:zoo.TSUMO].argmax(axis=1))
        return first, ahead, tiles

    @torch.no_grad()
    def decide(self, views, rows, players, legal, greedy=False, explore_share=0.0, wanderer=None,
               preview_reach=False, rules=None, **draws):
        if not preview_reach:
            raise AssertionError("this learner pushes the reaches' previews, and needs them")
        legal = np.atleast_2d(legal)
        n = len(rows)
        taught = np.frombuffer(views.arena.teacher(), dtype=np.uint8)[rows].astype(np.int64)
        allowed = zoo.translatable(legal)
        first, ahead, tiles = self.pushes(taught, allowed, legal)

        def score(planes, mask):
            if len(mask) != n + len(ahead):
                raise AssertionError("one question a step, the reaches' previews after the rows")
            logits, value = self.net.decision(planes, mask)[:2]
            push = torch.zeros(mask.shape)
            push[:n] = torch.from_numpy(first)
            push[n + torch.arange(len(ahead)), torch.from_numpy(tiles)] = PUSH
            return logits.float() + push, value

        before = 0
        if rules is not None:
            if not self.accounts or self.accounts[-1] is not rules:
                self.accounts.append(rules)
            before = len(rules.events)
        choice, records = mortal_learner.decide_in_mortal_space(
            score, views, rows, players, legal, greedy, "cpu", self.timing, explore_share=explore_share,
            wanderer=wanderer, preview_reach=preview_reach, rules=rules, **draws)
        if rules is not None:
            lines = getattr(views.arena, "lines", None)
            row_of = {int(game): row for row, game in enumerate(np.asarray(rows).tolist())}
            # The step's tile records follow its decidable rows' records.
            start = int(allowed.any(axis=1).sum())
            tile_slots = records.slots[start:]
            tile_of = dict(zip(tile_slots.tolist(), records.actions[start:].tolist()))
            dama_rows = set()
            for event in rules.events[before:]:
                position = None if lines is None else len(lines[event[0]])
                self.witnessed.append((event, position))
                row = row_of[event[0]]
                if event[3] == DAMA:
                    dama_rows.add(row)
                elif event[4] == zoo.MORTAL_RIICHI and row in tile_of:
                    self.reach_tiles[(event[0], position, event[2])] = tile_of[row]
            self.tile_records_on_dama_rows += len(dama_rows & set(tile_slots.tolist()))
            self.dama_rows += len(dama_rows)
        return choice, records


class Damaish(Reaching):
    """A learner that plays the heuristic player's moves with its riichi
    turned into the plain discard of the tile, and a win, above all, where
    the heuristic player takes one; the first open discard pushed by half as
    a fallback, and a riichi pushed above every move but the win only at a
    self-draw the heuristic player takes, so that a riichi is a declined
    self-draw's best other move wherever one is open; each preview's tile
    the first riichi tile open. It never declares of its own: every reach it
    makes is one a declined self-draw left, which dama then judges."""

    def pushes(self, taught: np.ndarray, allowed: np.ndarray, legal: np.ndarray):
        n = len(taught)
        mapped = TO_MORTAL[taught].copy()
        declares = (taught >= zoo.RIICHI_DISCARD) & (taught < zoo.TSUMO)
        mapped[declares] = taught[declares] - zoo.RIICHI_DISCARD
        first = np.zeros((n, zoo.MORTAL_ACTIONS), dtype=np.float32)
        first[np.arange(n), mapped] = PUSH
        discards = allowed[:, :34]
        some = np.flatnonzero(discards.any(axis=1))
        first[some, discards[some].argmax(axis=1)] += 0.5 * PUSH
        first[(taught == zoo.TSUMO) & allowed[:, zoo.MORTAL_RIICHI], zoo.MORTAL_RIICHI] += 2 * PUSH
        first[(taught == zoo.TSUMO) | (taught == zoo.RON), zoo.MORTAL_AGARI] += 4 * PUSH
        ahead = np.flatnonzero(allowed[:, zoo.MORTAL_RIICHI])
        return first, ahead, legal[ahead, zoo.RIICHI_DISCARD:zoo.TSUMO].argmax(axis=1)


class TeacherOpponent:
    """A seated other that plays the heuristic player's moves, asked with the
    four arguments a seated other takes and nothing more: handed the rules,
    it would fail. `asked` counts the decisions it was asked for."""

    kind = "engine"

    def __init__(self) -> None:
        self.asked = 0

    def eval(self):
        return self

    def choose(self, views, rows, players, legal):
        self.asked += len(rows)
        return np.frombuffer(views.arena.teacher(), dtype=np.uint8)[np.asarray(rows)].astype(np.int64)


def ruled_play(net, seed: int, rules=BOTH, games: int = E2E_GAMES, ledger: bool = True, learner=Reaching,
               opponents=None, seat_share: float = 0.0):
    """A round of `learner`, `Reaching` unless told, on the processor with
    the reach previewed, its events kept, seated with `opponents` by player
    at `seat_share` when given: the round, the learner and the arena."""
    made: list = []

    def arena(*args, **kwargs):
        made.append(Recording(*args, **kwargs))
        return made[-1]

    learner = learner(net)
    with patch.object(selfplay.riichi_py, "Arena", side_effect=arena):
        batch = selfplay.play(learner, games=games, seed=seed, device="cpu", preview_reach=True,
                              last_hand_rules=rules, ledger=ledger, opponents=opponents, seat_share=seat_share)
    return batch, learner, made[0]


def decisions_of(learner) -> dict[tuple, tuple]:
    """Each decision the rules played in a round, by (game, events read,
    player): the (hand, rule, move) of its last event, dama's where it threw
    undeclared the reach a declined self-draw left."""
    last = {}
    for (game, hand, player, rule, move), position in learner.witnessed:
        last[(game, position, player)] = (hand, rule, move)
    return last


def witness(lines: list[str], position: int, player: int, move: int, tile: int | None = None) -> str | None:
    """What is wrong, if anything, with a player's events after a rule's
    move, `move` in Mortal's moves, played when `position` of its game's
    events had been read: the move itself next, a discard of that tile and
    no reach (a reach then its discard, of `tile` when given, a quad), or,
    for the pass of a claim window, nothing from the player as the window
    resolves; never a win on the tile declined."""
    after = [json.loads(line) for line in lines[position:]]
    if move == zoo.MORTAL_PASS:
        # The window resolves before anyone draws or discards again, or the
        # hand ends.
        resolved = []
        for event in after:
            if event["type"] in ("tsumo", "dahai", "end_kyoku", "ryukyoku", "start_kyoku", "end_game"):
                break
            resolved.append(event)
        mine = [event for event in resolved if event.get("actor") == player]
        return f"the player answered the window it passed: {mine}" if mine else None
    mine = [event for event in after if event.get("actor") == player]
    if not mine:
        return "the player did nothing after the rule's move"
    if move < 34:
        want = [{"type": "dahai", "pai": TILE_NAMES[move]}]
    elif move == zoo.MORTAL_RIICHI:
        want = [{"type": "reach"}, {"type": "dahai", **({} if tile is None else {"pai": TILE_NAMES[tile]})}]
    elif move == zoo.MORTAL_KAN:
        want = [{"type": ("ankan", "kakan")}]
    else:
        return f"no witness for Mortal's move {move}"
    for got, wanted in zip(mine, want):
        for key, value in wanted.items():
            if got.get(key) != value and not (isinstance(value, tuple) and got.get(key) in value):
                return f"the player's next events {mine[:len(want)]} are not the rule's move {move}"
    return None


def witnessed_problems(learner, arena) -> list[tuple]:
    """What the table's own events say against each decision the rules
    played in a round (see `witness`), a reach a declined self-draw left
    read with the tile its record holds, which it must have."""
    problems = []
    for (game, position, player), (_hand, rule, move) in decisions_of(learner).items():
        tile = learner.reach_tiles.get((game, position, player))
        wrong = ("a reach the rules left has no tile record" if move == zoo.MORTAL_RIICHI and tile is None
                 else witness(arena.lines[game], position, player, move, tile))
        if wrong:
            problems.append((game, position, player, rule, move, wrong))
    return problems


class EndToEndTests(unittest.TestCase):
    """One round of `Reaching` with both rules, one without and one with
    both but no ledger, whose rules ask the engine about every row drawn, on
    the same deals; a round of `Damaish` with both rules; and a round of
    `Reaching` at mixed tables with both rules, with its ledger and without,
    shared by the tests below: a few minutes on the processor."""

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(7)
        cls.net = fusion(8, 1).eval()
        cls.batch, cls.learner, cls.arena = ruled_play(cls.net, E2E_SEED)
        cls.played = cls.learner.accounts[-1]
        cls.plain, _learner, _arena = ruled_play(cls.net, E2E_SEED, rules=())
        cls.unkept, cls.unkept_learner, _arena = ruled_play(cls.net, E2E_SEED, ledger=False)
        cls.second, cls.damaish, cls.damaish_arena = ruled_play(cls.net, E2E_DAMAISH, learner=Damaish)
        cls.others = [TeacherOpponent(), zoo.ClubPlayer()]
        cls.mixed, cls.mixed_learner, cls.mixed_arena = ruled_play(cls.net, E2E_MIXED, opponents=cls.others,
                                                                   seat_share=0.5)
        cls.mixed_unkept, cls.mixed_unkept_learner, _arena = ruled_play(
            cls.net, E2E_MIXED, ledger=False, opponents=[TeacherOpponent(), zoo.ClubPlayer()], seat_share=0.5)
        cls.rounds = (("reaching", cls.batch, cls.learner, cls.arena),
                      ("damaish", cls.second, cls.damaish, cls.damaish_arena),
                      ("mixed", cls.mixed, cls.mixed_learner, cls.mixed_arena))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_each_rule_fires_and_the_flags_add_up(self):
        report = self.batch.rules
        self.assertEqual(report, self.played.report())
        self.assertGreaterEqual(report[DAMA]["fired"], 1)
        self.assertGreaterEqual(report[DECLINE]["in_window"], 1)
        self.assertGreaterEqual(report[DECLINE]["at_turn"], 1)
        flagged = int(self.batch.ruled.sum())
        self.assertEqual(flagged, report["ruled_records"])
        self.assertEqual(flagged, report[DAMA]["fired"] + report[DECLINE]["fired"] - report["both"])
        self.assertEqual(report[DECLINE]["in_window"] + report[DECLINE]["at_turn"], report[DECLINE]["fired"])
        self.assertEqual(report["tile_records"], report["reaches_sent"])
        self.assertEqual(len(self.batch.ledger), self.batch.decisions)
        self.assertGreater(report["seconds"], 0.0)
        # Every step's tile records left out the rows dama threw undeclared.
        self.assertGreater(self.learner.dama_rows, 0)
        self.assertEqual(self.learner.tile_records_on_dama_rows, 0)

    def test_every_flagged_record_is_its_move_alone_at_probability_one(self):
        for name, batch, _learner, _arena in self.rounds:
            with self.subTest(round=name):
                flagged = batch.ruled.nonzero().flatten()
                self.assertGreater(len(flagged), 0)
                legal = batch.legal[flagged]
                self.assertTrue(bool((legal.sum(dim=1) == 1).all()))
                self.assertTrue(bool(legal[torch.arange(len(flagged)), batch.actions[flagged]].all()))
                self.assertTrue(torch.equal(batch.log_probs[flagged], torch.zeros(len(flagged))))
                # The rules' moves are the pass, a discard and the moves after
                # a declined self-draw, never a win.
                self.assertFalse(bool((batch.actions[flagged] == zoo.MORTAL_AGARI).any()))

    def test_each_flagged_record_is_the_decision_a_rule_played(self):
        """Every move these learners make is certain, so the test above would
        hold for flags on any rows: read through the ledger, the flagged
        records are exactly the decisions the rules played, by game, hand,
        player and move, a declined self-draw's reach that dama then threw
        undeclared once, as dama's discard."""
        for name, batch, learner, _arena in self.rounds:
            with self.subTest(round=name):
                led = batch.ledger
                flagged = collections.Counter(
                    (int(led.games[row]), int(led.hands[row]), int(led.players[row]), int(batch.actions[row]))
                    for row in batch.ruled.nonzero().flatten().tolist())
                decided = collections.Counter((game, hand, player, move) for (game, _position, player),
                                              (hand, _rule, move) in decisions_of(learner).items())
                self.assertGreater(sum(decided.values()), 0)
                self.assertEqual(flagged, decided)

    def test_the_tables_own_events_witness_every_firing(self):
        """After each decision a rule played, the player's next events are
        its move (see `witness`): a reach a declined self-draw left is the
        reach and then the tile its record holds thrown."""
        for name, _batch, learner, arena in self.rounds:
            with self.subTest(round=name):
                decided = decisions_of(learner)
                self.assertGreater(len(decided), 0)
                self.assertEqual(witnessed_problems(learner, arena), [])
                rules = {rule for _hand, rule, _move in decided.values()}
                if name == "damaish":
                    self.assertIn(DECLINE, rules)
                else:
                    self.assertEqual(rules, {DAMA, DECLINE})
        # The reaches declined self-draws left, each read with its tile.
        self.assertGreater(len(self.damaish.reach_tiles), 0)

    def test_a_declined_self_draw_that_leaves_a_riichi_is_judged_by_dama_and_keeps_its_tile_record(self):
        """With the real engine, a learner that declares only as a declined
        self-draw's best other move: each such reach is judged by dama, and
        each that stands is recorded alone, flagged, and followed by its
        tile record, the policy's own, unflagged, over the riichi tiles, for
        the same player and hand."""
        report, batch = self.second.rules, self.second
        to_riichi = report[DECLINE]["to_riichi"]
        self.assertGreaterEqual(to_riichi, 1)
        self.assertGreaterEqual(report[DAMA]["judged"], to_riichi)
        self.assertEqual(report["tile_records"], report["reaches_sent"])
        led, ruled, actions = batch.ledger, batch.ruled.numpy(), batch.actions.numpy()
        games = np.asarray(led.games)
        stood = np.flatnonzero(ruled & (actions == zoo.MORTAL_RIICHI)).tolist()
        self.assertEqual(len(stood), to_riichi - report["both"])
        for row in stood:
            # The game's next record, the step's tile record of the same row.
            after = row + 1 + np.flatnonzero(games[row + 1:] == games[row])
            self.assertTrue(len(after), row)
            tile = int(after[0])
            mask = batch.legal[tile].numpy()
            self.assertFalse(ruled[tile])
            self.assertEqual((int(led.players[tile]), int(led.hands[tile])),
                             (int(led.players[row]), int(led.hands[row])))
            self.assertTrue(mask[:34].any() and not mask[34:].any() and mask[actions[tile]], np.flatnonzero(mask))

    def test_at_mixed_tables_the_rules_play_only_the_learners_seats(self):
        """Seated by player at a share of a half, against a seated other that
        would fail if handed the rules and the Club bot, whose places the
        arena plays: both rules fire, every decision they played sits at a
        seat the learner holds, and the same round without a ledger, whose
        rules ask the engine about every row drawn, plays the same."""
        owners = self.mixed.ledger.owners
        self.assertGreater(int((owners == 0).sum()), 0)
        self.assertGreater(int((owners == 1).sum()), 0)
        self.assertGreater(self.others[0].asked, 0)
        decided = decisions_of(self.mixed_learner)
        report = self.mixed.rules
        self.assertGreaterEqual(report[DAMA]["fired"], 1)
        self.assertGreaterEqual(report[DECLINE]["in_window"], 1)
        self.assertGreaterEqual(report[DECLINE]["at_turn"], 1)
        self.assertEqual([(game, player) for (game, _position, player) in decided if owners[game, player] != -1], [])
        unkept = self.mixed_unkept.rules
        for name in BOTH:
            self.assertEqual({key: value for key, value in unkept[name].items() if key != "calls"},
                             {key: value for key, value in report[name].items() if key != "calls"})
            self.assertLess(report[name]["calls"], unkept[name]["calls"])
        for key in ("ruled_records", "both", "tile_records", "reaches_sent"):
            self.assertEqual(unkept[key], report[key])
        self.assertEqual(self.mixed_unkept_learner.accounts[-1].events, self.mixed_learner.accounts[-1].events)
        self.assertTrue(torch.equal(self.mixed_unkept.ruled, self.mixed.ruled))
        self.assertTrue(torch.equal(self.mixed_unkept.actions, self.mixed.actions))
        np.testing.assert_array_equal(self.mixed_unkept.final_scores, self.mixed.final_scores)

    def test_asking_the_engine_only_about_the_last_hand_changes_nothing(self):
        """Without a ledger every row drawn is asked about; with one, only
        a non-dealer's in South 4 (`selfplay_rules.last_hand`): the same
        round, the same judgements and firings, from fewer questions."""
        unkept = self.unkept.rules
        for name in BOTH:
            self.assertEqual({key: value for key, value in unkept[name].items() if key != "calls"},
                             {key: value for key, value in self.batch.rules[name].items() if key != "calls"})
            self.assertLess(self.batch.rules[name]["calls"], unkept[name]["calls"])
        for key in ("ruled_records", "both", "tile_records", "reaches_sent"):
            self.assertEqual(unkept[key], self.batch.rules[key])
        self.assertTrue(torch.equal(self.unkept.ruled, self.batch.ruled))
        self.assertTrue(torch.equal(self.unkept.actions, self.batch.actions))
        np.testing.assert_array_equal(self.unkept.final_scores, self.batch.final_scores)
        self.assertEqual(self.unkept_learner.accounts[-1].events, self.played.events)

    def test_with_the_rules_off_only_tables_a_rule_fired_at_end_otherwise(self):
        fired_at = self.played.dama.tables | self.played.decline.tables
        differ = set(np.flatnonzero((self.batch.final_scores != self.plain.final_scores).any(axis=1)).tolist())
        self.assertTrue(differ)
        self.assertLessEqual(differ, fired_at)
        self.assertIsNone(self.plain.rules)
        self.assertIsNone(self.plain.ruled)

    def test_the_trainers_epochs_learn_nothing_from_the_flagged_rows(self):
        """Their log-probabilities under the network with their masks are
        exactly nought; and the trainer's epochs over them and 64 other rows
        learn bit for bit the same weights and Adam's state with their
        advantages flipped."""
        from neural.tests.test_step_guard import differences

        flagged = self.batch.ruled.nonzero().flatten()
        others = (~self.batch.ruled).nonzero().flatten()[:64]
        with torch.no_grad():
            planes = self.batch.observations.rows(flagged.numpy()).dense("cpu")
            logits, _value, _hands = self.net.everything(planes, self.batch.legal[flagged])
            log_prob = torch.distributions.Categorical(logits=logits.float()).log_prob(self.batch.actions[flagged])
        self.assertTrue(torch.equal(log_prob, torch.zeros(len(flagged))))

        rows = torch.cat([flagged, others])
        part = SimpleNamespace(decisions=len(rows), observations=self.batch.observations.rows(rows.numpy()),
                               legal=self.batch.legal[rows], actions=self.batch.actions[rows],
                               returns=self.batch.returns[rows], log_probs=self.batch.log_probs[rows])
        rollout = ppo_loop.on_device(part, "cpu", card=False)
        args = SimpleNamespace(lr=1e-3, lr_ours=1e-3, lr_mortal=1e-3, epochs=2, batch=16, explore=0.0, clip=0.2,
                               value_weight=0.5, hands_weight=1.0, leash=0.0, entropy=0.0005, entropy_target=0.0,
                               entropy_rate=0.01, entropy_max=0.05)
        advantages, _spread = ppo_loop.standardised(rollout.returns, torch.zeros_like(rollout.returns))
        learned = []
        for sign in (3.0, -3.0):
            net = copy.deepcopy(self.net)
            net.set_mode("none")
            net.train()
            optimiser = torch.optim.AdamW(train_combined.optimiser_groups(net, args), lr=args.lr, weight_decay=1e-4,
                                          fused=True)
            given = advantages.clone()
            given[: len(flagged)] = sign
            torch.manual_seed(5)
            updates, _picks, _planes = train_combined.ppo_epochs(
                net, net.everything, rollout, self.batch.held[rows], optimiser,
                ppo_loop.Totals("cpu", "leash", "hands", "covered"), PolicyDrift(0.0), args, torch.float32, False,
                None, advantages=given, begun=None, explored=torch.zeros(len(rows), dtype=torch.bool),
                floor=train_combined.EntropyFloor(args))
            learned.append({"updates": updates, "weights": dict(net.state_dict()),
                            "adam": [optimiser.state[p] for g in optimiser.param_groups for p in g["params"]]})
        self.assertGreater(learned[0]["updates"], 0)
        self.assertEqual(differences(learned[0], learned[1]), [])

    def test_the_measurement_and_the_placement_reward_run_on_after_ruled_rounds(self):
        measured = selfplay.measure(self.learner, games=2, seed=7_000_000, device="cpu")
        self.assertEqual(set(measured), {"placement", "score", "wins", "hands"})
        placing = place_model.PlacementRun(place_model.PlaceModel((), hidden=8), 0.05, since=0, warm_up=0,
                                           refit_every=2, refit_rounds=2)
        for generation, batch in enumerate((self.batch, self.second)):
            self.assertIsNotNone(batch.rules)
            batch = copy.copy(batch)
            selfplay.settle_placement(batch, placing.reward)
            placing.remember(batch.ledger, generation)
        self.assertTrue(placing.refit_due(1))
        report = placing.refit(1, seed=3)
        self.assertEqual((report["generation"], report["rounds"]), (2, 2))
        self.assertIn("adopted", report)


if __name__ == "__main__":
    unittest.main()
