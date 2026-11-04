"""The promises self-play has to keep before anything is learned from it.

Every one of these was a real defect rather than a hypothetical: a hand
that ended on a step nobody recorded went unpaid and its decisions were
then paid by the next hand; a round that ran out of steps turned games
still in progress into final placements; an index naming no legal move was
replaced by one that was legal while the record kept the move that was
asked for; and a round too small for one minibatch trained on nothing and
said so nowhere.

    python -m unittest neural.tests.test_selfplay_contract -v
"""

from __future__ import annotations

import unittest

import numpy as np

import riichi_py

SEED = 4_242


class FirstLegal:
    """A player that always takes the first move the rules allow.

    Enough to drive the arena without a checkpoint: these tests are about
    what the collector records and refuses, not about how well anyone
    plays.
    """

    kind = "engine"

    def eval(self):
        return self

    def choose(self, views, rows, players, legal):
        return np.asarray(legal, dtype=bool).argmax(axis=1).astype(np.int64)

    def everything(self, planes, mask):
        """Policy, value and belief, shaped as the collector expects.

        The policy puts all its weight on the first legal move, so play is
        deterministic and never names an action the rules forbid — which
        matters now that the arena refuses one rather than substituting.
        """
        import torch

        rows = mask.shape[0]
        logits = torch.where(mask, 0.0, float("-inf"))
        logits[:, 0] = torch.where(mask[:, 0], torch.tensor(1.0), logits[:, 0])
        first = mask.float().argmax(dim=1)
        logits = torch.full_like(logits, float("-inf"))
        logits[torch.arange(rows), first] = 0.0
        value = torch.zeros(rows)
        guessed = torch.zeros(rows, riichi_py.OPPONENTS, riichi_py.POSITIONS)
        return logits, value, guessed


class StrictActions(unittest.TestCase):
    """An index that names no legal move must not be quietly replaced."""

    def test_an_illegal_index_is_refused_rather_than_replaced(self):
        """The step raises, so a record can never describe a move the
        engine did not play."""
        arena = riichi_py.Arena(games=1, seed=SEED, bot_places=[])
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(1, -1).astype(bool)
        illegal = int(np.nonzero(~mask[0])[0][0])
        with self.assertRaises(ValueError) as caught:
            arena.step([illegal])
        self.assertIn("illegal action", str(caught.exception))


class TruncationIsNotAnEnding(unittest.TestCase):
    """A game that did not finish has no result to report."""

    def test_self_play_refuses_a_round_that_ran_out_of_steps(self):
        from neural import selfplay, zoo  # noqa: F401  (import cost is the point)

        with self.assertRaises(RuntimeError) as caught:
            selfplay.play(FirstLegal(), games=4, seed=SEED, device="cpu", max_steps=3)
        self.assertIn("unfinished", str(caught.exception))

    def test_the_duel_refuses_a_run_that_ran_out_of_steps(self):
        from neural import duel

        with self.assertRaises(RuntimeError) as caught:
            duel.table(FirstLegal(), FirstLegal(), games=4, seed=SEED, place=0,
                       device="cpu", max_steps=3)
        self.assertIn("unfinished", str(caught.exception))

    def test_a_generous_limit_finishes_and_reports(self):
        """The guard must not fire on an ordinary run."""
        from neural import duel

        scores = duel.table(FirstLegal(), FirstLegal(), games=4, seed=SEED, place=0,
                            device="cpu")
        self.assertEqual(scores.shape, (4, 4))


class EveryHandIsPaidOnce(unittest.TestCase):
    """The defect this guards against, in its own words.

    When every live game's next decision belonged to a seated foreign
    opponent, the collector stepped the arena and went straight round the
    loop, skipping the block that pays out a hand that has just ended. The
    engine forgets a hand ended as soon as the next step is taken, so those
    points were lost — and the decisions waiting for them stayed waiting,
    to be paid instead by the hand after, which credits a decision with
    points from a hand it was not part of.

    `play` now asserts that nothing is still waiting once the games are
    over, so exercising the path is enough to catch a regression. A
    foreign opponent is seated in every game to make the path common: with
    two games, the steps where both owe their decision to that one player
    come round often.
    """

    def test_a_hand_ending_on_an_opponent_only_step_is_still_paid(self):
        from neural import selfplay

        batch = selfplay.play(
            FirstLegal(),
            games=2,
            seed=SEED,
            device="cpu",
            opponents=[FirstLegal()],
            opponent_share=1.0,
        )
        self.assertGreater(batch.hands, 0, "no hand was played")
        self.assertGreater(batch.decisions, 0, "nothing was recorded")
        self.assertEqual(len(batch.returns), batch.decisions)

    def test_the_same_holds_without_any_foreign_opponent(self):
        from neural import selfplay

        batch = selfplay.play(FirstLegal(), games=2, seed=SEED, device="cpu")
        self.assertGreater(batch.hands, 0)


class FirstLegalDecider(FirstLegal):
    """FirstLegal on the path of a learner that records its own decisions,
    as the Mortal-space learners do."""

    def decide(self, views, rows, players, legal, greedy=False, **_exploration):
        from types import SimpleNamespace

        from neural.observe import Planes

        chosen = self.choose(views, rows, players, legal)
        n = len(rows)
        records = SimpleNamespace(
            planes=Planes(np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.uint16),
                          np.zeros(0, dtype=np.float16)),
            masks=np.asarray(legal, dtype=bool).copy(), actions=chosen,
            log_probs=np.zeros(n, dtype=np.float32), slots=np.arange(n))
        return chosen, records


class TwiceRecordingDecider(FirstLegalDecider):
    """FirstLegalDecider recording every step's first row twice, as a
    riichi is recorded: two decisions from one row, the reach and the tile
    it throws. `sizes` keeps how many records each step made, so a test can
    find each pair: a step's first record and its last; and `games` the
    game of each step's rows, so a test can tell a step whose slots no
    longer name its games, once a game has finished."""

    def __init__(self):
        self.sizes = []
        self.games = []

    def decide(self, views, rows, players, legal, greedy=False, **_exploration):
        from neural.observe import Planes

        chosen, records = super().decide(views, rows, players, legal, greedy)
        n = len(rows)
        records.planes = Planes(np.zeros(n + 2, dtype=np.int64), np.zeros(0, dtype=np.uint16),
                                np.zeros(0, dtype=np.float16))
        records.masks = np.concatenate([records.masks, records.masks[:1]])
        records.actions = np.concatenate([records.actions, records.actions[:1]])
        records.log_probs = np.zeros(n + 1, dtype=np.float32)
        records.slots = np.concatenate([np.arange(n), [0]])
        self.sizes.append(n + 1)
        self.games.append(np.asarray(rows).copy())
        return chosen, records


NATIVE_ARENA = riichi_py.Arena


class CountingArena:
    """The real arena, counting how often each label is copied out of it."""

    def __init__(self, *args, **kwargs):
        self.native = NATIVE_ARENA(*args, **kwargs)
        self.calls = {"oracle": 0, "opponent_hands": 0}

    def __getattr__(self, name):
        return getattr(self.native, name)

    def oracle(self):
        self.calls["oracle"] += 1
        return self.native.oracle()

    def opponent_hands(self):
        self.calls["opponent_hands"] += 1
        return self.native.opponent_hands()


class LabelsAreReadOnlyWhenWanted(unittest.TestCase):
    """The oracle's planes and the opponents' hands are copied out of the
    engine for every game on every step: 95 ms a step at 4,096 games for
    the oracle alone, which no learner that decides for itself reads. They
    are fetched only for whoever reads them, and leaving them out changes
    no move and no reward."""

    def play(self, net, **kwargs):
        from unittest.mock import patch

        from neural import selfplay

        made = []

        def create(*args, **kw):
            made.append(CountingArena(*args, **kw))
            return made[-1]

        with patch.object(selfplay.riichi_py, "Arena", side_effect=create):
            batch = selfplay.play(net, games=2, seed=SEED, device="cpu", **kwargs)
        return batch, made[0].calls

    def assert_hidden_hands_are_held(self, batch):
        """The oracle's first twelve planes, each opponent's concealed tiles
        as four planes of a copy each, say what `held` says each opponent
        held, as shares of their hand, record by record: both were taken
        from the game that owed the record's decision."""
        oracle = batch.oracle.numpy().astype(np.float64)
        counts = oracle[:, :12].reshape(len(oracle), riichi_py.OPPONENTS, 4, riichi_py.POSITIONS).sum(axis=2)
        totals = counts.sum(axis=2, keepdims=True)
        held = batch.held.numpy()
        self.assertEqual(held.shape, counts.shape)
        self.assertTrue((totals > 0).all())
        np.testing.assert_allclose(held, counts / totals, rtol=0, atol=1e-6)

    def test_a_learner_that_decides_for_itself_gets_what_it_asks_for(self):
        plain, calls = self.play(FirstLegalDecider())
        self.assertEqual(calls["oracle"], 0)
        self.assertGreater(calls["opponent_hands"], 0)
        self.assertEqual(len(plain.held), plain.decisions)
        self.assertEqual(plain.oracle.numel(), 0)

        bare, calls = self.play(FirstLegalDecider(), want_held=False)
        self.assertEqual(calls, {"oracle": 0, "opponent_hands": 0})
        self.assertEqual(bare.held.numel(), 0)
        for name in ("actions", "log_probs", "returns", "legal"):
            self.assertTrue(np.array_equal(getattr(plain, name).numpy(), getattr(bare, name).numpy()), name)
        self.assertEqual(plain.final_scores.tolist(), bare.final_scores.tolist())

    def test_the_network_train_py_trains_still_gets_every_label(self):
        batch, calls = self.play(FirstLegal())
        self.assertGreater(calls["oracle"], 0)
        self.assertGreater(calls["opponent_hands"], 0)
        for name in ("held", "oracle"):
            self.assertEqual(len(getattr(batch, name)), batch.decisions, name)

    def test_a_learner_that_decides_for_itself_keeps_the_oracle_when_asked(self):
        """`keep_oracle` gives such a learner a row of the oracle's planes
        for every decision, in bytes of nought and one, and changes no move,
        no reward and no score. Kept beside what the opponents held, the
        two agree row by row."""
        import torch

        plain, calls = self.play(FirstLegalDecider(), want_held=False)
        kept, asked = self.play(FirstLegalDecider(), want_held=False, keep_oracle=True)
        self.assertEqual(calls["oracle"], 0)
        self.assertGreater(asked["oracle"], 0)
        self.assertEqual(plain.oracle.numel(), 0)
        self.assertEqual(tuple(kept.oracle.shape),
                         (kept.decisions, riichi_py.ORACLE_PLANES, riichi_py.POSITIONS))
        self.assertEqual(kept.oracle.dtype, torch.uint8)
        self.assertLessEqual(set(np.unique(kept.oracle.numpy()).tolist()), {0, 1})
        # The three opponents hold a concealed tile each at the least, in
        # the first twelve planes, four a player.
        self.assertTrue(bool((kept.oracle[:, :12].sum(dim=(1, 2)) >= 3).all()))
        for name in ("actions", "log_probs", "returns", "legal"):
            self.assertTrue(np.array_equal(getattr(plain, name).numpy(), getattr(kept, name).numpy()), name)
        self.assertEqual(plain.final_scores.tolist(), kept.final_scores.tolist())
        self.assertEqual(plain.steps, kept.steps)
        self.assertGreater(kept.steps, 0)
        self.assertEqual(plain.oracle_seconds, 0.0)
        self.assertGreater(kept.oracle_seconds, 0.0)
        both, _calls = self.play(FirstLegalDecider(), keep_oracle=True)
        self.assertTrue(torch.equal(both.oracle, kept.oracle))
        self.assertEqual(len(both.held), both.decisions)
        self.assert_hidden_hands_are_held(both)

    def test_a_riichis_two_records_share_their_rows_oracle(self):
        """Two records from one row, as a riichi makes, get that row's
        planes both; and the rows are not all the same planes. Each is the
        planes of the record's own game, not of the game its slot would
        name: they agree with what the opponents held, record by record,
        through steps after a game has finished, where the two differ."""
        learner = TwiceRecordingDecider()
        batch, _calls = self.play(learner, keep_oracle=True)
        oracle = batch.oracle.numpy()
        self.assertEqual(len(oracle), batch.decisions)
        self.assertEqual(sum(learner.sizes), batch.decisions)
        at = 0
        for size in learner.sizes:
            self.assertTrue(np.array_equal(oracle[at], oracle[at + size - 1]))
            at += size
        self.assertGreater(len(np.unique(oracle.reshape(len(oracle), -1), axis=0)), 1)
        self.assert_hidden_hands_are_held(batch)
        # Some step's first row is not game 0's, game 0 having finished: a
        # record read by its slot would have had game 0's planes there, all
        # nought, and would not have agreed.
        self.assertTrue(any(games[0] != 0 for games in learner.games))

    def test_a_learner_told_to_keep_no_oracle_is_never_asked_for_it(self):
        """`keep_oracle=False` leaves the oracle's planes out even for the
        network `train.py` trains: never copied out of the engine, never
        read, the other labels kept as before."""
        batch, calls = self.play(FirstLegal(), keep_oracle=False)
        self.assertEqual(calls["oracle"], 0)
        self.assertGreater(calls["opponent_hands"], 0)
        self.assertEqual(batch.oracle.numel(), 0)
        self.assertEqual(len(batch.held), batch.decisions)


if __name__ == "__main__":
    unittest.main()
