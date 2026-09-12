"""Record the real native hand ledger independently of Python reward assignment."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import riichi_py

from neural import selfplay, duel, searched
from neural.observe import Planes
from neural.outcomes import IncompleteGamesError

NATIVE_ARENA = riichi_py.Arena


class Heuristic:
    kind = 'engine'

    def eval(self):
        return self

    def choose(self, views, rows, players, legal):
        return np.frombuffer(views.arena.teacher(), dtype=np.uint8)[rows].astype(np.int64)

    def decide(self, views, rows, players, legal, greedy, **_exploration):
        chosen = self.choose(views, rows, players, legal)
        # Only bookkeeping is under test; the decisions and labels are native,
        # while compact placeholder observations avoid unnecessary NN inference.
        records = SimpleNamespace(
            planes=Planes(np.zeros(len(rows)+1, dtype=np.int64),
                          np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.float16)),
            masks=legal.copy(), actions=chosen, log_probs=np.zeros(len(rows)), slots=np.arange(len(rows)))
        return chosen, records

    def everything(self, planes, legal):
        logits = torch.zeros_like(legal, dtype=torch.float32).masked_fill(~legal, -1e9)
        return logits, torch.zeros(len(legal)), torch.zeros(len(legal), 3, 34)


class LedgerArena:
    def __init__(self, *args, **kwargs):
        self.native = NATIVE_ARENA(*args, **kwargs)
        self.games = kwargs['games']
        rng = np.random.default_rng(kwargs['seed'] ^ 0x0DDBA11)
        rng.random(self.games)
        self.foreign = rng.integers(0, 4, size=self.games)
        self.pending = [[[] for _ in range(4)] for _ in range(self.games)]
        self.people, self.game_ids, self.rewards = [], [], []
        self.hands = self.foreign_endings = self.steps = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def final_scores(self):
        if not self.native.all_finished():
            raise AssertionError('provisional scores were requested')
        return self.native.final_scores()

    def step(self, choices):
        seats = np.frombuffer(self.native.seats(), dtype=np.uint8)
        players = np.frombuffer(self.native.seat_players(), dtype=np.uint8).reshape(self.games, 4)
        owners = {}
        for game in np.flatnonzero(seats != 0xFF):
            person = int(players[game, seats[game]])
            owners[game] = person
            if person != self.foreign[game]:
                self.pending[game][person].append(len(self.rewards))
                self.people.append(person)
                self.game_ids.append(game)
                self.rewards.append(0.)
        self.native.step(choices)
        self.steps += 1
        ended = np.frombuffer(self.native.hand_ended(), dtype=np.uint8)
        deltas = np.frombuffer(self.native.hand_result(), dtype=np.int32).reshape(self.games, 4)
        for game in np.flatnonzero(ended):
            self.hands += 1
            self.foreign_endings += owners.get(game) == self.foreign[game]
            for person in range(4):
                for row in self.pending[game][person]:
                    self.rewards[row] += int(deltas[game, person]) * selfplay.HAND_SCALE
                self.pending[game][person].clear()


class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def play_with_ledger(self, games=1, max_steps=4000):
        engines = []
        def create(**kwargs):
            engine = LedgerArena(**kwargs)
            engines.append(engine)
            return engine
        with patch.object(selfplay.riichi_py, 'Arena', side_effect=create):
            batch = selfplay.play(Heuristic(), games, 1, 'cpu', greedy=True,
                                  opponents=[Heuristic()], opponent_share=1., max_steps=max_steps)
        return batch, engines[0]

    def test_every_hand_settles_even_when_only_external_opponents_move(self):
        for games in (1, 4):
            with self.subTest(games=games):
                batch, ledger = self.play_with_ledger(games)
                self.assertTrue(ledger.all_finished())
                self.assertGreater(ledger.foreign_endings, 0)
                self.assertEqual(batch.hands, ledger.hands)
                self.assertEqual(batch.decisions, len(ledger.rewards))
                expected = np.array(ledger.rewards)
                # Independent tie-aware reference, not the production helper.
                for row, (game, person) in enumerate(zip(ledger.game_ids, ledger.people)):
                    scores = batch.final_scores[game]
                    first = int((scores > scores[person]).sum())
                    tied = int((scores == scores[person]).sum())
                    expected[row] += np.mean(selfplay.PLACEMENT_VALUE[first:first+tied])
                np.testing.assert_allclose(batch.returns.numpy(), expected, atol=1e-6)
                self.assertTrue(all(not rows for game in ledger.pending for rows in game))

    def test_completing_on_the_last_allowed_step_is_not_truncation(self):
        batch, ledger = self.play_with_ledger()
        exact, _ = self.play_with_ledger(max_steps=ledger.steps)
        np.testing.assert_array_equal(exact.returns, batch.returns)
        with self.assertRaises(IncompleteGamesError):
            self.play_with_ledger(max_steps=ledger.steps-1)

    def test_all_simulators_reject_unfinished_games_before_reading_final_scores(self):
        calls = (
            lambda: selfplay.play(Heuristic(), 1, 1, 'cpu', max_steps=1),
            lambda: selfplay.measure(Heuristic(), 1, 1, 'cpu', max_steps=1),
            lambda: duel.table(Heuristic(), Heuristic(), 1, 1, 0, 'cpu', max_steps=1),
            lambda: searched.play(Heuristic(), 1, 1, None, 1, 1, 0., device='cpu', max_steps=1),
        )
        for call in calls:
            with self.subTest(simulator=call), patch.object(riichi_py, 'Arena', LedgerArena):
                with self.assertRaisesRegex(IncompleteGamesError, 'after 1 steps.*unfinished'):
                    call()
