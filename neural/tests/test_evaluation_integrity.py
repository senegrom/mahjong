"""Native ownership and real CLI coverage for both supported policy spaces."""
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import riichi_py

from neural import arena, selfplay
from neural.model import MORTAL_PLANES, PolicyValueNet
from neural.outcomes import IncompleteGamesError, placement_rewards

ROOT = Path(__file__).resolve().parents[2]


class NativeTeacher:
    kind = 'engine'

    def __init__(self):
        self.owners = Counter()

    def eval(self):
        return self

    def choose(self, views, rows, players, legal):
        self.owners.update(players.tolist())
        return np.frombuffer(views.arena.teacher(), dtype=np.uint8)[rows].astype(np.int64)


class EvaluationIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_native_bots_never_escape_through_claim_windows(self):
        saw_rotation = False
        for external in range(4):
            engine = riichi_py.Arena(games=3, seed=1,
                                     bot_places=[p for p in range(4) if p != external])
            for _ in range(4000):
                if engine.all_finished():
                    break
                seats = np.frombuffer(engine.seats(), dtype=np.uint8)
                players = np.frombuffer(engine.seat_players(), dtype=np.uint8).reshape(3, 4)
                live = np.flatnonzero(seats != 0xFF)
                self.assertGreater(len(live), 0)
                np.testing.assert_array_equal(players[live, seats[live]], external)
                saw_rotation |= bool(np.any(players != np.arange(4)))
                engine.step(np.frombuffer(engine.teacher(), dtype=np.uint8).tolist())
            self.assertTrue(engine.all_finished())
        self.assertTrue(saw_rotation)

    def test_score_loop_and_duplicate_only_ask_the_assigned_player(self):
        for place in range(4):
            policy = NativeTeacher()
            scores, hands = selfplay.evaluate_games(policy, 1, 1, 'cpu', place=place)
            self.assertEqual(set(policy.owners), {place})
            self.assertEqual(scores.shape, (1, 4))
            self.assertGreater(hands, 0)
        policy = NativeTeacher()
        result = arena.duplicate(policy, 1, 1, 'cpu')
        self.assertEqual(set(policy.owners), {0, 1, 2, 3})
        self.assertEqual(result['games_total'], 4)
        self.assertTrue(all(row['hands'] > 0 for row in result['by_seat']))

    def test_arena_cli_accepts_both_checkpoint_action_layouts(self):
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with tempfile.TemporaryDirectory() as folder:
            for planes, actions in [(riichi_py.PLANES, riichi_py.ACTIONS), (MORTAL_PLANES, 46)]:
                with self.subTest(actions=actions):
                    torch.manual_seed(0)
                    net = PolicyValueNet(8, 1, planes, actions=actions).eval()
                    path = Path(folder) / f'policy-{actions}.pt'
                    torch.save({'model': net.state_dict(), **net.payload_fields(), 'generation': 0}, path)
                    run = subprocess.run(
                        [sys.executable, '-m', 'neural.arena', str(path), '--games', '1',
                         '--seed', '1', '--device', 'cpu'],
                        cwd=ROOT, env=env, text=True, capture_output=True, timeout=120,
                    )
                    self.assertEqual(run.returncode, 0, run.stderr)
                    report = json.loads(run.stdout)
                    self.assertEqual(report['games_total'], 4)
                    self.assertEqual(len(report['by_seat']), 4)
                    self.assertTrue(1 <= report['placement'] <= 4)
                    self.assertTrue(0 <= report['wins'] <= 1)
                    self.assertTrue(all(row['hands'] > 0 for row in report['by_seat']))

    def test_shared_native_search_reward_fixtures(self):
        rows = np.loadtxt(ROOT / 'engine/riichi-core/tests/fixtures/placement-rewards.csv', delimiter=',')
        scores, expected = rows[:, :4], rows[:, 4:]
        for order in itertools.permutations(range(4)):
            order = list(order)
            np.testing.assert_allclose(
                placement_rewards(scores[:, order], riichi_py.PLACEMENT_VALUE), expected[:, order],
            )

    def test_duplicate_rejects_truncation_and_invalid_player(self):
        with self.assertRaises(IncompleteGamesError):
            arena.duplicate(NativeTeacher(), 1, 1, 'cpu', max_steps=1)
        for place in (-1, 4, True, .5):
            with self.assertRaises(ValueError):
                selfplay.evaluate_games(NativeTeacher(), 1, 1, 'cpu', place=place)
