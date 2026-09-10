"""CPU regression tests using the actual networks and both native engines."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import riichi_py
from libriichi.follow import Follower

from neural import combined, imitate, mortal_model, zoo
from neural.model import ENGINE_PLANES, MORTAL_PLANES, PolicyValueNet
from neural.observe import Planes


class BoundedArena:
    """Keep collection tests to two real decisions, without playing a match."""

    native = riichi_py.Arena

    def __init__(self, games, seed):
        self.arena = self.native(games=games, seed=seed)
        self.games = games
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.arena, name)

    def seats(self):
        return bytes([0xFF] * self.games) if self.steps >= 2 else self.arena.seats()

    def step(self, actions):
        self.arena.step(actions)
        self.steps += 1


def ready_follower():
    """A complete dealer hand that can either discard or declare riichi."""
    hand = ['1m', '2m', '3m', '4m', '5m', '6m', '7p', '8p', '9p', '2s', '3s', 'P', 'P']
    wall = [f'{rank}{suit}' for suit in 'mps' for rank in range(1, 10) for _ in range(4)]
    wall += [tile for tile in ['E', 'S', 'W', 'N', 'P', 'F', 'C'] for _ in range(4)]
    for tile in hand + ['1s', 'C']:
        wall.remove(tile)
    event = dict(type='start_kyoku', bakaze='E', dora_marker='C', kyoku=1,
                 honba=0, kyotaku=0, oya=0, scores=[30000] * 4,
                 tehais=[hand, wall[:13], wall[13:26], wall[26:39]])
    follower = Follower(1)
    follower.feed([[json.dumps(event), json.dumps(dict(type='tsumo', actor=0, pai='1s'))]])
    return follower


class FollowerViews:
    def __init__(self, follower):
        self.observer = SimpleNamespace(follower=follower)

    def sparse_and_masks(self, rows, players):
        who = [(int(g), int(p)) for g, p in zip(rows, players)]
        indptr, indices, values, mask = self.observer.follower.encode(who)
        return Planes.from_follower(indptr, indices, values), np.asarray(mask, dtype=bool)


class ReachTeacher:
    actions = 46

    def forward(self, planes, mask):
        # Half the first decision's mass is an ordinary discard. The other
        # half is riichi, whose separate tile choice has probabilities .2/.8.
        logits = torch.full(mask.shape, -torch.inf)
        if mask[:, zoo.MORTAL_RIICHI].any():
            logits[:, 0] = np.log(0.5)
            logits[:, zoo.MORTAL_RIICHI] = np.log(0.5)
        else:
            logits[:, 0] = np.log(0.2)
            logits[:, 1] = np.log(0.8)
        return logits.masked_fill(~mask, -torch.inf), torch.zeros(len(mask))


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_distribution(self, odds, legal):
        self.assertEqual(odds.shape, legal.shape)
        self.assertTrue(np.isfinite(odds).all())
        self.assertTrue((odds >= 0).all())
        self.assertTrue((odds[~legal] == 0).all())
        np.testing.assert_allclose(odds.sum(axis=1), 1, atol=1e-6)

    def test_riichi_mass_uses_the_second_policy_without_changing_the_follower(self):
        follower = ready_follower()
        who = [(0, p) for p in range(4)]
        before = [np.array(x, copy=True) for x in follower.encode(who)]
        legal = np.zeros((1, 78), dtype=bool)
        legal[0, [0, 34, 35]] = True
        odds = imitate.teacher_distribution(ReachTeacher(), FollowerViews(follower),
                                            np.array([0]), np.array([0]), legal)
        self.assert_distribution(odds, legal)
        np.testing.assert_allclose(odds[0, [0, 34, 35]], [0.5, 0.1, 0.4], atol=1e-6)
        self.assertEqual(odds.argmax(axis=1).tolist(), [0])
        for original, current in zip(before, follower.encode(who)):
            np.testing.assert_array_equal(original, current)
        follower.feed([[json.dumps(dict(type='dahai', actor=0, pai='1m', tsumogiri=False))]])
        # A real reach still arrives once and has exactly the previewed view.
        follower = ready_follower()
        preview = follower.encode([(0, 0)], after_reach=True)
        follower.feed([[json.dumps(dict(type='reach', actor=0))]])
        for expected, current in zip(preview, follower.encode([(0, 0)])):
            np.testing.assert_array_equal(expected, current)

    def test_mortal_calls_red_fives_and_untranslatable_rows_stay_in_engine_space(self):
        # Engine order: the three chii variants, pon, each kan, ron, tsumo,
        # pass, a red-five alias, and a position Mortal cannot express.
        targets = [71, 72, 73, 74, 75, 76, 77, 69, 68, 70, 4, 2]
        mortal = [40, 39, 38, 41, 42, 42, 42, 43, 43, 45, 34, 44]
        legal = np.zeros((len(targets), 78), dtype=bool)
        legal[np.arange(len(targets)), targets] = True
        own = np.zeros((len(targets), 46), dtype=bool)
        own[np.arange(len(targets)), mortal] = True
        planes = Planes.from_follower(np.zeros(len(targets) + 1), [], [])
        views = SimpleNamespace(sparse_and_masks=lambda rows, players: (planes, own))
        teacher = SimpleNamespace(actions=46, forward=lambda p, m:
                                  (torch.zeros(m.shape).masked_fill(~m, -torch.inf), None))
        odds = imitate.teacher_distribution(teacher, views, np.arange(len(targets)),
                                            np.zeros(len(targets), dtype=int), legal)
        self.assert_distribution(odds, legal)
        self.assertEqual(odds.argmax(axis=1).tolist(), targets)

    def test_checkpoint_teachers_collect_and_train_a_browser_student(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = PolicyValueNet(8, 1, ENGINE_PLANES)
            ours = PolicyValueNet(8, 1, MORTAL_PLANES, actions=46)
            fused = combined.Combined(ours, mortal_model.build(8, 1))
            fused.mortal_config = {'control': {'version': 4},
                                   'resnet': {'conv_channels': 8, 'num_blocks': 1}}
            teachers = {
                'engine': {'model': engine.state_dict(), **engine.payload_fields()},
                'reheaded': {'model': ours.state_dict(), **ours.payload_fields()},
                'fused': fused.state(),
            }
            for name, payload in teachers.items():
                with self.subTest(teacher=name):
                    path = root / f'{name}.pt'
                    torch.save(payload, path)
                    teacher = zoo.load_player(path, 'cpu')
                    self.assertEqual(teacher.channels, 8)
                    with patch.object(imitate.riichi_py, 'Arena', BoundedArena):
                        planes, masks, labels, _held, odds = imitate.collect(
                            1, 81, teacher, student='engine')
                    self.assertEqual(planes.dense('cpu').shape[1:], (ENGINE_PLANES, 34))
                    self.assert_distribution(odds, masks)
                    self.assertTrue(masks[np.arange(len(labels)), labels].all())
                    argv = ['imitate', '--teacher', str(path), '--student', 'engine',
                            '--channels', '8', '--blocks', '1', '--rounds', '1',
                            '--games', '1', '--batch', '2', '--out', str(root / name)]
                    with patch('sys.argv', argv), patch.object(imitate.riichi_py, 'Arena', BoundedArena), \
                         patch.object(imitate.selfplay, 'measure', return_value={
                             'placement': 2.5, 'score': 0., 'wins': 0.}), \
                         contextlib.redirect_stdout(io.StringIO()):
                        imitate.main()
                    saved = torch.load(root / name / 'latest.pt', weights_only=True)
                    self.assertEqual(saved['actions'], 78)
                    self.assertEqual(saved['planes'], ENGINE_PLANES)
                    self.assertTrue(all(torch.isfinite(t).all() for t in saved['model'].values()))
                    record = json.loads((root / name / 'log.jsonl').read_text())
                    self.assertGreaterEqual(record['positions'], 2)
                    self.assertTrue(np.isfinite(record['loss']))


if __name__ == '__main__':
    unittest.main()
