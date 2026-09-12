"""Evaluation and learning use engine legality before scores are masked."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from neural import zoo, combined, mortal_model, mortal_learner
from neural.model import PolicyValueNet, MORTAL_PLANES
from neural.observe import Planes


class ConflictingViews:
    def __init__(self):
        self.reached = False
        self.observer = SimpleNamespace(follower=self)
        self.planes = Planes.from_follower(np.array([0, 0]), [], [])

    def tell(self, game, player, event):
        self.reached = True

    def sparse_and_masks(self, rows, players, fresh=False):
        # Follower refuses reach before declaration, and tile 1 afterwards.
        mask = np.zeros((len(rows), 46), dtype=bool)
        mask[:, 0] = True
        return self.planes, mask

    def encode(self, who):
        return np.array([0, 0]), [], [], np.array([[True] + [False] * 45])


class EvaluationMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_evaluation_adapters_keep_core_legal_reach_and_second_discard(self):
        for kind in ('ours', 'mortal', 'combined'):
            with self.subTest(kind=kind):
                views = ConflictingViews()
                seen = []
                def score(planes, allowed):
                    seen.append(allowed.cpu().numpy())
                    logits = torch.zeros_like(allowed, dtype=torch.float32)
                    logits[:, 37] = 20
                    logits[:, 1] = 10
                    return logits.masked_fill(~allowed, -torch.inf), torch.zeros(len(allowed))
                net = PolicyValueNet(8, 1, MORTAL_PLANES, actions=46)
                if kind == 'ours':
                    player = zoo.MortalSpacePlayer(net, 'cpu')
                    player.forward = score
                elif kind == 'mortal':
                    with patch.object(zoo.mortal_model, 'load', return_value=object()):
                        player = zoo.MortalPlayer('unused', 'cpu')
                    player.forward = lambda planes, mask: score(planes, mask)[0]
                else:
                    player = combined.Combined(net, mortal_model.build(8, 1))
                    player.forward = score
                legal = np.zeros((1, 78), dtype=bool)
                legal[0, [0, 34, 35]] = True
                result = player.choose(views, np.array([0]), np.array([0]), legal)
                self.assertEqual(result.tolist(), [35])
                self.assertTrue(views.reached)
                self.assertTrue(seen[0][0, 37])
                self.assertEqual(np.flatnonzero(seen[1][0]).tolist(), [0, 1])

    def test_learner_also_keeps_every_core_legal_second_discard(self):
        views = ConflictingViews()
        legal = np.zeros((1, 78), dtype=bool)
        legal[0, [0, 34, 35]] = True
        def score(planes, allowed):
            logits = torch.zeros_like(allowed, dtype=torch.float32)
            logits[:, 37] = 20
            logits[:, 1] = 10
            return logits.masked_fill(~allowed, -torch.inf)
        result, records = mortal_learner.decide_in_mortal_space(
            score, views, np.array([0]), np.array([0]), legal, greedy=True, device='cpu')
        self.assertEqual(result.tolist(), [35])
        self.assertEqual(records.actions.tolist(), [37, 1])


if __name__ == '__main__':
    unittest.main()
