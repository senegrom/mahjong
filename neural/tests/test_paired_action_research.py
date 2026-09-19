"""Counterfactual actions are public predictions, not labels of hindsight winners."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from neural import model, paired_actions, league_gate, placement
from neural.checkpoints import atomic_save
from neural.seed_ledger import SeedLedger
from neural.tests.test_selfplay_experiments import sparse


class PairedActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_shortlist_is_policy_ordered_with_stable_ties(self):
        logits = np.zeros((1, 46)); logits[0, 7] = 2; logits[0, 4] = 1
        legal = np.zeros((1, 46), dtype=bool); legal[0, [1, 4, 7]] = True
        moves, valid = paired_actions.shortlist(logits, legal, 4)
        self.assertEqual(moves[0, :3].tolist(), [7, 4, 1])
        self.assertEqual(valid[0].tolist(), [True, True, True, False])

    def test_difference_loss_is_offset_invariant_and_masks_padding(self):
        scores = torch.zeros(2, 46); scores[:, 2] = .5
        moves = torch.tensor([[0, 2, 3], [0, 2, 4]])
        wanted = torch.tensor([[0., .5, 100.], [-1., -.5, 100.]])
        valid = torch.tensor([[True, True, False], [True, True, False]])
        self.assertEqual(float(paired_actions.paired_loss(scores, moves, wanted, valid)), 0)
        self.assertEqual(float(paired_actions.paired_loss(scores + 8, moves, wanted, valid)), 0)

    def test_real_paired_passes_have_identical_roots_before_intervention(self):
        net = model.PolicyValueNet(8, 1, actions=46).eval()
        options = dict(games=1, seed=124, root_step=0, candidates=2, device='cpu')
        root, _ = paired_actions.play_pass(net, **options)
        game = int(root['games'][0]); action = int(root['candidates'][0, 1])
        other, _ = paired_actions.play_pass(net, forced={game: action}, **options)
        self.assertEqual(paired_actions.root_fingerprint(root), paired_actions.root_fingerprint(other))

    def test_ranking_selects_predictions_not_best_realized_payoff(self):
        ranker = paired_actions.ActionDifferences(8, 1).eval()  # zero, so keep policy's first action
        n = 20
        data = {'games': torch.arange(n), 'candidates': torch.tensor([[0, 1]] * n),
                'valid': torch.ones(n, 2, dtype=torch.bool), 'payoffs': torch.tensor([[0., 1.5]] * n)}
        result = paired_actions.measure(ranker, data, sparse(n), np.arange(n), device='cpu', batch=8)
        self.assertEqual(result['selected_placement_edge'], 0.)
        self.assertGreater(result['paired_mse'], 0.)

    def test_ranker_fit_has_grouped_validation_and_one_final_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); n = 100; planes = sparse(n)
            source = {'paired_version': 1, 'actor': {'sha256': 'a' * 64}, 'shortlist': 2,
                      'objective': 'placement-v2', 'games': torch.arange(n),
                      'players': torch.arange(n) % 4, 'legal': torch.ones(n, 46, dtype=torch.bool),
                      'candidates': torch.tensor([[0, 1]] * n), 'valid': torch.ones(n, 2, dtype=torch.bool),
                      'payoffs': torch.tensor([[0., .5]] * n),
                      'planes': {key: torch.from_numpy(getattr(planes, key)) for key in planes.ARRAYS}}
            atomic_save(source, root / 'data.pt')
            with patch.object(paired_actions, 'measure', wraps=paired_actions.measure) as measured:
                report = paired_actions.fit([root / 'data.pt'], root / 'head.pt', width=8, blocks=1,
                                            epochs=2, batch=16, device='cpu')
                self.assertEqual(measured.call_count, 3)  # two validation reads; exactly one test
                first, second, test = [set(call.args[3].tolist()) for call in measured.call_args_list]
                self.assertEqual(first, second); self.assertFalse(first & test)
            self.assertGreater(report['test']['roots'], 0)
            paired_actions.load_ranker(root / 'head.pt', 'cpu')

    def test_judge_selects_only_on_validation_and_tests_once(self):
        net = model.PolicyValueNet(8, 1, actions=46)
        positions = placement.Positions(sparse(100), np.tile([-1.5, -.5, .5, 1.5], 25), np.arange(100))
        def measured(head, actor, positions, rows, device):
            return {'rows': len(rows), 'explained': .1}
        with patch.object(placement, 'measure', side_effect=measured) as check:
            _, history = placement.train(positions, net, epochs=2, batch=16, keep_best=True)
        self.assertEqual(check.call_count, 3)
        validation = set(check.call_args_list[0].args[3])
        self.assertEqual(validation, set(check.call_args_list[1].args[3]))
        self.assertFalse(validation & set(check.call_args_list[2].args[3]))
        self.assertIn('test', history[-1]); self.assertEqual(history[-1]['selected_epoch'], 0)

    def test_failed_gate_attempt_burns_seeds_and_cannot_reset_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'actor.pt'; ledger = root / 'evidence.json'
            atomic_save({'generation': 1}, source)
            with patch.object(league_gate.gate, 'compare', side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    league_gate.run(source, source, ledger=ledger, games=128)
            self.assertEqual(SeedLedger(ledger).snapshot()['counts']['promotion'], 1)
            def compare(*paths, **kwargs):
                return {'promote': False, 'candidate': {'sha256': 'a'*64}, 'champion': {'sha256': 'a'*64},
                        'attempt': kwargs['attempt'], 'seed': kwargs['seed'], 'checkpoint_written': False}
            with patch.object(league_gate.gate, 'compare', side_effect=compare):
                result = league_gate.run(source, source, ledger=ledger, games=128)
            self.assertEqual(result['attempt'], 2)
            self.assertFalse(result['checkpoint_written'])
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                league_gate.run(source, source, ledger=ledger, games=128, method='hoeffding')


if __name__ == '__main__': unittest.main()
