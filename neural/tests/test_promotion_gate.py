"""Promotion needs evidence, not a lucky smoothed heuristic-bot score."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import riichi_py

from neural import gate
from neural.model import PolicyValueNet


class GateTests(unittest.TestCase):
    def test_equal_policies_and_small_samples_do_not_promote(self):
        for n in (1, 10, 128, 1000):
            result=gate.assess([2.5]*n,[2.5]*n)
            self.assertFalse(result['promote'])
            self.assertLess(result['lower_bound'],0) # zero variance is not certainty
        self.assertFalse(gate.assess([1]*10,[4]*10)['promote'])

    def test_mean_bound_and_correlated_seating_count(self):
        result=gate.assess([1.5]*512,[3.5]*512)
        self.assertTrue(result['promote'])
        self.assertEqual(result['independent_deals'],512)
        self.assertEqual(result['games_total'],4096)
        radius=3*math.sqrt(math.log(40)/(2*512))
        self.assertAlmostEqual(result['lower_bound'],1-radius)
        reverse=gate.assess([3.5]*512,[1.5]*512)
        self.assertFalse(reverse['promote']);self.assertEqual(reverse['mean_placement_improvement'],-1)

    def test_role_asymmetry_does_not_pass_and_attempts_spend_alpha(self):
        self.assertFalse(gate.assess([2.]*512,[2.]*512)['promote'])
        a=gate.assess([2.]*512,[3.]*512,attempt=1)
        b=gate.assess([2.]*512,[3.]*512,attempt=10)
        self.assertLess(b['attempt_alpha'],a['attempt_alpha'])
        self.assertGreater(b['confidence_radius'],a['confidence_radius'])
        self.assertLessEqual(sum(.05/(i*(i+1)) for i in range(1,10000)),.05)

    def test_invalid_data_and_options_fail(self):
        for a,b in [([],[]),([1],[1,2]),([np.nan],[2]),([0],[2]),([4.1],[2]),([[2]],[[2]])]:
            with self.assertRaises(ValueError):gate.assess(a,b)
        for options in [{'confidence':1},{'attempt':0},{'attempt':True},
                        {'minimum_deals':1},{'minimum_edge':np.nan}]:
            with self.assertRaises(ValueError):gate.assess([2],[3],**options)

    def test_real_small_checkpoint_gate_leaves_models_unchanged(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'candidate.pt'
            net=PolicyValueNet(8,1,riichi_py.PLANES,actions=riichi_py.ACTIONS)
            torch.save({'model':net.state_dict(),**net.payload_fields(),'generation':1},path)
            original=path.read_bytes()
            report=gate.compare(path,path,games=1,seed=1)
            self.assertFalse(report['promote']);self.assertFalse(report['checkpoint_written'])
            self.assertEqual(report['games_total'],8)
            self.assertEqual(report['mean_placement_improvement'],0)
            self.assertEqual(report['candidate']['sha256'],report['champion']['sha256'])
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_truncation_cannot_be_reported_as_gate_evidence(self):
        from neural.outcomes import IncompleteGamesError
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'candidate.pt'
            net=PolicyValueNet(8,1,riichi_py.PLANES,actions=riichi_py.ACTIONS)
            torch.save({'model':net.state_dict(),**net.payload_fields()},path)
            with self.assertRaises(IncompleteGamesError):
                gate.compare(path,path,games=1,seed=1,max_steps=1)
