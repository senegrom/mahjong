"""CPU regressions for checkpoint boundaries, using real Torch and AdamW."""

import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural.training_state import capture_random_state, restore_random_state

MODES = ["none", "mortal", "ours", "head", "mortal+head", "ours+head"]


class TrainingStateTests(unittest.TestCase):
    def setUp(self):
        self.cuda = patch("torch.cuda.is_available", return_value=False)
        self.cuda.start()
        self.addCleanup(self.cuda.stop)
        self.before = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, self.before)

    def test_repeated_single_generation_resumes_keep_the_schedule(self):
        continuous = restore_random_state(None, seed=20260908, generation=0, modes=MODES)
        expected = [str(continuous.choice(MODES)) for _ in range(24)]
        saved, actual = None, []
        for generation in range(24):
            drawer = restore_random_state(saved, seed=20260908, generation=generation, modes=MODES)
            actual.append(str(drawer.choice(MODES)))
            saved = capture_random_state(drawer)
        self.assertEqual(actual, expected)
        self.assertGreater(len(set(actual)), 1)

    def test_legacy_schedule_advances_instead_of_restarting(self):
        drawer = np.random.default_rng(20260908 + 99)
        for _ in range(7):
            drawer.choice(MODES)
        with self.assertWarnsRegex(RuntimeWarning, "Legacy checkpoint"):
            resumed = restore_random_state(None, seed=20260908, generation=7, modes=MODES)
        self.assertEqual([drawer.choice(MODES) for _ in range(12)],
                         [resumed.choice(MODES) for _ in range(12)])

    def test_cpu_training_matches_after_serialized_resume(self):
        def create():
            net = torch.nn.Linear(3, 2)
            return net, torch.optim.AdamW(net.parameters(), lr=0.01)

        def steps(net, optimizer, drawer, count):
            trace = []
            for _ in range(count):
                mode = str(drawer.choice(MODES))
                x = torch.randn(5, 3)
                target = torch.randn(5, 2)
                order = torch.randperm(5)
                optimizer.zero_grad()
                loss = (net(x[order]) - target[order]).square().mean()
                loss.backward()
                optimizer.step()
                trace.append((mode, float(loss.detach())))
            return trace

        torch.manual_seed(12)
        net, optimizer = create()
        drawer = restore_random_state(None, seed=12, generation=0, modes=MODES)
        steps(net, optimizer, drawer, 3)
        buffer = io.BytesIO()
        torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "random_state": capture_random_state(drawer)}, buffer)
        expected = steps(net, optimizer, drawer, 8)
        expected_weights = {k: v.clone() for k, v in net.state_dict().items()}
        # Constructors and other setup consume random numbers before restore.
        torch.manual_seed(999)
        resumed, resumed_optimizer = create()
        torch.rand(100)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        resumed.load_state_dict(saved["model"])
        resumed_optimizer.load_state_dict(saved["optimizer"])
        drawer = restore_random_state(saved["random_state"], seed=12, generation=3, modes=MODES)
        self.assertEqual(steps(resumed, resumed_optimizer, drawer, 8), expected)
        for key, value in resumed.state_dict().items():
            self.assertTrue(torch.equal(value, expected_weights[key]), key)

    def test_snapshots_do_not_mutate_and_can_be_restored_twice(self):
        drawer = restore_random_state(None, seed=2, generation=0, modes=MODES)
        saved = capture_random_state(drawer)
        first = restore_random_state(saved, seed=2, generation=0, modes=MODES)
        expected = [first.choice(MODES), torch.rand(4)]
        first.choice(MODES, size=10)
        second = restore_random_state(saved, seed=2, generation=0, modes=MODES)
        self.assertEqual(second.choice(MODES), expected[0])
        self.assertTrue(torch.equal(torch.rand(4), expected[1]))

    def test_cuda_states_are_forwarded_and_mismatches_rejected(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_rng_state_all", return_value=[torch.tensor([1], dtype=torch.uint8)]), \
             patch("torch.cuda.set_rng_state_all") as restore, \
             patch("torch.cuda.device_count", return_value=1):
            saved = capture_random_state(np.random.default_rng(9))
            restore_random_state(saved, seed=9, generation=1, modes=MODES)
            restore.assert_called_once()
            self.assertTrue(torch.equal(restore.call_args.args[0][0], saved["cuda"][0]))
            with patch("torch.cuda.device_count", return_value=2):
                with self.assertRaisesRegex(ValueError, "device count"):
                    restore_random_state(saved, seed=9, generation=1, modes=MODES)

    def test_empty_modes_and_negative_generations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "--fixed"):
            restore_random_state(None, seed=1, generation=0, modes=[])
        with self.assertRaisesRegex(ValueError, "negative"):
            restore_random_state(None, seed=1, generation=-1, modes=MODES)


if __name__ == "__main__":
    unittest.main()
