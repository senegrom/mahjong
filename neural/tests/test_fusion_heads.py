"""Can the fusion's corrections change the answers they correct?

A head that cannot move its output is not a head, and this kind of fault
is invisible: the parameters exist, the checkpoint carries them, the loss
goes down because something *else* is learning, and nothing ever says that
one of the pieces is inert.

The reader's correction was that for the whole of the leashed run. Mortal's
vector was spread evenly across the thirty-four tiles and put through a
one-by-one convolution, which gives every tile the same number. A reading
is a distribution over tiles, so a constant added to every logit leaves the
softmax exactly where it was: measured, a logit moved by 244 and a
probability by 9e-16.

    python -m unittest neural.tests.test_fusion_heads -v
"""

from __future__ import annotations

import unittest

import torch

from neural.combined import Fuse
from neural.model import OPPONENTS, POSITIONS


def woken(channels: int = 320) -> Fuse:
    """A fusion whose corrections have real weights.

    They are initialised at zero so that a resumed checkpoint reads exactly
    as it did before, which is right — and which means a test that leaves
    them there proves only that zero times anything is zero.
    """
    torch.manual_seed(0)
    fuse = Fuse(channels=channels).double()
    torch.nn.init.normal_(fuse.hands_out.weight, std=0.5)
    torch.nn.init.normal_(fuse.hands_out.bias, std=0.5)
    torch.nn.init.normal_(fuse.value_fix[-1].weight, std=0.5)
    torch.nn.init.normal_(fuse.value_fix[-1].bias, std=0.5)
    return fuse


class TheReaderSCorrectionIsReal(unittest.TestCase):
    def setUp(self):
        self.fuse = woken()
        self.phi = torch.randn(4, 1024, dtype=torch.double)
        self.guessed = torch.randn(4, OPPONENTS, POSITIONS, dtype=torch.double)

    def test_it_changes_what_is_believed(self):
        read = self.fuse.read_hands(self.phi, self.guessed)
        before = torch.softmax(self.guessed, dim=2)
        after = torch.softmax(read, dim=2)
        self.assertGreater(
            float((after - before).abs().max()), 1e-3,
            "the correction moved the logits but not the distribution, which is "
            "what adding the same number to every tile does",
        )

    def test_mortal_s_vector_is_what_moves_it(self):
        """Not merely tile-dependent: dependent on the thing it corrects
        by. A correction that varies by tile only through the base reading
        still cannot let Mortal name a tile."""
        one = torch.softmax(self.fuse.read_hands(self.phi, self.guessed), dim=2)
        other = torch.softmax(
            self.fuse.read_hands(torch.randn_like(self.phi), self.guessed), dim=2
        )
        self.assertGreater(float((one - other).abs().max()), 1e-3)

    def test_a_belief_target_reaches_the_correction(self):
        read = self.fuse.read_hands(self.phi, self.guessed)
        wanted = torch.softmax(torch.randn_like(self.guessed), dim=2)
        loss = -(wanted * torch.log_softmax(read, dim=2)).sum(dim=2).mean()
        loss.backward()
        largest = max(
            float(p.grad.abs().max())
            for p in self.fuse.hands_out.parameters()
            if p.grad is not None
        )
        self.assertGreater(largest, 1e-4, "nothing can train a head nothing reaches")

    def test_it_starts_silent_so_a_resumed_run_is_unchanged(self):
        torch.manual_seed(0)
        fresh = Fuse(channels=320).double()
        read = fresh.read_hands(self.phi, self.guessed)
        self.assertLess(float((read - self.guessed).abs().max()), 1e-12)


class TheCriticSCorrectionIsReal(unittest.TestCase):
    def test_it_changes_what_a_position_is_worth(self):
        fuse = woken()
        phi = torch.randn(4, 1024, dtype=torch.double)
        value = torch.randn(4, dtype=torch.double)
        judged = fuse.judge(phi, value)
        self.assertGreater(float((judged - value).abs().max()), 1e-3)

    def test_it_starts_silent(self):
        torch.manual_seed(0)
        fuse = Fuse(channels=320).double()
        phi = torch.randn(4, 1024, dtype=torch.double)
        value = torch.randn(4, dtype=torch.double)
        self.assertLess(float((fuse.judge(phi, value) - value).abs().max()), 1e-12)


if __name__ == "__main__":
    unittest.main()
