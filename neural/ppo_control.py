"""Minibatch policy-drift checks and bounded baseline evaluation.

PPO clipping is not a hard trust region. The optional guard stops *before*
applying another update when the sampled reverse KL exceeds target_kl.
See https://spinningup.openai.com/en/latest/algorithms/ppo.html.
"""
from __future__ import annotations

import math
import torch


def add_training_controls(parser) -> None:
    parser.add_argument('--target-kl', type=float, default=0.0,
                        help='stop further PPO updates above this sampled KL; 0 disables the guard')
    parser.add_argument('--baseline-batch', type=int, default=None,
                        help='maximum value-baseline rows per forward; defaults to --batch')


def baseline_batch_size(decisions: int, batch_size: int, requested: int | None = None) -> int:
    """Never pad a tiny rollout to thousands of dense 1,012-plane rows."""
    for name, value in [('decisions', decisions), ('batch', batch_size),
                        ('baseline_batch', requested if requested is not None else batch_size)]:
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    return min(decisions, requested if requested is not None else batch_size)


class PolicyDrift:
    """One guard per collected round; stop all remaining epochs on excess drift.

    exp(delta)-1-delta is nonnegative and its expectation under the old policy
    is KL(old || new) when both policies have the same legal support. This is
    a sampled diagnostic, not exact full-policy KL or a rollback guarantee.
    All forwards and recorded probabilities must use the same precision.
    """
    def __init__(self, target_kl: float = 0.0) -> None:
        if not math.isfinite(target_kl) or target_kl < 0:
            raise ValueError('target_kl must be finite and nonnegative')
        self.target = target_kl
        self.stopped = False
        # Kept as tensors so that, with the guard off, nothing here waits
        # on the card: every other metric is read once at the round's end,
        # and a scalar synchronised on every minibatch would idle the card
        # through the next step's host work. The guard on has to read it,
        # since the answer decides whether the step is taken.
        self._last: torch.Tensor | float = 0.0
        self._maximum: torch.Tensor | float = 0.0

    @torch.no_grad()
    def check(self, old_log_prob: torch.Tensor, new_log_prob: torch.Tensor) -> bool:
        if old_log_prob.shape != new_log_prob.shape or old_log_prob.numel() == 0:
            raise ValueError('PPO probabilities need matching nonempty shapes')
        delta = new_log_prob.detach().float() - old_log_prob.detach().float()
        # expm1 avoids subtracting two near-equal numbers for small updates.
        estimate = (torch.expm1(delta) - delta).clamp_min(0).mean()
        if not self.target:
            self._last = estimate
            self._maximum = torch.maximum(torch.as_tensor(self._maximum, device=estimate.device), estimate)
            return False
        last = float(estimate)
        if not math.isfinite(last):
            raise FloatingPointError('Nonfinite PPO policy drift; no optimizer step was applied')
        self._last = last
        self._maximum = max(float(self._maximum), last)
        self.stopped |= last > self.target
        return self.stopped

    @property
    def last(self) -> float:
        return float(self._last)

    @property
    def maximum(self) -> float:
        return float(self._maximum)

    def metrics(self) -> dict:
        return {'target_kl': self.target, 'kl_early_stop': self.stopped,
                'sampled_kl_last': self.last, 'sampled_kl_max': self.maximum}
