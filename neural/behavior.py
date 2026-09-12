"""Likelihood of the policy/uniform exploration mixture, shared by actor and PPO.

The coefficient is per recorded decision: a reach declaration may explore while
its subsequent discard does not. The learner must not substitute one scalar for
both stages. Value targets describe continuations under this same behaviour.
"""
from __future__ import annotations

import math
import torch


def validate_exploration(epsilon: float) -> None:
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("exploration must be finite and between zero and one")


def action_log_prob(logits: torch.Tensor, legal: torch.Tensor,
                    actions: torch.Tensor, epsilon: float | torch.Tensor = 0.0) -> torch.Tensor:
    """The actual behaviour likelihood; epsilon zero preserves raw PPO exactly."""
    if legal.dtype != torch.bool or legal.shape != logits.shape:
        raise ValueError("behaviour logits require a matching boolean legal mask")
    distribution = torch.distributions.Categorical(logits=logits.float().masked_fill(~legal, -torch.inf))
    raw = distribution.log_prob(actions)
    if not isinstance(epsilon, torch.Tensor):
        validate_exploration(epsilon)
        if epsilon == 0:
            return raw
    share = torch.as_tensor(epsilon, device=logits.device, dtype=torch.float32)
    if share.ndim == 0:
        share = share.expand_as(raw)
    if share.shape != raw.shape:
        raise ValueError("exploration coefficients must have one entry per decision")
    if not torch.isfinite(share).all() or torch.any((share < 0) | (share > 1)):
        raise ValueError("exploration coefficients must be finite and in [0, 1]")
    counts = legal.sum(dim=1)
    if legal.shape != logits.shape or torch.any(counts == 0):
        raise ValueError("every behaviour row needs a legal-action mask")
    mixed = (1 - share) * raw.exp() + share / counts.to(raw.dtype)
    # The unused log branch must also have finite derivatives at epsilon zero.
    safe = mixed.clamp_min(torch.finfo(mixed.dtype).tiny).log()
    return torch.where(share == 0, raw, safe)
