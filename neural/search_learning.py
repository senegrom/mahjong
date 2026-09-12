"""Frozen search-policy targets and a finite masked cross-entropy objective."""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def improvement_targets(actor: torch.Tensor, actions: torch.Tensor,
                        legal: torch.Tensor, improve: float) -> torch.Tensor:
    if not math.isfinite(improve) or not 0 <= improve <= 1:
        raise ValueError("improve must be in [0, 1]")
    if actor.ndim != 2 or actor.shape != legal.shape or actions.shape != (len(actor),):
        raise ValueError("actor probabilities, legal masks and search actions do not align")
    if legal.dtype != torch.bool or not legal.any(dim=1).all():
        raise ValueError("each target needs a legal action")
    if not torch.isfinite(actor).all() or torch.any(actor < 0) or torch.any(actor[~legal] != 0):
        raise ValueError("actor probabilities must be finite, nonnegative and legal")
    if not torch.allclose(actor.sum(dim=1), torch.ones(len(actor), device=actor.device), atol=1e-5):
        raise ValueError("actor probabilities must sum to one")
    if actions.dtype != torch.int64 or torch.any((actions < 0) | (actions >= actor.shape[1])):
        raise ValueError("search actions are outside the action space")
    if not legal.gather(1, actions[:, None]).all():
        raise ValueError("search targets must name legal actions")
    target = actor.detach().clone() * (1 - improve)
    target.scatter_add_(1, actions[:, None], torch.full((len(actor), 1), improve,
                        device=actor.device, dtype=actor.dtype))
    return target


def masked_policy_loss(logits: torch.Tensor, target: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != legal.shape:
        raise ValueError("policy loss inputs must have matching shapes")
    # Zero mass times a masked -inf is NaN, not zero. Remove masked terms first.
    log_probs = torch.log_softmax(logits.float(), dim=1)
    log_probs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
    return -(target * log_probs).sum(dim=1).mean()
