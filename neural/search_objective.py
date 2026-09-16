"""Opt-in conservative fitting of existing, immutable search-replay targets.

Search agreement is not evidence that the actor should become more certain.
Preserve its distribution on agreement rows, and optionally penalize forward
KL from the frozen collecting actor on every row. This is a soft regularizer,
not a hard trust region or evidence that the search teacher is stronger.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .search_replay import action_contract, student_value_head


def validate_policy_options(policy_mode: str, actor_kl: float) -> None:
    if policy_mode not in ("all", "changed"):
        raise ValueError("policy_mode must be 'all' or 'changed'")
    if type(actor_kl) not in (int, float) or not math.isfinite(actor_kl) or actor_kl < 0:
        raise ValueError("actor_kl must be finite and nonnegative")


@dataclass(frozen=True)
class PolicyTerms:
    loss: torch.Tensor
    cross_entropy: torch.Tensor
    actor_kl: torch.Tensor
    changed_fraction: torch.Tensor


def policy_terms(logits: torch.Tensor, actor: torch.Tensor,
                 target: torch.Tensor, legal: torch.Tensor,
                 aliases: torch.Tensor, *, policy_mode: str = "changed",
                 actor_kl: float = 1.0) -> PolicyTerms:
    """Row-mean policy objective with detached, legal frozen distributions.

    A row is changed only when none of the selected engine move's policy
    aliases has maximal actor probability. Ties are conservatively preserved;
    red/plain aliases are not mistaken for different engine moves. Riichi's
    declaration and conditional tile rows are compared separately.

    'changed' retains the stored improvement target on changed rows and uses
    the frozen actor on all others. It does NOT divide by the number changed:
    an all-agreement batch remains defined, and splitting a batch into unequal
    replay shards cannot accidentally change its weighting.
    """
    validate_policy_options(policy_mode, actor_kl)
    if (logits.ndim != 2 or not len(logits) or logits.shape[1] == 0
            or any(t.shape != logits.shape for t in (actor, target, legal, aliases))
            or legal.dtype != torch.bool or aliases.dtype != torch.bool
            or not logits.is_floating_point()
            or not actor.is_floating_point() or not target.is_floating_point()
            or any(t.device != logits.device for t in (actor, target, legal, aliases))):
        raise ValueError("Policy inputs need matching nonempty shapes, devices and dtypes")
    if (not legal.any(dim=1).all() or not aliases.any(dim=1).all()
            or torch.any(aliases & ~legal)):
        raise ValueError("Every row needs legal actions and a legal selected-action alias")
    actor, target = actor.detach().float(), target.detach().float()
    for probabilities in (actor, target):
        if (not torch.isfinite(probabilities).all() or torch.any(probabilities < 0)
                or torch.any(probabilities[~legal] != 0)
                or not torch.allclose(probabilities.sum(dim=1),
                                      torch.ones(len(logits), device=logits.device),
                                      atol=1e-5, rtol=1e-5)):
            raise ValueError("Frozen policy distributions must be finite, normalized and legal")
    if not torch.isfinite(logits[legal]).all():
        raise FloatingPointError("Nonfinite legal policy logits")
    changed = actor.masked_fill(~aliases, -torch.inf).amax(dim=1) < actor.amax(dim=1)
    fitted = torch.where(changed[:, None], target, actor) if policy_mode == "changed" else target
    # Mask BEFORE normalization; even enormous/NaN illegal logits have no effect.
    logp = torch.log_softmax(logits.float().masked_fill(~legal, -torch.inf), dim=1)
    safe_logp = torch.where(legal, logp, torch.zeros_like(logp))
    cross_entropy = -(fitted * safe_logp).sum(dim=1).mean()
    # Zero actor mass contributes zero KL, including underflowed legal actions.
    actor_logp = torch.where(actor > 0, actor, torch.ones_like(actor)).log()
    divergence = (actor * (actor_logp - safe_logp)).sum(dim=1).mean()
    loss = cross_entropy + actor_kl * divergence
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite conservative policy loss")
    return PolicyTerms(loss, cross_entropy, divergence, changed.float().mean())


def replay_loss(replay, net, rows: np.ndarray, device: str = "cpu",
                value_weight: float = .5, *, policy_mode: str = "changed",
                actor_kl: float = 1.0) -> torch.Tensor:
    """Fit a minibatch without changing replay bytes, rewards or the value head."""
    validate_policy_options(policy_mode, actor_kl)
    if type(value_weight) not in (int, float) or not math.isfinite(value_weight) or value_weight < 0:
        raise ValueError("value_weight must be finite and nonnegative")
    rows = np.asarray(rows)
    if (rows.ndim != 1 or rows.dtype != np.int64 or not len(rows)
            or np.any((rows < 0) | (rows >= replay.metadata["rows"]))):
        raise ValueError("A minibatch needs valid int64 row indices")
    a = replay.arrays
    expected, aliases = action_contract(a["engine_legal"][rows], a["engine_action"][rows], a["stage"][rows])
    if not np.array_equal(expected, a["legal"][rows]):
        raise ValueError("Replay policy masks disagree with the selected engine actions")
    x = replay.observations().rows(rows).dense(device)
    tensor = lambda name: torch.from_numpy(a[name][rows].copy()).to(device)
    legal, actor, target, returns = (tensor(name) for name in (
        "legal", "actor_policy", "policy_target", "returns"))
    logits, value, _ = net.everything(x, legal)
    if hasattr(net, "value_only"):
        value = net.value_only(x, head=student_value_head(replay.metadata))
    if logits.shape != target.shape or value.shape != returns.shape:
        raise ValueError("Learner outputs do not match the replay action/value contract")
    terms = policy_terms(logits, actor, target, legal,
                         torch.from_numpy(aliases).to(device),
                         policy_mode=policy_mode, actor_kl=actor_kl)
    total = terms.loss + value_weight * torch.nn.functional.mse_loss(value.float(), returns)
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite search-supervision loss")
    return total
