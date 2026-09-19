"""Masked PPO arithmetic with full-distribution KL and actor-only clipping."""
from __future__ import annotations

import math
import torch
from torch import nn


def logits(net, planes, legal):
    fast = getattr(net, "policy_only", None)
    return fast(planes, legal) if callable(fast) else net(planes, legal)[0]


def log_distribution(scores: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    if scores.shape != legal.shape or legal.dtype != torch.bool or not legal.any(dim=1).all():
        raise ValueError("policy needs one nonempty legal mask per row")
    if not torch.isfinite(scores[legal]).all():
        raise FloatingPointError("nonfinite legal policy scores")
    return torch.log_softmax(scores.float().masked_fill(~legal, -torch.inf), dim=1)


def full_kl(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor,
            legal: torch.Tensor) -> torch.Tensor:
    """KL(old || new), per row; illegal -inf entries never enter subtraction."""
    if old_log_probs.shape != legal.shape or new_log_probs.shape != legal.shape:
        raise ValueError("KL distributions do not match the legal mask")
    old = torch.where(legal, old_log_probs, 0.0)
    new = torch.where(legal, new_log_probs, 0.0)
    if not torch.isfinite(old).all() or not torch.isfinite(new).all():
        raise FloatingPointError("nonfinite legal log probabilities")
    weights = torch.where(legal, old.exp(), 0.0)
    return (weights * (old - new)).sum(dim=1).clamp_min(0)


def decision_categories(legal: torch.Tensor) -> dict[str, torch.Tensor]:
    """Disjoint categories of decision STATES, not the sampled action."""
    if legal.ndim != 2 or legal.shape[1] != 46:
        raise ValueError("diagnostics require Mortal's 46-action layout")
    reach = legal[:, 37]
    call = legal[:, 38:43].any(dim=1) & ~reach
    discard = legal[:, :37].any(dim=1) & ~reach & ~call
    return {"riichi": reach, "call": call, "discard": discard,
            "other": ~reach & ~call & ~discard}


@torch.no_grad()
def old_distributions(net, batch, *, batch_size, device, amp=False):
    """Recompute the complete frozen policy and verify its chosen-action likelihoods."""
    net.eval()
    result = torch.empty_like(batch.legal, dtype=torch.float32)
    for start in range(0, batch.decisions, batch_size):
        stop = min(start + batch_size, batch.decisions)
        legal = batch.legal[start:stop].to(device)
        planes = batch.observations.slice(start, stop).dense(device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
            scores = logits(net, planes, legal)
        result[start:stop] = log_distribution(scores, legal).cpu()
    said = result.gather(1, batch.actions[:, None]).squeeze(1)
    delta = (said - batch.log_probs).abs()
    tolerance = 0.05 if amp else 5e-4
    if not torch.isfinite(delta).all() or float(delta.max()) > tolerance:
        raise RuntimeError(f"rollout/update policy mismatch: max log-prob difference {float(delta.max()):.6g}")
    return result


def update(net, optimizer, batch, advantages, old, *, epochs, batch_size,
           ratio_clip, target_kl, entropy_weight, grad_clip, device, amp=False,
           reference=None, reference_weight=0.0) -> dict:
    """An on-policy actor pass; all exploration-intervention batches are refused.

    The guard stops subsequent actor steps only. The caller still trains the
    independent critics and detached auxiliary heads after this function returns.
    It is an early stop, not a rollback or exact trust-region guarantee.
    """
    if getattr(batch, "explored", None) is not None and batch.explored.any():
        raise ValueError("forced-exploration trajectories are critic data, not on-policy PPO data")
    active = batch.legal.sum(dim=1) > 1
    rows = torch.nonzero(active).flatten()
    if not len(rows):
        raise ValueError("no actor choices in the rollout")
    net.train()
    steps, stopped = 0, False
    policy_total = entropy_total = grad_total = clip_total = leash_total = 0.0
    last_kl = max_kl = 0.0
    for _epoch in range(epochs):
        order = rows[torch.randperm(len(rows))]
        for start in range(0, len(order), batch_size):
            picks = order[start:start + batch_size]
            planes = batch.observations.rows(picks.numpy()).dense(device)
            legal = batch.legal[picks].to(device)
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
                scores = logits(net, planes, legal)
            logs = log_distribution(scores, legal)
            old_logs = old[picks].to(device)
            divergence = full_kl(old_logs, logs, legal)
            last_kl = float(divergence.mean().detach())
            max_kl = max(max_kl, last_kl)
            if not math.isfinite(last_kl):
                raise FloatingPointError("nonfinite actor KL")
            if target_kl and last_kl > target_kl:
                stopped = True
                break
            actions = batch.actions[picks].to(device)
            selected = logs.gather(1, actions[:, None]).squeeze(1)
            ratio = (selected - batch.log_probs[picks].to(device)).exp()
            advantage = advantages[picks].to(device)
            clipped = ratio.clamp(1 - ratio_clip, 1 + ratio_clip)
            policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            safe_logs = torch.where(legal, logs, 0.0)
            entropy = -(logs.exp() * safe_logs).sum(dim=1).mean()
            leash = scores.new_zeros((), dtype=torch.float32)
            if reference is not None and reference_weight:
                with torch.no_grad(), torch.autocast(torch.device(device).type,
                                                     dtype=torch.bfloat16, enabled=amp):
                    before = logits(reference, planes, legal)
                leash = full_kl(log_distribution(before, legal), logs, legal).mean()
            loss = policy_loss - entropy_weight * entropy + reference_weight * leash
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            params = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            grad = nn.utils.clip_grad_norm_(params, grad_clip, error_if_nonfinite=True)
            optimizer.step()
            policy_total += float(policy_loss.detach())
            entropy_total += float(entropy.detach())
            grad_total += float(grad)
            clip_total += float((ratio != clipped).float().mean())
            leash_total += float(leash.detach())
            steps += 1
        if stopped:
            break
    # The final measurement includes the last step (the pre-step guard cannot).
    net.eval()
    sums = {name: [0.0, 0] for name in ("riichi", "call", "discard", "other")}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            picks = rows[start:start + batch_size]
            legal = batch.legal[picks].to(device)
            planes = batch.observations.rows(picks.numpy()).dense(device)
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
                now = logits(net, planes, legal)
            divergence = full_kl(old[picks].to(device), log_distribution(now, legal), legal)
            for name, selected in decision_categories(legal).items():
                sums[name][0] += float(divergence[selected].sum())
                sums[name][1] += int(selected.sum())
    return {"actor_updates": steps, "kl_early_stop": stopped, "target_kl": target_kl,
            "pre_step_kl_last": last_kl, "pre_step_kl_max": max_kl,
            "full_kl_by_decision": {key: {"kl": total / count if count else None, "rows": count}
                                    for key, (total, count) in sums.items()},
            "policy_loss": policy_total / max(1, steps), "entropy": entropy_total / max(1, steps),
            "actor_gradient_norm": grad_total / max(1, steps),
            "clip_fraction": clip_total / max(1, steps), "reference_kl": leash_total / max(1, steps)}
