"""Acting-policy numerics shared by ordinary play and hypothetical search.

Training forwards deliberately do not use this module: their caller owns
mixed precision. Preserve ordinary Mortal-space play's CUDA bfloat16 policy,
CPU float32, and the legacy 78-action player's float32 path. Leaf evaluators
keep their separate, existing precision contract.
"""
from __future__ import annotations

import torch

VERSION = 1
DEFAULT_ROLLOUT_BATCH = 256


def precision(device, actions: int = 46) -> str:
    """Explicit acting precision, independent of an enclosing autocast context."""
    kind = torch.device(device).type
    if kind not in ("cpu", "cuda"):
        raise ValueError("Policy inference supports CPU and CUDA")
    if type(actions) is not int or actions not in (46, 78):
        raise ValueError("Unknown policy action layout")
    return "bfloat16" if kind == "cuda" and actions == 46 else "float32"


def autocast(device, actions: int = 46):
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                          enabled=precision(device, actions) == "bfloat16")


def describe(device, actions: int = 46) -> dict:
    return {"version": VERSION, "precision": precision(device, actions),
            "tie_break": "first_policy_index"}


def validate_batch(batch_size: int) -> None:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("rollout_batch must be a positive integer")


def order(logits: torch.Tensor) -> torch.Tensor:
    """Descending scores, retaining argmax's first-index convention on ties."""
    return torch.argsort(logits.float(), dim=1, descending=True, stable=True)


def everything(net, planes, legal):
    with autocast(planes.device, int(legal.shape[1])):
        return net.everything(planes, legal)


def policy_logits(net, planes, legal):
    with autocast(planes.device, int(legal.shape[1])):
        fast = getattr(net, "policy_only", None)
        return fast(planes, legal) if callable(fast) else net(planes, legal)[0]
