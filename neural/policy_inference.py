"""One explicit arithmetic and tie-breaking contract for evaluation policies.

This module is importable on thin cloud launchers; Torch is imported only when
an inference context is entered. Runtime precision is not a checkpoint weight or
a training-autocast setting. Configure the frozen evaluation player once per run.
"""
from __future__ import annotations

from contextlib import contextmanager

VERSION = 1
DEFAULT_ROLLOUT_BATCH = 256
PRECISIONS = ("auto", "float32", "bfloat16")


def validate(precision="auto", rollout_batch=DEFAULT_ROLLOUT_BATCH):
    if not isinstance(precision, str) or precision not in PRECISIONS:
        raise ValueError("policy_precision must be auto, float32 or bfloat16")
    if type(rollout_batch) is not int or rollout_batch <= 0:
        raise ValueError("rollout_batch must be a positive integer")


def underlying(net):
    # The normal Mortal-space player wraps the same network the teacher serves.
    return net.net if not hasattr(net, "everything") and hasattr(getattr(net, "net", None), "everything") else net


def resolved(precision, device, actions=46):
    validate(precision)
    kind = str(device).split(":", 1)[0]
    if precision == "auto":
        # Preserve ordinary-play defaults: Mortal policies use BF16 on CUDA;
        # legacy engine-action networks and CPU evaluation use float32.
        return "bfloat16" if kind == "cuda" and actions == 46 else "float32"
    if precision == "bfloat16" and kind not in ("cpu", "cuda"):
        raise ValueError("bfloat16 policy inference requires CPU or CUDA")
    return precision


def describe(net, device):
    net = underlying(net)
    precision = getattr(net, "policy_precision", "auto")
    actions = getattr(net, "actions", 46)
    return {"version": VERSION, "requested": precision,
            "resolved": resolved(precision, device, actions),
            "tie_break": "first_policy_index"}


def configure(net, precision, device):
    """Apply an explicitly requested runtime setting to the frozen player."""
    net = underlying(net)
    resolved(precision, device, getattr(net, "actions", 46))
    net.policy_precision = precision
    return describe(net, device)


@contextmanager
def context(net, device):
    """Override ambient autocast identically in ordinary play and search.

    Do not change gradient settings or training forwards here. Outputs are cast
    to float32 by the calling adapter before probabilities and comparisons.
    """
    import torch
    kind = torch.device(device).type
    mixed = describe(net, device)["resolved"] == "bfloat16"
    with torch.autocast(kind, dtype=torch.bfloat16, enabled=mixed):
        yield


def order(logits):
    """Descending policy order; exact ties use argmax's first-index rule."""
    import torch
    return torch.argsort(logits, dim=1, descending=True, stable=True)
