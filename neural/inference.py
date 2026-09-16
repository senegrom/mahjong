"""One evaluation precision and tie contract for ordinary play and search.

This module is importable by cloud controllers without Torch installed. It does
not change training autocast, stored model weights or the exporter (float32).
"""
from __future__ import annotations

from contextlib import contextmanager

VERSION = 1
DEFAULT_ROLLOUT_BATCH = 256
CONTRACT = {"version": VERSION, "ties": "lowest_policy_index",
            "mortal_cuda": "bfloat16", "engine_cuda": "float32", "cpu": "float32"}


def validate_batch(batch_size: int) -> None:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("rollout_batch must be a positive integer")


def describe(kind: str, device) -> dict:
    device_type = str(device).split(":", 1)[0]
    if device_type not in ("cpu", "cuda") or kind not in ("engine", "mortal"):
        raise ValueError("inference requires a known observation layout and CPU or CUDA")
    dtype = "bfloat16" if device_type == "cuda" and kind == "mortal" else "float32"
    return {"version": VERSION, "ties": CONTRACT["ties"],
            "device_type": device_type, "policy_dtype": dtype}


def validate_description(value) -> None:
    if not isinstance(value, dict) or type(value.get("version")) is not int:
        raise ValueError("invalid policy inference contract")
    if value not in [describe(kind, device) for kind in ("engine", "mortal")
                     for device in ("cpu", "cuda")]:
        raise ValueError("unsupported policy inference contract")


@contextmanager
def precision(net, device):
    """Match the ordinary player, even inside a caller's different autocast.

    Mortal-layout ordinary players already use CUDA bfloat16; legacy engine
    players use float32. CPU is explicitly float32, not inherited CPU autocast.
    """
    import torch
    spec = describe(getattr(net, "kind", "engine"), device)
    with torch.autocast(spec["device_type"], dtype=torch.bfloat16,
                        enabled=spec["policy_dtype"] == "bfloat16"):
        yield


def everything(net, planes, legal):
    with precision(net, planes.device):
        return net.everything(planes, legal)


def policy_logits(net, planes, legal):
    """Same arithmetic contract for the fast path and full-policy fallback."""
    with precision(net, planes.device):
        fast = getattr(net, "policy_only", None)
        return fast(planes, legal) if callable(fast) else net(planes, legal)[0]


def order(logits):
    """Descending scores, first policy index wins ties just as argmax does."""
    import torch
    return torch.argsort(logits.float(), dim=1, descending=True, stable=True)
