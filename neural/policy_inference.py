"""Acting-policy numerics for the players that choose moves at a table.

Training forwards deliberately do not use this module: their caller owns
mixed precision. Preserve ordinary Mortal-space play's CUDA bfloat16 policy,
CPU float32, and the legacy 78-action player's float32 path.
"""
from __future__ import annotations

import torch
from torch import nn


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


def precast(net: nn.Module) -> nn.Module:
    """`net`, its convolutions' and linear layers' weights and biases cast
    to bfloat16 once, here, instead of by autocast on every call.

    Autocast keeps the cast of a weight for the rest of its region only
    while the weight requires a gradient, so a player's weights were cast
    afresh on every forward: about four hundred small kernels a call for
    a seated Mortal. The cast is the one autocast makes, rounding to the
    nearest bfloat16, so the weights a convolution is handed, and every
    answer, are bit for bit what they were. Normalisation layers keep
    float32: autocast runs group norm in float32, and batch norm in the
    input's precision with float32 parameters, and either would be handed
    other numbers. Only for a network that plays on the card under
    autocast: one cast here cannot run in float32 any more."""
    for module in net.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            module.to(torch.bfloat16)
    return net
