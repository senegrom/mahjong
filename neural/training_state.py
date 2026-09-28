"""Checkpoint the independent freeze schedule and PyTorch sampling streams."""

from __future__ import annotations

from copy import deepcopy
import warnings

import numpy as np
import torch


def capture_random_state(drawer: np.random.Generator | None = None) -> dict:
    """The CPU generator and every CUDA one, with the freeze schedule's
    drawer when there is one: snapshots, not mutable references to running
    generators, taken at the completed checkpoint boundary."""
    state = {"freeze": deepcopy(drawer.bit_generator.state)} if drawer is not None else {}
    state["torch"] = torch.get_rng_state().clone()
    state["cuda"] = ([stream.clone() for stream in torch.cuda.get_rng_state_all()]
                     if torch.cuda.is_available() else None)
    return state


def restore_random_state(
    saved: dict | None, *, generation: int, seed: int | None = None, modes: list[str] | None = None
) -> np.random.Generator | None:
    """Restore after constructing every network, opponent, optimizer and
    compiler, before self-play: constructing a module consumes Torch's
    stream even when a checkpoint replaces its weights.

    Given `modes`, the trainer draws what stays fixed each generation, and
    the drawer comes back, seeded from `seed` and at the saved point of
    its schedule. Old checkpoints cannot reproduce their missing Torch
    stream; with the same seed and mode list the freeze schedule at least
    advances to the absolute generation instead of repeating its beginning
    every block.
    """
    drawer = None
    if modes is not None:
        if not modes:
            raise ValueError("Choose at least one training mode with --fixed")
        drawer = np.random.default_rng(seed + 99)
    if generation < 0:
        raise ValueError("Checkpoint generation cannot be negative")
    if saved is None:
        if drawer is not None:
            for _ in range(generation):
                drawer.choice(modes)
        if generation:
            warnings.warn(
                "Legacy checkpoint has no random state; advanced the freeze schedule "
                "using the current seed and modes, but Torch sampling cannot be reproduced"
                if drawer is not None else
                "Legacy checkpoint has no random state; exact continuation cannot be reconstructed",
                RuntimeWarning,
                stacklevel=2,
            )
        return drawer

    # Reject a device mismatch rather than claiming a reproducible GPU resume.
    cuda = saved.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        if len(cuda) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA random states do not match the device count")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda])
    elif cuda is not None or torch.cuda.is_available():
        warnings.warn(
            "Training device changed; the CPU streams resume, but GPU sampling cannot be reproduced",
            RuntimeWarning,
            stacklevel=2,
        )
    if drawer is not None:
        drawer.bit_generator.state = deepcopy(saved["freeze"])
    torch.set_rng_state(saved["torch"].cpu())
    return drawer


# Version 2: immutable per-generation ranges, independent of --games.
# Changing this stride changes every future deal; do not tune it with a run.
ROUND_SEED_STRIDE = 1 << 32


def round_seed(seed: int, generation: int, games: int) -> int:
    """First u64 deal seed in this generation's fixed, disjoint range.

    Reserving 2**32 seeds per generation lets --games grow or shrink on
    resume without revisiting earlier ranges. No mutable allocator is
    needed: the checkpoint's absolute generation identifies the range.

    This intentionally changes the old deal sequence, including small
    rounds. At a legacy resume G, every earlier range from either old
    formula ends below seed + G * stride, provided its rounds had at most
    stride games. Starting at seed + (G + 1) * stride clears those ranges
    too. Keep the same base seed when resuming, and do not downgrade to the
    old allocator. Reject overflow instead of letting Arena wrap its u64.
    """
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if type(generation) is not int or generation < 0:
        raise ValueError("generation must be a nonnegative integer")
    if type(games) is not int or not 1 <= games <= ROUND_SEED_STRIDE:
        raise ValueError("games must be between 1 and 2**32")
    first = seed + (generation + 1) * ROUND_SEED_STRIDE
    if first + games > 1 << 64:
        raise ValueError("The generation's deal seeds exceed the unsigned 64-bit range")
    return first


def peak_rss_gb() -> float | None:
    """The process's peak resident memory so far, in GiB, where the platform
    says (Linux reports kilobytes); None elsewhere."""
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    except (ImportError, AttributeError, OSError):
        return None


def peak_gpu_gb() -> float | None:
    """The most the card has held for this process so far, in GiB."""
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**30, 2)
