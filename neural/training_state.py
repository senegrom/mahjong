"""Checkpoint the independent freeze schedule and PyTorch sampling streams."""

from __future__ import annotations

from copy import deepcopy
import warnings

import numpy as np
import torch


def capture_random_state(drawer: np.random.Generator) -> dict:
    """Keep snapshots, not mutable references to a running generator."""
    return {
        "freeze": deepcopy(drawer.bit_generator.state),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    }


def restore_random_state(
    saved: dict | None, *, seed: int, generation: int, modes: list[str]
) -> np.random.Generator:
    """Restore after constructing every network/optimizer, before self-play.

    Old checkpoints cannot reproduce their missing Torch stream. With the
    same seed and mode list we can at least advance their freeze schedule to
    the absolute generation instead of repeating its beginning every block.
    """
    if not modes:
        raise ValueError("Choose at least one training mode with --fixed")
    if generation < 0:
        raise ValueError("Checkpoint generation cannot be negative")
    drawer = np.random.default_rng(seed + 99)
    if saved is None:
        for _ in range(generation):
            drawer.choice(modes)
        if generation:
            warnings.warn(
                "Legacy checkpoint has no random state; advanced the freeze schedule "
                "using the current seed and modes, but Torch sampling cannot be reproduced",
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
            "Training device changed; the freeze and CPU streams resume, "
            "but GPU sampling cannot be reproduced",
            RuntimeWarning,
            stacklevel=2,
        )
    drawer.bit_generator.state = deepcopy(saved["freeze"])
    torch.set_rng_state(saved["torch"].cpu())
    return drawer


def capture_sampling_state() -> dict:
    """CPU and every CUDA generator, copied at the completed checkpoint boundary."""
    return {"torch": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else None}


def restore_sampling_state(saved: dict | None, *, generation: int) -> None:
    """Restore only after all learners, opponents, optimizers and compilers exist."""
    if generation < 0:
        raise ValueError("Checkpoint generation cannot be negative")
    if saved is None:
        if generation:
            warnings.warn("Legacy Mortal checkpoint has no sampling state; exact continuation "
                          "cannot be reconstructed", RuntimeWarning, stacklevel=2)
        return
    cuda = saved.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        if len(cuda) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA random states do not match the device count")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda])
    elif cuda is not None or torch.cuda.is_available():
        warnings.warn("Training device changed; GPU sampling cannot be reproduced",
                      RuntimeWarning, stacklevel=2)
    torch.set_rng_state(saved["torch"].cpu())


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
