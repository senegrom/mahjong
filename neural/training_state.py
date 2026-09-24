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


def round_seed(seed: int, generation: int, games: int) -> int:
    """The first deal of a generation's round; its games take the seeds after
    it. The step between rounds is at least the round, so no deal is played
    twice: a step of a thousand under 4,096-game rounds replayed three
    quarters of each round's deals in the next, and every deal in four."""
    return seed + generation * max(1000, games)


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
