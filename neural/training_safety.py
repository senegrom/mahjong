"""Explicit contracts for the learner, environment and legacy search tools."""
from __future__ import annotations

from pathlib import Path

import torch

# Native strict steps and independent dealing/search random streams.
TRAINING_API_VERSION = 2


def require_training_engine() -> None:
    import riichi_py
    if getattr(riichi_py, "TRAINING_API_VERSION", 0) != TRAINING_API_VERSION:
        raise RuntimeError(
            "Rebuild and reinstall riichi_py: training requires API version 2 "
            "(strict actions and independent environment/search RNG streams)"
        )


def validate_training_options(args) -> None:
    """Reject configuration mistakes before creating output or playing games."""
    for name in ("batch", "epochs", "games", "measure_every", "measure_games"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("rounds", "generations"):
        if getattr(args, name, 0) < 0:
            raise ValueError(f"{name} cannot be negative")
    for name in ("resume", "mortal", "ours"):
        path = getattr(args, name, None)
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"{name} checkpoint does not exist: {path}")
    for path in getattr(args, "opponents", []):
        if not Path(path).is_file():
            raise FileNotFoundError(f"opponent checkpoint does not exist: {path}")
    share = getattr(args, "opponent_share", 0.0)
    if not 0.0 <= share <= 1.0:
        raise ValueError("opponent_share must be between zero and one")
    if share > 0 and not getattr(args, "opponents", []):
        raise ValueError("opponent_share requires at least one opponent checkpoint")
    if getattr(args, "freeze_policy", False) and getattr(args, "freeze_aux", False):
        raise ValueError("freeze_policy and freeze_aux cannot both be enabled")


def minibatch_indices(count: int, batch_size: int, *, compiled: bool = False) -> list[torch.Tensor]:
    """Eager learning uses all rows. Compiled learning retains fixed batches.

    The fixed-shape path intentionally drops a shuffled remainder, as before,
    but never silently trains on zero rows. Use a smaller batch or eager mode
    for smoke tests. Indices stay on the host for the sparse observation store.
    """
    if count <= 0 or batch_size <= 0:
        raise ValueError("minibatches need positive decision and batch counts")
    if compiled and count < batch_size:
        raise ValueError(
            f"Only {count} decisions for compiled batch {batch_size}; "
            "reduce --batch or omit --compile (no optimizer updates were performed)"
        )
    batches = list(torch.randperm(count).split(batch_size))
    return [rows for rows in batches if len(rows) == batch_size] if compiled else batches


def require_legacy_search(net) -> None:
    """Do not silently discard a combined player or feed it the wrong schema."""
    import riichi_py
    if (getattr(net, "kind", None) != "engine"
            or getattr(net, "actions", riichi_py.ACTIONS) != riichi_py.ACTIONS):
        raise ValueError(
            "Legacy search supports only the engine observation/action layout. "
            "Mortal and combined checkpoints need information-consistent search "
            "adapters; use neural.duel or neural.arena for policy-only evaluation."
        )


def require_search_payload(payload: dict) -> None:
    if "combined" in payload or "model" not in payload:
        raise ValueError("Legacy search cannot load a Mortal/combined checkpoint as a standalone model")
