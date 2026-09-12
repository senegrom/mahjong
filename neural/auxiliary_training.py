"""Entry and update contracts for imitation, re-heading and legacy distillation."""
from __future__ import annotations

import math
from pathlib import Path
from .training_batches import validate_learning


def validate_auxiliary_options(args) -> None:
    """Reject no-op runs and misspelled inputs before creating output or models."""
    validate_learning(args.batch, args.epochs)
    if args.batch < 2:
        raise ValueError("batch must be at least 2; singleton auxiliary batches are not trained")
    for name in ("rounds", "games", "measure_every", "measure_games", "max_steps",
                 "worlds", "candidates"):
        value = getattr(args, name, 1)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("lr", "temperature"):
        value = getattr(args, name, 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("margin", "value_weight"):
        value = getattr(args, name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    improve = getattr(args, "improve", 0.5)
    if not isinstance(improve, (int, float)) or isinstance(improve, bool) or not math.isfinite(improve) or not 0 <= improve <= 1:
        raise ValueError("improve must be finite and between zero and one")
    for name in ("resume", "teacher"):
        value = getattr(args, name, None)
        if value is not None and not Path(value).is_file():
            raise FileNotFoundError(f"{name} checkpoint does not exist: {value}")


def require_supervised_rows(rows: int) -> None:
    """Auxiliary loops can use partial batches, but must have at least two rows."""
    if type(rows) is not int or rows < 2:
        raise ValueError("At least two training rows are required; no learned checkpoint will be saved")
