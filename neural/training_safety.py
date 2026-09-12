"""Runtime and configuration contracts supplementing training_batches.

Keep the learning/batch policy in training_batches and checkpoint validation in
checkpoints. This module does not import Torch on thin cloud launchers.
"""
from __future__ import annotations

from pathlib import Path
import math
import warnings

TRAINING_API_VERSION = 2


def require_training_engine() -> None:
    import riichi_py

    if getattr(riichi_py, "TRAINING_API_VERSION", 0) != TRAINING_API_VERSION:
        raise RuntimeError(
            "Rebuild and reinstall riichi_py: training requires API version 2 "
            "(strict actions and independent environment/search RNG streams)"
        )


def benchmark_history(payload: dict) -> tuple[float | None, float]:
    """Do not compare benchmarks produced under incompatible native semantics.

    Only summary metrics are reset; the caller continues restoring the existing
    model, optimizer, generation and sampling state without modification.
    """
    if payload.get("training_api_version") != TRAINING_API_VERSION:
        warnings.warn(
            "Training environment changed: reset smoothed/best benchmark history; "
            "remeasure competing checkpoints under the same native API version",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, float("inf")
    return payload.get("smoothed"), float(payload.get("best_placement", float("inf")))


def validate_training_options(args) -> None:
    """Reject experiment-changing mistakes before output creation or self-play."""
    for name in ("batch", "epochs", "games", "measure_every", "measure_games"):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("rounds", "generations"):
        value = getattr(args, name, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("resume", "mortal", "ours"):
        path = getattr(args, name, None)
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"{name} checkpoint does not exist: {path}")
    roster = list(getattr(args, "opponents", []))
    roster += list(getattr(args, "recent", [])) + list(getattr(args, "older", []))
    if getattr(args, "champion", None):
        roster.append(args.champion)
    for path in roster:
        if not Path(path).is_file():
            raise FileNotFoundError(f"opponent checkpoint does not exist: {path}")
    share = getattr(args, "opponent_share", 0.0)
    if not 0.0 <= share <= 1.0:
        raise ValueError("opponent_share must be between zero and one")
    if share > 0 and not roster:
        raise ValueError("opponent_share requires at least one opponent checkpoint")
    if getattr(args, "freeze_policy", False) and getattr(args, "freeze_aux", False):
        raise ValueError("freeze_policy and freeze_aux cannot both be enabled")
    fixed = getattr(args, "fixed", None)
    if fixed is not None:
        if not fixed:
            raise ValueError("Choose at least one training mode with --fixed")
        for mode in fixed:
            if not isinstance(mode, str) or (mode != "none" and
                    not set(mode.split("+")) <= {"mortal", "ours", "head"}):
                raise ValueError(f"Invalid training mode: {mode}")

    # Reject nonfinite hyperparameters before allocating a model or creating output.
    for name in ('lr', 'lr_ours', 'lr_mortal', 'temperature'):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive')
    for name in ('target_kl', 'entropy', 'value_weight', 'hands_weight', 'reader_weight',
                 'distil_weight', 'leash'):
        value = getattr(args, name, 0.0)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    clip = getattr(args, 'clip', 0.2)
    if not math.isfinite(clip) or not 0 < clip < 1:
        raise ValueError('clip must be finite and between zero and one')
    baseline = getattr(args, 'baseline_batch', None)
    if baseline is not None and (type(baseline) is not int or baseline <= 0):
        raise ValueError('baseline_batch must be a positive integer')


def training_control_arguments(target_kl: float = 0.0, baseline_batch: int | None = None) -> list[str]:
    """Shared thin-launcher validation; never import Torch to deploy a cloud job."""
    if not math.isfinite(target_kl) or target_kl < 0:
        raise ValueError('target_kl must be finite and nonnegative')
    arguments = ['--target-kl', str(target_kl)]
    if baseline_batch is not None:
        if type(baseline_batch) is not int or baseline_batch <= 0:
            raise ValueError('baseline_batch must be a positive integer')
        arguments += ['--baseline-batch', str(baseline_batch)]
    return arguments
