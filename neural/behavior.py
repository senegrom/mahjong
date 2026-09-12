"""Validate exploration settings shared by collection and the combined trainer.

Forced rows are auxiliary-only, following the current main-branch policy.
"""
from __future__ import annotations

import math


def validate_exploration(epsilon: float) -> None:
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("exploration must be finite and between zero and one")
