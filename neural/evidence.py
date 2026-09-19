"""Variance-sensitive bounded-mean evidence and deal-grouped data partitions.

Empirical Bernstein: Maurer & Pontil (2009), Theorem 4, arXiv:0907.3740.
Apply the theorem to the negated sample for a lower bound, scaled to [lo,hi].
Settings/sample size are selected BEFORE reading outcomes. This is not an
anytime confidence sequence; use fresh deals and summable alpha per attempt.
"""
from __future__ import annotations
import math
import numpy as np


def lower_mean(values, *, alpha: float, lo: float, hi: float,
               method: str = "empirical-bernstein") -> dict:
    x = np.asarray(values, dtype=np.float64)
    if (x.ndim != 1 or not len(x) or not np.isfinite(x).all()
            or not all(math.isfinite(v) for v in (alpha, lo, hi))
            or not 0 < alpha < 1 or not lo < hi or np.any((x < lo) | (x > hi))):
        raise ValueError("invalid bounded-mean sample or confidence parameters")
    n, width = len(x), hi - lo
    variance = float(x.var(ddof=1)) if n > 1 else 0.0
    if method == "hoeffding":
        radius = width * math.sqrt(math.log(1 / alpha) / (2 * n))
    elif method == "empirical-bernstein":
        if n < 2:
            radius = width  # No empirical variance estimate; only the range is known.
        else:
            log = math.log(2 / alpha)
            radius = math.sqrt(2 * variance * log / n) + 7 * width * log / (3 * (n - 1))
    else:
        raise ValueError("unknown evidence method")
    mean = float(x.mean())
    return {"mean": mean, "lower": max(lo, mean - radius), "radius": radius,
            "sample_variance": variance, "independent_deals": n, "alpha": alpha,
            "method": method}


def split_games(games) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/10/10 train/validation/test by immutable environment game identity.

    Stable SplitMix64 mixing avoids dependence on input ordering, collection
    block boundaries, or a game number's last digit. Every seat and root from
    the same deal stays together, including duplicate seeds in other files.
    """
    raw = np.asarray(games)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or np.any(raw < 0):
        raise ValueError("nonnegative integer game identities required")
    x = raw.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    bucket = x % np.uint64(10)
    return np.flatnonzero(bucket >= 2), np.flatnonzero(bucket == 1), np.flatnonzero(bucket == 0)
