"""Diagnostics for search evidence, clustered by independent environment seed.

Repeated decisions, seats and world-splits in one game are not extra games.
The ratio-estimator error below targets the decision-weighted mean without
pretending games containing more decisions are more independent evidence.
These are diagnostic standard errors, not bounds on playing strength.
"""
from __future__ import annotations

import math
import numpy as np


def clustered_mean(values, games) -> dict:
    """Decision-weighted mean and game-cluster sandwich standard error.

    Let S_g be the sum and n_g the number of recorded decisions in game g.
    The mean is sum(S_g)/sum(n_g), not an unweighted mean of game means.
    SE^2 = G/(G-1) * sum((S_g - mean*n_g)^2) / N^2.
    With fewer than two independent games the error is unidentified, not zero.
    """
    values = np.asarray(values, dtype=np.float64)
    games = np.asarray(games)
    if (values.ndim != 1 or games.shape != values.shape
            or games.dtype.kind not in 'iu' or np.any(games < 0)
            or not np.isfinite(values).all()):
        raise ValueError('finite decision values and matching nonnegative integer game identities required')
    identities, inverse = np.unique(games, return_inverse=True)
    n, g = len(values), len(identities)
    if not n:
        return {'rows': 0, 'independent_deals': 0, 'mean': None,
                'standard_error': None, 'standard_errors': None,
                'method': 'game-clustered ratio-estimator sandwich'}
    mean = float(values.mean())
    error = None
    if g > 1:
        residual = np.bincount(inverse, weights=values - mean, minlength=g)
        error = math.sqrt(float(residual @ residual) * g / (g - 1)) / n
    return {'rows': n, 'independent_deals': g, 'mean': mean,
            'standard_error': error,
            'standard_errors': mean / error if error is not None and error > 0 else None,
            'method': 'game-clustered ratio-estimator sandwich'}


def validation_summary(values, games, *, total_deals: int) -> dict:
    """Compatibility names used by ``worth`` plus explicit evidence counts."""
    if type(total_deals) is not int or total_deals < 0:
        raise ValueError('total_deals must be a nonnegative integer')
    stats = clustered_mean(values, games)
    if stats['independent_deals'] > total_deals:
        raise ValueError('total_deals is smaller than the recorded evidence')
    return {**{k: v for k, v in stats.items() if k != 'mean'},
            'a_decision': stats['mean'],
            'a_deal_so_far': float(np.sum(values)) / total_deals if total_deals else None}
