"""Teacher proposal coverage for the immutable 46-action student contract."""
from __future__ import annotations

import numpy as np
from .search_replay import ENGINE_ACTIONS, MEANINGS


def representable_moves(engine_legal: np.ndarray) -> np.ndarray:
    """Engine moves the student can execute, without changing true legality.

    Several kans share one policy action, whose translation executes only its
    first legal meaning. Every legal riichi discard remains expressible through
    the conditional second question. Search evaluation may consider other moves;
    a completed supervised collection must not propose an unteachable target.
    """
    if (engine_legal.ndim != 2 or engine_legal.shape[1] != ENGINE_ACTIONS
            or engine_legal.dtype != np.bool_):
        raise ValueError("Invalid engine legal mask for student action coverage")
    represented = np.zeros_like(engine_legal)
    for policy_action, meanings in enumerate(MEANINGS):
        if not meanings:
            continue
        found = engine_legal[:, meanings]
        live = found.any(axis=1)
        if policy_action == 37:
            represented[:, 34:68] |= engine_legal[:, 34:68]
        else:
            first = np.asarray(meanings)[found.argmax(axis=1)]
            represented[np.nonzero(live)[0], first[live]] = True
    return represented
