"""Completed-game outcomes shared by training and all evaluation entry points.

Tied players occupy the average of their shared ranks and split the rewards
for those ranks. Win rates count a share of first place (1/k for k winners),
so ties do not create either extra wins or an arbitrary player-index winner.
"""
from __future__ import annotations

import numpy as np


class IncompleteGamesError(RuntimeError):
    """A simulation stopped before every game reached a terminal state."""


def validate_budget(games: int, max_steps: int) -> None:
    for name, value in (("games", games), ("max_steps", max_steps)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def require_finished(arena, *, steps: int, context: str) -> None:
    """Never read provisional scores as terminal targets or evaluation data."""
    if not arena.all_finished():
        live = int(np.count_nonzero(np.frombuffer(arena.seats(), dtype=np.uint8) != 0xFF))
        raise IncompleteGamesError(
            f"{context} stopped after {steps} steps with unfinished games "
            f"({live} still awaiting decisions); no final results were produced. "
            "Increase max_steps or investigate the stalled games."
        )


def _ties(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != 4 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite (games, 4) array")
    better = (scores[:, None, :] > scores[:, :, None]).sum(axis=2)
    tied = (scores[:, None, :] == scores[:, :, None]).sum(axis=2)
    return better, tied


def placements(scores: np.ndarray) -> np.ndarray:
    """One-based average ranks; a four-way tie places everyone at 2.5."""
    better, tied = _ties(scores)
    return 1.0 + better + (tied - 1) / 2.0


def placement_rewards(scores: np.ndarray, values) -> np.ndarray:
    """Share the sum of rewards for the places each tie occupies."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("placement values must contain four finite rewards")
    better, tied = _ties(scores)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return (cumulative[better + tied] - cumulative[better]) / tied


def win_shares(scores: np.ndarray) -> np.ndarray:
    """First-place credit, divided equally among tied winners."""
    better, tied = _ties(scores)
    return (better == 0) / tied
