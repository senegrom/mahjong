"""Exact, in-memory simulator/follower snapshots for opt-in research.

Snapshots include hidden information and must NEVER be passed to a policy.
They are process-local handles, not pickle files or portable saved games.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import riichi_py

from .observe import Observer, Views

RESEARCH_API_VERSION = 1


def require_research_engine():
    if getattr(riichi_py, "RESEARCH_API_VERSION", None) != RESEARCH_API_VERSION:
        raise RuntimeError("rebuild riichi_py with research API 1 for danger/snapshot experiments")


def context(arena) -> np.ndarray:
    require_research_engine()
    return np.asarray(riichi_py.research_context(arena), dtype=np.int64).reshape(arena.games, 16)


def decision(views):
    """Synchronize events, then expose public decision identities and legal masks."""
    views.advance()
    arena = views.arena
    seats = np.frombuffer(arena.seats(), np.uint8)
    rows = np.flatnonzero(seats != 255)
    masks = np.frombuffer(arena.legal_mask(), np.uint8).reshape(arena.games, 78).astype(bool)
    people = np.frombuffer(arena.seat_players(), np.uint8).reshape(arena.games, 4)
    players = people[rows, seats[rows]].astype(np.int64)
    views.prepare(rows, players)
    return rows, players, masks


@dataclass(frozen=True)
class Snapshot:
    """A private checkpoint. restore() returns independent copies each time."""
    _arena: object
    _follower: object | None

    @classmethod
    def capture(cls, views: Views, rows=None):
        require_research_engine()
        rows = list(range(views.games)) if rows is None else list(rows)
        if any(isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)) for row in rows):
            raise ValueError("snapshot rows must be integer game indices")
        rows = [int(row) for row in rows]
        arena = riichi_py.research_fork(views.arena, rows)
        follower = None
        if views.observer is not None:
            if not hasattr(views.observer.follower, "fork"):
                raise RuntimeError("rebuild libriichi with exact follower snapshots")
            follower = views.observer.follower.fork(rows)
        return cls(arena, follower)

    def restore(self) -> Views:
        rows = list(range(self._arena.games))
        arena = riichi_py.research_fork(self._arena, rows)
        # Avoid a fresh game or re-feeding already-consumed events. Cached encodings
        # are discarded; the cloned follower contains the exact pending reach flag.
        views = Views.__new__(Views)
        views.arena, views.games = arena, arena.games
        views._engine, views._step = None, None
        views.observer = None
        if self._follower is not None:
            views.observer = Observer.__new__(Observer)
            views.observer.arena = arena
            views.observer.follower = self._follower.fork(rows)
        return views


def public_flags(public_context, masks, log_probs=None, close_points=8000, close_logits=0.5):
    """Root selection uses only public pre-action information, never outcomes.

    Context columns are defined by the native research_context API. Categories
    can overlap; 'uniform' intentionally retains ordinary decisions.
    """
    c, m = np.asarray(public_context), np.asarray(masks, dtype=bool)
    if c.shape != (len(m), 16) or m.shape != (len(c), 46):
        raise ValueError("public root context/mask layout mismatch")
    rows = np.arange(len(c)); actor = c[:, 6]
    if np.any((actor < 0) | (actor > 3)):
        raise ValueError("root must belong to a live player")
    scores = c[:, 8:12]
    gap = np.abs(scores - scores[rows, actor, None])
    gap[rows, actor] = np.iinfo(np.int64).max
    danger = c[:, 12:16].sum(1) - c[rows, 12 + actor] > 0
    choices = m.sum(1) > 1
    late = (c[:, 0] > 1) | ((c[:, 0] == 1) & (c[:, 1] >= 3))
    flags = {"uniform": choices, "riichi_pressure": choices & danger,
             "close_endgame": choices & late & (gap.min(1) <= close_points),
             "call_pass": choices & m[:, 45] & m[:, 38:43].any(1),
             "riichi_choice": choices & m[:, 37]}
    if log_probs is not None:
        lp = np.asarray(log_probs)
        if lp.shape != m.shape or not np.isfinite(lp[m]).all():
            raise ValueError("invalid public policy scores")
        ordered = np.sort(np.where(m, lp, -np.inf), axis=1)
        # Avoid -inf - -inf for the single legal action case.
        margin = np.full(len(m), np.inf)
        margin[choices] = ordered[choices, -1] - ordered[choices, -2]
        flags["close_policy"] = choices & (margin <= close_logits)
    return flags
