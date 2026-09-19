"""Versioned actor targets and same-player, full-match advantage estimates.

GAE follows Schulman et al., arXiv:1506.02438. No discount is introduced:
placement is an undiscounted match objective. A hand boundary is NOT terminal.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass(frozen=True)
class Objective:
    name: str = "placement"

    def __post_init__(self):
        if self.name not in ("hybrid", "placement"):
            raise ValueError("objective must be hybrid or placement")

    @property
    def version(self) -> int:
        return {"hybrid": 1, "placement": 2}[self.name]

    def targets(self, batch) -> torch.Tensor:
        values = batch.placements if self.name == "placement" else batch.returns
        if values.ndim != 1 or not torch.isfinite(values).all():
            raise ValueError("invalid objective targets")
        return values.float()

    def contract(self) -> dict:
        return {"name": self.name, "version": self.version, "gamma": 1.0}


def trajectory_links(games, players, hands=None) -> tuple[np.ndarray, np.ndarray]:
    """Link each row to that player's next decision, including riichi stage 2.

    Rows may interleave tables and players. `hands` is only diagnostic; it
    deliberately does not break a trajectory. -1 denotes the end of a match.
    """
    games, players = np.asarray(games), np.asarray(players)
    if (games.ndim != 1 or games.shape != players.shape
            or games.dtype.kind not in "iu" or players.dtype.kind not in "iu"
            or np.any(games < 0) or np.any((players < 0) | (players > 3))):
        raise ValueError("rows require nonnegative game identities and players in [0,3]")
    if hands is not None and np.asarray(hands).shape != games.shape:
        raise ValueError("one hand identity per decision")
    nxt = np.full(len(games), -1, dtype=np.int64)
    last = {}
    for row in range(len(games) - 1, -1, -1):
        key = int(games[row]), int(players[row])
        nxt[row] = last.get(key, -1)
        last[key] = row
    return nxt, nxt < 0


def advantages(returns: torch.Tensor, values: torch.Tensor, *, method: str = "mc",
               next_index=None, gae_lambda: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw advantages and critic targets. The caller normalizes actor rows only.

    For GAE `returns` must be the same terminal placement utility at every
    row of a player's match. Full-return MC also supports the legacy hybrid.
    Values are old, pre-fit predictions; do not refit on this batch first.
    """
    if (returns.ndim != 1 or returns.shape != values.shape or not len(returns)
            or not torch.isfinite(returns).all() or not torch.isfinite(values).all()):
        raise ValueError("finite, matching nonempty one-dimensional returns/values required")
    if method == "mc":
        return returns.detach() - values.detach(), returns.detach().clone()
    if method != "gae" or not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
        raise ValueError("method must be mc/gae and lambda must lie in [0,1]")
    links = np.asarray(next_index)
    if links.shape != (len(returns),) or links.dtype.kind not in "iu":
        raise ValueError("GAE needs one next-decision index per row")
    rows = np.arange(len(links))
    if np.any((links != -1) & ((links <= rows) | (links >= len(links)))):
        raise ValueError("next indices must point forward or be -1")
    # The recurrence is cheap on CPU and avoids one CUDA synchronization per row.
    r = returns.detach().double().cpu().numpy()
    v = values.detach().double().cpu().numpy()
    adv = np.zeros_like(r)
    for i in range(len(links) - 1, -1, -1):
        j = int(links[i])
        if j < 0:
            adv[i] = r[i] - v[i]
        else:
            if not np.isclose(r[i], r[j], rtol=0, atol=1e-6):
                raise ValueError("GAE requires a constant terminal utility per trajectory")
            adv[i] = v[j] - v[i] + gae_lambda * adv[j]
    a = torch.as_tensor(adv, dtype=values.dtype, device=values.device)
    return a, a + values.detach()


def normalize_actor_advantages(raw: torch.Tensor, actor_rows: torch.Tensor) -> torch.Tensor:
    if actor_rows.shape != raw.shape or actor_rows.dtype != torch.bool:
        raise ValueError("actor mask does not match advantages")
    selected = raw[actor_rows]
    if not selected.numel():
        raise ValueError("no stochastic actor decisions in this batch")
    result = torch.zeros_like(raw)
    result[actor_rows] = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-6)
    return result
