"""The zoo: players that are not this project's network, at our tables.

Mortal first. A published Mortal (see `mortal_model.py`) reads the same
planes our network does, from the same follower, and answers in its own
action space of forty-six; this translates its answer into ours, with
our engine's legal mask as the authority on what may be done. Where the
two disagree, Mortal's next-best answer that our engine allows is taken.

One of Mortal's actions is a decision in two steps: it declares riichi
first and is then asked, from a state in which the reach is already
declared, which tile to throw with it. Our action space names the tile
with the declaration, so the follower is told the reach ahead of the
table (`Follower.tell`) and Mortal is asked again for the tile.

`choose` is the one call a table makes of any player, ours or the zoo's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import riichi_py

from . import mortal_model
from .observe import Planes, Views

ACTIONS = riichi_py.ACTIONS
DISCARD = 0
RIICHI_DISCARD = 34
TSUMO = 68
RON = 69
PASS = 70
CHII_LOW = 71
CHII_MIDDLE = 72
CHII_HIGH = 73
PON = 74
CLAIMED_KAN = 75
CONCEALED_KAN = 76
EXTENDED_KAN = 77

# Mortal's action space: 34 discards by kind, three red-five discards, then
# riichi, chi with the called tile lowest, middle and highest, pon, kan,
# a win, an abortive draw, and pass.
MORTAL_RED = {34: 4, 35: 13, 36: 22}
MORTAL_RIICHI = 37
MORTAL_CHI_LOW, MORTAL_CHI_MID, MORTAL_CHI_HIGH = 38, 39, 40
MORTAL_PON = 41
MORTAL_KAN = 42
MORTAL_AGARI = 43
MORTAL_RYUKYOKU = 44
MORTAL_PASS = 45


def translate(action: int, legal: np.ndarray) -> list[int]:
    """Our actions that Mortal's `action` could mean, best first, kept to
    the legal ones. Empty when none is legal, and for riichi, which is
    decided in a second step."""
    wanted: list[int]
    if action < 34:
        wanted = [DISCARD + action]
    elif action in MORTAL_RED:
        wanted = [DISCARD + MORTAL_RED[action]]
    elif action == MORTAL_RIICHI:
        return []
    elif action == MORTAL_CHI_LOW:
        # Mortal's low chi has the called tile lowest in the sequence,
        # which is the sequence that starts at the claimed tile: ours calls
        # that high. The names cross over.
        wanted = [CHII_HIGH]
    elif action == MORTAL_CHI_MID:
        wanted = [CHII_MIDDLE]
    elif action == MORTAL_CHI_HIGH:
        wanted = [CHII_LOW]
    elif action == MORTAL_PON:
        wanted = [PON]
    elif action == MORTAL_KAN:
        wanted = [CLAIMED_KAN, CONCEALED_KAN, EXTENDED_KAN]
    elif action == MORTAL_AGARI:
        wanted = [TSUMO, RON]
    elif action == MORTAL_PASS:
        wanted = [PASS]
    else:
        # An abortive draw, which our rules do not offer as a move.
        wanted = []
    return [index for index in wanted if legal[index]]


class MortalPlayer:
    """A published Mortal, choosing moves in our action space."""

    kind = "mortal"

    def __init__(self, path: Path | str, device: str = "cuda") -> None:
        self.net = mortal_model.load(path, device)
        self.device = device
        self.path = str(path)
        # How often the answer had to fall back on a later choice of
        # Mortal's, and how often nothing of Mortal's was legal at all.
        self.fallbacks = 0
        self.orphans = 0

    def eval(self) -> MortalPlayer:
        return self

    def _ask(self, views: Views, who: list[tuple[int, int]]) -> np.ndarray:
        """Mortal's Q values for those players, in its own action space."""
        follower = views.observer.follower
        indptr, indices, values, masks = follower.encode(who)
        planes = Planes.from_follower(indptr, indices, values).dense(self.device)
        mask = torch.from_numpy(np.asarray(masks, dtype=bool)).to(self.device)
        with torch.no_grad():
            q = self.net(planes, mask)
        return q.float().cpu().numpy()

    @torch.no_grad()
    def choose(
        self, views: Views, rows: np.ndarray, players: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        """One of our actions per row, for `players[i]` in game `rows[i]`,
        given our engine's legal mask for each."""
        who = list(zip(np.asarray(rows).tolist(), np.asarray(players).tolist()))
        q = self._ask(views, who)
        choice = np.full(len(who), PASS, dtype=np.int64)
        second: list[int] = []
        for i, (game, player) in enumerate(who):
            order = np.argsort(-q[i])
            allowed = legal[i]
            picked = None
            for rank, action in enumerate(order):
                if not np.isfinite(q[i][action]):
                    break
                if action == MORTAL_RIICHI:
                    if allowed[RIICHI_DISCARD:TSUMO].any():
                        second.append(i)
                        picked = -1
                        break
                    continue
                found = translate(int(action), allowed)
                if found:
                    picked = found[0]
                    if rank > 0:
                        self.fallbacks += 1
                    break
            if picked is None:
                self.orphans += 1
                picked = int(np.argmax(allowed))
            choice[i] = picked

        if second:
            # The reach declared ahead of the table, then the tile.
            follower = views.observer.follower
            for i in second:
                game, player = who[i]
                follower.tell(game, player, json.dumps({"type": "reach", "actor": player}))
            after = self._ask(views, [who[i] for i in second])
            for slot, i in enumerate(second):
                allowed = legal[i]
                order = np.argsort(-after[slot][:34])
                tile = next(
                    (int(t) for t in order if allowed[RIICHI_DISCARD + t] and np.isfinite(after[slot][t])),
                    None,
                )
                if tile is None:
                    self.orphans += 1
                    tile = int(np.argmax(allowed[RIICHI_DISCARD:TSUMO]))
                choice[i] = RIICHI_DISCARD + tile
        return choice


def choose(
    player,
    views: Views,
    rows: np.ndarray,
    players: np.ndarray,
    legal: np.ndarray,
    device: str,
    greedy: bool = True,
) -> np.ndarray:
    """One of our actions per row from any player: a zoo player answers
    for itself; a network of ours is shown the planes it sees and plays
    its best move."""
    if hasattr(player, "choose"):
        return player.choose(views, rows, players, legal)
    logits, _value = player(
        views.dense(player.kind, rows, players, device),
        torch.from_numpy(legal).to(device),
    )
    if greedy:
        return logits.argmax(dim=1).cpu().numpy()
    return torch.distributions.Categorical(logits=logits.float()).sample().cpu().numpy()


def load_player(path: Path | str, device: str, channels: int | None = None, blocks: int | None = None):
    """A player from a checkpoint of either kind: one of ours, rebuilt at
    the shape it says, or a Mortal, told apart by what the file holds."""
    from .model import from_payload

    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        payload = None
    if payload is not None and "model" in payload:
        net = from_payload(payload, device, channels, blocks)
        net.eval()
        return net
    return MortalPlayer(path, device)
