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


def meanings(action: int) -> list[int]:
    """Our actions that Mortal's `action` could mean, best first. Riichi
    means any of the riichi discards, the tile being decided in a second
    step; an abortive draw means nothing, our rules not offering one."""
    if action < 34:
        return [DISCARD + action]
    if action in MORTAL_RED:
        return [DISCARD + MORTAL_RED[action]]
    if action == MORTAL_RIICHI:
        return list(range(RIICHI_DISCARD, TSUMO))
    if action == MORTAL_CHI_LOW:
        # Mortal's low chi has the called tile lowest in the sequence,
        # which is the sequence that starts at the claimed tile: ours calls
        # that high. The names cross over.
        return [CHII_HIGH]
    if action == MORTAL_CHI_MID:
        return [CHII_MIDDLE]
    if action == MORTAL_CHI_HIGH:
        return [CHII_LOW]
    if action == MORTAL_PON:
        return [PON]
    if action == MORTAL_KAN:
        return [CLAIMED_KAN, CONCEALED_KAN, EXTENDED_KAN]
    if action == MORTAL_AGARI:
        return [TSUMO, RON]
    if action == MORTAL_PASS:
        return [PASS]
    return []


# Mortal's action by ours: the rank of our action among the meanings of
# Mortal's, and infinity where it is no meaning of it. Lets a whole step's
# rows be translated at once rather than a row at a time.
MORTAL_ACTIONS = 46
PRIORITY = np.full((MORTAL_ACTIONS, ACTIONS), np.inf, dtype=np.float32)
for _action in range(MORTAL_ACTIONS):
    for _rank, _ours in enumerate(meanings(_action)):
        PRIORITY[_action, _ours] = _rank
MEANS = np.isfinite(PRIORITY)
# As float32, because NumPy multiplies float matrices through BLAS and
# integer ones through a plain loop that is many times slower.
MEANS_BY_OURS = np.ascontiguousarray(MEANS.T, dtype=np.float32)


def translatable(legal: np.ndarray) -> np.ndarray:
    """Which of Mortal's actions our engine allows at each row: `legal`
    is (rows, 78) and the answer (rows, 46)."""
    legal = np.atleast_2d(legal)
    return (legal.astype(np.float32) @ MEANS_BY_OURS) > 0.5


def first_meaning(actions: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """For each row, the best legal one of ours that Mortal's action means,
    or -1 when it means nothing legal there."""
    ranked = np.where(np.atleast_2d(legal), PRIORITY[actions], np.inf)
    best = ranked.argmin(axis=1)
    found = np.isfinite(ranked[np.arange(len(best)), best])
    return np.where(found, best, -1)


def translate(action: int, legal: np.ndarray) -> list[int]:
    """Our actions that Mortal's `action` could mean, best first, kept to
    the legal ones. Empty when none is legal, and for riichi, which is
    decided in a second step."""
    if action == MORTAL_RIICHI:
        return []
    return [index for index in meanings(action) if legal[index]]


def choose_in_mortal_space(
    ask,
    views: Views,
    rows: np.ndarray,
    players: np.ndarray,
    legal: np.ndarray,
    stats: dict | None = None,
) -> np.ndarray:
    """One of our engine's actions per row, from values in Mortal's own
    action space.

    `ask(who, fresh)` answers with those values and Mortal's mask for the
    positions named. A riichi there names no tile, so when one is chosen
    the reach is told to the follower and the same question asked again,
    from the state in which it is declared, and the tile is decided by the
    discards of that second answer.
    """
    who = list(zip(np.asarray(rows).tolist(), np.asarray(players).tolist()))
    legal = np.atleast_2d(legal)
    values, own = ask(who, False)
    allowed = own & translatable(legal)
    orphan = ~allowed.any(axis=1)
    allowed[orphan, MORTAL_PASS] = True
    ranked = np.where(allowed, values, -np.inf)
    best = ranked.argmax(axis=1)
    if stats is not None:
        stats["orphans"] = stats.get("orphans", 0) + int(orphan.sum())
        # A fallback is its own first choice not being a move here.
        stats["fallbacks"] = stats.get("fallbacks", 0) + int(
            ((np.where(own, values, -np.inf).argmax(axis=1) != best) & ~orphan).sum()
        )
    choice = first_meaning(best, legal)
    choice = np.where(orphan | (choice < 0), legal.argmax(axis=1), choice)

    second = np.nonzero((best == MORTAL_RIICHI) & ~orphan)[0]
    if len(second):
        follower = views.observer.follower
        for i in second:
            game, player = who[i]
            follower.tell(game, player, json.dumps({"type": "reach", "actor": player}))
        after, _own_after = ask([who[i] for i in second], True)
        tiles = legal[second, RIICHI_DISCARD:TSUMO]
        ranked_tiles = np.where(tiles, after[:, :riichi_py.POSITIONS], -np.inf)
        choice[second] = RIICHI_DISCARD + ranked_tiles.argmax(axis=1)
    return choice.astype(np.int64)


class MortalSpacePlayer:
    """One of ours whose moves are Mortal's, played the same way.

    It reads the same planes and answers over the same forty-six actions,
    so nothing is translated: the only difference from a Mortal is which
    network is asked.
    """

    actions = MORTAL_ACTIONS

    def __init__(self, net, device: str = "cuda", compile: bool = False) -> None:
        self.net = net.eval()
        self.forward = torch.compile(net, dynamic=True) if compile else net
        self.device = device
        self.kind = net.kind
        self.fallbacks = 0
        self.orphans = 0

    def eval(self) -> MortalSpacePlayer:
        self.net.eval()
        return self

    def parameters(self):
        return self.net.parameters()

    @property
    def channels(self) -> int:
        return self.net.channels

    @property
    def blocks(self) -> int:
        return self.net.blocks

    @property
    def planes(self) -> int:
        return self.net.planes

    @torch.no_grad()
    def _ask(
        self, views: Views, who: list[tuple[int, int]], fresh: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.array([game for game, _player in who], dtype=np.int64)
        players = np.array([player for _game, player in who], dtype=np.int64)
        sparse, masks = views.sparse_and_masks(rows, players, fresh=fresh)
        mask = torch.from_numpy(masks).to(self.device)
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")
        ):
            logits, _value = self.forward(sparse.dense(self.device), mask)
        return logits.float().cpu().numpy(), masks

    @torch.no_grad()
    def choose(
        self, views: Views, rows: np.ndarray, players: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        stats: dict = {}
        choice = choose_in_mortal_space(
            lambda who, fresh: self._ask(views, who, fresh), views, rows, players, legal, stats
        )
        self.orphans += stats.get("orphans", 0)
        self.fallbacks += stats.get("fallbacks", 0)
        return choice


class MortalPlayer:
    """A published Mortal, choosing moves in our action space."""

    kind = "mortal"

    def __init__(self, path: Path | str, device: str = "cuda", compile: bool = False) -> None:
        self.net = mortal_model.load(path, device)
        # Compiled, the forty blocks' batch-norm, Mish and attention fuse
        # into a few kernels a block instead of a dozen; the batch varies
        # from step to step, so the shape is left symbolic.
        self.forward = torch.compile(self.net, dynamic=True) if compile else self.net
        self.device = device
        self.path = str(path)
        # How often the answer had to fall back on a later choice of
        # Mortal's, and how often nothing of Mortal's was legal at all.
        self.fallbacks = 0
        self.orphans = 0

    def eval(self) -> MortalPlayer:
        return self

    def _ask(
        self, views: Views, who: list[tuple[int, int]], fresh: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mortal's Q values for those players, in its own action space,
        and its own mask of what it believes it may do. From the step's
        shared encoding unless a fresh one is asked for."""
        rows = np.array([game for game, _player in who], dtype=np.int64)
        players = np.array([player for _game, player in who], dtype=np.int64)
        sparse, masks = views.sparse_and_masks(rows, players, fresh=fresh)
        planes = sparse.dense(self.device)
        mask = torch.from_numpy(masks).to(self.device)
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")
        ):
            q = self.forward(planes, mask)
        return q.float().cpu().numpy(), masks

    @torch.no_grad()
    def choose(
        self, views: Views, rows: np.ndarray, players: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        """One of our actions per row, for `players[i]` in game `rows[i]`,
        given our engine's legal mask for each: the best of Mortal's
        actions that our engine allows, translated."""
        stats: dict = {}
        choice = choose_in_mortal_space(
            lambda who, fresh: self._ask(views, who, fresh), views, rows, players, legal, stats
        )
        self.orphans += stats.get("orphans", 0)
        self.fallbacks += stats.get("fallbacks", 0)
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


def load_player(
    path: Path | str,
    device: str,
    channels: int | None = None,
    blocks: int | None = None,
    compile: bool = False,
):
    """A player from a checkpoint of either kind: one of ours, rebuilt at
    the shape it says, or a Mortal, told apart by what the file holds.
    With `compile`, the forward it plays with is compiled, the batch's
    size left symbolic."""
    from .model import from_payload

    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        payload = None
    if payload is not None and "combined" in payload:
        # A joined player: our network and a Mortal beneath a fusion head.
        from . import combined

        net, _state = combined.load(path, device)
        net.eval()
        if compile:
            net.backbones_forward = torch.compile(net.backbones, dynamic=True)
        return net
    if payload is not None and "model" in payload:
        net = from_payload(payload, device, channels, blocks)
        net.eval()
        if getattr(net, "speaks_mortal", False):
            # Its moves are Mortal's, so it is asked for them Mortal's way.
            return MortalSpacePlayer(net, device, compile=compile)
        if compile:
            net.forward = torch.compile(net.forward, dynamic=True)
        return net
    return MortalPlayer(path, device, compile=compile)
