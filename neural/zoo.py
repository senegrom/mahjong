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

from . import mortal_model, policy_inference
from .population import CLUB
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
#
# The red fives (34 to 36) mean nothing at our tables, which have none:
# they are never opened in a mask and never produced. They used to be
# opened as aliases of the plain fives, which split a five's weight across
# two actions and let a network pick a move whose value it had never been
# trained on. `engine/riichi-wasm` translates the same way; the two must
# not drift.
MORTAL_RED_FIVES = (34, 35, 36)
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
    step; a red five and an abortive draw mean nothing, our rules offering
    neither."""
    if action < 34:
        return [DISCARD + action]
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

    `ask(who, fresh, allowed)` scores using the core-derived mask and returns
    those values plus Mortal's original mask for the
    positions named. A riichi there names no tile, so when one is chosen
    the reach is told to the follower and the same question asked again,
    from the state in which it is declared, and the tile is decided by the
    discards of that second answer.

    A row with a single move open is not asked at all, at either step: the
    best of one move is that move, whatever the values, so its answer is
    the one asking would give. About one decision in fifteen is such a row,
    a declared riichi's draw most of all, and a step whose rows are all
    such asks nothing. `stats` counts a fallback only among the rows asked,
    since nothing was chosen in the others.
    """
    who = list(zip(np.asarray(rows).tolist(), np.asarray(players).tolist()))
    legal = np.atleast_2d(legal)
    # What our engine allows, named in Mortal's moves, and not what the two
    # rule sets agree on: Mortal will not reach with fewer than four tiles
    # left in the wall, and EMA 2025 section 3.3.10 allows it down to one.
    # `own` is still read below, to say how often Mortal would have chosen
    # otherwise.
    allowed = translatable(legal)
    orphan = ~allowed.any(axis=1)
    allowed[orphan, MORTAL_PASS] = True
    # The one open move of a row that has only one, and the asked rows'
    # best below.
    best = allowed.argmax(axis=1)
    asked = np.flatnonzero(allowed.sum(axis=1) > 1)
    if len(asked):
        values, own = ask([who[i] for i in asked], False, allowed[asked])
        best[asked] = np.where(allowed[asked], values, -np.inf).argmax(axis=1)
        if stats is not None:
            # A fallback is its own first choice not being a move here.
            stats["fallbacks"] = stats.get("fallbacks", 0) + int(
                (np.where(own, values, -np.inf).argmax(axis=1) != best[asked]).sum()
            )
    if stats is not None:
        stats["orphans"] = stats.get("orphans", 0) + int(orphan.sum())
    choice = first_meaning(best, legal)
    choice = np.where(orphan | (choice < 0), legal.argmax(axis=1), choice)

    second = np.nonzero((best == MORTAL_RIICHI) & ~orphan)[0]
    if len(second):
        follower = views.observer.follower
        for i in second:
            game, player = who[i]
            follower.tell(game, player, json.dumps({"type": "reach", "actor": player}))
        tiles = legal[second, RIICHI_DISCARD:TSUMO]
        # Likewise a reach that only one tile keeps ready.
        tile = tiles.argmax(axis=1)
        open_tiles = np.flatnonzero(tiles.sum(axis=1) > 1)
        if len(open_tiles):
            allowed_after = np.zeros((len(open_tiles), MORTAL_ACTIONS), dtype=bool)
            allowed_after[:, :34] = tiles[open_tiles]
            after, _own_after = ask([who[second[i]] for i in open_tiles], True, allowed_after)
            ranked_tiles = np.where(tiles[open_tiles], after[:, :riichi_py.POSITIONS], -np.inf)
            tile[open_tiles] = ranked_tiles.argmax(axis=1)
        choice[second] = RIICHI_DISCARD + tile
    return choice.astype(np.int64)


def unwrap(net):
    """The network that answers with `everything`: a player that only wraps
    one to be asked Mortal's way (`MortalSpacePlayer`) carries it as `net`,
    and whoever wants the network's own answers rather than the player's
    move, as the browser-parity replay does, asks the network."""
    if not hasattr(net, "everything") and hasattr(getattr(net, "net", None), "everything"):
        return net.net
    return net


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
        self, views: Views, who: list[tuple[int, int]], fresh: bool, allowed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.array([game for game, _player in who], dtype=np.int64)
        players = np.array([player for _game, player in who], dtype=np.int64)
        sparse, masks = views.sparse_and_masks(rows, players, fresh=fresh)
        mask = torch.from_numpy(allowed).to(self.device)
        with torch.no_grad(), policy_inference.autocast(self.device, MORTAL_ACTIONS):
            logits, _value = self.forward(sparse.dense(self.device), mask)
        return logits.float().cpu().numpy(), masks

    @torch.no_grad()
    def choose(
        self, views: Views, rows: np.ndarray, players: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        stats: dict = {}
        choice = choose_in_mortal_space(
            lambda who, fresh, allowed: self._ask(views, who, fresh, allowed), views, rows, players, legal, stats
        )
        self.orphans += stats.get("orphans", 0)
        self.fallbacks += stats.get("fallbacks", 0)
        return choice


class ClubPlayer:
    """The engine's Club-tier heuristic player, seated among the others by
    the name `club`. The arena plays its places itself (`table_bots`), so
    it is never asked for a move and needs no view of the table."""

    kind = "club"

    def eval(self) -> ClubPlayer:
        return self


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
        self, views: Views, who: list[tuple[int, int]], fresh: bool, allowed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mortal's Q values for those players, in its own action space,
        and its own mask of what it believes it may do. From the step's
        shared encoding unless a fresh one is asked for."""
        rows = np.array([game for game, _player in who], dtype=np.int64)
        players = np.array([player for _game, player in who], dtype=np.int64)
        sparse, masks = views.sparse_and_masks(rows, players, fresh=fresh)
        planes = sparse.dense(self.device)
        mask = torch.from_numpy(allowed).to(self.device)
        with torch.no_grad(), policy_inference.autocast(self.device, MORTAL_ACTIONS):
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
            lambda who, fresh, allowed: self._ask(views, who, fresh, allowed), views, rows, players, legal, stats
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
    with policy_inference.autocast(device, int(legal.shape[1])):
        logits, _value = player(
            views.dense(player.kind, rows, players, device),
            torch.from_numpy(legal).to(device),
        )
    if greedy:
        return logits.argmax(dim=1).cpu().numpy()
    return torch.distributions.Categorical(logits=logits.float()).sample().cpu().numpy()


def play_in_bfloat16(player, device: str):
    """`player`, with its network's weights cast to bfloat16 once (see
    `policy_inference.precast`) when it plays on the card in bfloat16: a
    Mortal, one of ours that answers in Mortal's moves, or a joined
    player. One of ours over the engine's seventy-eight moves plays in
    float32 and is left as it is, as is the heuristic player, which has no
    network. For a player that is only ever asked for moves: its network
    can no longer run in float32."""
    if policy_inference.precision(device, getattr(player, "actions", MORTAL_ACTIONS)) == "bfloat16":
        network = getattr(player, "net", player)
        if isinstance(network, torch.nn.Module):
            policy_inference.precast(network)
    return player


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
    size left symbolic. `club` names the engine's heuristic player, never
    a file."""
    from .model import from_payload

    if str(path) == CLUB:
        return ClubPlayer()
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
