"""What a network reads and answers in, and who can serve it.

A search has to put positions to a network and turn its answers back into
moves the engine will accept. Both halves depend on things about the
network that nothing used to write down: which planes it reads, which
action space it answers in, whether its moves need translating, whether it
has a critic to value a leaf with at all.

`neural.searched` assumed all four. It reshaped the engine's ninety-seven
planes, read the answer as one of our seventy-eight actions, and rebuilt
the checkpoint with `from_payload`, which takes the `model` entry and
ignores everything else. Hand it a network of the current lineage, which
reads Mortal's thousand and twelve and answers in Mortal's forty-six, and
it ran anyway: one backbone out of two, the wrong planes, the wrong action
space, and a placement figure at the end that looked like a measurement.

So the two are written down here, and a server is chosen from them. The
contract is asked once, at the root and at every imagined continuation
alike, because a search that serves the root correctly and the
continuations some other way is measuring a network that does not exist.

Two servers, and a combination that fits neither is refused by name rather
than approximated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import riichi_py

ENGINE_PLANES = riichi_py.PLANES
ENGINE_ACTIONS = riichi_py.ACTIONS
POSITIONS = riichi_py.POSITIONS


class UnsupportedSearchLayout(ValueError):
    """Nothing here can build this network's inputs for a search.

    A `ValueError` rather than an exit, so a caller can catch it before an
    arena is built and a tally is written; `neural.searched` re-exports it
    under this name.
    """


@dataclass(frozen=True)
class Contract:
    """What a checkpoint turned out to be."""

    #: Which observation it reads: "engine" for our ninety-seven planes,
    #: "mortal" for Mortal's thousand and twelve.
    reads: str
    planes: int
    #: How many moves it answers in. Ours is seventy-eight; Mortal's is
    #: forty-six and needs translating both ways.
    answers: int
    #: Whether its answers are already our engine's actions.
    speaks_our_moves: bool
    #: Whether it can value a position, which a search needs for its
    #: leaves, and whether it can read the opponents' hands, which a search
    #: needs to imagine worlds at all.
    has_value: bool
    has_belief: bool
    #: What the checkpoint actually held, so a loader that quietly built
    #: half of it can be told from one that built all of it.
    parts: tuple[str, ...]

    def describe(self) -> dict:
        return {
            "reads": self.reads,
            "planes": self.planes,
            "answers": self.answers,
            "speaks_our_moves": self.speaks_our_moves,
            "has_value": self.has_value,
            "has_belief": self.has_belief,
            "parts": list(self.parts),
        }


def unwrap(net):
    """The network that answers with `everything`: a player that only
    wraps one to be asked Mortal's way (`zoo.MortalSpacePlayer`) carries
    it as `net`, and the search asks the network, not the wrapper."""
    if not hasattr(net, "everything") and hasattr(getattr(net, "net", None), "everything"):
        return net.net
    return net


def of(net) -> Contract:
    """Reads the contract off a loaded player."""
    net = unwrap(net)
    kind = getattr(net, "kind", "engine")
    planes = getattr(net, "planes", None)
    planes = planes() if callable(planes) else planes
    if planes is None:
        planes = ENGINE_PLANES if kind == "engine" else 1012
    parts = ["policy"]
    if hasattr(net, "everything"):
        parts += ["value", "belief"]
    if hasattr(net, "ours") and hasattr(net, "mortal"):
        parts += ["ours", "mortal", "fusion"]
    return Contract(
        reads=kind,
        planes=int(planes),
        answers=int(getattr(net, "actions", ENGINE_ACTIONS)),
        speaks_our_moves=kind == "engine",
        has_value=hasattr(net, "everything"),
        has_belief=hasattr(net, "everything"),
        parts=tuple(parts),
    )


class EngineServed:
    """A network that reads our own planes and answers in our own moves.

    Everything it needs is on the arena already: the observation is the one
    the engine writes, and the lookahead's leaves come back in the same
    shape.
    """

    def __init__(self, net, contract: Contract) -> None:
        self.net = net
        self.contract = contract

    def root(self, arena, views, rows, deciding, legal, device):
        planes = np.frombuffer(arena.observations(), dtype=np.float32)
        planes = planes.reshape(-1, ENGINE_PLANES, POSITIONS)[rows]
        return (
            torch.from_numpy(planes.copy()).to(device),
            torch.from_numpy(legal[rows]).to(device),
        )

    def leaves(self, arena, leaf_bytes, counts, device):
        total = sum(counts)
        planes = np.frombuffer(leaf_bytes, dtype=np.float32)
        return torch.from_numpy(
            planes.reshape(total, ENGINE_PLANES, POSITIONS).copy()
        ).to(device)

    def to_engine(self, action: int, legal_row: np.ndarray, after_reach: bool = False) -> int:
        return int(action) if legal_row[action] else -1

    def to_engine_rows(self, own_order: np.ndarray, legal: np.ndarray) -> np.ndarray:
        """Every row's order named in our moves at once: its own, with
        the illegal ones marked -1."""
        rows = np.arange(len(own_order))[:, None]
        return np.where(legal[rows, own_order], own_order, -1)

    def order(self, logits) -> np.ndarray:
        return torch.argsort(logits, dim=1, descending=True).cpu().numpy()

    def value(self, planes, head: str = "critic"):
        """What the network makes of these positions.

        A network with a choice of heads is asked for the one named; one
        with a single value head answers with that, and saying otherwise
        would be inventing a distinction it does not have.
        """
        if hasattr(self.net, "value_only"):
            return self.net.value_only(planes, head=head)
        mask = torch.ones(
            planes.shape[0], self.contract.answers, dtype=torch.bool, device=planes.device
        )
        _logits, value, _hands = self.net.everything(planes, mask)
        return value


class MortalServed:
    """A network that reads Mortal's planes and answers in Mortal's moves.

    The root comes from the follower, which has been told everything that
    really happened. A leaf has no such history -- its world was imagined
    -- so the seat's real state is copied and advanced by the events that
    world invented, which `libriichi.follow.Imagined` does and which was
    checked against the follower's own answer to the last value.
    """

    def __init__(self, net, contract: Contract) -> None:
        from libriichi.follow import Imagined  # noqa: F401  (kept for the error)

        self.net = net
        self.contract = contract
        self._Imagined = Imagined

    def root(self, arena, views, rows, deciding, legal, device):
        from . import zoo

        planes, _own = views.sparse_and_masks(rows, deciding)
        allowed = zoo.translatable(legal[rows])
        allowed[~allowed.any(axis=1), zoo.MORTAL_PASS] = True
        return planes.dense(device), torch.from_numpy(allowed).to(device)

    def leaves(self, arena, leaf_bytes, counts, device):
        """Every leaf as Mortal sees it, built from its own continuation.

        A leaf whose world dealt a new hand offers no events -- its log
        replaced the one the cursor points into and its seats have moved --
        and is given zeros, which the engine ignores for the slots it does
        not want valued.
        """
        from .observe import Planes

        total = sum(counts)
        players, lines = arena.leaves_mjai()
        game_of = np.repeat(np.arange(len(counts)), counts)
        live = [at for at in range(total) if lines[at]]
        out = torch.zeros(total, self.contract.planes, POSITIONS, device=device)
        if not live:
            return out
        copies = self._Imagined.from_follower(
            arena_follower(arena),
            [(int(game_of[at]), int(players[at])) for at in live],
            4,
        )
        copies.feed([lines[at] for at in live])
        indptr, indices, values, _masks = copies.encode()
        built = Planes.from_follower(indptr, indices, values).dense(device)
        out[torch.tensor(live, device=device)] = built
        return out

    def to_engine(self, action: int, legal_row: np.ndarray, after_reach: bool = False) -> int:
        """One of Mortal's moves as one of ours, or -1 where it means
        nothing legal. A reach names no tile until the second question is
        asked (`after_reach`), so before that it is -1 rather than the
        lowest tile that happens to be legal, which is what a bare
        `first_meaning` would say and what nobody chose."""
        from . import zoo

        if after_reach:
            tile = action if action < 34 else zoo.MORTAL_RED.get(action, -1)
            if tile < 0:
                return -1
            index = zoo.RIICHI_DISCARD + tile
            return index if legal_row[index] else -1
        if action == zoo.MORTAL_RIICHI:
            return -1
        return int(zoo.first_meaning(np.array([action]), legal_row[None, :])[0])

    def to_engine_rows(self, own_order: np.ndarray, legal: np.ndarray) -> np.ndarray:
        """Every row's order in Mortal's moves named in ours at once, -1
        where a move means nothing legal there and for a reach, whose
        tile the second question decides (see `after_reach`)."""
        from . import zoo

        ranked = np.where(legal[:, None, :], zoo.PRIORITY[own_order], np.inf)
        best = ranked.argmin(axis=2)
        found = np.isfinite(np.take_along_axis(ranked, best[..., None], axis=2)[..., 0])
        found &= own_order != zoo.MORTAL_RIICHI
        return np.where(found, best, -1)

    def order(self, logits) -> np.ndarray:
        return torch.argsort(logits, dim=1, descending=True).cpu().numpy()

    def after_reach(self, arena, rows, deciding, legal, device):
        """The second question a reach asks, which tile to throw, put from
        a copy of each seat's real state told the declaration.

        A copy and not the follower: the table's player tells its follower
        the reach ahead of the table because it is about to make it, and
        the follower skips the duplicate when the table confirms it. A
        search has not decided yet, and a follower told of a reach that
        never happens carries it for the rest of the hand.
        """
        import json

        from . import zoo
        from .observe import Planes

        who = [(int(game), int(player)) for game, player in zip(rows, deciding)]
        copies = self._Imagined.from_follower(arena_follower(arena), who, 4)
        copies.feed([[json.dumps({"type": "reach", "actor": player})] for _game, player in who])
        indptr, indices, values, _masks = copies.encode()
        planes = Planes.from_follower(indptr, indices, values).dense(device)
        # What our engine allows the declaration to throw, named as Mortal
        # names a discard, as `zoo.choose_in_mortal_space` asks it.
        allowed = np.zeros((len(who), zoo.MORTAL_ACTIONS), dtype=bool)
        allowed[:, :POSITIONS] = legal[rows, zoo.RIICHI_DISCARD : zoo.TSUMO]
        return planes, torch.from_numpy(allowed).to(device)

    def value(self, planes, head: str = "critic"):
        """What the network makes of these positions.

        A network with a choice of heads is asked for the one named; one
        with a single value head answers with that, and saying otherwise
        would be inventing a distinction it does not have.
        """
        if hasattr(self.net, "value_only"):
            return self.net.value_only(planes, head=head)
        mask = torch.ones(
            planes.shape[0], self.contract.answers, dtype=torch.bool, device=planes.device
        )
        _logits, value, _hands = self.net.everything(planes, mask)
        return value


#: Where the follower lives on a `Views`. Set by `serve`, because the
#: server needs it and the arena does not carry one.
_FOLLOWERS: dict[int, object] = {}


def remember_follower(arena, follower) -> None:
    _FOLLOWERS[id(arena)] = follower


def arena_follower(arena):
    follower = _FOLLOWERS.get(id(arena))
    if follower is None:
        raise RuntimeError(
            "this search has no follower to copy a seat's state from; call "
            "contract.remember_follower(arena, views.observer.follower) first"
        )
    return follower


def serve(net, checkpoint: str = "the checkpoint"):
    """The server for this network, or an explicit refusal.

    A refusal names what the network reads and what is missing. It is not a
    limit to work around by padding an input or editing a constant: a
    search that cannot build the planes a network reads, for the root and
    for every continuation, cannot evaluate that network at all.
    """
    net = unwrap(net)
    contract = of(net)
    if not contract.has_value:
        raise UnsupportedSearchLayout(
            f"{checkpoint} has no value head, and a search values the positions its "
            "candidates lead to. Nothing here can search it."
        )
    if contract.reads == "engine":
        if contract.planes != ENGINE_PLANES or contract.answers != ENGINE_ACTIONS:
            # Engine planes with Mortal's moves is a layout nobody trained
            # and nothing serves: the engine's lookahead names our
            # seventy-eight, and the leaves it hands back have no Mortal
            # event history to translate the other way from.
            raise UnsupportedSearchLayout(
                f"{checkpoint} reads {contract.planes} planes and answers in "
                f"{contract.answers} moves; native lookahead serves {ENGINE_PLANES} "
                f"engine planes and {ENGINE_ACTIONS} actions, and its hypothetical "
                "leaves have no Mortal event history. It cannot be searched here; "
                "do not pad or relabel engine planes."
            )
        return EngineServed(net, contract)
    if contract.reads == "mortal":
        return MortalServed(net, contract)
    raise UnsupportedSearchLayout(
        f"{checkpoint} reads {contract.reads!r}, which no server here builds. "
        "Add one rather than approximating it with another's planes."
    )
