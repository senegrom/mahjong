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

import contextlib

import numpy as np
import torch

import riichi_py

from . import inference

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
    reader_planes: int | None = None

    def describe(self) -> dict:
        return {
            "reads": self.reads,
            "planes": self.planes,
            "answers": self.answers,
            "speaks_our_moves": self.speaks_our_moves,
            "has_value": self.has_value,
            "has_belief": self.has_belief,
            "parts": list(self.parts),
            "world_reader": self.reads if self.reader_planes is not None else "uniform",
            "reader_planes": self.reader_planes,
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
        speaks_our_moves=int(getattr(net, "actions", ENGINE_ACTIONS)) == ENGINE_ACTIONS,
        has_value=hasattr(net, "everything"),
        has_belief=hasattr(net, "everything"),
        parts=tuple(parts),
        reader_planes=int(planes) if callable(getattr(net, "read_plausibility", None)) else None,
    )



@torch.no_grad()
def proposal_scores(served, arena, hands_bytes, counts, device, batch_size=256):
    """Serve the reader's OWN layout, in bounded batches and native world order.

    Uniform weights are an explicit capability fallback, never a side effect of
    the observation layout. The hands stay in the encoder's relative-seat order.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("reader batch size must be positive")
    total = sum(counts)
    if getattr(served.contract, "reader_planes", None) is None:
        return np.zeros(total, dtype=np.float32)
    if served.contract.reader_planes != served.contract.planes:
        raise UnsupportedSearchLayout("reader observation layout does not match its server")
    hands = np.frombuffer(hands_bytes, dtype=np.float32).reshape(total, riichi_py.HIDDEN_HANDS_PLANES, POSITIONS)
    game_of = np.repeat(np.arange(len(counts)), counts)
    result = np.empty(total, dtype=np.float32)
    public = None
    if served.contract.reads == "engine":
        public = np.frombuffer(arena.observations(), dtype=np.float32).reshape(len(counts), ENGINE_PLANES, POSITIONS)
    elif served.contract.reads == "mortal":
        from .observe import Planes
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(len(counts), 4)
        follower = arena_follower(arena)
    else:
        raise UnsupportedSearchLayout("no encoder for this reader")
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        games = game_of[start:stop]
        if public is not None:
            planes = torch.from_numpy(public[games].copy()).to(device)
        else:
            unique, inverse = np.unique(games, return_inverse=True)
            if np.any(seats[unique] >= 4):
                raise ValueError("proposals came from a game with no deciding seat")
            who = [(int(game), int(players[game, seats[game]])) for game in unique]
            indptr, indices, values, _masks = follower.encode(who)
            planes = Planes.from_follower(indptr, indices, values).rows(inverse).dense(device)
        shown = torch.from_numpy(hands[start:stop].copy()).to(device)
        said = served.net.read_plausibility(planes, shown).float().cpu().numpy()
        if said.shape != (stop - start,) or not np.isfinite(said).all():
            raise ValueError("world reader must return one finite log weight per proposal")
        result[start:stop] = said
    return result


DEFAULT_LEAF_BATCH = 256


def leaf_slots(counts, wanted, batch_size):
    """Validate native leaf metadata and return the slots needing a value."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("leaf_batch must be a positive integer")
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("Native leaf counts must be nonnegative integers")
    total = sum(counts)
    if wanted is None:
        raise UnsupportedSearchLayout("Continuations require the native wanted-value mask")
    raw = (np.frombuffer(wanted, dtype=np.uint8)
           if isinstance(wanted, (bytes, bytearray, memoryview)) else np.asarray(wanted))
    if raw.shape != (total,) or not np.isin(raw, (0, 1)).all():
        raise ValueError("Native wanted-value mask has the wrong shape or values")
    return np.flatnonzero(raw)

class EngineServed:
    """A network that reads our own planes and answers in our own moves.

    Everything it needs is on the arena already: the observation is the one
    the engine writes, and the lookahead's leaves come back in the same
    shape.
    """

    def __init__(self, net, contract: Contract) -> None:
        self.net = net
        self.contract = contract
        self.placement_head = None

    def root(self, arena, views, rows, deciding, legal, device):
        planes = np.frombuffer(arena.observations(), dtype=np.float32)
        planes = planes.reshape(-1, ENGINE_PLANES, POSITIONS)[rows]
        return (
            torch.from_numpy(planes.copy()).to(device),
            torch.from_numpy(legal[rows]).to(device),
        )

    def leaf_batches(self, arena, leaf_bytes, counts, device, *, wanted, batch_size=DEFAULT_LEAF_BATCH):
        """Transfer only one bounded batch of wanted native leaves at a time."""
        live = leaf_slots(counts, wanted, batch_size)
        planes = np.frombuffer(leaf_bytes, dtype=np.float32).reshape(
            sum(counts), ENGINE_PLANES, POSITIONS
        )
        for start in range(0, len(live), batch_size):
            slots = live[start:start + batch_size]
            yield slots, torch.from_numpy(planes[slots].copy()).to(device)

    def leaves(self, arena, leaf_bytes, counts, device, *, wanted=None):
        """Small-call compatibility helper; search itself uses leaf_batches."""
        out = torch.zeros(sum(counts), ENGINE_PLANES, POSITIONS, device=device)
        for slots, planes in self.leaf_batches(arena, leaf_bytes, counts, device, wanted=wanted):
            out[torch.as_tensor(slots, device=device)] = planes
        return out

    def to_engine(self, action: int, legal_row: np.ndarray, after_reach: bool = False) -> int:
        return int(action) if legal_row[action] else -1

    def to_engine_rows(self, own_order: np.ndarray, legal: np.ndarray) -> np.ndarray:
        """Every row's order named in our moves at once: its own, with
        the illegal ones marked -1."""
        rows = np.arange(len(own_order))[:, None]
        return np.where(legal[rows, own_order], own_order, -1)

    def order(self, logits) -> np.ndarray:
        return inference.order(logits).cpu().numpy()

    def value(self, planes, head: str = "critic"):
        """What the network makes of these positions.

        A network with a choice of heads is asked for the one named; one
        with a single value head answers with that, and saying otherwise
        would be inventing a distinction it does not have. `placement` is
        none of the network's heads but the placement-only judge served
        beside it (`neural.placement`), asked on the same features.
        """
        if head == "placement":
            return placement_value(self, planes)
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
        # Retain upstream's opt-in count of reconstructed hand boundaries.
        self.count_crossings = False
        self.crossed = 0
        self.placement_head = None

    def root(self, arena, views, rows, deciding, legal, device):
        from . import zoo

        planes, _own = views.sparse_and_masks(rows, deciding)
        allowed = (legal[rows].copy() if self.contract.speaks_our_moves
                   else zoo.translatable(legal[rows]))
        if not allowed.any(axis=1).all():
            raise ValueError("Search roots must each have a legal action")
        return planes.dense(device), torch.from_numpy(allowed).to(device)

    def leaf_batches(self, arena, leaf_bytes, counts, device, *, wanted, batch_size=DEFAULT_LEAF_BATCH):
        """Reconstruct, encode, densify and transfer bounded batches together.

        Only root-hand positions reach the hybrid critic. Completed worlds
        already contain root-hand reward plus terminal placement and need
        neither a synthetic zero observation nor another network forward.
        """
        from .observe import Planes

        live = leaf_slots(counts, wanted, batch_size)
        total = sum(counts)
        players, lines = arena.leaves_mjai()
        if len(players) != total or len(lines) != total:
            raise ValueError("Native continuation metadata does not match the leaf count")
        if any(not lines[at] for at in live):
            raise UnsupportedSearchLayout(
                "nonterminal search leaves have no reconstructible Mortal history; "
                "refusing invented zero observations"
            )
        boundaries = np.cumsum(counts)
        for start in range(0, len(live), batch_size):
            slots = live[start:start + batch_size]
            game_of = np.searchsorted(boundaries, slots, side="right")
            copies = self._Imagined.from_follower(
                arena_follower(arena),
                [(int(game), int(players[at])) for game, at in zip(game_of, slots)],
                4,
            )
            copies.feed([lines[at] for at in slots])
            indptr, indices, values, _masks = copies.encode()
            del copies
            yield slots, Planes.from_follower(indptr, indices, values).dense(device)

    def leaves(self, arena, leaf_bytes, counts, device, *, wanted=None):
        """Small-call compatibility helper; large searches use leaf_batches."""
        # Validate before allocating, including when every slot is terminal.
        leaf_slots(counts, wanted, DEFAULT_LEAF_BATCH)
        out = torch.zeros(sum(counts), self.contract.planes, POSITIONS, device=device)
        for slots, planes in self.leaf_batches(arena, leaf_bytes, counts, device, wanted=wanted):
            out[torch.as_tensor(slots, device=device)] = planes
        return out

    def to_engine(self, action: int, legal_row: np.ndarray, after_reach: bool = False) -> int:
        """One of Mortal's moves as one of ours, or -1 where it means
        nothing legal. A reach names no tile until the second question is
        asked (`after_reach`), so before that it is -1 rather than the
        lowest tile that happens to be legal, which is what a bare
        `first_meaning` would say and what nobody chose."""
        from . import zoo

        if self.contract.speaks_our_moves:
            return EngineServed.to_engine(self, action, legal_row, after_reach)
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

        if self.contract.speaks_our_moves:
            return EngineServed.to_engine_rows(self, own_order, legal)
        ranked = np.where(legal[:, None, :], zoo.PRIORITY[own_order], np.inf)
        best = ranked.argmin(axis=2)
        found = np.isfinite(np.take_along_axis(ranked, best[..., None], axis=2)[..., 0])
        found &= own_order != zoo.MORTAL_RIICHI
        return np.where(found, best, -1)

    def order(self, logits) -> np.ndarray:
        return inference.order(logits).cpu().numpy()

    def after_reach(self, arena, rows, deciding, legal, device, follower=None):
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
        follower = follower if follower is not None else arena_follower(arena)
        copies = self._Imagined.from_follower(follower, who, 4)
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
        would be inventing a distinction it does not have. `placement` is
        none of the network's heads but the placement-only judge served
        beside it (`neural.placement`), asked on the same features.
        """
        if head == "placement":
            return placement_value(self, planes)
        if hasattr(self.net, "value_only"):
            return self.net.value_only(planes, head=head)
        mask = torch.ones(
            planes.shape[0], self.contract.answers, dtype=torch.bool, device=planes.device
        )
        _logits, value, _hands = self.net.everything(planes, mask)
        return value

    def play_lookahead(self, arena, device="cuda", temperature=0.0, passes=8000,
                       batch_size=inference.DEFAULT_ROLLOUT_BATCH) -> int:
        """The network moving every seat inside the lookahead, on Mortal's
        planes.

        A copy of each seat's state is kept per slot -- four per imagined
        world and candidate -- and told what that slot's world invents,
        hand boundaries included, as the engine streams it; the seat that
        owes a decision is encoded from its copy and answers in Mortal's
        moves, translated back to ours under the engine's legality. A
        reach is put its second question from a further copy, so the
        slot's own copies are never told a reach the engine then plays
        differently or not at all. Returns how many passes it took.
        """
        import json

        from . import zoo
        from .observe import Planes

        inference.validate_batch(batch_size)
        net = self.net
        follower = arena_follower(arena)
        base: dict[tuple[int, int], int] = {}
        who: list[tuple[int, int]] = []
        for game, count in enumerate(arena.lookahead_slots()):
            for slot in range(count):
                base[(game, slot)] = len(who)
                who.extend((game, player) for player in range(4))
        if not who:
            return 0
        copies = self._Imagined.from_follower(follower, who, 4)
        # Every copy holds what its world dealt that seat as the search
        # began, before the candidate: the seats the searcher cannot see
        # hold the world's tiles, and the searcher's own are the same in
        # every world.
        which: list[int] = []
        hands: list[list[str]] = []
        furiten: list[bool] = []
        initial = [state for per_game in arena.lookahead_initial_state() for state in per_game]
        for ((game, slot), dealt, (observer, flags)) in zip(base, (
            by_player for per_game in arena.lookahead_hands() for by_player in per_game
        ), initial):
            for player in range(4):
                if player == observer:
                    continue  # Preserve every field of the observer's known private state.
                which.append(base[(game, slot)] + player)
                hands.append(list(dealt[player]))
                furiten.append(flags[player])
        copies.replace_concealed(which, hands)
        copies.reset_search_private(which, furiten)
        needs_initial_decision = set(which)
        taken = 0
        while taken < passes:
            games, slots, players, masks_bytes, lines = arena.lookahead_owed_mjai()
            count = len(games)
            if self.count_crossings:
                self.crossed += sum('"start_kyoku"' in line for new in lines for line in new)
            if count == 0:
                break
            taken += 1
            masks = np.frombuffer(masks_bytes, dtype=np.uint8).reshape(count, ENGINE_ACTIONS)
            masks = masks.astype(bool)
            # Every seat's copy of each owed slot learns what its world did.
            which: list[int] = []
            told: list[list[str]] = []
            for game, slot, new in zip(games, slots, lines):
                if not new:
                    continue
                start = base[(game, slot)]
                for player in range(4):
                    which.append(start + player)
                    told.append(new)
            if which:
                copies.feed_some(which, told)
                needs_initial_decision.difference_update(which)
            deciding = [base[(game, slot)] + int(player) for game, slot, player in zip(games, slots, players)]
            # A root call can owe another response before any new event exists.
            # Only that initial reaction needs rebuilding explicitly. All later
            # action features come from normal events on the sampled state, not
            # expensive repeated reconstruction or changes to known observations.
            empty = [at for at, copy in enumerate(deciding) if copy in needs_initial_decision]
            if empty:
                flags, drawn = arena.lookahead_decision_state()
                copies.synchronize_search_decisions([deciding[at] for at in empty],
                    masks[empty].tolist(), [flags[at] for at in empty], [drawn[at] for at in empty])
                needs_initial_decision.difference_update(deciding[at] for at in empty)
            # What our engine allows, named in Mortal's moves, as the table
            # asks it (`zoo.choose_in_mortal_space`).
            allowed = masks if self.contract.speaks_our_moves else zoo.translatable(masks)
            if not allowed.any(axis=1).all():
                raise UnsupportedSearchLayout("An imagined decision has no translatable legal move")
            best = np.empty(count, dtype=np.int64)
            for start in range(0, count, batch_size):
                rows = np.arange(start, min(start + batch_size, count))
                indptr, indices, values, _own = copies.encode_some([deciding[at] for at in rows])
                planes = Planes.from_follower(indptr, indices, values)
                logits = policy_logits(net,
                    planes.dense(device), torch.from_numpy(allowed[rows]).to(device)
                )
                logits = logits.float()
                if temperature > 0:
                    odds = torch.softmax(logits / temperature, dim=1)
                    picked = torch.multinomial(odds, 1).squeeze(1)
                else:
                    picked = logits.argmax(dim=1)
                best[rows] = picked.cpu().numpy()
            actions = best.copy() if self.contract.speaks_our_moves else zoo.first_meaning(best, masks)
            if np.any(actions < 0):
                raise UnsupportedSearchLayout("The imagined policy chose an untranslatable action")
            second = (np.empty(0, dtype=np.int64) if self.contract.speaks_our_moves
                      else np.nonzero(best == zoo.MORTAL_RIICHI)[0])
            for start in range(0, len(second), batch_size):
                group = second[start:start + batch_size]
                asked = copies.clone_some([deciding[at] for at in group])
                asked.feed(
                    [[json.dumps({"type": "reach", "actor": int(players[at])})] for at in group]
                )
                indptr, indices, values, _masks = asked.encode()
                after = Planes.from_follower(indptr, indices, values).dense(device)
                tiles = masks[group, zoo.RIICHI_DISCARD : zoo.TSUMO]
                allowed_after = np.zeros((len(group), zoo.MORTAL_ACTIONS), dtype=bool)
                allowed_after[:, :POSITIONS] = tiles
                logits_after = policy_logits(net, after, torch.from_numpy(allowed_after).to(device))
                logits_after = logits_after.float()[:, :POSITIONS]
                ranked = torch.where(
                    torch.from_numpy(tiles).to(device), logits_after, torch.full_like(logits_after, -np.inf)
                )
                if temperature > 0:
                    tile = torch.multinomial(torch.softmax(ranked / temperature, dim=1), 1).squeeze(1)
                else:
                    tile = ranked.argmax(dim=1)
                actions[group] = zoo.RIICHI_DISCARD + tile.cpu().numpy()
            arena.lookahead_apply(actions.tolist())
        # The native leaf boundary rejects remaining/broken worlds, without
        # consuming the lookahead. A caller may continue with a larger budget.
        return taken


def policy_logits(net, planes, legal):
    """Use a verified fast path where available, without changing old models."""
    return inference.policy_logits(net, planes, legal)


def root_order(served, arena, views, rows, deciding, mask, device):
    """The network's order over our moves at the root, best first, for the
    live games, and what else its one pass gave: the logits in its own
    moves, the value, and its reading of the hands.

    A network of Mortal's lineage answers in Mortal's forty-six and the
    engine plays seventy-eight, so its order is named in ours; a reach
    there names no tile, and is put the second question the table puts
    it wherever one is legal, so the riichi discards take the reach's
    place in the order in the tile order of that answer. Every row of
    `order` is a full order of distinct moves -- the network's first, then
    the legal moves it did not reach, then the rest -- so a caller may read
    only its head. `arena` may be None when no reach's second question
    can arise for an engine-planes network; a Mortal-planes one needs it
    for the follower its copies come from.
    """
    from . import zoo

    net = served.net
    games = mask.shape[0]
    root_planes, root_mask = served.root(arena, views, rows, deciding, mask, device)
    logits, value, guessed = inference.everything(net, root_planes, root_mask)
    own_order = served.order(logits.float())
    named = served.to_engine_rows(own_order, mask[rows])
    reach_at: dict[int, int] = {}
    tile_order = None
    if not served.contract.speaks_our_moves:
        may_reach = mask[rows, zoo.RIICHI_DISCARD : zoo.TSUMO].any(axis=1)
        second = np.nonzero(may_reach)[0]
        if len(second):
            # The follower the copies come from: the arena's, or the one
            # the views hold when a player asks without an arena of its own.
            follower = views.observer.follower if views is not None and views.observer is not None else None
            after_planes, after_mask = served.after_reach(
                arena, rows[second], deciding[second], mask, device, follower=follower
            )
            after_logits, _after_value, _after_hands = inference.everything(net, after_planes, after_mask)
            tile_order = (
                inference.order(after_logits[:, :POSITIONS])
                .cpu()
                .numpy()
            )
            reach_at = {int(at): k for k, at in enumerate(second)}
    order = np.zeros((games, ENGINE_ACTIONS), dtype=np.int64)
    every = np.arange(ENGINE_ACTIONS)
    for at, game in enumerate(rows):
        ours = named[at]
        if at in reach_at:
            where = np.nonzero(own_order[at] == zoo.MORTAL_RIICHI)[0]
            riichi = zoo.RIICHI_DISCARD + tile_order[reach_at[at]]
            riichi = riichi[mask[game][riichi]]
            if len(where):
                ours = np.concatenate([ours[: where[0]], riichi, ours[where[0] + 1 :]])
        ours = ours[ours >= 0]
        _first, at_first = np.unique(ours, return_index=True)
        ours = ours[np.sort(at_first)]
        seen = np.zeros(ENGINE_ACTIONS, dtype=bool)
        seen[ours] = True
        rest = every[~seen]
        order[game] = np.concatenate([ours, rest[mask[game][rest]], rest[~mask[game][rest]]])
    return order, logits, value, guessed


#: Where the follower lives on a `Views`, by the arena it follows. Set by
#: the search, because the server needs it and the arena does not carry
#: one. The arena is kept beside its follower so that its identity cannot
#: pass to another arena while the entry stands, and whoever remembers a
#: follower forgets it when the search is over (`following`).
_FOLLOWERS: dict[int, tuple[object, object]] = {}


def remember_follower(arena, follower) -> None:
    _FOLLOWERS[id(arena)] = (arena, follower)


def forget_follower(arena) -> None:
    entry = _FOLLOWERS.get(id(arena))
    if entry is not None and entry[0] is arena:
        del _FOLLOWERS[id(arena)]


@contextlib.contextmanager
def following(arena, follower):
    """The follower remembered for the arena while a search runs and
    forgotten after, whatever happens: an entry left behind would hold a
    finished arena and its follower for good."""
    if follower is None:
        yield
        return
    remember_follower(arena, follower)
    try:
        yield
    finally:
        forget_follower(arena)


def arena_follower(arena):
    entry = _FOLLOWERS.get(id(arena))
    if entry is None or entry[0] is not arena:
        raise RuntimeError(
            "this search has no follower to copy a seat's state from; call "
            "contract.remember_follower(arena, views.observer.follower) first"
        )
    return entry[1]


def placement_value(served, planes):
    """Where the standings lead from these positions, by the placement-only
    head served beside the network. A search asks this at a hand boundary,
    where the hand's own result is already banked, and nowhere else."""
    head = getattr(served, "placement_head", None)
    if head is None:
        raise ValueError(
            "the search was asked to value leaves by placement, but no placement head "
            "was served beside the network: give `contract.serve` one (--placement-head)"
        )
    from . import placement

    features, pooled = placement.features_of(served.net, planes, head.feature_version)
    return head(features, pooled)


def serve(net, checkpoint: str = "the checkpoint", placement_head=None):
    """The server for this network, or an explicit refusal.

    `placement_head` is a judge from `neural.placement`, fitted to this
    network's features, served beside it for a search that values leaves by
    placement; whether it was fitted to this network is the caller's to
    check (`placement.require_head_for`).

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
        return _with_head(EngineServed(net, contract), placement_head)
    if contract.reads == "mortal":
        if contract.planes != 1012 or contract.answers not in (46, ENGINE_ACTIONS):
            raise UnsupportedSearchLayout(
                f"{checkpoint} has unsupported Mortal layout: {contract.planes} planes, "
                f"{contract.answers} actions; expected 1012 planes and 46 or {ENGINE_ACTIONS} actions"
            )
        return _with_head(MortalServed(net, contract), placement_head)
    raise UnsupportedSearchLayout(
        f"{checkpoint} reads {contract.reads!r}, which no server here builds. "
        "Add one rather than approximating it with another's planes."
    )


def _with_head(served, placement_head):
    served.placement_head = placement_head
    return served
