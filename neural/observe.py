"""Mortal's view of the table, for the network.

The network's observation is Mortal's: a thousand and twelve planes over the
34 tile kinds, built by Mortal's own engine from the mjai events our rules
engine writes as it plays. Ours describes the position in ninety-seven, and
the difference is not detail but kinds of information: each player's
discards in order with what was drawn and thrown from the hand, the last
tiles thrown by hand and the riichi tiles, the tiles every hand is waiting
on, furiten, shanten, and an efficiency lookahead that says for each discard
which draws advance the hand and how likely, by turn, it is to be ready, to
win, and for how much. A network can learn some of that from the raw
position and none of it cheaply. Our engine stays the authority on the rules
and on which moves are legal; Mortal's only reads the game and describes it.

Two costs come with the richer view, and this module pays both. The
lookahead takes milliseconds a decision, so the encoder runs across games
in parallel, in Rust and without the GIL, in `libriichi.follow.Follower`.
And a decision is 34,408 floats where it was 3,298, so the planes are kept
sparse: about one value in eighteen is not zero, and a round of them fits
where it used to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import libriichi
import riichi_py
from libriichi.follow import Follower

VERSION = 4
PLANES, POSITIONS = libriichi.consts.obs_shape(VERSION)
WIDTH = PLANES * POSITIONS


class Planes:
    """A batch of observations kept sparse.

    Row `r` has its non-zero entries at `indices[indptr[r]:indptr[r+1]]`,
    flat over plane times 34 plus position, with `values` alongside. A
    minibatch is made dense on the card, where the zeros cost nothing to
    write and the whole batch is a few hundred megabytes rather than the
    round being gigabytes on the host.
    """

    __slots__ = ("indptr", "indices", "values")
    ARRAYS = ("indptr", "indices", "values")

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, values: np.ndarray) -> None:
        self.indptr = indptr
        self.indices = indices
        self.values = values

    @classmethod
    def from_follower(cls, indptr, indices, values) -> Planes:
        """From what the follower's `encode` returned: the offsets widened
        so a round's worth of entries can be counted, the values halved."""
        return cls(
            np.asarray(indptr, dtype=np.int64),
            np.asarray(indices, dtype=np.uint16),
            np.asarray(values, dtype=np.float16),
        )

    @classmethod
    def empty(cls) -> Planes:
        return cls(
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.uint16),
            np.zeros(0, dtype=np.float16),
        )

    def __len__(self) -> int:
        return len(self.indptr) - 1

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    def nbytes(self) -> int:
        return self.indptr.nbytes + self.indices.nbytes + self.values.nbytes

    def rows(self, rows: np.ndarray) -> Planes:
        """The rows asked for, in that order. Works on memory maps: the
        entries of each row are contiguous, so a sorted `rows` reads the
        file forwards."""
        rows = np.asarray(rows, dtype=np.int64)
        starts = self.indptr[rows]
        counts = self.indptr[rows + 1] - starts
        indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        total = int(indptr[-1])
        # Every entry's place in the source: its row's start plus its
        # offset within the row.
        within = np.arange(total, dtype=np.int64) - np.repeat(indptr[:-1], counts)
        flat = np.repeat(starts, counts) + within
        return Planes(indptr, np.asarray(self.indices[flat]), np.asarray(self.values[flat]))

    def slice(self, start: int, stop: int) -> Planes:
        """Rows `start` to `stop`, a contiguous view."""
        stop = min(stop, len(self))
        lo, hi = int(self.indptr[start]), int(self.indptr[stop])
        return Planes(self.indptr[start : stop + 1] - lo, self.indices[lo:hi], self.values[lo:hi])

    def dense(self, device: str | torch.device) -> torch.Tensor:
        """The rows as float32 planes on `device`, shape (rows, PLANES, 34)."""
        n = len(self)
        out = torch.zeros(n * WIDTH, dtype=torch.float32, device=device)
        if self.nnz:
            counts = torch.from_numpy(np.diff(self.indptr)).to(device)
            row = torch.repeat_interleave(
                torch.arange(n, device=device, dtype=torch.int64), counts
            )
            columns = torch.from_numpy(np.asarray(self.indices, dtype=np.int64)).to(device)
            values = torch.from_numpy(np.asarray(self.values, dtype=np.float32)).to(device)
            out[row * WIDTH + columns] = values
        return out.reshape(n, PLANES, POSITIONS)

    @staticmethod
    def cat(blocks: list[Planes]) -> Planes:
        """Stacks blocks into one, freeing each as it is copied, as
        `selfplay.gather` does for dense fields."""
        if not blocks:
            return Planes.empty()
        rows = sum(len(block) for block in blocks)
        total = sum(block.nnz for block in blocks)
        indptr = np.empty(rows + 1, dtype=np.int64)
        indices = np.empty(total, dtype=np.uint16)
        values = np.empty(total, dtype=np.float16)
        indptr[0] = 0
        at_row = at = 0
        for index in range(len(blocks)):
            block = blocks[index]
            n, nnz = len(block), block.nnz
            indptr[at_row + 1 : at_row + n + 1] = block.indptr[1:] + at
            indices[at : at + nnz] = block.indices
            values[at : at + nnz] = block.values
            at_row += n
            at += nnz
            blocks[index] = None
        blocks.clear()
        return Planes(indptr, indices, values)

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self.ARRAYS}

    def save(self, root: Path, stem: str) -> None:
        for name, array in self.arrays().items():
            np.save(Path(root) / f"{stem}-{name}.npy", np.ascontiguousarray(array))

    @classmethod
    def load(cls, root: Path, stem: str, mmap: bool = True) -> Planes:
        mode = "r" if mmap else None
        return cls(*(np.load(Path(root) / f"{stem}-{name}.npy", mmap_mode=mode) for name in cls.ARRAYS))

    @classmethod
    def exists(cls, root: Path, stem: str) -> bool:
        return all((Path(root) / f"{stem}-{name}.npy").exists() for name in cls.ARRAYS)


class Observer:
    """Follows an arena's games through Mortal's engine and encodes what the
    deciding players see.

    Call `advance` once a step, after the arena has moved and before
    anyone is asked for a decision, so the follower has read everything up
    to the decision; then `encode` for whichever games' deciding seats
    want the view.
    """

    def __init__(self, arena, games: int) -> None:
        self.arena = arena
        self.follower = Follower(games, VERSION)

    def advance(self) -> None:
        """Reads whatever happened since last time."""
        self.follower.feed(self.arena.mjai_all())

    def encode(self, games: np.ndarray, players: np.ndarray) -> Planes:
        """The view of `players[i]` in `games[i]`, one row each, sparse."""
        who = list(zip(np.asarray(games).tolist(), np.asarray(players).tolist()))
        indptr, indices, values, _mask = self.follower.encode(who)
        return Planes.from_follower(indptr, indices, values)

    def finish(self) -> None:
        """Reads to the end, so a game that ended is read to its end_game
        and nothing is left in the arena."""
        self.advance()


class Views:
    """What each kind of network sees, from one arena.

    A network of the current lineage sees Mortal's planes; an older one
    sees the engine's. A table may seat both, as a duel between lineages
    does, so this serves either by the network's `kind`. Call `advance`
    once a step, after the arena has been asked who owes a decision and
    before anyone is asked for one.
    """

    ENGINE_PLANES = riichi_py.PLANES

    def __init__(self, arena, games: int, kinds: set[str] | None = None) -> None:
        self.arena = arena
        self.games = games
        kinds = set(kinds or {"mortal"})
        self.observer = Observer(arena, games) if "mortal" in kinds else None
        self._engine: np.ndarray | None = None

    def advance(self) -> None:
        if self.observer is not None:
            self.observer.advance()
        self._engine = None

    def engine(self) -> np.ndarray:
        """The engine's own planes for every game this step, dense."""
        if self._engine is None:
            flat = np.frombuffer(self.arena.observations(), dtype=np.float32)
            self._engine = flat.reshape(self.games, self.ENGINE_PLANES, POSITIONS)
        return self._engine

    def sparse(self, rows: np.ndarray, players: np.ndarray) -> Planes:
        """Mortal's view of `players[i]` in game `rows[i]`, sparse."""
        if self.observer is None:
            raise RuntimeError("this table was not set up to serve Mortal's planes")
        return self.observer.encode(rows, players)

    def dense(
        self, kind: str, rows: np.ndarray, players: np.ndarray, device: str | torch.device
    ) -> torch.Tensor:
        """The planes a network of `kind` wants for those rows, on `device`."""
        if kind == "mortal":
            return self.sparse(rows, players).dense(device)
        if kind == "engine":
            return torch.from_numpy(self.engine()[rows]).to(device)
        raise ValueError(f"no such kind of network: {kind}")
