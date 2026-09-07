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

import os
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
        """The rows as float32 planes on `device`, shape (rows, PLANES, 34).

        The entries cross to the card as they are stored, two bytes each,
        and are widened there: widening them on the host first cost more
        than the copy. Torch has no unsigned 16-bit kind, so the indices
        travel as signed and are put right on the card."""
        n = len(self)
        out = torch.zeros(n * WIDTH, dtype=torch.float32, device=device)
        nnz = self.nnz
        if nnz:
            counts = torch.from_numpy(np.diff(self.indptr)).to(device)
            row = torch.repeat_interleave(
                torch.arange(n, device=device, dtype=torch.int64), counts, output_size=nnz
            )
            signed = np.ascontiguousarray(self.indices).view(np.int16)
            columns = torch.from_numpy(signed).to(device).to(torch.int64) & 0xFFFF
            values = torch.from_numpy(np.ascontiguousarray(self.values)).to(device).float()
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


class DevicePlanes:
    """A round's sparse planes resident on the card, gathered there.

    A round of 800,000 decisions is about 2,500 entries each, twelve
    gigabytes at six bytes an entry, which the card has room for beside
    the network. Gathering a minibatch's rows from the host took longer
    than the step on the card even a few steps ahead; here the gather is a
    handful of kernels and the host does nothing per step.
    """

    def __init__(self, planes: Planes, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.indptr = torch.from_numpy(np.asarray(planes.indptr, dtype=np.int64)).to(self.device)
        signed = torch.from_numpy(np.ascontiguousarray(planes.indices).view(np.int16))
        self.indices = signed.to(self.device).to(torch.int32) & 0xFFFF
        self.values = torch.from_numpy(np.ascontiguousarray(planes.values)).to(self.device)

    def __len__(self) -> int:
        return int(self.indptr.numel()) - 1

    @property
    def nnz(self) -> int:
        return int(self.indices.numel())

    @staticmethod
    def bytes_needed(planes: Planes) -> int:
        """What holding `planes` on the card would take."""
        return planes.nnz * 6 + len(planes) * 8 + 8

    def rows(self, picks: torch.Tensor) -> torch.Tensor:
        """The rows `picks` (a tensor of indices on the card) as dense
        float32 planes, shape (rows, PLANES, 34)."""
        picks = picks.to(self.device, torch.int64)
        n = int(picks.numel())
        starts = self.indptr[picks]
        counts = self.indptr[picks + 1] - starts
        total = int(counts.sum())
        out = torch.zeros(n * WIDTH, dtype=torch.float32, device=self.device)
        if total:
            offsets = torch.cumsum(counts, 0) - counts
            row = torch.repeat_interleave(
                torch.arange(n, device=self.device), counts, output_size=total
            )
            within = torch.arange(total, device=self.device) - torch.repeat_interleave(
                offsets, counts, output_size=total
            )
            flat = torch.repeat_interleave(starts, counts, output_size=total) + within
            columns = self.indices[flat].to(torch.int64)
            out[row * WIDTH + columns] = self.values[flat].float()
        return out.reshape(n, PLANES, POSITIONS)

    def slice(self, start: int, stop: int) -> torch.Tensor:
        stop = min(stop, len(self))
        return self.rows(torch.arange(start, stop, device=self.device))


def resident(
    planes: Planes, device: str | torch.device, spare: int | None = None
) -> DevicePlanes | None:
    """`planes` on the card when it has room for them and `spare` bytes to
    spare afterwards for the step itself, else None and the host keeps
    them. Sixteen gigabytes by default, which the learning step's
    activations need; `RESIDENT_SPARE_GB` overrides it, so a small card
    can be made to take the path for a small round."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    if spare is None:
        spare = int(float(os.environ.get("RESIDENT_SPARE_GB", "16")) * (1 << 30))
    free, _total = torch.cuda.mem_get_info(device)
    if free - DevicePlanes.bytes_needed(planes) < spare:
        return None
    return DevicePlanes(planes, device)


def pad_rows(tensor: torch.Tensor, rows: int, value: float | bool = 0) -> torch.Tensor:
    """`tensor` with rows of `value` appended to make `rows` of them. A
    compiled graph is built for one shape, and the last chunk of a round
    was a new shape every generation, which had the compiler rebuild the
    graph now and then: minutes lost each time."""
    short = rows - tensor.shape[0]
    if short <= 0:
        return tensor
    filler = tensor.new_full((short, *tensor.shape[1:]), value)
    return torch.cat([tensor, filler])


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
        # A step's encoding of every deciding player at once: where each
        # (game, player) sits, the planes, and Mortal's own masks.
        self._step: tuple[dict[tuple[int, int], int], Planes, np.ndarray] | None = None

    def advance(self) -> None:
        if self.observer is not None:
            self.observer.advance()
        self._engine = None
        self._step = None

    def prepare(self, games: np.ndarray, players: np.ndarray) -> None:
        """Encodes the view of every deciding player named, in one call, so
        that the step's later asks for subsets are served from it. One
        call over a thousand rows spreads over the processors far better
        than a few calls over a few dozen each, as when the learner and a
        seated Mortal ask separately."""
        if self.observer is None or len(games) == 0:
            self._step = None
            return
        who = list(zip(np.asarray(games).tolist(), np.asarray(players).tolist()))
        indptr, indices, values, masks = self.observer.follower.encode(who)
        self._step = (
            {pair: index for index, pair in enumerate(who)},
            Planes.from_follower(indptr, indices, values),
            np.asarray(masks, dtype=bool),
        )

    def sparse_and_masks(
        self, rows: np.ndarray, players: np.ndarray, fresh: bool = False
    ) -> tuple[Planes, np.ndarray]:
        """Mortal's view of `players[i]` in game `rows[i]`, sparse, with
        Mortal's own action masks. From the step's prepared encoding when
        it covers them and `fresh` is not asked for; a player told an event
        ahead of the table wants a fresh one."""
        if self.observer is None:
            raise RuntimeError("this table was not set up to serve Mortal's planes")
        who = list(zip(np.asarray(rows).tolist(), np.asarray(players).tolist()))
        if not fresh and self._step is not None:
            where, planes, masks = self._step
            if all(pair in where for pair in who):
                picked = np.array([where[pair] for pair in who], dtype=np.int64)
                return planes.rows(picked), masks[picked]
        indptr, indices, values, masks = self.observer.follower.encode(who)
        return Planes.from_follower(indptr, indices, values), np.asarray(masks, dtype=bool)

    def engine(self) -> np.ndarray:
        """The engine's own planes for every game this step, dense."""
        if self._engine is None:
            flat = np.frombuffer(self.arena.observations(), dtype=np.float32)
            self._engine = flat.reshape(self.games, self.ENGINE_PLANES, POSITIONS)
        return self._engine

    def sparse(self, rows: np.ndarray, players: np.ndarray) -> Planes:
        """Mortal's view of `players[i]` in game `rows[i]`, sparse."""
        return self.sparse_and_masks(rows, players)[0]

    def dense(
        self, kind: str, rows: np.ndarray, players: np.ndarray, device: str | torch.device
    ) -> torch.Tensor:
        """The planes a network of `kind` wants for those rows, on `device`."""
        if kind == "mortal":
            return self.sparse(rows, players).dense(device)
        if kind == "engine":
            return torch.from_numpy(self.engine()[rows]).to(device)
        raise ValueError(f"no such kind of network: {kind}")
