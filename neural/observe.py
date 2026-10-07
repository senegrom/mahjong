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

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import functools
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

    `trusted` marks planes this process's encoder wrote this step, every
    value of which Mortal's encoder writes between 0 and 1: making them
    dense still checks that every column is in bounds, but not that every
    value is finite, which was most of the check's cost on each step's
    thousands of rows. The rows and slices of such planes are trusted too;
    anything stacked, saved or loaded is not, and is checked in full,
    except a round whose every row has already been checked in full once,
    by the pass that values it (see `ppo_loop.baseline`).
    """

    __slots__ = ("indptr", "indices", "values", "trusted")
    ARRAYS = ("indptr", "indices", "values")

    def __init__(
        self, indptr: np.ndarray, indices: np.ndarray, values: np.ndarray, trusted: bool = False
    ) -> None:
        self.indptr = indptr
        self.indices = indices
        self.values = values
        self.trusted = trusted

    @classmethod
    def from_follower(cls, indptr, indices, values, trusted: bool = False) -> Planes:
        """From what the follower's `encode` returned: the offsets widened
        so a round's worth of entries can be counted, the values halved.
        Values the follower already halved (`encode(..., half=True)`) are
        taken as they are."""
        return cls(
            np.asarray(indptr, dtype=np.int64),
            np.asarray(indices, dtype=np.uint16),
            np.asarray(values, dtype=np.float16),
            trusted=trusted,
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

    def validate(self, expected_rows: int | None = None, finite: bool = True) -> None:
        """Check untrusted arrays before a column can address another
        observation. `finite=False` leaves out the check that every value
        is finite, and only that."""
        if (any(not isinstance(a, np.ndarray) or a.ndim != 1
                for a in (self.indptr, self.indices, self.values))
                or self.indptr.dtype != np.dtype(np.int64)
                or self.indices.dtype != np.dtype(np.uint16)
                or self.values.dtype.kind != "f" or not self.values.dtype.isnative
                or not len(self.indptr) or self.indptr[0] != 0
                or expected_rows is not None and len(self) != expected_rows
                or np.any(self.indptr[1:] < self.indptr[:-1])
                or self.nnz != len(self.indices) or self.nnz != len(self.values)):
            raise ValueError("Inconsistent sparse replay observations or array dtypes")
        # Bound temporary boolean allocations for multi-gigabyte replay rounds.
        for start in range(0, self.nnz, 1_000_000):
            stop = start + 1_000_000
            if (np.any(self.indices[start:stop] >= WIDTH)
                    or finite and not np.isfinite(self.values[start:stop]).all()):
                raise ValueError("Sparse observation columns must be in bounds and values finite")

    def rows(self, rows: np.ndarray) -> Planes:
        """The rows asked for, in that order, copied. Works on memory maps:
        the entries of each row are contiguous, so each row is one slice
        of the source, and a sorted `rows` reads the file forwards."""
        rows = np.asarray(rows, dtype=np.int64)
        starts = self.indptr[rows]
        ends = self.indptr[rows + 1]
        counts = ends - starts
        if len(counts) and counts.min() < 0:
            raise ValueError("rows must be within the observations")
        indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        total = int(indptr[-1])
        indices = np.empty(total, dtype=self.indices.dtype)
        values = np.empty(total, dtype=self.values.dtype)
        if total:
            # A slice a row, copied in one pass. Naming every entry's
            # place in the source instead, and gathering by it, cost ten
            # to twenty times as much. The slices are cut from plain views
            # of the arrays, which copy nothing: a memory map's own slices
            # take twice as long to make.
            source_indices, source_values = np.asarray(self.indices), np.asarray(self.values)
            spans = list(zip(starts.tolist(), ends.tolist()))
            np.concatenate([source_indices[a:b] for a, b in spans], out=indices)
            np.concatenate([source_values[a:b] for a, b in spans], out=values)
        return Planes(indptr, indices, values, trusted=self.trusted)

    def slice(self, start: int, stop: int) -> Planes:
        """Rows `start` to `stop`, a contiguous view."""
        stop = min(stop, len(self))
        lo, hi = int(self.indptr[start]), int(self.indptr[stop])
        return Planes(self.indptr[start : stop + 1] - lo, self.indices[lo:hi], self.values[lo:hi],
                      trusted=self.trusted)

    def dense(
        self, device: str | torch.device, dtype: torch.dtype = torch.float32, pinned: bool = False
    ) -> torch.Tensor:
        """The rows as planes on `device`, shape (rows, PLANES, 34), in
        float32 or in `dtype`.

        bfloat16 is for a learning step that runs under autocast, whose
        convolutions would only cast float32 planes to it: the values are
        float16, which float32 holds exactly, so there is one rounding to
        bfloat16 either way, the same one, and the convolutions are handed
        the same bits. Planes read without autocast must be float32.

        The entries cross to the card as they are stored, two bytes each,
        and are widened there: widening them on the host first cost more
        than the copy. Torch has no unsigned 16-bit kind, so the indices
        travel as signed and are put right on the card. With `pinned` they
        cross from page-locked memory without waiting for the card, in the
        order of the current stream (see `ppo_loop.minibatches`); without,
        each copy waits until the card has done everything asked of it."""
        return Planes.dense_of([self], device, dtype, pinned)

    @staticmethod
    def dense_of(
        parts: list[Planes],
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
        pinned: bool = False,
    ) -> torch.Tensor:
        """The rows of `parts`, one part after another, as planes on
        `device`, made as `dense` makes them: the planes
        `Planes.cat(parts).dense(...)` would be, without first copying the
        parts into one on the host. A forward asked about rows from two
        encodings at once, as when a reach's tile is asked with the step's
        own question (see `zoo.choose_in_mortal_space`), is handed them
        this way; a step's thousand rows were a few megabytes to copy."""
        for part in parts:
            part.validate(finite=not part.trusted)
        n = sum(len(part) for part in parts)
        out = torch.zeros(n * WIDTH, dtype=dtype, device=device)

        def to_device(array: np.ndarray) -> torch.Tensor:
            host = torch.from_numpy(array)
            if pinned:
                return host.pin_memory().to(device, non_blocking=True)
            return host.to(device)

        first = 0
        for part in parts:
            rows, nnz = len(part), part.nnz
            if nnz:
                counts = to_device(np.diff(part.indptr))
                row = torch.repeat_interleave(
                    torch.arange(first, first + rows, device=device, dtype=torch.int64), counts,
                    output_size=nnz,
                )
                signed = np.ascontiguousarray(part.indices).view(np.int16)
                columns = to_device(signed).to(torch.int64) & 0xFFFF
                values = to_device(np.ascontiguousarray(part.values)).to(dtype)
                out[row * WIDTH + columns] = values
            first += rows
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
        planes = cls(*(np.load(Path(root) / f"{stem}-{name}.npy", mmap_mode=mode,
                               allow_pickle=False) for name in cls.ARRAYS))
        planes.validate()
        return planes


class FlatPlanes:
    """A batch of dense observations, answering what `Planes` answers.

    The engine's own planes are neither sparse nor large: a round of them
    is tens of megabytes where Mortal's would be gigabytes. They are kept
    half-width on the host and widened on the card, so a training loop can
    hold either container without knowing which it has.
    """

    __slots__ = ("array",)

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def __len__(self) -> int:
        return len(self.array)

    def rows(self, rows: np.ndarray) -> FlatPlanes:
        return FlatPlanes(self.array[np.asarray(rows, dtype=np.int64)])

    def slice(self, start: int, stop: int) -> FlatPlanes:
        return FlatPlanes(self.array[start:stop])

    def dense(
        self, device: str | torch.device, dtype: torch.dtype = torch.float32, pinned: bool = False
    ) -> torch.Tensor:
        """As `Planes.dense`; these are small enough to cross as they are."""
        return torch.from_numpy(np.ascontiguousarray(self.array)).to(device).to(dtype)

    @staticmethod
    def cat(blocks: list[FlatPlanes]) -> FlatPlanes:
        if not blocks:
            return FlatPlanes(np.zeros((0, riichi_py.PLANES, POSITIONS), dtype=np.float16))
        return FlatPlanes(np.concatenate([block.array for block in blocks]))

    @classmethod
    def of(cls, dense: np.ndarray) -> FlatPlanes:
        """From the engine's own float32 planes, halved for the host."""
        return cls(np.asarray(dense, dtype=np.float16))


class DevicePlanes:
    """A round's sparse planes resident on the card, gathered there.

    A round of 800,000 decisions is about 2,500 entries each, twelve
    gigabytes at six bytes an entry, which the card has room for beside
    the network. Gathering a minibatch's rows from the host took longer
    than the step on the card even a few steps ahead; here the gather is a
    handful of kernels and the host does nothing per step.
    """

    def __init__(self, planes: Planes, device: str | torch.device) -> None:
        planes.validate()
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

    def rows(self, picks: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """The rows `picks` (a tensor of indices on the card) as dense
        planes, shape (rows, PLANES, 34), in float32 or in `dtype` (see
        `Planes.dense`)."""
        picks = picks.to(self.device, torch.int64)
        n = int(picks.numel())
        starts = self.indptr[picks]
        counts = self.indptr[picks + 1] - starts
        total = int(counts.sum())
        out = torch.zeros(n * WIDTH, dtype=dtype, device=self.device)
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
            out[row * WIDTH + columns] = self.values[flat].to(dtype)
        return out.reshape(n, PLANES, POSITIONS)

    def slice(self, start: int, stop: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        stop = min(stop, len(self))
        return self.rows(torch.arange(start, stop, device=self.device), dtype)


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
    """Follows an arena's games through Mortal's engine, whose follower
    encodes what the deciding players see.

    Call `advance` once a step, after the arena has moved and before
    anyone is asked for a decision, so the follower has read everything up
    to the decision.
    """

    def __init__(self, arena, games: int) -> None:
        self.arena = arena
        self.follower = Follower(games, VERSION)

    def advance(self) -> None:
        """Reads whatever happened since last time."""
        self.follower.feed(self.arena.mjai_all())


@functools.cache
def encoder_thread() -> ThreadPoolExecutor:
    """The thread a step's views are encoded on beside the caller's (see
    `Views.prepare`), made when first wanted. One is enough: it hands the
    follower one batch at a time, and the follower spreads each over every
    processor itself. It serves one round of self-play at a time: a trainer
    that plays its rounds ahead plays every one of them on its own worker,
    one after another (`ppo_loop.PlayAhead`), and nothing else encodes
    aside, a measurement made meanwhile encoding on its own thread from a
    follower of its own."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="encoder")


@dataclass
class Encoded:
    """One of a step's encodings (see `Views.prepare`): the players it
    covers, in its order, and where each sits; whether each is as it would
    stand having declared riichi; and the planes with Mortal's own masks,
    or a worker's promise of them."""

    who: list[tuple[int, int]]
    where: dict[tuple[int, int], int]
    after_reach: bool
    made: tuple[Planes, np.ndarray] | Future

    def result(self) -> tuple[Planes, np.ndarray]:
        """The planes and the masks, waited for while a worker makes them."""
        if isinstance(self.made, Future):
            self.made = self.made.result()
        return self.made


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
        # The step's encodings of deciding players, each one call of the
        # encoder's, made here or by a worker (see `prepare`).
        self._step: list[Encoded] = []

    def advance(self) -> None:
        # Fed only once nothing is being encoded from it (see `prepare`).
        self.wait()
        if self.observer is not None:
            self.observer.advance()
        self._engine = None
        self._step = []

    def prepare(
        self,
        games: np.ndarray,
        players: np.ndarray,
        after_reach: bool = False,
        aside: bool = False,
    ) -> None:
        """Encodes the view of every deciding player named, in one call, so
        that the step's later asks for them are served from it. One call
        over a thousand rows spreads over the processors far better than a
        few calls over a few dozen each. Each call adds an encoding to the
        step's: an ask for exactly the players of one, in its order, is
        handed that encoding's own arrays, and an ask for others it covers
        has them gathered from it, so a caller that encodes each player's
        rows apart gathers nothing. With `after_reach`, each as it would
        stand having declared riichi (see `sparse_and_masks`).

        With `aside`, a worker makes it while this thread goes on: the
        encoder lets go of the interpreter as it works, so self-play's
        learner decides on the card while the seated others' views are made.
        Whatever is served from it waits for it, and so does whatever tells
        the follower anything, feeds it or encodes from it on this thread
        (`tell`, `advance`, `sparse_and_masks` afresh): the follower is
        borrowed while an encoding is made, so a change to it then is
        refused, and the encoder's workers finish the batch they began
        before they take up another, so a second batch would only queue."""
        if self.observer is None or len(games) == 0:
            return
        who = list(zip(np.asarray(games).tolist(), np.asarray(players).tolist()))
        made = (encoder_thread().submit(self._encode, who, after_reach) if aside
                else self._encode(who, after_reach))
        self._step.append(Encoded(who, {pair: index for index, pair in enumerate(who)}, after_reach, made))

    def wait(self) -> None:
        """Returns once none of the step's encodings is still being made,
        and raises whatever went wrong in one."""
        for encoded in self._step:
            encoded.result()

    def tell(self, game: int, player: int, line: str) -> None:
        """Tells `player` in `game` an event ahead of the table (see
        `Follower.tell`), once nothing is being encoded from the follower."""
        self.wait()
        self.observer.follower.tell(game, player, line)

    def _encode(
        self, who: list[tuple[int, int]], after_reach: bool = False
    ) -> tuple[Planes, np.ndarray]:
        """The follower's encoding of those players now, trusted (see
        `Planes`), with the values halved by the encoder's own workers
        rather than here on one thread; with `after_reach`, as each would
        stand having declared riichi."""
        indptr, indices, values, masks = self.observer.follower.encode(
            who, after_reach=after_reach, half=True
        )
        return (Planes.from_follower(indptr, indices, values, trusted=True),
                np.asarray(masks, dtype=bool))

    def sparse_and_masks(
        self, rows: np.ndarray, players: np.ndarray, fresh: bool = False, after_reach: bool = False
    ) -> tuple[Planes, np.ndarray]:
        """Mortal's view of `players[i]` in game `rows[i]`, sparse, with
        Mortal's own action masks. From one of the step's prepared
        encodings when it covers them and `fresh` is not asked for, as that
        encoding's own arrays when they are exactly its players in its
        order (see `prepare`); a player told an event ahead of the table
        wants a fresh one.

        With `after_reach`, each as it would stand having declared riichi:
        the follower previews the declaration on a copy of the player's
        state and changes nothing (see `Follower.encode`), so the tile a
        reach would throw can be asked about before the reach is chosen,
        and nobody need be told it ahead of the table. The table's own copy
        of the reach then leaves the player as telling it would have."""
        if self.observer is None:
            raise RuntimeError("this table was not set up to serve Mortal's planes")
        who = list(zip(np.asarray(rows).tolist(), np.asarray(players).tolist()))
        if not fresh:
            for encoded in self._step:
                if encoded.after_reach != after_reach:
                    continue
                if who == encoded.who:
                    return encoded.result()
                if all(pair in encoded.where for pair in who):
                    planes, masks = encoded.result()
                    picked = np.array([encoded.where[pair] for pair in who], dtype=np.int64)
                    return planes.rows(picked), masks[picked]
        self.wait()
        return self._encode(who, after_reach)

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
