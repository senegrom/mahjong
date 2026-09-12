"""A ring of the last few rounds on disk, for the heads that may learn from
stale play.

The policy is trained on the round it just played, as PPO wants. The value
heads, the reader and the head that reads the table are not so constrained:
what a position is worth, what the hidden hands were and how they read do
not go stale as the policy moves a little. Trained on one round at a time,
those heads memorised it. The public value head's error inside the epochs
fell to two thirds of its error on the next round, which was no better than
guessing the mean: a round is 370,000 decisions but only about two thousand
placements, and a network of thirteen million parameters learns those by
heart in three passes.

The ring keeps the last several rounds on disk as memory maps, a few
gigabytes each, and hands out minibatches drawn evenly from all of them, so
each head sees an order of magnitude more games than it did and no round
often enough to learn it by heart. This is the replay window every
AlphaZero-style trainer has, in the form the host's memory allows: the
rounds live on the SSD and only the minibatch crosses into memory.

The observations are kept sparse, as `observe.Planes`; the other fields are
dense arrays.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import warnings
from pathlib import Path

import numpy as np
import torch

from .observe import Planes
from .replay_schema import validate_dense_replay

FIELDS = ("legal", "held", "oracle", "imagined", "returns")
SPARSE = "observations"


class Ring:
    """Single-writer replay cache with immutable batches and atomic publication.

    Legacy slots remain readable but are never overwritten. New batches are
    durable before ring.json references them; retired data is removed only
    after that manifest is durable. Existing legacy mixed writes cannot be
    detected retrospectively when all array shapes happen to match.
    """

    def __init__(self, root: Path, rounds: int) -> None:
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0:
            raise ValueError("replay rounds must be a positive integer")
        self.root = Path(root)
        self.rounds = rounds
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "ring.json"
        self.entries: list[dict] = []
        self.next_slot = 0
        self.maps: dict[int, dict] = {}
        self._dirty = True
        retained = self._read_index()
        # Make a manifest left by an interrupted publication durable before
        # reclaiming generations which it no longer references.
        self._sync_directory(self.root)
        self._collect(retained)
        self._dirty = False

    def _read_index(self) -> list[dict]:
        """Reload committed state without reclaiming any potentially live data."""
        self.entries = []
        self.next_slot = 0
        self.maps = {}
        if self.index.exists():
            saved = json.loads(self.index.read_text(encoding="utf-8"))
            if saved.get("version", 1) not in (1, 2):
                raise ValueError("Unsupported replay manifest version")
            self.next_slot = saved["next_slot"]
            if type(self.next_slot) is not int or self.next_slot < 0:
                raise ValueError("Invalid replay slot counter")
            seen = set()
            for entry in saved["entries"]:
                slot, n = entry["slot"], entry["n"]
                if type(slot) is not int or slot < 0 or slot in seen or type(n) is not int or n <= 0:
                    raise ValueError("Invalid replay entry")
                seen.add(slot)
                # Reject path traversal before accessing a manifest's directory.
                generation = entry.get("generation")
                if generation is not None and (not isinstance(generation, str)
                        or not re.fullmatch(r"batch-[0-9a-f]{32}", generation)):
                    raise ValueError("Invalid replay generation")
                try:
                    maps = self._load(entry)
                    self._validate(maps, n)
                except (OSError, ValueError, EOFError) as error:
                    warnings.warn(f"Ignoring incomplete replay slot {slot}: {error}", RuntimeWarning)
                    continue
                self.entries.append(entry)
                self.maps[slot] = maps
            return saved["entries"]
        return []

    def _refresh(self) -> None:
        # A catchable signal may arrive after rename but before any following
        # Python assignment. Never reuse speculative in-memory state.
        if self._dirty:
            self._read_index()
            self._dirty = False

    @staticmethod
    def _validate(maps: dict, n: int) -> None:
        if n <= 0 or any(maps[field].ndim == 0 or len(maps[field]) != n for field in FIELDS):
            raise ValueError("Replay fields must have one row per decision")
        validate_dense_replay(maps, n)
        maps[SPARSE].validate(expected_rows=n)

    @staticmethod
    def _sync_directory(path: Path) -> None:
        # Windows has no portable directory-fsync API. File fsync and atomic
        # replace still prevent process-interruption mixing on that platform.
        if os.name != "nt":
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _load(self, entry: dict) -> dict:
        generation = entry.get("generation")
        root = self.root / generation if generation else self.root
        suffix = "" if generation else f"-{entry['slot']}"
        maps = {field: np.load(root / f"{field}{suffix}.npy", mmap_mode="r", allow_pickle=False)
                for field in FIELDS}
        maps[SPARSE] = Planes.load(root, f"{SPARSE}{suffix}")
        return maps

    def _remove(self, path: Path) -> None:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except OSError as error:
            # A Windows reader may still hold a retired mmap. The committed
            # manifest is valid; retry generation cleanup at the next opening.
            warnings.warn(f"Could not remove retired replay data {path.name}: {error}", RuntimeWarning)

    def _collect(self, entries: list[dict]) -> None:
        retained = {entry.get("generation") for entry in entries}
        for path in self.root.iterdir():
            if (re.fullmatch(r"(?:batch-|\.staging-)[0-9a-f]{32}", path.name)
                    and path.name not in retained):
                self._remove(path)
            elif re.fullmatch(r"\.ring-[0-9a-f]{32}\.json", path.name):
                self._remove(path)

    def __len__(self) -> int:
        self._refresh()
        return len(self.entries)

    def total(self) -> int:
        """How many decisions the ring holds."""
        self._refresh()
        return sum(entry["n"] for entry in self.entries)

    def push(self, batch) -> None:
        """Publish a complete generation without modifying any live batch."""
        self._refresh()
        n = int(batch.decisions)
        data = {field: getattr(batch, field).numpy() for field in FIELDS}
        data[SPARSE] = batch.observations
        self._validate(data, n)
        slot = self.next_slot % self.rounds
        token = uuid.uuid4().hex
        generation = f"batch-{token}"
        staging = self.root / f".staging-{token}"
        destination = self.root / generation
        manifest = self.root / f".ring-{token}.json"
        entry = {"slot": slot, "n": n, "generation": generation}
        entries = ([old for old in self.entries if old["slot"] != slot] + [entry])[-self.rounds:]
        previous = self.entries
        self._dirty = True
        try:
            staging.mkdir()
            for field in FIELDS:
                np.save(staging / f"{field}.npy", data[field], allow_pickle=False)
            batch.observations.save(staging, SPARSE)
            for path in staging.iterdir():
                with path.open("r+b") as stream:
                    os.fsync(stream.fileno())
            self._sync_directory(staging)
            os.replace(staging, destination)
            self._sync_directory(self.root)
            with manifest.open("w", encoding="utf-8") as stream:
                json.dump({"version": 2, "entries": entries, "next_slot": self.next_slot + 1}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(manifest, self.index)
            # Publication has occurred. Even if the directory sync fails, this
            # process must agree with the manifest now visible to new readers.
            self.entries = entries
            self.next_slot += 1
            self.maps = {old["slot"]: self.maps[old["slot"]] for old in entries
                         if old is not entry and old["slot"] in self.maps}
            self._sync_directory(self.root)
            self._dirty = False
        finally:
            self._remove(manifest)
            self._remove(staging)
            # Never delete destination here: rename may have committed it just
            # before KeyboardInterrupt. Recovery/next successful publication
            # collects only generations absent from the committed manifest.
        # Only retire data after successful, durable publication.
        for old in previous:
            if old not in entries and not old.get("generation"):
                for field in FIELDS:
                    self._remove(self.root / f"{field}-{old['slot']}.npy")
                for part in Planes.ARRAYS:
                    self._remove(self.root / f"{SPARSE}-{old['slot']}-{part}.npy")
        self._collect(entries)

    def _maps(self, slot: int) -> dict:
        if slot not in self.maps:
            entry = next(entry for entry in self.entries if entry["slot"] == slot)
            maps = self._load(entry)
            self._validate(maps, entry["n"])
            self.maps[slot] = maps
        return self.maps[slot]

    def sample(self, count: int, rng: np.random.Generator) -> dict:
        """Draw one round proportionally to its size, then rows uniformly."""
        self._refresh()
        if count <= 0 or not self.entries:
            raise ValueError("Sampling needs a positive count and a nonempty replay ring")
        weights = np.array([entry["n"] for entry in self.entries], dtype=np.float64)
        entry = self.entries[int(rng.choice(len(self.entries), p=weights / weights.sum()))]
        rows = np.sort(rng.choice(entry["n"], size=min(count, entry["n"]), replace=False))
        maps = self._maps(entry["slot"])
        out = {field: torch.from_numpy(np.ascontiguousarray(maps[field][rows])) for field in FIELDS}
        out[SPARSE] = maps[SPARSE].rows(rows)
        return out
