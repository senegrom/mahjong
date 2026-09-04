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
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

FIELDS = ("observations", "legal", "held", "oracle", "imagined", "returns")


class Ring:
    """The last `rounds` rounds, on disk under `root`, drawn from evenly."""

    def __init__(self, root: Path, rounds: int) -> None:
        self.root = Path(root)
        self.rounds = rounds
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "ring.json"
        self.entries: list[dict] = []
        self.next_slot = 0
        if self.index.exists():
            saved = json.loads(self.index.read_text(encoding="utf-8"))
            self.entries = [
                entry
                for entry in saved["entries"]
                if all((self.root / f"{field}-{entry['slot']}.npy").exists() for field in FIELDS)
            ]
            self.next_slot = int(saved["next_slot"])
        self.maps: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def total(self) -> int:
        """How many decisions the ring holds."""
        return sum(entry["n"] for entry in self.entries)

    def push(self, batch) -> None:
        """Writes a round into the oldest slot and forgets what was there."""
        slot = self.next_slot % self.rounds
        self.next_slot += 1
        self.entries = [entry for entry in self.entries if entry["slot"] != slot]
        # An open map would keep Windows from overwriting the file.
        self.maps.pop(slot, None)
        for field in FIELDS:
            np.save(self.root / f"{field}-{slot}.npy", getattr(batch, field).numpy())
        self.entries.append({"slot": slot, "n": int(batch.decisions)})
        self.index.write_text(
            json.dumps({"entries": self.entries, "next_slot": self.next_slot}),
            encoding="utf-8",
        )

    def _maps(self, slot: int) -> dict[str, np.ndarray]:
        if slot not in self.maps:
            self.maps[slot] = {
                field: np.load(self.root / f"{field}-{slot}.npy", mmap_mode="r") for field in FIELDS
            }
        return self.maps[slot]

    def sample(self, count: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        """A minibatch of `count` decisions from one round of the ring,
        the round chosen in proportion to its size, the rows evenly within
        it. One round per minibatch keeps the reads from the map local;
        over many minibatches every round is seen in proportion."""
        weights = np.array([entry["n"] for entry in self.entries], dtype=np.float64)
        entry = self.entries[int(rng.choice(len(self.entries), p=weights / weights.sum()))]
        rows = np.sort(rng.choice(entry["n"], size=min(count, entry["n"]), replace=False))
        maps = self._maps(entry["slot"])
        return {
            field: torch.from_numpy(np.ascontiguousarray(maps[field][rows])) for field in FIELDS
        }
