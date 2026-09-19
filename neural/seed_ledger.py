"""Durable, non-overlapping seed reservations for training and evaluation.

Reserve BEFORE work, so a crash consumes the block rather than reusing it.
All related runs should share --seed-ledger. Independent ledgers intentionally
reproduce the same deals and MUST NOT be combined as independent evidence.
A lock left by an interrupted writer is fail-closed: inspect before removing.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

SPAN = 1 << 60
LANE = 1 << 24
DOMAINS = {name: ((i + 1) * SPAN, (i + 2) * SPAN)
           for i, name in enumerate(("train", "validation", "test", "promotion", "counterfactual"))}
VERSION = 1


@contextmanager
def exclusive(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Seed ledger is locked: {lock}; do not run two writers") from error
    try:
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def initial_state(seed: int) -> dict:
    if type(seed) is not int or not 0 <= seed < SPAN // LANE:
        raise ValueError(f"seed must be an integer in [0,{SPAN // LANE})")
    return {"version": VERSION, "seed": seed,
            "next": {name: lo + seed * LANE for name, (lo, _hi) in DOMAINS.items()},
            "counts": {name: 0 for name in DOMAINS}, "reservations": []}


def validate(state: dict) -> None:
    if not isinstance(state, dict) or state.get("version") != VERSION:
        raise ValueError("unsupported seed ledger")
    if not isinstance(state.get("reservations"), list) or not isinstance(state.get("protocols", {}), dict):
        raise ValueError("invalid seed ledger history")
    expected = initial_state(state.get("seed"))
    if set(state.get("next", {})) != set(DOMAINS) or set(state.get("counts", {})) != set(DOMAINS):
        raise ValueError("incomplete seed ledger")
    for name, (_lo, hi) in DOMAINS.items():
        cursor, count = state["next"][name], state["counts"][name]
        if (type(cursor) is not int or not expected["next"][name] <= cursor <= min(hi, expected["next"][name] + LANE)
                or type(count) is not int or count < 0):
            raise ValueError("invalid seed ledger cursor or count")


class SeedLedger:
    def __init__(self, path: Path, *, seed: int = 0, saved: dict | None = None):
        self.path = Path(path)
        with exclusive(self.path):
            state = json.loads(self.path.read_text()) if self.path.exists() else initial_state(seed)
            validate(state)
            if state["seed"] != seed:
                raise ValueError("seed differs from the existing ledger")
            if saved is not None:
                validate(saved)
                if saved["seed"] != seed:
                    raise ValueError("checkpoint belongs to another seed ledger")
                for name, settings in saved.get("protocols", {}).items():
                    protocols = state.setdefault("protocols", {})
                    if name in protocols and protocols[name] != settings:
                        raise ValueError("checkpoint evidence protocol disagrees with ledger")
                    protocols[name] = deepcopy(settings)
                # Retain audit history when reconstructing a lost local ledger.
                known = {(row["domain"], row["seed"], row["end"]) for row in state["reservations"]}
                state["reservations"].extend(deepcopy(row) for row in saved["reservations"]
                    if (row["domain"], row["seed"], row["end"]) not in known)
                # A cloud checkpoint can restore a missing local ledger, while
                # a durable file may have newer reservations after a crash.
                for name in DOMAINS:
                    state["next"][name] = max(state["next"][name], saved["next"][name])
                    state["counts"][name] = max(state["counts"][name], saved["counts"][name])
            atomic_json(self.path, state)

    def reserve(self, domain: str, games: int, label: str = "") -> dict:
        if domain not in DOMAINS or type(games) is not int or games <= 0:
            raise ValueError("a known seed domain and positive game count are required")
        with exclusive(self.path):
            state = json.loads(self.path.read_text())
            validate(state)
            start = state["next"][domain]
            end = start + games
            if end > min(DOMAINS[domain][1], DOMAINS[domain][0] + (state["seed"] + 1) * LANE):
                raise OverflowError("seed domain exhausted")
            attempt = state["counts"][domain] + 1
            reservation = {"domain": domain, "seed": start, "games": games,
                           "end": end, "attempt": attempt, "label": str(label)}
            state["next"][domain] = end
            state["counts"][domain] = attempt
            state["reservations"].append(reservation)
            atomic_json(self.path, state)
        return reservation

    def snapshot(self) -> dict:
        # Atomic replacement permits lock-free readers of the complete state.
        state = json.loads(self.path.read_text())
        validate(state)
        return deepcopy(state)


def bind_protocol(ledger: SeedLedger, name: str, settings: dict) -> None:
    """A promotion series cannot reset its confidence budget or switch bounds."""
    with exclusive(ledger.path):
        state = json.loads(ledger.path.read_text())
        validate(state)
        protocols = state.setdefault("protocols", {})
        if name in protocols and protocols[name] != settings:
            raise ValueError("evaluation protocol changed within an existing evidence series")
        protocols[name] = deepcopy(settings)
        atomic_json(ledger.path, state)
