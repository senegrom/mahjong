"""Immutable policy snapshots and role-aware league provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

from .checkpoints import copy_checkpoint
from .population import Member, Population


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(source: Path, folder: Path) -> dict:
    """Load/evaluate the bytes we hashed, never re-open a mutable latest path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pin-", dir=folder) as where:
        staged = Path(where) / "policy.pt"
        generation = copy_checkpoint(Path(source), staged)
        sha = digest(staged)
        target = folder / f"{sha}.pt"
        if target.exists():
            if digest(target) != sha:
                raise ValueError(f"corrupted immutable policy snapshot: {target}")
        else:
            copy_checkpoint(staged, target)
    return {"requested": str(source), "sha256": sha, "generation": generation,
            "file": target.name}


def pin_population(requested: list[tuple[Path, str]], folder: Path, *, saved=None,
                   refresh=False) -> tuple[Population, list[Path], list[dict]]:
    """A mutable reference changes only on explicit --refresh-opponents.

    On resume with no new roster arguments, the saved roster is reused. Moving
    a run requires copying its snapshots directory as well as latest.pt.
    """
    folder = Path(folder)
    if saved is not None and not refresh:
        if requested and [(str(p), r) for p, r in requested] != [(x["requested"], x["role"]) for x in saved]:
            raise ValueError("roster changed on resume; use --refresh-opponents explicitly")
        entries = [dict(item) for item in saved]
    else:
        entries = []
        for source, role in requested:
            if role not in ("champion", "reference", "recent", "older"):
                raise ValueError("invalid opponent role")
            entries.append({**snapshot(source, folder), "role": role,
                            "weight": {"champion": 3.0, "reference": 2.0,
                                       "recent": 1.0, "older": 0.5}[role]})
    members, paths = [], []
    for item in entries:
        # Path traversal must not be possible through copied checkpoint metadata.
        sha = item["sha256"]
        if (not isinstance(sha, str) or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)
                or item["file"] != f"{sha}.pt"):
            raise ValueError("invalid snapshot provenance")
        path = folder / item["file"]
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f"missing or modified pinned opponent {path}")
        paths.append(path)
        members.append(Member(item["requested"], item["role"], f"sha256:{sha}", item["weight"]))
    return Population(members), paths, entries
