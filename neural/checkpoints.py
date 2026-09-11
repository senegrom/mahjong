"""Publish complete checkpoints, never truncate the previous recovery point.

Single-writer files. Atomic replacement protects against process interruption;
POSIX directory fsync additionally requests rename durability. Windows has no
portable directory fsync. A failed post-rename sync may leave the complete new
file visible, but cleanup must never unlink that destination.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator



def sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def staging_file(destination: Path) -> Iterator[Path]:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial",
                                        dir=destination.parent)
    os.close(descriptor)
    staged = Path(name)
    try:
        yield staged
    finally:
        # After replace this name is absent. Never clean up destination: a
        # catchable signal can arrive just after rename has committed it.
        staged.unlink(missing_ok=True)


def validate_checkpoint(path: Path, *, require_generation: bool = False) -> int | None:
    """Deserialize the *staged bytes* safely, not the concurrently writable source.

    Do not silently turn a damaged checkpoint into generation zero. Loading on
    CPU checks every tensor storage without allocating a second model on GPU.
    Only tensor/primitive checkpoints supported by weights_only are published.
    """
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not saved:
        raise ValueError(f"{path}: checkpoint must be a nonempty mapping")
    generation = saved.get("generation")
    if (generation is not None and (type(generation) is not int or generation < 0)
            or require_generation and generation is None):
        raise ValueError(f"{path}: checkpoint needs a nonnegative integer generation")
    return generation


def atomic_save(payload: dict, destination: Path, *, keep_previous: bool = True) -> None:
    """Validate the new file, retain the old bytes, then atomically publish.

    The optional .previous snapshot is local and single-writer, not a
    validated recovery guarantee for files that were already corrupt.
    Cloud publishing keeps its existing independent validation contract.
    """
    import torch

    destination = Path(destination)
    with staging_file(destination) as staged:
        with staged.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        validate_checkpoint(staged)
        if keep_previous and destination.exists():
            previous = destination.with_name(destination.name + ".previous")
            with staging_file(previous) as backup:
                with destination.open("rb") as original, backup.open("wb") as stream:
                    shutil.copyfileobj(original, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(backup, previous)
                sync_directory(destination.parent)
        os.replace(staged, destination)
        sync_directory(destination.parent)


def copy_checkpoint(source: Path, destination: Path, *, require_generation: bool = False) -> int | None:
    """Copy an immutable source snapshot; reject corrupt bytes before publication."""
    destination = Path(destination)
    with staging_file(destination) as staged:
        shutil.copyfile(source, staged)
        generation = validate_checkpoint(staged, require_generation=require_generation)
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(staged, destination)
        sync_directory(destination.parent)
    return generation


def publish_training_snapshot(source: Path, target: Path, minimum_generation: int) -> int:
    """Validate all model copies before changing any live volume checkpoint.

    A trainer may have advanced since its stdout notification. Metadata and
    archived names always use the generation in the copied latest.pt, not the
    notification's (historically zero-based) iteration number. Each checkpoint
    replacement is atomic; this is not a multi-file filesystem transaction.
    """
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=target) as folder:
        staged = Path(folder)
        generation = copy_checkpoint(source / "latest.pt", staged / "latest.pt",
                                     require_generation=True)
        if generation < minimum_generation:
            raise ValueError("Checkpoint is older than the completed generation notification")
        names = []
        for name in ("best.pt", "reference.pt"):
            if (source / name).exists():
                other = copy_checkpoint(source / name, staged / name)
                # A concurrently completed best from a later round can wait
                # for that round's notification; never pair it with this latest.
                if other is None or other <= generation:
                    names.append(name)
        log = source / "log.jsonl"
        if log.exists():
            shutil.copyfile(log, staged / "log.jsonl")
            names.append("log.jsonl")
        # All checkpoints have been fully deserialized before any live rename.
        # Retain each tenth complete generation from these exact copied bytes.
        if generation % 10 == 0:
            kept = target / "history" / f"gen-{generation:05d}.pt"
            if not kept.exists():
                copy_checkpoint(staged / "latest.pt", kept, require_generation=True)
        for name in (*names, "latest.pt"):
            with (staged / name).open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(staged / name, target / name)
        with staging_file(target / "generation.txt") as marker:
            with marker.open("w", encoding="utf-8") as stream:
                stream.write(str(generation))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(marker, target / "generation.txt")
        sync_directory(target)
    return generation
