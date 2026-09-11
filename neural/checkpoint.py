"""Crash-safe, single-writer publication of training checkpoints.

The live path is never opened for writing. A complete, flushed replacement
is published with os.replace on the same filesystem. The previous bytes are
kept at <name>.previous; recovery is explicit, never an automatic rollback.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import warnings

import torch


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _temporary(path: Path):
    return tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}-", suffix=".tmp",
        dir=path.parent, delete=False,
    )


def atomic_save(payload, destination: str | Path, *, keep_previous: bool = True) -> None:
    """Publish a fully serialized checkpoint without truncating the live one.

    A failed save before publication leaves destination byte-identical. An
    interruption after replace can expose the complete new file; cleanup
    never removes it. Single writer per destination. POSIX directory fsync
    strengthens durability; Windows guarantees here cover process interruption,
    not every filesystem/power-loss failure. Existing files are not validated
    or repaired retrospectively. This function does not unpickle checkpoints.
    """
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending: list[Path] = []
    try:
        with _temporary(path) as stream:
            temporary = Path(stream.name)
            pending.append(temporary)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if keep_previous and path.exists():
            previous = path.with_name(path.name + ".previous")
            with _temporary(previous) as stream:
                backup = Path(stream.name)
                pending.append(backup)
                with path.open("rb") as original:
                    shutil.copyfileobj(original, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(backup, previous)
            _sync_directory(path.parent)
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        # Only our temporary names: a signal can arrive immediately after a
        # successful replace, before any Python publication flag is updated.
        for temporary in pending:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                warnings.warn(f"Could not remove checkpoint temporary {temporary}: {error}",
                              RuntimeWarning, stacklevel=2)
