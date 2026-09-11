"""Writing a checkpoint so that a crash cannot cost you the last one.

`torch.save(payload, "latest.pt")` truncates the file and then writes into
it. Anything that goes wrong in between — the process killed, the machine
stopped, the disk full — leaves a file that exists, has the right name, and
cannot be loaded. The previous checkpoint is already gone, because it was
that file. A run can lose hours that way and only find out at the next
resume, which is the worst moment to find out.

So a checkpoint is written beside its destination, flushed to the disk
rather than merely to the kernel, and only then moved into place. The move
is atomic on every filesystem this runs on, so a reader sees either the old
checkpoint or the new one and never a half of either. The one it replaces
is kept as `<name>.prev.pt`, which costs a copy of the file and buys the
ability to go back one step when a checkpoint is complete but wrong.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def publish(payload: dict, destination: Path | str, keep_previous: bool = True) -> Path:
    """Writes `payload` to `destination` without ever leaving it partial.

    Returns the destination. `keep_previous` moves whatever was there to
    `<name>.prev.pt` first, so a complete but unwanted checkpoint can be
    stepped back from; it costs one rename, not a copy.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # On the destination's own filesystem, or the move at the end would be
    # a copy and lose its atomicity.
    writing = destination.with_name(destination.name + ".writing")

    with open(writing, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        # The bytes are the point. Without this they may sit in the
        # kernel's cache while the rename lands, which is exactly the
        # ordering that produces a named file with nothing in it.
        os.fsync(handle.fileno())

    if keep_previous and destination.exists():
        previous = destination.with_name(destination.stem + ".prev.pt")
        try:
            os.replace(destination, previous)
        except OSError:
            # Keeping a spare is a convenience; failing to keep one is not
            # a reason to refuse to save.
            pass

    os.replace(writing, destination)
    _sync_directory(destination.parent)
    return destination


def _sync_directory(where: Path) -> None:
    """Makes the rename itself durable, where the platform allows it.

    Windows has no directory handle to sync and raises; the rename is
    already ordered there, so there is nothing to do and nothing to report.
    """
    try:
        handle = os.open(where, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def loadable(path: Path | str) -> bool:
    """Whether a checkpoint can actually be read back.

    Cheap enough to run after writing one, and the only way to tell a
    complete file from a plausible-looking one.
    """
    try:
        torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return True
