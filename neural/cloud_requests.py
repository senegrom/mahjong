"""Validate cloud requests before launch; snapshot the complete opponent population.

Importing this module does not import Torch or Modal. Checkpoints are validated
lazily while staging, on the training host, not on the thin deployment client.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Callable, Sequence


def validate_cloud_request(generations: int, opponents: Sequence[str] | None,
                           opponent_share: float) -> None:
    """A cloud count is literal, unlike the CLI's zero-as-unspecified sentinel."""
    if type(generations) is not int or generations <= 0:
        raise ValueError("generations must be a positive integer of additional rounds")
    if (isinstance(opponent_share, bool) or not isinstance(opponent_share, Real)
            or not math.isfinite(opponent_share) or not 0 <= opponent_share <= 1):
        raise ValueError("opponent_share must be finite and between zero and one")
    if opponents is not None:
        if not isinstance(opponents, (list, tuple)):
            raise ValueError("opponents must be a list of checkpoint names")
        if any(not isinstance(name, str) or not name.strip() for name in opponents):
            raise ValueError("each opponent must be a nonempty checkpoint name")
    if opponent_share > 0 and not opponents:
        raise ValueError("opponent_share requires at least one opponent checkpoint")


def stage_opponents(where: Path, opponents: Sequence[str] | None,
                    opponent_share: float, resolve: Callable[[str], Path]) -> list[str]:
    """Stage every requested opponent or fail before launching a trainer.

    Ordinal directories cannot collide even when names share basenames or
    contain punctuation. Hash the validated copied bytes, not the mutable
    source. Duplicates are preserved deliberately (they affect sampling weights).
    The caller owns a fresh invocation workspace; no existing files are replaced.
    """
    from .checkpoints import copy_checkpoint

    validate_cloud_request(1, opponents, opponent_share)
    where = Path(where)
    names = list(opponents or [])
    # Resolve the entire population first: a partially missing population must
    # not be silently reduced to whichever paths happen to exist.
    sources = [Path(resolve(name)) for name in names]
    missing = [f"{name}: {source}" for name, source in zip(names, sources)
               if not source.is_file()]
    if missing:
        raise FileNotFoundError("Missing opponent checkpoints: " + "; ".join(missing))
    if not names:
        return ["--opponent-share", "0.0"]
    target = where / "opponents"
    manifest = where / "opponents.json"
    if target.exists() or manifest.exists():
        raise FileExistsError("opponent staging requires an unused invocation workspace")
    target.mkdir(parents=True)
    entries = []
    paths = []
    for index, (name, source) in enumerate(zip(names, sources)):
        basename = name.removesuffix(".pt").replace("/", "--") + ".pt"
        destination = target / f"{index:04d}" / basename
        destination.parent.mkdir()
        generation = copy_checkpoint(source, destination)
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({"index": index, "requested": name, "source": str(source),
                        "staged": str(destination.relative_to(where)),
                        "sha256": digest.hexdigest(), "generation": generation})
        paths.append(str(destination))
    # Staging failure aborts the invocation; its private workspace is cleaned by
    # the owning context manager. Only a complete manifest reaches publication.
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump({"version": 1, "opponent_share": float(opponent_share),
                   "opponents": entries}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    # Preserve the launchers' established flag order for existing callers.
    return ["--opponents", *paths, "--opponent-share", str(float(opponent_share))]
