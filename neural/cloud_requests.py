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


def round_arguments(generations: int, until: int) -> list[str]:
    """A trainer's `--rounds` and `--generations`: `generations` more from
    wherever it resumes, or, with `until`, the generation to stop at. A call
    the spend limit stalled and Modal started again from the top then
    finishes the run instead of playing its count again."""
    if type(until) is not int or until < 0:
        raise ValueError("until must be a generation, or nought for a count of rounds")
    return ["--rounds", str(0 if until else generations), "--generations", str(until or 1000000)]


def placement_arguments(reward: str = "hybrid", alpha: float = 0.0, place_model: str = "",
                        refit_every: int = 0, refit_rounds: int = 4, warm_up: int = 0,
                        warm_up_lr: float = 1e-4) -> list[str]:
    """`train_combined`'s placement reward options as a launch asks for them,
    checked before anything is staged: nothing at all for the default
    reward, so a launch that asks for none of them runs the command line it
    always did, and a refusal for any of them asked for without the
    placement reward, which would otherwise train on a reward nobody asked
    for. The place model's path is the caller's to add once it is staged."""
    if reward not in ("hybrid", "placement"):
        raise ValueError("reward must be 'hybrid' or 'placement'")
    for name, value in (("alpha", alpha), ("warm_up_lr", warm_up_lr)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if warm_up_lr <= 0:
        raise ValueError("warm_up_lr must be positive")
    for name, value in (("refit_every", refit_every), ("warm_up", warm_up)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if type(refit_rounds) is not int or refit_rounds < 1:
        raise ValueError("refit_rounds must be a positive integer")
    if not isinstance(place_model, str):
        raise ValueError("place_model must name a file on the volume, or be empty")
    if reward == "hybrid":
        asked = [name for name, value in (("alpha", alpha), ("place_model", place_model),
                                          ("refit_every", refit_every), ("warm_up", warm_up)) if value]
        if asked:
            raise ValueError(f"{', '.join(asked)} only with reward='placement'")
        return []
    arguments = ["--reward", "placement", "--alpha", repr(float(alpha))]
    if refit_every:
        arguments += ["--place-refit-every", str(refit_every), "--place-refit-rounds", str(refit_rounds)]
    if warm_up:
        arguments += ["--value-warm-up", str(warm_up), "--value-warm-up-lr", repr(float(warm_up_lr))]
    return arguments


def validate_cloud_request(generations: int, opponents: Sequence[str] | None,
                           opponent_share: float, seat_share: float = 0.0) -> None:
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
    if (isinstance(seat_share, bool) or not isinstance(seat_share, Real)
            or not math.isfinite(seat_share) or not 0 <= seat_share < 1):
        raise ValueError("seat_share must be finite, at least nought and below one")
    if seat_share > 0 and not opponents:
        raise ValueError("seat_share requires at least one opponent checkpoint")
    if seat_share > 0 and opponent_share > 0:
        raise ValueError("seat others by game (opponent_share) or by player (seat_share), not both")


def stage_opponents(where: Path, opponents: Sequence[str] | None,
                    opponent_share: float, resolve: Callable[[str], Path],
                    seat_share: float = 0.0) -> list[str]:
    """Stage every requested opponent or fail before launching a trainer.

    Ordinal directories cannot collide even when names share basenames or
    contain punctuation. Hash the validated copied bytes, not the mutable
    source. Duplicates are preserved deliberately (they affect sampling weights).
    The caller owns a fresh invocation workspace; no existing files are replaced.

    The manifest records how the population was seated, by game
    (`opponent_share`) or by player (`seat_share`), and the arguments
    returned carry the same numbers, so the two cannot disagree. A
    manifest written before `seat_share` was recorded has no such key:
    absent means not recorded, not nought.

    `club` names the engine's heuristic player rather than a checkpoint: it
    is passed on by name, and the manifest records it as heuristic.
    """
    from .checkpoints import copy_checkpoint
    from .population import CLUB

    validate_cloud_request(1, opponents, opponent_share, seat_share)
    where = Path(where)
    names = list(opponents or [])
    # Resolve the entire population first: a partially missing population must
    # not be silently reduced to whichever paths happen to exist.
    sources = [None if name == CLUB else Path(resolve(name)) for name in names]
    missing = [f"{name}: {source}" for name, source in zip(names, sources)
               if source is not None and not source.is_file()]
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
        if source is None:
            entries.append({"index": index, "requested": name, "heuristic": True})
            paths.append(name)
            continue
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
                   "seat_share": float(seat_share),
                   "opponents": entries}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    # Preserve the launchers' established flag order for existing callers.
    arguments = ["--opponents", *paths, "--opponent-share", str(float(opponent_share))]
    if seat_share:
        arguments += ["--seat-share", str(float(seat_share))]
    return arguments
