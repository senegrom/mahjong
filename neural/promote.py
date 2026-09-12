"""Whether a candidate has earned the champion's place.

The CLI uses neural.gate's immutable inputs and conservative evidence rule.
The paired-error object API documented below is retained as a diagnostic,
not as a substitute for the CLI's publication evidence.

The trainer used to keep `best.pt` by smoothed placement against three
heuristic players, a hundred and ninety-two games, the network always in
one seat. That number cannot do the job it was given. The heuristic table
compresses real differences — the README's own figure is sevenfold — so two
networks that differ by a tenth of a place at a real table look the same
against the bots; a single seat confuses strength with the dealer's luck;
and smoothing a series and keeping every new low means the lowest reading
wins, which over enough generations is whichever generation was luckiest.

So this asks the question the other way round: sit the candidate at a table
with the champion and see which one wins. The rules are here rather than in
the trainer so that they are written once, and written down before a
candidate is looked at:

- **Both directions.** The candidate plays one seat against three of the
  champion, and then the champion plays one seat against three of the
  candidate. One direction alone is not a ranking: a network can be good at
  being outnumbered and bad at outnumbering.
- **Every seat.** Each direction runs the same deals four times over with
  the measured network in each chair, so the dealer's advantage falls on
  both sides equally.
- **Paired by deal.** The error comes from the deals, not from the four
  seatings, because the seatings share their deals and are nowhere near
  independent.
- **A margin, and a direction that must agree.** The candidate has to be
  ahead by `margin` standard errors in the direction it is measured, and
  must not be behind by that much in the other. A tie is not a promotion:
  the champion keeps its place until something beats it.
- **Its own seeds.** Evaluation deals come from a range training never
  touches, so a candidate cannot be promoted for having learned the
  particular games it is about to be tested on.

Nothing here says how often to test. Testing a series of candidates against
one champion until one passes will eventually promote noise; the caller is
responsible for saying how many attempts a gate is allowed, and the report
records how many there have been.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import duel

#: Evaluation deals are drawn from here upwards. Training seeds live far
#: below it, so the two cannot collide by accident.
EVALUATION_SEED_BASE = 900_000_000

#: How far ahead the candidate has to be, in standard errors of the paired
#: difference, before it takes the champion's place.
DEFAULT_MARGIN = 2.0

#: Deals per seat per direction. Eight hundred deals a direction puts the
#: error on placement at roughly 0.02, which is the size of the differences
#: that have actually separated these networks.
DEFAULT_GAMES = 200


@dataclass
class Direction:
    """One network in one seat against three of the other."""

    #: Which network took the single seat.
    alone: str
    placement: float
    error: float
    #: How far from level, positive when `alone` did better.
    edge: float
    games_per_seat: int
    seed: int


@dataclass
class Verdict:
    """What the gate decided, and everything needed to check it."""

    promoted: bool
    reason: str
    margin: float
    candidate: str
    champion: str
    attempt: int
    directions: list[Direction] = field(default_factory=list)

    def as_json(self) -> str:
        return json.dumps(
            {
                **{k: v for k, v in asdict(self).items() if k != "directions"},
                "directions": [asdict(row) for row in self.directions],
            },
            indent=2,
        )


def _measure(alone, three, games: int, seed: int, device: str, label: str) -> Direction:
    """`alone` in each of the four seats against three of `three`."""
    report = duel.duel(alone, three, games=games, seed=seed, device=device)
    placement = float(report["placement"])
    error = float(report["standard_error"])
    return Direction(
        alone=label,
        placement=placement,
        error=error,
        # Lower placement is better, so being below level is the edge.
        edge=2.5 - placement,
        games_per_seat=games,
        seed=seed,
    )


def gate(
    candidate,
    champion,
    *,
    candidate_name: str = "candidate",
    champion_name: str = "champion",
    games: int = DEFAULT_GAMES,
    margin: float = DEFAULT_MARGIN,
    attempt: int = 1,
    device: str = "cuda",
) -> Verdict:
    """Runs both directions and says whether the candidate is promoted.

    The rule, stated before any number is read: the candidate is promoted
    when it is ahead by `margin` standard errors while outnumbered, and is
    not behind by `margin` standard errors while outnumbering. Anything
    else leaves the champion where it is.
    """
    seed = EVALUATION_SEED_BASE + attempt * 1_000
    outnumbered = _measure(candidate, champion, games, seed, device, candidate_name)
    outnumbering = _measure(champion, candidate, games, seed + 500, device, champion_name)

    ahead = outnumbered.edge > margin * outnumbered.error if outnumbered.error else False
    # The champion alone against three candidates: the candidate is behind
    # here exactly when the champion is ahead.
    behind = outnumbering.edge > margin * outnumbering.error if outnumbering.error else False

    if ahead and not behind:
        reason = (
            f"ahead by {outnumbered.edge:+.4f} at "
            f"{outnumbered.edge / outnumbered.error:.1f} errors while outnumbered, "
            "and not behind in the other direction"
        )
    elif ahead and behind:
        reason = (
            "ahead while outnumbered and behind while outnumbering, which is not a "
            "ranking but a difference in how each plays against its own kind"
        )
    else:
        reason = (
            f"only {outnumbered.edge:+.4f} at "
            f"{(outnumbered.edge / outnumbered.error if outnumbered.error else 0):.1f} "
            f"errors while outnumbered, short of the {margin:g} asked for"
        )

    return Verdict(
        promoted=bool(ahead and not behind),
        reason=reason,
        margin=margin,
        candidate=candidate_name,
        champion=champion_name,
        attempt=attempt,
        directions=[outnumbered, outnumbering],
    )


def promote(payload: dict, where: Path, verdict: Verdict) -> None:
    """Legacy in-memory publication; callers must supply the evaluated payload.

    The CLI uses immutable snapshots and the hash-bound gate below. This helper
    remains for callers already holding evaluated bytes; publication is atomic.
    """
    from .checkpoints import atomic_save

    if not verdict.promoted:
        raise ValueError("A rejected candidate cannot replace the champion")
    atomic_save({**payload, "promotion_verdict": asdict(verdict)}, where / "champion.pt")


def publish_evaluated_snapshot(snapshot: Path, where: Path, report: dict) -> None:
    """Publish exactly the bytes approved by gate.compare, never a mutable path.

    The immutable hash-named verdict is durable before the checkpoint rename.
    Individual files are atomic, not a multi-file transaction. Readers bind the
    verdict to the checkpoint SHA-256, not to the most recently edited JSON file.
    """
    import hashlib
    import os
    from .checkpoints import copy_checkpoint, staging_file, sync_directory

    if not report.get("promote"):
        raise ValueError("A rejected candidate cannot replace the champion")
    expected = report["candidate"]["sha256"]
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("The verdict needs a candidate SHA-256")
    where = Path(where)
    destination = where / "champion.pt"
    with staging_file(destination) as staged:
        copy_checkpoint(snapshot, staged)
        with staged.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Candidate bytes changed after evaluation; champion is unchanged")
        evidence = where / "promotion-reports" / f"{actual}.json"
        with staging_file(evidence) as record:
            with record.open("w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(record, evidence)
            sync_directory(evidence.parent)
        os.replace(staged, destination)
        sync_directory(where)


def main() -> None:
    """Use the shared conservative gate, then atomically publish its exact input.

    The object-level paired-error `gate` remains a diagnostic for compatibility;
    only neural.gate.compare supplies the CLI's publication evidence.
    """
    import argparse
    import tempfile
    import torch
    from . import gate as evidence_gate
    from .checkpoints import copy_checkpoint

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("champion", type=Path)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True, help="fresh held-out seed range")
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--minimum-edge", type=float, default=0.)
    parser.add_argument("--minimum-deals", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="mahjong-promotion-") as folder:
        snapshots = [Path(folder) / name for name in ("candidate.pt", "champion.pt")]
        for source, target in zip((args.candidate, args.champion), snapshots):
            copy_checkpoint(source, target)
        report = evidence_gate.compare(
            *snapshots, games=args.games, seed=args.seed, device=args.device,
            max_steps=args.max_steps, confidence=args.confidence,
            minimum_edge=args.minimum_edge, attempt=args.attempt,
            minimum_deals=args.minimum_deals,
        )
        report["candidate"]["path"] = str(args.candidate)
        report["champion"]["path"] = str(args.champion)
        if report["promote"]:
            publish_evaluated_snapshot(snapshots[0], args.out or args.candidate.parent, report)
            report["checkpoint_written"] = True
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
