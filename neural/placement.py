"""A head that judges a position by where it leads in the standings.

The search values a leaf. Every judge tried so far has been confidently
wrong at the table: the critic reads hand points and placement together
and prefers whatever ends a hand well, and playing the world out with the
club heuristic judges a move by how a weak player fares after it, which
gains 0.028 of a unit a decision by its own reckoning and loses a tenth
of a placement where it counts (`neural.worth`, and the measurements in
the plan).

What a search actually needs at a hand boundary is narrower than either.
The hand's own points are already banked there, so the only question left
is where the standings lead, and that is a quantity self-play already
labels: the placement part of every decision's return, kept apart from
the hand points (`selfplay.Batch.placements`). A head trained on that
alone, reading the same frozen features the policy reads, answers the
boundary question and nothing else -- and it costs one pass instead of a
match played out move by move.

    python -m neural.selfplay checkpoint.pt --games 256 --out round.pt
    python -m neural.placement checkpoint.pt round.pt [round.pt ...] --out placement.pt

It is not the hybrid critic relabelled: that head was trained on points
plus placement and cannot be told to forget the points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .observe import Planes

#: What this head is trained to predict, so a checkpoint cannot be
#: mistaken for one trained against a different target.
PLACEMENT_VERSION = 1


class Judge(nn.Module):
    """One number a position: where it leads in the standings, in the
    units the placement bonus is written in.

    Small on purpose. The features come from the policy's own tower,
    which is not trained here, so the head is the only thing fitted and
    a few thousand positions are enough to tell whether anything can be
    read off them at all.
    """

    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.body[2].weight)
        nn.init.zeros_(self.body[2].bias)

    def forward(self, features: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        # Both what the tower says per tile, averaged, and what it says
        # about the position as a whole: the standings live in the second.
        joined = torch.cat([features.mean(dim=2), pooled], dim=1)
        return self.body(joined).squeeze(1)


def backbone(net):
    """The network whose features the head reads."""
    return getattr(net, "ours", net)


@torch.no_grad()
def features_of(net, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ours = backbone(net)
    features = ours.tail(ours.tower(ours.stem(planes)))
    return features.float(), features.mean(dim=2).float()


class Positions:
    """Positions and the placement each led to, as self-play labelled
    them: `neural.selfplay` writes both into a round."""

    def __init__(self, planes: Planes, placements: np.ndarray, games: np.ndarray | None = None) -> None:
        if len(planes) != len(placements):
            raise ValueError("one placement a position")
        self.planes = planes
        self.placements = np.asarray(placements, dtype=np.float32)
        if games is None:
            raise ValueError("placement positions require stable environment game identities")
        identities = np.asarray(games)
        if (identities.shape != (len(placements),) or identities.dtype.kind not in "iu"
                or np.any(identities < 0) or self.placements.shape != (len(planes),)
                or not np.isfinite(self.placements).all()):
            raise ValueError("invalid placement labels or game identities")
        self.games = identities.astype(np.uint64)

    def __len__(self) -> int:
        return len(self.placements)

    @classmethod
    def of(cls, batch) -> Positions:
        placements = getattr(batch, "placements", None)
        if placements is None:
            raise ValueError(
                "this round does not carry the placement part of its returns; it was "
                "collected before they were kept apart, and points cannot be subtracted "
                "out of a return after the fact"
            )
        games = getattr(batch, "game_of", None)
        seed = getattr(batch, "seed", None)
        if games is None or type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("this round does not carry its environment game seeds")
        return cls(batch.observations, np.asarray(placements, dtype=np.float32),
                   np.asarray(games, dtype=np.uint64) + np.uint64(seed))

    def split(self, held_out_every: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Positions to fit and positions to hold back, by game: the
        decisions of one game share its result and are not independent."""
        if type(held_out_every) is not int or not 2 <= held_out_every < 2**64:
            raise ValueError("held_out_every must be an integer from 2 to 2**64-1")
        held = (self.games % np.uint64(held_out_every)) == 0
        return np.nonzero(~held)[0], np.nonzero(held)[0]


@torch.no_grad()
def fingerprint(net) -> str:
    """Which network's features a head reads, as a digest of the backbone's
    weights. A head fitted to one network's features says nothing about
    another's, and a checkpoint's name does not say which it was."""
    digest = hashlib.sha256()
    for name, tensor in sorted(backbone(net).state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def require_head_for(meta: dict, net) -> None:
    """Refuses a head that was fitted to another network's features."""
    fitted = meta.get("features")
    if fitted is None:
        raise ValueError(
            "this head does not say which network's features it was fitted to; train it again"
        )
    have = fingerprint(net)
    if fitted != have:
        raise ValueError(
            f"this head was fitted to a network with features {fitted}, and the network "
            f"served has {have}: a head reads one network's features and no other's"
        )


def validate_round(path: Path) -> dict:
    """Validate a round's arrays and real seed provenance before fitting/publishing.

    Version-1 rounds contain NumPy arrays, so these files must come from a trusted
    local collector (the existing round format is not an untrusted interchange).
    """
    from .selfplay import ROUND_VERSION
    from .ledger import REWARD_VERSION

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("round_version") != ROUND_VERSION:
        raise ValueError(f"{path}: unsupported round version")
    if payload.get("reward_version") != REWARD_VERSION:
        raise ValueError("round reward version is incompatible")
    for name, minimum in (("seed", 0), ("games", 1), ("decisions", 1), ("hands", 1)):
        if type(payload.get(name)) is not int or payload[name] < minimum:
            raise ValueError(f"round needs a valid {name}")
    if payload["seed"] + payload["games"] > 2**64:
        raise ValueError("round game seeds overflow u64")
    n = payload["decisions"]
    observations = payload.get("observations")
    if not isinstance(observations, dict) or set(observations) != set(Planes.ARRAYS):
        raise ValueError("invalid round observations")
    Planes(*(np.asarray(observations[name]) for name in Planes.ARRAYS)).validate(expected_rows=n)
    for name in ("placements", "returns"):
        value = np.asarray(payload.get(name))
        if value.dtype != np.float32 or value.shape != (n,) or not np.isfinite(value).all():
            raise ValueError(f"invalid round {name}")
    from .ledger import PLACEMENT_VALUE
    if (np.any(payload["placements"] < min(PLACEMENT_VALUE))
            or np.any(payload["placements"] > max(PLACEMENT_VALUE))):
        raise ValueError("round placements are outside the placement reward range")
    played = np.asarray(payload.get("games_of"))
    if (played.dtype != np.int64 or played.shape != (n,)
            or np.any((played < 0) | (played >= payload["games"]))):
        raise ValueError("invalid round games_of")
    scores = np.asarray(payload.get("final_scores"))
    if scores.shape != (payload["games"], 4) or scores.dtype.kind not in "iu":
        raise ValueError("invalid round final scores")
    return payload


def load_rounds(paths: list[Path]) -> Positions:
    """Split by actual environment seed, never input order or mutable paths.

    Overlapping seed ranges are rejected, including duplicate round bytes under
    different names. This deliberately refuses repeated deals under other actors
    too: they need an explicit weighting policy, not accidental double counting.
    """
    if not paths:
        raise ValueError("at least one round is required")
    blocks, placements, games, ranges = [], [], [], []
    for path in paths:
        payload = validate_round(path)
        lo, hi = payload["seed"], payload["seed"] + payload["games"]
        if any(lo < end and begin < hi for begin, end in ranges):
            raise ValueError("duplicate or overlapping environment game seeds in rounds")
        ranges.append((lo, hi))
        blocks.append(Planes(*(np.asarray(payload["observations"][name]) for name in Planes.ARRAYS)))
        placements.append(np.asarray(payload["placements"], dtype=np.float32))
        games.append(np.asarray(payload["games_of"], dtype=np.uint64) + np.uint64(lo))
    return Positions(Planes.cat(blocks), np.concatenate(placements), np.concatenate(games))


@torch.no_grad()
def measure(head: Judge, net, positions: Positions, rows: np.ndarray, device: str,
            step: int = 512) -> dict:
    """How well the head reads the standings on the rows named, against
    the two things that need no head at all: saying nothing, and saying
    the average."""
    if len(rows) == 0:
        return {"rows": 0}
    was = head.training
    head.eval()
    said, wanted = [], []
    for start in range(0, len(rows), step):
        picks = rows[start : start + step]
        planes = positions.planes.rows(picks).dense(device)
        features, pooled = features_of(net, planes)
        said.append(head(features, pooled).float().cpu())
        wanted.append(torch.from_numpy(positions.placements[picks]))
    if was:
        head.train()
    said = torch.cat(said)
    wanted = torch.cat(wanted)
    spread = float(wanted.var(unbiased=False))
    error = float(((said - wanted) ** 2).mean())
    return {
        "rows": int(len(rows)),
        "error": round(error, 5),
        # What the same error is for a head that answers with the mean of
        # the rows it was trained on, which is the thing to beat.
        "spread": round(spread, 5),
        "explained": round(1.0 - error / spread, 4) if spread > 0 else None,
        "mean_said": round(float(said.mean()), 4),
        "mean_wanted": round(float(wanted.mean()), 4),
    }


def train(positions: Positions, net, *, epochs: int = 8, lr: float = 1e-3, batch: int = 256,
          device: str = "cpu", head: Judge | None = None, log=None, seed: int = 3
          ) -> tuple[Judge, list[dict]]:
    """Fits the head on the positions and reads it on the held-back games
    after every pass."""
    net.eval()
    head = head or Judge(backbone(net).channels)
    head.to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    training, held = positions.split()
    if len(training) == 0:
        raise ValueError("no positions to fit")
    wanted_all = torch.from_numpy(positions.placements)
    drawer = np.random.default_rng(seed)
    history: list[dict] = []
    for epoch in range(epochs):
        head.train()
        order = drawer.permutation(training)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch):
            picks = np.sort(order[start : start + batch])
            planes = positions.planes.rows(picks).dense(device)
            features, pooled = features_of(net, planes)
            said = head(features, pooled)
            loss = nn.functional.mse_loss(said, wanted_all[picks].to(device))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
            optimiser.step()
            total += float(loss.detach()) * len(picks)
            seen += len(picks)
        record = {"epoch": epoch, "loss": round(total / max(seen, 1), 5),
                  "held_out": measure(head, net, positions, held, device)}
        history.append(record)
        if log is not None:
            print(json.dumps(record), file=log, flush=True)
    head.eval()
    return head, history


def save(head: Judge, path: Path, meta: dict) -> None:
    from .checkpoints import atomic_artifact
    if {"judge", "channels", "hidden", "placement_version"} & meta.keys():
        raise ValueError("metadata cannot replace the placement head schema")
    atomic_artifact({**meta, "judge": head.state_dict(), "channels": head.body[0].in_features // 2,
                     "hidden": head.body[0].out_features, "placement_version": PLACEMENT_VERSION},
                    path, lambda staged: load(staged, "cpu"))


def load(path: Path, device: str = "cpu") -> tuple[Judge, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if int(payload.get("placement_version", 0)) != PLACEMENT_VERSION:
        raise ValueError(
            f"{path} was trained against placement version "
            f"{payload.get('placement_version')}, and this is version {PLACEMENT_VERSION}"
        )
    if any(type(payload.get(key)) is not int or payload[key] <= 0 for key in ("channels", "hidden")):
        raise ValueError("invalid placement head dimensions")
    if any(not torch.isfinite(t).all() for t in payload["judge"].values()):
        raise ValueError("nonfinite placement head weights")
    head = Judge(payload["channels"], payload["hidden"])
    head.load_state_dict(payload["judge"])
    head.to(device).eval()
    return head, {k: v for k, v in payload.items() if k not in ("judge", "channels", "hidden")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("rounds", type=Path, nargs="+", help="self-play rounds saved by neural.selfplay")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from . import contract, zoo

    net = contract.unwrap(zoo.load_player(args.checkpoint, args.device))
    positions = load_rounds(args.rounds)
    training, held = positions.split()
    print(json.dumps({"positions": len(positions), "games": int(positions.games.max()) + 1,
                      "training": len(training), "held_out": len(held)}),
          file=sys.stderr, flush=True)
    head, history = train(positions, net, epochs=args.epochs, lr=args.lr, batch=args.batch,
                          device=args.device, log=sys.stderr)
    save(head, args.out, {"checkpoint": str(args.checkpoint), "rounds": [str(path) for path in args.rounds],
                          "features": fingerprint(net), "history": history})
    print(json.dumps({"out": str(args.out), "final": history[-1] if history else None}, indent=1))


if __name__ == "__main__":
    main()
