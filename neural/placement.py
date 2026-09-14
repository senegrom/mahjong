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
        self.games = (np.asarray(games, dtype=np.int64) if games is not None
                      else np.arange(len(placements), dtype=np.int64))

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
        return cls(batch.observations, np.asarray(placements, dtype=np.float32),
                   None if games is None else np.asarray(games, dtype=np.int64))

    def split(self, held_out_every: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Positions to fit and positions to hold back, by game: the
        decisions of one game share its result and are not independent."""
        held = (self.games % held_out_every) == 0
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


def load_rounds(paths: list[Path]) -> Positions:
    """The positions of one or more rounds written by `selfplay.save_round`,
    their games numbered apart so a hold-out by game stays one."""
    from .selfplay import ROUND_VERSION

    blocks, placements, games, offset = [], [], [], 0
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("round_version", 0)) != ROUND_VERSION:
            raise ValueError(
                f"{path} is not a round this reads: it says round version "
                f"{payload.get('round_version')}, and this is version {ROUND_VERSION}"
            )
        if payload.get("games_of") is None:
            raise ValueError(f"{path} does not say which game each decision belongs to")
        blocks.append(Planes(*(np.asarray(payload["observations"][name]) for name in Planes.ARRAYS)))
        placements.append(np.asarray(payload["placements"], dtype=np.float32))
        played = np.asarray(payload["games_of"], dtype=np.int64)
        games.append(played + offset)
        counted = payload.get("games")
        offset += played.max().item() + 1 if counted is None else counted
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
    torch.save({"judge": head.state_dict(), "channels": head.body[0].in_features // 2,
                "hidden": head.body[0].out_features, "placement_version": PLACEMENT_VERSION,
                **meta}, path)


def load(path: Path, device: str = "cpu") -> tuple[Judge, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if int(payload.get("placement_version", 0)) != PLACEMENT_VERSION:
        raise ValueError(
            f"{path} was trained against placement version "
            f"{payload.get('placement_version')}, and this is version {PLACEMENT_VERSION}"
        )
    head = Judge(int(payload["channels"]), int(payload.get("hidden", 64)))
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
