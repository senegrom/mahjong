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
#: How a head reads the tower. 1 and 2 average the per-tile features into one
#: vector; 3 reads the tiles themselves, because averaging was measured to be
#: the limit: at version 2 a head of 64, 256 and 1024 hidden units explained
#: 0.423, 0.424 and 0.424 of the held-out placement, so sixteen times the head
#: bought nothing and the mean was throwing the rest away.
FEATURE_VERSION = 3
READS_TILES = 3
#: A judge with a tower of its own, reading the planes rather than what the
#: policy's tower made of them. Version 3 was measured to gain nothing over
#: version 2 -- 0.423 against 0.424 of the held-out placement, whether it
#: weighed the tiles one way or sixteen -- so what the frozen features hold
#: about the standings, and not how a head reads them, is the ceiling.
OWN_TOWER = 4


class Judge(nn.Module):
    """One number a position: where it leads in the standings, in the
    units the placement bonus is written in.

    Small on purpose. The features come from the policy's own tower,
    which is not trained here, so the head is the only thing fitted and
    a few thousand positions are enough to tell whether anything can be
    read off them at all.
    """

    def __init__(self, channels: int, hidden: int = 64, *, context_channels: int | None = None,
                 feature_version: int = FEATURE_VERSION, looks: int = 4) -> None:
        super().__init__()
        self.channels = channels
        # What comes in beside the tiles: the pooled vector a frozen-feature
        # judge is given, or, for one with a tower of its own, the number of
        # planes it reads.
        self.context_channels = channels if context_channels is None else context_channels
        if (type(feature_version) is not int or feature_version not in (1, 2, READS_TILES, OWN_TOWER)
                or any(type(n) is not int or n <= 0 for n in (channels, hidden, self.context_channels))
                or type(looks) is not int or looks <= 0):
            raise ValueError("invalid placement head feature contract")
        if feature_version == 1 and self.context_channels != channels:
            raise ValueError("legacy placement features have one duplicated pooled vector")
        self.feature_version = feature_version
        self.looks = looks if feature_version == READS_TILES else 0
        if feature_version == OWN_TOWER:
            # Its own eyes. The policy's tower was trained to choose a move,
            # and what it keeps about the standings is what it keeps; a tower
            # fitted to this target sees the planes itself.
            from .model import Residual

            width = channels
            self.stem = nn.Sequential(nn.Conv1d(self.context_channels, channels, 3, padding=1),
                                      nn.GELU())
            self.tower = nn.Sequential(*[Residual(channels) for _ in range(max(looks, 1))])
            self.attention = None
            self.values = None
            width = channels * 2
        elif feature_version == READS_TILES:
            # Where a hand stands is not the average of its tiles: a lone
            # dangerous tile, or the one that completes a wait, is a fact
            # about one position and the mean divides it by thirty-four. So
            # the head is given a few ways to look along the tiles, each
            # weighing the positions itself, and reads what they return
            # beside the mean and the maximum it had before.
            self.attention = nn.Conv1d(channels, looks, 1)
            self.values = nn.Conv1d(channels, channels, 1)
            width = channels * (2 + looks) + self.context_channels
        else:
            self.attention = None
            self.values = None
            width = channels + self.context_channels
        self.body = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.body[2].weight)
        nn.init.zeros_(self.body[2].bias)

    def forward(self, features: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        # New heads receive distinct mean/max summaries and, for combined
        # policies, Mortal's global features. Legacy heads retain their exact
        # old inputs; changing feature meaning underneath saved weights is unsafe.
        if (features.ndim != 3 or features.shape[1] != self.channels
                or pooled.shape != (features.shape[0], self.context_channels)):
            raise ValueError("placement head inputs do not match its feature contract")
        if self.feature_version == OWN_TOWER:
            raise ValueError("a judge with its own tower reads the planes; call judge()")
        if self.feature_version != READS_TILES:
            joined = torch.cat([features.mean(dim=2), pooled], dim=1)
            return self.body(joined).squeeze(1)
        # One weighting of the tiles per look, and what each look draws out.
        weights = torch.softmax(self.attention(features), dim=2)
        drawn = torch.einsum("blp,bcp->blc", weights, self.values(features))
        joined = torch.cat([features.mean(dim=2), features.amax(dim=2),
                            drawn.reshape(drawn.shape[0], -1), pooled], dim=1)
        return self.body(joined).squeeze(1)


    def judge(self, net, planes: torch.Tensor) -> torch.Tensor:
        """Where these positions lead, however this judge reads them: through
        the policy's frozen tower, or through its own."""
        if self.feature_version == OWN_TOWER:
            seen = self.tower(self.stem(planes.float()))
            return self.body(torch.cat([seen.mean(dim=2), seen.amax(dim=2)], dim=1)).squeeze(1)
        features, pooled = features_of(net, planes, self.feature_version)
        return self(features, pooled)


def backbone(net):
    """The network whose features the head reads."""
    return getattr(net, "ours", net)


@torch.no_grad()
def features_of(net, planes: torch.Tensor, feature_version: int = FEATURE_VERSION
                ) -> tuple[torch.Tensor, torch.Tensor]:
    ours = backbone(net)
    features = ours.tail(ours.tower(ours.stem(planes))).float()
    if feature_version == 1:
        return features, features.mean(dim=2)
    if feature_version not in (2, READS_TILES):
        raise ValueError("unsupported placement feature version")
    context = features.amax(dim=2)
    if hasattr(net, "mortal"):
        context = torch.cat([context, net.mortal.features(planes).float()], dim=1)
    return features, context


def new_head(net, hidden: int = 64, feature_version: int = FEATURE_VERSION, looks: int = 4) -> Judge:
    channels = backbone(net).channels
    if feature_version == OWN_TOWER:
        # It reads the planes, so what it has to be told is how many there
        # are; `context_channels` carries that, and a saved head rebuilds
        # from the same field whichever kind it is.
        return Judge(channels, hidden, context_channels=int(backbone(net).planes),
                     feature_version=feature_version, looks=looks)
    context = channels
    if hasattr(net, "mortal"):
        context += int(net.fuse.phi_proj[0].in_features)
    return Judge(channels, hidden, context_channels=context,
                 feature_version=feature_version, looks=looks)



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
        # Set by `load_rounds` where the rounds say so: the first turn to act
        # after a hand boundary, which is where a search asks this judge.
        self.boundary = np.zeros(len(self.placements), dtype=bool)

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
def fingerprint(net, feature_version: int = 1) -> str:
    """Which network's features a head reads, as a digest of the backbone's
    weights. A head fitted to one network's features says nothing about
    another's, and a checkpoint's name does not say which it was."""
    if feature_version not in (1, 2, READS_TILES, OWN_TOWER):
        raise ValueError("unsupported placement feature version")
    if feature_version == OWN_TOWER:
        # It reads the planes, not the network, so what it must agree with is
        # the shape of the observation rather than anyone's weights.
        return f"planes-{int(backbone(net).planes)}"
    digest = hashlib.sha256()
    # Legacy digests are unchanged. New heads bind the entire combined model,
    # not just ours, since Mortal's encoder is part of their representation.
    source = backbone(net) if feature_version == 1 else net
    for name, tensor in sorted(source.state_dict().items()):
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
    have = fingerprint(net, meta.get("feature_version", 1))
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
    if "boundary" in payload:
        boundary = np.asarray(payload["boundary"])
        if boundary.dtype != np.bool_ or boundary.shape != (n,):
            raise ValueError("invalid boundary decision labels")
    scores = np.asarray(payload.get("final_scores"))
    if scores.shape != (payload["games"], 4) or scores.dtype.kind not in "iu":
        raise ValueError("invalid round final scores")
    return payload


def load_rounds(paths: list[Path], *, boundary_only: bool = False) -> Positions:
    """Split by actual environment seed, never input order or mutable paths.

    Overlapping seed ranges are rejected, including duplicate round bytes under
    different names. This deliberately refuses repeated deals under other actors
    too: they need an explicit weighting policy, not accidental double counting.
    """
    if not paths:
        raise ValueError("at least one round is required")
    blocks, placements, games, ranges, marks = [], [], [], [], []
    for path in paths:
        payload = validate_round(path)
        lo, hi = payload["seed"], payload["seed"] + payload["games"]
        if any(lo < end and begin < hi for begin, end in ranges):
            raise ValueError("duplicate or overlapping environment game seeds in rounds")
        ranges.append((lo, hi))
        if boundary_only and "boundary" not in payload:
            raise ValueError("boundary-only fitting needs recollected rounds with boundary labels")
        rows = (np.nonzero(payload["boundary"])[0] if boundary_only
                else np.arange(payload["decisions"], dtype=np.int64))
        blocks.append(Planes(*(np.asarray(payload["observations"][name]) for name in Planes.ARRAYS)).rows(rows))
        placements.append(np.asarray(payload["placements"], dtype=np.float32)[rows])
        games.append((np.asarray(payload["games_of"], dtype=np.uint64) + np.uint64(lo))[rows])
        # Which of the kept rows are the positions a search actually asks
        # about. A judge may be fitted on every decision and still be read
        # only where it is used.
        marks.append(np.asarray(payload["boundary"], dtype=bool)[rows] if "boundary" in payload
                     else np.zeros(len(rows), dtype=bool))
    positions = Positions(Planes.cat(blocks), np.concatenate(placements), np.concatenate(games))
    positions.boundary = np.concatenate(marks)
    return positions


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
        said.append(head.judge(net, planes).float().cpu())
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
          device: str = "cpu", head: Judge | None = None, log=None, seed: int = 3,
          read_at_boundary: bool = False) -> tuple[Judge, list[dict]]:
    """Fits the head on the positions and reads it on the held-back games
    after every pass.

    With `read_at_boundary` the reading is taken only where a search asks
    this judge -- the first turn of the hand after the root hand -- while the
    fitting still sees every position it was given. A judge with a tower of
    its own has far more to learn than a head on frozen features, and every
    decision of a game carries the same placement."""
    net.eval()
    head = head or new_head(net)
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
            said = head.judge(net, planes)
            loss = nn.functional.mse_loss(said, wanted_all[picks].to(device))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
            optimiser.step()
            total += float(loss.detach()) * len(picks)
            seen += len(picks)
        watched = held[positions.boundary[held]] if read_at_boundary else held
        record = {"epoch": epoch, "loss": round(total / max(seen, 1), 5),
                  "held_out": measure(head, net, positions, watched, device)}
        history.append(record)
        if log is not None:
            print(json.dumps(record), file=log, flush=True)
    head.eval()
    return head, history


def save(head: Judge, path: Path, meta: dict) -> None:
    from .checkpoints import atomic_artifact
    if {"judge", "channels", "hidden", "placement_version", "feature_version", "context_channels",
            "looks"} & meta.keys():
        raise ValueError("metadata cannot replace the placement head schema")
    atomic_artifact({**meta, "judge": head.state_dict(), "channels": head.channels,
                     "context_channels": head.context_channels, "feature_version": head.feature_version,
                     "looks": head.looks, "hidden": head.body[0].out_features,
                     "placement_version": PLACEMENT_VERSION},
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
    head = Judge(payload["channels"], payload["hidden"],
                 context_channels=payload.get("context_channels", payload["channels"]),
                 feature_version=payload.get("feature_version", 1),
                 looks=payload.get("looks") or 4)
    head.load_state_dict(payload["judge"])
    head.to(device).eval()
    return head, {k: v for k, v in payload.items() if k not in ("judge", "channels", "hidden")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("rounds", type=Path, nargs="+", help="self-play rounds saved by neural.selfplay")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--boundary-only", action="store_true", help="fit only first turn-to-act positions after hand boundaries")
    parser.add_argument("--read-at-boundary", action="store_true",
                        help="fit on every position given but report only where a search asks: "
                        "the first turn of the hand after the root hand")
    parser.add_argument("--looks", type=int, default=4,
                        help="how many ways a tile-reading head weighs the positions, or how "
                        "many blocks a judge with its own tower has")
    parser.add_argument("--feature-version", type=int, default=FEATURE_VERSION,
                        choices=(2, READS_TILES, OWN_TOWER),
                        help="2 averages the policy tower's tiles into one vector, 3 reads those "
                        "tiles, 4 gives the judge a tower of its own and reads the planes")
    parser.add_argument("--hidden", type=int, default=64,
                        help="width of the head's one hidden layer; whether more of it "
                        "reads more of the standings says if the features or the head is the limit")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from . import contract, zoo

    net = contract.unwrap(zoo.load_player(args.checkpoint, args.device))
    positions = load_rounds(args.rounds, boundary_only=args.boundary_only)
    training, held = positions.split()
    print(json.dumps({"positions": len(positions), "games": int(positions.games.max()) + 1,
                      "training": len(training), "held_out": len(held)}),
          file=sys.stderr, flush=True)
    head, history = train(positions, net, epochs=args.epochs, lr=args.lr, batch=args.batch,
                          device=args.device, log=sys.stderr,
                          head=new_head(net, args.hidden, args.feature_version, args.looks),
                          read_at_boundary=args.read_at_boundary)
    save(head, args.out, {"checkpoint": str(args.checkpoint), "rounds": [str(path) for path in args.rounds],
                          "features": fingerprint(net, head.feature_version), "boundary_only": args.boundary_only, "history": history})
    print(json.dumps({"out": str(args.out), "final": history[-1] if history else None}, indent=1))


if __name__ == "__main__":
    main()
