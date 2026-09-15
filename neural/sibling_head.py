"""A head that ranks siblings, taught by the search.

The value head that self-play trains sees one move per position, the one
the policy made, and its error on the moves the policy did not make is a
bias that a search cannot average away. The search that plays each world
to the end of its hand produces the thing self-play never does: one
position with several moves tried in the same worlds, each with what it
came to. `neural.searched --record` writes those down. This trains a
head on them to say, for a position, how much better or worse each of
our moves is than the average of the moves tried there -- a difference,
not a value, so the part of the position the moves share cancels out
and cannot bias the ranking.

The head reads the network's own per-tile and pooled features, like the
policy head does, and only it trains; the backbone stays what self-play
made it. What it learns is measured the way it will be used: how often
the move it ranks first is the one the rollouts ranked first, against how
often the policy's first choice is.

    python -m neural.sibling_head RECORDING checkpoint.pt --out head.pt
"""

from __future__ import annotations

import argparse
import json
import warnings
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

import riichi_py

from .observe import Planes

ACTIONS = riichi_py.ACTIONS
POSITIONS = riichi_py.POSITIONS
#: Our moves that name a tile: a discard and a riichi discard of each kind.
TILE_MOVES = 2 * POSITIONS


class Ranker(nn.Module):
    """A score for each of our moves from the features the policy reads.

    Two scores per tile from the per-tile features, one per remaining
    move from the pooled ones, laid out as the action space is: the same
    shape as the policy head, because it answers the same question with a
    different teacher.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.tiles = nn.Conv1d(channels, 2, 1)
        self.rest = nn.Linear(channels, ACTIONS - TILE_MOVES)
        # Starting at zero: a head that has learned nothing ranks every move
        # even, and the player that reads it keeps the policy's choice.
        for layer in (self.tiles, self.rest):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        tiles = self.tiles(features).reshape(features.shape[0], -1)
        return torch.cat([tiles, self.rest(pooled)], dim=1)


def backbone(net):
    """The network whose features the head reads: the fusion's own network
    beneath the head, or a plain one."""
    return getattr(net, "ours", net)


@torch.no_grad()
def features_of(net, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-tile and pooled features the policy head reads, without
    gradient: the backbone is not what trains here."""
    ours = backbone(net)
    features = ours.tail(ours.tower(ours.stem(planes)))
    return features.float(), features.mean(dim=2).float()


class Recorded:
    """A recording as `neural.searched --record` wrote it."""

    def __init__(self, folder: Path) -> None:
        from .recordings import resolve_recording, validate_snapshot

        folder = resolve_recording(folder)
        validate_snapshot(folder)
        self.folder = folder
        self.roots = Planes.load(folder, "root", mmap=False)
        self.candidates = np.load(folder / "candidates.npy")
        self.values = np.load(folder / "values.npy")
        per_world = folder / "per_world.npy"
        self.per_world = np.load(per_world) if per_world.exists() else None
        self.world_weights = np.load(folder / "world_weights.npy")
        self.policy = np.load(folder / "policy.npy")
        self.search = np.load(folder / "search.npy")
        self.game = np.load(folder / "game.npy")
        self.chair = np.load(folder / "chair.npy")
        meta_path = folder / "meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        rows = len(self.roots)
        # How sure the policy was of its first move at each root; one
        # where an older recording did not keep it.
        sure_path = folder / "sure.npy"
        self.sure = np.load(sure_path) if sure_path.exists() else np.ones(rows, dtype=np.float32)
        # What the engine allowed at each root, in our moves, where the
        # recording kept it; a policy taught from the rows needs it.
        legal_path = folder / "legal.npy"
        self.legal = np.load(legal_path) if legal_path.exists() else None
        if not (len(self.candidates) == len(self.values) == len(self.policy) == rows):
            raise ValueError(f"{folder}: the recording's arrays do not agree on the row count")

    def __len__(self) -> int:
        return len(self.roots)

    @property
    def targets(self) -> np.ndarray:
        """Each candidate's worth less the mean over the candidates tried
        there, NaN where there was no candidate: the difference the head
        learns.

        These are what the worlds came to, and the best of four of them is
        a best by luck: splitting the worlds so that one half chooses and
        the other scores, the best candidate is worth 0.031 of a unit less
        than the policy's own move on an eight-world recording, while the
        moves the search's margin let through are worth 0.034 more. A head
        fitted to these differences is fitted to that noise as well, which
        is what `precision` weighs down and what the margin in
        `neural.ranked` has to survive; `neural.teach` learns the search's
        decision instead.
        """
        valid = ~np.isnan(self.values)
        counts = valid.sum(axis=1, keepdims=True).clip(min=1)
        mean = np.where(valid, self.values, 0.0).sum(axis=1, keepdims=True) / counts
        return np.where(valid, self.values - mean, np.nan).astype(np.float32)

    def precision(self, floor: float = 0.05) -> np.ndarray:
        """How far each target can be trusted: one over the variance of the
        difference across the worlds it was averaged from, plus `floor`
        squared for the spread of true differences. A row whose candidates
        came to the same thing in every world speaks loudly; one whose
        worlds disagree wildly speaks softly. Even where the worlds were
        not kept. NaN where there is no candidate."""
        valid = ~np.isnan(self.values)
        if self.per_world is None:
            return np.where(valid, 1.0 / floor**2, np.nan).astype(np.float32)
        worlds = self.per_world.astype(np.float64)
        with warnings.catch_warnings():
            # Padding is all NaN, which numpy remarks on; it is masked below.
            warnings.simplefilter("ignore", RuntimeWarning)
            # The difference in each world between a candidate and the mean
            # of the candidates tried there, then its variance over worlds.
            centred = worlds - np.nanmean(worlds, axis=1, keepdims=True)
            counted = (~np.isnan(centred)).sum(axis=2)
            spread = np.nanvar(centred, axis=2)
        mass = np.where(np.isfinite(centred), self.world_weights[:, None, :], 0.0)
        total = mass.sum(axis=2, keepdims=True)
        mass = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
        mean = np.nansum(centred * mass, axis=2, keepdims=True)
        concentration = np.square(mass).sum(axis=2)
        spread = np.nansum(np.square(mass) * np.square(centred - mean), axis=2)
        error = np.divide(spread, 1 - concentration, out=np.full_like(spread, np.inf),
                          where=(counted > 1) & (concentration < 1))
        return np.where(valid, 1.0 / (error + floor**2), np.nan).astype(np.float32)

    def split(self, held_out_every: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Rows to train on and rows to hold out, split by deal rather than
        by row: a deal's decisions are not independent, and the chairs of
        one measurement play the same deals, so a held-out deal is held
        out in every chair."""
        held = (self.game % held_out_every) == 0
        return np.nonzero(~held)[0], np.nonzero(held)[0]


def scored(head: Ranker, net, roots: Planes, rows: np.ndarray, device: str, step: int = 512):
    """The head's scores over our moves for the rows named."""
    out = []
    for start in range(0, len(rows), step):
        picks = rows[start : start + step]
        planes = roots.rows(picks).dense(device)
        features, pooled = features_of(net, planes)
        out.append(head(features, pooled))
    return torch.cat(out) if out else torch.zeros(0, ACTIONS, device=device)


def chosen_by_search(recorded: Recorded) -> np.ndarray:
    """Which candidate the search took, as a column of the candidate
    table, and -1 where it took none of them. The move it made is the
    policy's own unless one beat it by two standard errors over the
    worlds, and that filter is what separates a lesson from the winner's
    curse (`neural.worth`)."""
    same = recorded.candidates == recorded.search[:, None]
    found = same.any(axis=1)
    return np.where(found, same.argmax(axis=1), -1)


@torch.no_grad()
def measure(head: Ranker, net, recorded: Recorded, rows: np.ndarray, device: str) -> dict:
    """How the head ranks against the rollouts, on the rows named.

    Three figures a position: the difference the rollouts gave the move
    the head ranks first, the move the policy chose first, and the best
    move there was. The head earns its place when its figure beats the
    policy's; the best is the ceiling.
    """
    if len(rows) == 0:
        return {"rows": 0}
    targets = torch.from_numpy(recorded.targets[rows]).to(device)
    candidates = torch.from_numpy(recorded.candidates[rows]).to(device)
    valid = ~torch.isnan(targets)
    scores = scored(head, net, recorded.roots, rows, device)
    picked = scores.gather(1, candidates.clamp(min=0))
    picked = torch.where(valid, picked, torch.full_like(picked, float("-inf")))
    by_head = picked.argmax(dim=1)
    filled = torch.where(valid, targets, torch.full_like(targets, float("-inf")))
    by_rollouts = filled.argmax(dim=1)
    at = torch.arange(len(rows), device=device)
    head_worth = targets[at, by_head]
    policy_worth = targets[:, 0]
    best_worth = filled.max(dim=1).values
    loss = nn.functional.mse_loss(picked[valid], targets[valid])
    # What a player that takes the head's favourite past a margin would
    # have gained on these rows, on the rollouts' own scale: held out of
    # the fit, an unbiased estimate of the ranked player's worth per asked
    # decision, to read before a duel is paid for. A head from too few
    # rows shows here as overriding often and gaining nothing.
    advantage = picked[at, by_head] - picked[:, 0]
    by_margin = {}
    for margin in (0.0, 0.02, 0.05, 0.1, 0.2):
        overriding = (by_head != 0) & (advantage > margin)
        rate = float(overriding.float().mean())
        gain = float((head_worth - policy_worth)[overriding].mean()) if bool(overriding.any()) else 0.0
        by_margin[f"{margin:g}"] = {
            "overrides": round(rate, 4),
            "gain_when_overriding": round(gain, 5),
            "gain_per_decision": round(rate * gain, 5),
        }
    # Whether the head names the move the search actually made, which is
    # the only judgement here that was filtered for noise. Read apart for
    # the decisions the search changed, where the base rate is zero
    # because the policy's own move is not the one to name.
    took = torch.from_numpy(chosen_by_search(recorded)[rows]).to(device)
    changed = torch.from_numpy(
        (recorded.search != recorded.policy)[rows]
    ).to(device)
    names_it = (by_head == took) & (took >= 0)
    overrode = {
        "rows": int(changed.sum()),
        "head_names_the_move": round(float(names_it[changed].float().mean()), 4) if bool(changed.any()) else None,
        # What naming one of the candidates at random would get there,
        # since the policy's own move is by definition the wrong answer
        # on a decision the search overrode.
        "by_chance": round(float((valid[changed].float().sum(dim=1).reciprocal()).mean()), 4)
        if bool(changed.any()) else None,
    }
    return {
        "rows": int(len(rows)),
        "loss": round(float(loss), 5),
        "head_names_the_search_move": round(float(names_it.float().mean()), 4),
        # Saying "the policy's move" everywhere scores this, which is what
        # the figure above has to beat to mean anything at all.
        "keeping_the_policys_move_would_score": round(float((~changed).float().mean()), 4),
        "where_the_search_overrode": overrode,
        "head_agrees_with_rollouts": round(float((by_head == by_rollouts).float().mean()), 4),
        "policy_agrees_with_rollouts": round(float((by_rollouts == 0).float().mean()), 4),
        "worth_of_heads_pick": round(float(head_worth.mean()), 5),
        "worth_of_policys_pick": round(float(policy_worth.mean()), 5),
        "worth_of_best": round(float(best_worth.mean()), 5),
        "by_margin": by_margin,
    }


def train(
    recorded: Recorded,
    net,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    batch: int = 256,
    device: str = "cpu",
    head: Ranker | None = None,
    log=None,
    weighted: bool = False,
    target: str = "values",
) -> tuple[Ranker, list[dict]]:
    """Trains the head on the recording's training rows and measures it on
    the held-out ones after every epoch. `weighted` counts each target by
    its precision (`Recorded.precision`), normalised to a mean of one over
    the targets, so the loss keeps its scale.

    `target` says what the head is asked to learn. `values` fits the
    differences the worlds came to, which carry the winner's curse along
    with the lesson. `decision` asks instead which move the search made
    under its two-standard-error margin -- a choice among the candidates,
    learned as one -- which is the only part of a recording measured to
    be worth anything (`neural.worth`).
    """
    net.eval()
    head = head or Ranker(backbone(net).channels)
    head.to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    training, held = recorded.split()
    targets_all = torch.from_numpy(recorded.targets)
    candidates_all = torch.from_numpy(recorded.candidates)
    if weighted:
        precision = recorded.precision()
        precision = precision / np.nanmean(precision)
        weights_all = torch.from_numpy(np.nan_to_num(precision, nan=0.0))
    else:
        weights_all = torch.ones_like(targets_all)
    if target not in ("values", "decision"):
        raise ValueError(f"no such target: {target}")
    took_all = torch.from_numpy(chosen_by_search(recorded))
    if target == "decision":
        training = training[took_all[training].numpy() >= 0]
        if len(training) == 0:
            raise ValueError("the search chose none of the candidates it was given")
    history: list[dict] = []
    generator = np.random.default_rng(7)
    for epoch in range(epochs):
        head.train()
        order = generator.permutation(training)
        total, count = 0.0, 0
        for start in range(0, len(order), batch):
            picks = np.sort(order[start : start + batch])
            planes = recorded.roots.rows(picks).dense(device)
            features, pooled = features_of(net, planes)
            targets = targets_all[picks].to(device)
            candidates = candidates_all[picks].to(device)
            weights = weights_all[picks].to(device)
            valid = ~torch.isnan(targets)
            scores = head(features, pooled).gather(1, candidates.clamp(min=0))
            if target == "decision":
                # A choice among the candidates: the head's scores are the
                # logits of which move the search would make.
                logits = torch.where(valid, scores, torch.full_like(scores, float("-inf")))
                loss = nn.functional.cross_entropy(logits, took_all[picks].to(device))
            else:
                squared = (scores[valid] - targets[valid]) ** 2
                loss = (weights[valid] * squared).sum() / weights[valid].sum().clamp(min=1e-6)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * int(valid.sum())
            count += int(valid.sum())
        head.eval()
        record = {
            "epoch": epoch,
            "train_loss": round(total / max(count, 1), 5),
            "held_out": measure(head, net, recorded, held, device),
        }
        history.append(record)
        if log is not None:
            print(json.dumps(record), file=log, flush=True)
    return head, history


def save(head: Ranker, path: Path, meta: dict) -> None:
    from .checkpoints import atomic_artifact
    if {"ranker", "channels"} & meta.keys():
        raise ValueError("metadata cannot replace the sibling head schema")
    atomic_artifact({**meta, "ranker": head.state_dict(), "channels": head.rest.in_features},
                    path, lambda staged: load(staged, "cpu"))


def load(path: Path, device: str = "cpu") -> tuple[Ranker, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if type(payload.get("channels")) is not int or payload["channels"] <= 0:
        raise ValueError("invalid sibling head dimensions")
    if any(not torch.isfinite(t).all() for t in payload["ranker"].values()):
        raise ValueError("nonfinite sibling head weights")
    head = Ranker(payload["channels"])
    head.load_state_dict(payload["ranker"])
    head.to(device).eval()
    return head, {k: v for k, v in payload.items() if k not in ("ranker", "channels")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, nargs="+", help="recording directories, taken together")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument(
        "--target",
        choices=("values", "decision"),
        default="decision",
        help="what the head learns: the differences the worlds came to, "
        "which carry the winner's curse, or which move the search made "
        "under its margin, which is the part of a recording measured to "
        "be worth something",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="count each target by how much its worlds agreed on it, so the "
        "rows the rollouts were sure of teach more than the ones they were "
        "not",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from . import contract, zoo

    net = contract.unwrap(zoo.load_player(args.checkpoint, args.device))
    recorded = Recorded(args.recording[0])
    if len(args.recording) > 1:
        recorded = gathered([Recorded(folder) for folder in args.recording])
    print(f"{len(recorded)} recorded decisions from {len(args.recording)} recording(s)", file=sys.stderr, flush=True)
    head, history = train(
        recorded, net, epochs=args.epochs, lr=args.lr, batch=args.batch, device=args.device, log=sys.stderr,
        weighted=args.weighted, target=args.target,
    )
    # The head is consulted where the search was: below the confidence
    # the recordings were gated at, which the player reads back.
    save(head, args.out, {"checkpoint": str(args.checkpoint), "recordings": [str(p) for p in args.recording],
                          "history": history, "sure": float(recorded.meta.get("sure", 1.0)),
                          "weighted": bool(args.weighted), "target": args.target})
    print(json.dumps({"out": str(args.out), "final": history[-1] if history else None}, indent=1))


def gathered(parts: list[Recorded]) -> Recorded:
    """Several recordings as one, their games kept apart for the split."""
    first = parts[0]
    joined = Recorded.__new__(Recorded)
    joined.folder = first.folder
    joined.roots = Planes.cat([part.roots for part in parts])
    width = max(part.candidates.shape[1] for part in parts)

    def padded(arrays, fill):
        out = []
        for array in arrays:
            pad = np.full((array.shape[0], width - array.shape[1]), fill, dtype=array.dtype)
            out.append(np.concatenate([array, pad], axis=1))
        return np.concatenate(out)

    joined.candidates = padded([part.candidates for part in parts], -1)
    joined.values = padded([part.values for part in parts], np.nan)
    if all(part.per_world is not None for part in parts):
        worlds = max(part.per_world.shape[2] for part in parts)
        blocks = []
        for part in parts:
            block = np.full((part.per_world.shape[0], width, worlds), np.nan, dtype=np.float32)
            block[:, : part.per_world.shape[1], : part.per_world.shape[2]] = part.per_world
            blocks.append(block)
        joined.per_world = np.concatenate(blocks)
        joined.world_weights = np.concatenate([
            np.pad(part.world_weights, ((0, 0), (0, worlds - part.world_weights.shape[1])))
            for part in parts])
    else:
        joined.per_world = None
    joined.policy = np.concatenate([part.policy for part in parts])
    joined.search = np.concatenate([part.search for part in parts])
    # The chairs of one measurement play the same deals, so their games
    # keep their numbers and a held-out deal is held out in every chair;
    # recordings of other deals are numbered on from the ones before.
    offsets = []
    by_deals: dict[tuple, int] = {}
    next_offset = 0
    for part in parts:
        deals = (part.meta.get("seed"), part.meta.get("games"))
        if None not in deals and deals in by_deals:
            offsets.append(by_deals[deals])
            continue
        by_deals[deals] = next_offset
        offsets.append(next_offset)
        next_offset += int(part.game.max()) + 1 if len(part.game) else 0
    joined.game = np.concatenate([part.game + offset for part, offset in zip(parts, offsets)])
    joined.chair = np.concatenate([part.chair for part in parts])
    joined.sure = np.concatenate([part.sure for part in parts])
    joined.legal = (np.concatenate([part.legal for part in parts])
                    if all(part.legal is not None for part in parts) else None)
    joined.meta = {
        "parts": [part.meta for part in parts],
        "sure": max(float(part.meta.get("sure", 1.0)) for part in parts),
    }
    return joined


if __name__ == "__main__":
    main()
