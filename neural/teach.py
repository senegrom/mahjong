"""The search teaching the policy itself.

A sibling head re-ranks what the policy proposes and can only ever pick
among its first few moves (`neural.ranked`). The end of this is a policy
that has absorbed what the rollouts know and plays it everywhere for
nothing, which is what teaches a searcher's strength into a network in
every game where that has worked.

What the recordings hold is exactly what a policy can be taught from: a
position, several of its moves, and what each came to when the hand was
played out in the same worlds. So the lesson is a distribution over those
moves -- softer or sharper by `--temperature`, in the rollouts' own units
of hand points over four thousand -- and the loss is the cross-entropy of
the policy restricted to the moves that were compared. Moves nobody tried
are not claimed to be bad; they are held where they were by the leash,
the same KL to the starting policy that keeps self-play from eroding this
lineage (`neural.train_combined`).

Which move the rollouts taught is not the one with the best average.
Four candidates valued over a handful of worlds have a best by luck: on
an eight-world recording, choosing on one half of the worlds and scoring
on the other, the best candidate loses 0.031 of a unit a decision, while
the moves the search's two-standard-error margin let through gain 0.034.
So `--target decision`, the default, teaches the move the search made
under that margin; `--target values` teaches the raw ordering, for
comparison rather than for use.

    python -m neural.teach recordings/ checkpoint.pt --out taught.pt

The rein is the learning rate as much as the leash's weight: gradients
are clipped to a norm of one, so a step moves the weights by about the
learning rate however heavy the leash, and a heavy leash on a large rate
makes the policy circle its starting point instead of settling on it.
Teach slowly. The drift the run reports (`leash_kl`) is the figure to
watch, and it is reported whatever the leash weighs.

Then `neural.duel taught.pt checkpoint.pt` says whether the lesson was
worth learning, which is the only figure that settles it.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

import riichi_py

from . import zoo
from .checkpoints import atomic_save
from .sibling_head import Recorded, gathered

ACTIONS = riichi_py.ACTIONS
MORTAL_ACTIONS = zoo.MORTAL_ACTIONS


def mortal_of_ours() -> np.ndarray:
    """Each of our moves as the Mortal move that means it, -1 for one that
    no Mortal move means. A discard is its own tile, a riichi discard is
    the reach -- whose tile is a second question -- and the calls are
    Mortal's calls. Where several of Mortal's mean ours, the plain one is
    taken: this rule set has no red fives."""
    table = np.full(ACTIONS, -1, dtype=np.int64)
    for ours in range(ACTIONS):
        means = np.nonzero(zoo.MEANS[:, ours])[0]
        if len(means):
            table[ours] = int(means.min())
    table[zoo.RIICHI_DISCARD : zoo.TSUMO] = zoo.MORTAL_RIICHI
    return table


OURS_TO_MORTAL = mortal_of_ours()


class Lesson:
    """A recording as the policy is asked it: per row, the moves that were
    compared named in Mortal's action space, what the rollouts gave each,
    and the target distribution over them.

    Two of our moves can be one of Mortal's -- a reach throwing this tile
    or that one is one reach, and which tile is a second question the
    table asks afterwards -- so such candidates are one move here, worth
    what the better of them was worth, which is what the policy would get
    by declaring the reach and then being asked.
    """

    def __init__(self, recorded: Recorded, temperature: float = 0.1, weighted: bool = False,
                 without_mask: bool = False, target: str = "decision") -> None:
        if recorded.legal is None and not without_mask:
            raise ValueError(
                "this recording did not keep the table's mask, and the fusion reads the mask "
                "itself: asked under the moves that were compared alone it answers something "
                "the table never asked. Record again, or pass without_mask to teach anyway"
            )
        self.recorded = recorded
        self.temperature = float(temperature)
        rows, width = recorded.candidates.shape
        named = np.where(recorded.candidates >= 0, OURS_TO_MORTAL[recorded.candidates.clip(min=0)], -1)
        values = recorded.values.astype(np.float64)
        moves = np.full((rows, width), -1, dtype=np.int64)
        worth = np.full((rows, width), np.nan, dtype=np.float64)
        for row in range(rows):
            best: dict[int, tuple[int, float]] = {}
            for at in range(width):
                move = int(named[row, at])
                value = values[row, at]
                if move < 0 or not np.isfinite(value):
                    continue
                if move not in best or value > best[move][1]:
                    best[move] = (at, value)
            for move, (at, value) in best.items():
                moves[row, at] = move
                worth[row, at] = value
        self.moves = moves
        self.worth = worth
        valid = moves >= 0
        self.valid = valid
        if target not in ("decision", "values"):
            raise ValueError(f"no such target: {target}")
        self.target = target
        # The lesson itself. Two ways to say what the rollouts taught:
        #
        # `decision` is the move the search actually made, which is the
        # policy's own unless a candidate beat it by two standard errors
        # over the worlds (`search::pick_by_margin`). That filter is the
        # difference between a lesson and noise: on an eight-world
        # recording, taking the best candidate outright loses 0.031 of a
        # unit a decision, chosen on one half of the worlds and scored on
        # the other, while the moves the margin let through gain 0.034.
        #
        # `values` is the raw ordering, softened by the temperature: every
        # candidate taught in proportion to what it came to. It teaches
        # the winner's curse along with the lesson, and is kept for
        # comparison rather than for use.
        if target == "values":
            filled = np.where(valid, worth, -np.inf)
            odds = np.exp((filled - filled.max(axis=1, keepdims=True)) / max(self.temperature, 1e-6))
            odds = np.where(valid, odds, 0.0)
            total = odds.sum(axis=1, keepdims=True)
            self.targets = np.where(total > 0, odds / np.maximum(total, 1e-12), 0.0).astype(np.float32)
        else:
            wanted = np.where(recorded.search >= 0, OURS_TO_MORTAL[recorded.search.clip(min=0)], -1)
            self.targets = np.zeros((rows, width), dtype=np.float32)
            for row in range(rows):
                where = np.nonzero(valid[row] & (moves[row] == wanted[row]))[0]
                if len(where):
                    self.targets[row, where[0]] = 1.0
        #: Where the search's move was not the policy's: the rows that
        #: teach something the policy did not already do.
        self.changed = recorded.search != recorded.policy
        self.rows = np.nonzero((valid.sum(axis=1) >= 2) & (self.targets.sum(axis=1) > 0))[0]
        if weighted:
            precision = recorded.precision()
            weight = np.nansum(np.where(valid, precision, np.nan), axis=1)
            self.weights = (weight / max(np.nanmean(weight[self.rows]), 1e-6)).astype(np.float32)
        else:
            self.weights = np.ones(rows, dtype=np.float32)
        # What the table allowed, in Mortal's moves: the question the
        # network is asked, not only what the leash holds it over. The
        # fusion's correction reads the mask as an input (`combined.Fuse`),
        # so a narrower one moves the logits of the moves that remain.
        if recorded.legal is not None:
            self.allowed = zoo.translatable(recorded.legal.astype(bool))
            self.whole_mask = True
        else:
            self.allowed = np.zeros((rows, MORTAL_ACTIONS), dtype=bool)
            for row in range(rows):
                self.allowed[row, moves[row][valid[row]]] = True
            self.whole_mask = False
        # Nothing is asked of a row whose mask lost every compared move.
        keep = self.allowed[self.rows].any(axis=1)
        self.rows = self.rows[keep]

    def __len__(self) -> int:
        return len(self.rows)

    def split(self, held_out_every: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Deals to teach on and deals to hold back, as the head's are."""
        training, held = self.recorded.split(held_out_every)
        mine = set(self.rows.tolist())
        return (
            np.array([row for row in training if row in mine], dtype=np.int64),
            np.array([row for row in held if row in mine], dtype=np.int64),
        )


def lesson_batch(lesson: Lesson, picks: np.ndarray, device: str):
    """One minibatch: the planes, the mask, the moves compared, the target
    over them and each row's weight."""
    planes = lesson.recorded.roots.rows(picks).dense(device)
    allowed = torch.from_numpy(lesson.allowed[picks]).to(device)
    moves = torch.from_numpy(lesson.moves[picks]).to(device)
    valid = torch.from_numpy(lesson.valid[picks]).to(device)
    targets = torch.from_numpy(lesson.targets[picks]).to(device)
    weights = torch.from_numpy(lesson.weights[picks]).to(device)
    return planes, allowed, moves, valid, targets, weights


def taught_loss(logits: torch.Tensor, moves, valid, targets, weights) -> torch.Tensor:
    """Cross-entropy of the policy restricted to the moves that were
    compared, against what the rollouts gave them."""
    picked = logits.gather(1, moves.clamp(min=0))
    picked = torch.where(valid, picked, torch.full_like(picked, float("-inf")))
    logs = picked - torch.logsumexp(picked, dim=1, keepdim=True)
    logs = torch.where(valid, logs, torch.zeros_like(logs))
    per_row = -(targets * logs).sum(dim=1)
    return (weights * per_row).sum() / weights.sum().clamp(min=1e-6)


def leash_to(before: torch.Tensor, now: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """KL from the starting policy to this one, counted the way that
    punishes abandoning a move the starting policy liked, as the trainer
    counts it."""
    before = torch.log_softmax(before.float(), dim=1)
    now = torch.log_softmax(now.float(), dim=1)
    weight = torch.where(allowed, before.exp(), torch.zeros_like(before))
    before = torch.where(allowed, before, torch.zeros_like(before))
    now = torch.where(allowed, now, torch.zeros_like(now))
    return (weight * (before - now)).sum(dim=1).mean()


@torch.no_grad()
def measure(net, lesson: Lesson, rows: np.ndarray, device: str, reference=None, step: int = 256) -> dict:
    """What the policy makes of the rows it is shown: how often its first
    move is the one the rollouts liked best, what the rollouts gave the
    move it would play, and how far it has come from where it started.

    The worth is on the search's own scale, each candidate's value less
    the mean of the candidates compared there, so zero is a policy that
    picks among them at random and the best is the ceiling.
    """
    if len(rows) == 0:
        return {"rows": 0}
    was = net.training
    net.eval()
    agreed = worth = drift = 0.0
    best_worth = policy_worth = 0.0
    counted = 0
    for start in range(0, len(rows), step):
        picks = rows[start : start + step]
        planes, allowed, moves, valid, targets, _weights = lesson_batch(lesson, picks, device)
        logits, _value, _hands = net.everything(planes, allowed)
        picked = logits.float().gather(1, moves.clamp(min=0))
        picked = torch.where(valid, picked, torch.full_like(picked, float("-inf")))
        chose = picked.argmax(dim=1)
        # Every candidate's worth less the mean of the candidates compared
        # there, ungrouped: a candidate that lost its group to a better one
        # keeps its own figure, and the winner's is the group's, so the
        # same array answers for the policy's first move and for the best.
        raw = lesson.recorded.values[picks].astype(np.float64)
        centred = torch.from_numpy(
            np.nan_to_num(raw - np.nanmean(raw, axis=1, keepdims=True), nan=-np.inf)
        ).to(device)
        at = torch.arange(len(picks), device=device)
        rollouts_best = centred.argmax(dim=1)
        agreed += float((chose == rollouts_best).float().sum())
        worth += float(centred[at, chose].sum())
        best_worth += float(centred.max(dim=1).values.sum())
        # The policy's own first move before any teaching: candidate zero,
        # which the search took from the order the policy gave it.
        policy_worth += float(centred[:, 0].sum())
        if reference is not None:
            before, _v, _h = reference.everything(planes, allowed)
            drift += float(leash_to(before, logits, allowed)) * len(picks)
        counted += len(picks)
    if was:
        net.train()
    out = {
        "rows": int(counted),
        "agrees_with_rollouts": round(agreed / counted, 4),
        "worth_of_its_pick": round(worth / counted, 5),
        "worth_of_policys_pick": round(policy_worth / counted, 5),
        "worth_of_best": round(best_worth / counted, 5),
    }
    if reference is not None:
        out["leash_kl"] = round(drift / counted, 5)
    return out


def teach(
    net,
    lesson: Lesson,
    *,
    epochs: int = 4,
    lr: float = 1e-4,
    batch: int = 128,
    device: str = "cpu",
    leash: float = 0.1,
    hold: str = "mortal+ours",
    log=None,
    seed: int = 11,
) -> list[dict]:
    """Teaches the policy the lesson, holding still what `hold` names, and
    measures it on the held-back deals after every epoch. The network is
    changed in place; the history is returned."""
    # The policy as it stands, kept whatever the leash weighs: what the
    # lesson is pulling against, and what every measure below says the
    # distance from. A teacher that cannot say how far it moved the policy
    # is not reporting the thing most likely to go wrong.
    reference = copy.deepcopy(net).eval()
    reference.requires_grad_(False)
    if hasattr(net, "set_mode"):
        net.set_mode(hold)
    trainable = [p for p in net.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(f"holding {hold} leaves nothing to teach")
    optimiser = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    training, held = lesson.split()
    if len(training) == 0:
        raise ValueError("the recording has no rows to teach from")
    history: list[dict] = []
    drawer = np.random.default_rng(seed)
    for epoch in range(epochs):
        net.train()
        order = drawer.permutation(training)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch):
            picks = np.sort(order[start : start + batch])
            planes, allowed, moves, valid, targets, weights = lesson_batch(lesson, picks, device)
            logits, _value, _hands = net.everything(planes, allowed)
            loss = taught_loss(logits.float(), moves, valid, targets, weights)
            if leash > 0:
                with torch.no_grad():
                    before, _v, _h = reference.everything(planes, allowed)
                loss = loss + leash * leash_to(before, logits, allowed)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimiser.step()
            total += float(loss.detach()) * len(picks)
            seen += len(picks)
        record = {
            "epoch": epoch,
            "loss": round(total / max(seen, 1), 5),
            "taught_on": int(seen),
            "held_out": measure(net, lesson, held, device, reference=reference),
        }
        history.append(record)
        if log is not None:
            print(json.dumps(record), file=log, flush=True)
    net.eval()
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, nargs="+", help="recording directories, taken together")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument(
        "--target",
        choices=("decision", "values"),
        default="decision",
        help="what the rollouts are taken to have taught: the move the "
        "search made, which is the policy's own unless a candidate beat "
        "it by two standard errors over the worlds, or the raw ordering "
        "of the candidates softened by the temperature. The filter is "
        "what separates the lesson from the winner's curse",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="how sharply the rollouts' values become a target, in their "
        "own units: near zero teaches the best move alone, larger teaches "
        "the ordering",
    )
    parser.add_argument(
        "--leash",
        type=float,
        default=0.1,
        help="how hard the starting policy is held on to, in nats; this "
        "lineage erodes without one (neural.train_combined)",
    )
    parser.add_argument(
        "--hold",
        default="mortal+ours",
        help="what stays fixed while the lesson is taught, parts joined "
        "with a plus: mortal, ours, head. The default teaches the fusion "
        "head alone, which is the smallest thing that changes the policy",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="count each row by how much its worlds agreed on it",
    )
    parser.add_argument(
        "--without-mask",
        action="store_true",
        help="teach from a recording that did not keep the table's mask, "
        "asking the network under the moves that were compared alone. The "
        "fusion reads the mask, so that is a different question from the "
        "one the table asked; for recordings made before the mask was kept",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    net = zoo.load_player(args.checkpoint, args.device)
    if getattr(net, "kind", None) != "mortal":
        raise SystemExit(f"{args.checkpoint} does not read Mortal's planes, which the recordings hold")
    recorded = Recorded(args.recording[0])
    if len(args.recording) > 1:
        recorded = gathered([Recorded(folder) for folder in args.recording])
    lesson = Lesson(recorded, temperature=args.temperature, weighted=args.weighted,
                    without_mask=args.without_mask, target=args.target)
    print(
        json.dumps({
            "rows": len(lesson), "of": len(recorded), "whole_mask": lesson.whole_mask,
            "sure": recorded.meta.get("sure"), "target": args.target,
            "temperature": args.temperature if args.target == "values" else None,
            "taught_a_new_move": int(lesson.changed[lesson.rows].sum()),
        }),
        file=sys.stderr, flush=True,
    )
    before = measure(net, lesson, lesson.split()[1], args.device)
    history = teach(
        net, lesson, epochs=args.epochs, lr=args.lr, batch=args.batch, device=args.device,
        leash=args.leash, hold=args.hold, log=sys.stderr,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kept = {
        **net.state(),
        "learner": "combined",
        "generation": int(payload.get("generation", 0)),
        "training_api_version": payload.get("training_api_version"),
        "taught": {
            "from": str(args.checkpoint),
            "recordings": [str(folder) for folder in args.recording],
            "rows": len(lesson),
            "target": args.target,
            "temperature": args.temperature,
            "leash": args.leash,
            "hold": args.hold,
            "epochs": args.epochs,
            "lr": args.lr,
            "weighted": bool(args.weighted),
            "history": history,
        },
    }
    atomic_save(kept, args.out)
    print(json.dumps({
        "out": str(args.out),
        "before": before,
        "after": history[-1]["held_out"] if history else None,
    }, indent=1))


if __name__ == "__main__":
    main()
