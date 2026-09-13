"""A player that asks the sibling head before it moves.

The policy proposes; among its first few moves, the head trained on the
search's recordings (`neural.sibling_head`) says how much better or worse
each is than the others, and its favourite is taken when it beats the
policy's choice by a margin in the units the rollouts spoke in -- hand
points over four thousand. Otherwise the policy's choice stands, as it
does with a head that has learned nothing, and wherever the policy is
sure of its move: the head learned from decisions the search was asked
about, and is asked about the same.

It costs one head on top of the pass the policy makes anyway, where the
search that taught the head costs a hand played out in every world; if
the head kept what the search knew, this is the search at no cost, and
the duel against the plain policy says whether it did.

    python -m neural.ranked checkpoint.pt head.pt --games 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import riichi_py

from . import contract, sibling_head

ACTIONS = riichi_py.ACTIONS


class RankedPlayer:
    """The network with the head's second opinion. Plays like the players
    the duel seats: `choose(views, rows, players, legal)` in our moves."""

    def __init__(self, net, head: sibling_head.Ranker, k: int = 4, margin: float = 0.05,
                 device: str = "cuda", sure: float = 1.0) -> None:
        self.served = contract.serve(net)
        self.net = self.served.net
        self.head = head.to(device).eval()
        self.k = k
        self.margin = margin
        #: The policy's probability on its first move at or above which the
        #: head is not asked; one asks it everywhere.
        self.sure = sure
        self.device = device
        self.kind = self.served.contract.reads
        self.planes = self.served.contract.planes
        self.actions = self.served.contract.answers
        self.asked = 0
        self.overrode = 0
        self.taken_sure = 0

    def eval(self) -> RankedPlayer:
        self.net.eval()
        self.head.eval()
        return self

    def parameters(self):
        return self.net.parameters()

    @property
    def channels(self) -> int:
        return self.net.channels

    @property
    def blocks(self) -> int:
        return self.net.blocks

    @torch.no_grad()
    def choose(self, views, rows: np.ndarray, players: np.ndarray, legal: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        players = np.asarray(players, dtype=np.int64)
        legal = np.atleast_2d(legal)
        # The order is built over every game's row for the games named,
        # so the mask is widened to the arena's shape the order expects.
        games = int(rows.max()) + 1 if len(rows) else 0
        mask = np.zeros((games, ACTIONS), dtype=bool)
        mask[rows] = legal
        order, logits, _value, _guessed = contract.root_order(
            self.served, None, views, rows, players, mask, self.device
        )
        top = np.nan_to_num(
            torch.softmax(logits.float(), dim=1).max(dim=1).values.cpu().numpy(), nan=1.0
        )
        planes, _mask = self.served.root(None, views, rows, players, mask, self.device)
        features, pooled = sibling_head.features_of(self.net, planes)
        scores = self.head(features, pooled).float().cpu().numpy()
        choice = np.empty(len(rows), dtype=np.int64)
        for at, game in enumerate(rows):
            first = order[game]
            candidates = first[: self.k]
            candidates = candidates[mask[game][candidates]]
            if len(candidates) == 0:
                choice[at] = int(np.nonzero(mask[game])[0][0]) if mask[game].any() else 0
                continue
            if len(candidates) < 2 or top[at] >= self.sure:
                self.taken_sure += 1
                choice[at] = int(candidates[0])
                continue
            self.asked += 1
            worth = scores[at][candidates]
            best = int(np.argmax(worth))
            if best != 0 and worth[best] - worth[0] > self.margin:
                self.overrode += 1
                choice[at] = int(candidates[best])
            else:
                choice[at] = int(candidates[0])
        return choice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("--games", type=int, default=200, help="deals per seating")
    parser.add_argument("--seed", type=int, default=555_000)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument(
        "--sure",
        type=float,
        default=None,
        help="the policy's probability on its first move at or above which "
        "the head is not asked; by default what the head's recordings were "
        "gated at, one for everywhere",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from . import duel, zoo

    plain = zoo.load_player(args.checkpoint, args.device)
    net = contract.unwrap(zoo.load_player(args.checkpoint, args.device))
    head, meta = sibling_head.load(args.head, args.device)
    sure = float(meta.get("sure", 1.0)) if args.sure is None else args.sure
    ranked = RankedPlayer(net, head, k=args.k, margin=args.margin, device=args.device, sure=sure)
    result = duel.duel(ranked, plain, args.games, args.seed, device=args.device)
    result["challenger"] = f"{args.checkpoint} with {args.head} (k={args.k}, margin={args.margin}, sure={sure})"
    result["incumbent"] = str(args.checkpoint)
    result["overrides"] = f"{ranked.overrode} of {ranked.asked}, {ranked.taken_sure} more taken sure"
    result["verdict"] = duel.verdict(result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
