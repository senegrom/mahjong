"""How strong the network really is, measured so the answer can be trusted.

A single measurement of average placement over a few hundred games carries
a standard error of about 0.05, which is the size of the improvement being
looked for. Keeping the best of many such measurements then picks the
luckiest network rather than the best one, and the number that made the
choice is the one number guaranteed to be optimistic.

This measures the same deals four times, once with the network in each
seat, so the luck of the deal falls on both sides equally. Against three
copies of the heuristic bot, a network no better than they are averages
2.5, and anything below that is a real edge.

How much of it to believe comes from the deals rather than from the four
seatings. The seatings share their deals and so are far from independent:
a network indistinguishable from the bots would play the same four games
and its placements would sum to exactly ten every time. Each deal therefore
contributes one figure, the four placements it produced averaged, and the
spread of those figures across deals is the error on the mean.

    python -m neural.arena E:/tmp-claude/mahjong/run4/best.pt --games 1000
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from .model import build
from .selfplay import play

SEATS = 4


def placements(scores: np.ndarray, seat: int) -> np.ndarray:
    """Where the player in `seat` finished each game, from 1 to 4."""
    order = (-scores).argsort(axis=1).argsort(axis=1) + 1
    return order[:, seat]


@torch.no_grad()
def duplicate(net, games: int, seed: int, device: str = "cuda") -> dict:
    """Plays the same deals with the network in each of the four seats."""
    per_seat = []
    for seat in range(SEATS):
        bots = [place for place in range(SEATS) if place != seat]
        batch = play(
            net,
            games=games,
            seed=seed,
            device=device,
            bot_places=bots,
            greedy=True,
        )
        got = placements(batch.final_scores, seat)
        per_seat.append(
            {
                "seat": seat,
                "placement": float(got.mean()),
                "score": float(batch.final_scores[:, seat].mean()),
                "wins": float((got == 1).mean()),
                "hands": batch.hands,
                # Kept for the error bar below, then dropped from the report.
                "placements": got.astype(float),
            }
        )

    means = [row["placement"] for row in per_seat]
    overall = float(statistics.fmean(means))

    # The error bar has to come from the deals, not from the four seatings.
    #
    # The seatings share their deals, so their placements are far from
    # independent: were the network no different from the bots, the four
    # would play the same games and sum to exactly ten for every deal, and
    # the average would be exactly 2.5 with no error at all. That is the
    # whole point of the design, and it means the spread between seatings
    # measures only the part that does not cancel, which is not the error on
    # their average. Using it claimed four standard errors where the honest
    # number was nearer one.
    #
    # So each deal gives one figure: the four placements it produced,
    # averaged. Those figures are independent across deals, and their spread
    # is the error on the mean.
    per_deal = np.stack([row.pop("placements") for row in per_seat]).mean(axis=0)
    error = float(per_deal.std(ddof=1) / (len(per_deal) ** 0.5)) if len(per_deal) > 1 else 0.0

    return {
        "games_per_seat": games,
        "games_total": games * SEATS,
        "placement": overall,
        "standard_error": error,
        "score": float(statistics.fmean(row["score"] for row in per_seat)),
        "wins": float(statistics.fmean(row["wins"] for row in per_seat)),
        "by_seat": per_seat,
    }


def verdict(result: dict) -> str:
    """What the numbers permit saying out loud.

    A network no stronger than the bots averages 2.5, so that is the line.
    Saying how many standard errors the difference comes to is more use than
    a yes or no: a result at one and a half is worth more games rather than
    a decision either way.
    """
    edge = 2.5 - result["placement"]
    error = result["standard_error"]
    if error == 0:
        return "not enough seatings to say"
    sigmas = edge / error
    size = f"{edge:+.4f} placement, {sigmas:+.1f} standard errors"
    if sigmas > 2:
        return f"stronger than the heuristic bot: {size}"
    if sigmas < -2:
        return f"weaker than the heuristic bot: {size}"
    return f"not settled either way, which needs more games: {size}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=500, help="deals per seating")
    parser.add_argument("--seed", type=int, default=555_000)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=10)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    net = build(channels=args.channels, blocks=args.blocks)
    net.load_state_dict(state["model"])
    net.eval()

    result = duplicate(net, games=args.games, seed=args.seed)
    result["checkpoint"] = str(args.checkpoint)
    result["verdict"] = verdict(result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
