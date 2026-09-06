"""Which of two networks is better, asked directly.

Measuring each network against the heuristic players and subtracting has a
floor: each figure carries an error near 0.017 over four thousand games,
so the difference carries 0.024, and separating two networks at two
standard errors needs a gap of about 0.05. Four times the games only
halves it. When two networks are close, that comparison never settles, and
the run reaches a point where every further generation is invisible to it.

Sitting them at the same table removes the problem. One network takes one
place and the other takes the remaining three, in the same games, so the
luck of the deal falls on both at once and cancels. What comes back is not
two placements to subtract but one: the challenger's, against the 2.50 it
would average if the two were the same player. Each deal contributes one
figure and the error comes from the deals, as in `arena.py`, and each set
of deals is played four times with the challenger in each seat.

    python -m neural.duel challenger.pt incumbent.pt --games 1000
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

import riichi_py

from .model import build, load_weights

PLANES = riichi_py.PLANES
POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS
SEATS = 4


@torch.no_grad()
def table(
    challenger,
    incumbent,
    games: int,
    seed: int,
    place: int,
    device: str = "cuda",
    max_steps: int = 4000,
) -> np.ndarray:
    """Plays `games` games with the challenger at `place` and the incumbent
    at the other three, and returns the final scores.

    Both networks play their best move rather than sampling, which is what
    the browser does and what the comparison is about.
    """
    challenger.eval()
    incumbent.eval()
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=[])
    steps = 0
    while not arena.all_finished() and steps < max_steps:
        steps += 1
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        if not live.any():
            break

        planes = np.frombuffer(arena.observations(), dtype=np.float32)
        planes = planes.reshape(games, PLANES, POSITIONS)
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        mask = mask.reshape(games, ACTIONS).astype(bool)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, SEATS)

        # Who owes each live game's decision, as a person rather than a
        # seat: the seats move between hands and the players do not.
        index = np.nonzero(live)[0]
        owner = np.array([players[game][seats[game]] for game in index])
        choice = np.zeros(games, dtype=np.int64)

        for net, wanted in ((challenger, owner == place), (incumbent, owner != place)):
            rows = index[wanted]
            if not len(rows):
                continue
            logits, _value = net(
                torch.from_numpy(planes[rows]).to(device),
                torch.from_numpy(mask[rows]).to(device),
            )
            choice[rows] = logits.argmax(dim=1).cpu().numpy()

        arena.step(choice.tolist())

    return np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, SEATS).copy()


def placements(scores: np.ndarray, place: int) -> np.ndarray:
    order = (-scores).argsort(axis=1).argsort(axis=1) + 1
    return order[:, place]


def duel(challenger, incumbent, games: int, seed: int, device: str = "cuda") -> dict:
    """The same deals four times, with the challenger in each seat."""
    per_seat = []
    per_deal = []
    for place in range(SEATS):
        scores = table(challenger, incumbent, games, seed, place, device=device)
        got = placements(scores, place)
        per_seat.append(
            {
                "place": place,
                "placement": float(got.mean()),
                "score": float(scores[:, place].mean()),
                "wins": float((got == 1).mean()),
            }
        )
        per_deal.append(got.astype(float))

    overall = float(statistics.fmean(row["placement"] for row in per_seat))
    paired = np.stack(per_deal).mean(axis=0)
    error = float(paired.std(ddof=1) / (len(paired) ** 0.5)) if len(paired) > 1 else 0.0
    return {
        "games_per_seat": games,
        "games_total": games * SEATS,
        "placement": overall,
        "standard_error": error,
        "by_seat": per_seat,
        "by_deal": [round(float(value), 4) for value in paired],
    }


def verdict(result: dict) -> str:
    """Two identical players average 2.50 at the same table, so that is the
    line, and the challenger is below it when it is the better network."""
    edge = 2.5 - result["placement"]
    error = result["standard_error"]
    if error == 0:
        return "not enough deals to say"
    sigmas = edge / error
    size = f"{edge:+.4f} placement, {sigmas:+.1f} standard errors"
    if sigmas > 2:
        return f"the challenger is stronger: {size}"
    if sigmas < -2:
        return f"the incumbent is stronger: {size}"
    return f"not settled either way, which needs more deals: {size}"


def load(path: Path, channels: int, blocks: int, device: str):
    payload = torch.load(path, map_location=device, weights_only=True)
    net = build(channels=channels, blocks=blocks, device=device)
    load_weights(net, payload["model"])
    net.eval()
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("challenger", type=Path)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("--games", type=int, default=1000, help="deals per seating")
    parser.add_argument("--seed", type=int, default=555_000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--channels", type=int, default=320)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--incumbent-channels", type=int, default=192)
    parser.add_argument("--incumbent-blocks", type=int, default=10)
    args = parser.parse_args()

    challenger = load(args.challenger, args.channels, args.blocks, args.device)
    incumbent = load(
        args.incumbent, args.incumbent_channels, args.incumbent_blocks, args.device
    )
    result = duel(challenger, incumbent, args.games, args.seed, device=args.device)
    result["challenger"] = str(args.challenger)
    result["incumbent"] = str(args.incumbent)
    result["verdict"] = verdict(result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
