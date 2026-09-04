"""Does the network play better when it thinks ahead?

The network supplies three things and the engine does the rest: an order
over the moves worth trying, what it takes the opponents to be holding, and
what a position is worth. The engine imagines the worlds and makes the
moves; the network values every position that results in one pass; the
engine picks. Nothing is played out to the end by a heuristic, which is
what the first version did and what measured worse than not searching.

The comparison is the same deals four times over with the searching player
in each chair, and its error bar comes from the deals rather than the four
seatings, for the reason set out in `arena.py`: the seatings share their
deals and are nowhere near independent.

    python -m neural.searched E:/tmp-claude/mahjong/big-run/best.pt --games 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np
import torch

import riichi_py

from .model import build, load_weights

PLANES = riichi_py.PLANES
POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS
OPPONENTS = riichi_py.OPPONENTS
HANDS = riichi_py.HANDS
SEATS = 4


def search_with_value_head(net, arena, ranked, belief_flat, *, worlds, candidates, margin, hurried, device="cuda"):
    """One searched decision for every live game, valued by the network.

    `ranked` is the network's move order per game, best first; `belief_flat`
    is its belief about the opponents' hands, one row of HANDS per game.
    The engine imagines the worlds and makes the moves; the network values
    every resulting position in a single pass; the engine picks, keeping the
    first move unless another beats it by `margin` standard errors of the
    world-by-world difference.
    """
    planes_bytes, counts, _settled, _wanted = arena.leaves(
        ranked, belief_flat, worlds=worlds, candidates=candidates, hurried=hurried
    )
    total = sum(counts)
    if total == 0:
        return arena.decide([], margin, ranked)
    planes = np.frombuffer(planes_bytes, dtype=np.float32).reshape(total, PLANES, POSITIONS)
    # Every slot is valued, including the few that want no value; the engine
    # adds what it settled itself, ignores the rest, and that is cheaper
    # than gathering.
    valued = np.empty(total, dtype=np.float32)
    step = 8192
    for start in range(0, total, step):
        chunk = torch.from_numpy(planes[start : start + step]).to(device)
        valued[start : start + step] = net.value_only(chunk).float().cpu().numpy()
    return arena.decide(valued.tolist(), margin, ranked)


@torch.no_grad()
def play(
    net,
    games: int,
    seed: int,
    searcher: int | None,
    worlds: int,
    candidates: int,
    margin: float,
    hurried: bool = True,
    device: str = "cuda",
) -> tuple[np.ndarray, tuple[int, int]]:
    """Plays `games` games out and returns the final scores.

    `searcher` is the place that thinks ahead, or None for nobody. Everyone
    else plays the network's first choice, so the only thing that differs
    between the two arms is whether that choice was checked.
    """
    net.eval()
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=[])
    steps = 0
    while not arena.all_finished() and steps < 4000:
        steps += 1
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        if not (seats != 0xFF).any():
            break

        planes = np.frombuffer(arena.observations(), dtype=np.float32)
        planes = planes.reshape(games, PLANES, POSITIONS)
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        mask = mask.reshape(games, ACTIONS).astype(bool)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, SEATS)

        logits, _value, guessed = net.everything(
            torch.from_numpy(planes).to(device),
            torch.from_numpy(mask).to(device),
        )
        # The network's order over the moves, best first.
        order = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
        belief = torch.softmax(guessed, dim=2).reshape(games, HANDS).cpu().numpy()

        if searcher is None:
            choice = order[:, 0].tolist()
        else:
            # Only the searching player's own turns are searched; the others
            # take the network's first choice, which is what `ranked` gives
            # back when a game is not theirs to think about.
            ranked = []
            for game in range(games):
                seat = int(seats[game])
                thinking = seat != 0xFF and int(players[game][seat]) == searcher
                ranked.append(
                    [int(index) for index in order[game][: candidates if thinking else 1]]
                )
            choice = search_with_value_head(
                net,
                arena,
                ranked,
                belief.reshape(-1).tolist(),
                worlds=worlds,
                candidates=candidates,
                margin=margin,
                hurried=hurried,
                device=device,
            )
        arena.step(list(choice))

    scores = np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, SEATS).copy()
    return scores, arena.search_tally()


def placements(scores: np.ndarray, place: int) -> np.ndarray:
    order = (-scores).argsort(axis=1).argsort(axis=1) + 1
    return order[:, place]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=200, help="deals per chair")
    parser.add_argument("--seed", type=int, default=90_210)
    parser.add_argument("--worlds", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--channels", type=int, default=320)
    parser.add_argument("--blocks", type=int, default=20)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    net = build(channels=args.channels, blocks=args.blocks)
    load_weights(net, state["model"])

    per_chair = []
    per_deal = []
    asked = overrode = 0
    for chair in range(SEATS):
        began = time.perf_counter()
        scores, tally = play(
            net,
            games=args.games,
            seed=args.seed,
            searcher=chair,
            worlds=args.worlds,
            candidates=args.candidates,
            margin=args.margin,
        )
        asked += tally[0]
        overrode += tally[1]
        got = placements(scores, chair)
        print(
            f"chair {chair}: placement {got.mean():.3f} over {args.games} games, "
            f"search changed {tally[1]} of {tally[0]} decisions, "
            f"{time.perf_counter() - began:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        per_chair.append(
            {
                "chair": chair,
                "placement": float(got.mean()),
                "score": float(scores[:, chair].mean()),
                "wins": float((got == 1).mean()),
            }
        )
        per_deal.append(got.astype(float))

    overall = statistics.fmean(row["placement"] for row in per_chair)
    paired = np.stack(per_deal).mean(axis=0)
    error = float(paired.std(ddof=1) / (len(paired) ** 0.5))
    edge = 2.5 - overall
    sigmas = edge / error if error else 0.0

    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "worlds": args.worlds,
                "candidates": args.candidates,
                "margin": args.margin,
                "games_total": args.games * SEATS,
                "placement": overall,
                "standard_error": error,
                "difference_from_level": edge,
                "standard_errors": sigmas,
                "by_chair": per_chair,
                "overrides": f"{overrode} of {asked}, {100.0 * overrode / max(asked, 1):.1f}%",
                "verdict": (
                    "searching helps"
                    if sigmas > 2
                    else "searching hurts"
                    if sigmas < -2
                    else "not settled either way, which needs more games"
                ),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
