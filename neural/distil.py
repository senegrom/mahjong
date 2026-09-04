"""Teaches the network the moves its own search came to.

This is the policy improvement half of the AlphaZero idea, in the form this
game allows. The network proposes an order over the moves; the search plays
the top few out in worlds drawn from what the network believes the
opponents hold; whichever survives that is a better move than the one
proposed, or the search is worth nothing. Training the network towards it
makes the next proposal better, and the search after that better again.

It also solves the practical problem with search, which is that half a
second a decision is hopeless in a browser. A network taught the search's
answers plays them without searching.

    python -m neural.distil --resume E:/tmp-claude/mahjong/big-run/best.pt \
        --rounds 200 --games 24 --out E:/tmp-claude/mahjong/distilled
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import riichi_py

from .model import build
from .selfplay import measure

PLANES = riichi_py.PLANES
POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS
OPPONENTS = riichi_py.OPPONENTS
HANDS = riichi_py.HANDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--games", type=int, default=24, help="tables per round")
    parser.add_argument("--worlds", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument(
        "--hurried",
        action="store_true",
        help="play the imagined worlds out without counting acceptance, "
        "which is most of what a rollout costs",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--channels", type=int, default=320)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--measure-every", type=int, default=10)
    parser.add_argument("--measure-games", type=int, default=384)
    parser.add_argument("--seed", type=int, default=4_040_404)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def collect(net, args, seed: int, device: str) -> tuple[np.ndarray, ...]:
    """Plays a round where every decision is searched, keeping the answers.

    Every seat searches, so a round yields four times the positions a single
    searching seat would, and the play the positions come from is the play
    the network will actually meet.
    """
    net.eval()
    arena = riichi_py.Arena(games=args.games, seed=seed, bot_places=[])
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    held: list[np.ndarray] = []
    proposed: list[int] = []

    for _step in range(4000):
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        if not live.any():
            break

        planes = np.frombuffer(arena.observations(), dtype=np.float32)
        planes = planes.reshape(args.games, PLANES, POSITIONS)
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        mask = mask.reshape(args.games, ACTIONS).astype(bool)
        truth = np.frombuffer(arena.opponent_hands(), dtype=np.float32)
        truth = truth.reshape(args.games, OPPONENTS, POSITIONS)

        logits, _value, guessed = net.everything(
            torch.from_numpy(planes).to(device),
            torch.from_numpy(mask).to(device),
        )
        order = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
        belief = torch.softmax(guessed, dim=2).reshape(args.games, HANDS).cpu().numpy()

        ranked = [
            [int(index) for index in order[game][: args.candidates]]
            for game in range(args.games)
        ]
        chosen = arena.search(
            ranked,
            belief.reshape(-1).tolist(),
            worlds=args.worlds,
            candidates=args.candidates,
            margin=args.margin,
            hurried=args.hurried,
        )

        for game in np.nonzero(live)[0]:
            observations.append(planes[game])
            masks.append(mask[game])
            labels.append(int(chosen[game]))
            held.append(truth[game])
            proposed.append(int(order[game][0]))

        arena.step(list(chosen))

    return (
        np.stack(observations),
        np.stack(masks),
        np.array(labels, dtype=np.int64),
        np.stack(held),
        np.array(proposed, dtype=np.int64),
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    net = build(channels=args.channels, blocks=args.blocks, device=device)
    payload = torch.load(args.resume, map_location=device, weights_only=True)
    net.load_state_dict(payload["model"])
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        f"device {device} | {net.channels}x{net.blocks} "
        f"| {net.parameter_count() / 1e6:.2f}M parameters | resumed {args.resume}",
        flush=True,
    )

    best_placement = float("inf")
    smoothed = None
    for round_index in range(args.rounds):
        began = time.time()
        planes, masks, labels, truth, proposed = collect(
            net, args, args.seed + round_index * 977, device
        )
        played = time.time() - began
        observations = torch.from_numpy(planes).to(device)
        legal = torch.from_numpy(masks).to(device)
        targets = torch.from_numpy(labels).to(device)
        wanted_hands = torch.from_numpy(truth).to(device)

        # How often the search disagreed with what the network proposed. If
        # this is zero there is nothing to learn and the search is a no-op;
        # if it is most of the time the margin is not doing its job.
        changed = float((labels != proposed).mean())

        net.train()
        total_loss = 0.0
        seen = 0
        for _epoch in range(args.epochs):
            order = torch.randperm(len(targets), device=device)
            for start in range(0, len(targets), args.batch):
                picks = order[start : start + args.batch]
                if picks.numel() < 2:
                    continue
                logits, _value, guessed = net.everything(
                    observations[picks], legal[picks]
                )
                loss = nn.functional.cross_entropy(logits, targets[picks])

                # The table-reading head keeps learning here too: the label
                # is exact and the search depends on it.
                mine = wanted_hands[picks]
                holding = mine.sum(dim=2) > 0
                log_guess = torch.log_softmax(guessed, dim=2)
                reading = -(mine * log_guess).sum(dim=2)
                loss = loss + (reading * holding).sum() / holding.sum().clamp(min=1)

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item() * picks.numel()
                seen += picks.numel()

        record = {
            "round": round_index,
            "positions": int(len(targets)),
            "search_changed": round(changed, 4),
            "loss": round(total_loss / max(seen, 1), 4),
            "seconds": round(time.time() - began, 1),
            "play_seconds": round(played, 1),
        }

        if (round_index + 1) % args.measure_every == 0 or round_index == 0:
            against = measure(net, games=args.measure_games, seed=9_000 + round_index)
            record.update(
                {
                    "placement": round(against["placement"], 3),
                    "score": round(against["score"], 1),
                    "win_rate": round(against["wins"], 3),
                }
            )
            smoothed = (
                against["placement"]
                if smoothed is None
                else against["placement"] / 3 + smoothed * 2 / 3
            )
            record["smoothed"] = round(smoothed, 3)
            saved = {
                "model": net.state_dict(),
                "generation": round_index + 1,
                "channels": net.channels,
                "blocks": net.blocks,
                "placement": against["placement"],
            }
            torch.save(saved, args.out / "latest.pt")
            if smoothed < best_placement:
                best_placement = smoothed
                torch.save(saved, args.out / "best.pt")
                record["best"] = True

        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    print("distilling finished", flush=True)


if __name__ == "__main__":
    main()
