"""A warm start: teach the network what a better player would do.

By default the teacher is the heuristic player that ships with the game.
With `--teacher` it is a checkpoint instead, which is how a network that
started its lineage from scratch can be given what an older lineage
already learned rather than rediscovering it a generation at a time.

Learning riichi from nothing by self-play alone works, but it starts from a
policy that discards at random, and most of what it must first discover is
not strategy at all: keep the tiles that go together, do not open a hand
that can never be declared, take a win when it is there. The heuristic
player already knows those, so the network is first taught to imitate it,
and self-play then has somewhere to improve from rather than somewhere to
begin.

This uses no human data and no outside model. The teacher is the same
handful of rules that ships with the game, and it is a floor, not a
ceiling: the point of the self-play that follows is to pass it.

Usage:
  python -m neural.imitate --rounds 200 --games 64 --out E:/tmp-claude/mahjong/clone
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

from . import selfplay
from .model import PolicyValueNet, load_weights

PLANES = riichi_py.PLANES
POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS
OPPONENTS = riichi_py.OPPONENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--games", type=int, default=64, help="tables per round")
    parser.add_argument("--channels", type=int, default=320)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--measure-every", type=int, default=20)
    parser.add_argument("--measure-games", type=int, default=192)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=None,
        help="a checkpoint to learn from instead of the heuristic player",
    )
    parser.add_argument("--teacher-channels", type=int, default=192)
    parser.add_argument("--teacher-blocks", type=int, default=10)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="how sharp the teacher's distribution is made before the "
        "student learns it. Above one keeps more of what it was unsure of",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out", type=Path, default=Path("E:/tmp-claude/mahjong/clone"))
    return parser.parse_args()


def collect(
    games: int, seed: int, teacher=None, device: str = "cpu", temperature: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Plays a round with one teacher at every place, keeping every position
    it saw, the move it made there, and what the other three were actually
    holding.

    With no teacher given that is the heuristic player and the label is its
    move. With a checkpoint, the label is its move and the full distribution
    comes back as well: a teacher torn between two discards says so, and
    that is worth more to a student than the argmax alone.
    """
    arena = riichi_py.Arena(games=games, seed=seed)
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    held: list[np.ndarray] = []
    spread: list[np.ndarray] = []

    for _step in range(4000):
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        if not live.any():
            break
        advice = (
            np.frombuffer(arena.teacher(), dtype=np.uint8)
            if teacher is None
            else np.zeros(games, dtype=np.uint8)
        )
        planes = np.frombuffer(arena.observations(), dtype=np.float32)
        planes = planes.reshape(games, PLANES, POSITIONS)
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        mask = mask.reshape(games, ACTIONS).astype(bool)
        truth = np.frombuffer(arena.opponent_hands(), dtype=np.float32)
        truth = truth.reshape(games, OPPONENTS, POSITIONS)

        odds = None
        if teacher is not None:
            rows = np.nonzero(live)[0]
            with torch.no_grad():
                logits, _value = teacher(
                    torch.from_numpy(planes[rows]).to(device),
                    torch.from_numpy(mask[rows]).to(device),
                )
                soft = torch.softmax(logits.float() / temperature, dim=1).cpu().numpy()
            odds = np.zeros((games, ACTIONS), dtype=np.float32)
            odds[rows] = soft
            advice = np.zeros(games, dtype=np.int64)
            advice[rows] = soft.argmax(axis=1)

        choice = np.zeros(games, dtype=np.int64)
        for index in np.nonzero(live)[0]:
            wanted = int(advice[index])
            # The teacher only ever names something it is allowed to do, but
            # a position it cannot express falls back to the first legal move.
            if wanted >= ACTIONS or not mask[index][wanted]:
                wanted = int(np.argmax(mask[index]))
            observations.append(planes[index])
            masks.append(mask[index])
            labels.append(wanted)
            held.append(truth[index])
            if odds is not None:
                spread.append(odds[index])
            choice[index] = wanted
        arena.step(choice.tolist())

    return (
        np.stack(observations),
        np.stack(masks),
        np.array(labels, dtype=np.int64),
        np.stack(held),
        np.stack(spread) if spread else None,
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    net = PolicyValueNet(args.channels, args.blocks).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    teacher = None
    if args.teacher is not None:
        teacher = PolicyValueNet(args.teacher_channels, args.teacher_blocks).to(device)
        load_weights(
            teacher, torch.load(args.teacher, map_location=device, weights_only=True)["model"]
        )
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        print(
            f"teacher {args.teacher} at {args.teacher_channels}x{args.teacher_blocks}",
            flush=True,
        )
    print(
        f"device {device} | {net.channels}x{net.blocks} "
        f"| {net.parameter_count() / 1e6:.2f}M parameters",
        flush=True,
    )

    for round_index in range(args.rounds):
        began = time.time()
        planes, masks, labels, truth, spread = collect(
            args.games,
            args.seed + round_index * 977,
            teacher=teacher,
            device=device,
            temperature=args.temperature,
        )
        played = time.time() - began
        observations = torch.from_numpy(planes).to(device)
        legal = torch.from_numpy(masks).to(device)
        targets = torch.from_numpy(labels).to(device)
        held = torch.from_numpy(truth).to(device)
        odds = torch.from_numpy(spread).to(device) if spread is not None else None

        net.train()
        total_loss = 0.0
        total_covered = 0.0
        agreed = 0
        seen = 0
        for _epoch in range(args.epochs):
            order = torch.randperm(len(targets), device=device)
            for start in range(0, len(targets), args.batch):
                picks = order[start : start + args.batch]
                if picks.numel() < 2:
                    continue
                logits, _value, guessed = net.everything(observations[picks], legal[picks])
                if odds is None:
                    loss = nn.functional.cross_entropy(logits, targets[picks])
                else:
                    # The teacher's whole distribution, not just its choice:
                    # what it was unsure of is worth as much as what it was
                    # sure of.
                    #
                    # An illegal move has a masked logit of negative
                    # infinity and no weight from the teacher, and zero
                    # times negative infinity is not zero, it is NaN, which
                    # is how this read NaN on its first run. Take the log
                    # only where the teacher put weight.
                    weight = odds[picks]
                    log_odds = torch.log_softmax(logits, dim=1)
                    log_odds = torch.where(weight > 0, log_odds, torch.zeros_like(log_odds))
                    loss = -(weight * log_odds).sum(dim=1).mean()

                # And what the other three were holding, which is exact and
                # costs nothing to know here.
                wanted = held[picks]
                holding = wanted.sum(dim=2) > 0
                log_guess = torch.log_softmax(guessed, dim=2)
                reading = -(wanted * log_guess).sum(dim=2)
                loss = loss + (reading * holding).sum() / holding.sum().clamp(min=1)
                with torch.no_grad():
                    overlap = torch.minimum(log_guess.exp(), wanted).sum(dim=2)
                    total_covered += float(
                        ((overlap * holding).sum() / holding.sum().clamp(min=1)) * picks.numel()
                    )
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item() * picks.numel()
                agreed += int((logits.argmax(dim=1) == targets[picks]).sum())
                seen += picks.numel()

        record = {
            "round": round_index,
            "positions": int(len(targets)),
            "loss": round(total_loss / max(seen, 1), 4),
            "agreement": round(agreed / max(seen, 1), 4),
            "hands_read": round(total_covered / max(seen, 1), 4),
            "seconds": round(time.time() - began, 1),
            # Kept apart so a slow round can be blamed on the right half:
            # the engine playing on the CPU, or the GPU being fed too slowly.
            "play_seconds": round(played, 1),
        }
        if (round_index + 1) % args.measure_every == 0 or round_index == 0:
            against = selfplay.measure(
                net, games=args.measure_games, seed=8_000_000 + round_index, device=device
            )
            record.update(
                {
                    "placement": round(against["placement"], 3),
                    "score": round(against["score"], 1),
                    "win_rate": round(against["wins"], 3),
                }
            )
            torch.save(
                {"model": net.state_dict(), "generation": 0,
                 "channels": net.channels, "blocks": net.blocks},
                args.out / "latest.pt",
            )
        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    torch.save(
        {"model": net.state_dict(), "generation": 0,
         "channels": net.channels, "blocks": net.blocks},
        args.out / "latest.pt",
    )
    print("imitation finished", flush=True)


if __name__ == "__main__":
    main()
