"""The training loop: play, learn, measure, repeat.

Advantage actor-critic on the game's own reward. The network plays every
place at every table, so a round of self-play produces four trajectories per
game and no opponent has to be found from anywhere. What a decision was
worth is the points the hand moved plus the placement the game ended in, and
the value head learns to predict that, so the policy is pushed toward the
decisions that did better than the network expected rather than merely
toward the ones that did well.

Every so often the network plays three heuristic opponents, which is the
only measurement that means anything on its own: average placement, where
2.5 is even and lower is better.

Usage:
  python -m neural.train --generations 200 --games 128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from . import selfplay
from .model import PolicyValueNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--games", type=int, default=128, help="tables per round")
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--clip", type=float, default=0.2, help="PPO ratio clip")
    parser.add_argument("--batch", type=int, default=4096, help="decisions per step")
    parser.add_argument("--epochs", type=int, default=3, help="passes over a round")
    parser.add_argument("--entropy", type=float, default=0.03)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--measure-every", type=int, default=10)
    parser.add_argument("--measure-games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out", type=Path, default=Path("E:/tmp-claude/mahjong/run1"))
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The environment runs on this thread and the network on the GPU, so a
    # couple of worker threads is plenty and leaves the machine usable.
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    net = PolicyValueNet(args.channels, args.blocks).to(device)
    start = 0
    best_placement = float("inf")
    if args.resume and args.resume.exists():
        payload = torch.load(args.resume, map_location=device, weights_only=True)
        net.load_state_dict(payload["model"])
        start = int(payload.get("generation", 0))
        if payload.get("channels") not in (None, net.channels):
            raise SystemExit("the checkpoint was trained at a different width")
        print(f"resumed from {args.resume} at generation {start}", flush=True)

    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        f"device {device} | {net.channels} channels x {net.blocks} blocks "
        f"| {net.parameter_count() / 1e6:.2f}M parameters",
        flush=True,
    )

    for generation in range(start, args.generations):
        began = time.time()
        batch = selfplay.play(
            net,
            games=args.games,
            seed=args.seed + generation * 1000,
            device=device,
        )
        played = time.time() - began

        observations = batch.observations.to(device)
        legal = batch.legal.to(device)
        actions = batch.actions.to(device)
        returns = batch.returns.to(device)
        old_log_probs = batch.log_probs.to(device)

        # A return that is mostly noise teaches nothing; standardising it
        # keeps the gradient the same size from one round to the next.
        normalised = (returns - returns.mean()) / (returns.std() + 1e-6)

        net.train()
        total_policy = total_value = total_entropy = 0.0
        total_clipped = 0.0
        steps = 0
        for _epoch in range(args.epochs):
            order = torch.randperm(batch.decisions, device=device)
            for start_index in range(0, batch.decisions, args.batch):
                picks = order[start_index : start_index + args.batch]
                if picks.numel() < 2:
                    continue
                logits, value = net(observations[picks], legal[picks])
                distribution = torch.distributions.Categorical(logits=logits)
                log_prob = distribution.log_prob(actions[picks])
                advantage = (normalised[picks] - value).detach()

                # The clipped objective: an update may improve an action's
                # odds, but only so far in one round, which is what keeps a
                # policy from narrowing onto a single action.
                ratio = torch.exp(log_prob - old_log_probs[picks])
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
                total_clipped += float((ratio != clipped).float().mean())
                value_loss = nn.functional.mse_loss(value, normalised[picks])
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + args.value_weight * value_loss
                    - args.entropy * entropy
                )

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimiser.step()

                total_policy += policy_loss.item()
                total_value += value_loss.item()
                total_entropy += entropy.item()
                steps += 1

        record = {
            "generation": generation,
            "decisions": batch.decisions,
            "hands": batch.hands,
            "seconds": round(time.time() - began, 1),
            "play_seconds": round(played, 1),
            "policy_loss": round(total_policy / max(steps, 1), 4),
            "value_loss": round(total_value / max(steps, 1), 4),
            "entropy": round(total_entropy / max(steps, 1), 4),
            "clipped": round(total_clipped / max(steps, 1), 3),
            "mean_return": round(float(returns.mean()), 4),
        }

        if (generation + 1) % args.measure_every == 0 or generation == 0:
            against = selfplay.measure(
                net, games=args.measure_games, seed=7_000_000 + generation, device=device
            )
            record.update(
                {
                    "placement": round(against["placement"], 3),
                    "score": round(against["score"], 1),
                    "win_rate": round(against["wins"], 3),
                }
            )
            payload = {
                "model": net.state_dict(),
                "generation": generation + 1,
                "channels": net.channels,
                "blocks": net.blocks,
                "placement": against["placement"],
            }
            torch.save(payload, args.out / "latest.pt")
            # The high-water mark is kept apart, so a run that wanders can
            # always be brought back to the best network it has produced.
            if against["placement"] < best_placement:
                best_placement = against["placement"]
                torch.save(payload, args.out / "best.pt")
                record["best"] = True

        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    torch.save(
        {"model": net.state_dict(), "generation": args.generations,
         "channels": net.channels, "blocks": net.blocks},
        args.out / "latest.pt",
    )
    print("training finished", flush=True)


if __name__ == "__main__":
    main()
