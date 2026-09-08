"""Training the joined player: our network and a Mortal beneath one head.

The same self-play loop as `train.py`, for `combined.Combined`. The head
and our value head train every generation; of the two networks beneath,
each generation draws at random which stays fixed, Mortal, ours, or
neither, so that each is moved by the other's company without either
being overwritten in one go. The policy gradient is PPO's clipped
objective on the head's output, the baseline is our value head as it
stood before the round, and the value head learns the return.

    python -m neural.train_combined --ours ours.pt --mortal mortal.pt --rounds 20 --out runs/joined
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import combined, selfplay, zoo
from .observe import pad_rows, resident
from .prefetch import Prefetcher

SMOOTHING = 1 / 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, default=None, help="our checkpoint to start from")
    parser.add_argument("--mortal", type=Path, default=None, help="the Mortal to start from")
    parser.add_argument("--resume", type=Path, default=None, help="a checkpoint this trainer wrote")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4, help="for the head and the value head")
    parser.add_argument("--lr-ours", type=float, default=4e-5, help="for our network beneath")
    parser.add_argument("--lr-mortal", type=float, default=3e-5, help="for the Mortal beneath")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--entropy", type=float, default=0.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument(
        "--fixed",
        nargs="*",
        default=list(combined.Combined.MODES),
        help="which of Mortal, ours or none stays fixed, drawn evenly each "
        "generation from these",
    )
    parser.add_argument("--opponents", type=Path, nargs="*", default=[])
    parser.add_argument("--opponent-share", type=float, default=0.0)
    parser.add_argument("--measure-every", type=int, default=5)
    parser.add_argument("--measure-games", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--out", type=Path, default=Path("E:/tmp-claude/mahjong/joined-run"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = args.amp and device == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    start = 0
    best_placement = float("inf")
    smoothed = None
    optimiser_state = None
    if args.resume is not None and args.resume.exists():
        net, payload = combined.load(args.resume, device)
        start = int(payload.get("generation", 0))
        smoothed = payload.get("smoothed")
        best_placement = float(payload.get("best_placement", float("inf")))
        optimiser_state = payload.get("optimizer_state")
        print(f"resumed from {args.resume} at generation {start}", flush=True)
    elif args.ours is not None and args.mortal is not None:
        net, _config = combined.build(args.ours, args.mortal, device)
        print(f"joined {args.ours} and {args.mortal}", flush=True)
    else:
        raise SystemExit("give --ours and --mortal to start, or --resume")

    # One optimiser over everything, in three groups; a group whose
    # parameters are fixed this generation gets no gradients and is left
    # alone by it.
    net.set_mode("none")
    optimiser = torch.optim.AdamW(
        [
            {"params": net.always_trained(), "lr": args.lr},
            {"params": net.ours_trained(), "lr": args.lr_ours},
            {"params": net.mortal_trained(), "lr": args.lr_mortal, "weight_decay": 0.01},
        ],
        lr=args.lr,
        weight_decay=1e-4,
        fused=device == "cuda",
    )
    if optimiser_state is not None:
        try:
            optimiser.load_state_dict(optimiser_state)
            for group, lr in zip(optimiser.param_groups, (args.lr, args.lr_ours, args.lr_mortal)):
                group["lr"] = lr
            print("restored AdamW state", flush=True)
        except (ValueError, RuntimeError) as error:
            print(f"could not restore AdamW state ({error}); starting it fresh", flush=True)

    if args.compile:
        # Deciding, with the batch's size left symbolic; the learning
        # forward compiles per mode, three graphs in all.
        net.backbones_forward = torch.compile(net.backbones, dynamic=True)
        learn = torch.compile(net.everything)
    else:
        learn = net.everything

    seated = []
    for path in args.opponents:
        if not Path(path).exists():
            print(f"no opponent at {path}, skipping", flush=True)
            continue
        other = zoo.load_player(path, device, compile=args.compile)
        other.eval()
        seated.append(other)
    if seated:
        print(f"{len(seated)} others seated in {args.opponent_share:.0%} of games", flush=True)
    print(
        f"device {device} | ours {net.ours.channels}x{net.ours.blocks} | Mortal beneath | "
        f"{net.parameter_count() / 1e6:.2f}M parameters | fixed by turns: {args.fixed}",
        flush=True,
    )
    drawer = np.random.default_rng(args.seed + 99)

    def checkpoint_payload(generation: int) -> dict:
        return {
            **net.state(),
            "learner": "combined",
            "generation": generation,
            "smoothed": smoothed,
            "best_placement": best_placement,
            "optimizer_state": optimiser.state_dict(),
        }

    end = start + args.rounds if args.rounds else args.generations
    for generation in range(start, end):
        began = time.time()
        batch = observations = on_card = None
        # Deciding is the joined player as it stands; what stays fixed in
        # the update that follows is drawn now and said in the record.
        fixed = str(drawer.choice(args.fixed))
        net.set_mode(fixed)
        net.eval()
        batch = selfplay.play(
            net,
            games=args.games,
            seed=args.seed + generation * 1000,
            device=device,
            amp=amp_enabled,
            opponents=seated,
            opponent_share=args.opponent_share,
        )
        played = time.time() - began
        observations = batch.observations
        legal = batch.legal.to(device)
        actions = batch.actions.to(device)
        returns = batch.returns.to(device)
        old_log_probs = batch.log_probs.to(device)
        on_card = resident(observations, device)
        loaded = time.time() - began - played

        # The baseline: our value head as it stands before the round.
        guess = torch.empty(batch.decisions, device=device)
        with torch.no_grad():
            for start_index in range(0, batch.decisions, 4096):
                chunk = slice(start_index, start_index + 4096)
                if on_card is not None:
                    planes = on_card.slice(start_index, start_index + 4096)
                else:
                    planes = observations.slice(start_index, start_index + 4096).dense(device)
                # The last chunk padded to the others' size, so the compiled
                # graph sees one shape all round.
                rows = planes.shape[0]
                planes = pad_rows(planes, 4096)
                mask = pad_rows(legal[chunk], 4096, True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    _logits, value, _guessed = learn(planes, mask)
                guess[chunk] = value.float()[:rows]
        value_error = float(((returns - guess) ** 2).mean())
        advantages = returns - guess
        advantage_spread = float(advantages.std())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        net.train()
        zero = lambda: torch.zeros((), device=device)
        total_policy, total_value, total_entropy = zero(), zero(), zero()
        total_clipped, total_kl, total_grad = zero(), zero(), zero()
        steps = 0
        for _epoch in range(args.epochs):
            order = torch.randperm(batch.decisions)
            slices = [
                order[start_index : start_index + args.batch]
                for start_index in range(0, batch.decisions, args.batch)
            ]
            # Whole minibatches only, so the compiled step sees one shape.
            slices = [drawn for drawn in slices if drawn.numel() == args.batch]

            def prepare(drawn: torch.Tensor):
                return drawn.to(device), observations.rows(drawn.numpy()).dense(device)

            def gather_on_card(drawn: torch.Tensor):
                picks = drawn.to(device)
                return picks, on_card.rows(picks)

            minibatches = (
                (gather_on_card(drawn) for drawn in slices)
                if on_card is not None
                else Prefetcher(slices, prepare)
            )
            for picks, planes in minibatches:
                optimiser.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    logits, value, _guessed = learn(planes, legal[picks])
                logits = logits.float()
                value = value.float()
                distribution = torch.distributions.Categorical(logits=logits)
                log_prob = distribution.log_prob(actions[picks])
                advantage = advantages[picks]
                ratio = torch.exp(log_prob - old_log_probs[picks])
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
                value_loss = nn.functional.mse_loss(value, returns[picks])
                entropy = distribution.entropy().mean()
                loss = policy_loss + args.value_weight * value_loss - args.entropy * entropy
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    [p for p in net.parameters() if p.requires_grad], 1.0
                )
                optimiser.step()
                with torch.no_grad():
                    total_policy += policy_loss
                    total_value += value_loss
                    total_entropy += entropy
                    total_clipped += (ratio != clipped).float().mean()
                    total_kl += (old_log_probs[picks] - log_prob).mean()
                    total_grad += grad_norm
                steps += 1

        # How much the head adds to our logits, on the last minibatch: the
        # mean over legal moves of what it changed, and the weight of the
        # straight road to Mortal's values.
        head_shift = 0.0
        with torch.no_grad():
            net.eval()
            parts = net.backbones(planes, legal[picks])
            phi, q, q_mask, pooled, features, a1, _value, _guessed = parts
            joined = net.fuse(
                phi.float(), q.float(), q_mask, pooled.float(), features.float(), a1.float(), legal[picks]
            )
            allowed = legal[picks]
            shift = (joined - a1).abs().masked_fill(~allowed, 0.0)
            head_shift = float(shift.sum() / allowed.sum().clamp(min=1))
            net.train()

        denom = max(steps, 1)
        record = {
            "generation": generation,
            "fixed": fixed,
            "head_shift": round(head_shift, 4),
            "mix": round(float(net.fuse.mix), 4),
            "decisions": batch.decisions,
            "hands": batch.hands,
            "seconds": round(time.time() - began, 1),
            "play_seconds": round(played, 1),
            "play_split": {name: round(value, 1) for name, value in batch.timing.items()},
            "load_seconds": round(loaded, 1),
            "resident": on_card is not None,
            "policy_loss": round(float(total_policy / denom), 4),
            "value_loss": round(float(total_value / denom), 4),
            "value_error": round(value_error, 4),
            "return_variance": round(float(returns.var()), 4),
            "advantage_spread": round(advantage_spread, 4),
            "entropy": round(float(total_entropy / denom), 4),
            "clipped": round(float(total_clipped / denom), 3),
            "approx_kl": round(float(total_kl / denom), 5),
            "grad_norm": round(float(total_grad / denom), 3),
            "mean_return": round(float(returns.mean()), 4),
        }

        measured = None
        is_best = False
        if (generation + 1) % args.measure_every == 0 or generation == 0:
            net.eval()
            measured = selfplay.measure(
                net, games=args.measure_games, seed=7_000_000 + generation, device=device
            )
            record.update(
                {
                    "placement": round(measured["placement"], 3),
                    "score": round(measured["score"], 1),
                    "win_rate": round(measured["wins"], 3),
                }
            )
            smoothed = (
                measured["placement"]
                if smoothed is None
                else SMOOTHING * measured["placement"] + (1 - SMOOTHING) * smoothed
            )
            record["smoothed"] = round(smoothed, 3)
            if smoothed < best_placement:
                best_placement = smoothed
                is_best = True
                record["best"] = True

        payload = checkpoint_payload(generation + 1)
        if measured is not None:
            payload["placement"] = measured["placement"]
        if is_best:
            torch.save(payload, args.out / "best.pt")
        torch.save(payload, args.out / "latest.pt")
        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    torch.save(checkpoint_payload(max(end, start)), args.out / "latest.pt")
    print("training finished", flush=True)


if __name__ == "__main__":
    main()
