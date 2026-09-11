"""Fine-tuning a published Mortal on our rules, by self-play.

The same loop as `train.py`, for a learner of the zoo's kind (see
`mortal_learner.py`): play a round with the learner at every seat, credit
each decision with what its hand moved and where its game placed, and
take a clipped policy-gradient step towards the decisions that did better
than the learner's own value head expected. Mortal's Q values are the
policy's logits and everything of Mortal's learns; there are no auxiliary
heads and no replay ring, since it has none of the heads those serve.

    python -m neural.train_mortal --mortal mortal.pth --rounds 30 --out runs/mortal
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import atomic_save
from .training_batches import validate_learning, require_trainable_round, require_updates
from torch import nn

from . import mortal_learner, selfplay, zoo
from .observe import pad_rows, resident
from .prefetch import Prefetcher
from .training_state import capture_sampling_state, restore_sampling_state

SMOOTHING = 1 / 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mortal", type=Path, default=None, help="the published Mortal to start from")
    parser.add_argument("--resume", type=Path, default=None, help="a checkpoint this trainer wrote")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=0, help="how many more generations, from where it resumes")
    parser.add_argument("--games", type=int, default=128, help="tables per round")
    # Mortal's own fine-tuning rate; a policy this good is moved gently.
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--entropy", type=float, default=0.005)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Mortal's Q values are divided by this to make the policy's logits",
    )
    parser.add_argument("--opponents", type=Path, nargs="*", default=[])
    parser.add_argument("--opponent-share", type=float, default=0.0)
    parser.add_argument("--measure-every", type=int, default=5)
    parser.add_argument("--measure-games", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--out", type=Path, default=Path("E:/tmp-claude/mahjong/mortal-run"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile Mortal's forward: its forty blocks' batch-norm, Mish and "
        "attention run as a dozen kernels each in eager mode, and a step of "
        "learning was bound by them rather than by the data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_learning(args.batch, args.epochs)
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
    sampling_state = None
    if args.resume is not None and args.resume.exists():
        net, payload = mortal_learner.load(args.resume, device, args.temperature)
        start = int(payload.get("generation", 0))
        smoothed = payload.get("smoothed")
        best_placement = float(payload.get("best_placement", float("inf")))
        optimiser_state = payload.get("optimizer_state")
        sampling_state = payload.get("sampling_state")
        print(f"resumed from {args.resume} at generation {start}", flush=True)
    elif args.mortal is not None and args.mortal.exists():
        net = mortal_learner.from_mortal(args.mortal, device, args.temperature)
        print(f"starting from {args.mortal}", flush=True)
    else:
        raise SystemExit("give --mortal, a published Mortal, or --resume, a checkpoint of this trainer's")
    source_state = torch.load(args.resume or args.mortal, map_location="cpu", weights_only=False)
    config = source_state["config"]

    optimiser = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=0.01, fused=device == "cuda"
    )
    if optimiser_state is not None:
        try:
            optimiser.load_state_dict(optimiser_state)
            for group in optimiser.param_groups:
                group["lr"] = args.lr
            print("restored AdamW state", flush=True)
        except (ValueError, RuntimeError) as error:
            print(f"could not restore AdamW state ({error}); starting it fresh", flush=True)

    # The learning step's forward with the minibatch's fixed shape, and
    # the deciding forward with the batch's size left symbolic, since it
    # changes every step.
    learn = torch.compile(net.policy) if args.compile else net.policy
    if args.compile:
        net.inference = torch.compile(net.policy, dynamic=True)

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
        f"device {device} | Mortal {config['resnet']['conv_channels']}x{config['resnet']['num_blocks']} "
        f"| {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters "
        f"| temperature {args.temperature}",
        flush=True,
    )

    restore_sampling_state(sampling_state, generation=start)

    def checkpoint_payload(generation: int) -> dict:
        return {
            **net.state(),
            "config": config,
            "learner": "mortal",
            "generation": generation,
            "smoothed": smoothed,
            "best_placement": best_placement,
            "temperature": args.temperature,
            "optimizer_state": optimiser.state_dict(),
            "sampling_state": capture_sampling_state(),
        }

    end = start + args.rounds if args.rounds else args.generations
    for generation in range(start, end):
        began = time.time()
        # Let go of the last round, on the host and on the card, before
        # the next is played.
        batch = observations = on_card = None
        batch = selfplay.play(
            net,
            games=args.games,
            seed=args.seed + generation * 1000,
            device=device,
            amp=amp_enabled,
            opponents=seated,
            opponent_share=args.opponent_share,
        )
        require_trainable_round(batch.decisions, args.batch, args.epochs)
        played = time.time() - began
        observations = batch.observations
        legal = batch.legal.to(device)
        actions = batch.actions.to(device)
        returns = batch.returns.to(device)
        old_log_probs = batch.log_probs.to(device)
        # The round's planes on the card whole when it has room, gathered
        # there; otherwise on the host, gathered a few steps ahead.
        on_card = resident(observations, device)
        loaded = time.time() - began - played

        # The baseline: the value head as it stands before the round.
        net.eval()
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
                    _logits, value = learn(planes, mask)
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
                # The gather on the host runs a few minibatches ahead of
                # the step on the card, in a thread of its own.
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
                    logits, value = learn(planes, legal[picks])
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
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimiser.step()
                with torch.no_grad():
                    total_policy += policy_loss
                    total_value += value_loss
                    total_entropy += entropy
                    total_clipped += (ratio != clipped).float().mean()
                    total_kl += (old_log_probs[picks] - log_prob).mean()
                    total_grad += grad_norm
                steps += 1

        require_updates(steps)
        denom = steps
        record = {
            "generation": generation,
            "checkpoint_generation": generation + 1,
            "optimizer_updates": steps,
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
            atomic_save(payload, args.out / "best.pt")
        atomic_save(payload, args.out / "latest.pt")
        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    atomic_save(checkpoint_payload(max(end, start)), args.out / "latest.pt")
    print("training finished", flush=True)


if __name__ == "__main__":
    main()
