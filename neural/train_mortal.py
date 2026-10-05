"""Fine-tuning a published Mortal on our rules, by self-play.

The same loop as `train.py`, for a learner of the zoo's kind (see
`mortal_learner.py`): play a round with the learner at every seat that
none of `--opponents` holds, credit each of its decisions with what its
hand moved and where its game placed, and take a clipped policy-gradient
step towards the decisions that did better than the learner's own value
head expected. Mortal's Q values are the policy's logits and everything
of Mortal's learns by the policy gradient; the value head reads its
encoder without training it. There are no auxiliary heads and no replay
ring, since it has none of the heads those serve.

    python -m neural.train_mortal --mortal mortal.pth --rounds 30 --out runs/mortal
"""

from __future__ import annotations

import argparse
from contextlib import closing
import time
from pathlib import Path

import torch
from torch import nn

from . import mortal_learner, ppo_loop, selfplay
from .observe import pad_rows
from .ppo_control import PolicyDrift, add_training_controls, baseline_batch_size
from .training_batches import require_trainable_round, require_updates
from .training_safety import TRAINING_API_VERSION, benchmark_history
from .training_state import (
    capture_random_state, peak_gpu_gb, peak_rss_gb, restore_random_state, round_seed,
)


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
        default=None,
        help="Mortal's Q values are divided by this to make the policy's logits. A resumed "
        "run keeps the one its checkpoint was trained at unless another is given, which "
        "is said when it differs; a run from a published Mortal starts at 1.0",
    )
    parser.add_argument("--opponents", type=Path, nargs="*", default=[])
    parser.add_argument("--opponent-share", type=float, default=0.0)
    parser.add_argument(
        "--seat-share", type=float, default=0.0,
        help="each player of each game is one of --opponents with this chance, so one table "
        "can hold several of them and the learner at once; instead of --opponent-share",
    )
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
    add_training_controls(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, amp_enabled, log_path = ppo_loop.setup(args)

    start = 0
    benchmark = ppo_loop.Benchmark()
    optimiser_state = None
    sampling_state = None
    if args.resume is not None and args.resume.exists():
        # At the temperature it was trained at: the flag used to default
        # to 1.0, so a run trained at another and resumed without it had
        # every logit changed and nothing said so.
        net, payload = mortal_learner.load(args.resume, device)
        if args.temperature is not None and args.temperature != net.temperature:
            print(
                f"temperature {args.temperature} as asked; {args.resume} was trained at "
                f"{net.temperature}, so every logit changes from this generation on",
                flush=True,
            )
            net.temperature = args.temperature
        start = int(payload.get("generation", 0))
        benchmark = ppo_loop.Benchmark(*benchmark_history(payload))
        optimiser_state = payload.get("optimizer_state")
        sampling_state = payload.get("sampling_state")
        # Mortal's config, which every checkpoint of this trainer carries
        # so that it loads wherever a published Mortal does.
        config = payload["config"]
        print(f"resumed from {args.resume} at generation {start}", flush=True)
    elif args.mortal is not None and args.mortal.exists():
        temperature = 1.0 if args.temperature is None else args.temperature
        net = mortal_learner.from_mortal(args.mortal, device, temperature)
        # Read once more for its config, and let go.
        config = torch.load(args.mortal, map_location="cpu", weights_only=False)["config"]
        print(f"starting from {args.mortal}", flush=True)
    else:
        raise SystemExit("give --mortal, a published Mortal, or --resume, a checkpoint of this trainer's")

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

    # Who else sits at the tables, as a roster with roles and shares, made
    # the way train_combined makes it.
    roster, seated = ppo_loop.seat_others(args, device)
    print(
        f"device {device} | Mortal {config['resnet']['conv_channels']}x{config['resnet']['num_blocks']} "
        f"| {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters "
        f"| temperature {net.temperature}",
        flush=True,
    )

    restore_random_state(sampling_state, generation=start)

    def checkpoint_payload(generation: int) -> dict:
        return {
            **net.state(),
            "config": config,
            "learner": "mortal",
            "generation": generation,
            "training_api_version": TRAINING_API_VERSION,
            "training_controls": {"target_kl": args.target_kl, "baseline_batch": args.baseline_batch,
                                  "seat_share": args.seat_share},
            "smoothed": benchmark.smoothed,
            "best_placement": benchmark.best,
            "temperature": net.temperature,
            "optimizer_state": optimiser.state_dict(),
            "sampling_state": capture_random_state(),
        }

    def measure(generation: int) -> dict:
        net.eval()
        return selfplay.measure(
            net, games=args.measure_games, seed=7_000_000 + generation, device=device
        )

    end = start + args.rounds if args.rounds else args.generations
    for generation in range(start, end):
        began = time.time()
        # Let go of the last round, on the host and on the card, before
        # the next is played.
        batch = rollout = None
        batch = selfplay.play(
            net,
            games=args.games,
            seed=round_seed(args.seed, generation, args.games),
            device=device,
            amp=amp_enabled,
            opponents=seated,
            opponent_share=args.opponent_share,
            seat_share=args.seat_share,
            population=roster,
            # Mortal has no head that reads the opponents' hands, and they
            # are a gigabyte of host memory on a large round.
            want_held=False,
        )
        require_trainable_round(batch.decisions, args.batch, args.epochs)
        played = time.time() - began
        rollout = ppo_loop.on_device(batch, device)
        loaded = time.time() - began - played

        # The baseline: the value head as it stands before the round.
        net.eval()
        rows = baseline_batch_size(batch.decisions, args.batch, args.baseline_batch)

        def values_of(chunk, planes):
            mask = pad_rows(rollout.legal[chunk], rows, True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                return (learn(planes, mask)[1],)

        valued = time.time()
        (guess,) = ppo_loop.baseline(rollout, rows, values_of)
        if device == "cuda":
            # Waited for, so that the time is the card's and not only the
            # launching of its work.
            torch.cuda.synchronize()
        valued = time.time() - valued
        value_error = float(((rollout.returns - guess) ** 2).mean())
        advantages, spread = ppo_loop.standardised(rollout.returns, guess)

        net.train()
        totals = ppo_loop.Totals(device)
        drift = PolicyDrift(args.target_kl)
        steps = 0
        for _epoch in range(args.epochs):
            with closing(ppo_loop.minibatches(rollout, args.batch)) as minibatches:
                for picks, planes in minibatches:
                    optimiser.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                        logits, value = learn(planes, rollout.legal[picks])
                    logits = logits.float()
                    value = value.float()
                    distribution = torch.distributions.Categorical(logits=logits)
                    log_prob = distribution.log_prob(rollout.actions[picks])
                    old_log_prob = rollout.old_log_probs[picks]
                    if drift.check(old_log_prob, log_prob):
                        break
                    policy_loss, ratio, clipped = ppo_loop.clipped_policy_loss(
                        log_prob, old_log_prob, advantages[picks], args.clip
                    )
                    value_loss = nn.functional.mse_loss(value, rollout.returns[picks])
                    entropy = distribution.entropy().mean()
                    loss = policy_loss + args.value_weight * value_loss - args.entropy * entropy
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
                    optimiser.step()
                    with torch.no_grad():
                        totals.add(
                            policy=policy_loss,
                            value=value_loss,
                            entropy=entropy,
                            clipped=(ratio != clipped).float().mean(),
                            kl=(old_log_prob - log_prob).mean(),
                            grad=grad_norm,
                        )
                    steps += 1
            if drift.stopped:
                break

        require_updates(steps)
        entry = ppo_loop.record(
            generation, steps, drift, rows, rollout, batch,
            {"began": began, "played": played, "loaded": loaded}, totals, spread,
            value_error=round(value_error, 4),
            # The pass that values the round before it is learned.
            baseline_seconds=round(valued, 1),
            # How the learner placed against each player it met, one row a
            # player, never summed: gaining on its own past while losing
            # to published Mortal is specialisation, and an average hides it.
            matchups=getattr(batch, "matchups", None),
            peak_rss_gb=peak_rss_gb(),
            peak_gpu_gb=peak_gpu_gb(),
        )
        ppo_loop.finish(args, generation, entry, measure, checkpoint_payload, benchmark, log_path)

    print("training finished", flush=True)


if __name__ == "__main__":
    main()
