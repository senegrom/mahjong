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
from contextlib import closing
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import riichi_py

from . import ppo_loop, selfplay
from .model import DEFAULT_BLOCKS, DEFAULT_CHANNELS, PolicyValueNet, load_weights, shape_of
from .observe import pad_rows
from .ppo_control import PolicyDrift, add_training_controls, baseline_batch_size
from .prefetch import Prefetcher
from .replay import Ring
from .training_batches import require_trainable_round, require_updates
from .training_safety import TRAINING_API_VERSION, benchmark_history
from .training_state import round_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="how many generations to train from where the run resumes, "
        "instead of running to --generations; for a short probe",
    )
    parser.add_argument("--games", type=int, default=128, help="tables per round")
    # 320 channels by 24 blocks with channel attention: about 15M parameters
    # in the policy tower. AlphaZero's twenty blocks of 256 came to roughly
    # 23M parameters on a Go board; on a line of thirty-four tiles the same
    # shape is a third of that, because the kernel is three rather than
    # three by three. The lineage before this one was twenty blocks over the
    # engine's ninety-seven planes; this one sees Mortal's thousand and
    # twelve, and four more blocks are the little more capacity that many
    # more inputs are given. Far too big for a phone, which is something to
    # distil away later rather than a reason to train something smaller now.
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--clip", type=float, default=0.2, help="PPO ratio clip")
    parser.add_argument("--batch", type=int, default=4096, help="decisions per step")
    parser.add_argument("--epochs", type=int, default=3, help="passes over a round")
    parser.add_argument("--entropy", type=float, default=0.03)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument(
        "--replay-rounds",
        type=int,
        default=8,
        help="how many past rounds the ring on disk keeps for the value "
        "heads and the head that reads the table to train on",
    )
    parser.add_argument(
        "--replay-steps",
        type=int,
        default=180,
        help="minibatches drawn from the ring after each round's policy "
        "update, for those heads only; about one pass over a round",
    )
    parser.add_argument(
        "--freeze-aux",
        action="store_true",
        help="train only the policy tower and its heads, by the policy loss "
        "alone: no value, table reading, critic, oracle or replay pass. For "
        "finding whether the PPO step by itself flattens the policy",
    )
    parser.add_argument(
        "--freeze-policy",
        action="store_true",
        help="train only the value heads and the head that reads the table; "
        "the policy tower and its heads keep their weights. For a night when "
        "the evaluator is worth improving and the policy is not to be "
        "trusted to improve itself",
    )
    parser.add_argument(
        "--distil-weight",
        type=float,
        default=1.0,
        help="how hard the public value head is pulled towards the oracle "
        "critic's estimate, on top of the return itself",
    )
    parser.add_argument(
        "--hands-weight",
        type=float,
        default=1.0,
        help="how much to weigh reading the opponents' hands, which is a "
        "free and dense label where the game's result is neither",
    )
    parser.add_argument(
        "--opponents",
        type=Path,
        nargs="*",
        default=[],
        help="older checkpoints to seat in a share of the self-play games. "
        "Four copies of one network playing only each other are never shown "
        "a position their own policy would not have created, and this run "
        "was measured getting worse against outside policies while getting "
        "better against fixed weak ones",
    )
    parser.add_argument(
        "--opponent-share",
        type=float,
        default=0.0,
        help="the share of games in which one seat is played by one of "
        "them, drawn at random. Nothing that seat does is recorded",
    )
    parser.add_argument("--measure-every", type=int, default=10)
    # Placement over sixty-four games wanders by about as much as the
    # improvements worth noticing, so the benchmark is wider.
    parser.add_argument("--measure-games", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out", type=Path, default=Path("E:/tmp-claude/mahjong/run1"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="run the learning step's forward pass in bfloat16; the losses "
        "and the optimiser stay in float32",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile the learning step's forward pass with torch.compile, "
        "which fuses its kernels and matters most when the launching "
        "thread is starved",
    )
    add_training_controls(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, amp_enabled, log_path = ppo_loop.setup(args)

    start = 0
    resume_payload = None
    benchmark = ppo_loop.Benchmark()
    if args.resume and args.resume.exists():
        # Keep the checkpoint on the host while its weights are copied into
        # the card. New checkpoints also carry Adam's moments, which are
        # larger than the model itself and should not transiently occupy the
        # card twice while loading.
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        # The checkpoint says what shape it is; the flags stand in only
        # where it is silent.
        shape = shape_of(resume_payload, args.channels, args.blocks)
        net = PolicyValueNet(**shape).to(device)
        load_weights(net, resume_payload["model"])
        start = int(resume_payload.get("generation", 0))
        benchmark = ppo_loop.Benchmark(*benchmark_history(resume_payload))
        if benchmark.smoothed is not None:
            print(
                f"carrying a smoothed placement of {benchmark.smoothed:.3f}, "
                f"best {benchmark.best:.3f}",
                flush=True,
            )
        print(f"resumed from {args.resume} at generation {start}", flush=True)
    else:
        net = PolicyValueNet(args.channels, args.blocks).to(device)
    if getattr(net, "actions", riichi_py.ACTIONS) != riichi_py.ACTIONS:
        raise SystemExit("neural.train requires a 78-action learner; use the Mortal-space trainer")
    if net.kind != "mortal":
        raise SystemExit("training needs a network that sees Mortal's planes")

    if args.freeze_policy:
        for name, parameter in net.named_parameters():
            if not name.startswith(("critic", "oracle_", "belief_", "hands.", "value.")):
                parameter.requires_grad_(False)
        print("policy frozen: training the value heads and the reading of the table", flush=True)
    if args.freeze_aux:
        for name, parameter in net.named_parameters():
            if name.startswith(("critic", "oracle_", "belief_", "hands.", "value.")):
                parameter.requires_grad_(False)
        args.value_weight = 0.0
        args.hands_weight = 0.0
        args.replay_steps = 0
        print("auxiliaries frozen: training the policy by the policy loss alone", flush=True)
    trainable = [parameter for parameter in net.parameters() if parameter.requires_grad]
    # PyTorch's fused CUDA AdamW performs the same update in far fewer kernel
    # launches. On CPU the ordinary implementation remains the right path.
    optimiser = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=1e-4,
        fused=device == "cuda",
    )
    if resume_payload is not None and "optimizer" in resume_payload:
        try:
            optimiser.load_state_dict(resume_payload["optimizer"])
            # A checkpoint carries the moments, not command-line policy.
            # Explicit options on a resumed run still win.
            for group in optimiser.param_groups:
                group["lr"] = args.lr
                group["weight_decay"] = 1e-4
                group["fused"] = device == "cuda"
            print("restored AdamW state", flush=True)
        except (ValueError, RuntimeError) as error:
            # Freezing a different set of heads changes the parameter groups;
            # weights still resume safely, but those experiments need fresh
            # optimiser state.
            print(f"could not restore AdamW state ({error}); starting it fresh", flush=True)

    # The full on-policy pass needs every head. Baseline evaluation and replay
    # do not: using `with_oracle` for them calculated policy logits and/or the
    # belief head across hundreds of thousands of rows and then threw those
    # tensors away. Keep purpose-built paths here so the model definition
    # stays about inference semantics rather than one trainer's scheduling.
    def value_heads(planes: torch.Tensor, seen: torch.Tensor):
        features = net.tail(net.tower(net.stem(planes)))
        pooled = features.mean(dim=2)
        public = net.value(pooled).squeeze(1)
        hidden = net.oracle_tail(
            net.oracle_tower(net.oracle_stem(torch.cat([planes, seen], dim=1)))
        ).mean(dim=2)
        oracle_value = net.oracle_value(torch.cat([pooled, hidden], dim=1)).squeeze(1)
        criticised = net.critic_value(planes, pooled)
        return public, oracle_value, criticised

    def auxiliary_heads(planes: torch.Tensor, seen: torch.Tensor):
        # Replay never trains the policy tower. Avoid constructing an autograd
        # graph for it merely to hand detached features to the auxiliary
        # towers.
        with torch.no_grad():
            features = net.tail(net.tower(net.stem(planes)))
            pooled = features.mean(dim=2)
        guessed = net.hands_from(planes, features)
        hidden = net.oracle_tail(
            net.oracle_tower(net.oracle_stem(torch.cat([planes, seen], dim=1)))
        ).mean(dim=2)
        oracle_value = net.oracle_value(torch.cat([pooled.detach(), hidden], dim=1)).squeeze(1)
        criticised = net.critic_value(planes, pooled)
        return guessed, oracle_value, criticised

    learn = torch.compile(net.with_oracle) if args.compile else net.with_oracle
    values = torch.compile(value_heads) if args.compile else value_heads
    auxiliary = torch.compile(auxiliary_heads) if args.compile else auxiliary_heads
    if args.compile:
        # The deciding forward too, its batch's size left symbolic since
        # it changes every step: the norms, activations and attention
        # gates fuse, where eager mode ran each as its own kernel.
        net.everything = torch.compile(net.everything, dynamic=True)

    # The last several rounds, on disk, for the heads that may learn from
    # stale play: see `replay.py`. The policy never trains on it.
    ring = Ring(args.out / "ring", args.replay_rounds)

    # The older selves that share the table, drawn evenly in a share of
    # the games; no roster weighs them here.
    seated = ppo_loop.load_others(args, device)
    if seated:
        print(
            f"{len(seated)} older checkpoints seated in {args.opponent_share:.0%} "
            "of games",
            flush=True,
        )
    replay_rng = np.random.default_rng(args.seed + 17)

    # Restore stochastic state only after constructing every network: module
    # initialisation consumes torch RNG even when its random weights are
    # immediately replaced by a checkpoint. Doing this last makes a process
    # restart continue the same sampling stream as an uninterrupted run.
    if resume_payload is not None:
        if "torch_rng_state" in resume_payload:
            torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
        if device == "cuda" and "cuda_rng_state_all" in resume_payload:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_payload["cuda_rng_state_all"]]
            )
        if "replay_rng_state" in resume_payload:
            replay_rng.bit_generator.state = resume_payload["replay_rng_state"]

    def checkpoint_payload(generation: int) -> dict:
        # Adam's moments are deliberately part of the checkpoint: restarting
        # them every Modal block used to turn every resume into a different
        # optimiser.
        payload = {
            "model": net.state_dict(),
            "optimizer": optimiser.state_dict(),
            "generation": generation,
            "training_api_version": TRAINING_API_VERSION,
            "training_controls": {"target_kl": args.target_kl, "baseline_batch": args.baseline_batch},
            **net.payload_fields(),
            "smoothed": benchmark.smoothed,
            "best_placement": benchmark.best,
            "torch_rng_state": torch.get_rng_state(),
            "replay_rng_state": replay_rng.bit_generator.state,
        }
        if device == "cuda":
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return payload

    def measure(generation: int) -> dict:
        return selfplay.measure(
            net,
            games=args.measure_games,
            seed=7_000_000 + generation,
            device=device,
            amp=args.amp,
        )

    print(
        f"device {device} | {net.channels} channels x {net.blocks} blocks "
        f"| {net.planes} planes | {net.parameter_count() / 1e6:.2f}M parameters",
        flush=True,
    )

    # A round of planes is a few gigabytes on the host, kept sparse. The
    # names below are cleared before the next round is played, so the
    # machine carries one round rather than two.
    batch = rollout = held = oracle = None
    end = start + args.rounds if args.rounds else args.generations
    for generation in range(start, end):
        began = time.time()
        batch = rollout = held = oracle = None
        batch = selfplay.play(
            net,
            games=args.games,
            seed=round_seed(args.seed, generation, args.games),
            device=device,
            amp=args.amp,
            opponents=seated,
            opponent_share=args.opponent_share,
        )
        require_trainable_round(batch.decisions, args.batch, args.epochs)
        played = time.time() - began
        # Each phase of the learning half is timed and said in the record:
        # a generation stalled by a near-constant five minutes now and
        # then, and the record could not say where.
        phase = time.time()
        ring.push(batch)
        ring_seconds = time.time() - phase

        rollout = ppo_loop.on_device(batch, device)
        held = batch.held.to(device)
        # The oracle's planes stay in bytes. When the round's planes went to
        # the card whole, these go too, so a step touches the host for
        # nothing.
        oracle = batch.oracle
        if rollout.on_card is not None:
            oracle = oracle.to(device)
        loaded = time.time() - began - played

        # The value head predicts the return in the reward's own units: the
        # points a hand moved over four thousand, plus the place bonus. It
        # used to predict the return standardised by each round's mean and
        # spread, which made a good training target and a useless number,
        # since nothing outside the round could say what a value of 0.7
        # meant. The rewards are already of order one, so nothing is lost
        # by leaving them alone; the advantage below is centred by the value
        # head itself.
        normalised = rollout.returns

        # The baseline for the policy gradient is whichever pre-update value
        # head predicts this fresh round best.
        net.eval()
        baseline_began = time.time()
        rows = baseline_batch_size(batch.decisions, args.batch, args.baseline_batch)

        def values_of(chunk, planes):
            seen = pad_rows(oracle[chunk].to(device).float(), rows)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                return values(planes, seen)

        public_guess, oracle_guess, critic_guess = ppo_loop.baseline(
            rollout, rows, values_of, heads=3
        )
        public_error = float(((normalised - public_guess) ** 2).mean())
        oracle_error = float(((normalised - oracle_guess) ** 2).mean())
        critic_error = float(((normalised - critic_guess) ** 2).mean())
        baseline_seconds = time.time() - baseline_began
        errors = {"public": public_error, "oracle": oracle_error, "critic": critic_error}
        chosen = min(errors, key=errors.get)
        baseline = {"public": public_guess, "oracle": oracle_guess, "critic": critic_guess}[chosen]
        oracle_better = chosen == "oracle"
        distil_weight = args.distil_weight if oracle_better else 0.0
        advantages, spread = ppo_loop.standardised(normalised, baseline)

        net.train()
        phase = time.time()
        totals = ppo_loop.Totals(
            device, "oracle", "distil", "critic", "hands", "covered",
            "sure", "likely", "unlikely", "sure_count", "likely_count", "unlikely_count",
        )
        drift = PolicyDrift(args.target_kl)
        steps = 0
        for _epoch in range(args.epochs):
            # Sparse on the host, dense float32 on the card: the tower's
            # weights are float32, and autocast takes it from there.
            with closing(ppo_loop.minibatches(
                rollout, args.batch, lambda index: (oracle[index].to(device).float(),)
            )) as minibatches:
                for picks, planes, seen in minibatches:
                    optimiser.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                        logits, value, guessed, oracle_value, criticised = learn(
                            planes, rollout.legal[picks], seen
                        )
                    # Whatever the forward pass ran in, the losses are float32.
                    logits = logits.float()
                    value = value.float()
                    guessed = guessed.float()
                    oracle_value = oracle_value.float()
                    criticised = criticised.float()
                    distribution = torch.distributions.Categorical(logits=logits)
                    log_prob = distribution.log_prob(rollout.actions[picks])
                    old_log_prob = rollout.old_log_probs[picks]
                    if drift.check(old_log_prob, log_prob):
                        break
                    advantage = advantages[picks]
                    # The mean advantage of the action taken, by how sure the
                    # policy was of it.
                    with torch.no_grad():
                        confidence = old_log_prob.exp()
                        sure = confidence > 0.5
                        likely = (confidence > 0.2) & ~sure
                        unlikely = confidence <= 0.2
                        totals.add(
                            sure=(advantage * sure).sum(), sure_count=sure.sum(),
                            likely=(advantage * likely).sum(), likely_count=likely.sum(),
                            unlikely=(advantage * unlikely).sum(), unlikely_count=unlikely.sum(),
                        )
                    policy_loss, ratio, clipped = ppo_loop.clipped_policy_loss(
                        log_prob, old_log_prob, advantage, args.clip
                    )
                    value_loss = nn.functional.mse_loss(value, normalised[picks])
                    oracle_loss = nn.functional.mse_loss(oracle_value, normalised[picks])
                    critic_loss = nn.functional.mse_loss(criticised, normalised[picks])
                    # The public head also learns from the oracle's estimate as
                    # it stood before the round, on rounds where that estimate
                    # was the better one; see the choice of baseline above.
                    distil_loss = nn.functional.mse_loss(value, oracle_guess[picks])
                    entropy = distribution.entropy().mean()

                    # What the opponents are holding.
                    hands_loss, covered = ppo_loop.hands_loss_of(guessed, held[picks])
                    loss = (
                        policy_loss
                        + args.value_weight * (value_loss + oracle_loss + critic_loss)
                        + args.value_weight * distil_weight * distil_loss
                        + args.hands_weight * hands_loss
                        - args.entropy * entropy
                    )

                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                    optimiser.step()

                    with torch.no_grad():
                        totals.add(
                            policy=policy_loss,
                            value=value_loss,
                            entropy=entropy,
                            # Standard PPO approximate KL. It does not alter
                            # the update, but makes a too-large policy step
                            # visible next to the clip fraction rather than
                            # only after an arena.
                            clipped=(ratio != clipped).float().mean(),
                            kl=(old_log_prob - log_prob).mean(),
                            grad=grad_norm,
                            oracle=oracle_loss,
                            critic=critic_loss,
                            distil=distil_loss,
                            hands=hands_loss,
                            covered=covered,
                        )
                    steps += 1
            if drift.stopped:
                break

        epochs_seconds = time.time() - phase
        phase = time.time()
        # The heads that may learn from stale play take a pass over the
        # ring: the value heads and the head that reads the table, on
        # minibatches drawn evenly from the last several rounds.
        # No policy term, and no distillation, since the oracle's pre-round
        # estimate exists only for the round just played.
        replay = ppo_loop.Totals(device, "critic", "oracle")
        replay_steps = 0
        if args.replay_steps and ring.total() >= args.batch:
            # Nothing in this pass reaches the policy tower: the critic, the
            # oracle and the head that reads the table each read it without
            # gradient. `auxiliary` also avoids calculating policy outputs
            # that this loss never reads.
            def prepare_replay(_step: int):
                # The ring's rows come off the disk's memory maps, which is
                # slower still than the host gather above; the same thread
                # runs ahead. The sampler is used from this thread alone,
                # in order, so its stream is what it always was.
                rows = ring.sample(args.batch, replay_rng)
                return (
                    rows["observations"].dense(device),
                    rows["oracle"].to(device).float(),
                    rows["returns"].to(device),
                    rows["held"].to(device),
                )

            with Prefetcher(range(args.replay_steps), prepare_replay) as minibatches:
                for planes, seen, target, held_rows in minibatches:
                    optimiser.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                        guessed, oracle_value, criticised = auxiliary(planes, seen)
                    guessed = guessed.float()
                    oracle_value = oracle_value.float()
                    criticised = criticised.float()
                    critic_loss = nn.functional.mse_loss(criticised, target)
                    oracle_loss = nn.functional.mse_loss(oracle_value, target)
                    hands_loss, _covered = ppo_loop.hands_loss_of(guessed, held_rows)
                    loss = (
                        args.value_weight * (critic_loss + oracle_loss)
                        + args.hands_weight * hands_loss
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                    optimiser.step()
                    with torch.no_grad():
                        replay.add(critic=critic_loss, oracle=oracle_loss)
                    replay_steps += 1
        replay_seconds = time.time() - phase

        require_updates(steps)
        confidence_total = (
            totals["sure_count"] + totals["likely_count"] + totals["unlikely_count"]
        ).clamp(min=1)
        entry = ppo_loop.record(
            generation, steps, drift, rows, rollout, batch,
            {"began": began, "played": played, "loaded": loaded}, totals, spread,
            baseline_seconds=round(baseline_seconds, 1),
            ring_seconds=round(ring_seconds, 1),
            epochs_seconds=round(epochs_seconds, 1),
            replay_seconds=round(replay_seconds, 1),
            oracle_loss=totals.mean("oracle", steps, 4),
            critic_loss=totals.mean("critic", steps, 4),
            # Each head's error on the round before it was trained on it,
            # against the return variance, and which was the baseline.
            public_error=round(public_error, 4),
            oracle_error=round(oracle_error, 4),
            critic_error=round(critic_error, 4),
            baseline=chosen,
            # Mean standardised advantage of the taken action by the policy's
            # confidence in it, and how much of the round each bin was.
            advantage_sure=round(float(totals["sure"] / totals["sure_count"].clamp(min=1)), 4),
            advantage_likely=round(float(totals["likely"] / totals["likely_count"].clamp(min=1)), 4),
            advantage_unlikely=round(
                float(totals["unlikely"] / totals["unlikely_count"].clamp(min=1)), 4
            ),
            share_sure=round(float(totals["sure_count"] / confidence_total), 3),
            share_likely=round(float(totals["likely_count"] / confidence_total), 3),
            share_unlikely=round(float(totals["unlikely_count"] / confidence_total), 3),
            # The same heads on the ring of past rounds, which is where
            # they must not learn a round by heart.
            replay_optimizer_updates=replay_steps,
            replay_rounds=len(ring),
            replay_critic_loss=replay.mean("critic", max(replay_steps, 1), 4),
            replay_oracle_loss=replay.mean("oracle", max(replay_steps, 1), 4),
            distil=totals.mean("distil", steps, 4),
            hands_loss=totals.mean("hands", steps, 4),
            hands_read=totals.mean("covered", steps, 4),
        )
        ppo_loop.finish(args, generation, entry, measure, checkpoint_payload, benchmark, log_path)

    print("training finished", flush=True)


if __name__ == "__main__":
    main()
