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

import numpy as np
import torch
from torch import nn

from . import selfplay, zoo
from .observe import resident
from .prefetch import Prefetcher
from .model import (
    DEFAULT_BLOCKS,
    DEFAULT_CHANNELS,
    HIDDEN_HANDS_PLANES,
    PolicyValueNet,
    load_weights,
    shape_of,
)
from .replay import Ring


# How much of a new measurement goes into the smoothed figure the best
# checkpoint is chosen on. A third means roughly the last three count, which
# cuts the noise about in half without lagging far behind a real gain.
SMOOTHING = 1 / 3


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
        "heads, the reader and the head that reads the table to train on",
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
        "alone: no value, table reading, critic, oracle, reader or replay "
        "pass. For finding whether the PPO step by itself flattens the "
        "policy",
    )
    parser.add_argument(
        "--freeze-policy",
        action="store_true",
        help="train only the critic, the oracle and the reader; the policy "
        "tower and its heads keep their weights. For a night when the "
        "evaluator is worth improving and the policy is not to be trusted "
        "to improve itself",
    )
    parser.add_argument(
        "--reader-weight",
        type=float,
        default=1.0,
        help="how hard the reader of hidden hands is trained to tell the "
        "real ones from the ones the proposal imagines",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The environment runs on this thread and the network on the GPU, so a
    # couple of worker threads is plenty and leaves the machine usable.
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = args.amp and device == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    # `--seed` used to seed only the rules engine and replay sampler. Model
    # initialisation, sampled policy moves and minibatch shuffles therefore
    # changed between nominally identical runs. Seed torch too; a resumed
    # run restores the exact RNG states saved in its checkpoint below.
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    start = 0
    resume_payload = None
    # Carried in the checkpoint, not reset per process: a run that resumes
    # has to judge a new measurement against what it has already reached,
    # or the first one after every restart becomes the new best whatever it
    # is. That replaced a 2.447 checkpoint with a 2.592 one, and on a
    # trainer that runs in blocks it would happen at every block.
    best_placement = float("inf")
    smoothed = None
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
        smoothed = resume_payload.get("smoothed")
        best_placement = float(resume_payload.get("best_placement", float("inf")))
        if smoothed is not None:
            print(
                f"carrying a smoothed placement of {smoothed:.3f}, "
                f"best {best_placement:.3f}",
                flush=True,
            )
        print(f"resumed from {args.resume} at generation {start}", flush=True)
    else:
        net = PolicyValueNet(args.channels, args.blocks).to(device)
    if net.kind != "mortal":
        raise SystemExit("training needs a network that sees Mortal's planes")

    if args.freeze_policy:
        for name, parameter in net.named_parameters():
            if not name.startswith(("critic", "oracle_", "reader", "belief_", "hands.", "value.")):
                parameter.requires_grad_(False)
        print("policy frozen: training the critic, the oracle and the reader", flush=True)
    if args.freeze_aux:
        for name, parameter in net.named_parameters():
            if name.startswith(("critic", "oracle_", "reader", "belief_", "hands.", "value.")):
                parameter.requires_grad_(False)
        args.value_weight = 0.0
        args.hands_weight = 0.0
        args.reader_weight = 0.0
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
    read = torch.compile(net.read_plausibility) if args.compile else net.read_plausibility
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

    # The older selves that share the table, loaded once, each at whatever
    # shape and of whatever kind its checkpoint says. They are only ever
    # asked for a move, so they need no optimiser and no gradients.
    seated = []
    for path in args.opponents:
        if not Path(path).exists():
            print(f"no opponent at {path}, skipping", flush=True)
            continue
        older = zoo.load_player(path, device, compile=args.compile)
        older.eval()
        for parameter in getattr(older, "parameters", list)():
            parameter.requires_grad_(False)
        seated.append(older)
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
        payload = {
            "model": net.state_dict(),
            "optimizer": optimiser.state_dict(),
            "generation": generation,
            **net.payload_fields(),
            "smoothed": smoothed,
            "best_placement": best_placement,
            "torch_rng_state": torch.get_rng_state(),
            "replay_rng_state": replay_rng.bit_generator.state,
        }
        if device == "cuda":
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return payload

    def hands_loss_of(guessed, wanted):
        """Cross-entropy against the distribution each opponent's hand
        actually was, over the 34 kinds, and how much of the hand the
        guess covers, which is readable where a cross-entropy is not.
        Positions where nobody was holding anything carry rows of zeros
        and are skipped."""
        holding = wanted.sum(dim=2) > 0
        log_guess = torch.log_softmax(guessed, dim=2)
        loss = -(wanted * log_guess).sum(dim=2)
        loss = (loss * holding).sum() / holding.sum().clamp(min=1)
        with torch.no_grad():
            overlap = torch.minimum(log_guess.exp(), wanted).sum(dim=2)
            covered = (overlap * holding).sum() / holding.sum().clamp(min=1)
        return loss, covered

    def reader_loss_of(planes, real, fake):
        """The reader, shown the position with the real hidden hands and
        with the hands the proposal imagined, learns to tell which is
        which; what it learns is the likelihood ratio a search weighs
        imagined worlds by. Returns the loss and how often it was right."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            verdict = read(torch.cat([planes, planes]), torch.cat([real, fake]))
        verdict = verdict.float()
        truth = torch.cat(
            [
                torch.ones(len(planes), device=device),
                torch.zeros(len(planes), device=device),
            ]
        )
        loss = nn.functional.binary_cross_entropy_with_logits(verdict, truth)
        with torch.no_grad():
            right = ((verdict > 0).float() == truth).float().mean()
        return loss, right

    print(
        f"device {device} | {net.channels} channels x {net.blocks} blocks "
        f"| {net.planes} planes | {net.parameter_count() / 1e6:.2f}M parameters",
        flush=True,
    )

    # A round of planes is a few gigabytes on the host, kept sparse. The
    # names below are cleared before the next round is played, so the
    # machine carries one round rather than two.
    batch = observations = oracle = imagined = on_card = None
    end = start + args.rounds if args.rounds else args.generations
    for generation in range(start, end):
        began = time.time()
        batch = observations = oracle = imagined = on_card = None
        batch = selfplay.play(
            net,
            games=args.games,
            seed=args.seed + generation * 1000,
            device=device,
            amp=args.amp,
            opponents=seated,
            opponent_share=args.opponent_share,
        )
        played = time.time() - began
        ring.push(batch)

        # The observations stay on the host, sparse, and each minibatch is
        # made dense on the card as it is drawn: a round of them dense would
        # be tens of gigabytes. A minibatch's entries are a few tens of
        # megabytes and cross in a few milliseconds. Not pinned: pinning
        # copies the round into page-locked memory, which once took the run
        # from twelve gigabytes of host memory to twenty-eight and the
        # machine to none.
        observations = batch.observations
        legal = batch.legal.to(device)
        actions = batch.actions.to(device)
        held = batch.held.to(device)
        # The oracle's planes likewise, in bytes, and the hands the proposal
        # imagined, the reader's negatives.
        oracle = batch.oracle
        imagined = batch.imagined
        returns = batch.returns.to(device)
        old_log_probs = batch.log_probs.to(device)
        # The round's planes go to the card whole when it has room, and
        # the minibatch is gathered there; otherwise they stay on the host
        # and are gathered a few steps ahead. On the card the small byte
        # planes go too, so a step touches the host for nothing.
        on_card = resident(observations, device)
        if on_card is not None:
            oracle = oracle.to(device)
            imagined = imagined.to(device)
        loaded = time.time() - began - played

        # The value head predicts the return in the reward's own units: the
        # points a hand moved over four thousand, plus the place bonus. It
        # used to predict the return standardised by each round's mean and
        # spread, which made a good training target and a useless number,
        # since nothing outside the round could say what a value of 0.7
        # meant. A search that evaluates positions with this head needs to
        # compare it with hands that actually ended, in points, so the units
        # have to be fixed. The rewards are already of order one, so nothing
        # is lost by leaving them alone; the advantage below is centred by
        # the value head itself.
        normalised = returns

        # The baseline for the policy gradient is whichever pre-update value
        # head predicts this fresh round best. It is computed once before any
        # head gets to learn the round, so the advantages cannot collapse as
        # the critic fits the very targets they are measured against.
        net.eval()
        public_guess = torch.empty(batch.decisions, device=device)
        oracle_guess = torch.empty(batch.decisions, device=device)
        critic_guess = torch.empty(batch.decisions, device=device)
        baseline_began = time.time()
        with torch.no_grad():
            for start_index in range(0, batch.decisions, 8192):
                chunk = slice(start_index, start_index + 8192)
                if on_card is not None:
                    planes = on_card.slice(start_index, start_index + 8192)
                else:
                    planes = observations.slice(start_index, start_index + 8192).dense(device)
                seen = oracle[chunk].to(device).float()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    guessed_value, judged, criticised = values(planes, seen)
                public_guess[chunk] = guessed_value.float()
                oracle_guess[chunk] = judged.float()
                critic_guess[chunk] = criticised.float()
        public_error = float(((normalised - public_guess) ** 2).mean())
        oracle_error = float(((normalised - oracle_guess) ** 2).mean())
        critic_error = float(((normalised - critic_guess) ** 2).mean())
        baseline_seconds = time.time() - baseline_began
        errors = {"public": public_error, "oracle": oracle_error, "critic": critic_error}
        chosen = min(errors, key=errors.get)
        baseline = {"public": public_guess, "oracle": oracle_guess, "critic": critic_guess}[chosen]
        oracle_better = chosen == "oracle"
        distil_weight = args.distil_weight if oracle_better else 0.0
        advantages = normalised - baseline
        advantage_mean = advantages.mean()
        advantage_std = advantages.std()
        advantage_spread = float(advantage_std)
        advantages = (advantages - advantage_mean) / (advantage_std + 1e-6)

        net.train()
        # Keep metrics on the card until the generation is over. The old loop
        # called `.item()` or `float()` around a dozen times per minibatch,
        # which synchronised the CPU with the GPU around a dozen times per
        # optimiser step merely to print one JSON line at the end.
        zero = lambda: torch.zeros((), device=device)
        total_policy = zero()
        total_value = zero()
        total_entropy = zero()
        total_oracle = zero()
        total_distil = zero()
        total_critic = zero()
        total_reader = zero()
        total_read_right = zero()
        total_clipped = zero()
        total_kl = zero()
        total_hands = zero()
        total_covered = zero()
        total_grad_norm = zero()
        sure_sum = zero()
        likely_sum = zero()
        unlikely_sum = zero()
        sure_count = zero()
        likely_count = zero()
        unlikely_count = zero()
        steps = 0
        for _epoch in range(args.epochs):
            # The large observations live on the host, so make the shuffle
            # there too. The old GPU permutation had to copy every minibatch
            # of indices back to the CPU before the observations could be
            # gathered, forcing one device synchronisation per step.
            order = torch.randperm(batch.decisions)
            slices = [
                order[start_index : start_index + args.batch]
                for start_index in range(0, batch.decisions, args.batch)
            ]
            slices = [drawn for drawn in slices if drawn.numel() >= 2]

            def prepare(drawn: torch.Tensor):
                # Sparse on the host, dense float32 on the card: the
                # tower's weights are float32, and autocast takes it from
                # there. Run a few minibatches ahead in a thread, so the
                # gather on the host overlaps the step on the card.
                return (
                    drawn.to(device),
                    observations.rows(drawn.numpy()).dense(device),
                    oracle[drawn].to(device).float(),
                    imagined[drawn].to(device).float(),
                )

            def gather_on_card(drawn: torch.Tensor):
                picks = drawn.to(device)
                return (
                    picks,
                    on_card.rows(picks),
                    oracle[picks].float(),
                    imagined[picks].float(),
                )

            minibatches = (
                (gather_on_card(drawn) for drawn in slices)
                if on_card is not None
                else Prefetcher(slices, prepare)
            )
            for picks, planes, seen, fake in minibatches:
                optimiser.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    logits, value, guessed, oracle_value, criticised = learn(
                        planes, legal[picks], seen
                    )
                # Whatever the forward pass ran in, the losses are float32.
                logits = logits.float()
                value = value.float()
                guessed = guessed.float()
                oracle_value = oracle_value.float()
                criticised = criticised.float()
                reader_loss, reader_right = reader_loss_of(
                    planes, seen[:, :HIDDEN_HANDS_PLANES], fake
                )
                distribution = torch.distributions.Categorical(logits=logits)
                log_prob = distribution.log_prob(actions[picks])
                advantage = advantages[picks]

                # The clipped objective: an update may improve an action's
                # odds, but only so far in one round, which is what keeps a
                # policy from narrowing onto a single action.
                with torch.no_grad():
                    confidence = old_log_probs[picks].exp()
                    sure = confidence > 0.5
                    likely = (confidence > 0.2) & ~sure
                    unlikely = confidence <= 0.2
                    sure_sum += (advantage * sure).sum()
                    sure_count += sure.sum()
                    likely_sum += (advantage * likely).sum()
                    likely_count += likely.sum()
                    unlikely_sum += (advantage * unlikely).sum()
                    unlikely_count += unlikely.sum()
                ratio = torch.exp(log_prob - old_log_probs[picks])
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
                with torch.no_grad():
                    total_clipped += (ratio != clipped).float().mean()
                    # Standard PPO approximate KL. It does not alter the
                    # update, but makes a too-large policy step visible next
                    # to the clip fraction rather than only after an arena.
                    total_kl += (old_log_probs[picks] - log_prob).mean()
                value_loss = nn.functional.mse_loss(value, normalised[picks])
                oracle_loss = nn.functional.mse_loss(oracle_value, normalised[picks])
                critic_loss = nn.functional.mse_loss(criticised, normalised[picks])
                # The public head also learns from the oracle's estimate as
                # it stood before the round, on rounds where that estimate
                # was the better one; see the choice of baseline above.
                distil_loss = nn.functional.mse_loss(value, oracle_guess[picks])
                entropy = distribution.entropy().mean()

                # What the opponents are holding.
                hands_loss, covered = hands_loss_of(guessed, held[picks])
                loss = (
                    policy_loss
                    + args.value_weight * (value_loss + oracle_loss + critic_loss)
                    + args.value_weight * distil_weight * distil_loss
                    + args.hands_weight * hands_loss
                    + args.reader_weight * reader_loss
                    - args.entropy * entropy
                )

                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(trainable, 1.0)
                optimiser.step()

                with torch.no_grad():
                    total_policy += policy_loss
                    total_value += value_loss
                    total_oracle += oracle_loss
                    total_critic += critic_loss
                    total_distil += distil_loss
                    total_reader += reader_loss
                    total_read_right += reader_right
                    total_entropy += entropy
                    total_hands += hands_loss
                    total_covered += covered
                    total_grad_norm += grad_norm
                steps += 1

        # The heads that may learn from stale play take a pass over the
        # ring: the value heads, the reader and the head that reads the
        # table, on minibatches drawn evenly from the last several rounds.
        # No policy term, and no distillation, since the oracle's pre-round
        # estimate exists only for the round just played.
        replay_critic = zero()
        replay_oracle = zero()
        replay_read_right = zero()
        replay_steps = 0
        if args.replay_steps and ring.total() >= args.batch:
            # Nothing in this pass reaches the policy tower: the critic, the
            # oracle and the reader each read it without gradient or not at
            # all. `auxiliary` also avoids calculating policy outputs that
            # this loss never reads.
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
                    rows["imagined"].to(device).float(),
                )

            for planes, seen, target, held_rows, fake in Prefetcher(
                range(args.replay_steps), prepare_replay
            ):
                optimiser.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    guessed, oracle_value, criticised = auxiliary(planes, seen)
                guessed = guessed.float()
                oracle_value = oracle_value.float()
                criticised = criticised.float()
                critic_loss = nn.functional.mse_loss(criticised, target)
                oracle_loss = nn.functional.mse_loss(oracle_value, target)
                hands_loss, _covered = hands_loss_of(guessed, held_rows)
                reader_loss, reader_right = reader_loss_of(
                    planes, seen[:, :HIDDEN_HANDS_PLANES], fake
                )
                loss = (
                    args.value_weight * (critic_loss + oracle_loss)
                    + args.hands_weight * hands_loss
                    + args.reader_weight * reader_loss
                )
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimiser.step()
                with torch.no_grad():
                    replay_critic += critic_loss
                    replay_oracle += oracle_loss
                    replay_read_right += reader_right
                replay_steps += 1

        # One synchronisation here replaces the many per-minibatch metric
        # synchronisations above. Once the first scalar is read, the rest are
        # already complete.
        denom = max(steps, 1)
        confidence_total = (sure_count + likely_count + unlikely_count).clamp(min=1)
        record = {
            "generation": generation,
            "decisions": batch.decisions,
            "hands": batch.hands,
            "seconds": round(time.time() - began, 1),
            "play_seconds": round(played, 1),
            # Where the play went: the engine and follower, the encoder,
            # the network, the seated others, and the bookkeeping.
            "play_split": {name: round(value, 1) for name, value in batch.timing.items()},
            # Moving the round to the card, and whether it went there.
            "load_seconds": round(loaded, 1),
            "resident": on_card is not None,
            "baseline_seconds": round(baseline_seconds, 1),
            "policy_loss": round(float(total_policy / denom), 4),
            "value_loss": round(float(total_value / denom), 4),
            "oracle_loss": round(float(total_oracle / denom), 4),
            "critic_loss": round(float(total_critic / denom), 4),
            # Each head's error on the round before it was trained on it,
            # against the return variance below, and which was the baseline.
            "public_error": round(public_error, 4),
            "oracle_error": round(oracle_error, 4),
            "critic_error": round(critic_error, 4),
            "baseline": chosen,
            "advantage_spread": round(advantage_spread, 4),
            # Mean standardised advantage of the taken action by the policy's
            # confidence in it, and how much of the round each bin was.
            "advantage_sure": round(float(sure_sum / sure_count.clamp(min=1)), 4),
            "advantage_likely": round(float(likely_sum / likely_count.clamp(min=1)), 4),
            "advantage_unlikely": round(float(unlikely_sum / unlikely_count.clamp(min=1)), 4),
            "share_sure": round(float(sure_count / confidence_total), 3),
            "share_likely": round(float(likely_count / confidence_total), 3),
            "share_unlikely": round(float(unlikely_count / confidence_total), 3),
            # The same heads on the ring of past rounds, which is where
            # they must not learn a round by heart.
            "replay_rounds": len(ring),
            "replay_critic_loss": round(float(replay_critic / max(replay_steps, 1)), 4),
            "replay_oracle_loss": round(float(replay_oracle / max(replay_steps, 1)), 4),
            "replay_reader_right": round(float(replay_read_right / max(replay_steps, 1)), 4),
            "distil": round(float(total_distil / denom), 4),
            # The reader's loss and how often it tells a real set of hidden
            # hands from an imagined one; a half is guessing.
            "reader_loss": round(float(total_reader / denom), 4),
            "reader_right": round(float(total_read_right / denom), 4),
            # What a constant guess would score, so the two losses above
            # read as how much of the return each head explains.
            "return_variance": round(float(returns.var()), 4),
            "entropy": round(float(total_entropy / denom), 4),
            "hands_loss": round(float(total_hands / denom), 4),
            "hands_read": round(float(total_covered / denom), 4),
            "clipped": round(float(total_clipped / denom), 3),
            "approx_kl": round(float(total_kl / denom), 5),
            "grad_norm": round(float(total_grad_norm / denom), 3),
            "mean_return": round(float(returns.mean()), 4),
        }

        measured = None
        is_best = False
        if (generation + 1) % args.measure_every == 0 or generation == 0:
            measured = selfplay.measure(
                net,
                games=args.measure_games,
                seed=7_000_000 + generation,
                device=device,
                amp=args.amp,
            )
            record.update(
                {
                    "placement": round(measured["placement"], 3),
                    "score": round(measured["score"], 1),
                    "win_rate": round(measured["wins"], 3),
                }
            )

            # The high-water mark is kept apart, so a run that wanders can
            # always be brought back to the best network it has produced.
            # It is chosen on a smoothed figure rather than on the single
            # measurement, because one measurement of a few hundred games
            # carries a standard error about as large as the improvement
            # being looked for.
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

        # Save after measurement so RNG and metadata describe exactly the
        # state from which the next generation will continue. Adam's moments
        # are deliberately part of the checkpoint: restarting them every
        # Modal block used to turn every resume into a different optimiser.
        payload = checkpoint_payload(generation + 1)
        if measured is not None:
            payload["placement"] = measured["placement"]
        if is_best:
            torch.save(payload, args.out / "best.pt")

        # Saved every generation, not only when measured, so that a restart
        # loses one generation at most rather than every one since the last
        # measurement.
        torch.save(payload, args.out / "latest.pt")

        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # The same fields the per-generation save writes. This one used to drop
    # the smoothed placement and the best it had reached, so every restart
    # began judging from nothing however carefully they were carried.
    torch.save(checkpoint_payload(max(end, start)), args.out / "latest.pt")
    print("training finished", flush=True)


if __name__ == "__main__":
    main()
