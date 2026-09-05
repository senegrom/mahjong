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

from . import selfplay
from .model import HIDDEN_HANDS_PLANES, PolicyValueNet, load_weights
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
# 320 channels by 20 blocks: 12.6M parameters, about fifty megabytes of
# float weights. AlphaZero's twenty blocks of 256 came to roughly 23M
# parameters on a Go board; on a line of thirty-four tiles the same shape is
# a third of that, because the kernel is three rather than three by three,
# and this width puts the count nearer the original's. Far too big for a
# phone, which is something to distil away later rather than a reason to
# train something smaller now.
    parser.add_argument("--channels", type=int, default=320)
    parser.add_argument("--blocks", type=int, default=20)
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
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    net = PolicyValueNet(args.channels, args.blocks).to(device)
    start = 0
    best_placement = float("inf")
    smoothed = None
    if args.resume and args.resume.exists():
        payload = torch.load(args.resume, map_location=device, weights_only=True)
        load_weights(net, payload["model"])
        start = int(payload.get("generation", 0))
        if payload.get("channels") not in (None, net.channels):
            raise SystemExit("the checkpoint was trained at a different width")
        print(f"resumed from {args.resume} at generation {start}", flush=True)

    if args.freeze_policy:
        for name, parameter in net.named_parameters():
            if not name.startswith(("critic", "oracle_", "reader")):
                parameter.requires_grad_(False)
        print("policy frozen: training the critic, the oracle and the reader", flush=True)
    if args.freeze_aux:
        for name, parameter in net.named_parameters():
            if name.startswith(("critic", "oracle_", "reader", "hands.", "value.")):
                parameter.requires_grad_(False)
        args.value_weight = 0.0
        args.hands_weight = 0.0
        args.reader_weight = 0.0
        args.replay_steps = 0
        print("auxiliaries frozen: training the policy by the policy loss alone", flush=True)
    trainable = [parameter for parameter in net.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    learn = torch.compile(net.with_oracle) if args.compile else net.with_oracle
    read = torch.compile(net.read_plausibility) if args.compile else net.read_plausibility

    # The last several rounds, on disk, for the heads that may learn from
    # stale play: see `replay.py`. The policy never trains on it.
    ring = Ring(args.out / "ring", args.replay_rounds)
    replay_rng = np.random.default_rng(args.seed + 17)

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
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
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
        f"| {net.parameter_count() / 1e6:.2f}M parameters",
        flush=True,
    )

    # A round of planes is about five gigabytes on the host. The names
    # below are cleared before the next round is played, so the machine
    # carries one round rather than two.
    batch = observations = oracle = imagined = None
    end = start + args.rounds if args.rounds else args.generations
    for generation in range(start, end):
        began = time.time()
        batch = observations = oracle = imagined = None
        batch = selfplay.play(
            net,
            games=args.games,
            seed=args.seed + generation * 1000,
            device=device,
            amp=args.amp,
        )
        played = time.time() - began
        ring.push(batch)

        # The observations stay on the host and each minibatch crosses to
        # the card as it is drawn: a round of them is five gigabytes and sat
        # on the card beside the step's own eight, which with the oracle
        # critic left a gigabyte spare of sixteen. A minibatch is 27 MB and
        # crosses in a few milliseconds. Not pinned: pinning copies the
        # round into page-locked memory, which took the run from twelve
        # gigabytes of host memory to twenty-eight and the machine to none.
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

        # The baseline for the policy gradient is the oracle critic as it
        # stands before this round's updates, computed once for the whole
        # round. It used to be recomputed inside the epochs from the head
        # being updated, and a critic that sees the hidden tiles fits a
        # round's returns within an epoch or two: the advantages shrank
        # towards nothing, the entropy bonus was all that was left, and the
        # policy drifted towards uniform for twenty-five generations,
        # entropy rising from 0.36 to 0.47 and placement worsening with it.
        # The advantages are then standardised, because the clipped
        # objective is not indifferent to a shift in them the way a plain
        # policy gradient is.
        #
        # Which head is the baseline is the round's to decide. The oracle
        # memorises: the hidden planes make nearly every position unique,
        # and its loss inside the epochs fell to two thirds of its error on
        # the next round, which was no better than guessing the mean. A
        # baseline worse than the public head adds noise rather than taking
        # it away, so both heads are measured on the round before either is
        # updated, the better one is the baseline, and the public head is
        # distilled towards the oracle only on rounds where the oracle is
        # the better of the two.
        net.eval()
        public_guess = torch.empty(batch.decisions, device=device)
        oracle_guess = torch.empty(batch.decisions, device=device)
        critic_guess = torch.empty(batch.decisions, device=device)
        with torch.no_grad():
            for start_index in range(0, batch.decisions, 8192):
                chunk = slice(start_index, start_index + 8192)
                planes = observations[chunk].to(device).float()
                seen = oracle[chunk].to(device).float()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                    _logits, guessed_value, _guessed, judged, criticised = net.with_oracle(
                        planes, legal[chunk], seen
                    )
                public_guess[chunk] = guessed_value.float()
                oracle_guess[chunk] = judged.float()
                critic_guess[chunk] = criticised.float()
        public_error = float(((normalised - public_guess) ** 2).mean())
        oracle_error = float(((normalised - oracle_guess) ** 2).mean())
        critic_error = float(((normalised - critic_guess) ** 2).mean())
        errors = {"public": public_error, "oracle": oracle_error, "critic": critic_error}
        chosen = min(errors, key=errors.get)
        baseline = {"public": public_guess, "oracle": oracle_guess, "critic": critic_guess}[chosen]
        oracle_better = chosen == "oracle"
        distil_weight = args.distil_weight if oracle_better else 0.0
        advantages = normalised - baseline
        advantage_spread = float(advantages.std())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        net.train()
        total_policy = total_value = total_entropy = 0.0
        total_oracle = total_distil = total_critic = 0.0
        # The advantage of the taken action, by how sure the policy was of
        # it: over a half, a fifth to a half, under a fifth.
        sure_sum = likely_sum = unlikely_sum = 0.0
        sure_count = likely_count = unlikely_count = 0
        total_reader = total_read_right = 0.0
        total_clipped = 0.0
        total_hands = total_covered = 0.0
        steps = 0
        for _epoch in range(args.epochs):
            order = torch.randperm(batch.decisions, device=device)
            for start_index in range(0, batch.decisions, args.batch):
                picks = order[start_index : start_index + args.batch]
                if picks.numel() < 2:
                    continue
                drawn = picks.cpu()
                # Half precision on the host, float32 on the card: the
                # tower's weights are float32, and autocast takes it from
                # there.
                planes = observations[drawn].to(device).float()
                seen = oracle[drawn].to(device).float()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
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
                    planes, seen[:, :HIDDEN_HANDS_PLANES], imagined[drawn].to(device).float()
                )
                distribution = torch.distributions.Categorical(logits=logits)
                log_prob = distribution.log_prob(actions[picks])
                # The oracle critic is the baseline, as it stood before the
                # round: see above. It depends on the hidden tiles but not
                # on the action, so it takes nothing away from the
                # gradient's expectation and a great deal from its noise.
                advantage = advantages[picks]

                # The clipped objective: an update may improve an action's
                # odds, but only so far in one round, which is what keeps a
                # policy from narrowing onto a single action.
                with torch.no_grad():
                    confidence = old_log_probs[picks].exp()
                    sure = confidence > 0.5
                    likely = (confidence > 0.2) & ~sure
                    unlikely = confidence <= 0.2
                    sure_sum += float((advantage * sure).sum())
                    sure_count += int(sure.sum())
                    likely_sum += float((advantage * likely).sum())
                    likely_count += int(likely.sum())
                    unlikely_sum += float((advantage * unlikely).sum())
                    unlikely_count += int(unlikely.sum())
                ratio = torch.exp(log_prob - old_log_probs[picks])
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
                total_clipped += float((ratio != clipped).float().mean())
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

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimiser.step()

                total_policy += policy_loss.item()
                total_value += value_loss.item()
                total_oracle += oracle_loss.item()
                total_critic += critic_loss.item()
                total_distil += distil_loss.item()
                total_reader += reader_loss.item()
                total_read_right += reader_right.item()
                total_entropy += entropy.item()
                total_hands += hands_loss.item()
                total_covered += covered.item()
                steps += 1

        # The heads that may learn from stale play take a pass over the
        # ring: the value heads, the reader and the head that reads the
        # table, on minibatches drawn evenly from the last several rounds.
        # No policy term, and no distillation, since the oracle's pre-round
        # estimate exists only for the round just played.
        replay_critic = replay_oracle = replay_read_right = 0.0
        replay_steps = 0
        if args.replay_steps and ring.total() >= args.batch:
            # Nothing in this pass reaches the policy tower: the critic, the
            # oracle and the reader each read it without gradient or not at
            # all. Every version that did reach it, however the policy's
            # output was held, crept the policy's entropy up a little each
            # generation, and the policy pass alone did not.
            for _ in range(args.replay_steps):
                rows = ring.sample(args.batch, replay_rng)
                planes = rows["observations"].to(device).float()
                seen = rows["oracle"].to(device).float()
                target = rows["returns"].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                    _logits, _value, _guessed, oracle_value, criticised = learn(
                        planes, rows["legal"].to(device), seen
                    )
                oracle_value = oracle_value.float()
                criticised = criticised.float()
                critic_loss = nn.functional.mse_loss(criticised, target)
                oracle_loss = nn.functional.mse_loss(oracle_value, target)
                reader_loss, reader_right = reader_loss_of(
                    planes, seen[:, :HIDDEN_HANDS_PLANES], rows["imagined"].to(device).float()
                )
                loss = (
                    args.value_weight * (critic_loss + oracle_loss)
                    + args.reader_weight * reader_loss
                )
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimiser.step()
                replay_critic += critic_loss.item()
                replay_oracle += oracle_loss.item()
                replay_read_right += reader_right.item()
                replay_steps += 1

        record = {
            "generation": generation,
            "decisions": batch.decisions,
            "hands": batch.hands,
            "seconds": round(time.time() - began, 1),
            "play_seconds": round(played, 1),
            "policy_loss": round(total_policy / max(steps, 1), 4),
            "value_loss": round(total_value / max(steps, 1), 4),
            "oracle_loss": round(total_oracle / max(steps, 1), 4),
            "critic_loss": round(total_critic / max(steps, 1), 4),
            # Each head's error on the round before it was trained on it,
            # against the return variance below, and which was the baseline.
            "public_error": round(public_error, 4),
            "oracle_error": round(oracle_error, 4),
            "critic_error": round(critic_error, 4),
            "baseline": chosen,
            "advantage_spread": round(advantage_spread, 4),
            # Mean standardised advantage of the taken action by the policy's
            # confidence in it, and how much of the round each bin was.
            "advantage_sure": round(sure_sum / max(sure_count, 1), 4),
            "advantage_likely": round(likely_sum / max(likely_count, 1), 4),
            "advantage_unlikely": round(unlikely_sum / max(unlikely_count, 1), 4),
            "share_sure": round(sure_count / max(steps * args.batch, 1), 3),
            # The same heads on the ring of past rounds, which is where
            # they must not learn a round by heart.
            "replay_rounds": len(ring),
            "replay_critic_loss": round(replay_critic / max(replay_steps, 1), 4),
            "replay_oracle_loss": round(replay_oracle / max(replay_steps, 1), 4),
            "replay_reader_right": round(replay_read_right / max(replay_steps, 1), 4),
            "distil": round(total_distil / max(steps, 1), 4),
            # The reader's loss and how often it tells a real set of hidden
            # hands from an imagined one; a half is guessing.
            "reader_loss": round(total_reader / max(steps, 1), 4),
            "reader_right": round(total_read_right / max(steps, 1), 4),
            # What a constant guess would score, so the two losses above
            # read as how much of the return each head explains.
            "return_variance": round(float(returns.var()), 4),
            "entropy": round(total_entropy / max(steps, 1), 4),
            "hands_loss": round(total_hands / max(steps, 1), 4),
            "hands_read": round(total_covered / max(steps, 1), 4),
            "clipped": round(total_clipped / max(steps, 1), 3),
            "mean_return": round(float(returns.mean()), 4),
        }

        payload = {
            "model": net.state_dict(),
            "generation": generation + 1,
            "channels": net.channels,
            "blocks": net.blocks,
        }
        if (generation + 1) % args.measure_every == 0 or generation == 0:
            against = selfplay.measure(
                net,
                games=args.measure_games,
                seed=7_000_000 + generation,
                device=device,
                amp=args.amp,
            )
            record.update(
                {
                    "placement": round(against["placement"], 3),
                    "score": round(against["score"], 1),
                    "win_rate": round(against["wins"], 3),
                }
            )
            payload["placement"] = against["placement"]

            # The high-water mark is kept apart, so a run that wanders can
            # always be brought back to the best network it has produced.
            # It is chosen on a smoothed figure rather than on the single
            # measurement, because one measurement of a few hundred games
            # carries a standard error about as large as the improvement
            # being looked for: keeping the best of forty such numbers keeps
            # the luckiest network, not the best one, and the number that
            # made the choice is then the one number guaranteed to flatter.
            smoothed = (
                against["placement"]
                if smoothed is None
                else SMOOTHING * against["placement"] + (1 - SMOOTHING) * smoothed
            )
            record["smoothed"] = round(smoothed, 3)
            if smoothed < best_placement:
                best_placement = smoothed
                torch.save(payload, args.out / "best.pt")
                record["best"] = True

        # Saved every generation, not only when measured, so that a restart
        # loses one generation at most rather than every one since the last
        # measurement.
        torch.save(payload, args.out / "latest.pt")

        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    torch.save(
        {"model": net.state_dict(), "generation": end,
         "channels": net.channels, "blocks": net.blocks},
        args.out / "latest.pt",
    )
    print("training finished", flush=True)


if __name__ == "__main__":
    main()
