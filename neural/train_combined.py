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

from . import checkpoint
from . import combined, population, selfplay, zoo
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
    parser.add_argument("--entropy", type=float, default=0.0005)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument(
        "--hands-weight",
        type=float,
        default=1.0,
        help="how much to weigh reading the opponents' hands, which is a "
        "free and dense label where the game's result is neither, and what "
        "a search would deal the unseen tiles from",
    )
    parser.add_argument(
        "--leash",
        type=float,
        default=0.0,
        help="what a nat of distance from the policy the run started with "
        "costs. PPO's clip holds one step near the last; this holds the "
        "run near its beginning, which is what stopped a good policy "
        "drifting away over twenty generations",
    )
    parser.add_argument(
        "--fixed",
        nargs="*",
        default=list(combined.Combined.MODES),
        help="what stays fixed, drawn evenly each generation from these: "
        "none, mortal, ours, head, or a pair joined with a plus such as "
        "mortal+head. Freezing a network without its head lets the head "
        "move the policy anyway",
    )
    parser.add_argument(
        "--opponents", type=Path, nargs="*", default=[],
        help="kept for launchers that name opponents directly; the roster "
             "in neural.population is what seats them now",
    )
    parser.add_argument(
        "--explore", type=float, default=0.0,
        help="how often a legal move is taken at random instead of the "
             "policy's, so the value head sees the positions a search asks "
             "it about rather than only the ones the policy reaches. The "
             "probability written down is the mixture's, not the policy's, "
             "so PPO divides by who actually chose",
    )
    parser.add_argument(
        "--champion", default=None,
        help="the checkpoint that last passed the gate, seated most often",
    )
    parser.add_argument(
        "--recent", nargs="*", default=[],
        help="checkpoints from the last few blocks of this lineage",
    )
    parser.add_argument(
        "--older", nargs="*", default=[],
        help="checkpoints from further back, which catch a policy going "
             "round in circles",
    )
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

    # The policy the run began with, kept beside the checkpoints so every
    # block of it leashes to the same place. Written once, on the first
    # block; a later block reads it rather than making a new one, or the
    # leash would only ever hold the run to where it last stopped.
    reference = None
    if args.leash > 0:
        kept = args.out / "reference.pt"
        if not kept.exists():
            checkpoint.publish(net.state(), kept)
            print(f"kept the starting policy at {kept}", flush=True)
        reference, _reference_state = combined.load(kept, device)
        reference.eval()
        reference.requires_grad_(False)
        print(f"leashed to {kept} at {args.leash} a nat", flush=True)

    # One optimiser over everything, in three groups; a group whose
    # parameters are fixed this generation gets no gradients and is left
    # alone by it.
    net.set_mode("none")

    def hands_loss_of(guessed, wanted):
        """Cross-entropy against the distribution each opponent's hand
        actually was, over the 34 kinds, and how much of the hand the guess
        covers, which is readable where a cross-entropy is not. Positions
        where nobody was holding anything are rows of zeros, and skipped."""
        holding = wanted.sum(dim=2) > 0
        log_guess = torch.log_softmax(guessed, dim=2)
        loss = -(wanted * log_guess).sum(dim=2)
        loss = (loss * holding).sum() / holding.sum().clamp(min=1)
        with torch.no_grad():
            overlap = torch.minimum(log_guess.exp(), wanted).sum(dim=2)
            covered = (overlap * holding).sum() / holding.sum().clamp(min=1)
        return loss, covered

    optimiser = torch.optim.AdamW(
        [
            # The head and the two things that always train share a rate;
            # what a generation holds still is decided by `set_mode`, not by
            # leaving it out here.
            {"params": net.head_trained() + net.always_trained(), "lr": args.lr},
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
            # A parameter whose shape has changed carries moments of the
            # old shape, and AdamW will not say so until it steps: it fails
            # with a complaint about dtype and layout, a generation's work
            # after the thing that was wrong. The reader's output layer
            # changed shape when it was made able to name a tile, so the
            # moments are checked here against the parameters they belong
            # to and any that no longer fit are started fresh.
            stale = []
            for group in optimiser.param_groups:
                for parameter in group["params"]:
                    kept = optimiser.state.get(parameter)
                    if not kept:
                        continue
                    for name in ("exp_avg", "exp_avg_sq"):
                        moment = kept.get(name)
                        if moment is not None and moment.shape != parameter.shape:
                            stale.append(parameter)
                            break
            for parameter in stale:
                optimiser.state.pop(parameter, None)
            if stale:
                print(
                    f"{len(stale)} parameters changed shape; their AdamW moments "
                    "start again",
                    flush=True,
                )
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

    # Who else sits at the tables, as a roster with roles and shares rather
    # than a list of paths typed on the launch line: see `neural.population`.
    # A member that is missing is dropped from the roster rather than
    # skipped silently at seating time, so the shares still add up and the
    # log says who was actually available.
    #
    # Checkpoints named with `--opponents` still seat: a launcher that
    # copies them to local files and passes the paths is how every run has
    # started one, and quietly seating nobody because the roster wanted
    # different flags would be a silent loss of the whole population.
    wanted = [
        (Path(path), member)
        for path, member in zip(
            args.opponents, population.Population.from_paths(args.opponents).members
        )
    ]
    for name in (args.champion, *args.recent, *args.older):
        if name:
            role = (
                "champion" if name == args.champion
                else "recent" if name in args.recent
                else "older"
            )
            weight = {"champion": 3.0, "recent": 1.0, "older": 0.5}[role]
            wanted.append(
                (Path(name), population.Member(str(name), role, f"named as {role}", weight))
            )
    seated, members = [], []
    for path, member in wanted:
        if not path.exists():
            print(f"no opponent at {path}, left out of the roster", flush=True)
            continue
        other = zoo.load_player(path, device, compile=args.compile)
        other.eval()
        seated.append(other)
        members.append(member)
    roster = population.Population(members=members)
    if seated:
        print(
            f"{len(seated)} others seated in {args.opponent_share:.0%} of games: "
            + json.dumps(roster.describe()),
            flush=True,
        )
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
            population=roster,
            explore_share=args.explore,
        )
        played = time.time() - began
        observations = batch.observations
        legal = batch.legal.to(device)
        actions = batch.actions.to(device)
        returns = batch.returns.to(device)
        old_log_probs = batch.log_probs.to(device)
        # What the three opponents were really holding at each decision: the
        # label the reading of the hands is trained against, which self-play
        # knows for free and which is far denser than the game's result.
        held = batch.held.to(device)
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
        total_leash, total_hands, total_covered = zero(), zero(), zero()
        steps = 0
        for _epoch in range(args.epochs):
            order = torch.randperm(batch.decisions)
            slices = [
                order[start_index : start_index + args.batch]
                for start_index in range(0, batch.decisions, args.batch)
            ]
            # Whole minibatches only, so the compiled step sees one shape.
            slices = [drawn for drawn in slices if drawn.numel() == args.batch]
            if not slices:
                # Every minibatch was a remainder, so this epoch would
                # train on nothing. A round smaller than one batch used to
                # pass through here in silence: the generation was written,
                # the checkpoint saved, the process exited nought, and not a
                # weight had moved. A smoke test that proves only that the
                # script runs is worse than no smoke test.
                raise RuntimeError(
                    f"a round of {batch.decisions} decisions makes no whole minibatch of "
                    f"{args.batch}; nothing would be learned from it. Lower --batch or "
                    "raise --games."
                )

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
                    logits, value, guessed = learn(planes, legal[picks])
                logits = logits.float()
                value = value.float()
                hands_loss, covered = hands_loss_of(guessed.float(), held[picks])
                distribution = torch.distributions.Categorical(logits=logits)
                log_prob = distribution.log_prob(actions[picks])
                advantage = advantages[picks]
                ratio = torch.exp(log_prob - old_log_probs[picks])
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
                value_loss = nn.functional.mse_loss(value, returns[picks])
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + args.value_weight * value_loss
                    + args.hands_weight * hands_loss
                    - args.entropy * entropy
                )
                # How far this has come from the policy the run began with,
                # counted the way that punishes abandoning a move the
                # starting policy liked, which is the drift that cost us.
                leash = torch.zeros((), device=device)
                if reference is not None:
                    with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=amp_enabled
                    ):
                        before, _before_value, _before_hands = reference.everything(
                            planes, legal[picks]
                        )
                    allowed = legal[picks]
                    before = torch.log_softmax(before.float(), dim=1)
                    now = torch.log_softmax(logits, dim=1)
                    # A move outside the mask holds minus infinity in both,
                    # and one infinity less another is not a number, so it
                    # is taken out by value; a zero weight does not cancel
                    # it, it only spreads the NaN.
                    weight = torch.where(allowed, before.exp(), torch.zeros_like(before))
                    before = torch.where(allowed, before, torch.zeros_like(before))
                    now = torch.where(allowed, now, torch.zeros_like(now))
                    leash = (weight * (before - now)).sum(dim=1).mean()
                    loss = loss + args.leash * leash
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
                    total_leash += leash
                    total_hands += hands_loss
                    total_covered += covered
                steps += 1

        # What the head is leaning on, on the last minibatch: how far it
        # moved our own logits, and the weight it puts on each of the three
        # answers it weighs, its own, Mortal's and ours.
        head_shift = 0.0
        weights = net.fuse.weights.mean(dim=1).tolist()
        # A round that gathered fewer decisions than one minibatch trains on
        # none of them, and there is then no last minibatch to read. Only the
        # weights can be reported, which are the network's rather than the
        # round's.
        if steps:
            with torch.no_grad():
                net.eval()
                phi, q, features, a1, _value, _guessed = net.backbones(planes, legal[picks])
                joined = net.fuse(phi.float(), q.float(), features.float(), a1.float(), legal[picks])
                allowed = legal[picks]
                shift = (joined - a1).abs().masked_fill(~allowed, 0.0)
                head_shift = float(shift.sum() / allowed.sum().clamp(min=1))
            net.train()

        denom = max(steps, 1)
        record = {
            "generation": generation,
            "fixed": fixed,
            "head_shift": round(head_shift, 4),
            "on_fusion": round(weights[0], 4),
            "on_mortal": round(weights[1], 4),
            "on_ours": round(weights[2], 4),
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
            "leash_kl": round(float(total_leash / denom), 5),
            "hands_loss": round(float(total_hands / denom), 4),
            "hands_covered": round(float(total_covered / denom), 4),
            "clipped": round(float(total_clipped / denom), 3),
            "approx_kl": round(float(total_kl / denom), 5),
            "optimiser_steps": steps,
            # One row a player met, never summed. Improving against your
            # own recent past while losing to the fine-tuned Mortal is
            # specialisation, and an average is what hides it.
            "matchups": batch.matchups,
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
            # `candidate.pt` is the generation the heuristic table liked
            # best so far, and that is all it is. The bots compress real
            # differences several-fold and this measures one seat, so
            # being the best of these readings is a reason to put a
            # checkpoint forward, not a finding that it is stronger.
            # `neural.promote` decides that, by sitting it opposite the
            # champion; nothing here may write `champion.pt`.
            checkpoint.publish(payload, args.out / "candidate.pt")
            checkpoint.publish(payload, args.out / "best.pt")
        checkpoint.publish(payload, args.out / "latest.pt")
        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    checkpoint.publish(checkpoint_payload(max(end, start)), args.out / "latest.pt")
    print("training finished", flush=True)


if __name__ == "__main__":
    main()
