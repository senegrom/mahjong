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
from contextlib import closing
import copy
import math
import time
from pathlib import Path

import torch
from torch import nn

from . import combined, policy_inference, ppo_loop, selfplay
from .behavior import validate_exploration
from .checkpoints import atomic_save
from .observe import pad_rows
from .ppo_control import PolicyDrift, add_training_controls, baseline_batch_size
from .training_batches import require_trainable_round, require_updates
from .training_safety import TRAINING_API_VERSION, benchmark_history
from .training_state import (
    capture_random_state, peak_gpu_gb, peak_rss_gb, restore_random_state, round_seed,
)


#: How many graphs one compiled forward may be given before the compiler
#: stops compiling it and runs the rest eagerly, unannounced. The learning
#: step's forward is compiled anew for every pattern of what requires a
#: gradient, which `Combined.set_mode` sets six ways (see
#: `compile_per_mode`); with gradients and without, for the baseline pass;
#: and with Mortal's vectors given and not (`--reuse-phi`): fourteen at
#: most, against the compiler's eight. Thirty-two leaves room for a
#: variant nobody foresaw.
RECOMPILE_LIMIT = 32


def guard_parameters() -> None:
    """Has the compiler check every parameter a compiled forward reads each
    time it is called, whether it requires a gradient included, so that
    each mode `Combined.set_mode` makes runs a graph traced in that mode.

    The cloud's torch 2.8 takes a parameter for unchanged while its
    module's dictionary of parameters is (its setting
    `skip_tensor_guards_with_matching_dict_tags`, on by default), and
    `set_mode` turns requires_grad on and off without touching that
    dictionary, so a graph traced in one mode was run in others, both
    ways. Where the mode held a part still that the graph trained, that
    part's backward ran and its gradients were thrown away, the very work
    holding it still is to save. Where the mode trained a part the graph
    had held still, that part was given no gradient at all and learned
    nothing that generation, without a word: under 2.8, eight of the
    thirty orders of two modes did this, a generation that held Mortal
    still followed by one that held nothing still leaving Mortal untrained
    in the second, for one. This desktop's torch 2.13 makes a graph a mode
    by itself."""
    config = torch._dynamo.config
    if hasattr(config, "skip_tensor_guards_with_matching_dict_tags"):
        config.skip_tensor_guards_with_matching_dict_tags = False


def compile_per_mode(limit: int = RECOMPILE_LIMIT) -> None:
    """Has the compiler make the learning step's graph anew for each mode
    (see `guard_parameters`), and lets it make `limit` of them (see
    `RECOMPILE_LIMIT`). `recompile_limit` is the setting's name in the
    cloud's torch 2.8 and in this desktop's 2.13; older releases called it
    `cache_size_limit`, which 2.8 keeps as another name for it."""
    config = torch._dynamo.config
    config.recompile_limit = max(config.recompile_limit, limit)
    guard_parameters()


def learning_forward(net: combined.Combined, args: argparse.Namespace):
    """The forward the learning step and the baseline pass call, compiled
    as the switches ask, every mode with graphs of its own (see
    `guard_parameters`).

    --compile compiles the joined player's forward for deciding as well,
    with the batch's size left symbolic, and keeps the compiler's own limit
    on graphs: past it, a variant runs eagerly, as it always did. Checking
    the parameters changes nothing else of it; this desktop's torch traced
    every mode anew already. --compile-learning compiles the learning step
    alone, at the one shape it is always handed: whole minibatches, and
    the pass's chunks padded to theirs."""
    if args.compile:
        guard_parameters()
        net.backbones_forward = torch.compile(net.backbones, dynamic=True)
        return torch.compile(net.everything)
    if args.compile_learning:
        compile_per_mode()
        return torch.compile(net.everything, dynamic=False)
    return net.everything


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
    parser.add_argument(
        "--entropy-target", type=float, default=0.0,
        help="hold the policy's entropy at or above this, nought for off. The "
        "bonus rises while entropy sits below the target and falls back to "
        "--entropy while it is above: a floor, never a push upwards",
    )
    parser.add_argument("--entropy-max", type=float, default=0.05,
                        help="the most the floor may raise the entropy bonus to")
    parser.add_argument("--entropy-rate", type=float, default=0.01,
                        help="how fast the floor's bonus moves, per update, in log units per nat")
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
        help="checkpoints to seat at the tables, as the cloud stages them. "
             "Each is a member of the roster (see neural.population): a "
             "name it knows keeps its role and weight there, any other is "
             "drawn with weight one",
    )
    parser.add_argument(
        "--explore", type=float, default=0.0,
        help="how often a legal move is taken at random instead of the "
             "policy's, so the value head and the reader of hands see "
             "positions the policy alone would not reach. The probability "
             "written down is the policy's own, and a forced move is kept "
             "out of the policy gradient; the other heads learn from it",
    )
    parser.add_argument("--opponent-share", type=float, default=0.0)
    parser.add_argument(
        "--seat-share", type=float, default=0.0,
        help="each player of each game is one of --opponents with this chance, so one table "
        "can hold several of them and the learner at once; instead of --opponent-share",
    )
    parser.add_argument(
        "--skip-forced", action="store_true",
        help="ask the seated --opponents nothing about a row with a single move open in "
        "Mortal's moves, whose move is then that one. Off by default: the rows still asked "
        "are scored in a smaller batch, which on the card can change their values in the "
        "last bit, and so turn a near tie",
    )
    parser.add_argument(
        "--preview-reach", action="store_true",
        help="ask which tile a reach would throw in the forward that decides on the reach, "
        "from the follower's preview of it, for the joined player and the seated --opponents, "
        "instead of telling the follower the reach and asking again: one forward where there "
        "were two. The same questions from the same states; the answers come from a larger "
        "batch, which in bfloat16 can move them in their last bits. Off by default",
    )
    parser.add_argument(
        "--baseline-from-play", action="store_true",
        help="measure the advantages against the value the joined player's head gave each "
        "decision as it played, instead of a pass over the round before it is learned. The "
        "same head, weights and planes; only the batches they went through differ, which in "
        "bfloat16 can move a value in its last bits. Off by default",
    )
    parser.add_argument(
        "--check-baseline", action="store_true",
        help="run the pass over the round as well and record how far play's values are from "
        "it (baseline_difference, its largest, and baseline_difference_mean): for the "
        "generation that checks --baseline-from-play before a run relies on it",
    )
    parser.add_argument(
        "--reuse-phi", action="store_true",
        help="in a generation that holds Mortal still (mortal, mortal+head), learn from Mortal's "
        "vector of each decision as play worked it out, kept on the card, instead of running "
        "Mortal again on every minibatch. Nothing moves Mortal in such a generation, so it is "
        "the same function of the same planes; only the batches differ, which in bfloat16 can "
        "move it in its last bits. Off by default; a card without room for it, about two "
        "kilobytes a decision, works it out as before",
    )
    parser.add_argument(
        "--play-ahead", action="store_true",
        help="play the next round on a worker while this one is learned, by the weights the "
        "learner has as this generation begins: play, bound by the processors, and learning, "
        "bound by the card, then overlap. Each round is played by the weights of a generation "
        "before the ones that learn it, a process's first excepted; PPO's ratio divides by the "
        "probabilities the round recorded, so its weighting stays exact, but its clip covers "
        "two generations' drift; the drift guard (--target-kl) counts from the learner's own "
        "probabilities as the generation begins, read in the pass, as without the switch. "
        "--baseline-from-play then takes play's values only from a round the learner played "
        "itself, a process's first, and makes the pass for the rest; --reuse-phi learns from "
        "Mortal's vectors only where nothing moved Mortal between the round and its learning. "
        "Off by default; not with --compile",
    )
    parser.add_argument("--measure-every", type=int, default=5)
    parser.add_argument("--measure-games", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--out", type=Path, default=Path("E:/tmp-claude/mahjong/joined-run"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--compile", action="store_true",
        help="compile the joined player's forward for deciding and for learning, and the "
        "seated others': over an hour of compiling in a fresh container",
    )
    parser.add_argument(
        "--compile-learning", action="store_true",
        help="compile the learning step's forward and nothing else: play and the seated others "
        "stay eager, with their weights cast to bfloat16 once and the step's planes made "
        "dense in bfloat16, as without compiling. Fuses the step's elementwise kernels; the "
        "numbers move by where bfloat16 rounds. Off by default; not with --compile",
    )
    add_training_controls(parser)
    args = parser.parse_args()
    if args.compile and args.compile_learning:
        parser.error("--compile already compiles the learning step; give one or the other")
    if args.compile and args.play_ahead:
        # --compile plays with the learner's own compiled forward and seats
        # compiled others, which a round played on a worker beside the
        # compiled learning would run, and compile, on a second thread.
        parser.error("--play-ahead plays eagerly beside the learning; with it, compile the "
                     "learning step alone (--compile-learning)")
    return args


def main() -> None:
    args = parse_args()
    device, amp_enabled, log_path = ppo_loop.setup(
        args, lambda options: validate_exploration(options.explore)
    )
    # In mixed precision the learning reads its planes only under autocast,
    # whose convolutions would cast float32 planes to bfloat16 first, so they
    # are made dense in bfloat16 straight away: half the bytes, and the same
    # bits handed to the convolutions (see `Planes.dense`). Not with
    # --compile, whose graphs were built for float32 planes: it gave the
    # same bits on this desktop's torch, but the compiler builds another
    # graph for inputs of another kind, which changed a compiled player's
    # answers when its weights were cast once (see `ppo_loop.load_others`),
    # and under the cloud's torch 2.8 it is untried. --compile-learning has
    # no float32 graphs to keep and is compiled for bfloat16 planes from the
    # start, which spares the prefetcher half its copying as the compiled
    # step shortens.
    planes_dtype = torch.bfloat16 if amp_enabled and not args.compile else torch.float32

    start = 0
    benchmark = ppo_loop.Benchmark()
    optimiser_state = None
    random_state = None
    entropy_log_coef = None
    if args.resume is not None and args.resume.exists():
        net, payload = combined.load(args.resume, device)
        start = int(payload.get("generation", 0))
        benchmark = ppo_loop.Benchmark(*benchmark_history(payload))
        optimiser_state = payload.get("optimizer_state")
        random_state = payload.get("random_state")
        entropy_log_coef = payload.get("entropy_log_coef")
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
            atomic_save(net.state(), kept)
            print(f"kept the starting policy at {kept}", flush=True)
        reference, _reference_state = combined.load(kept, device)
        reference.eval()
        reference.requires_grad_(False)
        print(f"leashed to {kept} at {args.leash} a nat", flush=True)

    # One optimiser over everything, in three groups; a group whose
    # parameters are fixed this generation gets no gradients and is left
    # alone by it.
    net.set_mode("none")
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
            optimiser.load_state_dict(combined.migrate_belief_optimizer(optimiser_state, optimiser, net))
            for group, lr in zip(optimiser.param_groups, (args.lr, args.lr_ours, args.lr_mortal)):
                group["lr"] = lr
            print("restored AdamW state", flush=True)
        except (ValueError, RuntimeError) as error:
            print(f"could not restore AdamW state ({error}); starting it fresh", flush=True)

    # The joined player as it plays: a copy whose convolutions and linear
    # layers hold their weights in bfloat16, refreshed from the learner
    # before every round (see `policy_inference.precast`). Played from the
    # learner itself, autocast cast those weights afresh at every step, five
    # hundred small kernels each time, to the same numbers, so the moves and
    # the probabilities recorded are bit for bit what they were. For eager
    # play in mixed precision only: with --compile the player is the
    # learner; --compile-learning leaves play eager, and the copy with it.
    actor = None
    if amp_enabled and not args.compile:
        actor = policy_inference.precast(copy.deepcopy(net)).requires_grad_(False)
    elif args.play_ahead:
        # A round played ahead is played by a copy in any precision: the
        # learner's own weights move while it is being played.
        actor = copy.deepcopy(net).requires_grad_(False)

    learn = learning_forward(net, args)

    # Who else sits at the tables, as a roster with roles and shares: see
    # `neural.population`. The checkpoints are the ones `--opponents`
    # names, which is how every launcher passes them.
    roster, seated = ppo_loop.seat_others(args, device)
    print(
        f"device {device} | ours {net.ours.channels}x{net.ours.blocks} | Mortal beneath | "
        f"{net.parameter_count() / 1e6:.2f}M parameters | fixed by turns: {args.fixed}"
        + (" | each round played ahead" if args.play_ahead else ""),
        flush=True,
    )
    # Constructors above consume Torch randomness. Restore only now, so the
    # next self-play decision and minibatch continue exactly where we saved.
    drawer = restore_random_state(
        random_state, seed=args.seed, generation=start, modes=args.fixed
    )

    def checkpoint_payload(generation: int) -> dict:
        return {
            **net.state(),
            "learner": "combined",
            "generation": generation,
            "training_api_version": TRAINING_API_VERSION,
            "training_controls": {"target_kl": args.target_kl, "baseline_batch": args.baseline_batch,
                                  "explore": args.explore, "ppo_reference": "unforced_policy_rows",
                                  "seat_share": args.seat_share},
            "smoothed": benchmark.smoothed,
            "best_placement": benchmark.best,
            "optimizer_state": optimiser.state_dict(),
            "random_state": capture_random_state(drawer),
            "entropy_log_coef": floor["log_coef"],
        }

    def measure(generation: int) -> dict:
        net.eval()
        return selfplay.measure(
            net, games=args.measure_games, seed=7_000_000 + generation, device=device
        )

    # The entropy floor: the bonus is exp(log_coef), never below the fixed
    # --entropy and never above --entropy-max, and it is a dual variable --
    # it climbs while the policy's entropy is under the target and sinks
    # back while it is over. Self-play here was measured to buy reward by
    # getting louder: 30 generations cut entropy by 29% and lost 0.0075.
    if args.entropy_target > 0 and args.entropy <= 0:
        raise SystemExit("--entropy-target needs a positive --entropy to fall back to")
    base_log = math.log(args.entropy) if args.entropy > 0 else float("-inf")
    floor = {"log_coef": float(entropy_log_coef) if entropy_log_coef is not None else base_log}
    ceiling_log = math.log(max(args.entropy_max, args.entropy)) if args.entropy > 0 else base_log

    def entropy_coef() -> float:
        return math.exp(floor["log_coef"]) if args.entropy_target > 0 else args.entropy

    def play(player, generation: int, **options):
        """Round `generation`, played by `player`: see `selfplay.play`,
        which `options` are handed on to."""
        return selfplay.play(
            player,
            games=args.games,
            seed=round_seed(args.seed, generation, args.games),
            device=device,
            amp=amp_enabled,
            opponents=seated,
            opponent_share=args.opponent_share,
            seat_share=args.seat_share,
            population=roster,
            explore_share=args.explore,
            preview_reach=args.preview_reach,
            **options,
        )

    end = start + args.rounds if args.rounds else args.generations
    # With --play-ahead each round is played on a worker while the one
    # before it is learned, by the copy above (see `ppo_loop.PlayAhead`).
    ahead = ppo_loop.PlayAhead(net, actor, device, play) if args.play_ahead else None
    try:
        for generation in range(start, end):
            began = time.time()
            batch = rollout = held = played_phi = None
            # Deciding is the joined player as it stands; what stays fixed in
            # the update that follows is drawn now and said in the record.
            fixed = str(drawer.choice(args.fixed))
            net.set_mode(fixed)
            net.eval()
            # A generation that holds Mortal still has nothing to move it, its
            # weights or its statistics, so its vector of every decision is
            # the same all generation: kept from play when asked to reuse it.
            reusing = args.reuse_phi and "mortal" in fixed.split("+")
            # Whether the round was played beside the last generation's
            # learning, by the weights from before it (--play-ahead).
            lagged = False
            if ahead is None:
                player = net if actor is None else actor
                if actor is not None:
                    # Copied with the same rounding autocast makes.
                    actor.load_state_dict(net.state_dict())
                player.keep_phi = reusing
                batch = play(player, generation)
                played = time.time() - began
            else:
                lagged = ahead.playing is not None
                if not lagged:
                    # A process's first round: nothing was played beside a
                    # generation before, so the learner plays it now.
                    ahead.start(generation, keep_phi=reusing)
                batch, played, waited = ahead.take(generation)
                if generation + 1 < end:
                    # The next round, played while this one is learned, by the
                    # weights the learner has now; never one past the last
                    # generation, which nothing would learn. Mortal's vectors
                    # are kept from it only when this generation and the next
                    # both hold Mortal still: the next can learn from them
                    # only if nothing moved Mortal between the copy and its
                    # own learning, this generation's learning being all that
                    # comes between, and only if its own learning moves
                    # Mortal no more. The next generation's mode is the one a
                    # copy of the drawer draws now, nothing else drawing
                    # from it.
                    following = str(copy.deepcopy(drawer).choice(args.fixed))
                    ahead.start(generation + 1,
                                keep_phi=reusing and "mortal" in following.split("+"))
            require_trainable_round(batch.decisions, args.batch, args.epochs)
            loading = time.time()
            # Beside a round being played the planes stay on the host: the
            # card holds that round's work as well, and keeping a round on
            # the card was never seen to shorten its learning.
            rollout = ppo_loop.on_device(batch, device, card=ahead is None)
            # What the three opponents were really holding at each decision: the
            # label the reading of the hands is trained against, which self-play
            # knows for free and which is far denser than the game's result.
            held = batch.held.to(device)
            if reusing and getattr(batch, "phi", None) is not None:
                # On the card beside the step, or not at all: gathered from the
                # host every minibatch it would hold the step up. Beside a round
                # being played, only with that round's room to spare as well.
                played_phi = (ppo_loop.kept_on_card(batch.phi, device) if ahead is None
                              else ppo_loop.kept_on_card(batch.phi, device, ahead.spare()))
                batch.phi = None
            loaded = time.time() - loading

            # The baseline: our value head as it stands before the round, from
            # a pass over it or, asked to, from play (see `ppo_loop.baseline_of`).
            rows = baseline_batch_size(batch.decisions, args.batch, args.baseline_batch)
            # For a round played ahead, the probability the learner as it
            # stands gives each move made, which the drift guard counts from
            # (see below): read in the pass, which such a round always gets.
            begun = torch.empty(batch.decisions, device=device) if lagged else None

            def values_of(chunk, planes):
                mask = pad_rows(rollout.legal[chunk], rows, True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    logits, value, _guessed = learn(planes, mask)
                if begun is not None:
                    taken = rollout.actions[chunk]
                    begun[chunk] = torch.distributions.Categorical(
                        logits=logits[: len(taken)].float(), validate_args=False
                    ).log_prob(taken)
                return (value,)

            guess, valued, baseline_said = ppo_loop.baseline_of(
                rollout, batch, rows, values_of,
                # Play's values are the head's as it stands only in a round
                # the learner played itself. One played ahead was valued by
                # the head of the generation before, and a generation's
                # learning moves that head far beyond the bfloat16 noise the
                # switch was accepted for: on this desktop, at fusion-long's
                # rates from random weights, such values stood 0.18 to 0.28
                # from the pass on average, where the learner's own stood
                # 0.0006 (a trained head's drift is unmeasured). Still a
                # baseline, since it does not depend on the move made, but
                # not the one the switch promised, so such a round gets the
                # pass; checked, the record says how far the head moved.
                from_play=args.baseline_from_play and not lagged,
                check=args.check_baseline, dtype=planes_dtype,
            )
            value_error = float(((rollout.returns - guess) ** 2).mean())
            advantages, spread = ppo_loop.standardised(rollout.returns, guess)

            net.train()
            totals = ppo_loop.Totals(device, "leash", "hands", "covered")
            drift = PolicyDrift(args.target_kl)
            # Rows whose move was forced by exploration are the value head's
            # and the hand reading's to learn from, not the policy's: the
            # policy did not choose them, and a ratio against its own
            # probability there sits far outside the clip, where a bad move
            # teaches nothing and a lucky one teaches the wrong thing. Kept out
            # of every policy term below; nothing is forced when --explore is
            # zero.
            explored = (
                batch.explored.to(device=device, dtype=torch.bool)
                if getattr(batch, "explored", None) is not None
                else torch.zeros(batch.decisions, dtype=torch.bool, device=device)
            )
            steps = 0
            for _epoch in range(args.epochs):
                with closing(ppo_loop.minibatches(rollout, args.batch, dtype=planes_dtype)) as minibatches:
                    for picks, planes in minibatches:
                        optimiser.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                            if played_phi is None:
                                logits, value, guessed = learn(planes, rollout.legal[picks])
                            else:
                                logits, value, guessed = learn(
                                    planes, rollout.legal[picks], played_phi[picks]
                                )
                        logits = logits.float()
                        value = value.float()
                        hands_loss, covered = ppo_loop.hands_loss_of(guessed.float(), held[picks])
                        # Unvalidated: checking the logits and the moves for the
                        # distribution waited on the card twice a step, and a
                        # logit that is not a number still stops the step, at the
                        # gradient's clip below.
                        distribution = torch.distributions.Categorical(logits=logits, validate_args=False)
                        log_prob = distribution.log_prob(rollout.actions[picks])
                        # The probabilities the round recorded as it was
                        # played: the policy that played it, which with
                        # --play-ahead is the generation before's, so the
                        # ratio below weighs each move by the right one.
                        old_log_prob = rollout.old_log_probs[picks]
                        # The drift guard counts how far this generation's
                        # learning has moved the policy: from those same
                        # probabilities in a round the learner played itself,
                        # and in one played ahead from the learner's own as
                        # the generation began (`begun`). Counted from the
                        # ones it was played by, a generation would begin with
                        # the last one's drift already spent: after one the
                        # guard stopped, the next could not take a step, and
                        # a generation that takes none stops the trainer.
                        since = old_log_prob if begun is None else begun[picks]
                        chosen = ~explored[picks]
                        own = chosen.sum().clamp(min=1)
                        if args.explore:
                            stop = bool(chosen.any()) and drift.check(since[chosen], log_prob[chosen])
                        else:
                            # Nothing was explored, so every row is the policy's;
                            # asking the card which ones would wait on it.
                            stop = drift.check(since, log_prob)
                        if stop:
                            break
                        # PPO's clipped objective over the rows the policy chose.
                        advantage = advantages[picks].masked_fill(~chosen, 0.0)
                        delta = (log_prob - old_log_prob).masked_fill(~chosen, 0.0)
                        ratio = torch.exp(delta)
                        clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                        surrogate = torch.min(ratio * advantage, clipped * advantage)
                        policy_loss = -(surrogate * chosen).sum() / own
                        value_loss = nn.functional.mse_loss(value, rollout.returns[picks])
                        entropy = (distribution.entropy() * chosen).sum() / own
                        loss = (
                            policy_loss
                            + args.value_weight * value_loss
                            + args.hands_weight * hands_loss
                            - entropy_coef() * entropy
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
                                    planes, rollout.legal[picks]
                                )
                            allowed = rollout.legal[picks]
                            before = torch.log_softmax(before.float(), dim=1)
                            now = torch.log_softmax(logits, dim=1)
                            # A move outside the mask holds minus infinity in both,
                            # and one infinity less another is not a number, so it
                            # is taken out by value; a zero weight does not cancel
                            # it, it only spreads the NaN.
                            weight = torch.where(allowed, before.exp(), torch.zeros_like(before))
                            before = torch.where(allowed, before, torch.zeros_like(before))
                            now = torch.where(allowed, now, torch.zeros_like(now))
                            leash = ((weight * (before - now)).sum(dim=1) * chosen).sum() / own
                            loss = loss + args.leash * leash
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(
                            [p for p in net.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
                        )
                        optimiser.step()
                        if args.entropy_target > 0:
                            # A dual step: under the target the bonus climbs,
                            # over it the bonus sinks back to the fixed floor.
                            shortfall = args.entropy_target - float(entropy.detach())
                            floor["log_coef"] = min(ceiling_log, max(
                                base_log, floor["log_coef"] + args.entropy_rate * shortfall))
                        with torch.no_grad():
                            totals.add(
                                policy=policy_loss,
                                value=value_loss,
                                entropy=entropy,
                                clipped=((ratio != clipped).float() * chosen).sum() / own,
                                kl=-delta.sum() / own,
                                grad=grad_norm,
                                leash=leash,
                                hands=hands_loss,
                                covered=covered,
                            )
                        steps += 1
                if drift.stopped:
                    break

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
                    allowed = rollout.legal[picks]
                    if planes.dtype != torch.float32:
                        # This runs without autocast, so the minibatch is made
                        # dense again as it always was.
                        planes = ppo_loop.planes_of(rollout, picks)
                    phi, q, features, a1, _value, _guessed = net.backbones(planes, allowed)
                    joined = net.fuse(phi.float(), q.float(), features.float(), a1.float(), allowed)
                    shift = (joined - a1).abs().masked_fill(~allowed, 0.0)
                    head_shift = float(shift.sum() / allowed.sum().clamp(min=1))
                net.train()

            require_updates(steps)
            entry = ppo_loop.record(
                generation, steps, drift, rows, rollout, batch,
                {"began": began, "played": played, "loaded": loaded}, totals, spread,
                fixed=fixed,
                head_shift=round(head_shift, 4),
                on_fusion=round(weights[0], 4),
                on_mortal=round(weights[1], 4),
                on_ours=round(weights[2], 4),
                value_error=round(value_error, 4),
                # The pass that values the round before it is learned, or the
                # values play recorded, and how far apart the two are when both
                # were asked for.
                baseline_seconds=round(valued, 1),
                **baseline_said,
                # Whether this generation learned from Mortal's vectors as
                # play worked them out, when the run reuses them.
                **({"reused_phi": played_phi is not None} if args.reuse_phi else {}),
                # With --play-ahead, whether the round was played beside the
                # last generation's learning, by the weights from before it,
                # and how long this generation waited for it: the part of
                # play_seconds, the round's own, that the learning did not hide.
                **({"played_ahead": lagged, "play_wait_seconds": round(waited, 1)}
                   if ahead is not None else {}),
                entropy_coef=round(entropy_coef(), 6),
                # Peaks, so the container's reservation can be sized from data:
                # memory is billed by what is reserved, not what is used.
                peak_rss_gb=peak_rss_gb(),
                peak_gpu_gb=peak_gpu_gb(),
                leash_kl=totals.mean("leash", steps, 5),
                hands_loss=totals.mean("hands", steps, 4),
                hands_covered=totals.mean("covered", steps, 4),
                # One row a player met, never summed. Improving against your
                # own recent past while losing to the fine-tuned Mortal is
                # specialisation, and an average is what hides it.
                matchups=getattr(batch, "matchups", None),
            )
            ppo_loop.finish(args, generation, entry, measure, checkpoint_payload, benchmark, log_path)
    finally:
        if ahead is not None:
            # A round still being played here is one the learning failed
            # beside: it is given up at its next step, not waited for.
            ahead.close()

    print("training finished", flush=True)


if __name__ == "__main__":
    main()
