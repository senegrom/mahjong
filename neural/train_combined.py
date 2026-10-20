"""Training the joined player: our network and a Mortal beneath one head.

The same self-play loop as `train.py`, for `combined.Combined`. Each
generation draws what stays fixed, evenly, from `--fixed`, by default the
six `Combined.MODES`: nothing, Mortal, ours, the head, or the head with
either network beneath it. Held still by turns, each part is moved in the
others' company without any being overwritten in one go. Three of the six
hold the head still. Only the critic and the reading of the hands, both
halves of each (`Combined.always_trained`), train whatever the draw.

The policy gradient is PPO's clipped objective on the head's output. The
baseline is the fusion's judge, our value head plus its correction from
Mortal's vector (`Fuse.judge`), as it stands before the round is learned;
the judge learns the return, and the reading what the opponents really
held.

The return is the default hybrid of every run so far unless `--reward
placement` asks for the placement reward, credited hand by hand (see
`place_model`), which a run starts from a U fitted on its own games and,
with `--value-warm-up`, a generation or more that teach the critic alone,
holding everything else still, the reading of the hands included.

    python -m neural.train_combined --ours ours.pt --mortal mortal.pt --rounds 20 --out runs/joined
"""

from __future__ import annotations

import argparse
from contextlib import closing
import copy
import functools
import json
import math
import time
import types
from pathlib import Path

import torch
from torch import nn

from . import block_convolution, combined, host, place_model, play_graphs, policy_inference, ppo_loop, selfplay
from .behavior import validate_exploration
from .checkpoints import atomic_save
from .observe import pad_rows
from .population import Population
from .ppo_control import PolicyDrift, StepGuard, add_training_controls, baseline_batch_size
from .training_batches import require_trainable_round, require_updates
from .training_safety import TRAINING_API_VERSION, benchmark_history
from .training_state import (
    capture_random_state, peak_gpu_gb, peak_rss_gb, restore_random_state, round_seed,
)


#: How many graphs one compiled forward may be given before the compiler
#: stops compiling it and runs the rest eagerly, unannounced. The learning
#: step's forward is compiled anew for every pattern of what requires a
#: gradient, which `Combined.set_mode` sets six ways (see
#: `compile_per_mode`) and `Combined.hold_all_but_value` a seventh, for the
#: placement reward's warm-up, and with Mortal's vectors given and not
#: (`--reuse-phi`): ten at most. The baseline pass's (`pass_forward`) is
#: compiled anew for each of the six modes, and shares its code with the
#: playing copy's own forward compiled for --compile-play: eight. The
#: compiler stops at eight; thirty-two leaves room for a variant nobody
#: foresaw.
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


def channels_last(module: nn.Module) -> int:
    """Has each one-dimensional convolution in `module` run, inside a
    compiled forward, as a two-dimensional one of 1x3 over a view of its
    input laid out channels-last, with a view of its own weight: the same
    convolution, in the layout the card's library works in, its sums
    perhaps taken in another order. Answers how many it changed.

    cuDNN works in NHWC on the card. Handed a one-dimensional
    convolution's input in NCL, it transposes it there and the answer back,
    around every convolution, forward and backward; and the compiler lays
    out only two-dimensional convolutions channels-last. So in the joined
    player's compiled learning step those transposes took 51 of the 256 ms
    an update kept an L40S busy in mode none. Over (N, C, 1, L) views kept
    channels-last there is nothing to transpose, and cuDNN picks its NHWC
    kernels.

    The weight is the convolution's own parameter, viewed with a dimension
    of one added, so the parameters, the state dict, a checkpoint and an
    optimiser see nothing new, and each gradient lands where it always did.
    Outside a compiled forward each convolution answers as it always did,
    bit for bit. A copy of `module` made before this keeps its own
    forwards, which is how the trainer leaves the playing copy, its graphs
    and its compiled forward as they were (see `main`). Only a convolution
    padded with zeros by a number of positions can be viewed so; any other
    is refused, and a subclass of Conv1d, whose forward may be its own, is
    left as it is."""
    changed = 0
    for conv in module.modules():
        if type(conv) is not nn.Conv1d:
            continue
        if conv.padding_mode != "zeros" or isinstance(conv.padding, str):
            raise ValueError(f"cannot lay out a convolution padded {conv.padding!r} with "
                             f"{conv.padding_mode} channels-last: {conv}")
        conv.forward = types.MethodType(_channels_last_forward, conv)
        changed += 1
    return changed


def _channels_last_forward(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """`conv` over `x`, as a 1x3 convolution over a channels-last view
    inside a compiled forward, and as itself everywhere else (see
    `channels_last`)."""
    if not torch.compiler.is_compiling():
        return nn.Conv1d.forward(conv, x)
    wide = x.unsqueeze(2).contiguous(memory_format=torch.channels_last)
    return nn.functional.conv2d(wide, conv.weight.unsqueeze(2), conv.bias, (1, conv.stride[0]),
                                (0, conv.padding[0]), (1, conv.dilation[0]), conv.groups).squeeze(2)


def learning_forward(net: combined.Combined, args: argparse.Namespace):
    """The forward the learning step calls, compiled as the switches ask,
    every mode with graphs of its own (see `guard_parameters`); the
    baseline pass calls its own (`pass_forward`).

    --compile compiles the joined player's forward for deciding as well,
    with the batch's size left symbolic, and keeps the compiler's own limit
    on graphs: past it, a variant runs eagerly, as it always did. Checking
    the parameters changes nothing else of it; this desktop's torch traced
    every mode anew already. --compile-learning compiles the learning step
    alone, at the one shape it is always handed: whole minibatches; the
    trainer has the convolutions of our network and of the head run
    channels-last first (`channels_last`), and those of Mortal's blocks
    work out their weight gradients as GEMMs
    (`block_convolution.gemm_weight_gradients`), once the playing copy is
    made."""
    if args.compile:
        guard_parameters()
        net.backbones_forward = torch.compile(net.backbones, dynamic=True)
        return torch.compile(net.everything)
    if args.compile_learning:
        compile_per_mode()
        return torch.compile(net.everything, dynamic=False)
    return net.everything


def pass_forward(net: combined.Combined, args: argparse.Namespace):
    """The forward the baseline pass calls, `Combined.decision`: the
    logits, the judge and Mortal's vector beneath them. Not `everything`,
    which the pass called before: that reads the opponents' hands as well,
    which the pass never reads, and leaves out Mortal's vector, which a
    generation that holds Mortal still learns from (`--reuse-phi`, see
    `value_round`). Compiled as the switches compile the learning step's
    forward, by `learning_forward`, which sets the compiler for both and
    is called first: --compile-learning at the one shape the pass is
    always handed, its chunks padded to the minibatches' size, every mode
    with a graph of its own. The logits and the value are `everything`'s
    layers on the same planes, from another graph, whose answers can
    differ from that one's in their last bits: on an L40S, from a005's
    gen-1240, the values were bit for bit those of `everything` compiled
    as the learning step's in every mode, the drift guard's probabilities
    97% so and the rest within 1e-6, and a chunk took 59.9 ms against
    60.8."""
    if args.compile:
        return torch.compile(net.decision)
    if args.compile_learning:
        return torch.compile(net.decision, dynamic=False)
    return net.decision


def value_round(passing, rollout: ppo_loop.Round, batch, rows: int, args: argparse.Namespace,
                dtype: torch.dtype, amp_enabled: bool, *, lagged: bool, keep=None):
    """The round's baseline: the fusion's judge, our value head plus its
    correction from Mortal's vector (`Fuse.judge`), as it stands before the
    round is learned, from a pass over the round through `passing` (see
    `pass_forward`), `rows` decisions at a time, or, asked to, from play
    (see `ppo_loop.baseline_of`). For a round played ahead (`lagged`), which
    always gets the pass, the pass also reads the probability the learner
    as it stands gives each move made, which the drift guard counts from
    (`begun`, see `ppo_epochs`).

    With `keep`, the pass keeps Mortal's vector of each decision as it
    works it out: in a generation that holds Mortal still, the vector the
    learning step would otherwise work out again from the same planes on
    every minibatch of every epoch. They are kept in a tensor that
    `keep(shape, dtype)` makes once the first chunk's show their width and
    precision (see `ppo_loop.kept_on_card`), or not at all where it answers
    None, having no room for them.

    Answers the baseline, the seconds it took, what the record says of it,
    `begun`, None for a round the learner played itself, and the vectors
    kept, None where none were. A function of its own, apart from `main`,
    so that a test can run the trainer's own pass on a card."""
    begun = torch.empty(rollout.decisions, device=rollout.device) if lagged else None
    # The vectors' tensor once the first chunk has asked for it, or None
    # where there was no room for it.
    kept: list[torch.Tensor | None] = []

    def values_of(chunk, planes):
        mask = pad_rows(rollout.legal[chunk], rows, True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            logits, value, phi = passing(planes, mask)
        taken = rollout.actions[chunk]
        if begun is not None:
            begun[chunk] = torch.distributions.Categorical(
                logits=logits[: len(taken)].float(), validate_args=False
            ).log_prob(taken)
        if keep is not None:
            if not kept:
                kept.append(keep((rollout.decisions, *phi.shape[1:]), phi.dtype))
            if kept[0] is not None:
                kept[0][chunk] = phi[: len(taken)]
        return (value,)

    guess, valued, said = ppo_loop.baseline_of(
        rollout, batch, rows, values_of,
        # Play's values are the head's as it stands only in a round the
        # learner played itself. One played ahead was valued by the head of
        # the generation before, and a generation's learning moves that
        # head far beyond the bfloat16 noise the switch was accepted for:
        # on this desktop, at fusion-long's rates from random weights, such
        # values stood 0.18 to 0.28 from the pass on average, where the
        # learner's own stood 0.0006 (a trained head's drift is
        # unmeasured). Still a baseline, since it does not depend on the
        # move made, but not the one the switch promised, so such a round
        # gets the pass; checked, the record says how far the head moved.
        from_play=args.baseline_from_play and not lagged,
        check=args.check_baseline, dtype=dtype,
    )
    return guess, valued, said, begun, (kept[0] if kept else None)


def optimiser_groups(net: combined.Combined, args) -> list[dict]:
    """The joined player's optimiser's three groups, as every checkpoint of
    this trainer saved its state in them: what the pre-test rebuilds to read
    a checkpoint's moments (`placement_pretest`) as well as the trainer."""
    return [
        # The head and the two things that always train share a rate; what a
        # generation holds still is decided by `set_mode`, not by leaving it
        # out here.
        {"params": net.head_trained() + net.always_trained(), "lr": args.lr},
        {"params": net.ours_trained(), "lr": args.lr_ours},
        {"params": net.mortal_trained(), "lr": args.lr_mortal, "weight_decay": 0.01},
    ]


def value_warm_up(net: combined.Combined, learn, rollout, held, optimiser, totals, args,
                  dtype: torch.dtype, amp_enabled: bool, played_phi,
                  clock: ppo_loop.LearningClock | None = None):
    """One generation's epochs that teach the critic alone (`--value-warm-up`):
    the value loss and nothing else, stepped by `optimiser`, which holds the
    critic's parameters alone (`Combined.value_parts`), everything else
    held still (`Combined.hold_all_but_value`). The critic reads features
    held back from it, so its gradient reaches nothing that chooses a move:
    the policy is the same after as before, bit for bit.

    A new reward changes the return the critic is asked for, and with it
    the baseline every advantage is measured from; at the run's own rates
    the critic would take many generations to catch up, its advantages
    wrong all the while. The policy's entropy and its drift are summed for
    the record, though nothing moves either. Answers the steps taken and the
    last minibatch's picks and planes, as the PPO epochs leave them; each
    step is noted on `clock` when one is given."""
    device = rollout.device
    nothing = torch.zeros((), device=device)
    steps = 0
    picks = planes = None
    for _epoch in range(args.epochs):
        with closing(ppo_loop.minibatches(rollout, args.batch, dtype=dtype)) as minibatches:
            for picks, planes in minibatches:
                optimiser.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    if played_phi is None:
                        logits, value, guessed = learn(planes, rollout.legal[picks])
                    else:
                        logits, value, guessed = learn(planes, rollout.legal[picks], played_phi[picks])
                value_loss = nn.functional.mse_loss(value.float(), rollout.returns[picks])
                (args.value_weight * value_loss).backward()
                grad_norm = nn.utils.clip_grad_norm_(net.value_parts(), 1.0, error_if_nonfinite=True)
                optimiser.step()
                with torch.no_grad():
                    distribution = torch.distributions.Categorical(logits=logits.float(), validate_args=False)
                    moved = distribution.log_prob(rollout.actions[picks]) - rollout.old_log_probs[picks]
                    hands_loss, covered = ppo_loop.hands_loss_of(guessed.float(), held[picks])
                    totals.add(policy=nothing, value=value_loss.detach(), entropy=distribution.entropy().mean(),
                               clipped=nothing, kl=-moved.mean(), grad=grad_norm, leash=nothing,
                               hands=hands_loss, covered=covered)
                steps += 1
                if clock is not None:
                    clock.update()
    return steps, picks, planes


class EntropyFloor:
    """The entropy bonus. Without --entropy-target it is --entropy, fixed.
    With it the bonus is exp(log_coef), never below the fixed --entropy and
    never above --entropy-max, and it is a dual variable -- it climbs while
    the policy's entropy is under the target and sinks back while it is
    over. Self-play here was measured to buy reward by getting louder: 30
    generations cut entropy by 29% and lost 0.0075. Its options were refused
    before anything was built where they leave it nothing to fall back to or
    no room to rise (see `validate_training_options`). The checkpoint keeps
    `log_coef`, where a resumed run starts it."""

    def __init__(self, args: argparse.Namespace, log_coef: float | None = None) -> None:
        self.target = args.entropy_target
        self.rate = args.entropy_rate
        self.fixed = args.entropy
        self.base = math.log(args.entropy) if args.entropy > 0 else float("-inf")
        self.ceiling = math.log(max(args.entropy_max, args.entropy)) if args.entropy > 0 else self.base
        self.log_coef = float(log_coef) if log_coef is not None else self.base

    @property
    def on(self) -> bool:
        """Whether the bonus moves, read from each update's entropy."""
        return self.target > 0

    def coef(self) -> float:
        return math.exp(self.log_coef) if self.on else self.fixed

    def step(self, entropy: torch.Tensor) -> None:
        """A dual step after an update whose policy had `entropy`: under the
        target the bonus climbs, over it the bonus sinks back to the fixed
        floor. The entropy is read on the host."""
        shortfall = self.target - float(entropy.detach())
        self.log_coef = min(self.ceiling, max(self.base, self.log_coef + self.rate * shortfall))


def ppo_epochs(net: combined.Combined, learn, rollout, held, optimiser, totals, drift: PolicyDrift, args,
               dtype: torch.dtype, amp_enabled: bool, played_phi, *, advantages: torch.Tensor,
               begun: torch.Tensor | None, explored: torch.Tensor, floor: EntropyFloor,
               reference: combined.Combined | None = None, clock: ppo_loop.LearningClock | None = None):
    """A generation's PPO epochs: an optimiser update a minibatch, of the
    clipped objective on the head's output over the moves the policy chose,
    the critic's and the reading's losses over every row, the entropy bonus
    (`floor`, stepped after each update when it moves) and, with
    `reference`, the leash; until the epochs are done or the drift guard
    (`drift`) stops them. `advantages` are the round's, standardised;
    `begun`, for a round played ahead, the probability the learner gave each
    move made as the generation began, which the guard counts from, and None
    for a round the learner played itself; `explored`, the rows whose move
    exploration forced. A function of its own, apart from `main`, so that a
    test can run the trainer's own epochs on a card, round after round.

    Whether each update is taken, as the guard and the gradient clip say,
    is decided on the card where the optimiser can refuse an update there,
    and read by the host a couple of updates later: that spares the two
    waits on the card an update that deciding on the host costs, and
    changes nothing learned (see `StepGuard`). Answers the updates taken
    and the last minibatch's picks and planes, as `value_warm_up` does,
    but with planes None where the learning stopped on a minibatch launched
    before the last, whose planes are gone; each update is noted on `clock`
    when one is given."""
    device = rollout.device
    # The entropy floor reads each update's entropy on the host, which waits
    # for the card anyway, and its step is not to be taken after an update
    # the guard refuses: with it, every verdict is had at once.
    guard = StepGuard(drift, optimiser, device, clock=clock, late=not floor.on)
    picks = planes = None
    for _epoch in range(args.epochs):
        with closing(ppo_loop.minibatches(rollout, args.batch, dtype=dtype)) as minibatches:
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
                    stop = bool(chosen.any()) and guard.check(since[chosen], log_prob[chosen])
                else:
                    # Nothing was explored, so every row is the policy's;
                    # asking the card which ones would wait on it.
                    stop = guard.check(since, log_prob)
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
                    - floor.coef() * entropy
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
                grad_norm = guard.clip([p for p in net.parameters() if p.requires_grad], 1.0)
                guard.step()
                if floor.on:
                    floor.step(entropy)
                with torch.no_grad():
                    totals.add(
                        unless=guard.refused,
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
                guard.launched(picks)
                if drift.stopped:
                    break
            # Every verdict is read before the next epoch draws its order from
            # torch's generator, which the checkpoint keeps: drawn after the
            # learning had stopped, it would leave the generator further on
            # than deciding at once does.
            guard.settle()
        if drift.stopped:
            break
    if guard.ended is not None:
        picks, planes = guard.ended, None
    return guard.updates, picks, planes


def placement_fields(place, ledger, returns: torch.Tensor, guess: torch.Tensor, warming: bool) -> dict:
    """What a generation's record says of the placement reward: the alpha it
    paid, whether it taught the critic alone, and the return's spread and
    the critic's error over the East round's decisions and the South's
    apart. An East decision's return carries less of the game's luck than
    it did, which is where the reward should show first."""
    stages = torch.from_numpy(ledger.stages[ledger.rows_of(ledger.games, ledger.hands)]).to(returns.device)
    fields = {"reward": "placement", "alpha": place.reward.alpha, "value_warm_up": warming}
    for name, rows in (("east", stages < 4), ("south", stages >= 4)):
        if rows.any():
            fields[f"return_variance_{name}"] = round(float(returns[rows].var(unbiased=False)), 4)
            fields[f"value_error_{name}"] = round(float(((returns[rows] - guess[rows]) ** 2).mean()), 4)
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, default=None, help="our checkpoint to start from")
    parser.add_argument("--mortal", type=Path, default=None, help="the Mortal to start from")
    parser.add_argument("--resume", type=Path, default=None, help="a checkpoint this trainer wrote")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="for the head, the critic and the reading of the hands")
    parser.add_argument("--lr-ours", type=float, default=4e-5,
                        help="for our network beneath, but for its critic and its reading of the hands (--lr)")
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
        "the site shows of the other hands",
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
        "vector of each decision as the pass that values the round worked it out, kept on the "
        "card, instead of running Mortal again on every minibatch. Nothing moves Mortal in such "
        "a generation, so it is the same function of the same planes; only the batches and the "
        "graphs differ, which in bfloat16 can move it in its last bits. Off by default; a card "
        "without room for it, about two kilobytes a decision, works it out as before, and so "
        "does a generation that makes no pass (--baseline-from-play)",
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
        "itself, a process's first, and makes the pass for the rest. Off by default; not with "
        "--compile",
    )
    parser.add_argument(
        "--play-graphs", action="store_true",
        help="answer play's small questions, of the joined player with --amp and of the seated "
        "--opponents, from CUDA graphs recorded once for each of a few sizes of batch, a question "
        "padded to the next of them, rather than launching a forward's thousand or two kernels "
        "one by one each time; a batch above 512 rows is asked eagerly as before. A graph "
        "answers what the forward answers eagerly at the padded size, bit for bit; that can "
        "differ from its answer at the size asked in the last bits of bfloat16. Off by "
        "default; not with --compile",
    )
    parser.add_argument(
        "--reward", choices=("hybrid", "placement"), default="hybrid",
        help="what a decision is paid. hybrid, as every run before: its hand's points over "
        "4,000 plus the game's placement bonus. placement: U, the bonus the scoreboard promises "
        "at the start of the next hand, plus --alpha times the hand's points over 4,000, and "
        "the bonus itself after a game's last hand (see neural.place_model)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0,
        help="with --reward placement, what the hand's points weigh: 0 pays placement alone, "
        "0.05 lets points only break ties, sixty thousand of them being worth less than a place",
    )
    parser.add_argument(
        "--place-model", type=Path, default=None,
        help="the U a run starting --reward placement settles by (python -m neural.place_model "
        "fit, or the pre-test's); a run that resumes settles by the one its checkpoint carries",
    )
    parser.add_argument(
        "--place-refit-every", type=int, default=0,
        help="with --reward placement, fit U again every this many generations, on the run's "
        "last --place-refit-rounds rounds, keeping the new one only if it reads an eighth of "
        "their games, held out, at least as well; nought never",
    )
    parser.add_argument(
        "--place-refit-rounds", type=int, default=4,
        help="how many of the run's last rounds a refit of U reads. The Club sits out about one "
        "round in four, and a refit on rounds that seated none of some kind of player keeps "
        "the U it has: two rounds both lack the Club one time in sixteen, four one in 256",
    )
    parser.add_argument(
        "--value-warm-up", type=int, default=0,
        help="with --reward placement, the first this many generations of the reward teach the "
        "critic alone, at --value-warm-up-lr in an optimiser of its own, everything else held "
        "still: the critic reads features held back from it, so no policy can move",
    )
    parser.add_argument("--value-warm-up-lr", type=float, default=1e-4,
                        help="the critic's rate in a --value-warm-up generation")
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
        "dense in bfloat16, as without compiling. Fuses the step's elementwise kernels, "
        "runs our network's and the head's convolutions channels-last, in the step and the "
        "pass, and works out the weight gradients of Mortal's block convolutions as GEMMs; "
        "the numbers move by where bfloat16 rounds. Off by default; not with --compile",
    )
    parser.add_argument(
        "--compile-play", action="store_true",
        help="answer play's questions bigger than its graphs (--play-graphs) from forwards "
        "compiled with the batch's size left symbolic, the playing copy's and the seated "
        "others', compiled before the first round: on an L40S about half the card's time of "
        "the eager forwards. They answer in other last bits. Needs --play-graphs and --amp",
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
    if args.compile and args.play_graphs:
        # --compile plays with compiled forwards, which the graphs would
        # have to record, and the compiler's own graphs are another matter.
        parser.error("--play-graphs records play's eager forwards; with it, compile the "
                     "learning step alone (--compile-learning)")
    if args.compile_play and not (args.play_graphs and args.amp):
        # The graphs keep the small questions from the compiled forwards: one
        # of a single row would have them compiled again, where play asks.
        parser.error("--compile-play compiles play's bigger questions beside the graphs that answer "
                     "its small ones, in bfloat16: it needs --play-graphs and --amp")
    check_placement_options(parser, args)
    return args


def check_placement_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuses the placement reward's options where they would be ignored
    or make no sense, before anything is built: an option of the placement
    reward given to a run paid the default one would leave it training on a
    reward it was not asked for, unannounced."""
    if not math.isfinite(args.alpha) or args.alpha < 0:
        parser.error("--alpha must be finite and nonnegative")
    for name in ("place_refit_every", "value_warm_up"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.place_refit_rounds < 1:
        parser.error("--place-refit-rounds must be at least one")
    if not math.isfinite(args.value_warm_up_lr) or args.value_warm_up_lr <= 0:
        parser.error("--value-warm-up-lr must be finite and positive")
    if args.reward == "hybrid":
        asked = [flag for flag, value in (
            ("--alpha", args.alpha), ("--place-model", args.place_model),
            ("--place-refit-every", args.place_refit_every), ("--value-warm-up", args.value_warm_up),
        ) if value]
        # These two have defaults of their own, so only a value other than the
        # default was asked for.
        asked += [f"--{name.replace('_', '-')}" for name in ("place_refit_rounds", "value_warm_up_lr")
                  if getattr(args, name) != parser.get_default(name)]
        if asked:
            parser.error(f"{', '.join(asked)} only with --reward placement")
    if args.place_model is not None and not args.place_model.is_file():
        parser.error(f"--place-model {args.place_model} does not exist")


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
    # What the checkpoint says of the placement reward, when it was trained
    # on it: its record and its U (see `place_model.PlacementRun`).
    carried: dict = {}
    if args.resume is not None and args.resume.exists():
        net, payload = combined.load(args.resume, device)
        start = int(payload.get("generation", 0))
        benchmark = ppo_loop.Benchmark(*benchmark_history(payload))
        optimiser_state = payload.get("optimizer_state")
        random_state = payload.get("random_state")
        entropy_log_coef = payload.get("entropy_log_coef")
        carried = {name: payload[name] for name in ("reward", "place_model") if name in payload}
        print(f"resumed from {args.resume} at generation {start}", flush=True)
    elif args.ours is not None and args.mortal is not None:
        net = combined.build(args.ours, args.mortal, device)
        print(f"joined {args.ours} and {args.mortal}", flush=True)
    else:
        raise SystemExit("give --ours and --mortal to start, or --resume")

    # The placement reward, when asked for (see `place_model.PlacementRun`).
    # A run trained on it is never resumed on the default reward by a
    # command line that forgot to say so.
    place = None
    if args.reward == "placement":
        place = place_model.PlacementRun.begin(
            args, carried, start, [member.name for member in Population.from_paths(args.opponents).members])
    elif carried.get("reward", {}).get("kind", "hybrid") != "hybrid":
        raise SystemExit(
            f"{args.resume} was trained on the {carried['reward']['kind']} reward at alpha "
            f"{carried['reward'].get('alpha')}; resume it with --reward {carried['reward']['kind']} "
            "--alpha as it was, or start a new run")

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
        # The network alone: the checkpoint it was read from is a copy of
        # every weight, and kept it stayed on the host for the whole run.
        reference = combined.load(kept, device)[0]
        reference.eval()
        reference.requires_grad_(False)
        print(f"leashed to {kept} at {args.leash} a nat", flush=True)

    # One optimiser over everything, in three groups; a group whose
    # parameters are fixed this generation gets no gradients and is left
    # alone by it.
    net.set_mode("none")
    optimiser = torch.optim.AdamW(
        optimiser_groups(net, args),
        lr=args.lr,
        weight_decay=1e-4,
        fused=device == "cuda",
    )
    ppo_loop.restore_optimiser(
        optimiser, optimiser_state,
        migrate=lambda saved: combined.migrate_belief_optimizer(saved, optimiser, net),
    )
    # Nothing reads the checkpoint from here on: its weights and Adam's
    # moments are in the network and the optimiser now, and kept they were a
    # third of a gigabyte of the host's memory held through every generation.
    payload = optimiser_state = None

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
    if args.play_graphs:
        # Small questions from CUDA graphs, every one recorded now, when play
        # asks in bfloat16 (--amp), and with --compile-play the bigger ones
        # from the forward compiled now too. The copy is given the learner's weights
        # where it holds them before each round, and the learner's own are
        # moved where they are by the optimiser, so the graphs answer by the
        # weights of the moment.
        play_graphs.graphed(net if actor is None else actor, device, amp=amp_enabled,
                            compile=args.compile_play)

    if args.compile_learning:
        # The learner's compiled forwards, the learning step's and the
        # pass's, run the convolutions of our network and of the head
        # channels-last, where the card's library does not transpose around
        # each of them (see `channels_last`). Mortal's stay one-dimensional.
        # Converted as well, on an L40S from a005's gen-1240, they took
        # another 7 ms off an update in each mode that trains Mortal and 5
        # ms off a chunk of the pass; but in those modes they moved the
        # step's logits from the one-dimensional step's by 0.023 on average
        # and 0.30 at most, where ours and the head alone moved them by
        # 0.0015 and 0.04, and the change was to stay within 0.15.
        # Only now that the playing copy is made: a copy made after would
        # carry the same forwards into its compiled one, while the copy, its
        # graphs and its compiled forward, the seated others and whatever is
        # exported are to answer as they always did. Outside the compiled
        # forwards the learner answers as it always did too, in a
        # measurement's games and the record's head_shift.
        channels_last(net.ours)
        channels_last(net.fuse)
        # The convolutions of Mortal's blocks, one-dimensional still, work
        # out their weight gradients in the learning step as GEMMs that
        # cuBLAS spreads over the card, where cuDNN's kernel for their
        # shape ran on six of its 142 SMs (see `block_convolution`): the
        # same sums in another order, their forwards and input gradients
        # cuDNN's own. On an L40S from a005's gen-1240 an update took 33
        # ms less in each mode that trains Mortal, and the step's answers
        # and every other gradient were bit for bit as before in all six
        # modes. Only now as well, for the same reasons; and only where
        # the step trains Mortal, so the pass and the generations that
        # hold Mortal still keep the graphs they had.
        block_convolution.gemm_weight_gradients(net.mortal)
    learn = learning_forward(net, args)
    passing = pass_forward(net, args)

    # Who else sits at the tables, as a roster with roles and shares: see
    # `neural.population`. The checkpoints are the ones `--opponents`
    # names, which is how every launcher passes them.
    roster, seated = ppo_loop.seat_others(args, device)
    # Play's forwards that answer from graphs, the learner's and the seated
    # others', for the record.
    graphed = play_graphs.graphs_of([net if actor is None else actor, *seated])
    print(
        f"device {device} | ours {net.ours.channels}x{net.ours.blocks} | Mortal beneath | "
        f"{net.parameter_count() / 1e6:.2f}M parameters | fixed by turns: {args.fixed}"
        + (" | each round played ahead" if args.play_ahead else "")
        + (f" | {len(graphed)} players' small questions from graphs: "
           f"{json.dumps(play_graphs.accounts(graphed))}" if graphed else ""),
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
            "entropy_log_coef": floor.log_coef,
            # The placement reward's record and the U it settles by, in a run
            # paid by it; a run paid the default carries nothing new.
            **(place.fields() if place is not None else {}),
        }

    def measure(generation: int) -> dict:
        net.eval()
        return selfplay.measure(
            net, games=args.measure_games, seed=7_000_000 + generation, device=device
        )

    # The entropy bonus, from where the checkpoint left it (see `EntropyFloor`).
    floor = EntropyFloor(args, entropy_log_coef)

    def play(player, generation: int, **options):
        """Round `generation`, played by `player`: see `selfplay.play`,
        which `options` are handed on to. A round of a run paid the
        placement reward keeps the ledger it is settled from."""
        if place is not None:
            options["ledger"] = True
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
    # The critic's own optimiser for a --value-warm-up generation, made when
    # the first such generation begins.
    warm_optimiser = None
    try:
        for generation in range(start, end):
            began = time.time()
            # The processors' time this process uses over the generation,
            # and the time the container is held back meanwhile, for the
            # record (see `host`).
            used = host.cpu_seconds()
            held_back = host.counters()
            batch = rollout = held = played_phi = None
            # Deciding is the joined player as it stands; what stays fixed in
            # the update that follows is drawn now and said in the record.
            fixed = str(drawer.choice(args.fixed))
            net.set_mode(fixed)
            net.eval()
            # A generation that holds Mortal still has nothing to move it, its
            # weights or its statistics, so its vector of every decision is
            # the same all generation: kept from the pass that values the
            # round, when asked to reuse it, for the learning step to read.
            reusing = args.reuse_phi and "mortal" in fixed.split("+")
            # Whether the round was played beside the last generation's
            # learning, by the weights from before it (--play-ahead).
            lagged = False
            if ahead is None:
                player = net if actor is None else actor
                if actor is not None:
                    # Copied with the same rounding autocast makes.
                    actor.load_state_dict(net.state_dict())
                batch = play(player, generation)
                played = time.time() - began
            else:
                lagged = ahead.playing is not None
                if not lagged:
                    # A process's first round: nothing was played beside a
                    # generation before, so the learner plays it now.
                    ahead.start(generation)
                batch, played, waited = ahead.take(generation)
                if generation + 1 < end:
                    # The next round, played while this one is learned, by the
                    # weights the learner has now; never one past the last
                    # generation, which nothing would learn.
                    ahead.start(generation + 1)
            # The learning begins: with --play-ahead, beside the next round
            # until that round is done (see `ppo_loop.LearningClock`).
            learning = ppo_loop.LearningClock()
            require_trainable_round(batch.decisions, args.batch, args.epochs)
            # A run paid the placement reward settles the round now, by the U
            # it has as the round is learned, and keeps the round's hand
            # starts for the next refit of U.
            warming = False
            if place is not None:
                selfplay.settle_placement(batch, place.reward)
                place.remember(batch.ledger, generation)
                warming = place.warming
            loading = time.time()
            # Beside a round being played the planes stay on the host: the
            # card holds that round's work as well, and keeping a round on
            # the card was never seen to shorten its learning.
            rollout = ppo_loop.on_device(batch, device, card=ahead is None)
            # What the three opponents were really holding at each decision: the
            # label the reading of the hands is trained against, which self-play
            # knows for free and which is far denser than the game's result.
            held = batch.held.to(device)
            loaded = time.time() - loading

            # The baseline: the fusion's judge as it stands before the round
            # is learned, from a pass over the round or, asked to, from play
            # (see `value_round`). For a round played ahead, the pass also
            # reads the probability the learner as it stands gives each move
            # made (`begun`), which the drift guard counts from (see
            # `ppo_epochs`).
            rows = baseline_batch_size(batch.decisions, args.batch, args.baseline_batch)
            guess, valued, baseline_said, begun, played_phi = value_round(
                passing, rollout, batch, rows, args, planes_dtype, amp_enabled, lagged=lagged,
                # Mortal's vectors on the card beside the step, or not at
                # all: gathered from the host every minibatch they would
                # hold the step up. Beside a round being played, only with
                # that round's room to spare as well.
                keep=functools.partial(ppo_loop.kept_on_card, device=device,
                                       spare=None if ahead is None else ahead.spare())
                if reusing else None,
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
            if warming:
                # A generation that teaches the critic alone the placement
                # reward's return, in an optimiser of its own and at a rate of
                # its own, everything else held still (see `value_warm_up`);
                # the PPO epochs below are not run.
                net.hold_all_but_value()
                if warm_optimiser is None:
                    warm_optimiser = torch.optim.AdamW(
                        net.value_parts(), lr=args.value_warm_up_lr, weight_decay=1e-4,
                        fused=device == "cuda",
                    )
                steps, picks, planes = value_warm_up(
                    net, learn, rollout, held, warm_optimiser, totals, args,
                    planes_dtype, amp_enabled, played_phi, clock=learning,
                )
            else:
                steps, picks, planes = ppo_epochs(
                    net, learn, rollout, held, optimiser, totals, drift, args, planes_dtype, amp_enabled,
                    played_phi, advantages=advantages, begun=begun, explored=explored, floor=floor,
                    reference=reference, clock=learning,
                )

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
                    if planes is None or planes.dtype != torch.float32:
                        # This runs without autocast, so the minibatch is made
                        # dense again as it always was; and the minibatch the
                        # learning stopped on, read late, is made again, its
                        # planes being gone (see `ppo_epochs`).
                        planes = ppo_loop.planes_of(rollout, picks)
                    phi, q, features, a1, _value, _guessed = net.backbones(planes, allowed)
                    joined = net.fuse(phi.float(), q.float(), features.float(), a1.float(), allowed)
                    shift = (joined - a1).abs().masked_fill(~allowed, 0.0)
                    head_shift = float(shift.sum() / allowed.sum().clamp(min=1))
                net.train()

            require_updates(steps)
            # What the log says of the placement reward, in a run paid by it:
            # the warm-up counted once its generation has learned, and U
            # fitted again when its time has come, before the checkpoint that
            # carries it is saved.
            placed = {}
            if place is not None:
                if warming:
                    place.warmed += 1
                placed = placement_fields(place, batch.ledger, rollout.returns, guess, warming)
                if place.refit_due(generation):
                    placed["place_model_refit"] = place.refit(generation, device, seed=args.seed)
            entry = ppo_loop.record(
                generation, steps, drift, rows, rollout, batch,
                {"began": began, "played": played, "loaded": loaded}, totals, spread,
                **placed,
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
                # the pass worked them out, when the run reuses them.
                **({"reused_phi": played_phi is not None} if args.reuse_phi else {}),
                # With --play-ahead, whether the round was played beside the
                # last generation's learning, by the weights from before it,
                # and how long this generation waited for it: the part of
                # play_seconds, the round's own, that the learning did not hide.
                **({"played_ahead": lagged, "play_wait_seconds": round(waited, 1)}
                   if ahead is not None else {}),
                # With --play-graphs, how many graphs play holds by now, the
                # seconds this process has spent recording them, and the
                # card memory they hold.
                **(play_graphs.accounts(graphed) if args.play_graphs else {}),
                # What bounds the generation: the card and the processors it
                # had, the processors' time it used and any the container was
                # denied, the seated others' encoding (the worker's share
                # hidden by the learner's deciding, so not a part of
                # play_split), and with --play-ahead how much of the learning
                # went on beside the next round and how much after it was
                # done, in seconds and in updates.
                card=host.card(device),
                host=host.processors(),
                cpu_seconds=round(host.cpu_seconds() - used, 1),
                throttling=host.change(held_back, host.counters()),
                seated_encode_seconds=round(getattr(batch, "seated_encode_seconds", 0.0), 1),
                **(learning.split(ahead.ended.get(generation + 1) if generation + 1 < end
                                  else learning.began)
                   if ahead is not None else {}),
                entropy_coef=round(floor.coef(), 6),
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
