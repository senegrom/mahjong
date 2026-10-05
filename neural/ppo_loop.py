"""What the three self-play trainers share.

`train` (our network), `train_mortal` (a published Mortal fine-tuned on
our rules) and `train_combined` (the two beneath one head) run one loop:
play a round, value every decision with the value head as it stood before
the round, take PPO's clipped steps on whole minibatches of it, then
measure, save and log. What each keeps to itself stays in its own module:
train's choice among three baselines, its distillation and its pass over
the replay ring; train_combined's forced moves kept out of the policy
gradient, its leash and its entropy floor; train_mortal's temperature.
Everything else is here, so that a fix to one trainer is a fix to all
three.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from . import population, zoo
from .checkpoints import atomic_save
from .observe import DevicePlanes, Planes, pad_rows, resident
from .prefetch import Prefetcher
from .training_batches import validate_learning
from .training_safety import (
    TRAINING_API_VERSION, require_training_engine, validate_training_options,
)

# How much of a new measurement goes into the smoothed figure the best
# checkpoint is chosen on. A third means roughly the last three count, which
# cuts the noise about in half without lagging far behind a real gain.
SMOOTHING = 1 / 3


def setup(args, check=None) -> tuple[str, bool, Path]:
    """Refuses a bad run before anything is built or written (`check`, when
    given, after the learning options), then picks the device, makes the
    run's directory and seeds torch. Returns the device, whether mixed
    precision is on, and where the log goes.

    `--seed` used to seed only the rules engine and replay sampler, so model
    initialisation, sampled moves and minibatch shuffles changed between
    nominally identical runs. Torch is seeded too, and a resumed run
    restores the states its checkpoint saved."""
    validate_learning(args.batch, args.epochs)
    if check is not None:
        check(args)
    validate_training_options(args)
    require_training_engine()
    # The environment runs on this thread and the network on the GPU, so a
    # couple of worker threads is plenty and leaves the machine usable.
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = args.amp and device == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    return device, amp_enabled, args.out / "log.jsonl"


def load_others(args, device: str) -> list:
    """The players `--opponents` names, each loaded once, at whatever shape
    and of whatever kind its checkpoint says. They are only ever asked for
    a move, so they need no gradients, and those that play on the card in
    bfloat16 hold their weights that way (see `zoo.play_in_bfloat16`).
    With `--skip-forced` they leave the rows with one move open unasked
    (see `zoo.choose_in_mortal_space`); without it, or in a trainer that
    has no such option, they ask about every row, as they always did.
    validate_training_options has refused a missing checkpoint already."""
    seated = []
    for path in args.opponents:
        other = zoo.load_player(path, device, compile=args.compile)
        other.eval()
        for parameter in getattr(other, "parameters", list)():
            parameter.requires_grad_(False)
        if getattr(args, "skip_forced", False) and hasattr(other, "skip_forced"):
            other.skip_forced = True
        seated.append(zoo.play_in_bfloat16(other, device))
    return seated


def seat_others(args, device: str) -> tuple[population.Population, list]:
    """The players `--opponents` names and the roster they make (see
    `neural.population`), which seats them by its weights: a name the
    roster knows, as the cloud stages `zoo/mortal_298k` at
    `.../zoo--mortal_298k.pt`, keeps its role and its weight there, so the
    Mortal-space trainers draw their others alike, and a round can say how
    the learner placed against each of them. The roster and the players
    line up one for one."""
    roster = population.Population.from_paths(args.opponents)
    seated = load_others(args, device)
    if seated:
        where = (f"{args.seat_share:.0%} of players" if args.seat_share
                 else f"{args.opponent_share:.0%} of games")
        print(f"{len(seated)} others seated in {where}: " + json.dumps(roster.describe()), flush=True)
    return roster, seated


@dataclass
class Round:
    """A round of self-play where the learning step can read it: the small
    fields on the device, and the planes there whole when it has room for
    them (`on_card`), gathered there; otherwise on the host, sparse, and
    made dense on the device a minibatch at a time, a few steps ahead.

    A round of planes dense would be tens of gigabytes, and a minibatch's
    entries are a few tens of megabytes that cross in a few milliseconds.
    They are not pinned: pinning copies the round into page-locked memory,
    which once took the run from twelve gigabytes of host memory to
    twenty-eight and the machine to none."""

    decisions: int
    device: str
    observations: Planes
    legal: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    old_log_probs: torch.Tensor
    on_card: DevicePlanes | None


def on_device(batch, device: str) -> Round:
    """`batch`, a round from `selfplay.play`, where the learning step reads it."""
    return Round(
        decisions=batch.decisions,
        device=device,
        observations=batch.observations,
        legal=batch.legal.to(device),
        actions=batch.actions.to(device),
        returns=batch.returns.to(device),
        old_log_probs=batch.log_probs.to(device),
        on_card=resident(batch.observations, device),
    )


@torch.no_grad()
def baseline(
    rollout: Round, rows: int, value, heads: int = 1, dtype: torch.dtype = torch.float32
) -> list[torch.Tensor]:
    """What the value heads made of every decision before any of them
    learned the round, so the advantages cannot collapse as a head fits the
    very targets they are measured against.

    The round goes through in chunks of `rows`: `value(chunk, planes)`
    answers one value a row for each of `heads` heads, for the decisions
    `chunk` names, with their planes padded to `rows`, so a compiled graph
    sees one shape all round. The last chunk was a new shape every
    generation, which had the compiler rebuild the graph now and then:
    minutes each time. The planes are float32, or `dtype` for a `value`
    that reads them only under autocast (see `Planes.dense`).

    This is the one pass over the whole round, and it checks every row of
    planes kept on the host in full as it makes it dense. The round is
    trusted from then on (see `Planes`): the minibatches that follow, each
    row twice over, check only that every column is in bounds."""
    guesses = [torch.empty(rollout.decisions, device=rollout.device) for _ in range(heads)]
    for start in range(0, rollout.decisions, rows):
        chunk = slice(start, start + rows)
        if rollout.on_card is not None:
            planes = rollout.on_card.slice(start, start + rows, dtype)
        else:
            planes = rollout.observations.slice(start, start + rows).dense(rollout.device, dtype)
        count = planes.shape[0]
        answers = value(chunk, pad_rows(planes, rows))
        for guess, answer in zip(guesses, answers):
            guess[chunk] = answer.float()[:count]
    observations = rollout.observations
    if isinstance(observations, Planes) and not observations.trusted:
        rollout.observations = Planes(
            observations.indptr, observations.indices, observations.values, trusted=True
        )
    return guesses


def standardised(returns: torch.Tensor, baseline: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Each decision's advantage over the baseline, standardised over the
    round, and how widely they spread before."""
    advantages = returns - baseline
    spread = advantages.std(unbiased=False)
    return (advantages - advantages.mean()) / (spread + 1e-6), float(spread)


_STAGING: dict[torch.device, torch.cuda.Stream] = {}


def staging_stream(device: str | torch.device) -> torch.cuda.Stream:
    """The stream minibatches are made dense on, one a device for the life
    of the process. The caching allocator keeps freed memory a stream at a
    time, and a fresh stream each epoch left the last one's minibatches'
    memory where no other stream could use it, over a gigabyte an epoch,
    until the card was full and every allocation stalled to free it."""
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device not in _STAGING:
        _STAGING[device] = torch.cuda.Stream(device)
    return _STAGING[device]


def minibatches(rollout: Round, size: int, extra=None, dtype: torch.dtype = torch.float32):
    """One pass over the round in a fresh order, as (picks, planes, ...)
    per minibatch, the rows `extra(index)` gathers for them last. The
    planes are float32, or `dtype` for a step that reads them only under
    autocast (see `Planes.dense`).

    The shuffle is made on the host, where the planes may be: a permutation
    made on the card had every minibatch's indices copied back before they
    could be gathered, one synchronisation a step. Only whole minibatches
    are drawn: the compiled step is built for one shape, and a remainder of
    a new size each generation had it rebuilt now and then, minutes each
    time; the few rows left over differ every epoch. Planes on the host are
    made dense on the device in a thread of their own, a few minibatches
    ahead of the step. Close what this returns when done with it.

    On the card, that thread copies from page-locked memory and makes the
    planes dense on a stream of its own, and the step's stream waits for
    each minibatch only as it takes it. Copied from ordinary memory, on the
    step's stream, every copy waited until the card had finished all the
    work queued before it, so the thread fell behind the step it was meant
    to run ahead of, and the step's kernels waited behind its copies."""
    order = torch.randperm(rollout.decisions)
    slices = [order[start : start + size] for start in range(0, rollout.decisions, size)]
    slices = [drawn for drawn in slices if drawn.numel() == size]
    more = extra if extra is not None else (lambda _index: ())
    if rollout.on_card is not None:
        def gather(drawn: torch.Tensor):
            picks = drawn.to(rollout.device)
            return (picks, rollout.on_card.rows(picks, dtype), *more(picks))

        return (gather(drawn) for drawn in slices)

    if torch.device(rollout.device).type != "cuda":
        def prepare(drawn: torch.Tensor):
            return (
                drawn.to(rollout.device),
                rollout.observations.rows(drawn.numpy()).dense(rollout.device, dtype),
                *more(drawn),
            )

        return Prefetcher(slices, prepare)

    side = staging_stream(rollout.device)
    step = torch.cuda.current_stream(rollout.device)

    def staged(drawn: torch.Tensor):
        rows = rollout.observations.rows(drawn.numpy())
        with torch.cuda.stream(side):
            picks = drawn.pin_memory().to(rollout.device, non_blocking=True)
            planes = rows.dense(rollout.device, dtype, pinned=True)
            ready = torch.cuda.Event()
            ready.record(side)
        # Made on the side stream and read on the step's: their memory is
        # not handed out again until the step has finished with them.
        picks.record_stream(step)
        planes.record_stream(step)
        return ready, picks, planes, *more(drawn)

    def receive(prepared):
        ready, *taken = prepared
        torch.cuda.current_stream(rollout.device).wait_event(ready)
        return tuple(taken)

    return Prefetcher(slices, staged, receive=receive)


def planes_of(rollout: Round, picks: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The planes of the decisions `picks` names, dense on the device."""
    if rollout.on_card is not None:
        return rollout.on_card.rows(picks, dtype)
    return rollout.observations.rows(picks.cpu().numpy()).dense(rollout.device, dtype)


def clipped_policy_loss(
    log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantage: torch.Tensor, clip: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO's clipped objective: an update may improve an action's odds, but
    only so far in one round, which is what keeps a policy from narrowing
    onto a single action. Returns the loss, the ratio and the ratio clipped."""
    ratio = torch.exp(log_prob - old_log_prob)
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    return -torch.min(ratio * advantage, clipped * advantage).mean(), ratio, clipped


def hands_loss_of(guessed: torch.Tensor, wanted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy against the distribution each opponent's hand actually
    was, over the 34 kinds, and how much of the hand the guess covers,
    which is readable where a cross-entropy is not. Positions where nobody
    was holding anything carry rows of zeros and are skipped."""
    holding = wanted.sum(dim=2) > 0
    log_guess = torch.log_softmax(guessed, dim=2)
    loss = -(wanted * log_guess).sum(dim=2)
    loss = (loss * holding).sum() / holding.sum().clamp(min=1)
    with torch.no_grad():
        overlap = torch.minimum(log_guess.exp(), wanted).sum(dim=2)
        covered = (overlap * holding).sum() / holding.sum().clamp(min=1)
    return loss, covered


class Totals:
    """A generation's sums, kept on the device until it is over. Reading one
    synchronises the host with the device, and the old loop did so around
    a dozen times a minibatch merely to print one line at the end. Add
    under `torch.no_grad()`."""

    #: What every trainer sums: the losses, how often the ratio was
    #: clipped, the approximate KL and the gradient's norm.
    COMMON = ("policy", "value", "entropy", "clipped", "kl", "grad")

    def __init__(self, device: str, *names: str) -> None:
        self.sums = {name: torch.zeros((), device=device) for name in (*self.COMMON, *names)}

    def add(self, **values: torch.Tensor) -> None:
        for name, value in values.items():
            self.sums[name] += value

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.sums[name]

    def mean(self, name: str, count: int, digits: int) -> float:
        return round(float(self.sums[name] / count), digits)


def record(
    generation: int,
    updates: int,
    drift,
    rows: int,
    rollout: Round,
    batch,
    timing: dict[str, float],
    totals: Totals,
    spread: float,
    **fields,
) -> dict:
    """A generation's record: what every trainer says, and then `fields`.

    `timing` holds when the generation `began` and how long it `played`
    and `loaded`, in seconds. `updates` is how many optimiser steps it
    took, which every mean is over."""
    return {
        **drift.metrics(),
        "baseline_batch": rows,
        "generation": generation,
        "training_api_version": TRAINING_API_VERSION,
        "checkpoint_generation": generation + 1,
        "optimizer_updates": updates,
        "decisions": rollout.decisions,
        "hands": batch.hands,
        "seconds": round(time.time() - timing["began"], 1),
        "play_seconds": round(timing["played"], 1),
        # Where the play went: the engine and follower, the encoder, the
        # network, the seated others, and the bookkeeping.
        "play_split": {name: round(value, 1) for name, value in batch.timing.items()},
        # Moving the round to the card, and whether it went there whole.
        "load_seconds": round(timing["loaded"], 1),
        "resident": rollout.on_card is not None,
        "policy_loss": totals.mean("policy", updates, 4),
        "value_loss": totals.mean("value", updates, 4),
        # What a constant guess would score, so the value losses read as
        # how much of the return each head explains.
        "return_variance": round(float(rollout.returns.var(unbiased=False)), 4),
        "advantage_spread": round(spread, 4),
        "entropy": totals.mean("entropy", updates, 4),
        "clipped": totals.mean("clipped", updates, 3),
        "approx_kl": totals.mean("kl", updates, 5),
        "grad_norm": totals.mean("grad", updates, 3),
        "mean_return": round(float(rollout.returns.mean()), 4),
        **fields,
    }


class Benchmark:
    """The placement against the heuristic players, smoothed, and the best
    the smoothed figure has reached.

    Carried in the checkpoint, not reset per process: a run that resumes
    has to judge a new measurement against what it has already reached, or
    the first one after every restart becomes the new best whatever it is.
    That replaced a 2.447 checkpoint with a 2.592 one, and on a trainer
    that runs in blocks it would happen at every block."""

    def __init__(self, smoothed: float | None = None, best: float = float("inf")) -> None:
        self.smoothed = smoothed
        self.best = best


def finish(
    args,
    generation: int,
    entry: dict,
    measure,
    payload_of,
    benchmark: Benchmark,
    log_path: Path,
) -> None:
    """Measures when it is time to, saves the checkpoint and logs the
    generation's record, `entry`.

    `measure(generation)` plays the heuristic players and answers as
    `selfplay.measure` does; `payload_of(generation)` is the checkpoint,
    made after measuring so its random states and figures describe exactly
    the state the next generation continues from."""
    measured = None
    is_best = False
    if (generation + 1) % args.measure_every == 0 or generation == 0:
        measured = measure(generation)
        entry.update(
            {
                "placement": round(measured["placement"], 3),
                "score": round(measured["score"], 1),
                "win_rate": round(measured["wins"], 3),
            }
        )
        # The high-water mark is kept apart, so a run that wanders can
        # always be brought back to the best network it has produced. It is
        # chosen on a smoothed figure rather than on the single measurement,
        # because one measurement of a few hundred games carries a standard
        # error about as large as the improvement being looked for.
        benchmark.smoothed = (
            measured["placement"]
            if benchmark.smoothed is None
            else SMOOTHING * measured["placement"] + (1 - SMOOTHING) * benchmark.smoothed
        )
        entry["smoothed"] = round(benchmark.smoothed, 3)
        if benchmark.smoothed < benchmark.best:
            benchmark.best = benchmark.smoothed
            is_best = True
            entry["best"] = True

    payload = payload_of(generation + 1)
    if measured is not None:
        payload["placement"] = measured["placement"]
    if is_best:
        # `best.pt` is the generation the heuristic table liked best so far,
        # and that is all it is. The bots compress real differences
        # several-fold and this measures one seat, so being the best of
        # these readings is a reason to put a checkpoint forward, not a
        # finding that it is stronger. Duels at one table (`neural.duel`),
        # pooled over many deals, decide that.
        atomic_save(payload, args.out / "best.pt")
    # Saved every generation, not only when measured, so that a restart
    # loses one generation at most rather than every one since the last
    # measurement.
    began = time.time()
    atomic_save(payload, args.out / "latest.pt")
    if time.time() - began > 30:
        print(f"saving the checkpoint took {time.time() - began:.0f}s", flush=True)
    print(json.dumps(entry), flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
