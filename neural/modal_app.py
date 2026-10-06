"""Self-play training on Modal.

The desktop card runs the learning step at about a tenth of its arithmetic:
the same generation took 481 milliseconds a step one round and 825 the
next, because the process runs at lowest priority on a machine whose
terminal holds half the cores. Nothing in the loop is wrong; the machine
is. This runs the same loop on a container that has the card to itself.

The image builds two Rust extensions from source: our rules engine, which
plays the games, and Mortal's, vendored, whose encoder is what the network
sees. Self-play is those two, and both must match the checkpoint exactly.

    modal deploy neural/modal_app.py
    modal run neural/modal_app.py::smoke
    modal run --detach neural/modal_app.py::train --generations 40

Runs live side by side on the volume, each in its own directory: `w1012-run`
is the lineage that sees Mortal's planes, and `w320-run` the retired one
that saw the engine's, whose checkpoints read an encoding of those planes
the engine no longer writes and are refused on loading (see
`model.require_engine_observation`). Every function takes the run it works
in, and a checkpoint from another run can be named by its path from the
volume's root, so one lineage can be duelled against another, or against
the published Mortal in `zoo`, and seat it as an opponent.

The run directory lives on the container's own disk, because the replay
ring is a few gigabytes a round and a network volume is the wrong place
for it. Only the checkpoints and the log go to the volume, after every
generation, so a container that is preempted costs one generation, which
is what the desktop supervisor already assumes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

from neural.training_safety import training_control_arguments
from neural.checkpoints import copy_checkpoint, publish_training_snapshot, validate_checkpoint
from neural.cloud_runs import workspace, validate_run, managed_process
from neural.cloud_requests import round_arguments, validate_cloud_request, stage_opponents

HERE = Path(__file__).parent.parent

# Debian with a Rust toolchain, because the engines are compiled and the
# wheels have to be built for the container's own libc.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config", "git")
    .run_commands(
        "curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable --profile minimal"
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("torch==2.8.0", "numpy>=2.0", "maturin>=1.7")
    # The sources the wheels are built from, and the training package itself.
    .add_local_dir(HERE / "engine", "/src/engine", copy=True, ignore=["**/target", "**/pkg"])
    .add_local_file(HERE / "Cargo.toml", "/src/Cargo.toml", copy=True)
    .add_local_file(HERE / "Cargo.lock", "/src/Cargo.lock", copy=True)
    .run_commands(
        "cd /src/engine/riichi-py && maturin build --release --out /src/wheels",
        # Mortal's engine, its own workspace and its own PyO3; see
        # engine/libriichi/NOTICE.md.
        "cd /src/engine/libriichi && maturin build --release --out /src/wheels",
        "pip install /src/wheels/*.whl",
    )
    # The training code last, so editing it does not rebuild the engines.
    .add_local_dir(HERE / "neural", "/src/neural", copy=True, ignore=["**/__pycache__"])
    .env({"PYTHONPATH": "/src", "PYTHONUNBUFFERED": "1"})
)

app = modal.App("mahjong-train", image=image)
volume = modal.Volume.from_name("mahjong-train", create_if_missing=True)
VOLUME = Path("/vol")
# The lineage that sees Mortal's planes, trained from September 2026.
DEFAULT_RUN = "w1012-run"


def _checkpoint(run: str, name: str) -> Path:
    """Where a checkpoint named by a caller is on the volume: in the run's
    own directory, or, named by its path from the volume's root, in
    another run's, so `zoo/mortal_298k` reaches across lineages."""
    # Named with or without the suffix: the launchers that predate the
    # runs living side by side say "latest.pt", and a name that resolved to
    # nothing would start a fresh network over a run's history.
    validate_run(run)
    # A leading slash is absolute on the volume whatever the host thinks
    # of it: `Path` on Windows calls "/tmp/other" relative.
    if (not isinstance(name, str) or not name or "\\" in name
            or name.startswith("/") or Path(name).is_absolute()
            or ".." in Path(name).parts):
        raise ValueError("checkpoint must be a relative path within the volume")
    name = name.removesuffix(".pt")
    own = VOLUME / run / f"{name}.pt"
    if own.exists():
        return own
    return VOLUME / f"{name}.pt"


def _publish(run: Path, generation: int, target_run: str, resumed_at: int) -> int:
    """Publish only copied, fully validated checkpoints of completed
    generations. `resumed_at` is the generation the block resumed from:
    archives past it belong to this timeline (see
    `publish_training_snapshot`)."""
    validate_run(target_run)
    actual = publish_training_snapshot(run, VOLUME / target_run, generation, resumed_at)
    volume.commit()
    return actual


def _same_lineage(run: str, source: Path) -> bool:
    """Whether a checkpoint a block resumes from is the run's own: its
    latest or best, or one of its archives under `history/`. Resuming from
    anything else starts another lineage, which must not inherit the run's
    log, best or leash."""
    return source.is_relative_to(VOLUME / run)


def _carry_log(run: str, where: Path, generation: int) -> None:
    """The run's log into the workspace, so the block appends to it, cut
    after the records of the generation it resumes from. A record past it
    belongs to the timeline a resume from the run's history abandons, or
    to a generation that finished after the checkpoint was copied, and the
    block is about to write that generation's record again."""
    history = VOLUME / run / "log.jsonl"
    if not history.exists():
        return
    kept, dropped = [], 0
    for line in history.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            done = int(record.get("checkpoint_generation", record["generation"] + 1))
        except (ValueError, TypeError, KeyError, AttributeError):
            done = None
        if done is not None and done <= generation:
            kept.append(line + "\n")
        elif line.strip():
            dropped += 1
    (where / "log.jsonl").write_text("".join(kept), encoding="utf-8")
    if dropped:
        print(f"the log carried stops at generation {generation}: {dropped} later "
              "records left behind", flush=True)


def _carry(run: str, source: Path, where: Path, *, leashed: bool = False) -> int:
    """The checkpoint a block resumes from, into its workspace as its
    `latest.pt`, and its generation. When the checkpoint is the run's own,
    the rest of what the block continues comes too: the run's best, its
    log cut at that generation, and, for a leashed trainer, the policy the
    run is leashed to. A start from another lineage inherits none of it:
    not the run's baseline, its record or its leash."""
    generation = copy_checkpoint(source, where / "latest.pt", require_generation=True)
    if _same_lineage(run, source):
        for name in ("best.pt", "reference.pt") if leashed else ("best.pt",):
            saved = VOLUME / run / name
            if saved.exists():
                copy_checkpoint(saved, where / name)
        # Carried so a resumed run appends to its record rather than
        # starting a fresh one every container.
        _carry_log(run, where, generation)
    return generation


def _generation_of(checkpoint: Path) -> int:
    """Missing starts at zero; an unreadable existing checkpoint is an error."""
    if not checkpoint.exists():
        return 0
    return validate_checkpoint(checkpoint, require_generation=True)


def _generation_or_zoo(checkpoint: Path) -> str:
    """A generation to print beside a player at a table. A player from the
    zoo -- a published Mortal -- has none, and a table is not a promotion:
    the file is checked, and it is described rather than refused."""
    generation = validate_checkpoint(checkpoint, require_generation=False)
    return str(generation) if generation is not None else "none (a player from the zoo)"


# Processors for the two trainers. The observation's efficiency lookahead
# costs about two milliseconds a decision on one of them, and the encoder
# spreads a step's decisions over all of them, so play is bound by their
# number: with sixteen, the first block of Mortal's fine-tuning spent 242
# of a generation's 390 seconds playing. Thirty-two is the measurement to
# make now; with the old encoding it was slower, but that encoding did not
# use them.
# Sixteen, measured: thirty-two made self-play slower (93 s against 78 s a
# round of 1024 tables) because the engine does not scale past sixteen, and
# processors are billed whether or not the engine can use them.
TRAINER_CPUS = 16

# The longest one call of a trainer may run. Modal ends it there, in the
# middle of a generation if one is under way, and a retry starts again
# from the last checkpoint published, the partial generation lost.
TRAINER_TIMEOUT = 24 * 60 * 60
# What is kept in hand when a trainer is stopped ahead of the timeout (see
# `_run_trainer`): a generation can run longer than the one before it, by
# a slower mode or by a measurement, and stopping the trainer and saving
# the compiler cache take a while themselves.
TRAINER_SPARE = 10 * 60


class StoppedBeforeTimeout(RuntimeError):
    """A trainer stopped on purpose between two generations, because the
    next would not have finished before the call's timeout. Raised so that
    the call fails and Modal's retries start it again at once, in a fresh
    container, from the checkpoint just published: a call that returned
    instead would simply have ended, and nothing would start it again."""


LOCAL_CACHE = Path("/tmp/inductor-cache")
SHARED_CACHE = VOLUME / "inductor-cache"


def _seed_cache() -> Path:
    """The local compiler cache, filled from the volume's copy once."""
    if not LOCAL_CACHE.exists():
        LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
        if SHARED_CACHE.exists():
            try:
                shutil.copytree(SHARED_CACHE, LOCAL_CACHE, dirs_exist_ok=True)
                print("compiler cache seeded from the volume", flush=True)
            except OSError as error:
                print(f"compiler cache not seeded: {error}", flush=True)
    return LOCAL_CACHE


def _save_cache() -> None:
    """What the compiler built here, back to the volume for the next
    container. Called when a trainer's block ends, not during it."""
    if not LOCAL_CACHE.exists():
        return
    try:
        SHARED_CACHE.mkdir(parents=True, exist_ok=True)
        shutil.copytree(LOCAL_CACHE, SHARED_CACHE, dirs_exist_ok=True)
        volume.commit()
        print("compiler cache saved to the volume", flush=True)
    except OSError as error:
        print(f"compiler cache not saved: {error}", flush=True)


def _environment(cpus: int | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    # The container reports the host's processors, not its share of them;
    # more threads than the share only queue.
    environment["RAYON_NUM_THREADS"] = str(int(cpus or min(os.cpu_count() or 16, 16)))
    environment["PYTHONPATH"] = "/src"
    # Compiled kernels are kept on the container's own disk and seeded from
    # the volume, so a block that resumes in a fresh container finds the
    # ones the last one built (compiling the network's graphs took the
    # first generation of a block a quarter of an hour) without the
    # compiler writing to the network volume while training: a recompile
    # that wrote there stalled a generation by about five minutes.
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(_seed_cache())
    # Say what the container actually has, since the count above is a
    # request: the cgroup's quota is the truth.
    try:
        quota = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()
        share = "unlimited" if quota[0] == "max" else f"{int(quota[0]) / int(quota[1]):.1f}"
    except Exception:
        share = "unknown"
    print(
        f"processors: {os.cpu_count()} visible, quota {share}, "
        f"rayon threads {environment['RAYON_NUM_THREADS']}",
        flush=True,
    )
    return environment


def _target(command: list[str], start: int) -> int:
    """The generation a trainer's command stops at, worked out as every
    trainer works it out: `--rounds` more from the one it resumes at, or,
    with none, `--generations` itself."""
    rounds = int(command[command.index("--rounds") + 1])
    return start + rounds if rounds else int(command[command.index("--generations") + 1])


def _run_trainer(command: list[str], where: Path, run: str, called: float) -> str:
    """Runs a trainer over the workspace `where` and publishes it to the
    run as it goes: each generation as its record arrives, so a preempted
    container costs a single round, and the last once the trainer exits
    cleanly. Answers how it ended.

    A call that cannot finish the run raises rather than answers, so that
    Modal's retries resume it from the checkpoint published last. A call
    that returns has ended, and on 1 October a trainer killed at generation
    147 left the run standing for hours with nothing to start it again.
    So a trainer that exits with an error short of the generation it was to
    reach raises `CalledProcessError`. And once the next generation would
    not finish before the call's timeout, counted from `called`, when the
    call began, the trainer is stopped as soon as a generation has been
    published, seconds into the next one, rather than killed by the timeout
    in the middle of it, and `StoppedBeforeTimeout` is raised."""
    environment = _environment(TRAINER_CPUS)
    print(" ".join(command), flush=True)
    saved_cache = False
    began = time.time()
    started_at = _generation_of(where / "latest.pt")
    target = _target(command, started_at)
    seen = started_at
    print(f"the checkpoint says generation {started_at}", flush=True)
    try:
        with managed_process(subprocess.Popen(
            command,
            cwd="/src",
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )) as process:
            assert process.stdout is not None
            # When the last generation was published, or the trainer began.
            last = time.time()
            for line in process.stdout:
                print(line.rstrip(), flush=True)
                # A finished generation is a JSON record.
                if line.startswith("{") and '"generation"' in line:
                    try:
                        record = json.loads(line)
                        seen = int(record.get("checkpoint_generation", record["generation"] + 1))
                    except Exception:
                        continue
                    seen = _publish(where, seen, run, started_at)
                    if not saved_cache:
                        _save_cache()
                        saved_cache = True
                    now = time.time()
                    lasted, last = now - last, now
                    # The next generation is judged by the last, which
                    # includes its measurement and its checkpoint's save.
                    if seen < target and now - called + 1.3 * lasted + TRAINER_SPARE > TRAINER_TIMEOUT:
                        raise StoppedBeforeTimeout(
                            f"stopped after publishing generation {seen}, {now - called:.0f}s into "
                            f"the call: the next, at about {lasted:.0f}s, would not finish before "
                            f"the {TRAINER_TIMEOUT}s timeout; a retry resumes from it"
                        )
            code = process.wait()
    except StoppedBeforeTimeout as stopped:
        # The trainer has been stopped on the way out of the block above.
        print(stopped, flush=True)
        _save_cache()
        raise
    _save_cache()
    # Publish only when a generation actually finished. A run that trained
    # nothing still rewrites its checkpoint with the target generation
    # stamped on it, and publishing that overwrote a good record with one
    # claiming to be two hundred generations younger.
    if code == 0 and seen > started_at:
        seen = _publish(where, seen, run, started_at)
    else:
        print(f"no final publication (exit={code}); last completed generation {seen}", flush=True)
    if code != 0 and seen < target:
        raise subprocess.CalledProcessError(
            code, command, output=f"the trainer stopped at generation {seen} of {target}"
        )
    return f"exit={code} generation={seen} from {started_at} after {time.time() - began:.0f}s"


@app.function(
    # An A100 when no H100 can be had: on 7 September two blocks sat two
    # hours with no container at all, their calls still marked running.
    gpu=["H100", "A100-80GB"],
    # Self-play is the long pole here, not the card. With the engine's own
    # planes a generation was about 180 seconds of which 80 were playing,
    # and thirty-two processors made it worse, 93 seconds against 78.
    # Mortal's encoder adds an efficiency lookahead that costs about two
    # milliseconds a decision on one processor, which the follower spreads
    # over all of them; whether more of them now pay is a measurement to
    # make on this lineage, not one to carry over.
    cpu=TRAINER_CPUS,
    memory=98304,
    timeout=TRAINER_TIMEOUT,
    volumes={str(VOLUME): volume},
    max_containers=1,
)
def train(
    generations: int = 40,
    games: int = 1024,
    batch: int = 4096,
    epochs: int = 2,
    lr: float = 4e-5,
    entropy: float = 0.005,
    measure_every: int = 5,
    measure_games: int = 512,
    replay_rounds: int = 8,
    replay_steps: int = 180,
    resume: str = "latest",
    opponents: list[str] | None = None,
    opponent_share: float = 0.0,
    run: str = DEFAULT_RUN,
    channels: int = 320,
    blocks: int = 24,
    target_kl: float = 0.0,
    baseline_batch: int | None = None,
) -> str:
    """Runs `generations` rounds of self-play and learning, resuming from the
    checkpoint of that name in the run's directory on the volume when it is
    there, and from a fresh network of `channels` by `blocks` when not.

    The defaults are not the desktop's. A container with the card to itself
    and sixteen processors can play many more games and hold a far larger
    batch, and fresh games are the thing the desktop run is short of: it
    passes over each round three times and its critic learns the round by
    heart.
    """
    # The timeout counts from here, not from when the trainer starts.
    called = time.time()
    validate_cloud_request(generations, opponents, opponent_share)
    controls = training_control_arguments(target_kl, baseline_batch)
    with workspace(run) as where:
        volume.reload()
        started_from = None
        source = _checkpoint(run, resume)
        if source.exists():
            started_from = str(source)
            _carry(run, source, where)
        print(f"resuming from {started_from or 'nothing: a fresh network'}", flush=True)

        command = [
            sys.executable,
            "-m",
            "neural.train",
            # `--rounds` is how many more to train from where the run resumes.
            # `--generations` is an absolute target, and a resumed run is
            # already past any small number, so it would stop at once.
            "--rounds",
            str(generations),
            "--generations",
            "1000000",
            "--games",
            str(games),
            "--batch",
            str(batch),
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--entropy",
            str(entropy),
            "--measure-every",
            str(measure_every),
            "--measure-games",
            str(measure_games),
            "--replay-rounds",
            str(replay_rounds),
            "--replay-steps",
            str(replay_steps),
            "--channels",
            str(channels),
            "--blocks",
            str(blocks),
            "--amp",
            "--compile",
            "--out",
            str(where),
        ]
        if started_from is not None:
            command += ["--resume", str(where / "latest.pt")]

        # Older selves to seat in a share of the games, named relative to the
        # run's directory on the volume, or to the volume's root for another
        # lineage's, so "history/gen-00290" and "zoo/mortal_298k" both work.
        # Measured 6 September: the old run recovered fully against its own
        # past and only halfway against a foreign network, so a large part of
        # what it gained was knowing its own family. An older self is foreign
        # enough to be worth playing, and another lineage more so.
        command += stage_opponents(
            where, opponents, opponent_share, lambda name: _checkpoint(run, name)
        )
        command += controls
        return _run_trainer(command, where, run, called)


@app.function(
    # An A100 when no H100 can be had: on 7 September two blocks sat two
    # hours with no container at all, their calls still marked running.
    gpu=["H100", "A100-80GB"],
    cpu=TRAINER_CPUS,
    memory=98304,
    timeout=TRAINER_TIMEOUT,
    volumes={str(VOLUME): volume},
    max_containers=1,
)
def train_mortal(
    generations: int = 40,
    games: int = 1024,
    batch: int = 2048,
    epochs: int = 2,
    lr: float = 1e-5,
    entropy: float = 0.005,
    temperature: float | None = None,
    measure_every: int = 5,
    measure_games: int = 512,
    resume: str = "latest",
    mortal: str = "zoo/mortal_298k",
    opponents: list[str] | None = None,
    opponent_share: float = 0.0,
    seat_share: float = 0.0,
    run: str = "mortal-run",
    target_kl: float = 0.0,
    baseline_batch: int | None = None,
    until: int = 0,
    skip_forced: bool = False,
    baseline_from_play: bool = False,
    check_baseline: bool = False,
) -> str:
    """Fine-tunes a published Mortal on our rules by self-play, in a run
    directory of its own: see `neural/train_mortal.py`. Resumes from the
    checkpoint of that name in the run when it is there, and starts from
    the published Mortal named otherwise. Its own function, so it runs
    beside the other lineage's training rather than queueing behind it.

    `until`, when given, is the generation to stop at rather than a count
    from wherever it resumes: a call the spend limit stalled and Modal
    started again from the top then finishes the run instead of playing
    `generations` more.

    `temperature` is passed on only when given: a run resumes at the one
    its checkpoint was trained at, and one from the published Mortal
    starts at 1.0 (see `neural/train_mortal.py`).

    `skip_forced` has the seated others leave the rows with one move open
    unasked, the trainer's `--skip-forced`; off, they play as they did.

    `baseline_from_play` takes the baseline from the values the learner
    recorded as it played rather than a pass over the round, and
    `check_baseline` makes the pass as well and records how far the two
    are apart: the trainer's `--baseline-from-play` and `--check-baseline`.
    Each is passed on only when true.
    """
    # The timeout counts from here, not from when the trainer starts.
    called = time.time()
    validate_cloud_request(generations, opponents, opponent_share, seat_share)
    rounds = round_arguments(generations, until)
    controls = training_control_arguments(target_kl, baseline_batch)
    with workspace(run) as where:
        volume.reload()
        source = _checkpoint(run, resume)
        command = [
            sys.executable, "-m", "neural.train_mortal", *rounds,
            "--games", str(games), "--batch", str(batch), "--epochs", str(epochs),
            "--lr", str(lr), "--entropy", str(entropy),
            "--measure-every", str(measure_every), "--measure-games", str(measure_games),
            "--amp", "--compile", "--out", str(where),
        ]
        if temperature is not None:
            command += ["--temperature", str(temperature)]
        if skip_forced:
            command.append("--skip-forced")
        if baseline_from_play:
            command.append("--baseline-from-play")
        if check_baseline:
            command.append("--check-baseline")
        if source.exists():
            _carry(run, source, where)
            command += ["--resume", str(where / "latest.pt")]
            print(f"resuming from {source}", flush=True)
        else:
            origin = _checkpoint(run, mortal)
            if not origin.exists():
                raise FileNotFoundError(f"no Mortal at {origin}")
            shutil.copyfile(origin, where / "origin.pt")
            command += ["--mortal", str(where / "origin.pt")]
            print(f"starting from {origin}", flush=True)

        # The seating by player goes in the staged manifest as well as on
        # the trainer's command line, so the run records how it was seated.
        command += stage_opponents(
            where, opponents, opponent_share, lambda name: _checkpoint(run, name),
            seat_share=seat_share,
        )
        command += controls
        return _run_trainer(command, where, run, called)


@app.function(
    # An A100 when no H100 can be had: on 7 September two blocks sat two
    # hours with no container at all, their calls still marked running.
    gpu=["H100", "A100-80GB"],
    cpu=TRAINER_CPUS,
    memory=98304,
    timeout=TRAINER_TIMEOUT,
    volumes={str(VOLUME): volume},
    max_containers=1,
)
def train_combined(
    generations: int = 20,
    games: int = 1024,
    batch: int = 2048,
    epochs: int = 2,
    lr: float = 1e-4,
    lr_ours: float = 4e-5,
    lr_mortal: float = 3e-5,
    entropy: float = 0.0005,
    leash: float = 0.0,
    fixed: list[str] | None = None,
    measure_every: int = 5,
    measure_games: int = 512,
    resume: str = "latest",
    # The re-headed checkpoint, which answers over Mortal's forty-six moves.
    # `w1012-run/latest` is the same trunk with our old seventy-eight, and
    # joining that one refuses rather than quietly mistranslating.
    ours: str = "w1012-run/mortal-space",
    mortal: str = "mortal-run/latest",
    opponents: list[str] | None = None,
    opponent_share: float = 0.0,
    seat_share: float = 0.0,
    explore: float = 0.0,
    run: str = "joined-run",
    compile: bool = True,
    target_kl: float = 0.0,
    baseline_batch: int | None = None,
    entropy_target: float = 0.0,
    entropy_max: float = 0.05,
    until: int = 0,
    skip_forced: bool = False,
    baseline_from_play: bool = False,
    check_baseline: bool = False,
    reuse_phi: bool = False,
) -> str:
    """Trains the joined player, our network and a Mortal beneath one
    fusion head, in a run directory of its own: see
    `neural/train_combined.py`. Resumes from the checkpoint of that name
    in the run when it is there, and otherwise joins the two checkpoints
    named, which may be any run's. `until`, `skip_forced`,
    `baseline_from_play` and `check_baseline` are as for `train_mortal`.

    `reuse_phi` has a generation that holds Mortal still learn from
    Mortal's vectors as play worked them out, the trainer's `--reuse-phi`;
    passed on only when true.
    """
    # The timeout counts from here, not from when the trainer starts.
    called = time.time()
    # Named after the run: a container that has already trained another
    # must not leave its log where this one will append to it.
    validate_cloud_request(generations, opponents, opponent_share, seat_share)
    rounds = round_arguments(generations, until)
    controls = training_control_arguments(target_kl, baseline_batch)
    with workspace(run) as where:
        volume.reload()
        source = _checkpoint(run, resume)
        command = [
            sys.executable, "-m", "neural.train_combined", *rounds,
            "--games", str(games), "--batch", str(batch), "--epochs", str(epochs),
            "--lr", str(lr), "--lr-ours", str(lr_ours), "--lr-mortal", str(lr_mortal),
            "--entropy", str(entropy), "--leash", str(leash),
            "--entropy-target", str(entropy_target), "--entropy-max", str(entropy_max),
            "--explore", str(explore),
            "--measure-every", str(measure_every), "--measure-games", str(measure_games),
            "--amp", "--out", str(where),
        ]
        # Six networks compile here, the joined player twice over and its four
        # seated others once, which took a fresh container over an hour before
        # its first generation; eager is the choice when that is not worth it.
        if compile:
            command.append("--compile")
        if skip_forced:
            command.append("--skip-forced")
        if baseline_from_play:
            command.append("--baseline-from-play")
        if check_baseline:
            command.append("--check-baseline")
        if reuse_phi:
            command.append("--reuse-phi")
        if fixed:
            command += ["--fixed", *fixed]
        if source.exists():
            _carry(run, source, where, leashed=True)
            command += ["--resume", str(where / "latest.pt")]
            print(f"resuming from {source}", flush=True)
        else:
            parts = []
            for label, name in (("--ours", ours), ("--mortal", mortal)):
                found = _checkpoint(run, name)
                if not found.exists():
                    raise FileNotFoundError(f"no checkpoint at {found}")
                local = where / (name.replace("/", "--") + ".pt")
                shutil.copyfile(found, local)
                parts += [label, str(local)]
            command += parts
            print(f"joining {ours} and {mortal}", flush=True)

        # As for `train_mortal`: the manifest and the command line both say
        # how the others were seated.
        command += stage_opponents(
            where, opponents, opponent_share, lambda name: _checkpoint(run, name),
            seat_share=seat_share,
        )
        command += controls
        return _run_trainer(command, where, run, called)


@app.function(
    gpu="L40S",
    cpu=16.0,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={str(VOLUME): volume},
)
def arena(
    which: str = "best",
    games: int = 1000,
    seed: int = 555_000,
    channels: int = 320,
    blocks: int = 24,
    run: str = DEFAULT_RUN,
) -> str:
    """Measures a checkpoint from the volume against the heuristic players.

    Its own container, so a strength check never shares a card with the
    learning step and never has to stop it. The training loop's own
    placement figure is 1024 games in one seat and its error is around
    0.035, which cannot separate this network from the one the browser
    plays; this is the same deals four times over with the network in each
    seat, and its error comes from the deals.

    The answer is the report. A missing checkpoint raises
    `FileNotFoundError`, and a measurement that exits with an error raises
    `CalledProcessError` carrying its output, as `duel` does.
    """
    volume.reload()
    source = _checkpoint(run, which)
    if not source.exists():
        raise FileNotFoundError(f"no checkpoint at {source}")
    with workspace("arena") as local:
        name = Path(which).name
        shutil.copyfile(source, local / f"{name}.pt")
        # Say which generation this is. The report named only the file it
        # copied, so two runs on an unchanged checkpoint were indistinguishable
        # from two on different ones, and at a fixed seed they give the same
        # number: one of them was a container spent to learn nothing.
        generation = _generation_of(local / f"{name}.pt")
        print(f"measuring {which}.pt, generation {generation}", flush=True)

        command = [
            sys.executable, "-m", "neural.arena", str(local / f"{name}.pt"),
            "--games", str(games), "--seed", str(seed),
            # The checkpoint says its own shape; these stand in only for the
            # oldest, which do not.
            "--channels", str(channels), "--blocks", str(blocks),
        ]
        result = subprocess.run(
            command,
            cwd="/src",
            env=_environment(),
            capture_output=True,
            text=True,
        )
    if result.returncode:
        output = (result.stdout or "") + (result.stderr or "")
        print(output, flush=True)
        raise subprocess.CalledProcessError(result.returncode, command, output=output)
    # Prefixed so the caller can tell one measurement from another without
    # parsing the report; the JSON still starts at the first brace.
    answer = f"generation {generation}\n{result.stdout or ''}"
    print(answer, flush=True)
    return answer


@app.function(
    gpu="L40S",
    cpu=16.0,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={str(VOLUME): volume},
)
def duel(
    challenger: str = "latest",
    incumbent: str = "zoo/mortal_298k",
    games: int = 1000,
    seed: int = 555_000,
    channels: int = 320,
    blocks: int = 24,
    incumbent_channels: int = 192,
    incumbent_blocks: int = 10,
    run: str = DEFAULT_RUN,
) -> str:
    """Sits two checkpoints from the volume at the same table.

    Measuring each against the heuristic players and subtracting has a
    floor of about 0.024 on the difference, so two close networks never
    separate. At one table the luck of the deal falls on both at once, and
    what comes back is a single placement against the 2.50 two identical
    players would average. The two may be of different lineages: each is
    served the planes it sees.

    The answer is the table's report and nothing else. A missing checkpoint
    raises `FileNotFoundError`, and a table that exits with an error raises
    `CalledProcessError` carrying its output, so a caller pooling many duels
    counts that call as failed instead of reading an error as a report.
    """
    volume.reload()
    sources = [_checkpoint(run, name) for name in (challenger, incumbent)]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"no checkpoint at {source}")
    with workspace("duel") as local:
        files = []
        for name, source in zip((challenger, incumbent), sources):
            # Two names may share a file name across runs, so keep the run's
            # name in the copy's.
            copied = local / (name.replace("/", "--") + ".pt")
            shutil.copyfile(source, copied)
            files.append(copied)
        print(
            f"challenger {challenger}.pt generation {_generation_or_zoo(files[0])} "
            f"against {incumbent}.pt generation {_generation_or_zoo(files[1])}",
            flush=True,
        )

        command = [
            sys.executable, "-m", "neural.duel", str(files[0]), str(files[1]),
            "--games", str(games), "--seed", str(seed),
            "--channels", str(channels), "--blocks", str(blocks),
            "--incumbent-channels", str(incumbent_channels),
            "--incumbent-blocks", str(incumbent_blocks),
        ]
        result = subprocess.run(
            command,
            cwd="/src",
            env=_environment(),
            capture_output=True,
            text=True,
        )
    if result.returncode:
        output = (result.stdout or "") + (result.stderr or "")
        print(output, flush=True)
        raise subprocess.CalledProcessError(result.returncode, command, output=output)
    answer = result.stdout or ""
    print(answer, flush=True)
    return answer


@app.function(
    # An A100 when no H100 can be had: on 7 September two blocks sat two
    # hours with no container at all, their calls still marked running.
    gpu=["H100", "A100-80GB"],
    cpu=16.0,
    memory=65536,
    timeout=6 * 60 * 60,
    volumes={str(VOLUME): volume},
    max_containers=1,
)
def distil(
    teacher: str,
    student: str = "",
    rounds: int = 60,
    games: int = 256,
    lr: float = 2e-4,
    teacher_channels: int = 192,
    teacher_blocks: int = 10,
    channels: int = 320,
    blocks: int = 24,
    student_planes: str = "mortal",
    attention: bool = True,
    run: str = DEFAULT_RUN,
    name: str = "distilled",
) -> str:
    """Teaches a network another's moves, or the heuristic player's.

    With a `student`, that checkpoint is fine-tuned; without one, a fresh
    network of `channels` by `blocks` is taught from nothing, which is how
    a lineage begins: a network that discards at random has most of the
    game still to discover, and imitation hands it the part that is not
    strategy at all. The teacher has no default and must be named: a
    network of either lineage, since each is served the planes it sees, or
    an empty name for the heuristic player that ships with the game. The
    old default, `w320-run/published`, reads an encoding of the engine's
    planes the engine no longer writes and is refused on loading. The
    student learns the teacher's whole distribution rather than its choice
    alone.

    With `student_planes` set to `engine` the student reads the engine's
    ninety-seven planes instead of Mortal's thousand. The browser plays
    only networks that read Mortal's (see `neural.export`).

    The result goes to the run's directory under `name`, so it can be
    duelled before anything decides to train on from it. A missing
    checkpoint raises `FileNotFoundError`, and a failed run raises
    `CalledProcessError` carrying its output, and publishes nothing.
    """
    with workspace(run) as where:
        volume.reload()
        local = where / "inputs"
        local.mkdir()
        out = where / "out"
        command = [
            sys.executable, "-m", "neural.imitate",
            "--rounds", str(rounds), "--games", str(games), "--lr", str(lr),
            "--channels", str(channels), "--blocks", str(blocks),
            "--student", student_planes,
            "--measure-every", "10", "--measure-games", "512",
            "--out", str(out),
        ]
        if not attention:
            command.append("--no-attention")
        for label, which in (("--resume", student), ("--teacher", teacher)):
            if not which:
                continue
            source = _checkpoint(run, which)
            if not source.exists():
                raise FileNotFoundError(f"no checkpoint at {source}")
            copied = local / (which.replace("/", "--") + ".pt")
            shutil.copyfile(source, copied)
            command += [label, str(copied)]
        if teacher:
            command += [
                "--teacher-channels", str(teacher_channels),
                "--teacher-blocks", str(teacher_blocks),
            ]
        print(" ".join(command), flush=True)

        result = subprocess.run(
            command,
            cwd="/src",
            env=_environment(),
            capture_output=True,
            text=True,
        )
        if result.returncode:
            output = (result.stdout or "")[-4000:] + (result.stderr or "")
            print(output, flush=True)
            raise subprocess.CalledProcessError(result.returncode, command, output=output)
        answer = (result.stdout or "")[-4000:]
        if (out / "latest.pt").exists():
            target = VOLUME / run
            target.mkdir(parents=True, exist_ok=True)
            copy_checkpoint(out / "latest.pt", target / f"{validate_run(name)}.pt")
            volume.commit()
            answer += f"\nwrote {run}/{name}.pt"
        print(answer, flush=True)
        return answer


@app.function(
    gpu=["H100", "A100-80GB"],
    cpu=16.0,
    memory=65536,
    timeout=6 * 60 * 60,
    volumes={str(VOLUME): volume},
    max_containers=1,
)
def rehead(
    teacher: str = "w1012-run/latest",
    resume: str = "",
    rounds: int = 40,
    games: int = 64,
    lr: float = 1e-3,
    batch: int = 2048,
    epochs: int = 2,
    temperature: float = 1.0,
    run: str = DEFAULT_RUN,
    name: str = "mortal-space",
) -> str:
    """Teaches one of our networks to answer in Mortal's action space.

    The trunk is the teacher's and does not move; only the last layer is
    replaced, by one over Mortal's forty-six moves, and taught to say what
    the old head said. See `neural/rehead.py`. The result goes to the run's
    directory under `name`, to be duelled against the teacher before
    anything is built on it. A missing checkpoint raises
    `FileNotFoundError`, and a failed run raises `CalledProcessError`
    carrying its output, and publishes nothing.
    """
    with workspace(run) as where:
        volume.reload()
        out = where / "out"
        source = _checkpoint(run, teacher)
        if not source.exists():
            raise FileNotFoundError(f"no checkpoint at {source}")
        local = where / "teacher.pt"
        shutil.copyfile(source, local)
        command = [
            sys.executable, "-m", "neural.rehead",
            "--teacher", str(local),
            "--rounds", str(rounds), "--games", str(games), "--lr", str(lr),
            "--batch", str(batch), "--epochs", str(epochs),
            "--temperature", str(temperature),
            "--out", str(out),
        ]
        if resume:
            found = _checkpoint(run, resume)
            if not found.exists():
                raise FileNotFoundError(f"no checkpoint at {found}")
            carried = where / "student.pt"
            shutil.copyfile(found, carried)
            command += ["--resume", str(carried)]
        print(" ".join(command), flush=True)

        result = subprocess.run(
            command,
            cwd="/src",
            env=_environment(),
            capture_output=True,
            text=True,
        )
        if result.returncode:
            output = (result.stdout or "")[-4000:] + (result.stderr or "")
            print(output, flush=True)
            raise subprocess.CalledProcessError(result.returncode, command, output=output)
        answer = (result.stdout or "")[-4000:]
        if (out / "latest.pt").exists():
            target = VOLUME / run
            target.mkdir(parents=True, exist_ok=True)
            copy_checkpoint(out / "latest.pt", target / f"{validate_run(name)}.pt")
            volume.commit()
            answer += f"\nwrote {run}/{name}.pt"
        print(answer, flush=True)
        return answer


@app.function(
    gpu="L4",
    cpu=8.0,
    memory=32768,
    timeout=45 * 60,
    volumes={str(VOLUME): volume},
)
def smoke() -> str:
    """A tiny round end to end: both engines load, self-play runs, the
    learning step runs, a checkpoint is written. Cheap, and it fails for
    the same reasons the real run would."""
    import torch

    import libriichi
    import riichi_py

    report = [
        f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        f"riichi_py planes={riichi_py.PLANES} actions={riichi_py.ACTIONS} "
        f"oracle={riichi_py.ORACLE_PLANES}",
        f"libriichi obs={libriichi.consts.obs_shape(4)}",
        f"cpus={os.cpu_count()}",
    ]
    out = Path("/scratch/smoke")
    out.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable, "-m", "neural.train",
            "--generations", "1", "--games", "24", "--batch", "512", "--epochs", "1",
            "--measure-every", "1", "--measure-games", "8",
            "--replay-rounds", "2", "--replay-steps", "4",
            "--amp", "--compile",
            "--out", str(out),
        ],
        cwd="/src",
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=40 * 60,
    )
    report.append(f"train exit={result.returncode}")
    report.append((result.stdout or "")[-2500:])
    if result.returncode != 0:
        report.append((result.stderr or "")[-2500:])
    # Mortal's fine-tuning too, from the published Mortal on the volume,
    # into scratch: nothing is published.
    volume.reload()
    origin = _checkpoint("mortal-run", "zoo/mortal_298k")
    if origin.exists():
        local = out / "origin.pt"
        shutil.copyfile(origin, local)
        result = subprocess.run(
            [
                sys.executable, "-m", "neural.train_mortal",
                "--mortal", str(local), "--rounds", "1", "--games", "16", "--batch", "512",
                "--measure-every", "1", "--measure-games", "8",
                "--amp", "--compile", "--out", str(out / "mortal"),
            ],
            cwd="/src",
            env=_environment(),
            capture_output=True,
            text=True,
            timeout=40 * 60,
        )
        report.append(f"train_mortal exit={result.returncode}")
        report.append((result.stdout or "")[-1500:])
        if result.returncode != 0:
            report.append((result.stderr or "")[-2500:])
    else:
        report.append(f"no Mortal at {origin}; its trainer not tried")
    answer = "\n".join(report)
    # Printed as well as returned: the container's output is what streams
    # back to a `modal run`, and a return value alone shows nothing.
    print(answer, flush=True)
    return answer


@app.local_entrypoint()
def main(generations: int = 40, games: int = 1024, batch: int = 4096,
         target_kl: float = 0.0, baseline_batch: int | None = None) -> None:
    print(train.remote(generations=generations, games=games, batch=batch,
                       target_kl=target_kl, baseline_batch=baseline_batch))
