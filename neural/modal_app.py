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

Runs live side by side on the volume, each in its own directory: `w320-run`
is the lineage that sees the engine's planes, `w1012-run` the one that sees
Mortal's. Every function takes the run it works in, and a checkpoint from
another run can be named by its path from the volume's root, so the new
lineage can be duelled against the old one and seat it as an opponent.

The run directory lives on the container's own disk, because the replay
ring is a few gigabytes a round and a network volume is the wrong place
for it. Only the checkpoints and the log go to the volume, after every
generation, so a container that is preempted costs one generation, which
is what the desktop supervisor already assumes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

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
RUN = Path("/scratch/run")
# The lineage that sees Mortal's planes, trained from September 2026.
DEFAULT_RUN = "w1012-run"


def _checkpoint(run: str, name: str) -> Path:
    """Where a checkpoint named by a caller is on the volume: in the run's
    own directory, or, named by its path from the volume's root, in
    another run's, so `w320-run/published` reaches across lineages."""
    # Named with or without the suffix: the launchers that predate the
    # runs living side by side say "latest.pt", and a name that resolved to
    # nothing would start a fresh network over a run's history.
    name = name.removesuffix(".pt")
    own = VOLUME / run / f"{name}.pt"
    if own.exists():
        return own
    return VOLUME / f"{name}.pt"


def _publish(run: Path, generation: int, target_run: str) -> None:
    """Copies the checkpoints and the log to the volume and commits them."""
    target = VOLUME / target_run
    target.mkdir(parents=True, exist_ok=True)
    # `log.jsonl` is the record the loop writes itself, one line a
    # generation. `train.log` is only what the desktop's shell redirect
    # captured, and nothing on this side appends to it.
    for name in ("latest.pt", "best.pt", "log.jsonl"):
        source = run / name
        if not source.exists():
            continue
        staged = target / f".{name}.partial"
        shutil.copyfile(source, staged)
        staged.replace(target / name)
    (target / "generation.txt").write_text(str(generation), encoding="utf-8")
    # Keep a checkpoint every tenth generation, for good. `best.pt` is
    # chosen on placement against the heuristic players, and that figure
    # has been measured moving opposite to real strength: over generations
    # 270 to 286 it improved by 0.088 while the network lost 0.13 at the
    # table. So the best network this run ever had was overwritten twice by
    # worse ones with better bot scores, and is gone. A run cannot be
    # rolled back to a peak it did not keep.
    if generation % 10 == 0:
        history = target / "history"
        history.mkdir(parents=True, exist_ok=True)
        kept = history / f"gen-{generation:05d}.pt"
        if not kept.exists() and (run / "latest.pt").exists():
            shutil.copyfile(run / "latest.pt", kept)
    volume.commit()


def _generation_of(checkpoint: Path) -> int:
    """The generation a checkpoint says it is, or zero.

    Read from the checkpoint rather than counted from the log: the
    desktop's log is UTF-16, so counting it returned zero, and a zero there
    is not harmless. It went into the published record and the next run
    resumed from the wrong place.
    """
    if not checkpoint.exists():
        return 0
    try:
        import torch

        return int(torch.load(checkpoint, map_location="cpu", weights_only=True)["generation"])
    except Exception:
        return 0


# Processors for the two trainers. The observation's efficiency lookahead
# costs about two milliseconds a decision on one of them, and the encoder
# spreads a step's decisions over all of them, so play is bound by their
# number: with sixteen, the first block of Mortal's fine-tuning spent 242
# of a generation's 390 seconds playing. Thirty-two is the measurement to
# make now; with the old encoding it was slower, but that encoding did not
# use them.
TRAINER_CPUS = 32


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
    timeout=24 * 60 * 60,
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
    RUN.mkdir(parents=True, exist_ok=True)
    volume.reload()
    started_from = None
    source = _checkpoint(run, resume)
    if source.exists():
        started_from = str(source)
        shutil.copyfile(source, RUN / "latest.pt")
        # Carry the history forward so a resumed run appends to it rather
        # than starting a fresh record every container.
        for name in ("log.jsonl", "train.log"):
            history = VOLUME / run / name
            if history.exists():
                shutil.copyfile(history, RUN / name)
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
        str(RUN),
    ]
    if (RUN / "latest.pt").exists():
        command += ["--resume", str(RUN / "latest.pt")]

    # Older selves to seat in a share of the games, named relative to the
    # run's directory on the volume, or to the volume's root for another
    # lineage's, so "history/gen-00290" and "w320-run/published" both work.
    # Measured 6 September: the old run recovered fully against its own
    # past and only halfway against a foreign network, so a large part of
    # what it gained was knowing its own family. An older self is foreign
    # enough to be worth playing, and another lineage more so.
    seated = []
    for name in opponents or []:
        source = _checkpoint(run, name)
        if not source.exists():
            print(f"no opponent at {source}", flush=True)
            continue
        local = RUN / "opponents" / f"{Path(name).name}.pt"
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local)
        seated.append(str(local))
    if seated:
        command += ["--opponents", *seated, "--opponent-share", str(opponent_share)]

    environment = _environment(TRAINER_CPUS)
    print(" ".join(command), flush=True)

    saved_cache = False
    began = time.time()
    started_at = _generation_of(RUN / "latest.pt")
    seen = started_at
    print(f"the checkpoint says generation {started_at}", flush=True)
    process = subprocess.Popen(
        command,
        cwd="/src",
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        # A finished generation is a JSON record; publish on each one so a
        # preempted container costs a single round.
        if line.startswith("{") and '"generation"' in line:
            try:
                import json

                seen = int(json.loads(line)["generation"])
            except Exception:
                continue
            _publish(RUN, seen, run)
            if not saved_cache:
                _save_cache()
                saved_cache = True
    code = process.wait()
    _save_cache()
    # Publish only when a generation actually finished. A run that trained
    # nothing still rewrites its checkpoint with the target generation
    # stamped on it, and publishing that overwrote a good record with one
    # claiming to be two hundred generations younger.
    if seen > started_at:
        _publish(RUN, seen, run)
    else:
        print(f"nothing trained: leaving the volume at generation {started_at}", flush=True)
    return (
        f"exit={code} generation={seen} from {started_at} "
        f"after {time.time() - began:.0f}s"
    )


@app.function(
    # An A100 when no H100 can be had: on 7 September two blocks sat two
    # hours with no container at all, their calls still marked running.
    gpu=["H100", "A100-80GB"],
    cpu=TRAINER_CPUS,
    memory=98304,
    timeout=24 * 60 * 60,
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
    temperature: float = 1.0,
    measure_every: int = 5,
    measure_games: int = 512,
    resume: str = "latest",
    mortal: str = "zoo/mortal_298k",
    opponents: list[str] | None = None,
    opponent_share: float = 0.0,
    run: str = "mortal-run",
) -> str:
    """Fine-tunes a published Mortal on our rules by self-play, in a run
    directory of its own: see `neural/train_mortal.py`. Resumes from the
    checkpoint of that name in the run when it is there, and starts from
    the published Mortal named otherwise. Its own function, so it runs
    beside the other lineage's training rather than queueing behind it.
    """
    where = Path("/scratch/mortal-run")
    where.mkdir(parents=True, exist_ok=True)
    volume.reload()
    source = _checkpoint(run, resume)
    command = [
        sys.executable, "-m", "neural.train_mortal",
        "--rounds", str(generations), "--generations", "1000000",
        "--games", str(games), "--batch", str(batch), "--epochs", str(epochs),
        "--lr", str(lr), "--entropy", str(entropy), "--temperature", str(temperature),
        "--measure-every", str(measure_every), "--measure-games", str(measure_games),
        "--amp", "--compile", "--out", str(where),
    ]
    if source.exists():
        shutil.copyfile(source, where / "latest.pt")
        history = VOLUME / run / "log.jsonl"
        if history.exists():
            shutil.copyfile(history, where / "log.jsonl")
        command += ["--resume", str(where / "latest.pt")]
        print(f"resuming from {source}", flush=True)
    else:
        origin = _checkpoint(run, mortal)
        if not origin.exists():
            return f"no Mortal at {origin}"
        shutil.copyfile(origin, where / "origin.pt")
        command += ["--mortal", str(where / "origin.pt")]
        print(f"starting from {origin}", flush=True)

    seated = []
    for name in opponents or []:
        found = _checkpoint(run, name)
        if not found.exists():
            print(f"no opponent at {found}", flush=True)
            continue
        local = where / "opponents" / (name.replace("/", "--") + ".pt")
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(found, local)
        seated.append(str(local))
    if seated:
        command += ["--opponents", *seated, "--opponent-share", str(opponent_share)]
    print(" ".join(command), flush=True)

    saved_cache = False
    began = time.time()
    started_at = _generation_of(where / "latest.pt")
    seen = started_at
    process = subprocess.Popen(
        command,
        cwd="/src",
        env=_environment(TRAINER_CPUS),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        if line.startswith("{") and '"generation"' in line:
            try:
                import json

                seen = int(json.loads(line)["generation"])
            except Exception:
                continue
            _publish(where, seen, run)
            if not saved_cache:
                _save_cache()
                saved_cache = True
    code = process.wait()
    _save_cache()
    if seen > started_at:
        _publish(where, seen, run)
    else:
        print(f"nothing trained: leaving the volume at generation {started_at}", flush=True)
    return f"exit={code} generation={seen} from {started_at} after {time.time() - began:.0f}s"


@app.function(
    # An A100 when no H100 can be had: on 7 September two blocks sat two
    # hours with no container at all, their calls still marked running.
    gpu=["H100", "A100-80GB"],
    cpu=TRAINER_CPUS,
    memory=98304,
    timeout=24 * 60 * 60,
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
    entropy: float = 0.01,
    fixed: list[str] | None = None,
    measure_every: int = 5,
    measure_games: int = 512,
    resume: str = "latest",
    ours: str = "w1012-run/latest",
    mortal: str = "mortal-run/latest",
    opponents: list[str] | None = None,
    opponent_share: float = 0.0,
    run: str = "joined-run",
    compile: bool = True,
) -> str:
    """Trains the joined player, our network and a Mortal beneath one
    fusion head, in a run directory of its own: see
    `neural/train_combined.py`. Resumes from the checkpoint of that name
    in the run when it is there, and otherwise joins the two checkpoints
    named, which may be any run's.
    """
    where = Path("/scratch/joined-run")
    where.mkdir(parents=True, exist_ok=True)
    volume.reload()
    source = _checkpoint(run, resume)
    command = [
        sys.executable, "-m", "neural.train_combined",
        "--rounds", str(generations), "--generations", "1000000",
        "--games", str(games), "--batch", str(batch), "--epochs", str(epochs),
        "--lr", str(lr), "--lr-ours", str(lr_ours), "--lr-mortal", str(lr_mortal),
        "--entropy", str(entropy),
        "--measure-every", str(measure_every), "--measure-games", str(measure_games),
        "--amp", "--out", str(where),
    ]
    # Six networks compile here, the joined player twice over and its four
    # seated others once, which took a fresh container over an hour before
    # its first generation; eager is the choice when that is not worth it.
    if compile:
        command.append("--compile")
    if fixed:
        command += ["--fixed", *fixed]
    if source.exists():
        shutil.copyfile(source, where / "latest.pt")
        history = VOLUME / run / "log.jsonl"
        if history.exists():
            shutil.copyfile(history, where / "log.jsonl")
        command += ["--resume", str(where / "latest.pt")]
        print(f"resuming from {source}", flush=True)
    else:
        parts = []
        for label, name in (("--ours", ours), ("--mortal", mortal)):
            found = _checkpoint(run, name)
            if not found.exists():
                return f"no checkpoint at {found}"
            local = where / (name.replace("/", "--") + ".pt")
            shutil.copyfile(found, local)
            parts += [label, str(local)]
        command += parts
        print(f"joining {ours} and {mortal}", flush=True)

    seated = []
    for name in opponents or []:
        found = _checkpoint(run, name)
        if not found.exists():
            print(f"no opponent at {found}", flush=True)
            continue
        local = where / "opponents" / (name.replace("/", "--") + ".pt")
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(found, local)
        seated.append(str(local))
    if seated:
        command += ["--opponents", *seated, "--opponent-share", str(opponent_share)]
    print(" ".join(command), flush=True)

    saved_cache = False
    began = time.time()
    started_at = _generation_of(where / "latest.pt")
    seen = started_at
    process = subprocess.Popen(
        command,
        cwd="/src",
        env=_environment(TRAINER_CPUS),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        if line.startswith("{") and '"generation"' in line:
            try:
                import json

                seen = int(json.loads(line)["generation"])
            except Exception:
                continue
            _publish(where, seen, run)
            if not saved_cache:
                _save_cache()
                saved_cache = True
    code = process.wait()
    _save_cache()
    if seen > started_at:
        _publish(where, seen, run)
    else:
        print(f"nothing trained: leaving the volume at generation {started_at}", flush=True)
    return f"exit={code} generation={seen} from {started_at} after {time.time() - began:.0f}s"


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
    """
    volume.reload()
    source = _checkpoint(run, which)
    if not source.exists():
        return f"no checkpoint at {source}"
    local = Path("/scratch/arena")
    local.mkdir(parents=True, exist_ok=True)
    name = Path(which).name
    shutil.copyfile(source, local / f"{name}.pt")
    # Say which generation this is. The report named only the file it
    # copied, so two runs on an unchanged checkpoint were indistinguishable
    # from two on different ones, and at a fixed seed they give the same
    # number: one of them was a container spent to learn nothing.
    generation = _generation_of(local / f"{name}.pt")
    print(f"measuring {which}.pt, generation {generation}", flush=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "neural.arena", str(local / f"{name}.pt"),
            "--games", str(games), "--seed", str(seed),
            # The checkpoint says its own shape; these stand in only for the
            # oldest, which do not.
            "--channels", str(channels), "--blocks", str(blocks),
        ],
        cwd="/src",
        env=_environment(),
        capture_output=True,
        text=True,
    )
    answer = (result.stdout or "") + (result.stderr or "" if result.returncode else "")
    # Prefixed so the caller can tell one measurement from another without
    # parsing the report; the JSON still starts at the first brace.
    answer = f"generation {generation}\n{answer}"
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
    incumbent: str = "w320-run/published",
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
    """
    volume.reload()
    local = Path("/scratch/duel")
    local.mkdir(parents=True, exist_ok=True)
    files = []
    for name in (challenger, incumbent):
        source = _checkpoint(run, name)
        if not source.exists():
            return f"no checkpoint at {source}"
        # Two names may share a file name across runs, so keep the run's
        # name in the copy's.
        copied = local / (name.replace("/", "--") + ".pt")
        shutil.copyfile(source, copied)
        files.append(copied)
    print(
        f"challenger {challenger}.pt generation {_generation_of(files[0])} "
        f"against {incumbent}.pt generation {_generation_of(files[1])}",
        flush=True,
    )

    result = subprocess.run(
        [
            sys.executable, "-m", "neural.duel", str(files[0]), str(files[1]),
            "--games", str(games), "--seed", str(seed),
            "--channels", str(channels), "--blocks", str(blocks),
            "--incumbent-channels", str(incumbent_channels),
            "--incumbent-blocks", str(incumbent_blocks),
        ],
        cwd="/src",
        env=_environment(),
        capture_output=True,
        text=True,
    )
    answer = (result.stdout or "") + (result.stderr or "" if result.returncode else "")
    print(answer, flush=True)
    return answer


@app.function(
    gpu="L40S",
    cpu=16.0,
    memory=32768,
    timeout=60 * 60,
    volumes={str(VOLUME): volume},
)
def discriminate(
    which: str = "latest",
    decisions: int = 40,
    worlds: int = 40,
    candidates: int = 4,
    warmup: int = 24,
    run: str = "w320-run",
) -> str:
    """Whether the value heads can tell two candidate moves apart.

    The search's whole comparison is between positions differing by one
    discarded tile, and a head can predict the return well while being
    nearly blind to that. This reports the separation against the error on
    it, per head, in minutes rather than the two hours a search arm takes.
    The search values positions the engine encodes, so this is for the
    lineage that sees the engine's planes.
    """
    volume.reload()
    source = _checkpoint(run, which)
    if not source.exists():
        return f"no checkpoint at {source}"
    local = Path("/scratch/discriminate")
    local.mkdir(parents=True, exist_ok=True)
    name = Path(which).name
    shutil.copyfile(source, local / f"{name}.pt")

    result = subprocess.run(
        [
            sys.executable, "-m", "neural.discriminate", str(local / f"{name}.pt"),
            "--decisions", str(decisions), "--worlds", str(worlds),
            "--candidates", str(candidates), "--warmup", str(warmup),
            "--channels", "320", "--blocks", "20",
        ],
        cwd="/src",
        env=_environment(),
        capture_output=True,
        text=True,
    )
    answer = (result.stdout or "") + (result.stderr or "" if result.returncode else "")
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
    teacher: str = "w320-run/published",
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
    strategy at all. The teacher may be of the other lineage, since each is
    served the planes it sees, and an empty teacher name means the
    heuristic player that ships with the game. The student learns the
    teacher's whole distribution rather than its choice alone.

    With `student_planes` set to `engine` the student reads the engine's
    ninety-seven planes instead of Mortal's thousand, which is the only
    kind of network the browser can run: a teacher of either lineage can
    then be distilled into something the page can play.

    The result goes to the run's directory under `name`, so it can be
    duelled before anything decides to train on from it.
    """
    volume.reload()
    local = Path("/scratch/distil")
    local.mkdir(parents=True, exist_ok=True)
    out = Path("/scratch/distilled")
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
            return f"no checkpoint at {source}"
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
    answer = (result.stdout or "")[-4000:] + (result.stderr or "" if result.returncode else "")
    if (out / "latest.pt").exists():
        target = VOLUME / run
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "latest.pt", target / f"{name}.pt")
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
def main(generations: int = 40, games: int = 1024, batch: int = 4096) -> None:
    print(train.remote(generations=generations, games=games, batch=batch))
