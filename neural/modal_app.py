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

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

from neural.training_safety import SEARCH_API_VERSION, training_control_arguments
from neural.checkpoints import copy_checkpoint, publish_training_snapshot, validate_checkpoint
from neural.cloud_runs import workspace, validate_run, managed_process
from neural.cloud_requests import validate_cloud_request, stage_opponents
from neural.recordings import atomic_json, copy_recording, experiment, resolve_recording

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
    another run's, so `w320-run/published` reaches across lineages."""
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


def _publish(run: Path, generation: int, target_run: str) -> int:
    """Publish only copied, fully validated checkpoints of completed generations."""
    validate_run(target_run)
    actual = publish_training_snapshot(run, VOLUME / target_run, generation)
    volume.commit()
    return actual


def _generation_of(checkpoint: Path) -> int:
    """Missing starts at zero; an unreadable existing checkpoint is an error."""
    if not checkpoint.exists():
        return 0
    return validate_checkpoint(checkpoint, require_generation=True)


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
    validate_cloud_request(generations, opponents, opponent_share)
    controls = training_control_arguments(target_kl, baseline_batch)
    with workspace(run) as where:
        volume.reload()
        started_from = None
        source = _checkpoint(run, resume)
        if source.exists():
            started_from = str(source)
            copy_checkpoint(source, where / "latest.pt", require_generation=True)
            for saved_name in ("best.pt", "reference.pt"):
                saved = VOLUME / run / saved_name
                # Cross-lineage starts must not inherit this run's old baseline.
                if source.parent == VOLUME / run and saved.exists():
                    copy_checkpoint(saved, where / saved_name)
            # Carry the history forward so a resumed run appends to it rather
            # than starting a fresh record every container.
            for name in ("log.jsonl", "train.log"):
                history = VOLUME / run / name
                if source.parent == VOLUME / run and history.exists():
                    shutil.copyfile(history, where / name)
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
        # lineage's, so "history/gen-00290" and "w320-run/published" both work.
        # Measured 6 September: the old run recovered fully against its own
        # past and only halfway against a foreign network, so a large part of
        # what it gained was knowing its own family. An older self is foreign
        # enough to be worth playing, and another lineage more so.
        command += stage_opponents(
            where, opponents, opponent_share, lambda name: _checkpoint(run, name)
        )

        environment = _environment(TRAINER_CPUS)
        command += controls
        print(" ".join(command), flush=True)

        saved_cache = False
        began = time.time()
        started_at = _generation_of(where / "latest.pt")
        seen = started_at
        print(f"the checkpoint says generation {started_at}", flush=True)
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
            for line in process.stdout:
                print(line.rstrip(), flush=True)
                # A finished generation is a JSON record; publish on each one so a
                # preempted container costs a single round.
                if line.startswith("{") and '"generation"' in line:
                    try:
                        import json

                        record = json.loads(line)
                        seen = int(record.get("checkpoint_generation", record["generation"] + 1))
                    except Exception:
                        continue
                    seen = _publish(where, seen, run)
                    if not saved_cache:
                        _save_cache()
                        saved_cache = True
            code = process.wait()
        _save_cache()
        # Publish only when a generation actually finished. A run that trained
        # nothing still rewrites its checkpoint with the target generation
        # stamped on it, and publishing that overwrote a good record with one
        # claiming to be two hundred generations younger.
        if code == 0 and seen > started_at:
            seen = _publish(where, seen, run)
        else:
            print(f"no final publication (exit={code}); last completed generation {seen}", flush=True)
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
    target_kl: float = 0.0,
    baseline_batch: int | None = None,
) -> str:
    """Fine-tunes a published Mortal on our rules by self-play, in a run
    directory of its own: see `neural/train_mortal.py`. Resumes from the
    checkpoint of that name in the run when it is there, and starts from
    the published Mortal named otherwise. Its own function, so it runs
    beside the other lineage's training rather than queueing behind it.
    """
    validate_cloud_request(generations, opponents, opponent_share)
    controls = training_control_arguments(target_kl, baseline_batch)
    with workspace(run) as where:
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
            copy_checkpoint(source, where / "latest.pt", require_generation=True)
            for saved_name in ("best.pt", "reference.pt"):
                saved = VOLUME / run / saved_name
                # Cross-lineage starts must not inherit this run's old baseline.
                if source.parent == VOLUME / run and saved.exists():
                    copy_checkpoint(saved, where / saved_name)
            history = VOLUME / run / "log.jsonl"
            if source.parent == VOLUME / run and history.exists():
                shutil.copyfile(history, where / "log.jsonl")
            else:
                (where / "log.jsonl").unlink(missing_ok=True)
            command += ["--resume", str(where / "latest.pt")]
            print(f"resuming from {source}", flush=True)
        else:
            origin = _checkpoint(run, mortal)
            if not origin.exists():
                return f"no Mortal at {origin}"
            shutil.copyfile(origin, where / "origin.pt")
            command += ["--mortal", str(where / "origin.pt")]
            print(f"starting from {origin}", flush=True)

        command += stage_opponents(
            where, opponents, opponent_share, lambda name: _checkpoint(run, name)
        )
        command += controls
        print(" ".join(command), flush=True)

        saved_cache = False
        began = time.time()
        started_at = _generation_of(where / "latest.pt")
        seen = started_at
        with managed_process(subprocess.Popen(
            command,
            cwd="/src",
            env=_environment(TRAINER_CPUS),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line.rstrip(), flush=True)
                if line.startswith("{") and '"generation"' in line:
                    try:
                        import json

                        record = json.loads(line)
                        seen = int(record.get("checkpoint_generation", record["generation"] + 1))
                    except Exception:
                        continue
                    seen = _publish(where, seen, run)
                    if not saved_cache:
                        _save_cache()
                        saved_cache = True
            code = process.wait()
        _save_cache()
        if code == 0 and seen > started_at:
            seen = _publish(where, seen, run)
        else:
            print(f"no final publication (exit={code}); last completed generation {seen}", flush=True)
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
    explore: float = 0.0,
    run: str = "joined-run",
    compile: bool = True,
    target_kl: float = 0.0,
    baseline_batch: int | None = None,
) -> str:
    """Trains the joined player, our network and a Mortal beneath one
    fusion head, in a run directory of its own: see
    `neural/train_combined.py`. Resumes from the checkpoint of that name
    in the run when it is there, and otherwise joins the two checkpoints
    named, which may be any run's.
    """
    # Named after the run: a container that has already trained another
    # must not leave its log where this one will append to it.
    validate_cloud_request(generations, opponents, opponent_share)
    controls = training_control_arguments(target_kl, baseline_batch)
    with workspace(run) as where:
        volume.reload()
        source = _checkpoint(run, resume)
        command = [
            sys.executable, "-m", "neural.train_combined",
            "--rounds", str(generations), "--generations", "1000000",
            "--games", str(games), "--batch", str(batch), "--epochs", str(epochs),
            "--lr", str(lr), "--lr-ours", str(lr_ours), "--lr-mortal", str(lr_mortal),
            "--entropy", str(entropy), "--leash", str(leash),
            "--explore", str(explore),
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
            copy_checkpoint(source, where / "latest.pt", require_generation=True)
            for saved_name in ("best.pt", "reference.pt"):
                saved = VOLUME / run / saved_name
                # Cross-lineage starts must not inherit this run's old baseline.
                if source.parent == VOLUME / run and saved.exists():
                    copy_checkpoint(saved, where / saved_name)
            history = VOLUME / run / "log.jsonl"
            if source.parent == VOLUME / run and history.exists():
                shutil.copyfile(history, where / "log.jsonl")
            else:
                (where / "log.jsonl").unlink(missing_ok=True)
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

        command += stage_opponents(
            where, opponents, opponent_share, lambda name: _checkpoint(run, name)
        )
        command += controls
        print(" ".join(command), flush=True)

        saved_cache = False
        began = time.time()
        started_at = _generation_of(where / "latest.pt")
        seen = started_at
        with managed_process(subprocess.Popen(
            command,
            cwd="/src",
            env=_environment(TRAINER_CPUS),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line.rstrip(), flush=True)
                if line.startswith("{") and '"generation"' in line:
                    try:
                        import json

                        record = json.loads(line)
                        seen = int(record.get("checkpoint_generation", record["generation"] + 1))
                    except Exception:
                        continue
                    seen = _publish(where, seen, run)
                    if not saved_cache:
                        _save_cache()
                        saved_cache = True
            code = process.wait()
        _save_cache()
        if code == 0 and seen > started_at:
            seen = _publish(where, seen, run)
        else:
            print(f"no final publication (exit={code}); last completed generation {seen}", flush=True)
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


#: Processors for the search container. `_environment` must be told, or
#: the engine's thread pool stays at sixteen whatever the container has.
SEARCH_CPUS = 16


@app.function(
    # Any of these cards will do: the search with the network moving every
    # seat is bound by Mortal's encoder, which runs on the processors, and a
    # chair of a hundred deals at sixteen worlds, each played to the end of
    # its hand, took 5.8 hours on sixteen of them. Thirty-two processors
    # with 64 GB beside an L40S was a shape Modal could not place for a day.
    gpu=["L40S", "A10G", "A100-40GB"],
    cpu=SEARCH_CPUS,
    memory=32768,
    # A half-chair of fifty deals at sixty-four worlds, on sixteen processors.
    timeout=24 * 60 * 60,
    volumes={str(VOLUME): volume},
)
def searched(
    which: str = "latest",
    games: int = 120,
    worlds: int = 16,
    candidates: int = 4,
    margin: float = 2.0,
    pool: int = 4,
    run: str = DEFAULT_RUN,
    played_by: str = "club",
    depth: int = 0,
    chair: int = -1,
    record: bool = False,
    sure: float = 1.0,
    save_every: int = 600,
    seed: int = 90_210,
    temperature: float = 0.0,
    valued_by: str = "critic",
    leaf_batch: int = 256,
    placement_head: str | None = None,
    objective: str = "hybrid",
    search_calls: bool = False,
    confirm_worlds: int = 0,
    extra_candidates: int = 0,
    audit_share: float = 0.0,
) -> str:
    """Evaluate search, keeping progress separate from successful complete runs.

    `placement_head` names a head written by `neural.placement` on the volume,
    by its path from the root or the run; `valued_by="placement"` judges the
    leaves with it at the hand after the root hand.

    Every invocation has an immutable identity bound to the copied checkpoint
    digest and its complete configuration. Progress survives interruption under
    progress/; only a successful child with a validated complete manifest can
    publish under complete/. Neither another run nor another generation is ever
    deleted. Filesystems that cannot publish atomically fail rather than falling
    back to overwriting a live recording in place.
    """
    import math
    from neural.recordings import digest_file, validate_snapshot

    from neural.teacher_options import validate_controls
    validate_controls(objective=objective, valued_by=valued_by, played_by=played_by,
                      depth=depth, search_calls=search_calls, confirm_worlds=confirm_worlds,
                      extra_candidates=extra_candidates, audit_share=audit_share)
    validate_run(run)
    for name, value in (("games", games), ("worlds", worlds), ("candidates", candidates),
                        ("pool", pool), ("leaf_batch", leaf_batch)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0 or seed + games > 2**64:
        raise ValueError("seed range must fit unsigned 64-bit game seeds")
    if type(depth) is not int or type(chair) is not int or not -1 <= chair < 4:
        raise ValueError("depth must be an integer and chair must be -1 or 0..3")
    for name, value in (("margin", margin), ("temperature", temperature), ("save_every", save_every)):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(sure) or not 0 <= sure <= 1:
        raise ValueError("sure must be a probability")
    if played_by not in ("club", "network") or valued_by not in ("critic", "public", "mean", "placement"):
        raise ValueError("invalid rollout policy or value head")
    if (valued_by == "placement") != (placement_head is not None):
        raise ValueError("valued_by placement needs a placement head, and a head needs that valuing")
    volume.reload()
    source = _checkpoint(run, which)
    if not source.exists():
        raise FileNotFoundError(f"no checkpoint at {source}")
    with workspace("searched") as where:
        copied = where / "checkpoint.pt"
        generation = copy_checkpoint(source, copied, require_generation=True)
        head_copy = None
        if placement_head is not None:
            head_source = _checkpoint(run, placement_head)
            if not head_source.exists():
                raise FileNotFoundError(f"no placement head at {head_source}")
            head_copy = where / "placement.pt"
            shutil.copyfile(head_source, head_copy)
        settings = dict(run=run, which=which, games=games, seed=seed, worlds=worlds,
                        candidates=candidates, margin=margin, pool=pool, played_by=played_by,
                        depth=depth, chair=chair, sure=sure, save_every=save_every,
                        temperature=temperature, valued_by=valued_by, leaf_batch=leaf_batch,
                        device="cuda", placement_head=placement_head,
                        objective=objective, search_api_version=SEARCH_API_VERSION, search_calls=search_calls, confirm_worlds=confirm_worlds,
                        extra_candidates=extra_candidates, audit_share=audit_share,
                        placement_head_sha256=None if head_copy is None else digest_file(head_copy))
        identity = experiment(copied, generation, settings)
        target = VOLUME / "searched-records" / identity["experiment_id"]
        if record:
            atomic_json(target / "experiment.json", identity)
            volume.commit()
        command = [
            sys.executable, "-m", "neural.searched", str(copied),
            "--games", str(games), "--seed", str(seed), "--worlds", str(worlds),
            "--candidates", str(candidates), "--margin", str(margin),
            "--pool", str(pool), "--played-by", played_by, "--depth", str(depth),
            "--chair", str(chair), "--sure", str(sure), "--save-every", str(save_every),
            "--temperature", str(temperature), "--valued-by", valued_by,
            "--leaf-batch", str(leaf_batch), "--device", "cuda",
            "--objective", objective, "--confirm-worlds", str(confirm_worlds),
            "--extra-candidates", str(extra_candidates), "--audit-share", str(audit_share),
        ]
        if search_calls:
            command += ["--search-calls"]
        if head_copy is not None:
            command += ["--placement-head", str(head_copy)]
        records = where / "records"
        kept_snapshot = None

        def keep_records(*, complete=False):
            nonlocal kept_snapshot
            try:
                snapshot = resolve_recording(records)
            except FileNotFoundError:
                if complete:
                    raise ValueError("Successful search produced no complete recording") from None
                return
            meta = validate_snapshot(snapshot, require_complete=complete)
            key = (meta["snapshot_id"], complete)
            if key == kept_snapshot:
                return
            copy_recording(snapshot, target / ("complete" if complete else "progress"),
                           require_complete=complete)
            volume.commit()
            kept_snapshot = key
            print(f"{meta['rows']} recorded decisions kept at {target}", flush=True)

        if record:
            command += ["--record", str(records)]
        log = where / "searched.log"
        with log.open("w", encoding="utf-8") as sink:
            with managed_process(subprocess.Popen(
                command, cwd="/src", env=_environment(SEARCH_CPUS), stdout=sink, stderr=subprocess.STDOUT,
            )) as process:
                while True:
                    try:
                        returncode = process.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        if record:
                            keep_records()
        answer = log.read_text(encoding="utf-8", errors="replace")
        if record:
            atomic_json(target / "result.json", {"returncode": returncode, "output": answer})
            volume.commit()
        if returncode:
            # Valid progress may already have been kept, but it is not a
            # successful experiment and never replaces an accepted dataset.
            raise subprocess.CalledProcessError(returncode, command, output=answer)
        if record:
            keep_records(complete=True)
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
    challenger_head: str | None = None,
    head_k: int = 4,
    head_margin: float = 0.05,
    head_sure: float | None = None,
) -> str:
    """Sits two checkpoints from the volume at the same table.

    With `challenger_head`, a sibling head on the volume (a `.pt` written
    by `neural.sibling_head`, named by its path from the volume's root),
    the challenger plays with the head's second opinion over its first
    `head_k` moves, taking the head's favourite past `head_margin`, and
    only where the policy's confidence is below `head_sure` (by default
    the gate the head's recordings were made under); see `neural.ranked`.

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

    if challenger_head is None:
        command = [
            sys.executable, "-m", "neural.duel", str(files[0]), str(files[1]),
            "--games", str(games), "--seed", str(seed),
            "--channels", str(channels), "--blocks", str(blocks),
            "--incumbent-channels", str(incumbent_channels),
            "--incumbent-blocks", str(incumbent_blocks),
        ]
    else:
        # The challenger with a sibling head's second opinion, against the
        # same network plain: the head's worth, at one table.
        head = _checkpoint(run, challenger_head)
        if not head.exists():
            return f"no head at {head}"
        head_copy = local / ("head--" + challenger_head.replace("/", "--") + ".pt")
        shutil.copyfile(head, head_copy)
        command = [
            sys.executable, "-m", "neural.ranked", str(files[0]), str(head_copy),
            "--games", str(games), "--seed", str(seed),
            "--k", str(head_k), "--margin", str(head_margin),
        ]
        if head_sure is not None:
            command += ["--sure", str(head_sure)]
    result = subprocess.run(
        command,
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
    cpu=8.0,
    memory=49152,
    timeout=4 * 60 * 60,
    volumes={str(VOLUME): volume},
)
def teach(
    recordings: list[str],
    checkpoint: str = "leashed-run/latest",
    out: str = "taught/latest",
    epochs: int = 6,
    lr: float = 1e-3,
    batch: int = 128,
    target: str = "decision",
    temperature: float = 0.1,
    leash: float = 4.0,
    hold: str = "mortal+ours",
    rows: str = "changed",
    weighted: bool = False,
    emphasis: float = 1.0,
    without_mask: bool = False,
    duel_games: int = 0,
    seed: int = 555_000,
) -> str:
    """Teaches the policy what the search found (`neural.teach`) from
    recordings on the volume, keeps the taught checkpoint at `out`, and
    with `duel_games` sits it against the network it was taught from at
    one table, which is the figure that settles whether the lesson was
    worth learning.
    """
    volume.reload()
    folders = [VOLUME / name for name in recordings]
    for folder in folders:
        folder = resolve_recording(folder)
        if not (folder / "meta.json").exists():
            return f"no recording at {folder}"
    run = checkpoint.rpartition("/")[0] or DEFAULT_RUN
    source = _checkpoint(run, checkpoint.rpartition("/")[2])
    if not source.exists():
        return f"no checkpoint at {source}"
    answer = ""
    with workspace("teach") as where:
        copied = where / "network.pt"
        copy_checkpoint(source, copied, require_generation=True)
        taught = where / "taught.pt"
        command = [
            sys.executable, "-m", "neural.teach", *[str(folder) for folder in folders],
            str(copied), "--out", str(taught), "--epochs", str(epochs), "--lr", str(lr),
            "--batch", str(batch), "--target", target, "--temperature", str(temperature),
            "--leash", str(leash), "--hold", hold, "--rows", rows,
            "--emphasis", str(emphasis), "--device", "cuda",
        ]
        if weighted:
            command.append("--weighted")
        if without_mask:
            command.append("--without-mask")
        print(" ".join(command), flush=True)
        result = subprocess.run(command, cwd="/src", env=_environment(8), capture_output=True, text=True)
        answer += (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0 or not taught.exists():
            print(answer, flush=True)
            return answer
        target = VOLUME / (out + ".pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(taught, target)
        volume.commit()
        answer += f"\ntaught network kept at {target}"
        if duel_games:
            # The taught policy against the one it was taught from, same
            # deals, one table: what the lesson was actually worth.
            duelled = subprocess.run(
                [
                    sys.executable, "-m", "neural.duel", str(taught), str(copied),
                    "--games", str(duel_games), "--seed", str(seed),
                ],
                cwd="/src", env=_environment(8), capture_output=True, text=True,
            )
            answer += "\n" + (duelled.stdout or "") + (duelled.stderr or "" if duelled.returncode else "")
    print(answer, flush=True)
    return answer


@app.function(
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={str(VOLUME): volume},
)
def train_head(
    recordings: list[str],
    checkpoint: str = "leashed-run/latest",
    out: str = "heads/latest-sibling",
    epochs: int = 10,
    lr: float = 1e-3,
    batch: int = 256,
    weighted: bool = False,
    target: str = "decision",
) -> str:
    """Trains the sibling head (`neural.sibling_head`) on recordings the
    search kept on the volume -- directories under `searched-records/`,
    named by their path from the volume's root -- against the network the
    recordings were made with, and keeps the head at `out` on the volume.
    """
    volume.reload()
    folders = [VOLUME / name for name in recordings]
    for folder in folders:
        folder = resolve_recording(folder)
        if not (folder / "meta.json").exists():
            return f"no recording at {folder}"
    source = _checkpoint(checkpoint.rpartition("/")[0] or DEFAULT_RUN, checkpoint.rpartition("/")[2])
    if not source.exists():
        return f"no checkpoint at {source}"
    with workspace("train-head") as where:
        copied = where / "network.pt"
        copy_checkpoint(source, copied, require_generation=True)
        head = where / "head.pt"
        command = [
            sys.executable, "-m", "neural.sibling_head", *[str(folder) for folder in folders],
            str(copied), "--out", str(head), "--epochs", str(epochs), "--lr", str(lr),
            "--batch", str(batch), "--device", "cuda",
        ]
        if weighted:
            command.append("--weighted")
        command += ["--target", target]
        print(" ".join(command), flush=True)
        result = subprocess.run(command, cwd="/src", env=_environment(8), capture_output=True, text=True)
        answer = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0 and head.exists():
            target = VOLUME / (out + ".pt")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(head, target)
            volume.commit()
            answer += f"\nhead kept at {target}"
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
        if result.returncode == 0 and (out / "latest.pt").exists():
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
    anything is built on it.
    """
    with workspace(run) as where:
        volume.reload()
        out = where / "out"
        source = _checkpoint(run, teacher)
        if not source.exists():
            return f"no checkpoint at {source}"
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
                return f"no checkpoint at {found}"
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
        answer = (result.stdout or "")[-4000:] + (result.stderr or "" if result.returncode else "")
        if result.returncode == 0 and (out / "latest.pt").exists():
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
