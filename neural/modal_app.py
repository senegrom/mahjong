"""Self-play training on Modal.

The desktop card runs the learning step at about a tenth of its arithmetic:
the same generation took 481 milliseconds a step one round and 825 the
next, because the process runs at lowest priority on a machine whose
terminal holds half the cores. Nothing in the loop is wrong; the machine
is. This runs the same loop on a container that has the card to itself.

The image builds the rules engine from source, because self-play is the
Rust extension and it must match the checkpoint's observation exactly.

    modal deploy neural/modal_app.py
    modal run neural/modal_app.py::smoke
    modal run --detach neural/modal_app.py::train --generations 40

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

# Debian with a Rust toolchain, because the engine is compiled and the
# wheel has to be built for the container's own libc.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config", "git")
    .run_commands(
        "curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable --profile minimal"
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("torch==2.8.0", "numpy>=2.0", "maturin>=1.7")
    # The sources the wheel is built from, and the training package itself.
    .add_local_dir(HERE / "engine", "/src/engine", copy=True, ignore=["**/target", "**/pkg"])
    .add_local_file(HERE / "Cargo.toml", "/src/Cargo.toml", copy=True)
    .add_local_file(HERE / "Cargo.lock", "/src/Cargo.lock", copy=True)
    .run_commands(
        "cd /src/engine/riichi-py && maturin build --release --out /src/wheels",
        "pip install /src/wheels/*.whl",
    )
    # The training code last, so editing it does not rebuild the engine.
    .add_local_dir(HERE / "neural", "/src/neural", copy=True, ignore=["**/__pycache__"])
    .env({"PYTHONPATH": "/src", "PYTHONUNBUFFERED": "1"})
)

app = modal.App("mahjong-train", image=image)
volume = modal.Volume.from_name("mahjong-train", create_if_missing=True)
VOLUME = Path("/vol")
RUN = Path("/scratch/run")


def _publish(run: Path, generation: int) -> None:
    """Copies the checkpoints and the log to the volume and commits them."""
    target = VOLUME / "w320-run"
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


@app.function(
    gpu="H100",
    # Self-play is the long pole here, not the card: a generation is about
    # 180 seconds of which 80 are playing, and the card idles through them.
    # Thirty-two processors were tried and made it worse, 93 seconds
    # against 78, so the engine does not scale past sixteen on this shape
    # of work and the extra ones only cost.
    cpu=16.0,
    memory=98304,
    timeout=24 * 60 * 60,
    volumes={str(VOLUME): volume},
    max_containers=1,
)
def train(
    generations: int = 40,
    games: int = 2048,
    batch: int = 8192,
    epochs: int = 2,
    lr: float = 4e-5,
    entropy: float = 0.0,
    measure_every: int = 5,
    measure_games: int = 1024,
    replay_rounds: int = 8,
    replay_steps: int = 180,
    resume: str = "latest.pt",
) -> str:
    """Runs `generations` rounds of self-play and learning, resuming from the
    checkpoint of that name on the volume when it is there.

    The defaults are not the desktop's. A container with the card to itself
    and sixteen processors can play four times the games and hold four
    times the batch, and fresh games are the thing the desktop run is short
    of: it passes over each round three times and its critic learns the
    round by heart.
    """
    RUN.mkdir(parents=True, exist_ok=True)
    volume.reload()
    started_from = None
    source = VOLUME / "w320-run" / resume
    if source.exists():
        started_from = str(source)
        shutil.copyfile(source, RUN / "latest.pt")
        # Carry the history forward so a resumed run appends to it rather
        # than starting a fresh record every container.
        for name in ("log.jsonl", "train.log"):
            history = VOLUME / "w320-run" / name
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
        "--amp",
        "--compile",
        "--out",
        str(RUN),
    ]
    if (RUN / "latest.pt").exists():
        command += ["--resume", str(RUN / "latest.pt")]

    environment = dict(os.environ)
    environment["RAYON_NUM_THREADS"] = str(int(os.cpu_count() or 16))
    environment["PYTHONPATH"] = "/src"
    print(" ".join(command), flush=True)

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
            _publish(RUN, seen)
    code = process.wait()
    # Publish only when a generation actually finished. A run that trained
    # nothing still rewrites its checkpoint with the target generation
    # stamped on it, and publishing that overwrote a good record with one
    # claiming to be two hundred generations younger.
    if seen > started_at:
        _publish(RUN, seen)
    else:
        print(f"nothing trained: leaving the volume at generation {started_at}", flush=True)
    return (
        f"exit={code} generation={seen} from {started_at} "
        f"after {time.time() - began:.0f}s"
    )


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
    blocks: int = 20,
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
    source = VOLUME / "w320-run" / f"{which}.pt"
    if not source.exists():
        return f"no checkpoint at {source}"
    local = Path("/scratch/arena")
    local.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, local / f"{which}.pt")
    # Say which generation this is. The report named only the file it
    # copied, so two runs on an unchanged checkpoint were indistinguishable
    # from two on different ones, and at a fixed seed they give the same
    # number: one of them was a container spent to learn nothing.
    generation = _generation_of(local / f"{which}.pt")
    print(f"measuring {which}.pt, generation {generation}", flush=True)

    environment = dict(os.environ)
    environment["RAYON_NUM_THREADS"] = str(int(os.cpu_count() or 16))
    environment["PYTHONPATH"] = "/src"
    result = subprocess.run(
        [
            sys.executable, "-m", "neural.arena", str(local / f"{which}.pt"),
            "--games", str(games), "--seed", str(seed),
            # The width is the caller's to say, so the network the browser
            # plays can be measured on the same deals as the one being
            # trained. That pairing is the whole point of the same seed.
            "--channels", str(channels), "--blocks", str(blocks),
        ],
        cwd="/src",
        env=environment,
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
    challenger: str = "best",
    incumbent: str = "published",
    games: int = 1000,
    seed: int = 555_000,
    channels: int = 320,
    blocks: int = 20,
    incumbent_channels: int = 192,
    incumbent_blocks: int = 10,
) -> str:
    """Sits two checkpoints from the volume at the same table.

    Measuring each against the heuristic players and subtracting has a
    floor of about 0.024 on the difference, so two close networks never
    separate. At one table the luck of the deal falls on both at once, and
    what comes back is a single placement against the 2.50 two identical
    players would average.
    """
    volume.reload()
    local = Path("/scratch/duel")
    local.mkdir(parents=True, exist_ok=True)
    for name in (challenger, incumbent):
        source = VOLUME / "w320-run" / f"{name}.pt"
        if not source.exists():
            return f"no checkpoint at {source}"
        shutil.copyfile(source, local / f"{name}.pt")
    print(
        f"challenger {challenger}.pt generation "
        f"{_generation_of(local / f'{challenger}.pt')} against {incumbent}.pt",
        flush=True,
    )

    environment = dict(os.environ)
    environment["RAYON_NUM_THREADS"] = str(int(os.cpu_count() or 16))
    environment["PYTHONPATH"] = "/src"
    result = subprocess.run(
        [
            sys.executable, "-m", "neural.duel",
            str(local / f"{challenger}.pt"), str(local / f"{incumbent}.pt"),
            "--games", str(games), "--seed", str(seed),
            "--channels", str(channels), "--blocks", str(blocks),
            "--incumbent-channels", str(incumbent_channels),
            "--incumbent-blocks", str(incumbent_blocks),
        ],
        cwd="/src",
        env=environment,
        capture_output=True,
        text=True,
    )
    answer = (result.stdout or "") + (result.stderr or "" if result.returncode else "")
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
    """A tiny round end to end: the engine loads, self-play runs, the
    learning step runs, a checkpoint is written. Cheap, and it fails for
    the same reasons the real run would."""
    import torch

    import riichi_py

    report = [
        f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        f"riichi_py planes={riichi_py.PLANES} actions={riichi_py.ACTIONS} "
        f"oracle={riichi_py.ORACLE_PLANES}",
        f"cpus={os.cpu_count()}",
    ]
    out = Path("/scratch/smoke")
    out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["RAYON_NUM_THREADS"] = str(int(os.cpu_count() or 8))
    environment["PYTHONPATH"] = "/src"
    result = subprocess.run(
        [
            sys.executable, "-m", "neural.train",
            "--generations", "1", "--games", "24", "--batch", "512", "--epochs", "1",
            "--measure-every", "1", "--measure-games", "8",
            "--replay-rounds", "2", "--replay-steps", "4",
            "--channels", "320", "--blocks", "20", "--amp",
            "--out", str(out),
        ],
        cwd="/src",
        env=environment,
        capture_output=True,
        text=True,
        timeout=40 * 60,
    )
    report.append(f"train exit={result.returncode}")
    report.append((result.stdout or "")[-2500:])
    if result.returncode != 0:
        report.append((result.stderr or "")[-2500:])
    answer = "\n".join(report)
    # Printed as well as returned: the container's output is what streams
    # back to a `modal run`, and a return value alone shows nothing.
    print(answer, flush=True)
    return answer


@app.local_entrypoint()
def main(generations: int = 40, games: int = 2048, batch: int = 8192) -> None:
    print(train.remote(generations=generations, games=games, batch=batch))
