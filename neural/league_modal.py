"""Explicit Modal entry point for the placement/league trainer.

    modal run neural/league_modal.py::train_league --initial joined-run/latest \
        --run placement-run --opponents mortal-run/latest --rounds 20

Importing/deploying does not start training. This uses the existing engine image,
volume, subprocess supervision and atomic checkpoint publisher. No champion is
written. Critic/optimizer state stays inside latest.pt, not separate sidecars.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import sys

from neural.modal_app import app, volume, VOLUME, TRAINER_CPUS, _checkpoint, _environment, _publish
from neural.cloud_runs import managed_process, workspace, validate_run
from neural.checkpoints import copy_checkpoint


@app.function(gpu=["H100", "A100-80GB"], cpu=TRAINER_CPUS, memory=98304,
              timeout=24 * 60 * 60, volumes={str(VOLUME): volume}, max_containers=1)
def train_league(initial: str = "joined-run/latest", run: str = "placement-run",
                 opponents: list[str] | None = None, rounds: int = 20, games: int = 1024,
                 batch: int = 2048, objective: str = "placement", critic: str = "privileged",
                 advantage: str = "mc", gae_lambda: float = .95, target_kl: float = .01,
                 schedule: str = "staged", seed: int = 20260919, compile: bool = False,
                 refresh_opponents: bool = False, reset_critics: bool = False,
                 collectors: int | None = None, inference_batch: int | None = None) -> str:
    validate_run(run)
    if any(type(n) is not int or n <= 0 for n in (rounds, games, batch)):
        raise ValueError("rounds, games and batch must be positive integers")
    if objective not in ("hybrid", "placement") or critic not in ("public", "privileged"):
        raise ValueError("invalid objective or critic")
    if advantage not in ("mc", "gae") or (advantage == "gae" and objective != "placement"):
        raise ValueError("GAE requires the placement objective")
    if collectors is not None and (type(collectors) is not int or not 1 <= collectors <= games):
        raise ValueError("collectors must be an integer in [1,games]")
    if inference_batch is not None and (type(inference_batch) is not int or inference_batch < 1):
        raise ValueError("inference_batch must be a positive integer")
    volume.reload()
    target = VOLUME / run
    resume = target / "latest.pt"
    sources = [_checkpoint(run, name) for name in opponents or []]
    if any(not path.is_file() for path in sources):
        raise FileNotFoundError("a requested opponent is missing")
    with workspace(run) as where:
        command = [sys.executable, "-m", "neural.train_league", "--out", str(where),
                   "--rounds", str(rounds), "--games", str(games), "--batch", str(batch),
                   "--objective", objective, "--critic", critic, "--advantage", advantage,
                   "--gae-lambda", str(gae_lambda), "--target-kl", str(target_kl),
                   "--schedule", schedule, "--seed", str(seed), "--device", "cuda", "--amp",
                   "--seed-ledger", str(VOLUME / "league-seeds.json")]
        if collectors is not None:
            command += ["--collectors", str(collectors)]
        if inference_batch is not None:
            command += ["--inference-batch", str(inference_batch)]
        if resume.exists():
            copy_checkpoint(resume, where / "latest.pt", require_generation=True)
            for name in ("reference.pt", "candidate.pt"):
                if (target / name).exists():
                    copy_checkpoint(target / name, where / name)
            for name in ("log.jsonl", "seeds.json"):
                if (target / name).exists():
                    shutil.copyfile(target / name, where / name)
            if (target / "snapshots").exists():
                shutil.copytree(target / "snapshots", where / "snapshots", dirs_exist_ok=True)
            command += ["--resume", str(where / "latest.pt")]
        else:
            source = _checkpoint(run, initial)
            if not source.is_file():
                raise FileNotFoundError(source)
            copy_checkpoint(source, where / "initial.pt")
            command += ["--initial", str(where / "initial.pt")]
        if sources:
            # The trainer copies/hashes these BEFORE loading any opponent.
            # Stable volume paths preserve roster identity across scratch dirs.
            command += ["--opponents", *map(str, sources)]
        if refresh_opponents:
            command.append("--refresh-opponents")
        if reset_critics:
            command.append("--reset-critics")
        if compile:
            command.append("--compile")
        last = None
        with managed_process(subprocess.Popen(command, cwd="/src", env=_environment(TRAINER_CPUS),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line.rstrip(), flush=True)
                if line.startswith("{"):
                    record = json.loads(line)
                    if "checkpoint_generation" in record:
                        last = _publish(where, int(record["checkpoint_generation"]), run)
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
        if last is None:
            raise RuntimeError("trainer completed without publishing a generation")
        return json.dumps({"generation": last, "run": run, "objective": objective,
                           "critic": critic, "champion_written": False})
