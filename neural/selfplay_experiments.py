"""Generate or execute the predeclared A-E self-play ablation sequence.

Prints commands by default. --execute explicitly runs them in this process,
sequentially, and refuses to overwrite a run. No cloud job is launched.
A is the new infrastructure control; the original trainers remain separate
legacy controls. A-E share infrastructure so each comparison has a clear change.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

PRESETS = {
    "A": ["--objective", "hybrid", "--critic", "public", "--schedule", "random", "--target-kl", "0", "--table-mix", "1", "0", "0"],
    "B": ["--objective", "placement", "--critic", "public", "--schedule", "random", "--target-kl", "0", "--table-mix", "1", "0", "0"],
    "C": ["--objective", "placement", "--critic", "privileged", "--schedule", "random", "--target-kl", "0", "--table-mix", "1", "0", "0"],
    "D": ["--objective", "placement", "--critic", "privileged", "--schedule", "random", "--target-kl", "0", "--table-mix", ".4", ".4", ".2"],
    "E": ["--objective", "placement", "--critic", "privileged", "--schedule", "staged", "--target-kl", ".01", "--table-mix", ".4", ".4", ".2"],
}


def commands(initial, opponents, out, seeds, *, rounds=20, games=1024, presets=None, mortal=None):
    if not seeds or any(type(value) is not int or value <= 0 for value in (rounds, games)):
        raise ValueError("nonempty seeds and positive rounds/games are required")
    if len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("training seeds must be distinct nonnegative integers")
    selected = list(presets or PRESETS)
    if any(name not in PRESETS for name in selected) or not selected:
        raise ValueError("choose presets A-E")
    if any(name in selected for name in ("D", "E")) and not opponents:
        raise ValueError("D/E require frozen opponents")
    tasks = []
    for label in selected:
        for seed in seeds:
            target = Path(out) / f"{label}-s{seed}"
            command = [sys.executable, "-m", "neural.train_league", "--initial", str(initial),
                       "--out", str(target), "--seed", str(seed), "--rounds", str(rounds),
                       "--games", str(games), "--seed-ledger", str(Path(out) / "seeds.json"),
                       "--advantage", "mc", *PRESETS[label]]
            if opponents:
                command += ["--opponents", *map(str, opponents)]
            tasks.append((target, command))
    if mortal is not None:
        for seed in seeds:
            target = Path(out) / f"Mortal-control-s{seed}"
            command = [sys.executable, "-m", "neural.train_league", "--initial", str(mortal),
                       "--out", str(target), "--seed", str(seed), "--rounds", str(rounds),
                       "--games", str(games), "--seed-ledger", str(Path(out) / "seeds.json"),
                       "--schedule", "joint", "--objective", "placement", "--critic", "privileged"]
            if opponents:
                command += ["--opponents", *map(str, opponents)]
            tasks.append((target, command))
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("initial", type=Path)
    p.add_argument("--opponents", type=Path, nargs="*", default=[])
    p.add_argument("--mortal", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--presets", choices=list(PRESETS), nargs="+", default=list(PRESETS))
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.execute:
        from .league_state import snapshot
        from .seed_ledger import atomic_json
        inputs = args.out / "inputs"
        originals = [args.initial, *args.opponents] + ([args.mortal] if args.mortal else [])
        pinned = [snapshot(path, inputs) for path in originals]
        args.initial = inputs / pinned[0]["file"]
        args.opponents = [inputs / item["file"] for item in pinned[1:1 + len(args.opponents)]]
        if args.mortal:
            args.mortal = inputs / pinned[-1]["file"]
        atomic_json(args.out / "inputs.json", {"snapshots": pinned})
    tasks = commands(args.initial, args.opponents, args.out, args.seeds, rounds=args.rounds,
                     games=args.games, presets=args.presets, mortal=args.mortal)
    for target, command in tasks:
        print(shlex.join(command), flush=True)
        if args.execute:
            if (target / "latest.pt").exists():
                raise FileExistsError(f"refusing to overwrite {target}")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
