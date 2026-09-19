"""Evaluate immutable candidates on fresh deals with a durable attempt budget.

    python -m neural.league_gate candidate.pt champion.pt --ledger evaluation.json --games 2048

Add --out RUN only to publish a passing evaluated snapshot as RUN/champion.pt.
The default only writes evidence. Never choose a different bound after seeing
its result; a ledger binds the statistical settings of the whole series.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile

from . import gate
from .checkpoints import copy_checkpoint
from .league_state import digest
from .seed_ledger import SeedLedger, atomic_json, bind_protocol


def run(candidate, champion, *, ledger, games=2048, confidence=.95,
        minimum_edge=0.0, minimum_deals=128, method="empirical-bernstein",
        device="cpu", out=None):
    # Validate everything before consuming a seed block or starting inference.
    gate.assess([2.5], [2.5], confidence=confidence, minimum_edge=minimum_edge,
                minimum_deals=minimum_deals, method=method)
    if type(games) is not int or games < minimum_deals:
        raise ValueError("games must be at least minimum_deals")
    with tempfile.TemporaryDirectory(prefix="league-gate-") as temporary:
        sources = [Path(temporary) / name for name in ("candidate.pt", "champion.pt")]
        for source, destination in zip((candidate, champion), sources):
            copy_checkpoint(Path(source), destination)
        candidate_sha, champion_sha = map(digest, sources)
        seeds = SeedLedger(Path(ledger), seed=0)
        bind_protocol(seeds, "promotion", {"confidence": confidence, "minimum_edge": minimum_edge,
                      "minimum_deals": minimum_deals, "method": method})
        reservation = seeds.reserve("promotion", games, f"{candidate_sha}-vs-{champion_sha}")
        report = gate.compare(*sources, games=games, seed=reservation["seed"], device=device,
                              confidence=confidence, minimum_edge=minimum_edge,
                              minimum_deals=minimum_deals, method=method,
                              attempt=reservation["attempt"])
        report.update(seed_reservation=reservation, ledger=str(ledger),
                      evidence_scope="symmetric 1-vs-3 matchup; not strength against arbitrary policies")
        report["candidate"]["path"], report["champion"]["path"] = str(candidate), str(champion)
        evidence = Path(ledger).parent / (Path(ledger).stem + "-reports") / f"attempt-{reservation['attempt']:06d}.json"
        atomic_json(evidence, report)
        if report["promote"] and out is not None:
            from .promote import publish_evaluated_snapshot
            publish_evaluated_snapshot(sources[0], Path(out), report)
            report["checkpoint_written"] = True
            atomic_json(evidence, report)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path)
    p.add_argument("champion", type=Path)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--games", type=int, default=2048)
    p.add_argument("--minimum-deals", type=int, default=128)
    p.add_argument("--minimum-edge", type=float, default=0.0)
    p.add_argument("--confidence", type=float, default=.95)
    p.add_argument("--method", choices=("hoeffding", "empirical-bernstein"), default="empirical-bernstein")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path)
    print(json.dumps(run(**vars(p.parse_args())), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
