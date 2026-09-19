"""A fixed, per-opponent validation/test panel; never an automatic promotion.

    python -m neural.league_evaluate candidate.pt --opponents champion.pt mortal.pt \
        --ledger runs/seeds.json --games 512 --out panel.json

All four seats and both directions are paired by deal. The same held-out panel
is useful for comparisons, but only fresh promotion deals justify publication.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import tempfile
import numpy as np

from . import duel, zoo
from .league_state import snapshot
from .seed_ledger import SeedLedger, atomic_json


def evaluate(candidate, opponents, *, ledger, games=512, domain="validation", device="cpu"):
    if domain not in ("validation", "test") or type(games) is not int or games < 2 or not opponents:
        raise ValueError("a validation/test domain, at least two deals and opponents are required")
    with tempfile.TemporaryDirectory(prefix="league-panel-") as folder:
        folder = Path(folder)
        entries = [snapshot(Path(path), folder) for path in [candidate, *opponents]]
        reservation = SeedLedger(Path(ledger), seed=0).reserve(domain, games, "league-evaluation")
        players = [zoo.load_player(folder / entry["file"], device) for entry in entries]
        rows = []
        for opponent, provenance in zip(players[1:], entries[1:]):
            forward = duel.duel(players[0], opponent, games, reservation["seed"], device)
            reverse = duel.duel(opponent, players[0], games, reservation["seed"], device)
            edge = (np.asarray(reverse["by_deal"]) - np.asarray(forward["by_deal"])) / 2
            rows.append({"opponent": provenance, "candidate_one_vs_three": forward,
                         "opponent_one_vs_three": reverse, "mean_edge": float(edge.mean()),
                         "standard_error": float(edge.std(ddof=1) / np.sqrt(games)),
                         "by_deal_edge": edge.tolist()})
    return {"candidate": entries[0], "seed_reservation": reservation, "domain": domain,
            "panel": rows, "diagnostic_only": True, "promotion": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path)
    p.add_argument("--opponents", type=Path, nargs="+", required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--domain", choices=("validation", "test"), default="validation")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, required=True)
    args = vars(p.parse_args())
    out = args.pop("out")
    atomic_json(out, evaluate(**args))


if __name__ == "__main__":
    main()
