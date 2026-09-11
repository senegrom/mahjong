"""Is the critic worse where a search asks it, and does the oracle help?

A one-ply search replaces the policy's choice with the critic's opinion of
the positions each candidate leads to. Measured, that made this network
worse in proportion to how often the opinion was taken: at a two-error bar
it overrode a third of decisions and lost 0.19 placement; at no bar at all,
a half and 0.43. A critic that were merely noisy would not do that, because
a two-error bar filters noise. The standing explanation is that the error
is a bias which correlates with the move being considered — the value head
has only ever seen positions the policy actually reached, and a search asks
it about the ones the policy avoids.

That is a hypothesis, and this measures it. A round is played with some
share of its moves forced: a legal move taken at random rather than the
policy's. The positions that follow a forced move are the positions a
search asks about. Both are scored against what the game really paid.

Two numbers come out.

**Does the error grow off the policy's path?** The critic's error on
decisions the policy made, against its error on decisions that were forced.
If they are the same, the standing explanation is wrong and the search's
failure needs another one. If the forced ones are much worse, coverage is
the fault and widening where self-play goes is the fix.

**Does the oracle survive out there?** The oracle critic sees the hidden
tiles, so it should predict the return far better; if that advantage
survives on forced positions, it can label them cheaply — one forward pass
instead of playing each alternative out — which is the difference between
an afternoon and a week of compute. If the oracle is equally lost off the
path, labelling with it buys nothing and that road is closed.

    python -m neural.counterfactual <checkpoint> [games=256] [share=0.25]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import selfplay, zoo


def error_of(guess: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """How far off, and how far off a constant would have been.

    The comparison that matters is not the error on its own but the error
    against the spread of what is being predicted. An error the size of the
    spread is a critic saying nothing about the position, only the average
    of the game.
    """
    if guess.size == 0:
        return {"rows": 0}
    residual = truth - guess
    spread = float(truth.std()) if truth.size > 1 else 0.0
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "rows": int(guess.size),
        "rmse": round(rmse, 4),
        "spread": round(spread, 4),
        # One minus the ratio of variances: what share of the spread the
        # critic accounts for. Zero is a constant, one is perfect, and
        # below zero is worse than saying nothing.
        "explained": round(1.0 - (rmse / spread) ** 2, 4) if spread else 0.0,
    }


@torch.no_grad()
def measure(net, games: int, seed: int, share: float, device: str) -> dict:
    """Plays a round with some moves forced and scores the critics on it."""
    batch = selfplay.play(
        net, games=games, seed=seed, device=device, explore_share=share
    )
    if batch.explored is None:
        raise SystemExit("this round recorded nothing about which moves were forced")

    forced = batch.explored.cpu().numpy().astype(bool)
    returns = batch.returns.cpu().numpy()

    # The critic's own reading of every position it recorded.
    guessed = np.empty(batch.decisions, dtype=np.float32)
    step = 4096
    for start in range(0, batch.decisions, step):
        rows = np.arange(start, min(start + step, batch.decisions))
        planes = batch.observations.rows(rows).dense(device)
        legal = batch.legal[rows].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            _logits, value, _hands = net.everything(planes, legal)
        guessed[rows] = value.float().cpu().numpy()

    report = {
        "checkpoint": None,
        "games": games,
        "decisions": batch.decisions,
        "forced_share": round(float(forced.mean()), 4),
        "on the policy's path": error_of(guessed[~forced], returns[~forced]),
        "where it was forced": error_of(guessed[forced], returns[forced]),
    }

    on = report["on the policy's path"]
    off = report["where it was forced"]
    if on.get("rows") and off.get("rows"):
        report["verdict"] = (
            "the critic explains "
            f"{on['explained']:.1%} of the spread where the policy went and "
            f"{off['explained']:.1%} where it was forced elsewhere"
        )
        report["coverage_is_the_fault"] = bool(off["explained"] < on["explained"] - 0.02)
    return report


def main() -> None:
    checkpoint = Path(sys.argv[1])
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 606_060
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = zoo.load_player(checkpoint, device)
    net.eval()
    report = measure(net, games, seed, share, device)
    report["checkpoint"] = str(checkpoint)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
