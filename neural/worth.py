"""What a search's rule is worth, measured on worlds it did not use.

A recording holds every candidate's value in every world, so it is
tempting to read off what the search gained as the best candidate's value
less the policy's. That number is mostly selection: the best of four
estimates over a handful of worlds beats the first even when every move
is identical, and the same is true of the two-standard-error margin when
the test and the score come from the same worlds.

So the worlds are split here. Half of them decide -- the paired mean, its
standard error and the margin, exactly as `search::compare` and
`pick_by_margin` do it -- and the other half scores the move that was
picked against the policy's own. Averaged over many splits, the
difference is an unbiased estimate of what the rule gains a decision, in
the units the rollouts spoke in: hand points over four thousand plus the
placement the hand led to. Deciding on half the worlds, it understates a
search that decides on all of them.

It is still the critic's yardstick, not the table's: a leaf is worth
what the value head says, so a rule that gains here has gained by the
network's own reckoning, and only a duel or a placement measurement says
whether that reckoning is right.

    python -m neural.worth searched-records/.../chair0 [...]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

#: What the engine's search asks of a candidate before it will override
#: the policy: this many standard errors of the paired difference.
MARGIN = 2.0

#: How many times the worlds are split. The split is noise of its own,
#: and the question is about the decision.
REPEATS = 16

#: Fewer worlds than this and neither half can say anything.
LEAST_WORLDS = 6


def edge_and_error(mine: np.ndarray, theirs: np.ndarray) -> tuple[float, float]:
    """A candidate's paired mean against the incumbent over the worlds
    both were tried in, and its standard error, as the engine counts them
    with even weights (`search::compare`). An error of infinity means too
    few worlds to say."""
    paired = mine - theirs
    paired = paired[np.isfinite(paired)]
    if len(paired) < 3:
        return 0.0, float("inf")
    mean = float(paired.mean())
    spread = float(((paired - mean) ** 2).sum()) / len(paired) ** 2
    return mean, float(np.sqrt(spread / (1.0 - 1.0 / len(paired))))


def picked_by_margin(table: np.ndarray, valid: np.ndarray, worlds: np.ndarray,
                     margin: float = MARGIN) -> int:
    """Which candidate the engine's rule takes, judging on the worlds
    named: the one with the largest edge over the policy's move that
    clears the margin, or the policy's move."""
    chose, best = 0, 0.0
    for candidate in range(1, len(valid)):
        if not valid[candidate]:
            continue
        edge, error = edge_and_error(table[candidate, worlds], table[0, worlds])
        if np.isfinite(error) and error > 0 and edge > margin * error and edge > best:
            chose, best = candidate, edge
    return chose


def measure(folder: Path, margin: float = MARGIN, repeats: int = REPEATS,
            seed: int = 11) -> dict:
    """What the margin's rule, and taking the best average, gain a
    decision on the recording named."""
    from .recordings import resolve_recording

    folder = resolve_recording(folder)
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    per_world = np.load(folder / "per_world.npy").astype(np.float64)
    values = np.load(folder / "values.npy").astype(np.float64)
    game = np.load(folder / "game.npy")
    drawer = np.random.default_rng(seed)
    rows, width, _count = per_world.shape
    by_margin, by_best, fired = [], [], []
    for row in range(rows):
        table = per_world[row]
        valid = ~np.isnan(values[row])
        if valid.sum() < 2 or not valid[0]:
            continue
        which = np.nonzero((~np.isnan(table)).any(axis=0))[0]
        if len(which) < LEAST_WORLDS:
            continue
        margins, bests, fires = [], [], []
        for _ in range(repeats):
            shuffled = drawer.permutation(which)
            half = len(shuffled) // 2
            deciding, scoring = shuffled[:half], shuffled[half:]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                scored = np.nanmean(table[:, scoring], axis=1)
                decided = np.nanmean(table[:, deciding], axis=1)
            if not np.isfinite(scored[0]):
                continue
            chose = picked_by_margin(table, valid, deciding, margin)
            if np.isfinite(scored[chose]):
                margins.append(scored[chose] - scored[0])
                fires.append(chose != 0)
            greedy = int(np.where(valid & np.isfinite(decided), decided, -np.inf).argmax())
            if np.isfinite(scored[greedy]):
                bests.append(scored[greedy] - scored[0])
        if margins:
            by_margin.append(float(np.mean(margins)))
            fired.append(float(np.mean(fires)))
        if bests:
            by_best.append(float(np.mean(bests)))
    deals = max(len(np.unique(game)), 1)

    def spoken(gains: list[float]) -> dict:
        gains = np.asarray(gains)
        if len(gains) < 2:
            return {"rows": int(len(gains))}
        error = float(gains.std(ddof=1) / len(gains) ** 0.5)
        return {
            "rows": int(len(gains)),
            "a_decision": round(float(gains.mean()), 5),
            "standard_error": round(error, 5),
            "standard_errors": round(float(gains.mean()) / error if error else 0.0, 2),
            "a_deal_so_far": round(float(gains.sum()) / deals, 4),
        }

    naive = float(
        np.nanmean(np.where(~np.isnan(values), values, -np.inf).max(axis=1) - values[:, 0])
    ) if rows else float("nan")
    return {
        "recording": str(folder),
        "search_backup_version": meta.get("search_backup_version", 1),
        "worlds": meta.get("worlds"),
        "sure": meta.get("sure"),
        "complete": meta.get("complete"),
        "deals": int(deals),
        "searched_decisions_a_deal": round(rows / deals, 1),
        # What the rule the engine plays gains, decided on half the worlds
        # and scored on the other.
        "by_margin": spoken(by_margin),
        "fired": round(float(np.mean(fired)), 4) if fired else None,
        # What taking the best average gains, likewise: the sibling head's
        # ceiling, and near zero wherever the worlds are few.
        "by_best": spoken(by_best),
        # And what the same difference reads as when the worlds that chose
        # also score, which is what a recording invites you to believe.
        "naive_gap": round(naive, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, nargs="+")
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    for folder in args.recording:
        print(json.dumps(measure(folder, margin=args.margin, repeats=args.repeats), indent=1))


if __name__ == "__main__":
    main()
