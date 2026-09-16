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
difference is a held-out estimate of what the rule gains a decision, in
the units the rollouts spoke in: hand points over four thousand plus the
placement the hand led to. This describes a half-budget diagnostic, not a guaranteed lower bound
on the gain at a larger search budget.

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
import math

from .teacher_statistics import validation_summary

import numpy as np

#: What the engine's search asks of a candidate before it will override
#: the policy: this many standard errors of the paired difference.
MARGIN = 2.0

#: How many times the worlds are split. The split is noise of its own,
#: and the question is about the decision.
REPEATS = 16

#: Fewer worlds than this and neither half can say anything.
LEAST_WORLDS = 6


def edge_and_error(mine: np.ndarray, theirs: np.ndarray, weights=None) -> tuple[float, float]:
    """Native weighted comparison over INDEPENDENT proposals, not repeats."""
    paired = np.asarray(mine, dtype=np.float64) - np.asarray(theirs, dtype=np.float64)
    weight = np.ones_like(paired) if weights is None else np.asarray(weights, dtype=np.float64)
    if weight.shape != paired.shape or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("invalid world weights")
    live = np.isfinite(paired) & (weight > 0)
    paired, weight = paired[live], weight[live]
    if len(paired) < 3:
        return 0.0, float("inf")
    weight = weight / weight.sum()
    mean = float((paired * weight).sum())
    concentration = float(np.square(weight).sum())
    if concentration >= 1:
        return 0.0, float("inf")
    spread = float((np.square(weight) * np.square(paired - mean)).sum())
    return mean, float(np.sqrt(spread / (1 - concentration)))


def weighted_means(table, weights):
    mass = np.where(np.isfinite(table), weights, 0.0)
    totals = mass.sum(axis=1)
    return np.divide((np.nan_to_num(table) * mass).sum(axis=1), totals,
                     out=np.full(len(table), np.nan), where=totals > 0)


def picked_by_margin(table: np.ndarray, valid: np.ndarray, worlds: np.ndarray,
                     margin: float = MARGIN, weights=None) -> int:
    """Which candidate the engine's rule takes, judging on the worlds
    named: the one with the largest edge over the policy's move that
    clears the margin, or the policy's move."""
    chose, best = 0, 0.0
    for candidate in range(1, len(valid)):
        if not valid[candidate]:
            continue
        edge, error = edge_and_error(table[candidate, worlds], table[0, worlds],
                                     None if weights is None else weights[worlds])
        if np.isfinite(error) and edge > margin * error and edge > best:
            chose, best = candidate, edge
    return chose


def measure(folder: Path, margin: float = MARGIN, repeats: int = REPEATS,
            seed: int = 11) -> dict:
    """What the margin's rule, and taking the best average, gain a
    decision on the recording named."""
    from .recordings import resolve_recording

    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    folder = resolve_recording(folder)
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    per_world = np.load(folder / "per_world.npy").astype(np.float64)
    values = np.load(folder / "values.npy").astype(np.float64)
    game = np.load(folder / "game.npy")
    drawer = np.random.default_rng(seed)
    rows, width, _count = per_world.shape
    weight_file = folder / "world_weights.npy"
    weights = np.load(weight_file) if weight_file.exists() else np.ones((rows, _count))
    if weights.shape != (rows, _count) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("invalid recorded world weights")
    by_margin, by_best, fired = [], [], []
    margin_games, best_games = [], []
    if game.shape != (rows,) or game.dtype.kind not in "iu" or np.any(game < 0):
        raise ValueError("one nonnegative environment game identity per recorded decision is required")
    for row in range(rows):
        table = per_world[row]
        valid = ~np.isnan(values[row])
        if valid.sum() < 2 or not valid[0]:
            continue
        which = np.nonzero((~np.isnan(table)).any(axis=0) & (weights[row] > 0))[0]
        if len(which) < LEAST_WORLDS:
            continue
        margins, bests, fires = [], [], []
        for _ in range(repeats):
            shuffled = drawer.permutation(which)
            half = len(shuffled) // 2
            deciding, scoring = shuffled[:half], shuffled[half:]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                scored = weighted_means(table[:, scoring], weights[row, scoring])
                decided = weighted_means(table[:, deciding], weights[row, deciding])
            if not np.isfinite(scored[0]):
                continue
            chose = picked_by_margin(table, valid, deciding, margin, weights=weights[row])
            if np.isfinite(scored[chose]):
                margins.append(scored[chose] - scored[0])
                fires.append(chose != 0)
            greedy = int(np.where(valid & np.isfinite(decided), decided, -np.inf).argmax())
            if np.isfinite(scored[greedy]):
                bests.append(scored[greedy] - scored[0])
        if margins:
            by_margin.append(float(np.mean(margins)))
            margin_games.append(game[row])
            fired.append(float(np.mean(fires)))
        if bests:
            by_best.append(float(np.mean(bests)))
            best_games.append(game[row])
    deals = max(len(np.unique(game)), 1)

    def spoken(gains, identities):
        return validation_summary(gains, np.asarray(identities, dtype=game.dtype),
                                  total_deals=deals)

    naive = float(
        np.nanmean(np.where(~np.isnan(values), values, -np.inf).max(axis=1) - values[:, 0])
    ) if rows else None
    return {
        "recording": str(folder),
        "teacher_objective": meta.get("teacher_objective", "hybrid"),
        "confirmation_worlds": meta.get("confirm_worlds", 0),
        "diagnostic_rule": "margin on a split of the recorded worlds; not a full two-stage replay",
        "search_backup_version": meta.get("search_backup_version", 1),
        "independent_worlds": weight_file.exists(),
        "worlds": meta.get("worlds"),
        "sure": meta.get("sure"),
        "complete": meta.get("complete"),
        "deals": int(deals),
        "searched_decisions_a_deal": round(rows / deals, 1),
        # What the rule the engine plays gains, decided on half the worlds
        # and scored on the other.
        "by_margin": spoken(by_margin, margin_games),
        "fired": round(float(np.mean(fired)), 4) if fired else None,
        # What taking the best average gains, likewise: the sibling head's
        # ceiling, and near zero wherever the worlds are few.
        "by_best": spoken(by_best, best_games),
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
