"""Choosing which imagined worlds to spend the search's effort on.

A search cannot afford to play out every world it can imagine, so it
proposes many and keeps a few. How those few are chosen decides what the
search is an estimate *of*.

What was done before: score every proposal with the reader of hidden hands,
keep the `keep` highest-scoring, and give them weights proportional to
those scores. That is not an estimator of anything in particular. Taking
the top few by weight is a deterministic choice that throws away all the
probability mass below the cut, and renormalising what survives does not
put it back; the result is systematically pulled towards whatever the
reader already found likely. In mahjong that is exactly the wrong bias to
have. The hand that decides whether a discard was a mistake is usually the
unlikely one — the opponent who was waiting on the tile nobody expected —
and a rule that keeps only the likely worlds is a rule that never sees it.

What is done instead is sampling importance resampling, which is the
standard answer to this. The proposals are drawn from the belief; the
reader says how much more likely each one's hidden hands are than the
proposal made them; those ratios are the importance weights; and `keep`
survivors are drawn *in proportion to* them. A world a hundred times less
likely than another is a hundred times less likely to survive — not
impossible, which is the whole difference. Survivors then count equally,
because the resampling has already spent the weights; weighting them a
second time would apply the same correction twice.

Systematic resampling is used rather than independent draws: one uniform
offset and evenly spaced cuts through the cumulative weights. It has the
same expectation and lower variance, and it guarantees that any world
holding more than `1 / keep` of the mass survives at least once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Chosen:
    """Which worlds survived, what they count for, and how healthy that is."""

    #: Indices into the proposals, with repeats where a heavy world was
    #: drawn more than once.
    kept: list[int]
    #: What each survivor counts for. Uniform, because resampling has
    #: already spent the weights.
    weights: list[float]
    #: Kish's effective sample size of the weights *before* resampling, as
    #: a share of the proposals. Near one means the reader found the
    #: proposals about equally likely and the resampling did little; near
    #: zero means a handful of worlds held all the mass and the search is
    #: really running on that handful, whatever `keep` says.
    efficiency: float
    #: How many distinct proposals survived. Far below `keep` means the
    #: same few worlds are being played out again and again.
    distinct: int


def resample(scores: np.ndarray, keep: int, rng: np.random.Generator) -> Chosen:
    """`keep` worlds drawn in proportion to `scores`, read as log weights.

    `scores` are the reader's log likelihood ratios, one per proposal.
    Uniform scores give a uniform draw, which is the unweighted estimator:
    a reader that has learned nothing cannot bias the search.

    Degenerate input — no proposals, or weights that are not finite, or a
    total of zero — falls back to an even draw rather than raising, because
    a search that cannot weigh its worlds should still play, and says so
    through an efficiency of zero.
    """
    scores = np.asarray(scores, dtype=np.float64)
    count = scores.size
    if count == 0 or keep <= 0:
        return Chosen(kept=[], weights=[], efficiency=0.0, distinct=0)

    finite = np.isfinite(scores)
    if not finite.all():
        # An infinity or a nan in the reader's output would poison every
        # weight through the exponential. Drop those proposals rather than
        # the whole search, and let the efficiency report the damage.
        scores = np.where(finite, scores, -np.inf)

    shifted = scores - scores.max(initial=0.0)
    weight = np.exp(np.where(np.isfinite(shifted), shifted, -np.inf))
    total = weight.sum()
    if not np.isfinite(total) or total <= 0.0:
        even = rng.choice(count, size=min(keep, count), replace=count < keep)
        kept = sorted(int(index) for index in np.atleast_1d(even))
        return Chosen(
            kept=kept,
            weights=[1.0 / len(kept)] * len(kept),
            efficiency=0.0,
            distinct=len(set(kept)),
        )

    weight = weight / total
    # Kish: one over the sum of the squares, as a share of the proposals.
    efficiency = float(1.0 / (count * np.square(weight).sum()))

    # Systematic resampling: one offset, evenly spaced cuts.
    cuts = (rng.random() + np.arange(keep)) / keep
    kept = np.searchsorted(np.cumsum(weight), cuts, side="right")
    kept = np.clip(kept, 0, count - 1)
    picked = [int(index) for index in kept]
    return Chosen(
        kept=picked,
        weights=[1.0 / keep] * keep,
        efficiency=efficiency,
        distinct=len(set(picked)),
    )
