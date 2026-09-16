"""Explicit teacher-only controls; safe to import without Torch/native engines."""
from __future__ import annotations

import math

from .inference import DEFAULT_ROLLOUT_BATCH, validate_batch


def validate_controls(*, objective="hybrid", valued_by="critic", played_by="network",
                      depth=0, search_calls=False, confirm_worlds=0,
                      extra_candidates=0, audit_share=0.0, rollout_batch=DEFAULT_ROLLOUT_BATCH):
    validate_batch(rollout_batch)
    if objective not in ("hybrid", "placement"):
        raise ValueError("teacher objective must be hybrid or placement")
    if valued_by not in ("critic", "public", "mean", "placement"):
        raise ValueError("unsupported teacher value head")
    if played_by not in ("network", "club"):
        raise ValueError("unsupported rollout player")
    if type(depth) is not int or depth < -1:
        raise ValueError("teacher depth must be -1 or nonnegative")
    if objective == "placement" and valued_by != "placement":
        raise ValueError("placement-only objective requires the placement value head")
    if valued_by == "placement" and played_by == "network" and depth != -1:
        raise ValueError("placement boundary leaves must play the root hand out: use network --depth -1")
    if type(search_calls) is not bool or search_calls and played_by != "network":
        raise ValueError("call search requires network rollouts")
    if type(confirm_worlds) is not int or confirm_worlds < 0 or confirm_worlds in (1, 2):
        raise ValueError("confirmation needs zero (disabled) or at least three fresh worlds")
    if type(extra_candidates) is not int or not 0 <= extra_candidates < 78:
        raise ValueError("extra_candidates must be an integer in [0, 77]")
    if type(audit_share) not in (int, float) or not math.isfinite(audit_share) or not 0 <= audit_share <= 1:
        raise ValueError("audit_share must be finite in [0, 1]")


def add_arguments(parser):
    parser.add_argument("--rollout-batch", type=int, default=DEFAULT_ROLLOUT_BATCH,
                        help="maximum policy inference rows, including conditional riichi; independent of leaf-batch")
    parser.add_argument("--objective", choices=("hybrid", "placement"), default="hybrid",
                        help="teacher utility; placement omits the separate root-hand points bonus")
    parser.add_argument("--search-calls", action="store_true", help="compare claims using normal simultaneous resolution")
    parser.add_argument("--confirm-worlds", type=int, default=0,
                        help="fresh fixed-size confirmation batch for each proposed override")
    parser.add_argument("--extra-candidates", type=int, default=0,
                        help="additional diverse/legal candidates beyond the policy prefix")
    parser.add_argument("--audit-share", type=float, default=0.0,
                        help="share of otherwise confident decisions to audit with search")


def candidate_set(order, legal, count, extra, rng):
    """Preserve the incumbent, prefer absent action families, then sample the tail.

    Candidates are proposals only; these probabilities are not behavior-policy
    likelihoods or a claim that the shortlist contains the globally best action.
    """
    if type(count) is not int or not 1 <= count <= 78 or type(extra) is not int or not 0 <= extra < 78:
        raise ValueError("invalid candidate budget")
    ranked = list(dict.fromkeys(int(a) for a in order if 0 <= int(a) < len(legal) and legal[int(a)]))
    if not ranked:
        raise ValueError("candidate ordering has no legal incumbent")
    chosen, rest = ranked[:count], ranked[count:]
    def family(action):
        return 0 if action < 34 else 1 if action < 68 else action
    for _ in range(extra):
        if not rest:
            break
        families = {family(a) for a in chosen}
        diverse = [a for a in rest if family(a) not in families]
        action = diverse[0] if diverse else rest[int(rng.integers(len(rest)))]
        chosen.append(action)
        rest.remove(action)
    return chosen
