"""Hierarchical action probabilities and conservative counterfactual targets."""
from __future__ import annotations
import math
import numpy as np


def joint_policy(actor, after, legal):
    """46-action policy -> executable 78-action distribution, including riichi.

    Do not give every conditional riichi discard the declaration's full mass.
    Policy aliases share one engine action. Unrepresentable choices are refused.
    """
    from .. import zoo
    actor, legal = np.asarray(actor, np.float64), np.asarray(legal, bool)
    if actor.shape != (46,) or legal.shape != (78,) or not np.isfinite(actor).all() or np.any(actor < 0) or not np.isclose(actor.sum(), 1):
        raise ValueError('invalid actor/action distribution')
    joint = np.zeros(78, np.float64)
    for i, probability in enumerate(actor):
        if i == 37 or probability == 0: continue
        action = int(zoo.first_meaning(np.array([i]), legal[None])[0])
        if action < 0: raise ValueError('actor mass on an untranslatable move')
        joint[action] += probability
    if actor[37] > 0:
        if after is None: raise ValueError('riichi needs a conditional policy')
        after = np.asarray(after, np.float64)
        if after.shape != (46,) or not np.isfinite(after).all() or np.any(after < 0) or not np.isclose(after.sum(), 1):
            raise ValueError('invalid conditional distribution')
        for i, probability in enumerate(after):
            if probability == 0: continue
            tile = i if i < 34 else zoo.MORTAL_RED.get(i, -1)
            if tile < 0 or not legal[34 + tile]: raise ValueError('conditional probability on illegal riichi tile')
            joint[34 + tile] += actor[37] * probability
    if not np.isclose(joint.sum(), 1) or np.any(joint[~legal]): raise ValueError('joint distribution lost mass')
    return joint


def lift_policy(joint, actor, after, legal):
    """Back to declaration and conditional targets, preserving alias proportions."""
    from .. import zoo
    old = joint_policy(actor, after, legal)
    joint = np.asarray(joint, np.float64)
    if joint.shape != (78,) or not np.isfinite(joint).all() or np.any(joint < 0) or not np.isclose(joint.sum(), 1) or np.any((old == 0) & (joint > 0)):
        raise ValueError('target must preserve actor support')
    first = np.zeros(46, np.float64)
    for i, probability in enumerate(actor):
        if i == 37 or probability == 0: continue
        action = int(zoo.first_meaning(np.array([i]), np.asarray(legal)[None])[0])
        first[i] = probability * joint[action] / old[action]
    reach_mass = float(joint[34:68].sum()); first[37] = reach_mass
    second = None if after is None else np.asarray(after, np.float64).copy()
    if second is not None and reach_mass > 0:
        for i, probability in enumerate(after):
            tile = i if i < 34 else zoo.MORTAL_RED.get(i, -1)
            second[i] = probability * joint[34 + tile] * actor[37] / (old[34 + tile] * reach_mass) if probability > 0 and tile >= 0 else 0.
    if not np.isclose(first.sum(), 1) or (second is not None and not np.isclose(second.sum(), 1)):
        raise ValueError('hierarchical target does not normalize')
    return first.astype(np.float32), None if second is None else second.astype(np.float32)


def confirmed_tilt(actor_joint, incumbent, challenger, lower_edge, eta=1.):
    """Redistribute ONLY the tested pair's mass, leaving all other moves intact.

    A positive confirmation lower margin gives a bounded exponential tilt. This
    margin is diagnostic, not a formal guarantee or a probability of winning.
    """
    p = np.asarray(actor_joint, np.float64)
    if p.shape != (78,) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError('invalid joint policy')
    if any(type(i) is not int or not 0 <= i < 78 for i in (incumbent, challenger)) or not np.isfinite(lower_edge) or not np.isfinite(eta) or eta < 0:
        raise ValueError('invalid confirmed comparison')
    q = p.copy()
    if incumbent == challenger or lower_edge <= 0: return q
    pair = np.array([incumbent, challenger]); mass = float(p[pair].sum())
    weights = p[pair] * np.exp([0., min(eta * lower_edge, 5.)])
    if weights.sum(): q[pair] = mass * weights / weights.sum()
    return q


def candidates(order, legal, probabilities, count, rng, method='top'):
    from ..teacher_actions import representable_moves
    allowed = representable_moves(np.asarray(legal, bool)[None])[0]
    ordered = list(dict.fromkeys(int(i) for i in order if 0 <= i < 78 and allowed[i]))
    if type(count) is not int or count < 2 or method not in ('top', 'gumbel') or not ordered:
        raise ValueError('invalid candidate request')
    if method == 'top': return ordered[:count]
    incumbent, rest = ordered[0], np.array(ordered[1:], np.int64)
    p = np.asarray(probabilities, np.float64)
    if p.shape != (78,) or not np.isfinite(p).all() or np.any(p < 0): raise ValueError('invalid candidate prior')
    score = np.log(np.maximum(p[rest], 1e-30)) + rng.gumbel(size=len(rest))
    return [incumbent] + rest[np.argsort(-score, kind='stable')[:count - 1]].tolist()


def evidence(judgements, weights):
    """One paired advantage per tested move, over independent native proposals."""
    from ..worth import edge_and_error
    if not judgements: return {}
    base = np.asarray(judgements[0][2], np.float64)
    weight = np.asarray(weights, np.float64)
    result = {}
    for action, _, values in judgements:
        values = np.asarray(values, np.float64)
        if values.shape != base.shape: raise ValueError('unpaired candidate evidence')
        edge, error = edge_and_error(values, base, weight)
        live = np.isfinite(values) & np.isfinite(base) & (weight > 0)
        w = weight[live]
        result[int(action)] = dict(edge=edge, error=error, worlds=int(live.sum()),
                                   effective_worlds=float(w.sum() ** 2 / (w @ w)) if len(w) else 0.)
    return result


def teach_root(evaluate, actions, *, worlds=8, confirm_worlds=16, audit_worlds=8, margin=2., race_budget=0):
    """Discovery -> one independent confirmation -> independent ranking labels.

    evaluate(actions, worlds) must create a NEW native world batch on every call.
    A sequential-halving budget counts candidate-world rollout slots, excluding
    confirmation/audit budgets. Incumbent is never eliminated; audit evidence is
    never fed back into the actor's acceptance decision.
    """
    if len(actions) < 2 or len(set(actions)) != len(actions): raise ValueError('need distinct candidate actions')
    for v in (worlds, confirm_worlds, audit_worlds):
        if type(v) is not int or v < 3: raise ValueError('each stage needs at least three independent proposals')
    if type(race_budget) is not int or race_budget < 0 or not np.isfinite(margin) or margin < 0:
        raise ValueError('invalid search budget/margin')
    incumbent = actions[0]; active = list(actions); allocated = 0
    if race_budget:
        # Allocate the remaining fixed budget across each remaining stage. No
        # stopping-time claim is attached; only the fresh confirmation can act.
        minimum, n = 0, len(active) - 1
        while True:
            minimum += 3 * (n + 1)
            if n == 1: break
            n = math.ceil(n / 2)
        if race_budget < minimum: raise ValueError('race budget too small for all stages')
        remaining = race_budget
        while True:
            stages = math.ceil(math.log2(len(active) - 1)) + 1
            per = max(3, remaining // (stages * len(active)))
            # Reserve the minimum for all future stages before allocating now.
            future = 0; n = math.ceil((len(active) - 1) / 2)
            if len(active) > 2:
                while True:
                    future += 3 * (n + 1)
                    if n == 1: break
                    n = math.ceil(n / 2)
            per = min(per, (remaining - future) // len(active))
            d = evaluate(active, per); allocated += per * len(active); remaining -= per * len(active)
            ranked = sorted(active[1:], key=lambda a: -d.get(a, {}).get('edge', -math.inf))
            if len(active) == 2: break
            active = [incumbent] + ranked[:math.ceil(len(ranked) / 2)]
        challenger = active[1]
    else:
        d = evaluate(active, worlds); allocated = len(active) * worlds
        challenger = max(active[1:], key=lambda a: d.get(a, {}).get('edge', -math.inf))
    lower = 0.; confirmed = False; confirmation = {}
    if d.get(challenger, {}).get('edge', 0.) > 0:
        confirmation = evaluate([incumbent, challenger], confirm_worlds)
        row = confirmation.get(challenger, {})
        edge, error = row.get('edge', 0.), row.get('error', math.inf)
        if np.isfinite(error) and row.get('worlds', 0) >= 3:
            lower = max(0., edge - margin * error); confirmed = lower > 0
    audit = evaluate(list(actions), audit_worlds)
    return dict(incumbent=incumbent, challenger=challenger, confirmed=confirmed, lower_edge=lower,
                audit=audit, confirmation=confirmation, discovery_slots=allocated)
