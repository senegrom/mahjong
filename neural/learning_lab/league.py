"""Development-only adaptive league, with explicit diversity floors and identity.

The matchup matrix is not promotion evidence. No member or champion is replaced.
An exploiter run uses the same learner against three copies of a frozen champion.
"""
from __future__ import annotations
from copy import deepcopy
import math

import numpy as np


class League:
    def __init__(self, members: list[dict], uniform: float = .3, adaptive: bool = True, state=None):
        if (type(uniform) not in (float, int) or not math.isfinite(uniform) or not 0 < uniform <= 1
                or type(adaptive) is not bool):
            raise ValueError('invalid league controls')
        self.members = deepcopy(members)
        self.uniform, self.adaptive = float(uniform), adaptive
        ids = []
        for m in members:
            if (set(m) != {'sha256', 'name', 'role'} or m['role'] not in ('reference', 'champion', 'recent', 'older', 'exploiter')
                    or not isinstance(m['name'], str) or not m['name']
                    or not isinstance(m['sha256'], str) or len(m['sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in m['sha256'])):
                raise ValueError('invalid frozen league member')
            ids.append(m['sha256'])
        if len(set(ids)) != len(ids): raise ValueError('duplicate league checkpoint identity')
        self.ema = {key: 2.5 for key in ids}
        self.matrix = {}
        if state is not None:
            if state.get('members') != self.members or state.get('uniform') != self.uniform or state.get('adaptive') != adaptive:
                raise ValueError('resumed league changed')
            self.ema = deepcopy(state['ema']); self.matrix = deepcopy(state['matrix'])
            if set(self.ema) != set(ids) or any(not 1 <= v <= 4 for v in self.ema.values()):
                raise ValueError('invalid resumed league estimates')
            for row in self.matrix.values():
                for key, stats in row.items():
                    if key not in ids or type(stats['games']) is not int or stats['games'] < 1 or not stats['games'] <= stats['sum_placement'] <= 4 * stats['games']:
                        raise ValueError('invalid matchup history')

    def weights(self) -> np.ndarray:
        if not self.members: return np.empty(0, np.float64)
        prior = {'reference': 2., 'champion': 3., 'recent': 1., 'older': 1., 'exploiter': 1.}
        raw = np.asarray([prior[m['role']] for m in self.members], np.float64)
        if self.adaptive:
            # Greater learner placement means a harder opponent. The uniform
            # component guarantees each checkpoint at least uniform/N exposure.
            raw *= np.asarray([max(.05, (self.ema[m['sha256']] - 1.) / 3.) ** 2 for m in self.members])
        return self.uniform / len(raw) + (1 - self.uniform) * raw / raw.sum()

    def seat(self, games, share, rng):
        if type(games) is not int or games < 1 or not 0 <= share <= 1: raise ValueError('invalid seating request')
        seated = np.full(games, -1, np.int64)
        if self.members:
            takes = rng.random(games) < share
            seated[takes] = rng.choice(len(self.members), int(takes.sum()), p=self.weights())
        return seated

    def update(self, actor_id: str, seated: np.ndarray, learner_placements: np.ndarray):
        seated, places = np.asarray(seated), np.asarray(learner_placements)
        if (not isinstance(actor_id, str) or len(actor_id) != 64 or seated.ndim != 1
                or places.shape != seated.shape or seated.dtype != np.int64
                or np.any((seated < -1) | (seated >= len(self.members)))
                or not np.isfinite(places).all() or np.any((places < 1) | (places > 4))):
            raise ValueError('invalid development matchups')
        row = self.matrix.setdefault(actor_id, {})
        for i, member in enumerate(self.members):
            observed = places[seated == i]
            if not len(observed): continue
            key = member['sha256']
            old = row.setdefault(key, dict(games=0, sum_placement=0.))
            old['games'] += len(observed); old['sum_placement'] += float(observed.sum())
            a = 1 - math.exp(-len(observed) / 32.)
            self.ema[key] = (1 - a) * self.ema[key] + a * float(observed.mean())

    def state(self):
        return deepcopy(dict(members=self.members, uniform=self.uniform, adaptive=self.adaptive, ema=self.ema, matrix=self.matrix))


def select_roots(roots, maximum: int, uniform_share: float, rng):
    """A uniform quota plus priority sampling without replacement.

    Deliberately not an IID/on-policy dataset; no fake inverse-probability
    correction is attached. At least one uniform root survives a positive cap.
    """
    if type(maximum) is not int or maximum < 0 or not 0 < uniform_share <= 1:
        raise ValueError('invalid root selector')
    if not roots or maximum == 0: return []
    n = min(maximum, len(roots)); uniform_n = min(n, max(1, math.ceil(n * uniform_share)))
    first = rng.choice(len(roots), uniform_n, replace=False).tolist()
    rest = np.asarray([i for i in range(len(roots)) if i not in set(first)], np.int64)
    if n > uniform_n:
        weights = np.asarray([roots[int(i)]['priority'] for i in rest], np.float64)
        if not np.isfinite(weights).all() or np.any(weights < 0): raise ValueError('invalid public root priorities')
        weights = weights + .01
        first += rng.choice(rest, n - uniform_n, replace=False, p=weights / weights.sum()).tolist()
    return sorted((deepcopy(roots[i]) for i in first), key=lambda r: (r['step'], r['game']))
