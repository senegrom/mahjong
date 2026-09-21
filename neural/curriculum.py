"""Public-state-targeted, paired counterfactual practice; NEVER PPO replay.

A bounded reservoir per requested category chooses roots without looking at
future payoffs. Exact simulator/follower snapshots avoid replaying the prefix
for every candidate. Both ordinary decisions and conditional riichi discards
are supported. Outcome labels are noisy action differences, not best actions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import riichi_py

from . import mortal_learner, rl_policy, zoo
from .observe import Views, Planes
from .outcomes import placements, require_finished, validate_budget
from .paired_actions import shortlist, load_data
from .league_state import snapshot
from .seed_ledger import SeedLedger
from .checkpoints import atomic_artifact
from .research_state import Snapshot, context, decision, public_flags, require_research_engine

CATEGORIES = ('uniform', 'riichi_pressure', 'close_endgame', 'call_pass',
              'riichi_choice', 'close_policy', 'riichi_discard')


@dataclass
class Root:
    snapshot: Snapshot
    planes: Planes
    legal: np.ndarray
    candidates: np.ndarray
    valid: np.ndarray
    game: int
    player: int
    step: int
    stage: int
    category: str


def policy_step(actor, views, *, forced=None, device='cpu'):
    rows, players, legal = decision(views)
    first = True
    def score(planes, mask):
        nonlocal first
        logits = rl_policy.logits(actor, planes, mask).float()
        if first and forced is not None:
            logits = logits.clone()
            for slot, game in enumerate(rows):
                action = forced.get(int(game))
                if action is not None:
                    if not 0 <= action < 46 or not mask[slot, action]:
                        raise ValueError('illegal root intervention')
                    logits[slot].fill_(-torch.inf); logits[slot, action] = 0.
        first = False
        return logits
    chosen, records = mortal_learner.decide_in_mortal_space(
        score, views, rows, players, legal[rows], greedy=True, device=device)
    actions = np.zeros(views.games, np.int64); actions[rows] = chosen
    views.arena.step(actions.tolist())
    return records


@torch.no_grad()
def continuation(actor, root: Root, action: int, *, device='cpu', max_steps=4000):
    """All branches restore the same hidden world and both RNG streams."""
    views = root.snapshot.restore(); arena = views.arena
    before = views.sparse(np.array([0]), np.array([root.player]))
    if any(not np.array_equal(getattr(before, k), getattr(root.planes, k)) for k in Planes.ARRAYS):
        raise RuntimeError('snapshot changed the public root observation')
    if not 0 <= action < 46 or not root.legal[action]: raise ValueError('illegal candidate')
    if root.stage == 1:
        # The snapshot follower has already seen this player's declaration.
        # The real engine has not: its composite action is riichi + this tile.
        if action >= 34: raise ValueError('riichi stage two must discard a tile')
        mask = np.frombuffer(arena.legal_mask(), np.uint8).reshape(1, 78)
        if not mask[0, 34 + action]: raise ValueError('conditional tile not legal in core')
        arena.step([34 + action])
    else:
        policy_step(actor, views, forced={0: action}, device=device)
    steps = 1
    while not arena.all_finished() and steps < max_steps:
        policy_step(actor, views, device=device); steps += 1
    require_finished(arena, steps=steps, context='snapshot continuation')
    finals = np.frombuffer(arena.final_scores(), np.int32).reshape(1, 4)
    return float(2.5 - placements(finals)[0, root.player]), steps


@torch.no_grad()
def select_roots(actor, *, games, seed, per_category=16, categories=CATEGORIES,
                 candidates=4, device='cpu', max_steps=4000):
    validate_budget(games, max_steps); require_research_engine()
    if (type(per_category) is not int or per_category < 1 or not categories
            or len(set(categories)) != len(categories) or any(c not in CATEGORIES for c in categories)):
        raise ValueError('invalid root reservoir')
    shortlist(np.zeros((1,46)), np.ones((1,46),bool), candidates)  # validate first
    arena = riichi_py.Arena(games=games, seed=seed); views = Views(arena, games, {'mortal'})
    actor.eval(); rng = np.random.default_rng(seed ^ 0xCA771)
    reservoirs = {c: [] for c in categories}; seen = {c: 0 for c in categories}
    steps = 0

    def consider(category, make_root):
        seen[category] += 1
        count = seen[category]; kept = reservoirs[category]
        pick = len(kept) if len(kept) < per_category else int(rng.integers(count))
        if pick < per_category:
            root = make_root(category)
            if pick < len(kept): kept[pick] = root
            else: kept.append(root)

    while not arena.all_finished() and steps < max_steps:
        rows, players, engine_mask = decision(views)
        if not len(rows): break
        planes = views.sparse(rows, players); allowed = zoo.translatable(engine_mask[rows])
        scores = rl_policy.logits(actor, planes.dense(device), torch.from_numpy(allowed).to(device)).float().cpu().numpy()
        moves, valid = shortlist(scores, allowed, candidates)
        flags = public_flags(context(arena)[rows], allowed, scores)
        for i, (game, player) in enumerate(zip(rows, players)):
            def make_root(category, i=i, game=int(game), player=int(player)):
                return Root(Snapshot.capture(views, [game]), planes.rows(np.array([i])), allowed[i].copy(),
                            moves[i].copy(), valid[i].copy(), seed+game, player, steps, 0, category)
            for category in categories:
                if category != 'riichi_discard' and flags[category][i]: consider(category, make_root)
            if 'riichi_discard' in categories and allowed[i, 37] and engine_mask[game, 34:68].sum() > 1:
                def conditional(category, i=i, game=int(game), player=int(player)):
                    after = Snapshot.capture(views, [game]).restore()
                    after.observer.follower.tell(0, player, json.dumps({'type':'reach','actor':player}))
                    p = after.sparse(np.array([0]), np.array([player]))
                    m = np.zeros((1,46), bool); m[0,:34] = engine_mask[game,34:68]
                    q = rl_policy.logits(actor, p.dense(device), torch.from_numpy(m).to(device)).float().cpu().numpy()
                    a, v = shortlist(q, m, candidates)
                    return Root(Snapshot.capture(after), p, m[0], a[0], v[0], seed+game,
                                player, steps, 1, category)
                consider('riichi_discard', conditional)
        # Ordinary continuation is independent of selected roots. No forced
        # actions or hypothetical reach notifications enter the original game.
        chosen, _ = mortal_learner.decide_in_mortal_space(
            lambda p,m: rl_policy.logits(actor,p,m), views, rows, players,
            engine_mask[rows], greedy=True, device=device)
        actions = np.zeros(games,np.int64); actions[rows] = chosen
        arena.step(actions.tolist()); steps += 1
    require_finished(arena, steps=steps, context='curriculum root selection')
    # A root qualifying for several categories is stored once, not treated as
    # independent evidence several times. Retain all its category annotations.
    unique = {}; tags = {}
    for category in categories:
        for root in reservoirs[category]:
            key = (root.game, root.step, root.player, root.stage)
            unique.setdefault(key, root); tags.setdefault(key, []).append(category)
    roots = [unique[k] for k in sorted(unique)]
    return roots, [tags[k] for k in sorted(unique)], seen


def collect(actor_path, out, *, ledger, games=64, per_category=16, categories=CATEGORIES,
            candidates=4, device='cpu', max_steps=4000):
    from .train_league import load_actor
    if Path(out).exists(): raise FileExistsError(out)
    require_research_engine(); validate_budget(games, max_steps)
    with tempfile.TemporaryDirectory(prefix='curriculum-') as folder:
        info = snapshot(Path(actor_path), Path(folder))
        actor, _, _ = load_actor(Path(folder)/info['file'], device)
        reservation = SeedLedger(Path(ledger)).reserve('counterfactual', games, 'targeted-roots')
        roots, tags, seen = select_roots(actor, games=games, seed=reservation['seed'], per_category=per_category,
            categories=categories, candidates=candidates, device=device, max_steps=max_steps)
        if not roots: raise ValueError('no roots matched; increase games or widen categories')
        payoffs = np.zeros((len(roots), candidates), np.float32); continuation_steps = 0
        for i, root in enumerate(roots):
            for j, action in enumerate(root.candidates):
                if root.valid[j]:
                    payoffs[i,j], steps = continuation(actor, root, int(action), device=device, max_steps=max_steps)
                    continuation_steps += steps
        planes = Planes.cat([r.planes for r in roots])
        payload = {'paired_version': 1, 'actor': info, 'shortlist': candidates, 'objective': 'placement-v2',
                   'root_step': None, 'stages': sorted({r.stage for r in roots}), 'root_stages': torch.tensor([r.stage for r in roots]),
                   'root_steps': torch.tensor([r.step for r in roots]), 'categories': tags, 'eligible': seen,
                   'reservation': reservation, 'collector': 'public-targeted-snapshot-v1',
                   'planes': {k:torch.from_numpy(getattr(planes,k).copy()) for k in Planes.ARRAYS},
                   'games': torch.tensor([r.game for r in roots],dtype=torch.int64),
                   'players': torch.tensor([r.player for r in roots],dtype=torch.int64),
                   'legal': torch.from_numpy(np.stack([r.legal for r in roots])),
                   'candidates': torch.from_numpy(np.stack([r.candidates for r in roots])),
                   'valid': torch.from_numpy(np.stack([r.valid for r in roots])),
                   'payoffs': torch.from_numpy(payoffs), 'label': 'paired noisy placement; never hindsight labels'}
        atomic_artifact(payload, Path(out), load_data)
        return {'roots':len(roots), 'eligible':seen, 'continuation_steps':continuation_steps,
                'seed_reservation':reservation, 'prefix_replays_per_candidate':0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('actor_path',type=Path); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True); p.add_argument('--games',type=int,default=64)
    p.add_argument('--per-category',type=int,default=16); p.add_argument('--candidates',type=int,default=4)
    p.add_argument('--categories',nargs='+',choices=CATEGORIES,default=list(CATEGORIES))
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=vars(p.parse_args()); torch.set_num_threads(2)
    print(json.dumps(collect(**args),allow_nan=False))


if __name__ == '__main__': main()
