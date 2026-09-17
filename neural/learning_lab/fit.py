"""Held-out critic fitting and supervised policy distillation, never PPO replay."""
from __future__ import annotations
from pathlib import Path
import tempfile

import numpy as np
import torch

from ..checkpoints import atomic_save, copy_checkpoint
from ..observe import Planes
from . import artifacts
from .collection import Round
from .config import Config
from .critics import Critics, UTILITY
from .reanalysis import Lessons
from .train import _cpu, _policy, masked_kl


def split(games, every):
    games = np.asarray(games)
    if type(every) is not int or every < 0 or every == 1 or every >= 2**64:
        raise ValueError('validation_every must be zero (explicitly off) or >=2')
    if games.ndim != 1 or games.dtype != np.uint64: raise ValueError('stable uint64 game identities required')
    held = games % np.uint64(every) == 0 if every else np.zeros(len(games), bool)
    if not (~held).any() or (every and not held.any()): raise ValueError('need both whole-game splits, or explicitly disable validation')
    return np.flatnonzero(~held), np.flatnonzero(held)


@torch.no_grad()
def critic_report(critics, public, hidden, ranks, hands):
    output = critics.public(public)
    truth = ranks @ critics.public.utility
    if critics.cfg.objective == 'hybrid': truth = truth + hands
    result = dict(mse=(output['value'] - truth).square())
    if output['probabilities'] is not None:
        result.update(rank_brier=(output['probabilities'] - ranks).square().sum(1),
                      rank_cross_entropy=-(ranks * output['rank'].log_softmax(1)).sum(1))
    if critics.oracle is not None:
        result['oracle_mse'] = (critics.baseline(public, hidden) - truth).square()
    return result


def fit_critics(paths, out, cfg: Config, *, epochs=3, validation_every=5, device='cpu'):
    cfg.validate()
    if type(epochs) is not int or epochs < 1 or Path(out).exists(): raise ValueError('positive epochs and new output required')
    entries = []
    for path in paths:
        m, _, identity = artifacts.load(path, 'learning-round')
        entries.append((identity, Round.load(path)))
    entries.sort(key=lambda pair: pair[0])
    if not entries or len({i for i, _ in entries}) != len(entries): raise ValueError('empty/duplicate critic data')
    actors = {r.trace['actor_id'] for _, r in entries}
    if len(actors) != 1: raise ValueError('Prefit the oracle for one fixed policy, not a mix of policies')
    all_games, seen, planes, fields = [], set(), [], {k: [] for k in ('oracle', 'ranks', 'hand_return', 'defence')}
    for _, r in entries:
        seeds = r.arrays['game'].astype(np.uint64) + np.uint64(r.trace['seed'])
        identities = set(seeds.tolist())
        if seen & identities: raise ValueError('overlapping game seeds in critic training/holdout')
        seen |= identities; all_games.append(seeds); planes.append(r.observations)
        if r.trace['engine'] != artifacts.engine_identity(): raise ValueError('round engine contract changed')
        if cfg.oracle != 'none' and r.arrays['oracle'].shape[1] == 0: raise ValueError('round did not record oracle labels')
        if cfg.defence_weight and not r.trace['defence_labels']: raise ValueError('round did not record defensive labels')
        for k in fields: fields[k].append(r.arrays[k])
    p = Planes.cat(planes); data = {k: np.concatenate(v) for k, v in fields.items()}
    games = np.concatenate(all_games); training, validation = split(games, validation_every)
    torch.manual_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    critics = Critics(cfg).to(device)
    optimizer = torch.optim.AdamW(critics.parameters(), lr=cfg.critic_learning_rate, weight_decay=1e-4)
    for _ in range(epochs):
        critics.train(); order = rng.permutation(training)
        for start in range(0, len(order), cfg.batch):
            rows = order[start:start + cfg.batch]; get = lambda k: artifacts.tensor(data[k][rows], device)
            loss, _ = critics.losses(p.rows(rows).dense(device), get('oracle'), get('ranks'), get('hand_return'), get('defence'))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(critics.parameters(), cfg.max_grad_norm, error_if_nonfinite=True); optimizer.step()
    critics.eval(); reports = {}
    for name, rows in (('training', training), ('validation', validation)):
        sums = {}; total = 0
        for start in range(0, len(rows), cfg.batch):
            selected = rows[start:start + cfg.batch]; get = lambda k: artifacts.tensor(data[k][selected], device)
            metrics = critic_report(critics, p.rows(selected).dense(device), get('oracle'), get('ranks'), get('hand_return'))
            for key, values in metrics.items(): sums[key] = sums.get(key, 0.) + float(values.sum())
            total += len(selected)
        reports[name] = dict(rows=total, games=int(len(np.unique(games[rows]))),
                             metrics={k: v / total for k, v in sums.items()} if total else None)
    payload = dict(learning_critics=dict(version=1, config=cfg.describe(), policy_id=next(iter(actors)),
                        data=[i for i, _ in entries], state=critics.state_dict(), reports=reports,
                        epochs=epochs, validation_every=validation_every, engine=artifacts.engine_identity()))
    artifacts.finite(payload); atomic_save(_cpu(payload), Path(out), keep_previous=False)
    return reports


def distill(source, lesson_path, out, *, epochs=2, cfg=None, device='cpu'):
    cfg = cfg or Config()
    cfg.validate()
    if type(epochs) is not int or epochs < 1 or Path(out).exists(): raise ValueError('positive epochs and NEW output required')
    lessons = Lessons.load(lesson_path)
    with tempfile.TemporaryDirectory(prefix='mahjong-distill-lab-') as temporary:
        snapshot = Path(temporary) / 'actor.pt'; copy_checkpoint(source, snapshot)
        original = torch.load(snapshot, map_location='cpu', weights_only=True)
        torch.manual_seed(cfg.seed)
        net = artifacts.actor_from_payload(original, device).eval()
        if hasattr(net, 'set_mode'): net.set_mode(cfg.fixed)
        reference = artifacts.actor_from_payload(original, device).eval().requires_grad_(False) if cfg.anchor_weight else None
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=cfg.learning_rate, weight_decay=1e-4)
        rng = np.random.default_rng(cfg.seed); updates = 0
        for _ in range(epochs):
            order = rng.permutation(len(lessons))
            for start in range(0, len(order), cfg.batch):
                rows = order[start:start + cfg.batch]; planes, legal, target = lessons.batch(rows, device)
                logits = _policy(net, planes, legal, cfg)
                loss = -(target * logits.log_softmax(1).masked_fill(~legal, 0.)).sum(1).mean()
                if reference is not None:
                    with torch.no_grad(): old = _policy(reference, planes, legal, cfg).softmax(1)
                    loss = loss + cfg.anchor_weight * masked_kl(old, logits, legal)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite distillation loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm, error_if_nonfinite=True); optimizer.step(); updates += 1
        payload = dict(**artifacts.actor_payload(net), generation=artifacts.generation(original) + 1,
                       learner='learning_lab_distillation', learning_distillation=dict(version=1, config=cfg.describe(),
                       lessons=lessons.identity, teacher=lessons.meta['teacher_id'], epochs=epochs, updates=updates,
                       objective='supervised_counterfactual_policy_only', requires_critic_refit=True))
        artifacts.finite(payload); Path(out).mkdir(parents=True, exist_ok=False)
        atomic_save(_cpu(payload), Path(out) / 'candidate.pt', keep_previous=False)
    return Path(out) / 'candidate.pt'
