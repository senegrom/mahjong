"""Paired-advantage rankers with fused frozen features or a dedicated tower."""
from __future__ import annotations
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .. import contract
from ..checkpoints import atomic_save, copy_checkpoint
from . import artifacts
from .fit import split
from .reanalysis import Lessons
from .train import _cpu


class Ranker(nn.Module):
    def __init__(self, net, mode='fused', channels=32):
        super().__init__()
        if mode not in ('ours', 'fused', 'own-tower') or type(channels) is not int or channels < 8 or channels % 8:
            raise ValueError('invalid ranker architecture')
        self.mode, self.width = mode, channels
        ours = getattr(net, 'ours', net)
        self.features_id = artifacts.digest_state(artifacts.actor_payload(net))
        self.channels = channels if mode == 'own-tower' else ours.channels
        self.mortal_dim = int(net.fuse.phi_proj[0].in_features) if mode == 'fused' and hasattr(net, 'mortal') else 0
        if mode == 'own-tower':
            from ..model import Residual
            self.encoder = nn.Sequential(nn.Conv1d(1012, channels, 1), nn.GroupNorm(8, channels), nn.GELU(), Residual(channels))
        self.tiles = nn.Conv1d(self.channels, 2, 1)
        self.global_scores = nn.Linear(self.channels * 2 + self.mortal_dim, 78)
        for layer in (self.tiles, self.global_scores):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, net, planes):
        if self.mode == 'own-tower':
            features = self.encoder(planes.detach())
            context = torch.cat([features.mean(2), features.amax(2)], 1)
        else:
            with torch.no_grad():
                ours = getattr(net, 'ours', net)
                features = ours.tail(ours.tower(ours.stem(planes))).detach()
                parts = [features.mean(2), features.amax(2)]
                if self.mortal_dim: parts.append(net.mortal.features(planes).detach())
                context = torch.cat(parts, 1)
        tile = self.tiles(features).reshape(len(planes), 68)
        return self.global_scores(context) + F.pad(tile, (0, 10))

    def metadata(self):
        return dict(version=1, features_id=self.features_id, mode=self.mode, channels=self.width)


def fit(source, lesson_path, out, *, mode='fused', channels=32, epochs=3, batch=128,
        learning_rate=1e-4, validation_every=5, seed=27, device='cpu'):
    if (Path(out).exists() or any(type(n) is not int or n < 1 for n in (epochs, batch))
            or not np.isfinite(learning_rate) or learning_rate <= 0): raise ValueError('invalid ranker fit or existing output')
    data = Lessons.load(lesson_path); a = data.arrays
    eligible = (a['tested'].sum(1) >= 2)
    if not eligible.any(): raise ValueError('No independently audited candidate comparisons')
    ids = np.flatnonzero(eligible); training, held = split(a['games'][ids], validation_every)
    training, held = ids[training], ids[held]
    with tempfile.TemporaryDirectory(prefix='mahjong-lab-ranker-') as temporary:
        snapshot = Path(temporary) / 'actor.pt'; copy_checkpoint(source, snapshot)
        torch.manual_seed(seed)
        net = artifacts.actor_from_payload(torch.load(snapshot, map_location='cpu', weights_only=True), device).eval().requires_grad_(False)
        head = Ranker(net, mode, channels).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
        rng = np.random.default_rng(seed); observations = data.roots()
        bases = np.array([d['incumbent'] for d in data.meta['diagnostics']], np.int64)
        for _ in range(epochs):
            order = rng.permutation(training)
            for start in range(0, len(order), batch):
                rows = order[start:start + batch]
                scores = head(net, observations.rows(rows).dense(device))
                base = artifacts.tensor(bases[rows], device)
                delta = scores - scores[torch.arange(len(rows), device=device), base, None]
                target = artifacts.tensor(a['advantage'][rows], device)
                tested = artifacts.tensor(a['tested'][rows], device)
                weight = tested / (1 + artifacts.tensor(a['stderr'][rows], device).square())
                loss = (F.smooth_l1_loss(delta, target, reduction='none') * weight).sum() / weight.sum().clamp_min(1)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite paired-ranker loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        from ..teacher_statistics import clustered_mean
        gains = []
        with torch.no_grad():
            head.eval()
            for start in range(0, len(held), batch):
                rows = held[start:start + batch]
                scores = head(net, observations.rows(rows).dense(device)).cpu().numpy()
                best = np.where(a['tested'][rows], scores, -np.inf).argmax(1)
                base = bases[rows]
                improve = scores[np.arange(len(rows)), best] > scores[np.arange(len(rows)), base]
                choice = np.where(improve, best, base)
                gains.extend(a['advantage'][rows, choice].tolist())
        report = clustered_mean(np.asarray(gains, np.float64), a['games'][held])
        report['meaning'] = 'held-out game audit labels with positive-gap overrides; not whole-game strength or promotion evidence'
        payload = dict(learning_ranker={**head.metadata(), 'state': head.state_dict(), 'lessons': data.identity,
                        'epochs': epochs, 'batch': batch, 'learning_rate': learning_rate, 'seed': seed,
                        'validation_every': validation_every, 'report': report})
        artifacts.finite(payload); atomic_save(_cpu(payload), Path(out), keep_previous=False)
    return report


def load(path, net, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True).get('learning_ranker')
    if not isinstance(saved, dict) or saved.get('version') != 1: raise ValueError('unsupported experimental ranker')
    head = Ranker(net, saved['mode'], saved['channels']).to(device)
    if saved['features_id'] != head.features_id: raise ValueError('ranker supporting policy changed; refit')
    head.load_state_dict(saved['state'], strict=True)
    artifacts.finite(head.state_dict()); return head.eval()


class Player:
    """Evaluate a trained ranker at the real table, with the policy as fallback."""
    kind = 'mortal'
    def __init__(self, net, head, *, candidates=4, margin=.05, device='cpu'):
        if (head.features_id != artifacts.digest_state(artifacts.actor_payload(net))
                or type(candidates) is not int or candidates < 2 or not np.isfinite(margin) or margin < 0):
            raise ValueError('invalid or mismatched ranker player')
        self.net, self.head, self.device, self.k, self.margin = net, head, device, candidates, margin
        self.served = contract.serve(net)
    def eval(self): self.net.eval(); self.head.eval(); return self
    def parameters(self): return self.net.parameters()
    @torch.no_grad()
    def choose(self, views, rows, players, legal):
        rows = np.asarray(rows, np.int64)
        if not len(rows): return np.empty(0, np.int64)
        mask = np.zeros((int(rows.max()) + 1, 78), bool); mask[rows] = legal
        order, _, _, _ = contract.root_order(self.served, None, views, rows, players, mask, self.device)
        planes, _ = self.served.root(None, views, rows, players, mask, self.device)
        scores = self.head(self.net, planes).cpu().numpy()
        choices = []
        from ..teacher_actions import representable_moves
        supported = representable_moves(np.asarray(legal, bool))
        for i, game in enumerate(rows):
            actions = [int(a) for a in order[game] if supported[i, a]][:self.k]
            if not actions: raise ValueError('no representable ranker move')
            best = max(actions, key=lambda a: scores[i, a])
            choices.append(best if scores[i, best] - scores[i, actions[0]] > self.margin else actions[0])
        return np.asarray(choices, np.int64)
