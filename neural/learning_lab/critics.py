"""Separate public/asymmetric value estimators and public defensive predictors.

No actor parameters or features are trainable from these losses. Privileged
inputs are combined with the player's own history encoding, not substituted
for it. An oracle is a baseline only: it never chooses or labels policy moves.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .config import Config

UTILITY = (1.5, .5, -.5, -1.5)
DEFENCE_WIDTH = 104


def rank_targets(scores: np.ndarray, games: np.ndarray, players: np.ndarray) -> np.ndarray:
    from ..outcomes import _ties
    scores, games, players = np.asarray(scores), np.asarray(games), np.asarray(players)
    if (games.ndim != 1 or players.shape != games.shape or games.dtype != np.int64
            or players.dtype != np.int64 or np.any((games < 0) | (games >= len(scores)))
            or np.any((players < 0) | (players >= 4))):
        raise ValueError('invalid game/player labels')
    better, tied = _ties(scores)
    start, size = better[games, players], tied[games, players]
    rank = np.arange(4)[None, :]
    return (((rank >= start[:, None]) & (rank < (start + size)[:, None])) / size[:, None]).astype(np.float32)


def validate_defence(labels: np.ndarray):
    a = np.asarray(labels)
    if (a.ndim != 3 or a.shape[1:] != (3, DEFENCE_WIDTH) or a.dtype != np.float32
            or not np.isfinite(a).all() or np.any(a < 0)
            or not np.isin(a[:, :, :70], [0., 1.]).all()):
        raise ValueError('invalid defensive labels')
    valid, known, ron, payment = a[:, :, :1], a[:, :, 2:36], a[:, :, 36:70], a[:, :, 70:]
    if (np.any(a[:, :, 1:2] > valid) or np.any(known > valid) or np.any(ron > known)
            or np.any(payment[ron == 0] != 0)):
        raise ValueError('missing defensive labels cannot become safe/paid examples')


class SignalCritic(nn.Module):
    def __init__(self, cfg: Config, hidden_planes: int = 0, defence: bool = False):
        super().__init__()
        cfg.validate()
        self.cfg, self.hidden_planes, self.defence = cfg, hidden_planes, defence
        c = cfg.channels
        layers = [nn.Conv1d(1012 + hidden_planes, c, 1), nn.GroupNorm(8, c), nn.GELU()]
        from ..model import Residual
        layers.extend(Residual(c) for _ in range(cfg.blocks))
        self.encoder = nn.Sequential(*layers)
        self.rank = nn.Linear(2 * c, 4 if cfg.rank_head == 'categorical' else 1)
        self.hand = nn.Linear(2 * c, 1) if cfg.objective == 'hybrid' else None
        self.register_buffer('utility', torch.tensor(UTILITY))
        nn.init.zeros_(self.rank.weight); nn.init.zeros_(self.rank.bias)
        if self.hand is not None:
            nn.init.zeros_(self.hand.weight); nn.init.zeros_(self.hand.bias)
        if defence:
            self.ready = nn.Linear(2 * c, 3)
            self.ron = nn.Conv1d(c, 3, 1)
            self.payment = nn.Conv1d(c, 3, 1)

    def forward(self, public, hidden=None):
        if public.ndim != 3 or public.shape[1:] != (1012, 34):
            raise ValueError('critic requires player-history observations')
        if self.hidden_planes:
            if hidden is None or hidden.shape != (len(public), self.hidden_planes, 34):
                raise ValueError('oracle baseline needs its declared privileged planes')
            public = torch.cat([public.detach().float(), hidden.detach().float()], dim=1)
        elif hidden is not None:
            raise ValueError('public critic must not receive privileged inputs')
        features = self.encoder(public.detach().float())
        pooled = torch.cat([features.mean(2), features.amax(2)], dim=1)
        raw = self.rank(pooled)
        probs = raw.softmax(1) if self.cfg.rank_head == 'categorical' else None
        value = probs @ self.utility if probs is not None else raw[:, 0]
        hand = self.hand(pooled)[:, 0] if self.hand is not None else torch.zeros_like(value)
        return dict(rank=raw, probabilities=probs, placement=value, hand=hand, value=value + hand,
                    tenpai=self.ready(pooled) if self.defence else None,
                    ron=self.ron(features) if self.defence else None,
                    log_payment=F.softplus(self.payment(features)) if self.defence else None)

    def loss(self, output, ranks, hand_returns, defence=None):
        if (ranks.shape != (len(output['value']), 4) or hand_returns.shape != (len(ranks),)
                or not torch.isfinite(ranks).all() or not torch.isfinite(hand_returns).all()
                or (ranks < 0).any() or not torch.allclose(ranks.sum(1), torch.ones_like(hand_returns))):
            raise ValueError('invalid complete-game critic targets')
        rank_loss = (-(ranks.detach() * output['rank'].log_softmax(1)).sum(1).mean()
                     if self.cfg.rank_head == 'categorical'
                     else F.mse_loss(output['placement'], ranks.detach() @ self.utility))
        hand_loss = F.mse_loss(output['hand'], hand_returns.detach()) if self.hand is not None else rank_loss * 0
        stats = dict(rank_loss=rank_loss, hand_loss=hand_loss)
        total = rank_loss + hand_loss
        if self.defence:
            if defence is None or defence.shape != (len(ranks), 3, DEFENCE_WIDTH):
                raise ValueError('defensive head needs explicit labelled/missing targets')
            defence = defence.detach()
            flags = defence[:, :, :70]
            if (not torch.isfinite(defence).all() or (defence < 0).any()
                    or not ((flags == 0) | (flags == 1)).all()):
                raise ValueError('defensive targets must be finite binary flags and nonnegative payments')
            valid, tenpai = defence[:, :, 0], defence[:, :, 1]
            known, ron, paid = defence[:, :, 2:36], defence[:, :, 36:70], defence[:, :, 70:]
            if ((tenpai > valid).any() or (known > valid[:, :, None]).any() or (ron > known).any()
                    or (paid[ron == 0] != 0).any()):
                raise ValueError('missing/illegal defensive targets cannot train as known outcomes')
            ready_loss = (F.binary_cross_entropy_with_logits(output['tenpai'], tenpai, reduction='none') * valid).sum() / valid.sum().clamp_min(1)
            ron_loss = (F.binary_cross_entropy_with_logits(output['ron'], ron, reduction='none') * known).sum() / known.sum().clamp_min(1)
            paid_loss = ((output['log_payment'] - torch.log1p(paid / 4000)).square() * ron).sum() / ron.sum().clamp_min(1)
            stats.update(tenpai_loss=ready_loss, ron_loss=ron_loss, payment_loss=paid_loss)
            total = total + self.cfg.defence_weight * (ready_loss + ron_loss + paid_loss)
        if not torch.isfinite(total): raise FloatingPointError('nonfinite critic loss')
        return total, stats


class Critics(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        import riichi_py
        self.cfg = cfg
        self.public = SignalCritic(cfg, defence=cfg.defence_weight > 0)
        self.hidden_planes = {'none': 0, 'hands': riichi_py.HIDDEN_HANDS_PLANES, 'full': riichi_py.ORACLE_PLANES}[cfg.oracle]
        self.oracle = SignalCritic(cfg, self.hidden_planes) if self.hidden_planes else None

    def baseline(self, public, hidden=None):
        if self.oracle is None: return self.public(public)['value']
        if hidden is None: raise ValueError('missing oracle data')
        return self.oracle(public, hidden[:, :self.hidden_planes])['value']

    def losses(self, public, hidden, ranks, hand_returns, defence=None):
        loss, metrics = self.public.loss(self.public(public), ranks, hand_returns, defence)
        if self.oracle is not None:
            extra, _ = self.oracle.loss(self.oracle(public, hidden[:, :self.hidden_planes]), ranks, hand_returns)
            loss = loss + extra
            metrics['oracle_loss'] = extra
        return loss, metrics
