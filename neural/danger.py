"""Public legal-ron probability and conditional-payment prediction.

collect labels cloned real hands; fit reads ONLY public observations. attach
freezes the fitted predictor for an entire PPO run and adds a zero-initialized
residual policy. Neither actual hidden tiles nor labels can enter that actor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn

from .observe import Planes, Views, PLANES
from .rl_critics import Residual
from .checkpoints import atomic_artifact, atomic_save
from .evidence import split_games
from .league_state import snapshot, digest
from .seed_ledger import SeedLedger
from . import mortal_learner, rl_policy
from .research_state import require_research_engine, decision

VERSION = 1
PAYMENT_SCALE = 1000.0


class Predictor(nn.Module):
    def __init__(self, width=64, blocks=2):
        super().__init__()
        if type(width) is not int or width < 8 or width % 8 or type(blocks) is not int or blocks < 1:
            raise ValueError("invalid danger predictor dimensions")
        self.width, self.blocks = width, blocks
        self.stem = nn.Conv1d(PLANES, width, 1)
        self.tower = nn.Sequential(*(Residual(width) for _ in range(blocks)))
        self.head = nn.Conv1d(width, 6, 1)

    def forward(self, observation):
        if observation.ndim != 3 or observation.shape[1:] != (PLANES, 34):
            raise ValueError("danger predictor requires public Mortal observation planes")
        raw = self.head(self.tower(self.stem(observation.float())))
        return raw[:, :3], nn.functional.softplus(raw[:, 3:])

    def features(self, observation):
        logits, payment = self(observation)
        probability = logits.sigmoid()
        # Conditional payment and probability stay distinct; actor can learn
        # an expected-loss feature without making danger a hard prohibition.
        return torch.cat([probability, probability * payment], dim=1)


def loss_of(logits, payment, labels, mask):
    if (logits.shape != payment.shape or logits.shape != labels.shape[:-1]
            or labels.shape[-1] != 2 or mask.shape != (len(labels), 34)):
        raise ValueError("danger target shapes disagree")
    active = mask[:, None, :].expand_as(logits)
    hit = labels[..., 0]
    probability_loss = nn.functional.binary_cross_entropy_with_logits(logits, hit, reduction="none")
    probability_loss = probability_loss[active].mean() if active.any() else logits.sum() * 0
    won = active & (hit > .5)
    payment_loss = ((payment - labels[..., 1] / PAYMENT_SCALE).square()[won].mean()
                    if won.any() else payment.sum() * 0)
    return probability_loss + payment_loss


def load_data(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if type(data.get("danger_version")) is not int or data["danger_version"] != VERSION:
        raise ValueError("unsupported danger data")
    planes = Planes(**{k: v.numpy() for k, v in data["planes"].items()})
    n = len(planes); planes.validate()
    y, m = data["labels"], data["mask"]
    if (n < 1 or y.dtype != torch.float32 or y.shape != (n, 3, 34, 2) or m.shape != (n, 34) or m.dtype != torch.bool
            or data["games"].shape != (n,) or data["games"].dtype != torch.int64
            or (data["games"] < 0).any() or not torch.isfinite(y).all()
            or not ((y[..., 0] == 0) | (y[..., 0] == 1)).all() or (y[..., 1] < 0).any()
            or not m.any(1).all() or (y[..., 1][y[..., 0] == 0] != 0).any()):
        raise ValueError("invalid danger data contract")
    return data, planes


def collect(actor_path, out, *, ledger, games=64, every=8, device="cpu", max_steps=4000):
    import riichi_py
    from .train_league import load_actor
    from .outcomes import validate_budget, require_finished
    require_research_engine(); validate_budget(games, max_steps)
    if type(every) is not int or every < 1: raise ValueError("every must be positive")
    if Path(out).exists(): raise FileExistsError(out)
    with tempfile.TemporaryDirectory(prefix="danger-source-") as folder:
        info = snapshot(Path(actor_path), Path(folder))
        actor, _, _ = load_actor(Path(folder) / info['file'], device)
        actor.eval()
        reserved = SeedLedger(Path(ledger)).reserve("train", games, "danger-labels")
        arena = riichi_py.Arena(games=games, seed=reserved['seed'])
        views = Views(arena, games, {"mortal"})
        blocks, labels, masks, identities = [], [], [], []
        generator = torch.Generator(device=device).manual_seed(reserved['seed'] ^ 0xDA93)
        steps = 0
        with torch.no_grad():
            while not arena.all_finished() and steps < max_steps:
                rows, players, legal = decision(views)
                if not len(rows): break
                # Decide sampling before labels/outcomes. All observed stages
                # in a sampled step are retained together.
                targets = None
                if steps % every == 0:
                    valid, truth = riichi_py.research_danger(arena)
                    targets = (np.frombuffer(valid, np.uint8).reshape(games, 2, 34).astype(bool),
                               np.frombuffer(truth, np.float32).reshape(games, 2, 3, 34, 2))
                chosen, records = mortal_learner.decide_in_mortal_space(
                    lambda p, m: rl_policy.logits(actor, p, m), views, rows, players, legal[rows],
                    device=device, generator=generator)
                if targets is not None:
                    stage = np.zeros(len(records.slots), dtype=np.int64)
                    seen = set()
                    for i, slot in enumerate(records.slots):
                        stage[i] = int(int(slot) in seen); seen.add(int(slot))
                    game = rows[records.slots]
                    valid = targets[0][game, stage]
                    keep = np.flatnonzero(valid.any(1))
                    if len(keep):
                        blocks.append(records.planes.rows(keep)); masks.append(valid[keep].copy())
                        labels.append(targets[1][game[keep], stage[keep]].copy())
                        identities.append(game[keep] + reserved['seed'])
                actions = np.zeros(games, np.int64); actions[rows] = chosen
                arena.step(actions.tolist()); steps += 1
        require_finished(arena, steps=steps, context="danger collection")
        if not blocks: raise ValueError("collection produced no discard positions")
        planes = Planes.cat(blocks)
        data = {"danger_version": VERSION, "actor": info, "reservation": reserved,
                "label": "legal ron and discarder payment for each opponent alone; not chosen ron",
                "planes": {k: torch.from_numpy(getattr(planes, k).copy()) for k in Planes.ARRAYS},
                "labels": torch.from_numpy(np.concatenate(labels)), "mask": torch.from_numpy(np.concatenate(masks)),
                "games": torch.from_numpy(np.concatenate(identities).astype(np.int64))}
        atomic_artifact(data, Path(out), load_data)
        return {"rows": len(planes), "games": games, "seed_reservation": reserved}


@torch.no_grad()
def measure(net, data, planes, rows, device, batch):
    if not len(rows): return {"rows": 0}
    n = count = hits = 0; brier = total = payment_error = 0.
    bins = np.zeros((10, 3), dtype=np.float64)
    net.eval()
    for start in range(0, len(rows), batch):
        pick = rows[start:start+batch]
        logits, payment = net(planes.rows(pick).dense(device))
        y, mask = data['labels'][pick].to(device), data['mask'][pick].to(device)
        active = mask[:, None, :].expand_as(logits)
        prob = logits.sigmoid(); truth = y[..., 0]
        total += float(loss_of(logits, payment, y, mask)) * len(pick); n += len(pick)
        brier += float((prob - truth).square()[active].sum()); count += int(active.sum())
        positive = active & (truth > .5)
        payment_error += float((payment * PAYMENT_SCALE - y[..., 1]).square()[positive].sum())
        hits += int(positive.sum())
        p = prob[active].cpu().numpy(); t = truth[active].cpu().numpy()
        index = np.minimum((p * 10).astype(int), 9)
        for i in range(10):
            selected = index == i
            bins[i] += (selected.sum(), p[selected].sum(), t[selected].sum())
    return {'rows': n, 'loss': total/n, 'brier': brier/max(count, 1), 'legal_targets': count,
            'positive_targets': hits, 'conditional_payment_rmse': (payment_error/hits)**.5 if hits else None,
            'calibration': [{'count': int(c), 'predicted': float(p/c) if c else None, 'observed': float(t/c) if c else None}
                            for c, p, t in bins]}


def load_predictor(path, device="cpu"):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (type(data.get('predictor_version')) is not int or data['predictor_version'] != VERSION
            or type(data.get('fitted_epochs')) is not int or data['fitted_epochs'] < 1):
        raise ValueError("attach only a fitted, versioned public danger predictor")
    net = Predictor(data['width'], data['blocks']); net.load_state_dict(data['predictor'])
    if any(not torch.isfinite(t).all() for t in net.state_dict().values()):
        raise ValueError("nonfinite danger predictor")
    return net.to(device).eval(), data


def fit(paths, out, *, epochs=8, batch=256, width=64, blocks=2, lr=1e-3, device="cpu", seed=1):
    if (not paths or type(epochs) is not int or type(batch) is not int or epochs < 1
            or batch < 1 or not np.isfinite(lr) or lr <= 0):
        raise ValueError("invalid fitting budget")
    if Path(out).exists(): raise FileExistsError(out)
    hashes = [digest(Path(p)) for p in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError('duplicate danger data')
    loaded = [load_data(p) for p in paths]
    reservations = [d['reservation'] for d, _ in loaded if 'reservation' in d]
    for i, a in enumerate(reservations):
        if any(max(a['seed'], b['seed']) < min(a['end'], b['end']) for b in reservations[:i]):
            raise ValueError('danger data seed blocks overlap')
    planes = Planes.cat([x[1] for x in loaded])
    data = {key: torch.cat([d[key] for d, _ in loaded]) for key in ('labels', 'mask', 'games')}
    train, val, test = split_games(data['games'].numpy())
    if not all(len(part) for part in (train, val, test)):
        raise ValueError("need more independent games for train/validation/test")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Predictor(width, blocks).to(device); optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    best = None; history = []
    for epoch in range(epochs):
        net.train(); order = rng.permutation(train)
        for start in range(0, len(order), batch):
            pick = order[start:start+batch]
            logits, payment = net(planes.rows(pick).dense(device))
            loss = loss_of(logits, payment, data['labels'][pick].to(device), data['mask'][pick].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        metrics = measure(net, data, planes, val, device, batch); history.append(metrics)
        if best is None or metrics['loss'] < best[0]:
            best = (metrics['loss'], epoch, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[2])
    saved = {'predictor_version': VERSION, 'fitted_epochs': epochs, 'selected_epoch': best[1],
             'width': width, 'blocks': blocks, 'predictor': best[2], 'history': history,
             'test': measure(net, data, planes, test, device, batch),
             'source_hashes': hashes, 'input': 'public-observation-only'}
    atomic_artifact(saved, Path(out), load_predictor)
    return {k: v for k, v in saved.items() if k != 'predictor'}


def predictor_fingerprint(predictor):
    """Tensor-content identity survives serialization and device transfers."""
    sha = hashlib.sha256()
    sha.update(f"danger-{VERSION}-{predictor.width}-{predictor.blocks}".encode())
    for name, value in sorted(predictor.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        sha.update(name.encode()); sha.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        sha.update(tensor.numpy().tobytes())
    return sha.hexdigest()


class DangerPolicy(nn.Module):
    kind = "mortal"
    actions = 46
    speaks_mortal = True

    def __init__(self, base, base_kind, base_origin, predictor, predictor_meta, predictor_sha):
        super().__init__()
        self.base, self.base_kind, self.base_origin = base, base_kind, base_origin
        self.predictor = predictor.eval().requires_grad_(False)
        self.predictor_meta, self.predictor_sha = predictor_meta, predictor_sha
        self.frozen_predictor_hash = predictor_fingerprint(predictor)
        self.residual = nn.Sequential(nn.Linear(6*34 + 2*46, 128), nn.SiLU(), nn.Linear(128, 46))
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)

    @property
    def planes(self): return self.base.planes if hasattr(self.base, 'planes') else PLANES

    @property
    def channels(self): return getattr(self.base, 'channels', None)

    @property
    def blocks(self): return getattr(self.base, 'blocks', None)

    def train(self, mode=True):
        super().train(mode); self.predictor.eval().requires_grad_(False); return self

    def policy_only(self, observation, legal):
        base = rl_policy.logits(self.base, observation, legal).float()
        with torch.no_grad():
            features = self.predictor.features(observation).flatten(1)
        safe = torch.where(legal, base, 0.)
        correction = self.residual(torch.cat([features, safe, legal.float()], dim=1))
        return (base + correction).masked_fill(~legal, -torch.inf)

    def forward(self, observation, legal):
        # Value/hand displays retain their base model's documented semantics.
        return self.policy_only(observation, legal), self.base(observation, legal)[1]

    def state(self):
        if predictor_fingerprint(self.predictor) != self.frozen_predictor_hash:
            raise ValueError('frozen danger predictor changed during actor training')
        from .train_league import actor_state
        return {'danger_policy': VERSION, 'base': actor_state(self.base, self.base_kind, self.base_origin),
                'predictor_artifact': {**self.predictor_meta, 'predictor': self.predictor.state_dict()},
                'predictor_sha256': self.predictor_sha, 'predictor_tensor_sha256': self.frozen_predictor_hash,
                'residual': self.residual.state_dict()}


def from_payload(data, device):
    from .train_league import load_actor
    if type(data.get('danger_policy')) is not int or data['danger_policy'] != VERSION or data['base'].get('danger_policy'):
        raise ValueError("unsupported/nested danger policy")
    with tempfile.TemporaryDirectory(prefix='danger-load-') as folder:
        base = Path(folder)/'base.pt'; predictor = Path(folder)/'predictor.pt'
        torch.save(data['base'], base); torch.save(data['predictor_artifact'], predictor)
        net, origin, kind = load_actor(base, device)
        pred, meta = load_predictor(predictor, device)
    joined = DangerPolicy(net, kind, origin, pred, meta, data['predictor_sha256']).to(device)
    if joined.frozen_predictor_hash != data.get('predictor_tensor_sha256'):
        raise ValueError('frozen danger predictor fingerprint mismatch')
    joined.residual.load_state_dict(data['residual'])
    if any(not torch.isfinite(v).all() for v in joined.residual.state_dict().values()):
        raise ValueError('nonfinite danger policy residual')
    return joined


def attach(actor_path, predictor_path, out, device="cpu"):
    from .train_league import load_actor
    if Path(out).exists(): raise FileExistsError(out)
    with tempfile.TemporaryDirectory(prefix='danger-attach-') as folder:
        folder = Path(folder)
        a = snapshot(Path(actor_path), folder); p = snapshot(Path(predictor_path), folder)
        base, origin, kind = load_actor(folder/a['file'], device)
        if kind == 'danger': raise ValueError('danger predictors cannot be nested')
        pred, meta = load_predictor(folder/p['file'], device)
        joined = DangerPolicy(base, kind, origin, pred, meta, p['sha256']).to(device)
        atomic_save({**joined.state(), 'generation': 0, 'actor_source': a}, Path(out))
    return {'out': str(out), 'predictor_sha256': p['sha256'], 'initial_policy': 'unchanged'}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect'); c.add_argument('actor_path', type=Path)
    c.add_argument('--ledger', type=Path, required=True); c.add_argument('--games', type=int, default=64)
    c.add_argument('--every', type=int, default=8)
    f = sub.add_parser('fit'); f.add_argument('paths', nargs='+', type=Path)
    f.add_argument('--epochs', type=int, default=8); f.add_argument('--batch', type=int, default=256)
    f.add_argument('--width', type=int, default=64); f.add_argument('--blocks', type=int, default=2)
    f.add_argument('--lr', type=float, default=1e-3)
    a = sub.add_parser('attach'); a.add_argument('actor_path', type=Path); a.add_argument('predictor_path', type=Path)
    for cmd in (c, f, a):
        cmd.add_argument('--out', type=Path, required=True); cmd.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = vars(p.parse_args()); cmd = args.pop('cmd'); torch.set_num_threads(2)
    print(json.dumps({'collect': collect, 'fit': fit, 'attach': attach}[cmd](**args), allow_nan=False))


if __name__ == '__main__': main()
