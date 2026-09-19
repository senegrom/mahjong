"""Separate counterfactual action-difference research, never PPO replay.

Collect paired FINAL-PLACEMENT returns for a frozen public policy's top-k
first-stage actions. Intervene BEFORE the two-stage riichi translator acts,
so a forced alternative never leaves a fictitious reach in its follower.
Fit public action differences, not labels of which move won on one hidden wall.

    python -m neural.paired_actions collect actor.pt --ledger seeds.json --games 256 --out pairs.pt
    python -m neural.paired_actions fit pairs.pt --out ranker.pt
    python -m neural.paired_actions evaluate actor.pt ranker.pt --ledger seeds.json --out evaluation.json

The ranker is experimental and off by default. No training or browser path
loads it automatically. Its actor checkpoint and shortlist width are immutable.
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
import riichi_py

from . import mortal_learner, rl_policy, zoo
from .checkpoints import atomic_artifact
from .evidence import split_games
from .league_state import snapshot, digest
from .observe import Planes, Views, PLANES
from .outcomes import placements, require_finished, validate_budget
from .rl_critics import Residual
from .seed_ledger import SeedLedger, atomic_json
from .train_league import load_actor

VERSION = 1


def shortlist(scores, legal, k):
    """Policy-ordered, padded top-k legal actions; ties keep action order."""
    if type(k) is not int or not 2 <= k <= 46:
        raise ValueError('candidates must be between 2 and 46')
    legal = np.asarray(legal, dtype=bool)
    scores = np.asarray(scores)
    if scores.shape != legal.shape or not legal.any(axis=1).all() or not np.isfinite(scores[legal]).all():
        raise ValueError('invalid policy scores or legal masks')
    order = np.argsort(-np.where(legal, scores, -np.inf), axis=1, kind='stable')[:, :k]
    valid = np.take_along_axis(legal, order, axis=1)
    return order.astype(np.int64), valid


@torch.no_grad()
def play_pass(actor, *, games, seed, root_step, candidates, device, forced=None, max_steps=4000):
    """Return root information and final scores from one deterministic continuation."""
    validate_budget(games, max_steps)
    if type(root_step) is not int or not 0 <= root_step < max_steps:
        raise ValueError('root_step is outside the rollout budget')
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=[])
    views = Views(arena, games, {'mortal'})
    actor.eval()
    root = None
    steps = 0
    while not arena.all_finished() and steps < max_steps:
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        rows = np.flatnonzero(seats != 255)
        if not len(rows):
            break
        views.advance()
        legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, 78).astype(bool)
        people = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
        players = people[rows, seats[rows]].astype(np.int64)
        views.prepare(rows, players)
        calls = 0
        first = {}

        def score(planes, mask):
            nonlocal calls
            logits = rl_policy.logits(actor, planes, mask).float()
            if steps == root_step and calls == 0:
                raw = logits.cpu().numpy()
                allowed = mask.cpu().numpy()
                choices, valid = shortlist(raw, allowed, candidates)
                first.update(candidates=choices, valid=valid, legal=allowed)
                if forced:
                    logits = logits.clone()
                    for index, game in enumerate(rows):
                        action = forced.get(int(game))
                        if action is not None:
                            if not 0 <= action < 46 or not allowed[index, action]:
                                raise ValueError('counterfactual action is illegal at its root')
                            logits[index].fill_(-torch.inf)
                            logits[index, action] = 0.
            calls += 1
            return logits

        chosen, records = mortal_learner.decide_in_mortal_space(
            score, views, rows, players, legal[rows], greedy=True, device=device)
        if steps == root_step:
            # First-stage records precede any additional reach-discard rows.
            indices = []
            for slot in range(len(rows)):
                found = np.flatnonzero(records.slots == slot)
                if not len(found):
                    raise ValueError('unrecorded first-stage action cannot be a paired root')
                indices.append(int(found[0]))
            root = {'planes': records.planes.rows(np.asarray(indices)),
                    'games': rows.copy(), 'players': players.copy(), **first}
        action = np.zeros(games, dtype=np.int64)
        action[rows] = chosen
        arena.step(action.tolist())
        steps += 1
    require_finished(arena, steps=steps, context='paired action continuation')
    final = np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, 4).copy()
    return root, final


def root_fingerprint(root):
    h = hashlib.sha256()
    for array in [root['games'], root['players'], root['legal'], root['candidates'], root['valid'],
                  root['planes'].indptr, root['planes'].indices, root['planes'].values]:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def collect(actor_path, out, *, ledger, games=256, root_step=40, candidates=4,
            device='cpu', max_steps=4000):
    validate_budget(games, max_steps)
    if type(candidates) is not int or not 2 <= candidates <= 46 or not 0 <= root_step < max_steps:
        raise ValueError("invalid root or candidate budget")
    if Path(out).exists():
        raise FileExistsError('paired data is immutable; select a new output file')
    with tempfile.TemporaryDirectory(prefix='pairs-') as folder:
        info = snapshot(Path(actor_path), Path(folder))
        actor, _, _ = load_actor(Path(folder) / info['file'], device)
        reservation = SeedLedger(Path(ledger)).reserve('counterfactual', games, info['sha256'])
        options = dict(games=games, seed=reservation['seed'], root_step=root_step,
                       candidates=candidates, device=device, max_steps=max_steps)
        roots, finals = play_pass(actor, **options)
        if roots is None:
            raise ValueError('no roots at selected step')
        expected = root_fingerprint(roots)
        payoffs = np.zeros((len(roots['games']), candidates), dtype=np.float32)
        payoffs[:, 0] = 2.5 - placements(finals)[roots['games'], roots['players']]
        for slot in range(1, candidates):
            forced = {int(game): int(roots['candidates'][i, slot])
                      for i, game in enumerate(roots['games']) if roots['valid'][i, slot]}
            if not forced:
                continue
            observed, finals = play_pass(actor, forced=forced, **options)
            if observed is None or root_fingerprint(observed) != expected:
                raise RuntimeError('paired passes disagree before intervention')
            payoffs[:, slot] = 2.5 - placements(finals)[roots['games'], roots['players']]
        keep = np.flatnonzero(roots['valid'].sum(axis=1) >= 2)
        if not len(keep):
            raise ValueError('no root offers multiple legal alternatives')
        planes = roots['planes'].rows(keep)
        payload = {'paired_version': VERSION, 'actor': info, 'shortlist': candidates,
                   'root_step': root_step, 'reservation': reservation, 'objective': 'placement-v2',
                   'planes': {name: torch.from_numpy(getattr(planes, name).copy()) for name in Planes.ARRAYS},
                   'games': torch.tensor(reservation['seed'] + roots['games'][keep], dtype=torch.int64),
                   'players': torch.from_numpy(roots['players'][keep]),
                   'legal': torch.from_numpy(roots['legal'][keep]),
                   'candidates': torch.from_numpy(roots['candidates'][keep]),
                   'valid': torch.from_numpy(roots['valid'][keep]), 'payoffs': torch.from_numpy(payoffs[keep]),
                   'label': 'paired noisy placement returns; NOT hindsight-optimal actions'}
        atomic_artifact(payload, Path(out), load_data)
        return {'roots': len(keep), 'candidate_slots': candidates, 'actor_sha256': info['sha256'],
                'seed_reservation': reservation}


def load_data(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('paired_version') != VERSION or saved.get('objective') != 'placement-v2':
        raise ValueError('not a paired-action dataset')
    n = len(saved['games']); k = saved['shortlist']
    if not n or type(k) is not int or not 2 <= k <= 46:
        raise ValueError('invalid paired dataset size')
    planes = Planes(**{key: value.numpy() for key, value in saved['planes'].items()})
    if len(planes) != n or saved['legal'].shape != (n, 46):
        raise ValueError('paired observations and masks disagree')
    for key in ('valid', 'candidates', 'payoffs'):
        if saved[key].shape != (n, k): raise ValueError('invalid paired target shape')
    if saved['candidates'].dtype != torch.int64 or saved['valid'].dtype != torch.bool:
        raise ValueError('invalid paired target types')
    moves, valid = saved['candidates'], saved['valid']
    if ((moves < 0) | (moves >= 46)).any() or not valid[:, 0].all() or (valid.sum(dim=1) < 2).any():
        raise ValueError('invalid candidate actions')
    if not saved['legal'].gather(1, moves)[valid].all():
        raise ValueError('illegal paired candidate')
    payoff = saved['payoffs']
    if not torch.isfinite(payoff).all() or (payoff[valid].abs() > 1.5).any():
        raise ValueError('invalid placement payoffs')
    return saved, planes


class ActionDifferences(nn.Module):
    """Public-only action values; their common offset is intentionally unidentified."""
    def __init__(self, width=64, blocks=2):
        super().__init__()
        if type(width) is not int or width < 8 or width % 8 or type(blocks) is not int or blocks < 1:
            raise ValueError('invalid ranker architecture')
        self.width, self.blocks = width, blocks
        self.stem = nn.Conv1d(PLANES, width, 1)
        self.tower = nn.Sequential(*(Residual(width) for _ in range(blocks)))
        self.head = nn.Linear(width * 2, 46)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, observation):
        features = self.tower(self.stem(observation.float()))
        return self.head(torch.cat([features.mean(dim=2), features.amax(dim=2)], dim=1))


def paired_loss(scores, moves, payoffs, valid):
    selected = scores.gather(1, moves)
    predicted = selected[:, 1:] - selected[:, :1]
    wanted = payoffs[:, 1:] - payoffs[:, :1]
    mask = valid[:, 1:]
    # Equal root weight, not extra weight for roots with more legal candidates.
    return (((predicted - wanted).square() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).mean()


@torch.no_grad()
def measure(ranker, payload, planes, rows, *, device, batch):
    if not len(rows): return {'roots': 0}
    ranker.eval(); loss = 0.; gains = []
    for start in range(0, len(rows), batch):
        pick = rows[start:start + batch]
        score = ranker(planes.rows(pick).dense(device))
        moves = payload['candidates'][pick].to(device); valid = payload['valid'][pick].to(device)
        payoff = payload['payoffs'][pick].to(device)
        loss += float(paired_loss(score, moves, payoff, valid)) * len(pick)
        # Select by the learned PUBLIC prediction, never by the realized return.
        choice = score.gather(1, moves).masked_fill(~valid, -torch.inf).argmax(dim=1)
        gain = payoff.gather(1, choice[:, None]).squeeze(1) - payoff[:, 0]
        gains.extend(gain.cpu().tolist())
    grouped = {}
    for game, gain in zip(payload['games'][rows].tolist(), gains):
        grouped.setdefault(game, []).append(gain)
    by_deal = np.asarray([np.mean(values) for values in grouped.values()])
    return {'roots': len(rows), 'independent_deals': len(by_deal), 'paired_mse': loss / len(rows),
            'selected_placement_edge': float(by_deal.mean()),
            'standard_error': float(by_deal.std(ddof=1) / np.sqrt(len(by_deal))) if len(by_deal) > 1 else None,
            'diagnostic_only': True}


def fit(paths, out, *, epochs=8, batch=256, width=64, blocks=2, lr=1e-3, device='cpu', seed=3):
    if not paths or min(epochs, batch) < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('invalid fitting controls')
    loaded = [load_data(path) for path in paths]
    source = loaded[0][0]
    for data, _ in loaded:
        if (data['actor']['sha256'] != source['actor']['sha256'] or data['shortlist'] != source['shortlist']):
            raise ValueError('cannot mix actors or shortlist definitions')
    planes = Planes.cat([p for _, p in loaded])
    fields = ('games', 'players', 'legal', 'candidates', 'valid', 'payoffs')
    data = {key: torch.cat([p[key] for p, _ in loaded]) for key in fields}
    training, validation, test = split_games(data['games'].numpy())
    if not all(len(part) for part in (training, validation, test)):
        raise ValueError('collect more independent games: train/validation/test must be nonempty')
    torch.manual_seed(seed)
    ranker = ActionDifferences(width, blocks).to(device)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed); best = None; history = []
    for epoch in range(epochs):
        ranker.train()
        order = rng.permutation(training)
        for start in range(0, len(order), batch):
            rows = order[start:start + batch]
            score = ranker(planes.rows(rows).dense(device))
            loss = paired_loss(score, data['candidates'][rows].to(device),
                               data['payoffs'][rows].to(device), data['valid'][rows].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(ranker.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        watched = measure(ranker, data, planes, validation, device=device, batch=batch)
        history.append({'epoch': epoch, 'validation': watched})
        if best is None or watched['paired_mse'] < best[0]:
            best = (watched['paired_mse'], epoch, {k: v.detach().cpu().clone() for k, v in ranker.state_dict().items()})
    ranker.load_state_dict(best[2])
    result = {'ranker_version': VERSION, 'actor_sha256': source['actor']['sha256'],
              'shortlist': source['shortlist'], 'width': width, 'blocks': blocks, 'model': best[2],
              'objective': 'placement-v2', 'selected_epoch': best[1], 'history': history,
              'test': measure(ranker, data, planes, test, device=device, batch=batch),
              'sources': [str(path) for path in paths]}
    atomic_artifact(result, Path(out), lambda path: load_ranker(path, 'cpu'))
    return {key: value for key, value in result.items() if key != 'model'}


def load_ranker(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('ranker_version') != VERSION or saved.get('objective') != 'placement-v2':
        raise ValueError('unsupported action ranker')
    net = ActionDifferences(saved['width'], saved['blocks'])
    net.load_state_dict(saved['model'])
    return net.to(device).eval(), saved


class RankedPlayer:
    kind = 'mortal'
    def __init__(self, actor, ranker, k, device):
        self.actor, self.ranker, self.k, self.device = actor, ranker, k, device

    def eval(self):
        self.actor.eval(); self.ranker.eval(); return self

    @torch.no_grad()
    def choose(self, views, rows, players, legal):
        first = True
        def score(planes, mask):
            nonlocal first
            logits = rl_policy.logits(self.actor, planes, mask).float()
            if not first: return logits  # The training intervention was FIRST stage only.
            first = False
            moves, valid = shortlist(logits.cpu().numpy(), mask.cpu().numpy(), self.k)
            indices = torch.from_numpy(moves).to(self.device)
            allowed = torch.from_numpy(valid).to(self.device)
            scores = self.ranker(planes).gather(1, indices).masked_fill(~allowed, -torch.inf)
            picked = indices.gather(1, scores.argmax(dim=1)[:, None])
            answer = torch.full_like(logits, -torch.inf); answer.scatter_(1, picked, 0.)
            return answer
        chosen, _ = mortal_learner.decide_in_mortal_space(
            score, views, rows, players, legal, greedy=True, device=self.device)
        return chosen


def evaluate(actor_path, ranker_path, *, ledger, games=512, device='cpu'):
    from . import duel
    with tempfile.TemporaryDirectory(prefix='ranker-evaluation-') as folder:
        actor_info = snapshot(Path(actor_path), Path(folder))
        ranker_info = snapshot(Path(ranker_path), Path(folder))
        ranker, saved = load_ranker(Path(folder) / ranker_info['file'], device)
        if actor_info['sha256'] != saved['actor_sha256']:
            raise ValueError('ranker belongs to a different frozen actor')
        actor, _, _ = load_actor(Path(folder) / actor_info['file'], device)
        player = RankedPlayer(actor, ranker, saved['shortlist'], device).eval()
        control = zoo.load_player(Path(folder) / actor_info['file'], device)
        reservation = SeedLedger(Path(ledger)).reserve('test', games, 'paired-ranker-evaluation')
        forward = duel.duel(player, control, games, reservation['seed'], device)
        reverse = duel.duel(control, player, games, reservation['seed'], device)
        return {'actor': actor_info, 'ranker': ranker_info, 'seed_reservation': reservation,
                'ranked_one_vs_three': forward, 'control_one_vs_three': reverse,
                'diagnostic_only': True, 'champion_written': False}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    collect_p = sub.add_parser('collect'); collect_p.add_argument('actor_path', type=Path)
    collect_p.add_argument('--ledger', type=Path, required=True); collect_p.add_argument('--games', type=int, default=256)
    collect_p.add_argument('--root-step', type=int, default=40); collect_p.add_argument('--candidates', type=int, default=4)
    fit_p = sub.add_parser('fit'); fit_p.add_argument('paths', type=Path, nargs='+')
    fit_p.add_argument('--epochs', type=int, default=8); fit_p.add_argument('--batch', type=int, default=256)
    fit_p.add_argument('--width', type=int, default=64); fit_p.add_argument('--blocks', type=int, default=2)
    fit_p.add_argument('--lr', type=float, default=1e-3)
    evaluation_p = sub.add_parser('evaluate'); evaluation_p.add_argument('actor_path', type=Path)
    evaluation_p.add_argument('ranker_path', type=Path); evaluation_p.add_argument('--ledger', type=Path, required=True)
    evaluation_p.add_argument('--games', type=int, default=512)
    for command in (collect_p, fit_p, evaluation_p):
        command.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        command.add_argument('--out', type=Path, required=True)
    args = vars(p.parse_args()); command = args.pop('command'); torch.set_num_threads(2)
    if command == 'collect': result = collect(**args)
    elif command == 'fit': result = fit(**args)
    else:
        out = args.pop('out'); result = evaluate(**args); atomic_json(out, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
