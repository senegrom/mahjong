"""Opt-in learning experiments; no subcommand promotes or deploys a model.

Run ``python -m neural.learning_lab --help``. The original training commands are
unchanged. Every output path must be new; exact training resume inherits all
configuration and requires identical source/runtime/data identities.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile

import torch

from ..checkpoints import atomic_save, copy_checkpoint
from . import artifacts
from .config import Config


def configuration(args, default=True):
    values = {}
    if getattr(args, 'preset', None):
        path = Path(__file__).parent / 'presets' / (args.preset + '.json')
        values.update(json.loads(path.read_text()))
    if getattr(args, 'config', None):
        supplied = json.loads(args.config.read_text())
        if not isinstance(supplied, dict): raise ValueError('config must be a JSON object')
        values.update(supplied)
    for assignment in getattr(args, 'set', []) or []:
        key, sep, raw = assignment.partition('=')
        if not sep: raise ValueError('--set requires field=value')
        try: value = json.loads(raw)
        except json.JSONDecodeError: value = raw
        values[key] = value
    return Config(**values).validate() if values or default else None


def add_config(parser):
    parser.add_argument('--preset', choices=('baseline', 'oracle', 'categorical', 'placement', 'defence', 'league', 'all', 'exploiter'))
    parser.add_argument('--config', type=Path, help='JSON overrides of Config defaults')
    parser.add_argument('--set', action='append', default=[], help='one configuration override: field=value')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=2)
    sub = p.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train', help='fresh on-policy PPO with independent critics')
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path); source.add_argument('--resume', type=Path)
    train.add_argument('--out', type=Path, required=True)
    train.add_argument('--rounds', type=int, default=1, help='additional policy rounds, after any configured critic warm-up')
    train.add_argument('--opponents', nargs='*', type=Path, default=[])
    train.add_argument('--roles', nargs='*', default=[])
    train.add_argument('--lessons', type=Path)
    train.add_argument('--critic-init', type=Path, help='prefit critics for exactly this frozen starting policy')
    add_config(train)
    collect = sub.add_parser('collect', help='complete frozen-policy games with public root traces and training labels')
    collect.add_argument('--checkpoint', type=Path, required=True)
    collect.add_argument('--peer', type=Path, help='optional public-only disagreement selector')
    collect.add_argument('--out', type=Path, required=True)
    add_config(collect)
    critics = sub.add_parser('fit-critics', help='supervised critic prefit, with a whole-game holdout')
    critics.add_argument('--round', dest='data', nargs='+', type=Path, required=True)
    critics.add_argument('--out', type=Path, required=True)
    critics.add_argument('--epochs', type=int, default=3)
    critics.add_argument('--validation-every', type=int, default=5)
    add_config(critics)
    again = sub.add_parser('reanalyse', help='new teacher labels on old public roots, written as a new artifact')
    again.add_argument('--trace', type=Path, required=True)
    again.add_argument('--teacher', type=Path, required=True)
    again.add_argument('--placement-head', type=Path)
    again.add_argument('--out', type=Path, required=True)
    for name, default in (('worlds', 8), ('candidates', 4), ('confirm-worlds', 16), ('audit-worlds', 8),
                          ('race-budget', 0), ('batch', 256), ('seed', 81231)):
        again.add_argument('--' + name, type=int, default=default)
    again.add_argument('--margin', type=float, default=2.)
    again.add_argument('--eta', type=float, default=1.)
    again.add_argument('--candidate-method', choices=('top', 'gumbel'), default='top')
    ranker = sub.add_parser('fit-ranker', help='paired advantages from independently audited root candidates')
    ranker.add_argument('--checkpoint', type=Path, required=True)
    ranker.add_argument('--lessons', type=Path, required=True)
    ranker.add_argument('--out', type=Path, required=True)
    ranker.add_argument('--features', choices=('ours', 'fused', 'own-tower'), default='fused')
    for name, default in (('channels', 32), ('epochs', 3), ('batch', 128), ('validation-every', 5), ('seed', 27)):
        ranker.add_argument('--' + name, type=int, default=default)
    ranker.add_argument('--learning-rate', type=float, default=1e-4)
    distill = sub.add_parser('distill', help='supervised lessons only; never reuse their old PPO likelihoods')
    distill.add_argument('--checkpoint', type=Path, required=True)
    distill.add_argument('--lessons', type=Path, required=True)
    distill.add_argument('--out', type=Path, required=True)
    distill.add_argument('--epochs', type=int, default=2)
    add_config(distill)
    evaluate = sub.add_parser('evaluate-ranker', help='explicit development duel in both seat-role directions; no promotion')
    evaluate.add_argument('--checkpoint', type=Path, required=True)
    evaluate.add_argument('--ranker', type=Path, required=True)
    evaluate.add_argument('--opponent', type=Path, required=True)
    evaluate.add_argument('--out', type=Path, required=True)
    evaluate.add_argument('--games', type=int, default=32)
    evaluate.add_argument('--seed', type=int, default=9_100_000)
    evaluate.add_argument('--candidates', type=int, default=4)
    evaluate.add_argument('--margin', type=float, default=.05)
    for child in (train, collect, critics, again, ranker, distill, evaluate):
        child.add_argument('--device', default='cpu', help='explicit CPU or CUDA device; no cloud job is started')
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.threads < 1: p.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if torch.device(args.device).type not in ('cpu', 'cuda'): p.error('CPU/CUDA devices only')
    if args.command == 'train':
        from .train import run
        result = run(args.resume or args.checkpoint, args.out, cfg=configuration(args, not bool(args.resume)),
                     rounds=args.rounds, resume=bool(args.resume), device=args.device, opponents=args.opponents,
                     roles=args.roles, lessons=args.lessons, critic_init=args.critic_init)
    elif args.command == 'collect':
        from .collection import collect
        cfg = configuration(args)
        if cfg.role == 'exploiter' or cfg.adaptive_league:
            p.error('league/exploiter collection is performed by train with its frozen --opponents')
        if args.out.exists(): raise FileExistsError('collection output must be new')
        with tempfile.TemporaryDirectory(prefix='mahjong-lab-collect-') as temporary:
            path = Path(temporary) / 'actor.pt'; copy_checkpoint(args.checkpoint, path)
            net = artifacts.actor_from_payload(torch.load(path, map_location='cpu', weights_only=True), args.device).eval()
            peer = None
            if args.peer:
                path = Path(temporary) / 'peer.pt'; copy_checkpoint(args.peer, path)
                peer = artifacts.actor_from_payload(torch.load(path, map_location='cpu', weights_only=True), args.device).eval()
            torch.manual_seed(cfg.seed)
            data = collect(net, cfg, cfg.seed, peer=peer)
            result = dict(identity=data.save(args.out), decisions=len(data.observations), roots=len(data.trace['roots']))
    elif args.command == 'fit-critics':
        from .fit import fit_critics
        result = fit_critics(args.data, args.out, configuration(args), epochs=args.epochs,
                             validation_every=args.validation_every, device=args.device)
    elif args.command == 'reanalyse':
        from .reanalysis import run
        controls = {name: getattr(args, name) for name in ('worlds', 'candidates', 'confirm_worlds', 'audit_worlds',
                     'race_budget', 'batch', 'seed', 'margin', 'eta', 'candidate_method', 'device', 'placement_head')}
        data = run(args.trace, args.teacher, args.out, **controls)
        result = dict(identity=data.identity, roots=len(data.roots()), policy_rows=len(data),
                      confirmed=int(data.arrays['confirmed'].sum()))
    elif args.command == 'fit-ranker':
        from .ranker import fit
        controls = {name: getattr(args, name) for name in ('channels', 'epochs', 'batch', 'validation_every', 'seed', 'learning_rate', 'device')}
        result = fit(args.checkpoint, args.lessons, args.out, mode=args.features, **controls)
    elif args.command == 'distill':
        from .fit import distill
        result = distill(args.checkpoint, args.lessons, args.out, epochs=args.epochs,
                         cfg=configuration(args), device=args.device)
    else:
        from .. import duel, zoo
        from .ranker import load, Player
        if args.out.exists() or args.games < 2 or args.seed < 9_000_000:
            raise ValueError('evaluation needs a new output and >=2 deals in the held-out seed range')
        with tempfile.TemporaryDirectory(prefix='mahjong-lab-evaluation-') as temporary:
            snapshots = []
            for i, path in enumerate((args.checkpoint, args.ranker, args.opponent)):
                snapshot = Path(temporary) / f'{i}.pt'; copy_checkpoint(path, snapshot); snapshots.append(snapshot)
            net = artifacts.actor_from_payload(torch.load(snapshots[0], map_location='cpu', weights_only=True), args.device).eval()
            player = Player(net, load(snapshots[1], net, args.device), candidates=args.candidates, margin=args.margin, device=args.device)
            other = zoo.load_player(snapshots[2], args.device).eval()
            result = dict(version=1, seed=args.seed, kind='development_duel_not_promotion',
                          identities=[artifacts.digest_file(s) for s in snapshots],
                          forward=duel.duel(player, other, args.games, args.seed, args.device),
                          reverse=duel.duel(other, player, args.games, args.seed, args.device))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open('x') as stream: stream.write(artifacts.canonical(result).decode())
    print(json.dumps(str(result) if isinstance(result, Path) else result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
