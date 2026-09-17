"""Executable opt-in PPO experiments with separate critics and exact checkpoints.

PPO consumes only fresh actions sampled by this actor. Reanalysis lessons, when
explicitly supplied, add a separate supervised term, never fabricated PPO ratios.
Each new run writes a new directory; only completed rounds can be checkpointed.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from ..checkpoints import atomic_save, copy_checkpoint
from . import artifacts
from .collection import collect
from .config import Config
from .critics import Critics, UTILITY
from .league import League


def _policy(net, planes, legal, cfg):
    with torch.autocast(planes.device.type, dtype=torch.bfloat16, enabled=cfg.amp and planes.device.type == 'cuda'):
        fast = getattr(net, 'policy_only', None)
        logits = fast(planes, legal) if callable(fast) else net(planes, legal)[0]
    logits = logits.float().masked_fill(~legal, -torch.inf)
    if not torch.isfinite(logits[legal]).all(): raise FloatingPointError('nonfinite legal policy logits')
    return logits


def masked_kl(before, logits, legal):
    logp = logits.log_softmax(1).masked_fill(~legal, 0.)
    before = before.detach().masked_fill(~legal, 0.)
    oldlog = torch.where(before > 0, before, torch.ones_like(before)).log()
    return (before * (oldlog - logp)).sum(1).mean()


def runtime(device):
    return dict(torch=str(torch.__version__), numpy=str(np.__version__), device=str(torch.device(device)),
                threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled())


def source_identity():
    h = hashlib.sha256(artifacts.canonical(artifacts.engine_identity()))
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / 'neural').glob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))
    for p in files:
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def _restore_optimizer(optimizer, state, active):
    """No silent Adam reset or slot rebinding on exact continuation."""
    if not isinstance(state, dict) or not isinstance(state.get('state'), dict):
        raise ValueError('missing optimizer state')
    if not isinstance(active, list) or any(type(i) is not int for i in active) or sorted(state['state']) != active:
        raise ValueError('missing or extra active optimizer slots')
    wanted = optimizer.state_dict()['param_groups']
    groups = state.get('param_groups', [])
    if groups != wanted:
        raise ValueError('optimizer settings/parameter slots changed on resume')
    allowed = {i for group in groups for i in group['params']}
    if any(type(i) is not int or i not in allowed for i in state.get('state', {})):
        raise ValueError('unknown optimizer slots')
    optimizer.load_state_dict(state)
    for p, data in optimizer.state.items():
        if set(data) != {'step', 'exp_avg', 'exp_avg_sq'} or any(data[k].shape != p.shape for k in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('damaged Adam moments')
        if data['step'].ndim or not torch.isfinite(data['step']) or data['step'].item() <= 0 or data['step'].item() % 1:
            raise ValueError('damaged Adam update count')
    artifacts.finite(state)


def _cpu(tree):
    if isinstance(tree, torch.Tensor): return tree.detach().cpu().clone()
    if isinstance(tree, dict): return {k: _cpu(v) for k, v in tree.items()}
    if isinstance(tree, list): return [_cpu(v) for v in tree]
    if isinstance(tree, tuple): return tuple(_cpu(v) for v in tree)
    return deepcopy(tree)


class Learner:
    def __init__(self, payload, cfg: Config, *, device='cpu', saved=None,
                 opponent_payloads=(), members=(), lessons=None, critic_init=None):
        cfg.validate()
        if torch.device(device).type not in ('cpu', 'cuda'): raise ValueError('CPU/CUDA only')
        self.cfg, self.device = cfg, device
        self.actor = artifacts.actor_from_payload(payload, device)
        if hasattr(self.actor, 'set_mode'): self.actor.set_mode(cfg.fixed)
        self.actor.eval()
        self.critics = Critics(cfg).to(device)
        self.optimizer = torch.optim.AdamW([p for p in self.actor.parameters() if p.requires_grad], lr=cfg.learning_rate, weight_decay=1e-4)
        self.critic_optimizer = torch.optim.AdamW(self.critics.parameters(), lr=cfg.critic_learning_rate, weight_decay=1e-4)
        self.signature = [[n, list(p.shape), str(p.dtype), p.requires_grad] for n, p in self.actor.named_parameters()]
        self.reference_payload = _cpu(artifacts.actor_payload(self.actor)) if cfg.anchor_weight else None
        self.reference = None
        self.opponent_payloads = list(opponent_payloads)
        self.members = list(members)
        self.actors = []
        self.league = League(self.members, cfg.league_uniform, cfg.adaptive_league)
        self.rounds = self.actor_rounds = self.updates = self.critic_updates = 0
        self.initial_generation = artifacts.generation(payload)
        self.critic_origin = None
        self.initial_actor_id = artifacts.digest_state(artifacts.actor_payload(self.actor))
        self.ready = True
        self.lessons = lessons
        self.lesson_id = lessons.identity if lessons is not None else None
        self.reports = []
        self.data = []
        self.rng = np.random.default_rng(cfg.seed)
        self.runtime = runtime(device); self.source = source_identity()
        if saved is not None:
            for key, expected in (('version', 1), ('config', cfg.describe()), ('runtime', self.runtime),
                                  ('source', self.source), ('signature', self.signature), ('lesson_id', self.lesson_id)):
                if artifacts.canonical(saved.get(key)) != artifacts.canonical(expected):
                    raise ValueError(f'Exact resume {key} differs')
            self.reference_payload = saved['reference']
            self.opponent_payloads = saved['opponents']; self.members = saved['members']
            self.league = League(self.members, cfg.league_uniform, cfg.adaptive_league, saved['league'])
            self.critics.load_state_dict(saved['critics'], strict=True)
            _restore_optimizer(self.optimizer, saved['optimizer'], saved.get('active_optimizer'))
            _restore_optimizer(self.critic_optimizer, saved['critic_optimizer'], saved.get('active_critic_optimizer'))
            for key in ('rounds', 'actor_rounds', 'updates', 'critic_updates', 'initial_generation'):
                if type(saved[key]) is not int or saved[key] < 0: raise ValueError('damaged completion counters')
                setattr(self, key, saved[key])
            if self.actor_rounds != max(0, self.rounds - cfg.warmup_rounds) or not self.critic_updates or (self.actor_rounds and not self.updates):
                raise ValueError('checkpoint is not a completed experimental round')
            # Slot history must agree with published update counters.
            for optim, count in ((self.optimizer, self.updates), (self.critic_optimizer, self.critic_updates)):
                steps = [int(s['step'].item()) for s in optim.state.values()]
                if (count and (not steps or max(steps) != count)) or (not count and steps):
                    raise ValueError('optimizer history was reset or corrupted')
            self.initial_actor_id = saved['initial_actor_id']
            self.critic_origin = saved.get('critic_origin')
            if not artifacts.is_digest(self.initial_actor_id): raise ValueError('missing initial policy identity')
            self.reports, self.data = deepcopy(saved['reports']), deepcopy(saved['data'])
            if len(self.reports) != self.rounds or len(self.data) != self.rounds or not all(artifacts.is_digest(i) for i in self.data):
                raise ValueError('incomplete round history')
            self.rng.bit_generator.state = deepcopy(saved['shuffle'])
        if critic_init is not None:
            if saved is not None: raise ValueError('resumed critics cannot be replaced')
            fitted = critic_init.get('learning_critics')
            architecture = ('objective', 'rank_head', 'oracle', 'defence_weight', 'channels', 'blocks')
            if (not isinstance(fitted, dict) or type(fitted.get('version')) is not int or fitted['version'] != 1
                    or fitted.get('policy_id') != self.initial_actor_id
                    or fitted.get('engine') != artifacts.engine_identity()
                    or any(fitted['config'].get(k) != cfg.describe()[k] for k in architecture)
                    or type(fitted.get('epochs')) is not int or fitted['epochs'] < 1):
                raise ValueError('prefitted critic is not compatible with this policy and objective')
            self.critics.load_state_dict(fitted['state'], strict=True)
            self.critic_origin = artifacts.digest_state(critic_init)
        if bool(self.reference_payload) != bool(cfg.anchor_weight): raise ValueError('missing policy anchor')
        if self.reference_payload is not None:
            if artifacts.digest_state(self.reference_payload) != self.initial_actor_id:
                raise ValueError('policy anchor changed from the initial checkpoint')
            self.reference = artifacts.actor_from_payload(self.reference_payload, device).eval().requires_grad_(False)
        if len(self.members) != len(self.opponent_payloads): raise ValueError('missing frozen league members')
        # The artifacts contain weights_only-compatible tensor/primitive payloads.
        # Zoo supports additional fixed reference layouts; load its own snapshot.
        from .. import zoo
        with tempfile.TemporaryDirectory(prefix='mahjong-lab-opponents-') as temporary:
            for i, (state, member) in enumerate(zip(self.opponent_payloads, self.members)):
                if artifacts.digest_state(state) != member['sha256']: raise ValueError('opponent identity mismatch')
                path = Path(temporary) / f'{i}.pt'; atomic_save(state, path, keep_previous=False)
                self.actors.append(zoo.load_player(path, device).eval())
        if cfg.adaptive_league and not self.actors: raise ValueError('adaptive league needs at least one fixed opponent')
        if cfg.role == 'exploiter' and len(self.actors) != 1: raise ValueError('exploiter needs exactly one champion')
        if bool(cfg.lesson_weight) != bool(lessons is not None): raise ValueError('lesson data and positive lesson_weight must be supplied together')
        artifacts.finite(artifacts.actor_payload(self.actor)); artifacts.finite(self.critics.state_dict())
        if saved is not None:
            torch.set_rng_state(saved['torch_rng'].cpu())
            cuda = saved['cuda_rng']
            if torch.device(device).type == 'cuda':
                if cuda is None or len(cuda) != torch.cuda.device_count(): raise ValueError('CUDA RNG layout changed')
                torch.cuda.set_rng_state_all([v.cpu() for v in cuda])
            elif cuda is not None: raise ValueError('CPU checkpoint contains CUDA RNG state')
        else:
            torch.manual_seed(cfg.seed)

    def step(self, data_root: Path):
        if not self.ready: raise RuntimeError('Reload the last completed checkpoint after a failed round')
        self.ready = False
        cfg, device = self.cfg, self.device
        seed = cfg.seed + self.rounds * cfg.games
        if seed + cfg.games > 1_000_000: raise ValueError('training seed range exhausted; do not enter held-out ranges')
        round_data = collect(self.actor, cfg, seed, opponents=self.actors, league=self.league)
        identity = round_data.save(Path(data_root) / f'round-{self.rounds:06d}')
        a, n = round_data.arrays, len(round_data.observations)
        target = a['ranks'] @ np.asarray(UTILITY, np.float32)
        if cfg.objective == 'hybrid': target = target + a['hand_return']
        baseline, public_value = np.empty(n, np.float32), np.empty(n, np.float32)
        self.critics.eval()
        with torch.no_grad():
            for start in range(0, n, cfg.batch):
                rows = np.arange(start, min(n, start + cfg.batch))
                p = round_data.observations.rows(rows).dense(device)
                hidden = artifacts.tensor(a['oracle'][rows], device)
                baseline[rows] = self.critics.baseline(p, hidden).cpu().numpy()
                public_value[rows] = self.critics.public(p)['value'].cpu().numpy()
        # These predictions are frozen BEFORE either optimiser sees this round.
        advantages = target - baseline
        spread = float(advantages.std())
        advantages = (advantages - advantages.mean()) / (spread + 1e-6)
        warmup = self.rounds < cfg.warmup_rounds
        policy_steps = 0; largest_kl = 0.; stopped = False
        if not warmup:
            self.actor.eval()  # Keep collection-time BN/dropout semantics while differentiating.
            for _ in range(cfg.epochs):
                order = self.rng.permutation(n)
                for start in range(0, n, cfg.batch):
                    rows = order[start:start + cfg.batch]
                    p = round_data.observations.rows(rows).dense(device)
                    legal = artifacts.tensor(a['legal'][rows], device)
                    actions = artifacts.tensor(a['action'][rows], device)
                    before = artifacts.tensor(a['log_prob'][rows], device)
                    logits = _policy(self.actor, p, legal, cfg)
                    distribution = torch.distributions.Categorical(logits=logits)
                    now = distribution.log_prob(actions)
                    delta = now - before
                    kl = float((torch.expm1(delta.detach()) - delta.detach()).mean())
                    if not np.isfinite(kl): raise FloatingPointError('nonfinite PPO drift')
                    largest_kl = max(largest_kl, kl)
                    if kl > cfg.target_kl:
                        stopped = True; break
                    ratio = delta.exp()
                    advantage = artifacts.tensor(advantages[rows], device)
                    loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage).mean()
                    loss = loss - cfg.entropy * distribution.entropy().mean()
                    if self.reference is not None:
                        with torch.no_grad(): old = _policy(self.reference, p, legal, cfg).softmax(1)
                        loss = loss + cfg.anchor_weight * masked_kl(old, logits, legal)
                    if self.lessons is not None:
                        indices = self.rng.integers(0, len(self.lessons), size=min(cfg.batch, len(self.lessons)))
                        lp, ll, lq = self.lessons.batch(indices, device)
                        logp = _policy(self.actor, lp, ll, cfg).log_softmax(1).masked_fill(~ll, 0.)
                        loss = loss + cfg.lesson_weight * -(lq * logp).sum(1).mean()
                    if not torch.isfinite(loss): raise FloatingPointError('nonfinite experimental policy loss')
                    self.optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                    self.optimizer.step(); self.updates += 1; policy_steps += 1
                if stopped: break
            if not policy_steps: raise ValueError('No policy update accepted; inspect inference precision/drift')
        self.critics.train()
        last_metrics = {}
        for _ in range(cfg.critic_epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, cfg.batch):
                rows = order[start:start + cfg.batch]
                p = round_data.observations.rows(rows).dense(device)
                get = lambda k: artifacts.tensor(a[k][rows], device)
                loss, metrics = self.critics.losses(p, get('oracle'), get('ranks'), get('hand_return'), get('defence'))
                self.critic_optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.critic_optimizer.step(); self.critic_updates += 1
                last_metrics = {k: float(v.detach()) for k, v in metrics.items()}
        self.league.update(round_data.trace['actor_id'], a['seated'], a['learner_placement'])
        self.rounds += 1
        if not warmup: self.actor_rounds += 1
        report = dict(round=self.rounds, actor_round=self.actor_rounds, critic_only=warmup,
                      decisions=n, seed=seed, objective=cfg.objective, oracle=cfg.oracle,
                      rank_head=cfg.rank_head, policy_updates=policy_steps, stopped_on_kl=stopped,
                      sampled_kl=largest_kl, frozen_baseline_mse=float(np.mean((target - baseline) ** 2)),
                      frozen_public_mse=float(np.mean((target - public_value) ** 2)), advantage_spread=spread,
                      fit_last_batch=last_metrics, roots=len(round_data.trace['roots']),
                      development_placement=float(a['learner_placement'].mean()), league_weights=self.league.weights().tolist())
        artifacts.finite(artifacts.actor_payload(self.actor)); artifacts.finite(self.critics.state_dict())
        artifacts.finite(self.optimizer.state_dict()); artifacts.finite(self.critic_optimizer.state_dict())
        artifacts.canonical(report)
        self.reports.append(report); self.data.append(identity); self.ready = True
        return report

    def checkpoint(self):
        if not self.ready or not self.rounds: raise RuntimeError('Only completed rounds can be published')
        saved = dict(version=1, config=self.cfg.describe(), runtime=self.runtime, source=self.source,
                     signature=self.signature, critics=self.critics.state_dict(), optimizer=self.optimizer.state_dict(),
                     active_optimizer=sorted(self.optimizer.state_dict()['state']),
                     active_critic_optimizer=sorted(self.critic_optimizer.state_dict()['state']), critic_origin=self.critic_origin,
                     critic_optimizer=self.critic_optimizer.state_dict(), reference=self.reference_payload,
                     opponents=self.opponent_payloads, members=self.members, league=self.league.state(),
                     rounds=self.rounds, actor_rounds=self.actor_rounds, updates=self.updates, critic_updates=self.critic_updates,
                     initial_generation=self.initial_generation, initial_actor_id=self.initial_actor_id,
                     shuffle=deepcopy(self.rng.bit_generator.state), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.device(self.device).type == 'cuda' else None,
                     lesson_id=self.lesson_id, reports=self.reports, data=self.data)
        result = dict(**artifacts.actor_payload(self.actor), learner='learning_lab', learning_lab=saved,
                      generation=self.initial_generation + self.actor_rounds, training_api_version=2)
        artifacts.finite(result)
        return _cpu(result)


def run(source: Path, out: Path, *, cfg=None, rounds=1, resume=False, device='cpu',
        opponents=(), roles=(), lessons=None, critic_init=None):
    if type(rounds) is not int or rounds < 1: raise ValueError('rounds must be positive')
    out = Path(out)
    if out.exists(): raise FileExistsError('Use a new experiment output directory')
    with tempfile.TemporaryDirectory(prefix='mahjong-learning-lab-') as folder:
        snapshot = Path(folder) / 'actor.pt'; copy_checkpoint(source, snapshot)
        payload = torch.load(snapshot, map_location='cpu', weights_only=True)
        saved = payload.get('learning_lab') if resume else None
        if resume and (payload.get('learner') != 'learning_lab' or not isinstance(saved, dict)):
            raise ValueError('Exact resume needs a learning-lab checkpoint')
        cfg = cfg or (Config(**saved['config']) if saved else Config())
        cfg.validate()
        if saved and payload.get('generation') != saved['initial_generation'] + saved['actor_rounds']:
            raise ValueError('outer checkpoint generation disagrees with training state')
        if resume and opponents: raise ValueError('resumed league is inherited, not replaced')
        if roles and len(roles) != len(opponents): raise ValueError('one role per opponent required')
        states, members = [], []
        for i, path in enumerate(opponents):
            copied = Path(folder) / f'opponent-{i}.pt'; copy_checkpoint(path, copied)
            state = torch.load(copied, map_location='cpu', weights_only=True)
            states.append(state); members.append(dict(sha256=artifacts.digest_state(state), name=Path(path).stem,
                                                      role=roles[i] if roles else 'reference'))
        from .reanalysis import Lessons
        data = Lessons.load(lessons) if lessons is not None else None
        # Constructors are seeded identically; a resume restores RNG after them.
        initial = None
        if critic_init is not None:
            copied = Path(folder) / 'critics.pt'; copy_checkpoint(critic_init, copied)
            initial = torch.load(copied, map_location='cpu', weights_only=True)
        torch.manual_seed(cfg.seed)
        learner = Learner(payload, cfg, device=device, saved=saved, opponent_payloads=states, members=members, lessons=data, critic_init=initial)
        out.mkdir(parents=True, exist_ok=False)
        until = learner.actor_rounds + rounds
        while learner.actor_rounds < until:
            report = learner.step(out / 'data')
            atomic_save(learner.checkpoint(), out / 'latest.pt')
            print(json.dumps(report, allow_nan=False), flush=True)
    return out / 'latest.pt'
