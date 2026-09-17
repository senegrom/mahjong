"""Fresh, immutable teacher labels on selected roots from completed games.

Reconstruct the recorded game with its seed and actual engine-action history,
verify the player's public state, then ask the new teacher. Never change the
recorded trajectory, reuse old labels as new evidence, or invent PPO likelihoods.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
import torch

from .. import contract, searched
from ..collect_search import _Inputs, _sparse
from ..observe import Planes, Views
from ..checkpoints import copy_checkpoint
from . import artifacts, targets
from .collection import Round, root_digest
from .config import Config
from .critics import Critics


@dataclass
class Lessons:
    observations: Planes
    arrays: dict
    meta: dict
    identity: str = ''

    def __len__(self): return len(self.observations)

    def validate(self):
        a, n = self.arrays, len(self)
        self.observations.validate()
        expected = {'legal', 'actor', 'target', 'stage', 'root', 'root_legal', 'advantage', 'stderr', 'tested',
                    'worlds', 'games', 'confirmed', 'root_indptr', 'root_indices', 'root_values'}
        if set(a) != expected or not n or self.meta.get('label_kind') != 'counterfactual_policy_only_v1':
            raise ValueError('not a supported supervised lesson artifact')
        if a['legal'].shape != (n, 46) or a['legal'].dtype != np.bool_ or not a['legal'].any(1).all():
            raise ValueError('invalid lesson masks')
        for field in ('actor', 'target'):
            p = a[field]
            if (p.shape != (n, 46) or p.dtype != np.float32 or not np.isfinite(p).all()
                    or np.any(p < 0) or np.any(p[~a['legal']] != 0) or not np.allclose(p.sum(1), 1., atol=1e-5)):
                raise ValueError('invalid legal lesson distribution')
        roots = self.roots(); r = len(roots); roots.validate()
        if a['root_legal'].shape != (r, 78) or a['root_legal'].dtype != np.bool_ or a['tested'].shape != (r, 78) or a['tested'].dtype != np.bool_:
            raise ValueError('invalid candidate support')
        if np.any(a['tested'] & ~a['root_legal']): raise ValueError('illegal candidate labelled')
        for name in ('advantage', 'stderr'):
            if a[name].shape != (r, 78) or a[name].dtype != np.float32 or not np.isfinite(a[name]).all():
                raise ValueError('invalid ranking evidence')
        if (a['worlds'].shape != (r, 78) or a['worlds'].dtype != np.int64 or np.any(a['worlds'][a['tested']] < 3)
                or np.any(a['stderr'] < 0) or a['games'].shape != (r,) or a['games'].dtype != np.uint64
                or a['confirmed'].shape != (r,) or a['confirmed'].dtype != np.bool_):
            raise ValueError('invalid evidence counts or identities')
        if (a['root'].shape != (n,) or a['root'].dtype != np.int64 or np.any((a['root'] < 0) | (a['root'] >= r))
                or a['stage'].shape != (n,) or a['stage'].dtype != np.int64 or not np.isin(a['stage'], [0, 1]).all()):
            raise ValueError('invalid hierarchical lesson linkage')
        diagnostics = self.meta.get('diagnostics')
        if not isinstance(diagnostics, list) or len(diagnostics) != r:
            raise ValueError('missing per-root teacher evidence')
        if np.any((a['actor'] == 0) & (a['target'] > 0)):
            raise ValueError('lesson silently expands policy support')
        for i, item in enumerate(diagnostics):
            linked = np.flatnonzero(a['root'] == i)
            if len(linked) not in (1, 2) or a['stage'][linked].tolist() != list(range(len(linked))):
                raise ValueError('each root needs one ordinary and at most one conditional lesson')
            actions = item.get('actions', [])
            if (len(actions) < 2 or len(set(actions)) != len(actions)
                    or any(type(v) is not int or not 0 <= v < 78 or not a['root_legal'][i, v] for v in actions)
                    or item.get('incumbent') != actions[0] or item.get('challenger') not in actions
                    or type(item.get('confirmed')) is not bool or item['confirmed'] != bool(a['confirmed'][i])):
                raise ValueError('invalid diagnostic candidate linkage')
            if not a['confirmed'][i] and not np.allclose(a['target'][linked], a['actor'][linked], atol=1e-7):
                raise ValueError('unconfirmed comparisons cannot sharpen the policy')
            if np.any(a['tested'][i] & ~np.isin(np.arange(78), actions)):
                raise ValueError('untested alternatives cannot acquire audit labels')
        return self

    def roots(self):
        return Planes(self.arrays['root_indptr'], self.arrays['root_indices'], self.arrays['root_values'])

    def batch(self, indices, device='cpu'):
        return (self.observations.rows(indices).dense(device), artifacts.tensor(self.arrays['legal'][indices], device),
                artifacts.tensor(self.arrays['target'][indices], device))

    def save(self, path):
        self.validate()
        self.identity = artifacts.save(path, 'learning-lessons', self.meta,
                                        {**self.arrays, **{'obs_' + k: v for k, v in self.observations.arrays().items()}})
        return self.identity

    @classmethod
    def load(cls, path):
        meta, a, identity = artifacts.load(path, 'learning-lessons')
        p = Planes(a.pop('obs_indptr'), a.pop('obs_indices'), a.pop('obs_values'))
        return cls(p, a, meta, identity).validate()


class PlacementAdapter:
    """Public-only placement expectation, regardless of PPO reward ablation."""
    def __init__(self, critics): self.critics = critics.eval()
    def judge(self, net, planes): return self.critics.public(planes)['placement']


def _teacher(path, device, placement_head):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    artifacts.finite(payload)
    net = artifacts.actor_from_payload(payload, device).eval()
    head = None; head_identity = None
    if placement_head is not None:
        fitted = torch.load(placement_head, map_location='cpu', weights_only=True).get('learning_critics')
        if fitted is not None:
            if (not isinstance(fitted, dict) or type(fitted.get('version')) is not int or fitted['version'] != 1
                    or fitted.get('policy_id') != artifacts.digest_state(artifacts.actor_payload(net))
                    or fitted.get('engine') != artifacts.engine_identity()
                    or type(fitted.get('epochs')) is not int or fitted['epochs'] < 1):
                raise ValueError('placement sidecar does not belong to this policy/engine')
            critics = Critics(Config(**fitted['config'])).to(device)
            critics.load_state_dict(fitted['state'], strict=True); artifacts.finite(critics.state_dict())
            head = PlacementAdapter(critics)
        else:
            from .. import placement
            head, meta = placement.load(placement_head, device)
            placement.require_head_for(meta, net); head.eval()
        head_identity = artifacts.digest_file(placement_head)
    elif 'learning_lab' in payload:
        saved = payload['learning_lab']
        if saved.get('version') != 1 or not saved.get('rounds'):
            raise ValueError('untrained or unsupported experimental critic')
        critics = Critics(Config(**saved['config'])).to(device)
        critics.load_state_dict(saved['critics'], strict=True)
        head = PlacementAdapter(critics)
        head_identity = artifacts.digest_state(saved['critics'])
    if head is None:
        raise ValueError('Supply a learning-lab checkpoint or an explicitly fitted placement head')
    return net, contract.serve(net, placement_head=head), head_identity


@torch.no_grad()
def run(trace_path, teacher_path, out, *, placement_head=None, worlds=8, candidates=4,
        confirm_worlds=16, audit_worlds=8, margin=2., eta=1., race_budget=0,
        candidate_method='top', seed=81231, batch=256, device='cpu'):
    if Path(out).exists(): raise FileExistsError('Write reanalysis into a NEW directory')
    if type(candidates) is not int or not 2 <= candidates <= 78 or type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('invalid reanalysis controls')
    if type(batch) is not int or batch < 1 or not np.isfinite(eta) or eta < 0 or candidate_method not in ('top', 'gumbel'):
        raise ValueError('invalid reanalysis batch/target/candidate controls')
    # Validate the complete stage budget without starting expensive simulation.
    targets.teach_root(lambda actions, n: {a: dict(edge=0., error=1., worlds=n) for a in actions},
                       list(range(candidates)), worlds=worlds, confirm_worlds=confirm_worlds,
                       audit_worlds=audit_worlds, margin=margin, race_budget=race_budget)
    data = Round.load(trace_path)
    trace_meta, _, trace_id = artifacts.load(trace_path, 'learning-round')
    if data.trace['engine'] != artifacts.engine_identity():
        raise ValueError('Recorded trace uses different native/follower semantics; recollect it')
    if not data.trace['roots']: raise ValueError('trace retained no decision roots')
    with tempfile.TemporaryDirectory(prefix='mahjong-reanalysis-') as temporary:
        checkpoint = Path(temporary) / 'teacher.pt'; copy_checkpoint(teacher_path, checkpoint)
        head_path = None
        if placement_head is not None:
            head_path = Path(temporary) / 'placement.pt'; copy_checkpoint(placement_head, head_path)
        net, served, head_id = _teacher(checkpoint, device, head_path)
        teacher_id = artifacts.digest_state(artifacts.actor_payload(net))
        import riichi_py
        arena = riichi_py.Arena(games=data.trace['games'], seed=data.trace['seed'], bot_places=[])
        if getattr(riichi_py, 'LEARNING_LABEL_API_VERSION', None) != 1:
            raise ValueError('Rebuild riichi-py for independently seeded reanalysis')
        arena.learning_seed_search(seed)
        views = Views(arena, data.trace['games'], {'mortal'})
        by_step = {}
        for root in data.trace['roots']: by_step.setdefault(root['step'], []).append(root)
        rng = np.random.default_rng(seed)
        blocks, root_blocks, policy, masks, desired, stages, links = [], [], [], [], [], [], []
        root_legal, edges, errors, support, counts, ids, confirmed, diagnostics = [], [], [], [], [], [], [], []
        for step, actual in enumerate(data.arrays['trajectory']):
            seats = np.frombuffer(arena.seats(), np.uint8)
            live = np.flatnonzero(seats != 255).astype(np.int64)
            people = np.frombuffer(arena.seat_players(), np.uint8).reshape(data.trace['games'], 4)
            deciding = np.zeros(data.trace['games'], np.int64); deciding[live] = people[live, seats[live]]
            legal = np.frombuffer(arena.legal_mask(), np.uint8).reshape(data.trace['games'], 78).astype(bool)
            views.advance(); views.prepare(live, deciding[live])
            for root in by_step.get(step, []):
                game = root['game']; rows = np.array([game], np.int64); who = deciding[rows]
                observed = views.sparse(rows, who)
                if game not in live or int(who[0]) != root['player'] or root_digest(observed, legal[game]) != root['public_sha256']:
                    raise ValueError('Reconstructed root differs from the recorded public information')
                captured = _Inputs(served)
                ordered, logits, _, guessed = contract.root_order(captured, arena, views, rows, who, legal, device)
                planes, mask = captured.root_inputs
                pi = logits.float().softmax(1)[0].cpu().numpy()
                after_pi = None; after_planes = after_mask = None
                if captured.reach_inputs is not None:
                    after_planes, after_mask = captured.reach_inputs
                    from ..policy_inference import everything
                    after_pi = everything(net, after_planes, after_mask)[0].float().softmax(1)[0].cpu().numpy()
                joint = targets.joint_policy(pi, after_pi, legal[game])
                actions = targets.candidates(ordered[game], legal[game], joint, candidates, rng, candidate_method)
                if len(actions) < 2: continue
                beliefs = np.zeros((data.trace['games'], riichi_py.HANDS), np.float32)
                beliefs[game] = guessed.float().softmax(2).reshape(-1).cpu().numpy()
                def evaluate(choices, count):
                    ranked = [[int(np.flatnonzero(l)[0])] if l.any() else [0] for l in legal]
                    ranked[game] = choices
                    with contract.following(arena, views.observer.follower):
                        searched._search_once(net, arena, ranked, beliefs.reshape(-1).tolist(),
                            worlds=count, candidates=len(choices), margin=0., hurried=False, device=device,
                            pool=1, played_by='network', depth=-1, temperature=0., valued_by='placement',
                            served=served, leaf_batch=batch, objective='placement', search_calls=True, rollout_batch=batch)
                    return targets.evidence(arena.judgements()[game], arena.judgement_weights()[game])
                result = targets.teach_root(evaluate, actions, worlds=worlds, confirm_worlds=confirm_worlds,
                                           audit_worlds=audit_worlds, margin=margin, race_budget=race_budget)
                q_joint = targets.confirmed_tilt(joint, result['incumbent'], result['challenger'], result['lower_edge'], eta)
                if not result['confirmed'] or eta == 0:
                    q, after_q = pi.copy(), None if after_pi is None else after_pi.copy()
                else:
                    q, after_q = targets.lift_policy(q_joint, pi, after_pi, legal[game])
                r = len(root_blocks); root_blocks.append(observed); root_legal.append(legal[game].copy())
                blocks.append(_sparse(planes)); policy.append(pi); masks.append(mask[0].cpu().numpy()); desired.append(q); stages.append(0); links.append(r)
                if after_q is not None:
                    blocks.append(_sparse(after_planes)); policy.append(after_pi); masks.append(after_mask[0].cpu().numpy()); desired.append(after_q); stages.append(1); links.append(r)
                e, s = np.zeros(78, np.float32), np.zeros(78, np.float32)
                tested, n = np.zeros(78, bool), np.zeros(78, np.int64)
                for action, row in result['audit'].items():
                    if np.isfinite(row['error']) and row['worlds'] >= 3:
                        e[action], s[action], n[action], tested[action] = row['edge'], row['error'], row['worlds'], True
                edges.append(e); errors.append(s); counts.append(n); support.append(tested)
                ids.append(data.trace['seed'] + game); confirmed.append(result['confirmed'])
                diagnostics.append(dict(step=step, game=game, actions=actions, incumbent=result['incumbent'],
                                        challenger=result['challenger'], confirmed=result['confirmed'],
                                        lower_edge=result['lower_edge'], discovery_slots=result['discovery_slots']))
            arena.step(actual.tolist())
        if not arena.all_finished(): raise ValueError('Trace did not replay to completion')
        final = np.frombuffer(arena.final_scores(), np.int32).reshape(data.trace['games'], 4)
        if not np.array_equal(final, data.arrays['final_scores']): raise ValueError('Trace final outcomes changed')
        if not blocks: raise ValueError('No retained root had two representable actions')
        roots = Planes.cat(root_blocks)
        a = dict(legal=np.stack(masks), actor=np.stack(policy).astype(np.float32), target=np.stack(desired),
                 stage=np.array(stages, np.int64), root=np.array(links, np.int64), root_legal=np.stack(root_legal),
                 advantage=np.stack(edges), stderr=np.stack(errors), tested=np.stack(support), worlds=np.stack(counts),
                 games=np.array(ids, np.uint64), confirmed=np.array(confirmed, bool),
                 **{'root_' + k: v for k, v in roots.arrays().items()})
        meta = dict(label_kind='counterfactual_policy_only_v1', trace_id=trace_id, teacher_id=teacher_id,
                    placement_head_id=head_id, engine=artifacts.engine_identity(),
                    settings=dict(worlds=worlds, candidates=candidates, confirm_worlds=confirm_worlds,
                                  audit_worlds=audit_worlds, margin=margin, eta=eta, race_budget=race_budget,
                                  candidate_method=candidate_method, seed=seed, batch=batch, device=device),
                    objective='placement', diagnostics=diagnostics,
                    note='fresh confirmation and separate audit; standard-error margins are not formal confidence bounds')
        lesson = Lessons(Planes.cat(blocks), a, meta).validate(); lesson.save(out)
        return lesson
