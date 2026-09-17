"""Completed experimental on-policy games and replayable public decision roots.

Reuse the existing two-stage action recorder and reward ledger. The experimental
format keeps player identities, full rank targets, private training labels and
raw action traces separate. Neither private tensor is passed to the actor.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .. import ledger, zoo
from ..mortal_learner import decide_in_mortal_space
from ..observe import Planes, Views
from ..outcomes import placements, require_finished, placement_rewards
from . import artifacts
from .config import Config
from .critics import DEFENCE_WIDTH, UTILITY, rank_targets, validate_defence
from .league import League, select_roots


@dataclass
class Round:
    observations: Planes
    arrays: dict
    trace: dict

    def save(self, path):
        self.validate()
        return artifacts.save(path, 'learning-round', self.trace,
                              {**self.arrays, **{'obs_' + k: v for k, v in self.observations.arrays().items()}})

    @classmethod
    def load(cls, path):
        meta, a, _ = artifacts.load(path, 'learning-round')
        planes = Planes(a.pop('obs_indptr'), a.pop('obs_indices'), a.pop('obs_values'))
        result = cls(planes, a, meta); result.validate(); return result

    def validate(self):
        a, m, n = self.arrays, self.trace, len(self.observations)
        self.observations.validate()
        expected = {'legal', 'action', 'log_prob', 'game', 'player', 'stage', 'return', 'hand_return',
                    'ranks', 'oracle', 'defence', 'trajectory', 'final_scores', 'seated', 'learner_placement'}
        if set(a) != expected or type(m.get('format')) is not int or m.get('format') != 1 or not n or m.get('reward') != 'hybrid_components_v1':
            raise ValueError('invalid learning round contract')
        g = m['games']
        if type(g) is not int or g < 1 or type(m['seed']) is not int or not 0 <= m['seed'] < 2**64 - g:
            raise ValueError('invalid recorded game seeds')
        for name in ('game', 'player', 'stage', 'action'):
            if a[name].shape != (n,) or a[name].dtype != np.int64: raise ValueError(f'invalid {name}')
        for name in ('return', 'hand_return', 'log_prob'):
            if a[name].shape != (n,) or a[name].dtype != np.float32 or not np.isfinite(a[name]).all(): raise ValueError(f'invalid {name}')
        if (a['legal'].shape != (n, 46) or a['legal'].dtype != np.bool_
                or np.any((a['action'] < 0) | (a['action'] >= 46)) or not a['legal'][np.arange(n), a['action']].all()
                or np.any((a['stage'] < 0) | (a['stage'] > 1))
                or a['final_scores'].shape != (g, 4) or a['final_scores'].dtype != np.int32):
            raise ValueError('invalid action/stage/outcome data')
        ranks = rank_targets(a['final_scores'], a['game'], a['player'])
        if a['ranks'].dtype != np.float32 or not np.array_equal(ranks, a['ranks']):
            raise ValueError('rank labels disagree with actual tied finishes')
        if not np.allclose(a['return'], a['hand_return'] + ranks @ np.asarray(UTILITY), atol=2e-6):
            raise ValueError('hybrid components disagree; never relabel old returns')
        if (a['trajectory'].dtype != np.int64 or a['trajectory'].ndim != 2 or a['trajectory'].shape[1] != g
                or np.any((a['trajectory'] < 0) | (a['trajectory'] >= 78))):
            raise ValueError('invalid exact action trace')
        import riichi_py
        if m.get('oracle') not in ('none', 'hands', 'full') or type(m.get('defence_labels')) is not bool:
            raise ValueError('missing private-label collection contract')
        expected_hidden = {'none': 0, 'hands': riichi_py.HIDDEN_HANDS_PLANES, 'full': riichi_py.ORACLE_PLANES}[m['oracle']]
        if a['oracle'].shape != (n, expected_hidden, 34) or a['oracle'].dtype != np.uint8 or np.any(a['oracle'] > 1):
            raise ValueError('invalid privileged observation data')
        if a['defence'].shape != (n, 3, DEFENCE_WIDTH): raise ValueError('invalid defensive shape')
        validate_defence(a['defence'])
        if not m['defence_labels'] and np.any(a['defence']): raise ValueError('unrecorded defence labels cannot appear')
        if (a['seated'].shape != (g,) or a['seated'].dtype != np.int64
                or np.any((a['seated'] < -1) | (a['seated'] >= len(m['opponents'])))
                or a['learner_placement'].shape != (g,) or a['learner_placement'].dtype != np.float64
                or not np.isfinite(a['learner_placement']).all() or np.any((a['learner_placement'] < 1) | (a['learner_placement'] > 4))):
            raise ValueError('invalid development outcomes')
        if not artifacts.is_digest(m.get('actor_id')) or not isinstance(m.get('roots'), list):
            raise ValueError('missing root/policy identity')
        seen = set()
        for r in m['roots']:
            if (any(type(r.get(k)) is not int for k in ('step', 'game', 'player'))
                    or not 0 <= r['step'] < len(a['trajectory']) or not 0 <= r['game'] < g
                    or not 0 <= r['player'] < 4 or not artifacts.is_digest(r.get('public_sha256'))
                    or not np.isfinite(r.get('priority', np.nan)) or r['priority'] < 0
                    or (r['step'], r['game']) in seen):
                raise ValueError('invalid retained root')
            seen.add((r['step'], r['game']))
        return self


def root_digest(planes, legal):
    h = hashlib.sha256()
    for a in planes.arrays().values(): h.update(a.tobytes())
    h.update(np.asarray(legal, dtype=np.bool_).tobytes())
    return h.hexdigest()


@torch.no_grad()
def collect(net, cfg: Config, seed: int, *, opponents=(), league=None, peer=None) -> Round:
    import riichi_py
    from ..training_safety import require_training_engine
    from ..teacher_actions import representable_moves
    cfg.validate(); require_training_engine()
    if ledger.REWARD_VERSION != 1 or ledger.HAND_SCALE != 1 / 4000 or tuple(ledger.PLACEMENT_VALUE) != UTILITY:
        raise ValueError('unsupported reward components; do not relabel observations')
    if cfg.defence_weight and getattr(riichi_py, 'LEARNING_LABEL_API_VERSION', None) != 1:
        raise ValueError('Rebuild riichi-py for learning label API 1 before defensive training')
    if net.actions != 46 or net.planes != 1012: raise ValueError('unsupported actor')
    if type(seed) is not int or not 0 <= seed < 2**64 - cfg.games: raise ValueError('invalid seed')
    net.eval()
    device = str(next(net.parameters()).device)
    league = league or League([])
    if len(opponents) != len(league.members): raise ValueError('league and loaded opponents differ')
    if cfg.role == 'exploiter' and len(opponents) != 1: raise ValueError('exploiter needs exactly one frozen champion')
    for other in opponents: other.eval()
    if peer is not None: peer.eval()
    picker = np.random.default_rng(seed ^ 0x0DDBA11)
    seated = league.seat(cfg.games, 1. if cfg.role == 'exploiter' else cfg.opponent_share, picker)
    foreign = np.full((cfg.games, 4), -1, np.int64)
    for game, who in enumerate(seated):
        if who < 0: continue
        person = int(picker.integers(4))
        if cfg.role == 'exploiter':
            foreign[game] = who; foreign[game, person] = -1
        else: foreign[game, person] = who
    arena = riichi_py.Arena(games=cfg.games, seed=seed, bot_places=[])
    views = Views(arena, cfg.games, {'mortal'} | {o.kind for o in opponents})
    account = ledger.Ledger(cfg.games)
    blocks, trajectory, roots = [], [], []
    parts = {k: [] for k in ('legal', 'action', 'log_prob', 'game', 'player', 'stage', 'oracle', 'defence')}
    actor_id = artifacts.digest_state(artifacts.actor_payload(net))
    for step in range(cfg.max_steps):
        if arena.all_finished(): break
        seats = np.frombuffer(arena.seats(), np.uint8)
        live = np.flatnonzero(seats != 255).astype(np.int64)
        rotation = np.frombuffer(arena.seat_players(), np.uint8).reshape(cfg.games, 4)
        person = np.zeros(cfg.games, np.int64); person[live] = rotation[live, seats[live]]
        legal = np.frombuffer(arena.legal_mask(), np.uint8).reshape(cfg.games, 78).astype(bool)
        views.advance(); views.prepare(live, person[live])
        actions = np.zeros(cfg.games, np.int64)
        own = live[foreign[live, person[live]] < 0]
        for i, opponent in enumerate(opponents):
            rows = live[foreign[live, person[live]] == i]
            if len(rows): actions[rows] = zoo.choose(opponent, views, rows, person[rows], legal[rows], device, greedy=True)
        hidden = (np.frombuffer(arena.oracle(), np.float32).reshape(cfg.games, riichi_py.ORACLE_PLANES, 34).astype(np.uint8)
                  if cfg.oracle != 'none' else np.zeros((cfg.games, 0, 34), np.uint8))
        if cfg.oracle == 'hands': hidden = hidden[:, :riichi_py.HIDDEN_HANDS_PLANES]
        defence = (np.frombuffer(arena.learning_defence(), np.float32).reshape(cfg.games, 3, DEFENCE_WIDTH)
                   if cfg.defence_weight else np.zeros((cfg.games, 3, DEFENCE_WIDTH), np.float32))
        # Public engine context: last 17 planes include four round winds;
        # the final four encode hand number. This selector never reads labels.
        public = np.frombuffer(arena.observations(), np.float32).reshape(cfg.games, riichi_py.PLANES, 34)
        late = (public[:, -16, 0] > 0) & (public[:, -2:, 0].sum(1) > 0)
        for start in range(0, len(own), cfg.batch):
            rows = own[start:start + cfg.batch]
            if not zoo.translatable(legal[rows]).any(1).all(): raise ValueError('unrepresentable on-policy decision')
            seen = []
            def score(planes, mask):
                fast = getattr(net, 'policy_only', None)
                logits = fast(planes, mask) if callable(fast) else net(planes, mask)[0]
                disagreement = np.zeros(len(planes), bool)
                if not seen and peer is not None:
                    from ..policy_inference import policy_logits
                    other = policy_logits(peer, planes, mask)
                    disagreement = (other.argmax(1) != logits.argmax(1)).cpu().numpy()
                seen.append((logits.float().softmax(1).cpu().numpy(), disagreement))
                return logits
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=cfg.amp and device.startswith('cuda')):
                picked, records = decide_in_mortal_space(score, views, rows, person[rows], legal[rows], device=device)
            actions[rows] = picked
            n, slots = len(records.actions), records.slots
            stage = np.r_[np.zeros(len(rows), np.int64), np.ones(n - len(rows), np.int64)]
            games, people = rows[slots], person[rows[slots]]
            blocks.append(records.planes)
            vals = dict(legal=records.masks.copy(), action=records.actions.copy(), log_prob=records.log_probs.copy(),
                        game=games.copy(), player=people.copy(), stage=stage, oracle=hidden[games].copy(), defence=defence[games].copy())
            for k, v in vals.items(): parts[k].append(v)
            for game, player in zip(games, people): account.open(int(game), int(player))
            probs, disagree = seen[0]
            searchable = representable_moves(legal[rows]).sum(1) >= 2
            for i, game in enumerate(rows):
                if not searchable[i]: continue
                p = probs[i]; entropy = -float((p * np.log(np.maximum(p, 1e-30))).sum()) / np.log(46)
                key_decision = bool(legal[game, 34:68].any() or (legal[game, 70] and legal[game, 71:75].any()))
                roots.append(dict(step=step, game=int(game), player=int(person[game]),
                                  public_sha256=root_digest(records.planes.rows(np.array([i])), legal[game]),
                                  priority=entropy + .5 * key_decision + .5 * bool(late[game]) + float(disagree[i])))
        trajectory.append(actions.copy())
        arena.step(actions.tolist()); account.settle(arena)
    require_finished(arena, steps=len(trajectory), context='learning lab collection')
    returns = account.close(arena)
    if not len(returns): raise ValueError('no learner decisions')
    a = {k: np.concatenate(v) for k, v in parts.items()}
    final_scores = np.frombuffer(arena.final_scores(), np.int32).reshape(cfg.games, 4).copy()
    ranks = rank_targets(final_scores, a['game'], a['player'])
    places = placements(final_scores)
    learner_place = np.array([places[g, foreign[g] < 0].mean() for g in range(cfg.games)], np.float64)
    a.update({'return': returns, 'hand_return': (returns - ranks @ np.asarray(UTILITY, np.float32)).astype(np.float32),
              'ranks': ranks, 'trajectory': np.stack(trajectory), 'final_scores': final_scores,
              'seated': seated, 'learner_placement': learner_place})
    kept = select_roots(roots, cfg.roots, cfg.uniform_roots, np.random.default_rng(seed ^ 0xC011EC7))
    trace = dict(format=1, games=cfg.games, seed=seed, actor_id=actor_id, engine=artifacts.engine_identity(),
                 reward='hybrid_components_v1', collection_config=cfg.describe(), oracle=cfg.oracle, defence_labels=bool(cfg.defence_weight),
                 opponents=league.members, role=cfg.role, roots=kept, eligible_roots=len(roots),
                 selector=dict(uniform_share=cfg.uniform_roots, cap=cfg.roots, rule='uniform_quota_plus_public_priority_v1'),
                 precision='bfloat16' if cfg.amp and device.startswith('cuda') else 'float32')
    return Round(Planes.cat(blocks), a, trace).validate()
