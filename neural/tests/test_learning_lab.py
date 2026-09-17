"""Learning-signal experiments: masks, information boundaries, evidence and resume."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import combined, model, mortal_model
from neural.checkpoints import atomic_save
from neural.collect_search import _sparse
from neural.learning_lab import artifacts, targets
from neural.learning_lab.config import Config
from neural.learning_lab.critics import Critics, SignalCritic, UTILITY, rank_targets, validate_defence
from neural.learning_lab.league import League, select_roots
from neural.learning_lab.fit import split
from neural.learning_lab.ranker import Ranker
from neural.learning_lab.train import Learner, _restore_optimizer, masked_kl


def config(**kw):
    base = dict(games=1, batch=32, epochs=1, critic_epochs=1, warmup_rounds=0,
                channels=8, blocks=1, roots=3, seed=421)
    base.update(kw)
    return Config(**base).validate()


def actor(joined=False):
    net = model.PolicyValueNet(8, 1, actions=46)
    if joined:
        net = combined.Combined(net, mortal_model.build(16, 1))
        net.mortal_config = {'resnet': {'conv_channels': 16, 'num_blocks': 1}, 'control': {'version': 4}}
    return net.eval()


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)
    def equal_tree(self, left, right):
        if isinstance(left, torch.Tensor): self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for k in left: self.equal_tree(left[k], right[k])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.equal_tree(a, b)
        else: self.assertEqual(left, right)


class SignalTests(Base):
    def test_rank_targets_share_exact_occupied_ranks_not_rounded_average(self):
        scores = np.array([[40, 30, 30, 0], [0, 0, 0, 0]], np.int32)
        games = np.repeat(np.arange(2, dtype=np.int64), 4)
        people = np.tile(np.arange(4, dtype=np.int64), 2)
        result = rank_targets(scores, games, people)
        np.testing.assert_array_equal(result[1], [0, .5, .5, 0])
        np.testing.assert_array_equal(result[4:], np.full((4, 4), .25))
        np.testing.assert_array_equal(result @ np.array(UTILITY), [1.5, 0, 0, -1.5, 0, 0, 0, 0])
        with self.assertRaises(ValueError): rank_targets(scores, games.astype(float), people)

    def test_categorical_probabilities_normalize_and_hybrid_keeps_separate_hand_term(self):
        public = torch.randn(2, 1012, 34)
        for objective in ('hybrid', 'placement'):
            net = SignalCritic(config(rank_head='categorical', objective=objective))
            answer = net(public)
            torch.testing.assert_close(answer['probabilities'].sum(1), torch.ones(2))
            self.assertEqual(net.hand is not None, objective == 'hybrid')
            torch.testing.assert_close(answer['value'], answer['placement'] + answer['hand'])

    def test_privileged_features_and_critic_losses_cannot_change_actor_or_inputs(self):
        import riichi_py
        net = actor(True); identity = artifacts.digest_state(artifacts.actor_payload(net))
        public = torch.randn(2, 1012, 34, requires_grad=True)
        hidden = torch.randn(2, riichi_py.ORACLE_PLANES, 34, requires_grad=True)
        ranks = torch.tensor([[1., 0, 0, 0], [0, 0, 0, 1.]])
        defence = torch.zeros(2, 3, 104); defence[:, :, 0] = 1
        critic = Critics(config(oracle='full', rank_head='categorical', defence_weight=.1))
        loss, _ = critic.losses(public, hidden, ranks, torch.tensor([1., -1.]), defence)
        loss.backward()
        self.assertIsNone(public.grad); self.assertIsNone(hidden.grad)
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        self.assertTrue(any(p.grad is not None for p in critic.parameters()))
        self.assertEqual(identity, artifacts.digest_state(artifacts.actor_payload(net)))
        with self.assertRaises(ValueError): critic.public(public, hidden)
        with self.assertRaises(ValueError): critic.oracle(public)

    def test_hands_oracle_does_not_consume_wall_planes(self):
        import riichi_py
        net = Critics(config(oracle='hands'))
        self.assertEqual(net.hidden_planes, riichi_py.HIDDEN_HANDS_PLANES)
        with torch.no_grad(): net.oracle.rank.weight.fill_(.01)
        public = torch.randn(2, 1012, 34)
        hidden = torch.randn(2, riichi_py.ORACLE_PLANES, 34)
        before = net.baseline(public, hidden)
        hidden[:, riichi_py.HIDDEN_HANDS_PLANES:] += 100
        torch.testing.assert_close(before, net.baseline(public, hidden), rtol=0, atol=0)

    def test_missing_defence_is_not_safe_and_payment_uses_only_legal_ron(self):
        net = SignalCritic(config(defence_weight=1.), defence=True)
        inputs = torch.randn(2, 1012, 34)
        ranks = torch.full((2, 4), .25)
        empty = torch.zeros(2, 3, 104)
        _, report = net.loss(net(inputs), ranks, torch.zeros(2), empty)
        for k in ('tenpai_loss', 'ron_loss', 'payment_loss'): self.assertEqual(float(report[k].detach()), 0)
        labelled = empty.clone(); labelled[:, :, 0] = 1
        labelled[0, 0, 2 + 5] = 1; labelled[0, 0, 36 + 5] = 1; labelled[0, 0, 70 + 5] = 12000
        validate_defence(labelled.numpy())
        loss, report = net.loss(net(inputs), ranks, torch.zeros(2), labelled)
        self.assertGreater(float(report['payment_loss'].detach()), 0)
        loss.backward(); self.assertTrue(torch.isfinite(net.payment.weight.grad).all())
        for index in (1, 2, 36, 70):
            bad = np.zeros((1, 3, 104), np.float32); bad[0, 0, index] = 1
            with self.subTest(index=index), self.assertRaises(ValueError): validate_defence(bad)

    def test_all_architectures_and_objectives_have_finite_gradients(self):
        import riichi_py
        for objective in ('hybrid', 'placement'):
            for head in ('scalar', 'categorical'):
                for oracle in ('none', 'hands', 'full'):
                    with self.subTest(objective=objective, head=head, oracle=oracle):
                        net = Critics(config(objective=objective, rank_head=head, oracle=oracle))
                        loss, _ = net.losses(torch.randn(2, 1012, 34), torch.zeros(2, riichi_py.ORACLE_PLANES, 34),
                                             torch.eye(4)[:2], torch.tensor([1., -1.]))
                        loss.backward()
                        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_richer_ranker_uses_distinct_fused_inputs_and_detaches_actor(self):
        net = actor(True)
        for mode in ('ours', 'fused', 'own-tower'):
            head = Ranker(net, mode, 8)
            self.assertEqual(head.mortal_dim > 0, mode == 'fused')
            loss = (head(net, torch.randn(2, 1012, 34)) - 1).square().mean(); loss.backward()
            self.assertTrue(any(p.grad is not None for p in head.parameters()))
            self.assertTrue(all(p.grad is None for p in net.parameters()))


class TeacherTests(Base):
    def test_pairwise_tilt_preserves_every_unsearched_probability(self):
        p = np.zeros(78); p[:3] = [.5, .3, .2]
        q = targets.confirmed_tilt(p, 0, 1, .4)
        self.assertGreater(q[1], p[1]); self.assertEqual(q[2], p[2]); self.assertAlmostEqual(q.sum(), 1)
        np.testing.assert_array_equal(p, targets.confirmed_tilt(p, 0, 1, 0))

    def test_alias_and_two_stage_riichi_probabilities_roundtrip(self):
        legal = np.zeros(78, bool); legal[[0, 4, 34, 35]] = True
        p = np.zeros(46); p[[0, 4, 34, 37]] = [.2, .2, .1, .5]
        after = np.zeros(46); after[:2] = [.7, .3]
        joint = targets.joint_policy(p, after, legal)
        self.assertAlmostEqual(joint[4], .3); self.assertAlmostEqual(joint[34], .35)
        first, second = targets.lift_policy(joint, p, after, legal)
        np.testing.assert_allclose(first, p); np.testing.assert_allclose(second, after)
        q = targets.confirmed_tilt(joint, 0, 35, .2)
        first, second = targets.lift_policy(q, p, after, legal)
        np.testing.assert_allclose(targets.joint_policy(first, second, legal), q, atol=1e-7)
        self.assertAlmostEqual(float(first[4] / first[34]), 2.)
        with self.assertRaises(ValueError): targets.joint_policy(p, None, legal)

    def test_confirmation_and_audit_are_distinct_and_audit_cannot_force_override(self):
        calls = []
        def evaluate(actions, worlds):
            calls.append((list(actions), worlds))
            edge = 1. if len(calls) != 2 else -.2
            return {a: dict(edge=0. if a == actions[0] else edge, error=.01, worlds=worlds) for a in actions}
        result = targets.teach_root(evaluate, [3, 4, 5], worlds=3, confirm_worlds=7, audit_worlds=11)
        self.assertEqual(calls, [([3,4,5],3), ([3,4],7), ([3,4,5],11)])
        self.assertFalse(result['confirmed']); self.assertEqual(result['lower_edge'], 0)

    def test_sequential_halving_honours_slot_budget_and_keeps_incumbent(self):
        for k in range(2, 24):
            calls = []
            def evaluate(actions, worlds):
                calls.append((list(actions), worlds))
                return {a: dict(edge=float(a), error=.01, worlds=worlds) for a in actions}
            result = targets.teach_root(evaluate, list(range(k)), race_budget=k * 40, worlds=3, confirm_worlds=5, audit_worlds=7)
            discovery = calls[:-2]
            self.assertTrue(all(a[0] == 0 and n >= 3 for a, n in discovery))
            self.assertLessEqual(sum(len(a) * n for a, n in discovery), k * 40)
            self.assertEqual(result['challenger'], k-1); self.assertTrue(result['confirmed'])

    def test_gumbel_explores_beyond_policy_prefix_but_preserves_incumbent(self):
        legal = np.zeros(78, bool); legal[:10] = True
        p = np.zeros(78); p[:10] = .1
        rng = np.random.default_rng(52); visited = set()
        for _ in range(100):
            actions = targets.candidates(range(78), legal, p, 3, rng, 'gumbel')
            self.assertEqual(actions[0], 0); visited.update(actions)
        self.assertEqual(visited, set(range(10)))

    def test_insufficient_or_missing_evidence_does_not_confirm(self):
        def evaluate(actions, worlds):
            return {a: dict(edge=1., error=float('inf'), worlds=2) for a in actions}
        self.assertFalse(targets.teach_root(evaluate, [0, 1], worlds=3, confirm_worlds=3, audit_worlds=3)['confirmed'])
        for kw in (dict(worlds=0), dict(confirm_worlds=2), dict(audit_worlds=True), dict(race_budget=1)):
            with self.assertRaises(ValueError): targets.teach_root(evaluate, [0, 1], **kw)


class LeagueAndArtifactTests(Base):
    def members(self):
        return [dict(sha256=c*64, name=c, role=role) for c, role in [('a','champion'),('b','reference'),('c','older')]]
    def test_adaptive_league_keeps_diversity_and_exact_matrix(self):
        league = League(self.members(), uniform=.3)
        before = league.weights()
        league.update('d'*64, np.zeros(32, np.int64), np.full(32, 4.))
        after = league.weights()
        self.assertGreater(after[0], before[0]); self.assertTrue((after >= .1).all())
        np.testing.assert_array_equal(after, League(self.members(), .3, True, league.state()).weights())
        self.assertEqual(league.matrix['d'*64]['a'*64]['games'], 32)
        self.assertNotIn('b'*64, league.matrix['d'*64])

    def test_selector_has_uniform_floor_and_never_duplicates_roots(self):
        roots = [dict(priority=float(i), step=i, game=0) for i in range(100)]
        got = select_roots(roots, 20, .5, np.random.default_rng(20))
        self.assertEqual(len(got), 20); self.assertEqual(len({r['step'] for r in got}), 20)
        self.assertTrue(any(r['step'] < 40 for r in got))
        self.assertEqual(select_roots([], 20, .5, np.random.default_rng(1)), [])

    def test_artifacts_are_exclusive_checked_and_missing_manifest_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data'; a = dict(test=np.arange(6, dtype=np.int64))
            identity = artifacts.save(path, 'test', {'hello': 'world'}, a)
            meta, loaded, read_id = artifacts.load(path, 'test')
            self.assertEqual(identity, read_id); np.testing.assert_array_equal(loaded['test'], a['test'])
            with self.assertRaises(FileExistsError): artifacts.save(path, 'test', {}, a)
            (path / 'test.npy').write_bytes(b'broken')
            with self.assertRaises(ValueError): artifacts.load(path, 'test')
            (path / 'manifest.json').unlink()
            with self.assertRaises(ValueError): artifacts.load(path, 'test')

    def test_artifact_write_failure_publishes_no_manifest(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(np, 'save', side_effect=OSError('disk full')):
            path = Path(temp) / 'failed'
            with self.assertRaises(OSError): artifacts.save(path, 'test', {}, dict(test=np.zeros(1)))
            self.assertFalse((path/'manifest.json').exists())

    def test_whole_game_holdout_handles_uint64_and_refuses_empty_split(self):
        ids = np.array([2**63 + 1, 2**63 + 1, 2**63 + 2, 2**63 + 2], np.uint64)
        training, held = split(ids, 2)
        self.assertEqual(set(training), {0, 1}); self.assertEqual(set(held), {2, 3})
        with self.assertRaises(ValueError): split(ids[:2], 2)
        self.assertEqual(len(split(ids[:2], 0)[0]), 2)

    def test_configuration_presets_are_explicit_and_reject_bad_controls(self):
        from neural.learning_lab.__main__ import parser, configuration
        for name in ('baseline','oracle','categorical','placement','defence','league','all','exploiter'):
            args = parser().parse_args(['train','--checkpoint','actor.pt','--out','new','--preset',name])
            configuration(args).validate()
        for kw in (dict(games=True), dict(oracle='cheat'), dict(league_uniform=0), dict(amp=1), dict(channels=7), dict(seed=9_000_000)):
            with self.subTest(kw=kw), self.assertRaises(ValueError): Config(**kw).validate()
        self.assertIsNone(configuration(parser().parse_args(['train','--resume','x','--out','new']), False))

    def test_optimizer_restore_rejects_missing_active_slots(self):
        net = torch.nn.Linear(2, 1); optimizer = torch.optim.AdamW(net.parameters())
        net(torch.ones(2)).sum().backward(); optimizer.step()
        state = deepcopy(optimizer.state_dict()); active = sorted(state['state'])
        bad = deepcopy(state); del bad['state'][active[0]]
        with self.assertRaises(ValueError): _restore_optimizer(optimizer, bad, active)
        bad = deepcopy(state); bad['state'][active[0]]['step'] = torch.tensor(1.5)
        with self.assertRaises(ValueError): _restore_optimizer(optimizer, bad, active)

    def test_all_actor_payloads_roundtrip_without_experiment_parameters(self):
        for fused in (False, True):
            original = actor(fused); payload = artifacts.actor_payload(original)
            restored = artifacts.actor_from_payload(payload)
            self.equal_tree(payload, artifacts.actor_payload(restored))
            self.assertNotIn('learning_lab', artifacts.actor_payload(restored))
            self.assertTrue(all('oracle' not in key for key in restored.fuse.state_dict()) if fused else True)


if __name__ == '__main__': unittest.main()
