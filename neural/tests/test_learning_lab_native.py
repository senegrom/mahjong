"""Small real engines/networks through the experimental learning lifecycle.

These are integration/continuation checks, not trained-player strength evidence.
No external weights, paid execution, production model, or network is required.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import riichi_py

from neural import contract, duel, zoo
from neural.checkpoints import atomic_save
from neural.learning_lab import artifacts
from neural.learning_lab.collection import collect, Round
from neural.learning_lab.config import Config
from neural.learning_lab.critics import validate_defence
from neural.learning_lab.fit import fit_critics, distill
from neural.learning_lab.league import League
from neural.learning_lab.ranker import fit as fit_ranker, load as load_ranker, Player
from neural.learning_lab.reanalysis import run as reanalyse, Lessons
from neural.learning_lab.train import Learner, run
from neural.tests.test_learning_lab import Base, actor, config


@unittest.skipUnless(getattr(riichi_py, 'LEARNING_LABEL_API_VERSION', None) == 1,
                     'requires the optional rebuilt learning label API 1')
class NativeLifecycleTests(Base):
    def test_native_defensive_labels_do_not_move_the_actual_table(self):
        arena = riichi_py.Arena(games=2, seed=71, bot_places=[])
        before, oracle = arena.observations(), arena.oracle()
        labels = np.frombuffer(arena.learning_defence(), np.float32).reshape(2, 3, 104)
        validate_defence(labels)
        self.assertEqual(before, arena.observations()); self.assertEqual(oracle, arena.oracle())
        legal = np.frombuffer(arena.legal_mask(), np.uint8).reshape(2, 78)
        for i in range(2):
            np.testing.assert_array_equal(labels[i, :, 2:36], np.tile(legal[i, :34], (3, 1)))
        choices = [int(np.flatnonzero(row[:34])[0]) for row in legal]
        arena.step(choices)
        for _ in range(100):
            legal = np.frombuffer(arena.legal_mask(), np.uint8).reshape(2, 78)
            owed = legal[:, 70] > 0
            calls = np.frombuffer(arena.learning_defence(), np.float32).reshape(2, 3, 104)
            if owed.any():
                self.assertTrue((calls[owed] == 0).all()); break
            arena.step([int(np.flatnonzero(row)[0]) if row.any() else 0 for row in legal])
        else: self.fail('fixture did not encounter a claim decision')

    def test_reanalysis_rng_is_separate_from_actual_game_and_reproducible(self):
        one = riichi_py.Arena(games=1, seed=71, bot_places=[])
        two = riichi_py.Arena(games=1, seed=71, bot_places=[])
        one.learning_seed_search(11); two.learning_seed_search(12)
        same = riichi_py.Arena(games=1, seed=71, bot_places=[])
        same.learning_seed_search(11)
        first = one.search_resample_seeds([True])
        self.assertEqual(first, same.search_resample_seeds([True]))
        self.assertNotEqual(first, two.search_resample_seeds([True]))
        self.assertEqual(one.observations(), two.observations())
        self.assertEqual(one.oracle(), two.oracle())
        # The normal game remains identical through actual actions.
        for _ in range(50):
            legal = np.frombuffer(one.legal_mask(), np.uint8).reshape(1, 78)
            moves = [int(np.flatnonzero(legal[0])[0])]
            one.step(moves); two.step(moves)
            self.assertEqual(one.observations(), two.observations())
            self.assertEqual(one.oracle(), two.oracle())

    def test_completed_collector_retains_components_and_selected_privileged_inputs(self):
        for oracle in ('none', 'hands', 'full'):
            with self.subTest(oracle=oracle):
                torch.manual_seed(30); net = actor()
                before = artifacts.digest_state(artifacts.actor_payload(net))
                cfg = config(oracle=oracle, defence_weight=.1, rank_head='categorical')
                data = collect(net, cfg, cfg.seed)
                self.assertEqual(before, artifacts.digest_state(artifacts.actor_payload(net)))
                self.assertEqual(data.arrays['oracle'].shape[1], {'none':0, 'hands':riichi_py.HIDDEN_HANDS_PLANES, 'full':riichi_py.ORACLE_PLANES}[oracle])
                self.assertEqual(len(data.trace['roots']), cfg.roots)
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / 'round'; data.save(path)
                    restored = Round.load(path)
                    np.testing.assert_array_equal(data.arrays['return'], restored.arrays['return'])
                    bad = deepcopy(data); bad.arrays['ranks'][0] = 0
                    with self.assertRaises(ValueError): bad.save(Path(temp) / 'bad')
                    self.assertFalse((Path(temp)/'bad').exists())

    def test_fused_ppo_exact_serialized_resume_and_damaged_state_rejection(self):
        cfg = config(oracle='hands', rank_head='categorical', defence_weight=.1, objective='placement', batch=128)
        torch.manual_seed(111); original = dict(**artifacts.actor_payload(actor(True)), generation=7)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); source = root / 'actor.pt'; atomic_save(original, source)
            initial_bytes = source.read_bytes()
            continuous = run(source, root / 'continuous', cfg=cfg, rounds=2)
            one = run(source, root / 'one', cfg=cfg, rounds=1)
            two = run(one, root / 'two', rounds=1, resume=True)
            a, b = (torch.load(path, weights_only=True) for path in (continuous, two))
            self.equal_tree(a, b)
            self.assertEqual(source.read_bytes(), initial_bytes)
            self.assertEqual(a['generation'], 9)
            with self.assertRaises(ValueError): run(two, root / 'changed', resume=True, cfg=replace(cfg, objective='hybrid'))
            self.assertFalse((root/'changed').exists())
            bad = deepcopy(a); slots = bad['learning_lab']['active_optimizer']
            del bad['learning_lab']['optimizer']['state'][slots[0]]
            with self.assertRaises(ValueError): Learner(bad, cfg, saved=bad['learning_lab'])
            bad = deepcopy(a)
            value = next(v for v in bad['learning_lab']['reference']['combined'].values() if v.is_floating_point())
            value.reshape(-1)[0] += 1
            with self.assertRaises(ValueError): Learner(bad, cfg, saved=bad['learning_lab'])

    def test_critic_only_warmup_never_updates_actor_parameters(self):
        cfg = config(oracle='hands', rank_head='categorical', warmup_rounds=1, batch=128)
        torch.manual_seed(cfg.seed); payload = artifacts.actor_payload(actor(True))
        learner = Learner(payload, cfg)
        before = artifacts.digest_state(artifacts.actor_payload(learner.actor))
        with tempfile.TemporaryDirectory() as temp:
            report = learner.step(Path(temp)/'data')
            self.assertTrue(report['critic_only']); self.assertEqual(learner.updates, 0)
            self.assertGreater(learner.critic_updates, 0)
            self.assertEqual(before, artifacts.digest_state(artifacts.actor_payload(learner.actor)))
            saved = learner.checkpoint(); resumed = Learner(saved, cfg, saved=saved['learning_lab'])
            self.assertEqual(resumed.rounds, 1); self.assertEqual(resumed.actor_rounds, 0)

    def test_all_features_prefit_reanalyse_rank_distill_and_reload(self):
        cfg = config(oracle='hands', rank_head='categorical', defence_weight=.1, objective='placement', batch=128,
                     adaptive_league=True, opponent_share=1., role='exploiter')
        torch.manual_seed(21); initial = artifacts.actor_payload(actor(True))
        opponent = actor(); opponent_state = artifacts.actor_payload(opponent)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); source = root/'actor.pt'; reference = root/'opponent.pt'
            atomic_save(dict(**initial, generation=1), source); atomic_save(opponent_state, reference)
            trained = run(source, root/'first', cfg=cfg, rounds=1, opponents=[reference], roles=['champion'])
            payload = torch.load(trained, weights_only=True)
            self.assertTrue(payload['learning_lab']['league']['matrix'])
            trace = root/'first/data/round-000000'
            reports = fit_critics([trace], root/'prefit.pt', cfg, epochs=1, validation_every=0)
            self.assertEqual(reports['validation']['rows'], 0)
            self.assertIsNone(reports['validation']['metrics'])
            with self.assertRaises(ValueError): fit_critics([trace, trace], root/'duplicate.pt', cfg, epochs=1, validation_every=0)
            prefitted = run(source, root/'prefitted', cfg=cfg, rounds=1, opponents=[reference], roles=['champion'], critic_init=root/'prefit.pt')
            self.assertIsNotNone(torch.load(prefitted, weights_only=True)['learning_lab']['critic_origin'])
            lesson_path = root/'lessons'
            lessons = reanalyse(trace, trained, lesson_path, worlds=3, candidates=2, confirm_worlds=3,
                                audit_worlds=3, race_budget=12, candidate_method='gumbel', batch=32)
            self.assertGreater(len(lessons.roots()), 0)
            self.assertEqual(lessons.identity, Lessons.load(lesson_path).identity)
            with self.assertRaises(FileExistsError): reanalyse(trace, trained, lesson_path)
            for mode in ('fused', 'own-tower'):
                fit_ranker(trained, lesson_path, root/(mode+'.pt'), mode=mode, channels=8,
                           epochs=1, batch=2, validation_every=0)
            model = artifacts.actor_from_payload(payload).eval()
            head = load_ranker(root/'fused.pt', model)
            ranked = Player(model, head)
            # One complete real table verifies that the new ranker can act.
            scores = duel.table(ranked, zoo.load_player(reference, 'cpu'), 1, 9_100_123, 0, device='cpu')
            self.assertEqual(scores.shape, (1, 4))
            distilled = distill(trained, lesson_path, root/'distilled', cfg=cfg, epochs=1)
            output = torch.load(distilled, weights_only=True)
            self.assertNotIn('learning_lab', output)
            self.assertTrue(output['learning_distillation']['requires_critic_refit'])
            self.assertEqual(artifacts.actor_from_payload(output).actions, 46)
            mix = run(trained, root/'mixed', cfg=replace(cfg, lesson_weight=.1), rounds=1,
                      opponents=[reference], roles=['champion'], lessons=lesson_path)
            self.assertEqual(torch.load(mix, weights_only=True)['learning_lab']['lesson_id'], lessons.identity)

    def test_failed_round_cannot_publish_checkpoint(self):
        learner = Learner(artifacts.actor_payload(actor()), config())
        with tempfile.TemporaryDirectory() as temp, patch('neural.learning_lab.train.collect', side_effect=RuntimeError('truncated')):
            with self.assertRaises(RuntimeError): learner.step(Path(temp))
            with self.assertRaises(RuntimeError): learner.checkpoint()
            with self.assertRaises(RuntimeError): learner.step(Path(temp))


if __name__ == '__main__': unittest.main()
