"""Regression contracts for placement PPO, league sampling and reliable evidence."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import combined, model, mortal_model, selfplay, rl_policy, rl_critics, train_league
from neural.checkpoints import atomic_save, publish_training_snapshot
from neural.evidence import lower_mean, split_games
from neural.league_state import pin_population, digest
from neural.observe import Planes
from neural.population import Member, Population, table_seats
from neural.rl_objective import Objective, advantages, normalize_actor_advantages, trajectory_links
from neural.seed_ledger import SeedLedger, bind_protocol, DOMAINS, LANE


def sparse(n):
    return Planes(np.arange(n + 1, dtype=np.int64), np.arange(n, dtype=np.uint16),
                  np.ones(n, dtype=np.float16))


def mortal_config():
    return {"resnet": {"conv_channels": 16, "num_blocks": 1}, "control": {"version": 4}}


class ObjectiveTests(unittest.TestCase):
    def test_placement_does_not_read_hand_points(self):
        batch = SimpleNamespace(returns=torch.tensor([20., -10.]), placements=torch.tensor([-.5, 1.5]))
        self.assertTrue(torch.equal(Objective().targets(batch), batch.placements))
        self.assertEqual(Objective().version, 2)
        self.assertTrue(torch.equal(Objective('hybrid').targets(batch), batch.returns))

    def test_interleaved_players_and_two_stage_reach_cross_hand_boundaries(self):
        games = [0, 0, 1, 0, 0, 0]
        players = [0, 1, 0, 0, 0, 1]
        links, done = trajectory_links(games, players, [0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(links, [3, 5, -1, 4, -1, -1])
        np.testing.assert_array_equal(done, [False, False, True, False, True, True])
        returns = torch.tensor([1.5, -.5, .5, 1.5, 1.5, -.5])
        values = torch.tensor([.1, .2, .3, .4, .5, .6])
        gae, target = advantages(returns, values, method='gae', next_index=links, gae_lambda=1)
        torch.testing.assert_close(gae, returns - values)
        torch.testing.assert_close(target, returns)
        one_step, _ = advantages(returns, values, method='gae', next_index=links, gae_lambda=0)
        torch.testing.assert_close(one_step, torch.tensor([.3, .4, .2, .1, 1., -1.1]))

    def test_gae_refuses_broken_chain_or_mixed_rewards(self):
        with self.assertRaises(ValueError):
            advantages(torch.ones(2), torch.zeros(2), method='gae', next_index=[1, 0])
        with self.assertRaisesRegex(ValueError, 'constant terminal'):
            advantages(torch.arange(2.), torch.zeros(2), method='gae', next_index=[1, -1])

    def test_normalize_only_actor_choices(self):
        actual = normalize_actor_advantages(torch.tensor([1., 1000., 3.]), torch.tensor([True, False, True]))
        torch.testing.assert_close(actual, torch.tensor([-1., 0., 1.]))


class SeedAndEvidenceTests(unittest.TestCase):
    def test_changed_batch_size_restart_and_domains_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'seeds.json'
            ledger = SeedLedger(path, seed=17)
            first = ledger.reserve('train', 1024)
            saved = ledger.snapshot()
            consumed_by_crash = ledger.reserve('train', 7)
            resumed = SeedLedger(path, seed=17, saved=saved)
            next_block = resumed.reserve('train', 4096)
            self.assertEqual(first['end'], consumed_by_crash['seed'])
            self.assertEqual(consumed_by_crash['end'], next_block['seed'])
            self.assertLess(next_block['end'], resumed.reserve('validation', 32)['seed'])
            reconstructed = SeedLedger(Path(folder) / 'restored.json', seed=17, saved=resumed.snapshot())
            self.assertEqual(reconstructed.reserve('train', 1)['seed'], next_block['end'])
            other = SeedLedger(Path(folder) / 'other.json', seed=18)
            self.assertLess(next_block['end'], other.reserve('train', 1)['seed'])

    def test_lanes_are_bounded_and_lock_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'seeds.json'; ledger = SeedLedger(path)
            ledger.reserve('train', LANE)
            with self.assertRaises(OverflowError): ledger.reserve('train', 1)
            path.with_name(path.name + '.lock').write_text('interrupted')
            with self.assertRaisesRegex(RuntimeError, 'locked'): ledger.reserve('test', 1)

    def test_evidence_protocol_survives_checkpoint_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            first = SeedLedger(Path(folder) / 'one.json')
            settings = {'confidence': .95, 'method': 'empirical-bernstein'}
            bind_protocol(first, 'promotion', settings)
            first.reserve('promotion', 2048)
            second = SeedLedger(Path(folder) / 'two.json', saved=first.snapshot())
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                bind_protocol(second, 'promotion', {**settings, 'confidence': .9})
            self.assertEqual(second.reserve('promotion', 2048)['attempt'], 2)

    def test_small_stable_edge_has_power_without_replacing_the_null(self):
        a = lower_mean(np.full(2048, .05), alpha=.025, lo=-1.5, hi=1.5)
        b = lower_mean(np.full(2048, .05), alpha=.025, lo=-1.5, hi=1.5, method='hoeffding')
        self.assertGreater(a['lower'], 0)
        self.assertLess(b['lower'], 0)
        self.assertLess(lower_mean(np.zeros(2048), alpha=.025, lo=-1.5, hi=1.5)['lower'], 0)
        later = lower_mean(np.full(2048, .05), alpha=.05/(5*6), lo=-1.5, hi=1.5)
        self.assertGreater(later['radius'], a['radius'])

    def test_entire_deal_and_every_duplicate_stays_in_one_split(self):
        ids = np.repeat(np.arange(500, dtype=np.uint64) + np.uint64(DOMAINS['train'][0]), 4)
        parts = split_games(ids)
        sets = [set(ids[p].tolist()) for p in parts]
        self.assertTrue(all(sets))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(len(p) for p in parts), len(ids))
        reversed_parts = split_games(ids[::-1])
        for wanted, rows in zip(sets, reversed_parts): self.assertEqual(wanted, set(ids[::-1][rows].tolist()))


class PopulationTests(unittest.TestCase):
    def test_mixture_and_player_chairs(self):
        roster = [Member('a', 'reference', '', 1), Member('b', 'older', '', .5)]
        seats = table_seats(10000, roster, (.4, .4, .2), np.random.default_rng(9))
        counts = (seats < 0).sum(axis=1)
        self.assertEqual(set(counts), {1, 3, 4})
        for n, proportion in [(4, .4), (1, .4), (3, .2)]:
            self.assertLess(abs(float((counts == n).mean()) - proportion), .02)
        for chair in range(4):
            self.assertGreater(int((seats[counts == 1, chair] < 0).sum()), 800)
        with self.assertRaises(ValueError): table_seats(1, [], (.4, .4, .2), np.random.default_rng(0))

    def test_latest_path_cannot_silently_replace_a_pinned_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'latest.pt'
            atomic_save({'generation': 1, 'value': torch.ones(1)}, source)
            _, paths, saved = pin_population([(source, 'reference')], root / 'pins')
            old_hash = digest(paths[0])
            atomic_save({'generation': 2, 'value': torch.zeros(1)}, source)
            _, resumed, _ = pin_population([], root / 'pins', saved=saved)
            self.assertEqual(digest(resumed[0]), old_hash)
            _, changed, _ = pin_population([(source, 'reference')], root / 'pins', saved=saved, refresh=True)
            self.assertNotEqual(digest(changed[0]), old_hash)

    def test_publisher_keeps_high_generation_reference_and_checks_existing_pins(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'scratch'; target = root / 'volume'
            atomic_save({'generation': 1}, source / 'latest.pt')
            atomic_save({'generation': 900}, source / 'reference.pt')
            _, pins, _ = pin_population([(source / 'reference.pt', 'reference')], source / 'snapshots')
            publish_training_snapshot(source, target, 1)
            self.assertEqual(digest(source / 'reference.pt'), digest(target / 'reference.pt'))
            (target / 'snapshots' / pins[0].name).write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'snapshot was modified'):
                publish_training_snapshot(source, target, 1)


class ModelAndTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_full_kl_ignores_illegal_infinities_and_has_all_categories(self):
        legal = torch.zeros(4, 46, dtype=torch.bool)
        legal[0, :2] = True; legal[1, [0, 37]] = True
        legal[2, [41, 45]] = True; legal[3, [43, 45]] = True
        old = rl_policy.log_distribution(torch.zeros(4, 46), legal)
        new = old.clone(); new[~legal] = float('nan')
        self.assertTrue(torch.equal(rl_policy.full_kl(old, new, legal), torch.zeros(4)))
        self.assertEqual([int(v.sum()) for v in rl_policy.decision_categories(legal).values()], [1, 1, 1, 1])

    def test_privileged_critic_never_backpropagates_into_observation_or_actor(self):
        observation = torch.randn(2, 1012, 34, requires_grad=True)
        hidden = torch.randn(2, selfplay.ORACLE_PLANES, 34, requires_grad=True)
        critic = rl_critics.TrainingCritic(rl_critics.CriticConfig(1012, selfplay.ORACLE_PLANES, 8, 1))
        with torch.no_grad(): critic.value[-1].weight.fill_(.1)
        critic(observation, hidden).sum().backward()
        self.assertIsNone(observation.grad); self.assertIsNone(hidden.grad)
        self.assertIsNotNone(critic.stem.weight.grad)
        with self.assertRaisesRegex(ValueError, 'requires pre-action'): critic(observation)
        public = rl_critics.TrainingCritic(rl_critics.CriticConfig(1012, 0, 8, 1))
        with self.assertRaisesRegex(ValueError, 'must not receive'): public(observation, hidden)

    def test_combined_policy_only_is_identical_and_optimizers_are_disjoint(self):
        net = combined.Combined(model.PolicyValueNet(8, 1, actions=46), mortal_model.build(16, 1))
        net.eval(); planes = sparse(2).dense('cpu'); legal = torch.ones(2, 46, dtype=torch.bool)
        with torch.no_grad():
            fast = net.policy_only(planes, legal); full = net.everything(planes, legal)[0]
        torch.testing.assert_close(fast, full, rtol=0, atol=0)
        args = SimpleNamespace(lr=1e-4, lr_ours=4e-5, lr_mortal=3e-5)
        actor, aux = train_league.parameter_groups(net, 'combined', args)
        self.assertFalse({id(p) for g in actor for p in g['params']} & {id(p) for p in aux})

    def make_source(self, root, kind):
        source = root / 'initial.pt'; torch.manual_seed(19)
        if kind == 'native':
            net = model.PolicyValueNet(8, 1, actions=46)
            payload = {'model': net.state_dict(), **net.payload_fields()}
        elif kind == 'combined':
            net = combined.Combined(model.PolicyValueNet(8, 1, actions=46), mortal_model.build(16, 1))
            net.mortal_config = mortal_config(); payload = net.state()
        else:
            net = mortal_model.build(16, 1); payload = {**net.state(), 'config': mortal_config()}
        atomic_save(payload, source)
        return source

    def fake_rollout(self, wrapper, games, seed, device, **kwargs):
        n = 8; observations = sparse(n); legal = torch.zeros(n, 46, dtype=torch.bool); legal[:, :3] = True
        wrapper.eval()
        with torch.no_grad():
            logits = rl_policy.logits(wrapper.net, observations.dense(device), legal)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample(); logs = distribution.log_prob(actions)
        players = torch.arange(n) % 4; games_of = torch.zeros(n, dtype=torch.int64)
        links, terminal = trajectory_links(games_of.numpy(), players.numpy())
        placement = (1.5 - players.float())
        return SimpleNamespace(observations=observations, legal=legal, actions=actions, log_probs=logs,
            placements=placement, returns=placement + .2, game_of=games_of, player_of=players,
            next_index=torch.from_numpy(links), terminal=torch.from_numpy(terminal), decisions=n,
            explored=torch.zeros(n, dtype=torch.bool), oracle=torch.zeros(n, selfplay.ORACLE_PLANES, 34),
            held=torch.ones(n, 3, 34)/34, hands=2, matchups=[], timing={})

    def train(self, source, out, rounds, resume=False, extra=()):
        args = ['--resume' if resume else '--initial', str(source), '--out', str(out), '--rounds', str(rounds),
                '--games', '1', '--batch', '4', '--epochs', '1', '--critic-width', '8', '--critic-blocks', '1',
                '--critic-warmup', '0', '--aux-epochs', '1', '--seed', '7', '--device', 'cpu', *extra]
        with patch.object(selfplay, 'play', side_effect=self.fake_rollout), redirect_stdout(io.StringIO()):
            return train_league.main(args)

    def test_checkpoint_resume_exactly_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = self.make_source(root, 'native')
            continuous = self.train(source, root / 'continuous', 2)
            self.train(source, root / 'split', 1)
            resumed = self.train(root / 'split/latest.pt', root / 'split', 1, resume=True)
            self.assertEqual(resumed['generation'], 2)
            for key, value in continuous['model'].items():
                torch.testing.assert_close(value, resumed['model'][key], rtol=0, atol=0)
            for name in continuous['training_critics']:
                for key, value in continuous['training_critics'][name]['state'].items():
                    torch.testing.assert_close(value, resumed['training_critics'][name]['state'][key], rtol=0, atol=0)
            self.assertEqual(continuous['seed_state'], resumed['seed_state'])

    def test_combined_and_mortal_controls_train_with_separate_critics(self):
        for kind in ('combined', 'mortal'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); source = self.make_source(root, kind)
                state = self.train(source, root / 'run', 1)
                self.assertEqual(state['last_metrics']['actor_updates'], 2)
                self.assertEqual(state['last_metrics']['critics']['privileged']['updates'], 2)
                # Existing zoo/player loaders must still accept the actor's layout.
                from neural import zoo
                zoo.load_player(root / 'run/latest.pt', 'cpu')

    def test_objective_change_refuses_silent_hybrid_critic_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = self.make_source(root, 'native'); out = root / 'run'
            self.train(source, out, 1)
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                self.train(out / 'latest.pt', out, 1, resume=True, extra=('--objective', 'hybrid'))
            state = self.train(out / 'latest.pt', out, 1, resume=True,
                               extra=('--objective', 'hybrid', '--reset-critics', '--critic-warmup', '1'))
            self.assertEqual(state['last_metrics']['actor_updates'], 0)
            self.assertGreater(state['last_metrics']['critics']['privileged']['updates'], 0)

    def test_critics_continue_after_actor_kl_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = self.make_source(root, 'native')
            with patch.object(rl_policy, 'update', return_value={'actor_updates': 0, 'kl_early_stop': True}):
                state = self.train(source, root / 'run', 1)
            self.assertGreater(state['last_metrics']['critics']['public']['updates'], 0)
            self.assertGreater(state['last_metrics']['auxiliary']['updates'], 0)

    def test_forced_rollout_rejected_by_actor(self):
        batch = SimpleNamespace(explored=torch.tensor([True]))
        with self.assertRaisesRegex(ValueError, 'forced-exploration'):
            rl_policy.update(None, None, batch, None, None, epochs=1, batch_size=1, ratio_clip=.2,
                             target_kl=.01, entropy_weight=0, grad_clip=1, device='cpu')

    def test_real_three_frozen_opponent_game_keeps_player_identity(self):
        net = model.PolicyValueNet(8, 1, actions=46).eval()
        from neural.zoo import MortalSpacePlayer
        opponent = MortalSpacePlayer(net, 'cpu')
        roster = Population([Member('frozen', 'reference', '', 1)])
        batch = selfplay.play(train_league.RolloutActor(net, 'cpu'), games=1, seed=711,
                              device='cpu', opponents=[opponent], population=roster,
                              table_mix=(0., 1., 0.), want_oracle=True)
        self.assertEqual(int((batch.opponent_seats[0] < 0).sum()), 1)
        self.assertEqual(len(set(batch.player_of.tolist())), 1)
        self.assertEqual(int(batch.terminal.sum()), 1)
        self.assertEqual(batch.oracle.shape[0], batch.decisions)
        links, done = trajectory_links(batch.game_of.numpy(), batch.player_of.numpy())
        np.testing.assert_array_equal(links, batch.next_index.numpy())
        self.assertEqual(batch.matchups[0]['learner_seats'], 1)


if __name__ == '__main__':
    unittest.main()
