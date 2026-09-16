"""Exercise discovery/confirmation all the way through native hand boundaries."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
import riichi_py

from neural import contract, model, placement, searched


def deal_signature(lines):
    event = next(json.loads(line) for line in lines if json.loads(line)['type'] == 'start_kyoku')
    # Seats/scores/honba can depend on the candidate. Chance is the dealt tiles
    # (put in wind order) and indicator, not those deterministic consequences.
    hands = event['tehais']
    dealer = event['oya']
    return hands[dealer:] + hands[:dealer], event['dora_marker']


class TracingServer(contract.EngineServed):
    def __init__(self, net, *, zero=False):
        super().__init__(net, contract.of(net))
        self.placement_head = placement.new_head(net)
        self.deals = []
        self.zero = zero

    def leaf_batches(self, arena, leaf_bytes, counts, device, **options):
        _, lines = arena.leaves_mjai()
        self.deals.append([deal_signature(slot) for slot in lines])
        yield from super().leaf_batches(arena, leaf_bytes, counts, device, **options)

    def value(self, planes, head='critic'):
        # A deliberately scripted critic makes the native comparator propose
        # an override, so the real confirmation branch is necessarily exercised.
        # These numbers are not a learned value/playing-strength measurement.
        self.assert_count = len(planes)
        if self.zero:
            return torch.zeros(len(planes))
        return torch.cat([torch.zeros(len(planes)//2), torch.ones(len(planes)//2)])


class ContinuationChanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def run_confirmation(self, played_by):
        torch.manual_seed(77)
        net = model.PolicyValueNet(8, 1, planes=riichi_py.PLANES, actions=78).eval()
        arena = riichi_py.Arena(games=1, seed=100, bot_places=[])
        untouched = riichi_py.Arena(games=1, seed=100, bot_places=[])
        legal = np.frombuffer(arena.legal_mask(), np.uint8).reshape(1, 78).astype(bool)
        ranking = [np.flatnonzero(legal[0])[:2].tolist()]
        before = arena.observations()
        served = TracingServer(net)
        evidence = {}
        # Fast, legal continuation decisions. Search, chance generation, claim
        # resolution, boundary transitions, leaf serialization and comparison
        # are the real production paths, not mocked _search_once helpers.
        def first_legal(planes, masks):
            return torch.zeros_like(masks, dtype=torch.float32).masked_fill(~masks, -torch.inf)
        with patch.object(net, 'policy_only', side_effect=first_legal):
            choices = searched.search_with_value_head(net, arena, ranking, [1/34]*102,
                worlds=3, candidates=2, margin=0., hurried=True, device='cpu', pool=1,
                played_by=played_by, depth=-1, valued_by='placement', objective='placement',
                served=served, confirm_worlds=3, evidence=evidence)
        self.assertEqual(choices, [ranking[0][1]])
        self.assertEqual(evidence['confirmed'], [True])
        self.assertEqual(len(served.deals), 2)
        for stage in served.deals:
            self.assertEqual(len(stage), 6)
            for world in range(3): self.assertEqual(stage[world], stage[3 + world])
        self.assertNotEqual(served.deals[0][:3], served.deals[1][:3])
        self.assertEqual(arena.observations(), before)
        # Future actual play is unchanged by either evidence stage.
        for _ in range(80):
            mask = np.frombuffer(arena.legal_mask(), np.uint8)
            action = int(np.flatnonzero(mask)[0])
            arena.step([action]); untouched.step([action])
            self.assertEqual(arena.observations(), untouched.observations())
        return served.deals

    def test_network_and_club_confirmation_refresh_future_deals_and_replay_exactly(self):
        for player in ('network', 'club'):
            with self.subTest(player=player):
                self.assertEqual(self.run_confirmation(player), self.run_confirmation(player))

    def test_club_boundary_placement_only_does_not_bank_root_points(self):
        net = model.PolicyValueNet(8, 1, planes=riichi_py.PLANES, actions=78).eval()
        # Club continuations give nonzero root points in this fixture. With a
        # zero placement head, every first-hand boundary value must still be 0.
        for objective in ('placement', 'hybrid'):
            arena = riichi_py.Arena(games=1, seed=100, bot_places=[])
            mask = np.frombuffer(arena.legal_mask(), np.uint8).reshape(1, 78)
            ranking = [np.flatnonzero(mask[0])[:2].tolist()]
            evidence = {}
            searched.search_with_value_head(net, arena, ranking, [1/34]*102,
                worlds=3, candidates=2, margin=0., hurried=True, device='cpu', pool=1,
                played_by='club', valued_by='placement', objective=objective,
                served=TracingServer(net, zero=True), evidence=evidence)
            values = np.array([worlds for _, _, worlds in evidence['judgements'][0]])
            if objective == 'placement': np.testing.assert_array_equal(values, np.zeros_like(values))
            else: self.assertTrue(np.any(values != 0), 'fixture must exercise nonzero root points')


if __name__ == '__main__': unittest.main()
