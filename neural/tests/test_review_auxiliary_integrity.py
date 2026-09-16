"""Atomic auxiliary artifacts and placement splits by real environment seeds."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import checkpoints, placement, selfplay, sibling_head, model
from neural.observe import Planes


def batch(seed=11):
    return SimpleNamespace(seed=seed, game_of=torch.tensor([0, 0, 1]),
        observations=Planes.from_follower([0, 1, 2, 3], [0, 1, 2], [1., 1., 1.]),
        decisions=3, placements=torch.tensor([.5, .5, -1.5]), returns=torch.tensor([1., 2., 0.]),
        games=2, hands=2, reward_version=1, final_scores=np.zeros((2, 4), dtype=np.int32))


class ArtifactTests(unittest.TestCase):
    def test_all_auxiliary_writers_preserve_old_bytes_on_serialization_failure(self):
        head, ranker = placement.Judge(8), sibling_head.new_head(model.PolicyValueNet(8, 1, actions=46))
        writers = [lambda path: placement.save(head, path, {}),
                   lambda path: sibling_head.save(ranker, path, {}),
                   lambda path: selfplay.save_round(batch(), path)]
        for writer in writers:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'saved.pt'
                writer(path)
                before = path.read_bytes()
                def fail(payload, stream):
                    stream.write(b'incomplete tensor archive')
                    raise OSError('injected serialization failure')
                with patch.object(torch, 'save', side_effect=fail), self.assertRaises(OSError):
                    writer(path)
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse(list(path.parent.glob('*.partial')))
                torch.load(path, weights_only=False)

    def test_validation_and_rename_failures_keep_the_previous_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'head.pt'
            head = placement.Judge(8)
            placement.save(head, path, {})
            before = path.read_bytes()
            with torch.no_grad(): head.body[2].bias.fill_(float('nan'))
            with self.assertRaises(ValueError): placement.save(head, path, {})
            self.assertEqual(path.read_bytes(), before)
            with torch.no_grad(): head.body[2].bias.fill_(1.)
            real_replace = checkpoints.os.replace
            for after in (False, True):
                def fail(source, target):
                    if after: real_replace(source, target)
                    raise KeyboardInterrupt('rename boundary')
                with patch.object(checkpoints.os, 'replace', side_effect=fail), self.assertRaises(KeyboardInterrupt):
                    placement.save(head, path, {})
                recovered, _ = placement.load(path)
                self.assertEqual(float(recovered.body[2].bias.detach()), 1. if after else 0.)

    def test_saving_does_not_consume_random_state_or_allow_schema_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            head = placement.Judge(8)
            state = torch.get_rng_state().clone()
            placement.save(head, Path(tmp) / 'head.pt', {})
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            with self.assertRaises(ValueError): placement.save(head, Path(tmp) / 'bad.pt', {'channels': 100})
            with self.assertRaises(ValueError): selfplay.save_round(batch(), Path(tmp) / 'round.pt', {'seed': 12})


class PlacementIdentityTests(unittest.TestCase):
    def test_duplicate_copy_overlaps_and_missing_seed_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, copy, overlap = (Path(tmp) / n for n in ('first.pt', 'copy.pt', 'overlap.pt'))
            selfplay.save_round(batch(), first)
            shutil.copyfile(first, copy)
            selfplay.save_round(batch(seed=12), overlap)
            for path in (first, copy, overlap):
                with self.assertRaisesRegex(ValueError, 'duplicate or overlapping'):
                    placement.load_rounds([first, path])
            payload = torch.load(first, weights_only=False)
            del payload['seed']
            torch.save(payload, copy)
            with self.assertRaisesRegex(ValueError, 'seed'): placement.load_rounds([copy])

    def test_order_and_chunking_do_not_change_validation_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for seed in (19, 30, 2**64 - 2):
                path = Path(tmp) / f'{seed}.pt'
                selfplay.save_round(batch(seed), path)
                paths.append(path)
            for order in (paths, list(reversed(paths))):
                positions = placement.load_rounds(order)
                training, held = positions.split(10)
                self.assertEqual(set(positions.games[held]), {20, 30})
                self.assertTrue(set(positions.games[training]).isdisjoint(positions.games[held]))


if __name__ == '__main__': unittest.main()
