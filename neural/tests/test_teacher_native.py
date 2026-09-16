"""Real native teacher decisions and completed placement-teacher replay contracts."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import riichi_py

from neural import contract, model, placement, searched
from neural.observe import Views
from neural.collect_search import collect, SearchSettings
from neural.search_replay import SearchReplay
from neural.train_search import run
from neural.checkpoints import atomic_save


@unittest.skipUnless(getattr(riichi_py, 'SEARCH_API_VERSION', 0) == 4, 'requires rebuilt search API 4')
class NativeTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_real_root_call_is_searched_without_mutating_table_or_follower(self):
        net=model.PolicyValueNet(8,1,actions=46).eval()
        arena=riichi_py.Arena(games=1,seed=2,bot_places=[])
        views=Views(arena,1,{'mortal'})
        for _ in range(200):
            legal=np.frombuffer(arena.legal_mask(),np.uint8).reshape(1,78).astype(bool)
            views.advance()
            # A claim window with a real chii/pon alternative to Pass.
            if legal[0,70] and legal[0,71:75].any(): break
            arena.step([int(np.nonzero(legal[0])[0][0])])
        else: self.fail('no claim-window fixture')
        seats=np.frombuffer(arena.seats(),np.uint8)
        players=np.frombuffer(arena.seat_players(),np.uint8).reshape(1,4)
        rows=np.array([0],np.int64); deciding=players[rows,seats].astype(np.int64)
        views.prepare(rows,deciding)
        before=arena.observations()
        root_before,_=views.sparse_and_masks(rows,deciding)
        copied={name: array.copy() for name,array in root_before.arrays().items()}
        served=contract.serve(net)
        call=int(np.nonzero(legal[0,71:75])[0][0]+71)
        evidence={}
        with contract.following(arena,views.observer.follower):
            choices=searched.search_with_value_head(net,arena,[[70,call]], [1/34]*102,
                worlds=3,candidates=2,margin=2.,hurried=False,device='cpu',pool=1,
                played_by='network',search_calls=True,served=served,evidence=evidence)
        self.assertIn(choices[0], [70,call])
        self.assertEqual(len(evidence['judgements'][0]),2)
        self.assertEqual(arena.observations(),before)
        after,_=views.sparse_and_masks(rows,deciding)
        for name,array in after.arrays().items(): np.testing.assert_array_equal(array,copied[name])
        arena.step(choices)  # The actual call is still owed, and remains legal.

    def test_real_placement_only_search_returns_only_placement_scale(self):
        net=model.PolicyValueNet(8,1,actions=46).eval()
        arena=riichi_py.Arena(games=1,seed=7,bot_places=[])
        views=Views(arena,1,{'mortal'}); views.advance()
        legal=np.frombuffer(arena.legal_mask(),np.uint8).reshape(1,78).astype(bool)
        seats=np.frombuffer(arena.seats(),np.uint8)
        players=np.frombuffer(arena.seat_players(),np.uint8).reshape(1,4)
        rows=np.array([0],np.int64); deciding=players[rows,seats].astype(np.int64)
        views.prepare(rows,deciding)
        head=placement.new_head(net).eval()  # Zero predictions, never a hand-points bonus.
        served=contract.serve(net,placement_head=head)
        evidence={}; ranking=[np.nonzero(legal[0])[0][:2].tolist()]
        with contract.following(arena,views.observer.follower):
            searched.search_with_value_head(net,arena,ranking,[1/34]*102,worlds=3,
                candidates=2,margin=2.,hurried=False,device='cpu',pool=1,played_by='network',
                depth=-1,valued_by='placement',objective='placement',served=served,evidence=evidence)
        for _, _, worlds in evidence['judgements'][0]:
            self.assertTrue(np.isfinite(worlds).all())
            self.assertTrue((np.abs(worlds)<=1.5).all())
            # First hand of the match cannot finish it: zero boundary head, zero utility.
            np.testing.assert_array_equal(worlds,np.zeros(len(worlds)))

    def test_completed_placement_teacher_replay_fits_and_resumes_hybrid_student(self):
        net=model.PolicyValueNet(8,1,actions=46).eval()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root=Path(temp)
            head=placement.new_head(net)
            placement.save(head,root/'head.pt',dict(features=placement.fingerprint(net,head.feature_version)))
            atomic_save({**net.payload_fields(),'model':net.state_dict(),'generation':1},root/'actor.pt')
            # No search needed to exercise provenance and complete-game accounting;
            # the search objective/claim paths have dedicated native tests above.
            replay=collect(net,games=1,seed=77,settings=SearchSettings(valued_by='placement',
                           objective='placement',depth=-1,sure=0), actor_sha256='a'*64,
                           source_revision='b'*40,placement_head=root/'head.pt',device='cpu')
            replay.save(root/'replay'); loaded=SearchReplay.load(root/'replay')
            self.assertEqual(loaded.metadata['teacher']['student_value_head'],'critic')
            one=run(root/'actor.pt',[root/'replay'],root/'one',epochs=1,
                    overrides=dict(batch=32,validation_every=0,policy_mode='changed',actor_kl=1.))
            two=run(one,[root/'replay'],root/'two',epochs=1,resume=True)
            payload=torch.load(two,weights_only=True)
            self.assertEqual(payload['search_training']['epochs'],2)


if __name__=='__main__': unittest.main()
