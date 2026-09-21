"""Opt-in danger, targeted snapshots, synchronous collection and table evidence."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import riichi_py

from neural import (danger, curriculum, parallel_selfplay, chance_control, table_evaluate,
                    model, train_league, rl_policy, selfplay, zoo, paired_actions)
from neural.checkpoints import atomic_save
from neural.evidence import split_games
from neural.observe import Planes, Views
from neural.population import Member, Population
from neural.research_state import Snapshot, context, decision, public_flags
from neural.rl_objective import trajectory_links


def small_actor():
    return model.PolicyValueNet(channels=8, blocks=1, planes=1012, attention=False, actions=46).eval()


def save_actor(path):
    actor = small_actor()
    atomic_save({'model': actor.state_dict(), **actor.payload_fields(), 'generation': 0}, path)
    return actor


def teacher_position(seed, steps):
    arena = riichi_py.Arena(games=1, seed=seed)
    for _ in range(steps):
        arena.step(np.frombuffer(arena.teacher(), np.uint8).tolist())
    views = Views(arena, 1, {'mortal'})
    rows, players, masks = decision(views)
    return views, int(players[0]), masks[0]


def arrays_equal(test, a, b):
    for key in Planes.ARRAYS:
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_snapshot_reproduces_teacher_rng_claims_scores_and_future_deals(self):
        original, _, _ = teacher_position(1, 46)
        snap = Snapshot.capture(original)
        left, right = snap.restore(), snap.restore()
        for _ in range(700):
            for views in (left, right):
                rows, players, _ = decision(views)
            np.testing.assert_array_equal(context(left.arena), context(right.arena))
            self.assertEqual(left.arena.legal_mask(), right.arena.legal_mask())
            rows, players, _ = decision(left)
            if len(rows):
                arrays_equal(self, left.sparse(rows, players), right.sparse(rows, players))
            if left.arena.all_finished():
                self.assertTrue(right.arena.all_finished())
                break
            a = np.frombuffer(left.arena.teacher(), np.uint8).tolist()
            b = np.frombuffer(right.arena.teacher(), np.uint8).tolist()
            self.assertEqual(a, b)
            left.arena.step(a); right.arena.step(b)
        self.assertGreater(sum(left.arena.hands_done()), 1)
        # Consuming a restored game cannot consume or change the saved root.
        self.assertEqual(snap.restore().arena.legal_mask(), original.arena.legal_mask())
        np.testing.assert_array_equal(context(snap.restore().arena), context(original.arena))

    def test_duplicate_rows_are_independent_and_indices_are_validated(self):
        views, _, _ = teacher_position(1, 45)
        with self.assertRaises(ValueError): Snapshot.capture(views, [1])
        doubled = Snapshot.capture(views, [0, 0]).restore()
        masks = np.frombuffer(doubled.arena.legal_mask(), np.uint8).reshape(2, 78)
        legal = np.flatnonzero(masks[0])
        self.assertGreater(len(legal), 1)
        doubled.arena.step([int(legal[0]), int(legal[-1])])
        self.assertEqual(Snapshot.capture(views).restore().arena.legal_mask(), views.arena.legal_mask())

    def test_conditional_riichi_snapshot_does_not_replay_or_duplicate_declaration(self):
        views, player, engine = teacher_position(1, 45)
        self.assertGreater(engine[34:68].sum(), 1)
        before = context(views.arena).copy()
        after = Snapshot.capture(views).restore()
        after.observer.follower.tell(0, player, json.dumps({'type': 'reach', 'actor': player}))
        planes = after.sparse(np.array([0]), np.array([player]))
        mask = np.zeros(46, bool); mask[:34] = engine[34:68]
        moves, valid = paired_actions.shortlist(np.zeros((1, 46)), mask[None], 2)
        root = curriculum.Root(Snapshot.capture(after), planes, mask, moves[0], valid[0], 1, player, 45, 1, 'riichi_discard')
        actor = small_actor()
        first, steps = curriculum.continuation(actor, root, int(moves[0, 0]))
        again, other_steps = curriculum.continuation(actor, root, int(moves[0, 0]))
        self.assertEqual((first, steps), (again, other_steps))
        self.assertTrue(-1.5 <= first <= 1.5)
        # Another tile can be played from exactly the same conditional observation.
        second, _ = curriculum.continuation(actor, root, int(moves[0, 1]))
        self.assertTrue(np.isfinite(second))
        np.testing.assert_array_equal(context(views.arena), before)

    def test_targeted_root_flags_are_public_and_reservoirs_are_bounded(self):
        c = np.zeros((3, 16), np.int64); c[:, 6] = [0, 1, 2]
        c[:, 8:12] = [30000, 29000, 22000, 19000]
        c[0, 13] = 1; c[1, 0] = 1; c[1, 1] = 3
        m = np.zeros((3, 46), bool); m[:, :2] = True
        m[2, 38] = True; m[2, 45] = True
        flags = public_flags(c, m, np.zeros_like(m, dtype=float))
        self.assertTrue(flags['riichi_pressure'][0]); self.assertTrue(flags['close_endgame'][1])
        self.assertTrue(flags['call_pass'][2]); self.assertTrue(flags['close_policy'].all())
        roots, tags, seen = curriculum.select_roots(small_actor(), games=2, seed=13,
            per_category=1, categories=['uniform', 'call_pass'], candidates=2)
        self.assertTrue(roots); self.assertLessEqual(len(roots), 2)
        self.assertGreater(seen['uniform'], 1)
        self.assertEqual(len({(r.game, r.step, r.player, r.stage) for r in roots}), len(roots))
        self.assertTrue(all(r.category in tag for r, tag in zip(roots, tags)))

    def test_truncated_collection_and_bad_categories_never_save_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); actor = root/'actor.pt'; save_actor(actor)
            with self.assertRaises((RuntimeError, ValueError)):
                curriculum.collect(actor, root/'pairs.pt', ledger=root/'seeds.json', games=1, max_steps=1)
            self.assertFalse((root/'pairs.pt').exists())
        with self.assertRaises(ValueError):
            curriculum.select_roots(small_actor(), games=1, seed=1, categories=['unknown'])


class DangerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_labels_are_nonmutating_and_ron_deposit_is_not_payment(self):
        # A naturally dealt position, no handcrafted illegal simulator state.
        views, _, _ = teacher_position(26, 71)
        arena = views.arena
        before = Snapshot.capture(views)
        valid, truth = riichi_py.research_danger(arena)
        mask = np.frombuffer(valid, np.uint8).reshape(1, 2, 34)
        labels = np.frombuffer(truth, np.float32).reshape(1, 2, 3, 34, 2)
        self.assertTrue(mask[0, :, 17].all())
        np.testing.assert_array_equal(labels[0, :, 2, 17, 0], [1., 1.])
        # Riichi on the same losing discard is refunded; it does not make ron cheaper.
        np.testing.assert_array_equal(labels[0, :, 2, 17, 1], [5200., 5200.])
        self.assertEqual(arena.legal_mask(), before.restore().arena.legal_mask())
        np.testing.assert_array_equal(context(arena), context(before.restore().arena))

    def test_native_can_ron_label_matches_actual_single_claim_payment(self):
        views, player, _ = teacher_position(26, 71)
        clone = Snapshot.capture(views).restore().arena
        opening = context(clone)[0, 8+player]
        clone.step([17])
        while not np.frombuffer(clone.hand_ended(), np.uint8).any():
            m = np.frombuffer(clone.legal_mask(), np.uint8).reshape(1, 78)
            who = int(context(clone)[0, 6])
            opponent = (player+3) % 4
            clone.step([69 if who == opponent and m[0, 69] else 70])
        # Hand results are already mapped back to original player identity.
        self.assertEqual(np.frombuffer(clone.hand_result(), np.int32).reshape(1, 4)[0, player], -5200)

    def test_probability_and_payment_losses_use_only_their_valid_labels(self):
        logits = torch.zeros(2, 3, 34, requires_grad=True)
        payment = torch.ones(2, 3, 34, requires_grad=True)
        labels = torch.zeros(2, 3, 34, 2); mask = torch.zeros(2, 34, dtype=torch.bool); mask[:, 2] = True
        labels[0, 1, 2] = torch.tensor([1., 8000.])
        danger.loss_of(logits, payment, labels, mask).backward()
        self.assertTrue(torch.isfinite(logits.grad).all()); self.assertTrue(torch.isfinite(payment.grad).all())
        self.assertEqual(int(torch.count_nonzero(payment.grad)), 1)
        self.assertEqual(float(logits.grad[:, :, 0].abs().sum()), 0.)
        with self.assertRaises(ValueError): danger.loss_of(logits, payment, labels, mask[:, :1])

    def test_danger_attachment_preserves_policy_and_freezes_predictor_under_ppo(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); base_path=root/'base.pt'; base=save_actor(base_path)
            pred=danger.Predictor(8,1)
            fitted={'predictor_version':1,'fitted_epochs':1,'width':8,'blocks':1,'predictor':pred.state_dict()}
            torch.save(fitted,root/'predictor.pt')
            danger.attach(base_path,root/'predictor.pt',root/'joined.pt')
            joined, payload, kind=train_league.load_actor(root/'joined.pt','cpu')
            self.assertEqual(kind,'danger');joined.eval()
            p=torch.randn(4,1012,34);m=torch.ones(4,46,dtype=torch.bool);m[:,45]=False
            torch.testing.assert_close(rl_policy.logits(base,p,m),joined.policy_only(p,m),rtol=0,atol=0)
            args=train_league.parse_args(['--initial',str(root/'joined.pt'),'--out',str(root/'run'),
                '--batch','32','--games','2','--collectors','2','--inference-batch','16',
                '--rounds','1','--critic-width','8','--critic-blocks','1','--critic-warmup','0',
                '--critic','public','--aux-epochs','0','--epochs','1','--device','cpu'])
            groups,aux=train_league.parameter_groups(joined,kind,args)
            active={id(p) for g in groups for p in g['params']}|{id(p) for p in aux}
            self.assertFalse(active & {id(p) for p in joined.predictor.parameters()})
            before={k:v.clone() for k,v in joined.predictor.state_dict().items()}
            with redirect_stdout(io.StringIO()): result=train_league.main([
                '--initial',str(root/'joined.pt'),'--out',str(root/'run'),'--batch','32','--games','2',
                '--collectors','2','--inference-batch','16','--rounds','1','--critic-width','8',
                '--critic-blocks','1','--critic-warmup','0','--critic','public',
                '--aux-epochs','0','--epochs','1','--device','cpu'])
            trained,_,_=train_league.load_actor(root/'run/latest.pt','cpu')
            for key,value in before.items(): torch.testing.assert_close(value,trained.predictor.state_dict()[key],rtol=0,atol=0)
            self.assertGreater(result['last_metrics']['actor_updates'],0)
            self.assertGreater(float(trained.residual[-1].weight.detach().abs().sum()),0.)
            self.assertFalse(any(p.requires_grad for p in trained.predictor.parameters()))
            # Normal zoo/evaluation loading must retain the residual, not the base alone.
            served=zoo.load_player(root/'run/latest.pt','cpu')
            self.assertIsInstance(served.net,danger.DangerPolicy)
            self.assertIn('old_policy',result['last_metrics']['learning_seconds'])

    def test_fit_splits_whole_games_and_tests_only_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);n=100
            planes=Planes(np.arange(n+1,dtype=np.int64),np.arange(n,dtype=np.uint16),np.ones(n,np.float16))
            y=torch.zeros(n,3,34,2);m=torch.ones(n,34,dtype=torch.bool)
            y[:,0,0]=torch.tensor([1.,3900.])
            data={'danger_version':1,'planes':{k:torch.from_numpy(getattr(planes,k)) for k in Planes.ARRAYS},
                  'labels':y,'mask':m,'games':torch.arange(n)}
            torch.save(data,root/'data.pt')
            original=danger.measure;called=[]
            def measured(net,data,planes,rows,*args):
                called.append(np.asarray(rows).copy());return original(net,data,planes,rows,*args)
            with patch.object(danger,'measure',side_effect=measured):
                report=danger.fit([root/'data.pt'],root/'fitted.pt',epochs=2,batch=32,width=8,blocks=1)
            train,val,test=split_games(np.arange(n))
            self.assertEqual(len(called),3)
            np.testing.assert_array_equal(called[0],val);np.testing.assert_array_equal(called[1],val)
            np.testing.assert_array_equal(called[2],test)
            self.assertEqual(sum(b['count'] for b in report['test']['calibration']),len(test)*3*34)
            self.assertEqual(danger.load_predictor(root/'fitted.pt')[0].width,8)

    def test_real_collection_uses_public_rows_with_private_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);save_actor(root/'actor.pt')
            summary=danger.collect(root/'actor.pt',root/'data.pt',ledger=root/'seeds.json',games=2,every=30)
            data,planes=danger.load_data(root/'data.pt')
            self.assertEqual(summary['rows'],len(planes))
            self.assertEqual(data['labels'].shape[1:],(3,34,2))
            self.assertNotIn('oracle',data);self.assertGreater(len(planes),0)


class ParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_service_batches_without_crossing_policies_or_losing_order(self):
        seen=[]
        def score(p,m):
            seen.append(len(p));return p[:,0,0,None].expand(-1,46).masked_fill(~m,-torch.inf)
        with parallel_selfplay.InferenceService({0:score,1:lambda p,m:score(p,m)+10},max_batch=4,capacity=2) as service:
            with ThreadPoolExecutor(3) as workers:
                futures=[]
                for key in (0,1,0):
                    p=torch.arange(9,dtype=torch.float32)[:,None,None].expand(-1,1012,34)
                    m=torch.ones(9,46,dtype=torch.bool);m[:,5]=False
                    futures.append(workers.submit(service.ask,key,p,m))
                for f,key in zip(futures,(0,1,0)):
                    result=f.result();torch.testing.assert_close(result[:,0],torch.arange(9,dtype=torch.float32)+key*10)
                    self.assertTrue(torch.isneginf(result[:,5]).all())
        self.assertTrue(seen);self.assertLessEqual(max(seen),4);self.assertFalse(service.thread.is_alive())

    def test_worker_failure_joins_service_and_produces_no_partial_rollout(self):
        class Broken(nn.Module):
            def policy_only(self,p,m): raise RuntimeError('inference failed')
        with self.assertRaisesRegex(RuntimeError,'inference'):
            parallel_selfplay.play(Broken(),games=2,seed=100,collectors=2,inference_batch=4)
        self.assertFalse(any(t.name.startswith(('mahjong-inference','mahjong-collector')) for t in threading.enumerate()))
        with self.assertRaises((ValueError,RuntimeError)):
            parallel_selfplay.play(small_actor(),games=2,seed=100,collectors=2,max_steps=1)
        self.assertFalse(any(t.name.startswith(('mahjong-inference','mahjong-collector')) for t in threading.enumerate()))

    def test_real_parallel_rollout_preserves_frozen_weights_rng_and_trajectory_identity(self):
        actor=small_actor();before={k:v.clone() for k,v in actor.state_dict().items()}
        rng=torch.get_rng_state().clone()
        batch=parallel_selfplay.play(actor,games=3,seed=110,collectors=2,inference_batch=8,want_oracle=True)
        self.assertEqual(batch.games,3);self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        for key,value in before.items():torch.testing.assert_close(value,actor.state_dict()[key],rtol=0,atol=0)
        links,terminal=trajectory_links(batch.game_of.numpy(),batch.player_of.numpy())
        np.testing.assert_array_equal(links,batch.next_index.numpy());np.testing.assert_array_equal(terminal,batch.terminal.numpy())
        self.assertEqual(batch.oracle.shape[0],batch.decisions)
        self.assertLessEqual(batch.timing['max_inference_rows'],8)
        self.assertGreater(batch.timing['matches_per_second'],0.)
        rl_policy.old_distributions(actor,batch,batch_size=32,device='cpu')

    def test_parallel_population_seats_and_contracts_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'other.pt';save_actor(path)
            population=Population([Member(str(path),'reference','test',1.)])
            batch=parallel_selfplay.play(small_actor(),games=2,seed=222,collectors=2,inference_batch=8,
                opponents=[zoo.load_player(path,'cpu')],population=population,table_mix=(0.,1.,0.))
            self.assertTrue(((batch.opponent_seats>=0).sum(1)==3).all())
            self.assertTrue(batch.matchups)
            for game in range(2): self.assertEqual(len(torch.unique(batch.player_of[batch.game_of==game])),1)
        with self.assertRaises(ValueError): parallel_selfplay.merge([batch,copy.copy(batch)],222)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_identity_panels_are_level_and_count_deals_not_rotations(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);save_actor(root/'actor.pt')
            report=table_evaluate.evaluate(root/'actor.pt',[root/'actor.pt']*3,ledger=root/'seeds.json',
                games=2,lineups=[[0,1,2,3],[0,0,1,1]],want_controls=True)
            for row in report['panel']:
                self.assertEqual(row['independent_deals'],2);self.assertEqual(row['games_total'],8)
                self.assertEqual(row['mean_placement'],2.5);self.assertEqual(row['standard_error'],0.)
                np.testing.assert_allclose(row['placement_distribution'],[.25]*4)
                self.assertEqual(np.asarray(row['control_features']).shape,(2,140))
                self.assertGreater(row['diagnostics']['discards'],0)
            self.assertFalse(report['promotion'])

    def test_event_diagnostics_deduplicate_double_ron_and_define_pressure(self):
        d=table_evaluate.Diagnostics(1)
        events=[{'type':'start_kyoku'}, {'type':'reach_accepted','actor':1},
                {'type':'dahai','actor':0}, {'type':'hora','actor':1,'target':0},
                {'type':'hora','actor':2,'target':0}]
        d.feed([[json.dumps(e) for e in events]])
        self.assertEqual(d.dealins[0,0],1);self.assertEqual(d.pressure_dealins[0,0],1)
        self.assertEqual(d.pressure_discards[0,0],1)
        c=np.zeros((1,16),np.int64);c[0,0]=1;c[0,1]=3;c[0,8:12]=[40000,30000,20000,10000]
        d.before(c,np.array([],dtype=int),np.array([],dtype=int),np.zeros((1,78),bool),np.array([0]))
        self.assertTrue(d.late_leading[0,0]);self.assertTrue(d.late_trailing[0,3])

    def test_chance_correction_has_zero_expectation_and_small_deck_pair_mean_is_exact(self):
        from itertools import combinations
        probabilities=np.array([.1,.2,.7]);v=np.array([-2.,4.,1.])
        c=np.array([chance_control.finite_correction(probabilities,v,i) for i in range(3)])
        self.assertAlmostEqual(float(probabilities@c),0.,places=14)
        pair_counts=[]
        for pick in combinations(range(6),3):
            counts=np.bincount(np.array(pick)//2,minlength=3)
            pair_counts.append((counts*(counts-1)/2).sum())
        self.assertAlmostEqual(np.mean(pair_counts),chance_control.pair_expectation(3,2,3))
        with self.assertRaises(ValueError):chance_control.finite_correction([.3,.3],[1,2],0)

    def test_initial_counts_are_pre_draw_and_unchanged_by_feature_reads(self):
        arena=riichi_py.Arena(games=3,seed=42)
        mask=arena.legal_mask();f=chance_control.initial_features(arena)
        self.assertEqual(f.shape,(3,140));self.assertEqual(arena.legal_mask(),mask)
        counts=f[:,:136].reshape(3,4,34)+13/34
        np.testing.assert_allclose(counts.sum(2),13.)
        arena.step(np.frombuffer(arena.teacher(),np.uint8).tolist())
        with self.assertRaises(ValueError):chance_control.initial_features(arena)

    def test_pilot_coefficients_are_frozen_disjoint_and_never_promotion_evidence(self):
        rng=np.random.default_rng(1);n=250;x=rng.normal(size=(n,140));y=np.tanh(x[:,0])
        row={'lineup':[0,1,2,3],'control_features':x.tolist(),'by_deal_utility':y.tolist()}
        pilot={'table_panel_version':1,'domain':'validation','policies':[{'sha256':str(i)*64} for i in range(4)],
               'panel':[row],'seed_reservation':{'seed':1000,'end':1250}}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'pilot.json').write_text(json.dumps(pilot))
            saved=chance_control.fit(root/'pilot.json',root/'control.json',ridge=10.)
            with self.assertRaisesRegex(ValueError,'overlap'):chance_control.apply(pilot,saved)
            test=copy.deepcopy(pilot);test['seed_reservation']={'seed':2000,'end':2250};test['domain']='test'
            x2=rng.normal(size=(n,140));test['panel'][0]['control_features']=x2.tolist()
            test['panel'][0]['by_deal_utility']=np.tanh(x2[:,0]).tolist()
            result=chance_control.apply(test,saved)
            self.assertLess(result['panels'][0]['variance_ratio'],1.)
            self.assertFalse(result['promotion']);self.assertTrue(result['diagnostic_only'])
            self.assertGreater(result['panels'][0]['conservative_range'][1],1.5)
            test['policies'][0]['sha256']='changed'
            with self.assertRaises(ValueError):chance_control.apply(test,saved)
            with self.assertRaises(ValueError):
                (root/'test.json').write_text(json.dumps(test));chance_control.fit(root/'test.json',root/'bad.json')

    def test_bad_panel_and_partial_games_never_produce_evidence(self):
        with self.assertRaises(ValueError):table_evaluate.evaluate('x',['y'],ledger='unused',lineups=[],games=2)
        with self.assertRaises(ValueError):table_evaluate.evaluate('x',['y'],ledger='unused',lineups=[[0,2,1,1]],games=2)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'actor.pt';save_actor(path);player=zoo.load_player(path,'cpu')
            with self.assertRaises(RuntimeError):table_evaluate.table([player],[0,0,0,0],games=1,seed=1,max_steps=1)


if __name__=='__main__':unittest.main()
