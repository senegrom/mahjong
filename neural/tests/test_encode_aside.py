"""The seated others' views are made by a worker while the learner decides
(`Views.prepare(..., aside=True)`): the encoder lets go of the interpreter
as it works, so the learner's forward runs on the card meanwhile. Each
player is served an encoding of its own, its own arrays rather than rows
gathered from everyone's, and nothing changes: the same planes to the bit,
the same moves, the same round. Nothing tells the follower anything, feeds
it or encodes from it on the caller's thread while a worker encodes from
it: it would refuse the first two, and the third would only queue."""

from concurrent.futures import Future, ThreadPoolExecutor
import contextlib
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import riichi_py
import torch

from neural import observe, ppo_loop, selfplay, zoo
from neural.observe import Views
from neural.tests.test_baseline_from_play import varied
from neural.tests.test_bfloat16_players import checkpoints, fusion
from neural.tests.test_preview_reach import Steered
from neural.tests.test_step_planes import assert_same, deciding, play_on


class Inline:
    """Makes each encoding at once, on the caller's thread: a round as it
    is played without the worker."""

    def submit(self, work, *args):
        made = Future()
        made.set_result(work(*args))
        return made


class Watched:
    """The encoder's thread, slowed, counting the encodings handed to it
    that it has not yet finished: a test of what the caller does in the
    meantime that does not hang on how the threads happen to be scheduled.
    `while_pending` counts the calls to `mark` made while one was."""

    def __init__(self, pause: float = 0.02) -> None:
        self.thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="watched")
        self.pause = pause
        self.pending = 0
        self.handed = 0
        self.lock = threading.Lock()
        self.marks: list[bool] = []

    def submit(self, work, *args):
        with self.lock:
            self.pending += 1
            self.handed += 1

        def slowly():
            try:
                time.sleep(self.pause)
                return work(*args)
            finally:
                with self.lock:
                    self.pending -= 1

        return self.thread.submit(slowly)

    def mark(self) -> None:
        self.marks.append(self.pending > 0)


class Guarded:
    """The real follower, noting whatever tells it, feeds it or encodes
    from it on the main thread while the worker has an encoding to finish."""

    def __init__(self, follower, worker: Watched) -> None:
        self.follower = follower
        self.worker = worker
        self.clashes: list[str] = []

    def check(self, what: str) -> None:
        if threading.current_thread() is threading.main_thread() and self.worker.pending:
            self.clashes.append(what)

    def tell(self, *args):
        self.check("tell")
        return self.follower.tell(*args)

    def feed(self, *args):
        self.check("feed")
        return self.follower.feed(*args)

    def encode(self, *args, **kwargs):
        self.check("encode")
        return self.follower.encode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.follower, name)


#: Mortal's move for each of ours: the one it is a meaning of.
MORTAL_OF = np.isfinite(zoo.PRIORITY).argmax(axis=0)


class SteeredMortal(zoo.MortalPlayer):
    """A seated Mortal pushed, in each step's first question, towards the
    heuristic player's move and towards a reach wherever one is open, so it
    reaches often; each reach's tile is its network's own. Counts the rows
    it is asked a second time, after telling, and the rows asked ahead."""

    asked_again = asked_ahead = 0

    def _ask(self, views, who, fresh, allowed, ahead=()):
        values, own = super()._ask(views, who, fresh, allowed, ahead)
        if fresh:
            self.asked_again += len(who)
        else:
            self.asked_ahead += len(ahead)
            teacher = np.frombuffer(views.arena.teacher(), dtype=np.uint8)
            for row, (game, _player) in enumerate(who):
                values[row, MORTAL_OF[int(teacher[game])]] += 100.0
                values[row, zoo.MORTAL_RIICHI] += 200.0 * allowed[row, zoo.MORTAL_RIICHI]
        return values, own


def ready_to_reach(arena, games: int, live: np.ndarray) -> np.ndarray:
    """Which of the step's rows may declare riichi."""
    legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, -1)
    return legal[live, zoo.RIICHI_DISCARD:zoo.TSUMO].any(axis=1)


class ViewsTests(unittest.TestCase):
    def test_views_made_aside_are_the_views_made_here(self):
        """Encoded by a worker or here, in one call or several, as they
        stand or as they would stand having declared: the same planes and
        masks to the bit. An ask for the very players of one encoding, in
        its order, is handed that encoding's own arrays; any other is
        gathered from it."""
        games = 8
        arena = riichi_py.Arena(games=games, seed=606)
        views = Views(arena, games)
        served = previewed = 0
        for _ in range(80):
            views.advance()
            live, players = deciding(arena, games)
            mine = np.arange(len(live)) % 2 == 0
            ready = ready_to_reach(arena, games, live)
            views.prepare(live[mine], players[mine])
            views.prepare(live[~mine], players[~mine], aside=True)
            views.prepare(live[ready], players[ready], after_reach=True, aside=True)
            for part in (mine, ~mine, np.ones(len(live), dtype=bool)):
                planes, masks = views.sparse_and_masks(live[part], players[part])
                want, want_masks = views.sparse_and_masks(live[part], players[part], fresh=True)
                assert_same(self, planes, want)
                np.testing.assert_array_equal(masks, want_masks)
                self.assertTrue(planes.trusted)
                served += int(part.sum())
            for part in (mine, ~mine):
                if part.any():
                    first, _masks = views.sparse_and_masks(live[part], players[part])
                    again, _masks = views.sparse_and_masks(live[part], players[part])
                    self.assertIs(first, again)
            if ready.any():
                planes, masks = views.sparse_and_masks(live[ready], players[ready], after_reach=True)
                want, want_masks = views.sparse_and_masks(live[ready], players[ready], fresh=True,
                                                          after_reach=True)
                assert_same(self, planes, want)
                np.testing.assert_array_equal(masks, want_masks)
                # Not the views as they stand: the reach is in them.
                stand, _masks = views.sparse_and_masks(live[ready], players[ready])
                self.assertFalse(np.array_equal(planes.indices, stand.indices)
                                 and np.array_equal(planes.values, stand.values))
                previewed += int(ready.sum())
            play_on(arena, games)
        self.assertGreater(served, 400)
        self.assertGreater(previewed, 0)

    def test_nothing_touches_the_follower_while_a_worker_encodes(self):
        """Feeding the table, telling a player and encoding afresh here all
        wait until the worker has finished: the follower is borrowed while
        an encoding is made, and the encoder's workers finish one batch
        before they take up another."""
        worker = Watched()
        games = 8
        arena = riichi_py.Arena(games=games, seed=707)
        views = Views(arena, games)
        follower = views.observer.follower = Guarded(views.observer.follower, worker)
        with patch.object(observe, "encoder_thread", return_value=worker):
            for _ in range(20):
                # Fed while last step's encodings may still be in the making.
                views.advance()
                live, players = deciding(arena, games)
                views.prepare(live, players, aside=True)
                views.sparse_and_masks(live[:1], players[:1], fresh=True)
                views.prepare(live, players, aside=True)
                play_on(arena, games)
            views.advance()
            live, players = deciding(arena, games)
            views.prepare(live, players, aside=True)
            views.tell(int(live[0]), int(players[0]),
                       json.dumps({"type": "reach", "actor": int(players[0])}))
        self.assertGreater(worker.handed, 40)
        self.assertEqual(follower.clashes, [])

    def test_a_workers_failure_is_raised_where_it_is_waited_for(self):
        games = 2
        views = Views(riichi_py.Arena(games=games, seed=1), games)
        views.advance()
        views.prepare(np.array([games + 5]), np.array([0]), aside=True)
        with self.assertRaises(ValueError):
            views.wait()


class RoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.folder = tempfile.TemporaryDirectory()
        cls.paths = checkpoints(Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        cls.folder.cleanup()

    def seated(self, **flags):
        """A Mortal and a joined player seated by player at half the
        places, with the Club bot, as the trainers seat them; the Mortal
        pushed towards reaches (see `SteeredMortal`)."""
        args = SimpleNamespace(opponents=[self.paths["mortal"], self.paths["fused"], "club"],
                               compile=False, seat_share=0.5, opponent_share=0.0, **flags)
        with contextlib.redirect_stdout(None):
            roster, seated = ppo_loop.seat_others(args, "cpu")
        steered = SteeredMortal(self.paths["mortal"], "cpu")
        steered.skip_forced, steered.preview_reach = seated[0].skip_forced, seated[0].preview_reach
        seated[0] = steered
        return roster, seated

    @staticmethod
    def learner():
        """A joined learner pushed towards reaches (see `Steered`)."""
        torch.manual_seed(31)
        net = varied(fusion(16, 2)).eval()
        return Steered(lambda planes, mask: net.decision(planes, mask)[:2])

    def test_the_learner_decides_while_the_others_are_encoded(self):
        """On every step with a seated other at it the learner decides
        while the worker is still making the others' views, and nothing
        touches the follower meanwhile, telling a reach included."""
        worker = Watched()
        made: list = []

        def views_of(*args, **kwargs):
            views = Views(*args, **kwargs)
            views.observer.follower = Guarded(views.observer.follower, worker)
            made.append(views)
            return views

        learner = self.learner()
        decide = learner.decide

        def decided(views, *args, **kwargs):
            worker.mark()
            return decide(views, *args, **kwargs)

        learner.decide = decided
        roster, seated = self.seated()
        torch.manual_seed(5)
        with patch.object(observe, "encoder_thread", return_value=worker), \
                patch.object(selfplay, "Views", side_effect=views_of):
            batch = selfplay.play(learner, games=4, seed=808, device="cpu", opponents=seated,
                                  seat_share=0.5, population=roster)
        self.assertGreater(int((batch.actions == zoo.MORTAL_RIICHI).sum()), 5)
        self.assertGreater(seated[0].asked_again, 5)
        self.assertGreater(worker.handed, 100)
        # Still encoding as the learner began, on a good share of its steps:
        # those with a seated other at a table.
        self.assertGreater(sum(worker.marks), 100)
        self.assertGreater(sum(worker.marks), len(worker.marks) // 4)
        self.assertEqual(made[0].observer.follower.clashes, [])

    def test_a_round_is_played_as_without_the_worker(self):
        """A joined learner with a Mortal and another joined player seated
        by player, played with the others' views made aside and made at
        once: every array of the round the same to the bit, by telling or
        asking ahead, with or without the forced rows. The learner and the
        Mortal reach often, so their reaches are told, or asked ahead,
        while views are being made."""
        for flags in ({}, {"preview_reach": True, "skip_forced": True}):
            rounds = []
            for inline in (False, True):
                learner = self.learner()
                roster, seated = self.seated(**flags)
                torch.manual_seed(5)
                with patch.object(observe, "encoder_thread", return_value=Inline()) if inline \
                        else contextlib.nullcontext():
                    rounds.append(selfplay.play(
                        learner, games=4, seed=808, device="cpu", opponents=seated, seat_share=0.5,
                        population=roster, explore_share=0.1,
                        preview_reach=flags.get("preview_reach", False)))
            aside, here = rounds
            with self.subTest(**flags):
                self.assertGreater(aside.decisions, 500)
                self.assertGreater(int((aside.actions == zoo.MORTAL_RIICHI).sum()), 5)
                self.assertGreater(seated[0].asked_ahead if flags else seated[0].asked_again, 5)
                for name in ("indptr", "indices", "values"):
                    np.testing.assert_array_equal(getattr(aside.observations, name).view(np.uint8),
                                                  getattr(here.observations, name).view(np.uint8))
                for field in ("legal", "actions", "explored", "returns", "log_probs", "values", "held"):
                    self.assertTrue(torch.equal(getattr(aside, field), getattr(here, field)), field)
                np.testing.assert_array_equal(aside.final_scores, here.final_scores)
                self.assertEqual(aside.matchups, here.matchups)


if __name__ == "__main__":
    unittest.main()
