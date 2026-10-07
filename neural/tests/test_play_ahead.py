"""Played ahead (`train_combined --play-ahead`), each round is played on a
worker while the round before it is learned, by the weights the learner had
before it learned that one; a process's first round, which nothing
precedes, by the learner as it stands. The round is exactly the one those
weights play, drawn from a generator of its own whatever the learning does
meanwhile. What stops a round is raised where it is taken, a round still
being played when the learning fails is given up, nothing is played past
the last generation, play's values are the baseline whichever weights
played, Mortal's vectors are learned from only where they are the
learner's Mortal's and kept only where they will be, and a run resumes
from what either way of playing wrote."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import combined, mortal_model, policy_inference, ppo_loop, selfplay, train_combined, zoo
from neural.observe import Planes
from neural.tests import test_mixed_tables
from neural.tests.test_baseline_from_play import varied
from neural.tests.test_bfloat16_players import fusion
from neural.tests.test_preview_reach import Steered
from neural.training_state import round_seed

CUDA = torch.cuda.is_available()
SEED = 5
MEASURED = {"placement": 2.5, "score": 0.0, "wins": 0.25}


def generation_of(seed: int) -> int:
    """The generation a round's deal seed belongs to (see `round_seed`)."""
    return (seed - SEED) // (1 << 32) - 1


def fake_round(values=None, phi: bool = False):
    """A round of eight decisions with known returns, and the values and
    vectors play kept when given."""
    n = 8
    planes = Planes(np.arange(n + 1, dtype=np.int64), np.zeros(n, dtype=np.uint16),
                    np.ones(n, dtype=np.float16))
    return SimpleNamespace(
        decisions=n, observations=planes, legal=torch.ones(n, 46, dtype=torch.bool),
        actions=torch.zeros(n, dtype=torch.int64), returns=torch.linspace(-1, 1, n),
        log_probs=torch.zeros(n), held=torch.full((n, 3, 34), 1 / 34),
        games=1, hands=1, timing={}, values=values,
        phi=torch.ones(n, 1024) if phi else None)


class Rounds:
    """Stands in for `selfplay.play` in a trainer: a fake round each call,
    with what it was asked, on which thread, and the joined player's head
    as the player held it then. `then(generation, options)` may raise, or
    wait, before the round is handed back."""

    def __init__(self, values=None, then=None):
        self.values = values
        self.then = then
        self.calls = []

    def __call__(self, player, **options):
        generation = generation_of(options["seed"])
        self.calls.append(SimpleNamespace(
            generation=generation, thread=threading.current_thread().name,
            head={name: value.detach().clone() for name, value in player.fuse.state_dict().items()},
            keep_phi=player.keep_phi, own_draws=options.get("own_draws", False),
            abandon=options.get("abandon")))
        if self.then is not None:
            self.then(generation, options)
        return fake_round(self.values, phi=player.keep_phi)


def same(one: dict, two: dict) -> bool:
    return one.keys() == two.keys() and all(torch.equal(one[name], two[name]) for name in one)


class Trainer:
    """`train_combined.main` on the processor over stand-in rounds, in a
    folder of its own: the generations' records, and the joined player's
    head as each checkpoint saved it, by the generation it starts."""

    def __init__(self, test: unittest.TestCase) -> None:
        folder = tempfile.TemporaryDirectory()
        test.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.origin = self.root / "source.pt"
        torch.manual_seed(3)
        torch.save(fusion(8, 1).state(), self.origin)
        self.start = torch.load(self.origin, map_location="cpu", weights_only=False)["combined"]

    def run(self, flags, rounds=3, play=None, resume=None, out="run", patches=()):
        torch.set_num_threads(1)
        out = self.root / out
        argv = ["trainer", "--resume", str(resume or self.origin), "--out", str(out),
                "--rounds", str(rounds), "--batch", "4", "--epochs", "1", "--games", "1",
                "--measure-games", "1", "--seed", str(SEED), *flags]
        if "--fixed" not in flags:
            argv += ["--fixed", "none"]
        heads = {}
        save = ppo_loop.atomic_save

        def saving(payload, path):
            if Path(path).name == "latest.pt":
                heads[payload["generation"]] = {
                    name: value.detach().clone() for name, value in payload["combined"].items()}
            return save(payload, path)

        with contextlib.ExitStack() as stack:
            for patched in (
                patch.object(sys, "argv", argv),
                patch.object(torch.cuda, "is_available", return_value=False),
                patch.object(train_combined.selfplay, "play", side_effect=play or Rounds()),
                patch.object(train_combined.selfplay, "measure", return_value=MEASURED),
                patch.object(ppo_loop, "atomic_save", side_effect=saving),
                *patches,
            ):
                stack.enter_context(patched)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:
                train_combined.main()
            finally:
                self.records = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()] \
                    if (out / "log.jsonl").exists() else []
                self.heads = heads
        return self.records


class SamplingTests(unittest.TestCase):
    def test_a_generator_of_its_own_draws_what_torchs_would(self):
        """From the same seed, the same moves: `sampled` is
        `Categorical.sample` with the generator named, on either device;
        without one it is `sample` itself, and draws from torch's."""
        for device in ("cpu", "cuda") if CUDA else ("cpu",):
            with self.subTest(device=device):
                logits = torch.randn(300, 46, device=device)
                logits[:, 5] = -torch.inf
                distribution = torch.distributions.Categorical(logits=logits)
                torch.manual_seed(7)
                want = distribution.sample()
                generator = torch.Generator(device=device).manual_seed(7)
                got = selfplay.sampled(distribution, generator)
                self.assertTrue(torch.equal(want, got))
                self.assertEqual(got.shape, (300,))
                torch.manual_seed(7)
                self.assertTrue(torch.equal(selfplay.sampled(distribution), want))

    def test_a_round_with_draws_of_its_own_leaves_torchs_alone(self):
        """A round that draws its moves apart is the same round whatever
        torch's generator holds, and leaves it as it found it."""
        torch.set_num_threads(2)
        torch.manual_seed(1)
        net = fusion(8, 1)
        rounds = []
        for seed in (100, 200):
            torch.manual_seed(seed)
            before = torch.get_rng_state().clone()
            rounds.append(selfplay.play(net, games=1, seed=11, device="cpu", explore_share=0.1,
                                        own_draws=True))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
        one, two = rounds
        self.assertGreater(one.decisions, 100)
        self.assertGreater(int(one.explored.sum()), 0)
        for name in ("actions", "log_probs", "returns", "explored", "values"):
            self.assertTrue(torch.equal(getattr(one, name), getattr(two, name)), name)

    def test_a_round_reaching_often_draws_its_tiles_apart_too(self):
        """Steered to reach often (see `test_preview_reach.Steered`), the
        reaches' tiles come from the round's generator as well, by telling
        and by asking ahead: the same round whatever torch's generator
        holds, and torch's left alone."""
        torch.set_num_threads(2)
        torch.manual_seed(31)
        joined = varied(fusion(16, 2)).eval()
        for preview in (False, True):
            rounds = []
            for seed in (100, 200):
                torch.manual_seed(seed)
                before = torch.get_rng_state().clone()
                learner = Steered(lambda planes, mask: joined.decision(planes, mask)[:2])
                rounds.append(selfplay.play(learner, games=4, seed=41, device="cpu", explore_share=0.05,
                                            preview_reach=preview, own_draws=True))
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
            one, two = rounds
            with self.subTest(preview_reach=preview):
                self.assertGreater(int((one.actions == zoo.MORTAL_RIICHI).sum()), 20)
                for name in ("actions", "log_probs", "returns", "explored", "values"):
                    self.assertTrue(torch.equal(getattr(one, name), getattr(two, name)), name)

    def test_an_abandoned_round_stops_at_its_next_step(self):
        torch.manual_seed(1)
        abandon = threading.Event()
        abandon.set()
        with self.assertRaises(selfplay.Abandoned):
            selfplay.play(fusion(8, 1), games=1, seed=11, device="cpu", abandon=abandon)


class PlayAheadTests(unittest.TestCase):
    """The worker itself, with a stand-in for playing."""

    def setUp(self):
        torch.manual_seed(2)
        self.learner = torch.nn.Linear(3, 2)
        self.actor = copy.deepcopy(self.learner).requires_grad_(False)

    def test_a_round_is_played_by_the_weights_the_learner_had_as_it_began(self):
        go = threading.Event()
        seen = []

        def play(player, generation, **options):
            # Read only once the learner has moved on.
            self.assertTrue(go.wait(10))
            seen.append((generation, player.weight.clone(), player.keep_phi, options,
                         threading.current_thread().name))
            return "round"

        ahead = ppo_loop.PlayAhead(self.learner, self.actor, "cpu", play)
        self.addCleanup(ahead.close)
        weight = self.learner.weight.detach().clone()
        ahead.start(4, keep_phi=True)
        self.assertEqual(ahead.playing, 4)
        with torch.no_grad():
            self.learner.weight.add_(1.0)
        go.set()
        batch, played, waited = ahead.take(4)
        self.assertEqual(batch, "round")
        self.assertIsNone(ahead.playing)
        self.assertGreaterEqual(played, 0.0)
        self.assertGreaterEqual(waited, 0.0)
        ((generation, held, kept, options, thread),) = seen
        self.assertEqual(generation, 4)
        self.assertTrue(torch.equal(held, weight))
        self.assertTrue(kept)
        self.assertIs(options["own_draws"], True)
        self.assertIs(options["abandon"], ahead.abandon)
        self.assertTrue(thread.startswith("play-ahead"), thread)
        # The next round takes the weights as they are by then.
        ahead.start(5)
        ahead.take(5)
        self.assertTrue(torch.equal(seen[-1][1], self.learner.weight.detach()))
        self.assertFalse(seen[-1][2])

    def test_what_stops_a_round_is_raised_where_it_is_taken(self):
        def play(_player, generation, **_options):
            raise ValueError(f"round {generation} broke")

        ahead = ppo_loop.PlayAhead(self.learner, self.actor, "cpu", play)
        self.addCleanup(ahead.close)
        ahead.start(2)
        with self.assertRaisesRegex(ValueError, "round 2 broke"):
            ahead.take(2)
        self.assertIsNone(ahead.playing)
        # And the worker plays on, for a trainer that would.
        ahead.play = lambda _player, generation, **_options: generation
        ahead.start(3)
        self.assertEqual(ahead.take(3)[0], 3)

    def test_one_round_at_a_time_and_only_the_one_begun_is_taken(self):
        release = threading.Event()
        ahead = ppo_loop.PlayAhead(
            self.learner, self.actor, "cpu", lambda *_args, **_options: release.wait(10))
        self.addCleanup(ahead.close)
        self.addCleanup(release.set)
        ahead.start(1)
        with self.assertRaises(RuntimeError):
            ahead.start(2)
        with self.assertRaises(RuntimeError):
            ahead.take(2)
        release.set()
        self.assertIs(ahead.take(1)[0], True)
        with self.assertRaises(RuntimeError):
            ahead.take(1)

    def test_closing_gives_up_a_round_still_being_played(self):
        stopped = []

        def play(_player, _generation, abandon, **_options):
            # As `selfplay.play` does, at the start of each step.
            for _step in range(1000):
                if abandon.is_set():
                    stopped.append(True)
                    raise selfplay.Abandoned("as asked")
                time.sleep(0.01)
            return "finished"

        ahead = ppo_loop.PlayAhead(self.learner, self.actor, "cpu", play)
        ahead.start(1)
        began = time.time()
        ahead.close()
        self.assertLess(time.time() - began, 5.0)
        self.assertEqual(stopped, [True])
        self.assertFalse(any(thread.name.startswith("play-ahead") for thread in threading.enumerate()))

    def test_closing_gives_up_a_round_no_longer_held(self):
        """`take` lets go of the round before it waits for it; a wait
        interrupted there leaves the round playing, and closing gives it up
        all the same."""
        stopped = []

        def play(_player, _generation, abandon, **_options):
            for _step in range(1000):
                if abandon.is_set():
                    stopped.append(True)
                    raise selfplay.Abandoned("as asked")
                time.sleep(0.01)
            return "finished"

        ahead = ppo_loop.PlayAhead(self.learner, self.actor, "cpu", play)
        ahead.start(1)
        # As `take` leaves it the moment before it waits.
        ahead.round = None
        began = time.time()
        ahead.close()
        self.assertLess(time.time() - began, 5.0)
        self.assertEqual(stopped, [True])

    @unittest.skipUnless(CUDA, "streams are the card's")
    def test_on_the_card_a_round_is_played_on_a_stream_of_its_own(self):
        streams = []
        learner = self.learner.cuda()
        actor = copy.deepcopy(learner).requires_grad_(False)

        def play(player, _generation, **_options):
            streams.append(torch.cuda.current_stream())
            return player.weight.sum().item()

        ahead = ppo_loop.PlayAhead(learner, actor, "cuda", play)
        self.addCleanup(ahead.close)
        ahead.start(0)
        total, _played, _waited = ahead.take(0)
        self.assertEqual(total, learner.weight.sum().item())
        (stream,) = streams
        self.assertEqual(stream, ahead.stream)
        self.assertNotEqual(stream, torch.cuda.current_stream())
        self.assertLess(stream.priority, torch.cuda.current_stream().priority)


class TrainerTests(unittest.TestCase):
    def test_each_round_is_played_ahead_by_the_weights_from_before_and_none_past_the_last(self):
        trainer = Trainer(self)
        rounds = Rounds()
        records = trainer.run(["--play-ahead"], rounds=3, play=rounds)
        self.assertEqual([call.generation for call in rounds.calls], [0, 1, 2])
        self.assertEqual([record["played_ahead"] for record in records], [False, True, True])
        for call in rounds.calls:
            self.assertTrue(call.thread.startswith("play-ahead"), call.thread)
            self.assertIs(call.own_draws, True)
            self.assertIsInstance(call.abandon, threading.Event)
        for record in records:
            self.assertGreaterEqual(record["play_wait_seconds"], 0.0)
            self.assertIs(record["resident"], False)
        heads = trainer.heads
        self.assertEqual(sorted(heads), [1, 2, 3])
        # The first round by the learner as it began, the second by the
        # same weights, played beside the first's learning, and the third
        # by the weights the first round's learning left.
        self.assertTrue(same(rounds.calls[0].head, trainer.start))
        self.assertTrue(same(rounds.calls[1].head, trainer.start))
        self.assertTrue(same(rounds.calls[2].head, heads[1]))
        self.assertFalse(same(heads[1], trainer.start))
        self.assertFalse(any(thread.name.startswith("play-ahead") for thread in threading.enumerate()))

    def test_without_the_switch_each_round_is_played_here_by_the_learner_as_it_stands(self):
        trainer = Trainer(self)
        rounds = Rounds()
        records = trainer.run([], rounds=3, play=rounds)
        self.assertEqual([call.generation for call in rounds.calls], [0, 1, 2])
        for call in rounds.calls:
            self.assertEqual(call.thread, threading.current_thread().name)
            self.assertIs(call.own_draws, False)
            self.assertIsNone(call.abandon)
        for record in records:
            self.assertNotIn("played_ahead", record)
            self.assertNotIn("play_wait_seconds", record)
        self.assertTrue(same(rounds.calls[0].head, trainer.start))
        self.assertTrue(same(rounds.calls[1].head, trainer.heads[1]))
        self.assertTrue(same(rounds.calls[2].head, trainer.heads[2]))

    def test_what_stops_a_round_played_ahead_stops_the_trainer_where_it_is_taken(self):
        def then(generation, _options):
            if generation == 2:
                raise ValueError("the round of generation 2 broke")

        trainer = Trainer(self)
        with self.assertRaisesRegex(ValueError, "generation 2 broke"):
            trainer.run(["--play-ahead"], rounds=4, play=Rounds(then=then))
        # The two generations before it were learned and saved.
        self.assertEqual([record["generation"] for record in trainer.records], [0, 1])
        self.assertEqual(sorted(trainer.heads), [1, 2])
        self.assertFalse(any(thread.name.startswith("play-ahead") for thread in threading.enumerate()))

    def test_a_learning_that_fails_gives_up_the_round_being_played_beside_it(self):
        waited = []

        def then(generation, options):
            if generation == 2:
                # A round as long as it likes, unless it is given up.
                waited.append(options["abandon"].wait(60))
                raise selfplay.Abandoned("as asked")

        updates = []

        def require_updates(steps):
            updates.append(steps)
            if len(updates) == 2:
                raise RuntimeError("the learning of generation 1 failed")

        trainer = Trainer(self)
        began = time.time()
        with self.assertRaisesRegex(RuntimeError, "generation 1 failed"):
            trainer.run(["--play-ahead"], rounds=4, play=Rounds(then=then),
                        patches=[patch.object(train_combined, "require_updates", side_effect=require_updates)])
        self.assertLess(time.time() - began, 50)
        self.assertEqual(waited, [True])
        self.assertEqual([record["generation"] for record in trainer.records], [0])
        self.assertFalse(any(thread.name.startswith("play-ahead") for thread in threading.enumerate()))

    def test_plays_values_are_the_baseline_whichever_weights_played_the_round(self):
        """The first round's values are the learner's own as it stands;
        the second's, played ahead, those of the weights from before.
        Either is the baseline, with no pass, as without the switch."""
        values = torch.linspace(-0.5, 0.5, 8)
        returns = torch.linspace(-1, 1, 8)
        passes = []
        real = ppo_loop.baseline

        def baseline(*args, **kwargs):
            guesses = real(*args, **kwargs)
            passes.append(guesses[0].clone())
            return guesses

        counted = patch.object(ppo_loop, "baseline", side_effect=baseline)
        records = Trainer(self).run(["--play-ahead", "--baseline-from-play"], rounds=3,
                                    play=Rounds(values=values), patches=[counted])
        self.assertEqual([record["played_ahead"] for record in records], [False, True, True])
        self.assertEqual(passes, [])
        for record in records:
            self.assertIs(record["baseline_from_play"], True)
            self.assertAlmostEqual(record["value_error"], round(float(((returns - values) ** 2).mean()), 4))
        # Checked, the pass is made for each as well, and how far play's
        # values were from it is said.
        records = Trainer(self).run(["--play-ahead", "--baseline-from-play", "--check-baseline"], rounds=2,
                                    play=Rounds(values=values), patches=[counted])
        self.assertEqual(len(passes), 2)
        for record, passed in zip(records, passes):
            self.assertIs(record["baseline_from_play"], True)
            self.assertAlmostEqual(record["baseline_difference"], round(float((values - passed).abs().max()), 6))
        # Without the switch, the pass, for a round played ahead too.
        passes.clear()
        records = Trainer(self).run(["--play-ahead"], rounds=2, play=Rounds(values=values), patches=[counted])
        self.assertEqual(len(passes), 2)
        for record, passed in zip(records, passes):
            self.assertNotIn("baseline_from_play", record)
            self.assertAlmostEqual(record["value_error"], round(float(((returns - passed) ** 2).mean()), 4))

    def test_mortals_vectors_are_learned_from_only_where_nothing_moved_mortal_since_the_round_began(self):
        """Modes none, mortal, mortal, ours, mortal, mortal+head. The first
        round is the learner's own in a generation that moves Mortal, so
        its vectors are not kept. The second was begun before the first
        generation's learning, which moved Mortal; the fifth before the
        fourth's, likewise: neither keeps its vectors, though each is
        learned in a generation that holds Mortal still. The third and the
        sixth were begun in generations that held Mortal still and are
        learned in such generations, so theirs are kept and learned from.
        The fourth was begun in such a generation too, but its own trains
        Mortal, so it would never read them, and they are not kept."""
        class Scripted:
            def __init__(self, modes):
                self.modes = list(modes)
                self.bit_generator = np.random.default_rng(0).bit_generator

            def choice(self, _modes):
                return self.modes.pop(0)

        given = []
        everything = combined.Combined.everything

        def watched(self, planes, legal, phi=None):
            if torch.is_grad_enabled():
                given.append(phi is not None)
            return everything(self, planes, legal, phi)

        spares = []

        def kept_on_card(tensor, device, spare=None):
            spares.append(spare)
            return tensor

        resident = []
        rounds = Rounds()
        modes = ["none", "mortal", "mortal", "ours", "mortal", "mortal+head"]
        records = Trainer(self).run(
            ["--play-ahead", "--reuse-phi", "--fixed", "none", "mortal", "ours", "mortal+head"], rounds=6,
            play=rounds,
            patches=[
                patch.object(train_combined, "restore_random_state", return_value=Scripted(modes)),
                patch.object(combined.Combined, "everything", watched),
                patch.object(ppo_loop, "kept_on_card", side_effect=kept_on_card),
                patch.object(ppo_loop, "resident", side_effect=lambda *args: resident.append(args)),
            ])
        # Each generation's mode is the one drawn for it: looking at the next
        # took nothing from the drawer.
        self.assertEqual([record["fixed"] for record in records], modes)
        self.assertEqual([call.keep_phi for call in rounds.calls], [False, False, True, False, False, True])
        # Two minibatches a generation.
        self.assertEqual(given, [False, False, False, False, True, True, False, False, False, False, True, True])
        self.assertEqual([record["reused_phi"] for record in records], [False, False, True, False, False, True])
        # On the card only with the round's room to spare too; the planes
        # are never put there beside a round being played.
        self.assertEqual(spares, [ppo_loop.step_spare() + (ppo_loop.AHEAD_SPARE_GB << 30)] * 2)
        self.assertEqual(resident, [])

    def test_a_run_resumes_from_what_either_way_of_playing_wrote(self):
        trainer = Trainer(self)
        trainer.run(["--play-ahead"], rounds=2, out="ahead")
        written = trainer.root / "ahead" / "latest.pt"
        saved = torch.load(written, map_location="cpu", weights_only=False)
        self.assertEqual(saved["generation"], 2)
        for flags in (["--play-ahead"], []):
            with self.subTest(flags=flags):
                rounds = Rounds()
                (record,) = trainer.run(flags, rounds=1, play=rounds, resume=written,
                                        out="resumed" + "".join(flags))
                self.assertEqual(record["generation"], 2)
                (call,) = rounds.calls
                self.assertEqual(call.generation, 2)
                # The first round after a resume is the learner's own, as it stands.
                self.assertTrue(same(call.head, saved["combined"]))
                if flags:
                    self.assertIs(record["played_ahead"], False)
        trainer.run([], rounds=1, out="plain")
        rounds = Rounds()
        (record,) = trainer.run(["--play-ahead"], rounds=1, play=rounds,
                                resume=trainer.root / "plain" / "latest.pt", out="then-ahead")
        self.assertEqual((record["generation"], record["played_ahead"]), (1, False))

    def test_refused_with_the_compiled_player_and_off_unless_asked(self):
        with patch.object(sys, "argv", ["trainer"]):
            self.assertFalse(train_combined.parse_args().play_ahead)
        with patch.object(sys, "argv", ["trainer", "--play-ahead", "--compile-learning"]):
            self.assertTrue(train_combined.parse_args().play_ahead)
        with patch.object(sys, "argv", ["trainer", "--play-ahead", "--compile"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_combined.parse_args()

    def test_the_cloud_passes_it_on_only_when_asked_and_never_with_the_compiled_player(self):
        launch = test_mixed_tables.CloudLaunchTests.launch
        self.assertNotIn("--play-ahead", launch(self, "train_combined"))
        command = launch(self, "train_combined", play_ahead=True, compile=False)
        self.assertIn("--play-ahead", command)
        self.assertNotIn("--compile", command)
        command = launch(self, "train_combined", play_ahead=True, compile_learning=True)
        self.assertIn("--play-ahead", command)
        self.assertIn("--compile-learning", command)
        with self.assertRaisesRegex(ValueError, "play_ahead"):
            launch(self, "train_combined", play_ahead=True)


class RealRoundTests(unittest.TestCase):
    """Real rounds on the processor: a small joined player, a Mortal seated
    by player, whose views are encoded aside while the learner decides,
    exploration on, and a measurement made beside a round."""

    CONFIG = {"resnet": {"conv_channels": 8, "num_blocks": 1}, "control": {"version": 4}}

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        torch.manual_seed(13)
        self.origin = self.root / "source.pt"
        torch.save(fusion(8, 1).state(), self.origin)
        self.other = self.root / "mortal.pt"
        torch.save({**mortal_model.build(8, 1).state(), "config": self.CONFIG}, self.other)

    def train(self, flags, rounds, resume=None, out="run", measure_every=2):
        """The trainer's rounds by generation, as played, and its
        checkpoints' payloads by the generation they start."""
        played, payloads = {}, {}
        play, save = selfplay.play, ppo_loop.atomic_save

        def playing(player, **options):
            batch = play(player, **options)
            played[generation_of(options["seed"])] = batch
            return batch

        def saving(payload, path):
            if Path(path).name == "latest.pt":
                payloads[payload["generation"]] = copy.deepcopy(payload)
            return save(payload, path)

        argv = ["trainer", "--resume", str(resume or self.origin), "--out", str(self.root / out),
                "--rounds", str(rounds), "--games", "1", "--batch", "64", "--epochs", "1",
                "--fixed", "none", "--explore", "0.1", "--opponents", str(self.other),
                "--seat-share", "0.5", "--measure-every", str(measure_every), "--measure-games", "1",
                "--seed", str(SEED), *flags]
        with patch.object(sys, "argv", argv), \
                patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(train_combined.selfplay, "play", side_effect=playing), \
                patch.object(ppo_loop, "atomic_save", side_effect=saving), \
                contextlib.redirect_stdout(io.StringIO()):
            train_combined.main()
        records = [json.loads(line) for line in (self.root / out / "log.jsonl").read_text().splitlines()]
        return played, payloads, records

    def replay(self, payload, generation):
        """Round `generation` as the weights in `payload` play it alone, here
        on this thread, drawing apart as a round played ahead does."""
        path = self.root / f"weights-{generation}.pt"
        torch.save(payload, path)
        net, _state = combined.load(path, "cpu")
        args = SimpleNamespace(opponents=[self.other], compile=False, seat_share=0.5, opponent_share=0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            roster, seated = ppo_loop.seat_others(args, "cpu")
            return selfplay.play(net, games=1, seed=round_seed(SEED, generation, 1), device="cpu",
                                 opponents=seated, seat_share=0.5, population=roster,
                                 explore_share=0.1, own_draws=True)

    def assertSameRound(self, got, want):
        for name in ("actions", "log_probs", "returns", "legal", "explored", "values", "held"):
            self.assertTrue(torch.equal(getattr(got, name), getattr(want, name)), name)
        for name in ("indptr", "indices", "values"):
            self.assertTrue(np.array_equal(getattr(got.observations, name), getattr(want.observations, name)),
                            name)
        self.assertEqual(got.matchups, want.matchups)

    def test_each_round_is_the_one_the_weights_it_was_given_play(self):
        torch.set_num_threads(2)
        played, payloads, records = self.train(["--play-ahead"], rounds=3)
        self.assertEqual([record["played_ahead"] for record in records], [False, True, True])
        # The seated Mortal sat, its views encoded aside while the learner decided.
        for record in records:
            self.assertTrue(any(row["role"] != "self" for row in record["matchups"]), record["matchups"])
        start = torch.load(self.origin, map_location="cpu", weights_only=False)
        self.assertSameRound(played[0], self.replay(start, 0))
        self.assertSameRound(played[1], self.replay(start, 1))
        self.assertSameRound(played[2], self.replay(payloads[1], 2))
        # Measured beside the second and third rounds, with the weights the
        # first and second generations left: as measured alone.
        for generation in (0, 1):
            torch.save(payloads[generation + 1], self.root / f"after-{generation}.pt")
            net, _state = combined.load(self.root / f"after-{generation}.pt", "cpu")
            alone = selfplay.measure(net, games=1, seed=7_000_000 + generation, device="cpu")
            self.assertEqual(records[generation]["placement"], round(alone["placement"], 3))
            self.assertEqual(records[generation]["score"], round(alone["score"], 1))

    def test_a_resumed_run_repeats_itself(self):
        """From a checkpoint written while a round was being played beside
        the learning, the same round and the same learning, however often
        it is resumed: the round draws from a generator of its own, and the
        learning from the generators the checkpoint kept, which the round
        being played as it was written had not touched."""
        torch.set_num_threads(2)
        _played, payloads, _records = self.train(["--play-ahead"], rounds=2, out="first", measure_every=100)
        written = self.root / "written-beside-a-round.pt"
        torch.save(payloads[1], written)
        runs = []
        for out in ("again", "once-more"):
            played, payloads, records = self.train(["--play-ahead"], rounds=1, resume=written, out=out,
                                                   measure_every=100)
            runs.append((played[1], payloads[2], records))
        (one_played, one, one_records), (two_played, two, _two_records) = runs
        self.assertSameRound(one_played, two_played)
        self.assertEqual([record["played_ahead"] for record in one_records], [False])
        for part in ("combined", "model", "mortal", "current_dqn"):
            for name, value in one[part].items():
                self.assertTrue(torch.equal(value, two[part][name]), f"{part}.{name}")

    @unittest.skipUnless(CUDA, "the card's streams and mixed precision")
    def test_on_the_card_a_round_played_ahead_is_the_round_played_here(self):
        """Played on the worker on a stream of its own by the learner's
        copy with its weights cast once, while this thread keeps the card
        busy: the round played here on the learner's stream, bit for bit."""
        torch.manual_seed(17)
        net = fusion(8, 1).cuda().eval()
        args = SimpleNamespace(opponents=[self.other, "club"], compile=False, seat_share=0.5,
                               opponent_share=0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            roster, seated = ppo_loop.seat_others(args, "cuda")

        def play(player, generation, **options):
            return selfplay.play(player, games=1, seed=round_seed(SEED, generation, 1), device="cuda",
                                 amp=True, opponents=seated, seat_share=0.5, population=roster,
                                 explore_share=0.1, **options)

        actor = policy_inference.precast(copy.deepcopy(net)).requires_grad_(False)
        ahead = ppo_loop.PlayAhead(net, actor, "cuda", play)
        self.addCleanup(ahead.close)
        ahead.start(0)
        # A learning step's worth of work queued on this thread's stream
        # while the round is played.
        busy = torch.randn(2048, 2048, device="cuda")
        for _ in range(100):
            busy = torch.tanh(busy @ busy * 1e-3)
        batch, _played, _waited = ahead.take(0)
        torch.cuda.synchronize()
        here = policy_inference.precast(copy.deepcopy(net)).requires_grad_(False)
        want = play(here, 0, own_draws=True)
        self.assertGreater(batch.decisions, 300)
        self.assertSameRound(batch, want)


if __name__ == "__main__":
    unittest.main()
