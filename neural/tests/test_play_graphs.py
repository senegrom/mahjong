"""Play's small questions answered from CUDA graphs (`--play-graphs`): a
graph answers what the forward answers eagerly at the padded size, bit for
bit, for every kind of player asked in bfloat16; it answers by the weights
of the moment when new ones are copied in or an optimiser moves them, and
is recorded again when weights are put somewhere else; a cast autocast
kept is never what it reads; it can be recorded on a worker beside work
that waits on its own stream, as the learning does; a whole round played
from graphs is the round played eagerly at the padded sizes, with every
graph recorded before it; and everything else, the processor, float32,
gradients, training, bigger batches, is answered eagerly as before. Off
unless a trainer is told, and refused with the compiled player."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural import (
    combined, mortal_learner, mortal_model, play_graphs, policy_inference, ppo_loop, selfplay,
    train_combined, train_mortal, zoo,
)
from neural.model import MORTAL_PLANES, PolicyValueNet
from neural.observe import pad_rows
from neural.tests import test_mixed_tables
from neural.tests.test_baseline_from_play import fake_round, varied
from neural.tests.test_bfloat16_players import checkpoints, fusion
from neural.tests.test_preview_reach import Steered

CUDA = torch.cuda.is_available()
MORTAL_CONFIG = {"resnet": {"conv_channels": 8, "num_blocks": 1}, "control": {"version": 4}}


def questions(n: int, seed: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """`n` rows of planes about as sparse as Mortal's, and masks with a
    few moves open and passing always among them."""
    generator = torch.Generator().manual_seed(seed)
    planes = (torch.rand(n, MORTAL_PLANES, 34, generator=generator) < 0.06).float()
    planes *= torch.rand(n, MORTAL_PLANES, 34, generator=generator)
    mask = torch.rand(n, zoo.MORTAL_ACTIONS, generator=generator) < 0.3
    mask[:, zoo.MORTAL_PASS] = True
    return planes.to(device), mask.to(device)


def parts(answer) -> tuple:
    return (answer,) if isinstance(answer, torch.Tensor) else tuple(answer)


def bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16 if tensor.element_size() == 2 else torch.int32)


def padded(forward, planes: torch.Tensor, mask: torch.Tensor, rows: int) -> tuple:
    """The forward's answer, eagerly, to the question padded to `rows` as
    a graph's is: planes of nought and every move open."""
    return tuple(part[: len(planes)] for part in
                 parts(forward(pad_rows(planes, rows), pad_rows(mask, rows, True))))


def size_of(n: int, rows=play_graphs.ROWS) -> int:
    return next(size for size in rows if size >= n)


class PaddedEagerly(play_graphs.Graphed):
    """Answers as a graph would, eagerly: the question padded to the size
    its graph would be recorded at, the forward run, the first rows kept.
    What a round played from graphs is compared with."""

    def __call__(self, planes, mask):
        n = planes.shape[0]
        if (not planes.is_cuda or not 0 < n <= self.rows[-1] or torch.is_grad_enabled()
                or self.module.training):
            return self.forward(planes, mask)
        answer = padded(self.forward, planes, mask, size_of(n, self.rows))
        return answer[0] if len(answer) == 1 else answer


def networks():
    """One small network of every kind a trainer plays from graphs, on the
    card, and what its deciding forward is: a joined player, a Mortal, one
    of ours in Mortal's moves, and a learner of Mortal's kind."""
    torch.manual_seed(41)
    yield "joined", varied(fusion(16, 2)).cuda().eval(), lambda net: net.decision
    yield "mortal", mortal_model.build(16, 2).cuda().eval(), lambda net: net
    yield "reheaded", PolicyValueNet(16, 2, actions=46).cuda().eval(), lambda net: net
    learner = mortal_learner.MortalLearner(mortal_model.build(16, 2)).cuda().eval()
    with torch.no_grad():
        learner.value_head.weight.normal_()
    yield "learner", learner, lambda net: net.policy


class FallbackTests(unittest.TestCase):
    """Whatever no graph answers is answered eagerly, by the forward itself."""

    def test_the_processor_gradients_training_and_bigger_batches_go_eagerly(self):
        asked = []

        def forward(planes, mask):
            asked.append(len(planes))
            return planes.sum(dim=(1, 2)), mask

        module = torch.nn.Linear(2, 2)
        graphed = play_graphs.Graphed(forward, module)
        planes, mask = questions(5, 1, "cpu")
        with torch.no_grad():
            got = graphed(planes, mask)
        self.assertTrue(torch.equal(got[0], planes.sum(dim=(1, 2))))
        self.assertEqual(asked, [5])
        self.assertEqual(graphed.graphs, {})
        if not CUDA:
            return
        planes, mask = questions(600, 2)
        cases = (
            ("with gradients", planes[:5], mask[:5], True, False),
            ("training", planes[:5], mask[:5], False, True),
            ("bigger than the largest graph", planes, mask, False, False),
            ("no rows", planes[:0], mask[:0], False, False),
        )
        for name, rows, masks, gradients, training in cases:
            with self.subTest(name):
                asked.clear()
                module.train(training)
                with torch.set_grad_enabled(gradients), policy_inference.autocast("cuda"):
                    graphed(rows, masks)
                self.assertEqual(asked, [len(rows)])
                self.assertEqual(graphed.graphs, {})
                self.assertEqual(graphed.replays, 0)

    def test_on_the_processor_no_player_is_changed(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = checkpoints(Path(folder))
            args = SimpleNamespace(opponents=[paths["mortal"], paths["reheaded"], paths["fused"], "club"],
                                   compile=False, play_graphs=True)
            seated = ppo_loop.load_others(args, "cpu")
        self.assertEqual(play_graphs.graphs_of(seated), [])
        self.assertIsNone(seated[2].deciding)
        net = fusion(8, 1)
        self.assertIs(play_graphs.graphed(net, "cpu", amp=False), net)
        self.assertIsNone(net.deciding)


@unittest.skipUnless(CUDA, "CUDA graphs are the card's")
class GraphTests(unittest.TestCase):
    def test_a_graph_answers_what_the_forward_answers_at_the_padded_size(self):
        """Every kind under autocast in bfloat16, its weights cast once as
        the trainers' copies and seated players hold them, or in float32 as
        a learner that plays itself holds them; at sizes on and between the
        graphs', and above them, where the forward answers eagerly at the
        size asked."""
        for name, net, forward_of in networks():
            for cast in (True, False):
                player = policy_inference.precast(copy.deepcopy(net)) if cast else copy.deepcopy(net)
                deciding = forward_of(player)
                graphed = play_graphs.Graphed(deciding, player)
                for n in (1, 7, 16, 100, 512, 600):
                    planes, mask = questions(n, n)
                    with self.subTest(player=name, cast=cast, rows=n), torch.no_grad(), \
                            policy_inference.autocast("cuda"):
                        got = parts(graphed(planes, mask))
                        if n <= play_graphs.ROWS[-1]:
                            want = padded(deciding, planes, mask, size_of(n))
                        else:
                            want = parts(deciding(planes, mask))
                        self.assertEqual(len(got), len(want))
                        for one, two in zip(got, want):
                            self.assertEqual(one.shape, two.shape)
                            self.assertEqual(one.dtype, two.dtype)
                            self.assertTrue(torch.equal(bits(one), bits(two)))
                # One, seven and sixteen rows from the graph of sixteen; six
                # hundred eagerly.
                self.assertEqual(len(graphed.graphs), 3)
                self.assertEqual(graphed.replays, 5)

    def test_only_questions_asked_in_bfloat16_are_answered_from_graphs(self):
        """A graph keeps the workspaces its convolutions were recorded
        with: in float32 under torch 2.8 on this card, gigabytes of them.
        Asked in float32 or under autocast in float16, the forward answers
        eagerly, at the size asked."""
        net = policy_inference.precast(mortal_model.build(16, 2).cuda().eval())
        float32 = mortal_model.build(16, 2).cuda().eval()
        graphed = play_graphs.Graphed(net, net)
        eager = play_graphs.Graphed(float32, float32)
        planes, mask = questions(20, 3)
        with torch.no_grad():
            for _ in range(3):
                with policy_inference.autocast("cuda"):
                    graphed(planes, mask)
                    graphed(planes[:3], mask[:3])
            self.assertEqual((len(graphed.graphs), graphed.replays), (2, 6))
            with torch.autocast("cuda", dtype=torch.float16):
                got = graphed(planes, mask)
                want = net(planes, mask)
            self.assertTrue(torch.equal(bits(got), bits(want)))
            got = eager(planes, mask)
            want = float32(planes, mask)
            self.assertTrue(torch.equal(bits(got), bits(want)))
        self.assertEqual((len(graphed.graphs), graphed.replays), (2, 6))
        self.assertEqual((eager.graphs, eager.replays), ({}, 0))
        # A learner that plays in float32, as without --amp, is left as it is.
        learner = mortal_learner.MortalLearner(mortal_model.build(16, 2)).cuda()
        self.assertEqual(play_graphs.graphed(learner, "cuda", amp=False).inference, learner.policy)

    def test_weights_copied_in_or_moved_by_an_optimiser_are_the_ones_answered_by(self):
        """As a trainer copies the learner's weights into its playing copy
        before each round, and as an optimiser steps a learner that plays
        itself: where they are stored, so nothing is recorded again."""
        for name, net, forward_of in networks():
            player = policy_inference.precast(copy.deepcopy(net))
            deciding = forward_of(player)
            graphed = play_graphs.Graphed(deciding, player)
            planes, mask = questions(30, 5)
            with torch.no_grad(), policy_inference.autocast("cuda"):
                before = parts(graphed(planes, mask))
                recorded, seconds = len(graphed.graphs), graphed.seconds
                moved = copy.deepcopy(net)
                for parameter in moved.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.1)
                player.load_state_dict(moved.state_dict())
                after = parts(graphed(planes, mask))
                want = padded(deciding, planes, mask, 32)
            with self.subTest(player=name, by="copy"):
                self.assertEqual((len(graphed.graphs), graphed.seconds), (recorded, seconds))
                self.assertFalse(torch.equal(bits(before[0]), bits(after[0])))
                for one, two in zip(after, want):
                    self.assertTrue(torch.equal(bits(one), bits(two)))
        # An optimiser's step, on a learner of float32 weights that plays
        # itself under autocast, as train_mortal's does.
        torch.manual_seed(43)
        learner = mortal_learner.MortalLearner(mortal_model.build(16, 2)).cuda()
        play_graphs.graphed(learner, "cuda", amp=True)
        self.assertIsInstance(learner.inference, play_graphs.Graphed)
        # Every size recorded at once, in training mode as it was found.
        self.assertEqual(len(learner.inference.graphs), len(play_graphs.ROWS))
        self.assertTrue(learner.training)
        optimiser = torch.optim.AdamW(learner.parameters(), lr=1e-2, fused=True)
        planes, mask = questions(40, 7)
        for step in range(3):
            learner.eval()
            with torch.no_grad(), policy_inference.autocast("cuda"):
                got = parts(learner.inference(planes, mask))
                want = padded(learner.policy, planes, mask, 64)
            with self.subTest(by="optimiser", step=step):
                for one, two in zip(got, want):
                    self.assertTrue(torch.equal(bits(one), bits(two)))
            learner.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, value = learner.policy(planes, mask)
            loss = logits.float().masked_fill(~mask, 0.0).pow(2).mean() + value.float().pow(2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        # Nothing recorded again: the graphs were asked as play asks.
        self.assertEqual(len(learner.inference.graphs), len(play_graphs.ROWS))

    def test_weights_put_somewhere_else_are_recorded_again(self):
        """A cast to another precision moves the weights; the graphs would
        read where they were, so the next question records them again."""
        torch.manual_seed(47)
        net = mortal_model.build(16, 2).cuda().eval()
        graphed = play_graphs.Graphed(net, net)
        planes, mask = questions(30, 9)
        with torch.no_grad(), policy_inference.autocast("cuda"):
            graphed(planes, mask)
            first = list(graphed.places)
            policy_inference.precast(net)
            got = graphed(planes, mask)
            want = padded(net, planes, mask, 32)[0]
        self.assertNotEqual(graphed.places, first)
        self.assertEqual(len(graphed.graphs), 1)
        self.assertTrue(torch.equal(bits(got), bits(want)))

    def test_a_cast_that_autocast_kept_is_not_what_the_graph_reads(self):
        """Weights that require a gradient have their casts kept by autocast
        for the rest of its region, and an eager forward there keeps them.
        A graph recorded in that region reading those would read memory let
        go when the region ends: here filled with what is not a number."""
        torch.manual_seed(53)
        net = mortal_model.build(16, 2).cuda().eval()
        for parameter in net.parameters():
            parameter.requires_grad_(True)
        graphed = play_graphs.Graphed(net, net)
        planes, mask = questions(20, 11)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                net(planes, mask)
                graphed(planes, mask)
            # The region left, its casts let go: fill what they held.
            junk = [torch.full((parameter.numel(),), float("nan"), device="cuda", dtype=torch.bfloat16)
                    for _ in range(4) for parameter in net.parameters()]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                got = graphed(planes, mask)
                want = padded(net, planes, mask, 32)[0]
        del junk
        self.assertFalse(torch.isnan(got[mask]).any())
        self.assertTrue(torch.equal(bits(got), bits(want)))

    def test_asked_on_one_stream_and_then_another(self):
        torch.manual_seed(59)
        net = policy_inference.precast(mortal_model.build(16, 2).cuda().eval())
        graphed = play_graphs.Graphed(net, net)
        side = torch.cuda.Stream()
        with torch.no_grad(), policy_inference.autocast("cuda"):
            for round_ in range(3):
                for stream in (torch.cuda.current_stream(), side):
                    planes, mask = questions(12 + round_, round_)
                    with torch.cuda.stream(stream):
                        got = graphed(planes, mask)
                        want = padded(net, planes, mask, 16)[0]
                        same = torch.equal(bits(got), bits(want))
                    with self.subTest(round=round_, stream=stream):
                        self.assertTrue(same)
        self.assertEqual(len(graphed.graphs), 1)

    def test_recorded_on_a_worker_beside_work_that_waits_on_its_stream(self):
        """As the learning does beside a round played ahead: this thread
        keeps the card busy, allocates, copies from page-locked memory and
        waits for its own stream, while a worker records graph after graph
        on a stream of its own. Recording forbids only its own thread a
        wait; forbidding every thread's, as recording does unless told
        otherwise, failed this one's waits and the recordings. A wait for
        the whole card is refused to every thread while anything is
        recorded, which is why the trainers record every graph before play
        begins."""
        torch.manual_seed(61)
        net = policy_inference.precast(mortal_model.build(16, 2).cuda().eval())
        graphed = play_graphs.Graphed(net, net)
        planes, mask = questions(500, 13)
        done = threading.Event()
        failed = []
        checked = []

        def worker():
            try:
                with torch.cuda.stream(torch.cuda.Stream()), torch.no_grad(), \
                        policy_inference.autocast("cuda"):
                    for _round in range(4):
                        graphed.forget()
                        for n in (1, 20, 40, 70, 100, 150, 200, 300, 400, 500):
                            got = graphed(planes[:n], mask[:n])
                            want = padded(net, planes[:n], mask[:n], size_of(n))[0]
                            checked.append(torch.equal(bits(got), bits(want)))
            except BaseException as error:  # noqa: BLE001 - raised below
                failed.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=worker)
        thread.start()
        busy = torch.randn(512, 512, device="cuda")
        host = torch.randn(1 << 16)
        waits = 0
        try:
            while not done.is_set():
                busy = torch.tanh(busy @ busy * 1e-3)
                float(busy.sum())
                torch.cuda.current_stream().synchronize()
                torch.empty((1 + waits % 7) << 20, device="cuda").fill_(1.0)
                host.pin_memory().to("cuda", non_blocking=True).sum()
                waits += 1
        finally:
            thread.join()
        self.assertEqual(failed, [])
        self.assertEqual(len(checked), 40)
        self.assertTrue(all(checked))
        self.assertGreater(waits, 10)

    def test_what_graphs_hold_is_their_pools_and_their_inputs(self):
        torch.manual_seed(83)
        forwards = []
        for _ in range(2):
            net = policy_inference.precast(mortal_model.build(16, 2).cuda().eval())
            forwards.append(play_graphs.Graphed(net, net))
        self.assertEqual(play_graphs.held(forwards), 0)
        inputs = play_graphs.ROWS[-1] * (MORTAL_PLANES * 34 * 4 + zoo.MORTAL_ACTIONS)
        forwards[0].prepare("cuda")
        one = play_graphs.held(forwards)
        self.assertGreater(one, inputs)
        self.assertLess(one, inputs + (1 << 30))
        forwards[1].prepare("cuda")
        self.assertGreater(play_graphs.held(forwards), one + inputs)

    def test_a_copy_answers_by_its_own_weights(self):
        torch.manual_seed(67)
        net = policy_inference.precast(varied(fusion(16, 2)).cuda().eval())
        play_graphs.graphed(net, "cuda", amp=True)
        planes, mask = questions(9, 17)
        with torch.no_grad(), policy_inference.autocast("cuda"):
            net.deciding(planes, mask)
            twin = copy.deepcopy(net)
            self.assertIsInstance(twin.deciding, play_graphs.Graphed)
            self.assertIs(twin.deciding.module, twin)
            self.assertEqual(twin.deciding.graphs, {})
            for parameter in twin.fuse.parameters():
                parameter.add_(0.5)
            got = parts(twin.deciding(planes, mask))
            want = padded(twin.decision, planes, mask, 16)
            mine = parts(net.deciding(planes, mask))
        for one, two in zip(got, want):
            self.assertTrue(torch.equal(bits(one), bits(two)))
        self.assertFalse(torch.equal(bits(got[0]), bits(mine[0])))


@unittest.skipUnless(CUDA, "CUDA graphs are the card's")
class RoundTests(unittest.TestCase):
    """Whole rounds on the card, played as train_combined plays them with
    --amp: the learner's copy with its weights cast once, a Mortal, one of
    ours and another joined player seated by player with the Club bot."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.paths = checkpoints(Path(folder.name))
        torch.manual_seed(71)
        self.net = varied(fusion(16, 2)).cuda().eval()

    def play(self, graphs: type, preview_reach=False, steered=False):
        """A round played with `graphs` standing for `play_graphs.Graphed`;
        the players' graphed forwards; and how many questions each answered
        from a graph, and how many graphs each recorded, in the round."""
        with patch.object(play_graphs, "Graphed", graphs):
            actor = policy_inference.precast(copy.deepcopy(self.net)).requires_grad_(False)
            play_graphs.graphed(actor, "cuda", amp=True)
            args = SimpleNamespace(
                opponents=[self.paths["mortal"], self.paths["reheaded"], self.paths["fused"], "club"],
                compile=False, seat_share=0.5, opponent_share=0.0, preview_reach=preview_reach,
                play_graphs=True)
            with contextlib.redirect_stdout(io.StringIO()):
                roster, seated = ppo_loop.seat_others(args, "cuda")
        player = actor
        if steered:
            player = Steered(lambda planes, mask: actor.deciding(planes, mask)[:2], "cuda")
        forwards = play_graphs.graphs_of([actor, *seated])
        before = [(forward.replays, len(forward.graphs)) for forward in forwards]
        torch.manual_seed(73)
        batch = selfplay.play(player, games=4, seed=20261007, device="cuda", amp=True, opponents=seated,
                              seat_share=0.5, population=roster, explore_share=0.1,
                              preview_reach=preview_reach)
        during = [(forward.replays - replays, len(forward.graphs) - graphs)
                  for forward, (replays, graphs) in zip(forwards, before)]
        return batch, forwards, during

    def test_a_round_from_graphs_is_the_round_eager_at_the_padded_sizes(self):
        for preview_reach, steered in ((False, False), (False, True), (True, True)):
            with self.subTest(preview_reach=preview_reach, steered=steered):
                got, graphed, during = self.play(play_graphs.Graphed, preview_reach, steered)
                want, padded_eagerly, _during = self.play(PaddedEagerly, preview_reach, steered)
                # The learner's copy and three seated networks; the Club bot
                # has none. Every graph was recorded before the round, under
                # the autocast it was then asked under, and the learner asked
                # every question of the round from them.
                self.assertEqual(len(graphed), 4)
                self.assertEqual([recorded for _replays, recorded in during], [0, 0, 0, 0])
                self.assertGreater(during[0][0], 100)
                self.assertGreater(sum(replays for replays, _recorded in during[1:]), 0, during)
                self.assertTrue(all(forward.graphs == {} for forward in padded_eagerly))
                self.assertGreater(got.decisions, 300)
                if steered:
                    self.assertGreater(int((got.actions == zoo.MORTAL_RIICHI).sum()), 10)
                for name in ("actions", "log_probs", "returns", "legal", "explored", "values", "held"):
                    self.assertTrue(torch.equal(getattr(got, name), getattr(want, name)), name)
                for name in ("indptr", "indices", "values"):
                    np.testing.assert_array_equal(getattr(got.observations, name),
                                                  getattr(want.observations, name))
                np.testing.assert_array_equal(got.final_scores, want.final_scores)
                self.assertEqual(got.matchups, want.matchups)


class SwitchTests(unittest.TestCase):
    def test_the_seated_others_answer_from_graphs_only_when_the_trainer_says(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = checkpoints(Path(folder))
            opponents = [paths["mortal"], paths["reheaded"], paths["fused"], paths["ours, our moves"], "club"]
            device = "cuda" if CUDA else "cpu"
            for options in ({}, {"play_graphs": False}, {"play_graphs": True}):
                args = SimpleNamespace(opponents=opponents, compile=False, **options)
                seated = ppo_loop.load_others(args, device)
                found = play_graphs.graphs_of(seated)
                with self.subTest(**options):
                    if options.get("play_graphs") and CUDA:
                        # The Mortal, ours in Mortal's moves and the joined
                        # player; ours in the engine's moves and the Club bot
                        # are left as they are.
                        self.assertEqual(len(found), 3)
                        self.assertIs(seated[0].forward.module, seated[0].net)
                        self.assertIs(seated[1].forward.module, seated[1].net)
                        self.assertIs(seated[2].deciding.module, seated[2])
                        # Recorded over the weights as cast for play.
                        self.assertEqual(seated[0].net.brain.encoder.net[0].weight.dtype, torch.bfloat16)
                    else:
                        self.assertEqual(found, [])
        self.assertIsNone(combined.Combined.deciding)

    def test_the_trainers_play_from_graphs_with_the_player_that_plays(self):
        """train_combined graphs its playing copy where it has one, the
        learner itself where it plays its own rounds; train_mortal its
        learner; on the device the trainer chose, only when told."""
        for module in (train_combined, train_mortal):
            for flags in ([], ["--play-graphs"], ["--play-graphs", "--play-ahead"]):
                if module is train_mortal and "--play-ahead" in flags:
                    continue
                graphed, played = [], []

                def graph(player, device, amp):
                    graphed.append((player, device, amp))
                    return player

                def play(player, **_options):
                    played.append(player)
                    return fake_round()

                with self.subTest(trainer=module.__name__, flags=flags), \
                        tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    origin, out = root / "source.pt", root / "run"
                    if module is train_mortal:
                        torch.save({**mortal_model.build(8, 1).state(), "config": MORTAL_CONFIG}, origin)
                    else:
                        torch.save(fusion(8, 1).state(), origin)
                    argv = ["trainer", "--mortal" if module is train_mortal else "--resume", str(origin),
                            "--out", str(out), "--rounds", "2", "--batch", "4", "--epochs", "1",
                            "--games", "1", "--measure-games", "1", *flags]
                    if module is train_combined:
                        argv += ["--fixed", "none"]
                    torch.set_num_threads(1)
                    with patch.object(sys, "argv", argv), \
                            patch.object(torch.cuda, "is_available", return_value=False), \
                            patch.object(play_graphs, "graphed", side_effect=graph), \
                            patch.object(module.selfplay, "play", side_effect=play), \
                            patch.object(module.selfplay, "measure",
                                         return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
                            contextlib.redirect_stdout(io.StringIO()):
                        module.main()
                    records = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
                    if flags:
                        ((player, device, amp),) = graphed
                        self.assertEqual((device, amp), ("cpu", False))
                        self.assertTrue(all(one is player for one in played))
                        for record in records:
                            self.assertEqual(
                                (record["play_graphs"], record["graph_seconds"], record["graph_gb"]),
                                (0, 0.0, 0.0))
                    else:
                        self.assertEqual(graphed, [])
                        for record in records:
                            for field in ("play_graphs", "graph_seconds", "graph_gb"):
                                self.assertNotIn(field, record)

    def test_off_unless_asked_and_refused_with_the_compiled_player(self):
        for module in (train_combined, train_mortal):
            with self.subTest(trainer=module.__name__):
                with patch.object(sys, "argv", ["trainer"]):
                    self.assertFalse(module.parse_args().play_graphs)
                with patch.object(sys, "argv", ["trainer", "--play-graphs"]):
                    self.assertTrue(module.parse_args().play_graphs)
                with patch.object(sys, "argv", ["trainer", "--play-graphs", "--compile"]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    module.parse_args()
        with patch.object(sys, "argv", ["trainer", "--play-graphs", "--compile-learning", "--play-ahead"]):
            self.assertTrue(train_combined.parse_args().play_graphs)

    def test_the_cloud_passes_it_on_only_when_asked_and_never_with_the_compiled_player(self):
        launch = test_mixed_tables.CloudLaunchTests.launch
        self.assertNotIn("--play-graphs", launch(self, "train_combined"))
        command = launch(self, "train_combined", play_graphs=True, compile=False)
        self.assertIn("--play-graphs", command)
        self.assertNotIn("--compile", command)
        command = launch(self, "train_combined", play_graphs=True, compile_learning=True)
        self.assertIn("--play-graphs", command)
        self.assertIn("--compile-learning", command)
        with self.assertRaisesRegex(ValueError, "play_graphs"):
            launch(self, "train_combined", play_graphs=True)


@unittest.skipUnless(CUDA, "CUDA graphs are the card's")
class TrainerOnTheCardTests(unittest.TestCase):
    """The trainers on the card with the switch, small networks and a real
    round: their records say how many graphs play holds."""

    def run_trainer(self, module, flags):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            origin, out = root / "source.pt", root / "run"
            other = root / "mortal.pt"
            torch.manual_seed(79)
            torch.save({**mortal_model.build(8, 1).state(), "config": MORTAL_CONFIG}, other)
            if module is train_mortal:
                torch.save({**mortal_model.build(8, 1).state(), "config": MORTAL_CONFIG}, origin)
            else:
                torch.save(fusion(8, 1).state(), origin)
            argv = ["trainer", "--mortal" if module is train_mortal else "--resume", str(origin),
                    "--out", str(out), "--rounds", "3", "--batch", "64", "--epochs", "1", "--games", "2",
                    "--opponents", str(other), "--seat-share", "0.5", "--measure-every", "2",
                    "--measure-games", "1", "--amp", "--play-graphs", *flags]
            if module is train_combined:
                argv += ["--fixed", "none", "mortal"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            return [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]

    def test_each_trainer_plays_from_graphs(self):
        """Every size of the learner's and of the seated Mortal's recorded
        before the first round, and nothing more, beside the learning or
        not; train_mortal's measurement asks its learner in float32, which
        answers eagerly."""
        for module, flags in ((train_combined, []), (train_combined, ["--play-ahead"]), (train_mortal, [])):
            with self.subTest(trainer=module.__name__, flags=flags):
                records = self.run_trainer(module, flags)
                self.assertEqual([record["generation"] for record in records], [0, 1, 2])
                self.assertEqual([record["play_graphs"] for record in records],
                                 [2 * len(play_graphs.ROWS)] * 3)
                for record in records:
                    self.assertGreaterEqual(record["graph_seconds"], 0.0)
                    self.assertGreater(record["graph_gb"], 0.0)


if __name__ == "__main__":
    unittest.main()
