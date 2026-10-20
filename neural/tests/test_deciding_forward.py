"""The joined player decides with less of its forward: the reading of the
opponents' hands, which only the learning step reads, is left out, and the
logits and the value it decides by are bit for bit `everything`'s, eagerly
on the processor and on the card, its weights cast once or not. A forward a
trainer or a seated player's loader compiled is still asked for all of it,
and the compiler traces the deciding forward whole, as it runs eagerly,
without the reading."""

from contextlib import nullcontext
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import riichi_py
import torch

from neural import combined, policy_inference, selfplay, train_combined, zoo
from neural.observe import Views
from neural.tests.test_bfloat16_players import fusion
from neural.tests.test_compiled_learning import compiler_settings_kept
from neural.tests.test_step_planes import deciding, play_on

CUDA = torch.cuda.is_available()


def same_bits(test, got: torch.Tensor, want: torch.Tensor, name: str = ""):
    test.assertEqual(got.dtype, want.dtype, name)
    test.assertEqual(got.shape, want.shape, name)
    width = {2: torch.int16, 4: torch.int32}[got.element_size()]
    test.assertTrue(torch.equal(got.view(width), want.view(width)), name)


def positions(games: int = 8, steps: int = 40, seed: int = 404):
    """Real positions: each step's deciding players' planes, sparse, and
    the moves our engine allows them in Mortal's, from tables played on by
    the heuristic player."""
    arena = riichi_py.Arena(games=games, seed=seed)
    views = Views(arena, games)
    found = []
    for _ in range(steps):
        views.advance()
        live, deciders = deciding(arena, games)
        legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, -1).astype(bool)
        allowed = zoo.translatable(legal[live])
        allowed[~allowed.any(axis=1), zoo.MORTAL_PASS] = True
        sparse, _own = views.sparse_and_masks(live, deciders)
        found.append((sparse, allowed))
        play_on(arena, games)
    return found


def as_it_was(self, planes, legal):
    """`Combined.decision` as deciding was before it: the whole forward,
    and the logits and the value it returns. The reference."""
    logits, value = self.forward(planes, legal)
    return logits, value, None


class DecidingForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.positions = positions()

    def kinds(self):
        """The joined player as it decides: on the processor in float32,
        and on the card under autocast, its weights cast by autocast or
        cast once (the trainer's copy, the seated others)."""
        yield "cpu", False
        if CUDA:
            yield "cuda", False
            yield "cuda", True

    def test_the_logits_and_the_value_are_everythings_bit_for_bit(self):
        for device, cast in self.kinds():
            torch.manual_seed(17)
            net = fusion(32, 3).to(device).eval()
            if cast:
                policy_inference.precast(net)
            rows = 0
            for sparse, allowed in self.positions[::4]:
                planes = sparse.dense(device)
                legal = torch.from_numpy(allowed).to(device)
                with self.subTest(device=device, cast=cast, rows=len(allowed)), torch.no_grad(), \
                        torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    want_logits, want_value, _guessed = net.everything(planes, legal)
                    logits, value, phi = net.decision(planes, legal)
                    same_bits(self, logits, want_logits, "logits")
                    same_bits(self, value, want_value, "value")
                    # Mortal's vector beneath them, as Mortal works it out.
                    same_bits(self, phi, net.mortal.features(planes), "phi")
                rows += len(allowed)
            self.assertGreater(rows, 50)

    def test_the_reading_of_the_hands_is_not_run(self):
        net = fusion().eval()
        ran = []
        net.ours.belief_stem.register_forward_hook(lambda *_args: ran.append("belief"))
        net.fuse.hands_fix.register_forward_hook(lambda *_args: ran.append("hands"))
        sparse, allowed = self.positions[0]
        with torch.no_grad():
            net.decision(sparse.dense("cpu"), torch.from_numpy(allowed))
            self.assertEqual(ran, [])
            net.everything(sparse.dense("cpu"), torch.from_numpy(allowed))
            self.assertEqual(ran, ["belief", "hands"])

    def test_a_compiled_forward_is_asked_for_all_of_it(self):
        """Compiled, a forward without the hands would be another graph,
        with answers of its own bits; the one compiled is asked as before.
        The compiler itself is not run: a stand-in records the call."""
        net = fusion().eval()
        asked = []

        def compiled(*args, **kwargs):
            asked.append((len(args), kwargs))
            return net.backbones(*args, **kwargs)

        with patch.object(torch, "compile", lambda _function, **_options: compiled):
            net.compile_backbones(dynamic=True)
        sparse, allowed = self.positions[0]
        planes, legal = sparse.dense("cpu"), torch.from_numpy(allowed)
        with torch.no_grad():
            logits, value, _phi = net.decision(planes, legal)
            want_logits, want_value, _guessed = net.everything(planes, legal)
        self.assertEqual(asked, [(2, {}), (2, {})])
        same_bits(self, logits, want_logits)
        same_bits(self, value, want_value)

    def test_a_copy_still_leaves_the_reading_out(self):
        """The playing copy is a deep copy of the learner: the forward
        beneath it is its own method, bound to it, and it decides without
        the reading of the hands as the learner does."""
        net = copy.deepcopy(fusion().eval())
        ran = []
        net.ours.belief_stem.register_forward_hook(lambda *_args: ran.append("belief"))
        sparse, allowed = self.positions[0]
        with torch.no_grad():
            net.decision(sparse.dense("cpu"), torch.from_numpy(allowed))
        self.assertEqual(ran, [])

    def test_the_trainer_and_the_loader_say_they_compiled_the_networks_beneath(self):
        """--compile, and a seated player loaded to be compiled, put the
        compiled forward in place through the network, which sets the flag
        `decision` reads; --compile-learning, nothing compiled, and a player
        loaded as it is leave both as they were. The compiler itself is not
        run: a stand-in records what it was given."""
        compiled = []

        def stand_in(function, **options):
            compiled.append((function.__name__, options))
            return function

        for switches, beneath, wanted in (
            ({"compile": True, "compile_learning": False}, True,
             [("backbones", {"dynamic": True}), ("everything", {})]),
            ({"compile": False, "compile_learning": True}, False, [("everything", {"dynamic": False})]),
            ({"compile": False, "compile_learning": False}, False, []),
        ):
            compiled.clear()
            net = fusion(8, 1)
            with self.subTest(**switches), compiler_settings_kept(), patch.object(torch, "compile", stand_in):
                train_combined.learning_forward(net, SimpleNamespace(**switches))
                self.assertEqual(compiled, wanted)
                self.assertIs(net.backbones_compiled, beneath)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fused.pt"
            torch.save(fusion(8, 1).state(), path)
            for asked in (True, False):
                compiled.clear()
                with self.subTest(compile=asked), patch.object(torch, "compile", stand_in):
                    seated = zoo.load_player(path, "cpu", compile=asked)
                    self.assertEqual(compiled, [("backbones", {"dynamic": True})] if asked else [])
                    self.assertIs(seated.backbones_compiled, asked)

    def test_the_compiler_traces_it_whole(self):
        """One graph, unbroken, wherever it is compiled. The cloud's torch
        2.8 broke it in three where it compared the forward beneath with
        the networks' own method."""
        net = fusion(8, 1).eval()
        sparse, allowed = self.positions[0]
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        with torch.no_grad():
            explained = torch._dynamo.explain(net.decision)(sparse.dense("cpu"), torch.from_numpy(allowed))
        self.assertEqual((explained.graph_count, explained.graph_break_count), (1, 0))

    def test_the_compiler_traces_what_deciding_runs_and_not_the_reading(self):
        """Traced, the deciding forward holds the convolutions it runs
        eagerly and no more, none of the reading of the hands among them.
        The cloud's torch 2.8 traced the reading in, ten convolutions more,
        while `decision` asked whether the function beneath was the
        networks' own method: it took the answer to be no in every forward
        it compiled (see `Combined.decision`)."""
        net = fusion(8, 1).eval()
        sparse, allowed = self.positions[0]
        planes, legal = sparse.dense("cpu"), torch.from_numpy(allowed)
        reading = {*net.ours.belief_stem.modules(), *net.ours.belief_tower.modules(), net.ours.hands}
        ran = []
        hooks = [conv.register_forward_hook(lambda conv, *_args: ran.append(conv))
                 for conv in net.modules() if isinstance(conv, torch.nn.Conv1d)]
        with torch.no_grad():
            net.decision(planes, legal)
        for hook in hooks:
            hook.remove()
        traced = []

        def backend(graph, _inputs):
            traced.extend(node for node in graph.graph.nodes
                          if node.op == "call_function" and getattr(node.target, "__name__", "") == "conv1d")
            return graph.forward

        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        with torch.no_grad():
            torch.compile(net.decision, backend=backend, dynamic=False)(planes, legal)
        self.assertGreater(len(ran), 5)
        self.assertFalse(reading & set(ran))
        self.assertEqual(len(traced), len(ran))

    def test_seated_asks_are_answered_as_before(self):
        games = 8
        arena = riichi_py.Arena(games=games, seed=505)
        views = Views(arena, games)
        for device, cast in self.kinds():
            torch.manual_seed(19)
            net = fusion().to(device).eval()
            if cast:
                policy_inference.precast(net)
            asked = 0
            for step in range(30):
                views.advance()
                live, deciders = deciding(arena, games)
                legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, -1)
                allowed = zoo.translatable(legal.astype(bool)[live])
                allowed[~allowed.any(axis=1), zoo.MORTAL_PASS] = True
                views.prepare(live, deciders)
                who = list(zip(live.tolist(), deciders.tolist()))
                if step % 5 == 0:
                    with self.subTest(device=device, cast=cast, step=step):
                        got, got_own = net._ask(views, who, False, allowed)
                        with patch.object(combined.Combined, "decision", as_it_was):
                            want, want_own = net._ask(views, who, False, allowed)
                        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
                        np.testing.assert_array_equal(got_own, want_own)
                    asked += len(who)
                play_on(arena, games)
            self.assertGreater(asked, 20)

    def test_whole_rounds_are_played_as_before(self):
        """Self-play by the joined player, deciding with the shorter
        forward and with the whole one: the same moves, the same recorded
        probabilities to the bit, the same games. Played as the trainer
        plays, on the card by a copy with its weights cast once, or on the
        processor where there is no card; every kind is compared batch by
        batch above."""
        for device, cast in [("cuda", True)] if CUDA else [("cpu", False)]:
            torch.manual_seed(23)
            net = fusion().to(device)
            if cast:
                policy_inference.precast(net)
            rounds = []
            for reference in (False, True):
                torch.manual_seed(29)
                with patch.object(combined.Combined, "decision", as_it_was) if reference else nullcontext():
                    rounds.append(selfplay.play(net, games=1, seed=99, device=device,
                                                amp=device == "cuda"))
            new, old = rounds
            with self.subTest(device=device, cast=cast):
                self.assertGreater(new.decisions, 100)
                self.assertTrue(torch.equal(new.actions, old.actions))
                same_bits(self, new.log_probs, old.log_probs)
                self.assertTrue(torch.equal(new.legal, old.legal))
                np.testing.assert_array_equal(new.final_scores, old.final_scores)


if __name__ == "__main__":
    unittest.main()
