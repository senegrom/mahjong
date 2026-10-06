"""The joined player decides with less of its forward: the reading of the
opponents' hands, which only the learning step reads, is left out, and the
logits and the value it decides by are bit for bit `everything`'s, eagerly
on the processor and on the card, its weights cast once or not. A forward a
trainer compiled is still asked for all of it."""

from contextlib import nullcontext
import unittest
from unittest.mock import patch

import numpy as np
import riichi_py
import torch

from neural import combined, policy_inference, selfplay, zoo
from neural.observe import Views
from neural.tests.test_bfloat16_players import fusion
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

        net.backbones_forward = compiled
        sparse, allowed = self.positions[0]
        planes, legal = sparse.dense("cpu"), torch.from_numpy(allowed)
        with torch.no_grad():
            logits, value, _phi = net.decision(planes, legal)
            want_logits, want_value, _guessed = net.everything(planes, legal)
        self.assertEqual(asked, [(2, {}), (2, {})])
        same_bits(self, logits, want_logits)
        same_bits(self, value, want_value)

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
        probabilities to the bit, the same games."""
        for device, cast in self.kinds():
            torch.manual_seed(23)
            net = fusion(32, 3).to(device)
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
