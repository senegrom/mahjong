"""The joined player's learner runs convolutions channels-last in its
compiled forwards (`train_combined.channels_last`): each a 1x3 convolution
over a view of its input, on a view of its own weight. Compiled, it answers
and learns as the one-dimensional forward does, as nearly as the rounding
allows; outside a compiled forward it answers bit for bit as before; its
parameters, its state dict and its optimiser see nothing new; a copy made
before keeps its own forwards, its graphs and its compiled forward
answering bit for bit as before; and the trainer converts those of our
network and of the head in the learner alone, with --compile-learning,
after the playing copy is made, and leaves Mortal's one-dimensional."""

import copy
import importlib.util
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from neural import combined, play_graphs, policy_inference, train_combined
from neural.tests.test_bfloat16_players import fusion
from neural.tests.test_compiled_learning import compiler_settings_kept, fake_round, forget_whether_triton_runs, train
from neural.tests.test_deciding_forward import same_bits
from neural.tests.test_learning_planes import encoded_like

CUDA = torch.cuda.is_available()
TRITON = importlib.util.find_spec("triton") is not None


def as_the_trainer_converts(net: combined.Combined) -> None:
    """The learner's convolutions the trainer runs channels-last: our
    network's and the head's, not Mortal's (see `train_combined.main`)."""
    train_combined.channels_last(net.ours)
    train_combined.channels_last(net.fuse)


def converted(conv: nn.Module) -> bool:
    """Whether `conv` runs channels-last when compiled: its forward is the
    one `channels_last` gives it."""
    forward = conv.__dict__.get("forward")
    return isinstance(forward, types.MethodType) and forward.__func__ is train_combined._channels_last_forward


def convolutions(module: nn.Module) -> list[nn.Conv1d]:
    return [conv for conv in module.modules() if type(conv) is nn.Conv1d]


def inputs(rows: int, seed: int, device: str = "cpu"):
    planes = encoded_like(rows, seed=seed).dense(device)
    legal = torch.rand(rows, combined.Combined.actions, generator=torch.Generator().manual_seed(seed)) < 0.4
    legal[:, 0] = True
    return planes, legal.to(device)


def traced(forward, *arguments) -> list[str]:
    """The convolutions, by name, in the graph the compiler traces
    `forward` into: what it runs compiled. Nothing is generated; the graph
    runs as it was traced."""
    names: list[str] = []

    def backend(graph, _inputs):
        names.extend(node.target.__name__ for node in graph.graph.nodes
                     if node.op == "call_function" and "conv" in getattr(node.target, "__name__", ""))
        return graph.forward

    torch._dynamo.reset()
    try:
        with torch.no_grad():
            torch.compile(forward, backend=backend, dynamic=False)(*arguments)
    finally:
        torch._dynamo.reset()
    return names


class LayoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        self.net = fusion()
        self.planes, self.legal = inputs(12, seed=4)

    def test_compiled_every_convolution_runs_as_a_1x3_one(self):
        """The learning step's forward and the pass's, traced as the
        compiler traces them: every convolution two-dimensional once
        converted, and every one one-dimensional before. How many the pass
        traces is the compiler's to say: the cloud's torch 2.8 traces the
        reading of the hands into it too, which nothing reads and its
        kernels leave out."""
        net = self.net.train()
        forwards = (("everything", net.everything), ("decision", net.decision))
        before = {name: traced(forward, self.planes, self.legal) for name, forward in forwards}
        self.assertEqual(train_combined.channels_last(net), len(convolutions(net)))
        self.assertTrue(all(converted(conv) for conv in convolutions(net)))
        for name, forward in forwards:
            self.assertGreater(len(before[name]), 10, name)
            self.assertEqual(before[name], ["conv1d"] * len(before[name]), name)
            self.assertEqual(traced(forward, self.planes, self.legal), ["conv2d"] * len(before[name]), name)

    def test_outside_a_compiled_forward_it_answers_bit_for_bit_as_before(self):
        net = self.net.eval()
        with torch.no_grad():
            before = [*net.everything(self.planes, self.legal), *net.decision(self.planes, self.legal)]
            train_combined.channels_last(net)
            after = [*net.everything(self.planes, self.legal), *net.decision(self.planes, self.legal)]
        for index, (got, want) in enumerate(zip(after, before)):
            same_bits(self, got, want, str(index))

    def test_compiled_it_answers_and_learns_as_the_one_dimensional_forward(self):
        """In float32 on the processor, through the compiler's own tracing
        of the backward: the answers and every gradient as the eager
        one-dimensional step's, but for the order the sums are taken in.

        Each gradient is held to the one-dimensional step's as a whole, by
        how far apart the two stand as a share of its size, and not element
        by element. Summed in another order, a value that a ReLU reads
        within a few units in its last place of nought can land on the
        other side, and the gradient through it is then there in one step
        and not in the other. The cloud's torch 2.8 sums the two layouts in
        other orders on a processor with AVX2 and not AVX-512, as CI's are:
        there one of the 52,224 values the second ReLU of the reading
        tower's second block reads crossed (-4.1e-8 against 2.5e-7), which
        moved that block's gradients by up to 1.03e-3 of their size and put
        7,157 of the 388,608 elements of the gradient of the reading's stem
        past the bound each was held to, 1e-4 of it and 1e-6; with that one
        value as the one-dimensional step had it, every gradient stood
        within 1.6e-6 of its size. A gradient worked out wrong stands its
        own size away."""
        from torch._dynamo.backends.common import aot_autograd
        from functorch.compile import make_boxed_func

        backend = aot_autograd(fw_compiler=lambda gm, _inputs: make_boxed_func(gm.forward),
                               bw_compiler=lambda gm, _inputs: make_boxed_func(gm.forward))
        actions = torch.multinomial(self.legal.float(), 1).squeeze(1)

        def step(net, forward):
            net.zero_grad(set_to_none=True)
            logits, value, guessed = forward(self.planes, self.legal)
            loss = (-torch.log_softmax(logits, dim=1).gather(1, actions[:, None]).mean() + value.square().mean()
                    + torch.log_softmax(guessed, dim=2).mean())
            loss.backward()
            return (logits, value, guessed), {name: p.grad for name, p in net.named_parameters()
                                              if p.grad is not None}

        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        for mode in ("none", "mortal+head"):
            with self.subTest(mode=mode):
                plain = self.net
                plain.set_mode(mode)
                plain.train()
                wide = copy.deepcopy(plain)
                train_combined.channels_last(wide)
                want, want_grads = step(plain, plain.everything)
                got, got_grads = step(wide, torch.compile(wide.everything, backend=backend, dynamic=False))
                for name, one, two in zip(("logits", "value", "hands"), got, want):
                    torch.testing.assert_close(one, two, rtol=1e-5, atol=1e-5, msg=name)
                self.assertEqual(sorted(got_grads), sorted(want_grads))
                self.assertGreater(len(want_grads), 20)
                for name in want_grads:
                    size = float(want_grads[name].norm())
                    apart = float((got_grads[name] - want_grads[name]).norm())
                    # Within a hundredth of its size, and 1e-7 more for
                    # the bias of our reading, which has no gradient but
                    # the rounding's: a softmax over the tiles cannot tell
                    # a constant added to every one of them.
                    self.assertLessEqual(apart, 1e-2 * size + 1e-7, name)

    def test_the_parameters_the_state_dict_and_the_optimiser_see_nothing_new(self):
        net = self.net
        parameters = list(net.named_parameters())
        state = net.state_dict()
        groups = train_combined.optimiser_groups(net, types.SimpleNamespace(lr=1.0, lr_ours=2.0, lr_mortal=3.0))
        saved = net.state()
        train_combined.channels_last(net)
        self.assertEqual([(name, id(p)) for name, p in net.named_parameters()],
                         [(name, id(p)) for name, p in parameters])
        now = net.state_dict()
        self.assertEqual(list(now), list(state))
        for name, value in now.items():
            self.assertEqual(value.data_ptr(), state[name].data_ptr(), name)
            self.assertTrue(torch.equal(value, state[name]), name)
        again = train_combined.optimiser_groups(net, types.SimpleNamespace(lr=1.0, lr_ours=2.0, lr_mortal=3.0))
        self.assertEqual([[id(p) for p in group["params"]] for group in again],
                         [[id(p) for p in group["params"]] for group in groups])
        self.assertEqual(sorted(net.state()), sorted(saved))
        # A copy given the learner's weights, as the playing copy is before
        # each round, takes them as it always did.
        copied = fusion()
        copied.load_state_dict(net.state_dict())
        for name, value in copied.state_dict().items():
            self.assertTrue(torch.equal(value, now[name]), name)

    def test_a_copy_made_before_keeps_its_own_forwards(self):
        """The playing copy, made before the learner is converted, as the
        processor makes it to play a round ahead: none of its convolutions
        converted, traced one-dimensional, and answering bit for bit as
        before. On the card it is cast once and graphed as well (see
        `OnTheCardTests`)."""
        net = self.net.eval()
        actor = copy.deepcopy(net).requires_grad_(False)
        with torch.no_grad():
            before = actor.decision(self.planes, self.legal)
        train_combined.channels_last(net)
        self.assertFalse(any(converted(conv) for conv in convolutions(actor)))
        names = traced(actor.decision, self.planes, self.legal)
        self.assertGreater(len(names), 10)
        self.assertEqual(names, ["conv1d"] * len(names))
        with torch.no_grad():
            after = actor.decision(self.planes, self.legal)
        for index, (got, want) in enumerate(zip(after, before)):
            same_bits(self, got, want, str(index))

    def test_a_convolution_it_cannot_view_so_is_refused_and_a_subclass_left_alone(self):
        for conv in (nn.Conv1d(4, 4, 3, padding="same"), nn.Conv1d(4, 4, 3, padding=1, padding_mode="circular")):
            with self.subTest(conv=conv), self.assertRaises(ValueError):
                train_combined.channels_last(nn.Sequential(conv))

        class Own(nn.Conv1d):
            pass

        mine = nn.Sequential(Own(4, 4, 3, padding=1), nn.Conv1d(4, 4, 3, padding=1))
        self.assertEqual(train_combined.channels_last(mine), 1)
        self.assertEqual((converted(mine[0]), converted(mine[1])), (False, True))


class TrainerTests(unittest.TestCase):
    def test_the_trainer_converts_ours_and_the_head_in_the_learner_alone_to_compile_its_learning(self):
        """With --compile-learning, every convolution of our network and of
        the head in the learner, none of Mortal's, and none of the copy that
        plays the rounds, which is made before; with --compile, or nothing
        compiled, none at all."""
        for flags, compiling in ((["--compile-learning", "--play-ahead"], True), (["--play-ahead"], False),
                                 (["--compile"], False)):
            learners, players = [], []

            def play(player):
                players.append(player)
                return fake_round()

            def learning_forward(net, args):
                learners.append(net)
                return real(net, args)

            real = train_combined.learning_forward
            with self.subTest(flags=flags), compiler_settings_kept(), \
                    patch.object(train_combined, "learning_forward", learning_forward):
                train(["--rounds", "2", "--fixed", "none", *flags], play, lambda function, **_options: function)
                (learner,) = learners
                self.assertTrue(players)
                for part, wanted in (("ours", compiling), ("fuse", compiling), ("mortal", False)):
                    self.assertEqual({converted(conv) for conv in convolutions(getattr(learner, part))}, {wanted},
                                     part)
                for player in players:
                    self.assertEqual(player is learner, "--play-ahead" not in flags)
                    if player is not learner:
                        self.assertFalse(any(converted(conv) for conv in convolutions(player)))


@unittest.skipUnless(CUDA and TRITON, "the compiler's kernels are the card's")
class OnTheCardTests(unittest.TestCase):
    """The learning step compiled as the trainer compiles it, in bfloat16 on
    the card, with its convolutions channels-last and without; and the
    playing copy beside it, its graphs and its compiled forward."""

    def setUp(self):
        torch._dynamo.reset()
        forget_whether_triton_runs()

    def tearDown(self):
        torch._dynamo.reset()

    def test_compiled_it_learns_as_the_one_dimensional_step(self):
        """As near the float32 step as the one-dimensional step comes, give
        or take, its logits and its gradients, and its gradients along the
        same direction: what bfloat16 moves, it moves by other roundings."""
        def step(forward, net, planes, legal, actions, amp=True):
            net.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, value, guessed = forward(planes if amp else planes.float(), legal)
            loss = (-torch.log_softmax(logits.float(), dim=1).gather(1, actions[:, None]).mean()
                    + value.float().square().mean() + torch.log_softmax(guessed.float(), dim=2).mean())
            loss.backward()
            grads = torch.cat([p.grad.reshape(-1) for p in net.parameters() if p.grad is not None])
            return float(loss.detach()), grads, logits.float().detach()

        args = types.SimpleNamespace(compile=False, compile_learning=True)
        with compiler_settings_kept():
            for mode in ("none", "mortal"):
                torch.manual_seed(3)
                plain = fusion().cuda()
                plain.set_mode(mode)
                plain.train()
                wide = copy.deepcopy(plain)
                as_the_trainer_converts(wide)
                planes = encoded_like(256, seed=4).dense("cuda", torch.bfloat16)
                legal = torch.rand(256, 46, device="cuda") < 0.4
                legal[:, 0] = True
                actions = torch.multinomial(legal.float(), 1).squeeze(1)
                exact_loss, exact, exact_logits = step(plain.everything, plain, planes, legal, actions, amp=False)
                plain_loss, plain_grads, plain_logits = step(train_combined.learning_forward(plain, args), plain,
                                                             planes, legal, actions)
                # Traced afresh, whatever the compiler's checks make of a
                # forward put on a convolution itself.
                torch._dynamo.reset()
                loss, grads, logits = step(train_combined.learning_forward(wide, args), wide, planes, legal,
                                           actions)
                torch._dynamo.reset()
                with self.subTest(mode=mode):
                    self.assertLess(abs(loss - plain_loss), 2.0 ** -7 * abs(plain_loss))
                    finite = torch.isfinite(exact_logits)
                    self.assertTrue(torch.equal(torch.isfinite(logits), finite))
                    plain_apart = float((plain_logits - exact_logits)[finite].abs().mean())
                    apart = float((logits - exact_logits)[finite].abs().mean())
                    self.assertLess(apart, 2 * plain_apart + 1e-3)
                    plain_off = float((plain_grads - exact).norm() / exact.norm())
                    off = float((grads - exact).norm() / exact.norm())
                    self.assertLess(off, 2 * plain_off + 1e-3)
                    self.assertGreater(float(nn.functional.cosine_similarity(grads, plain_grads, dim=0)), 0.99)

    def test_the_playing_copy_answers_bit_for_bit_as_before(self):
        """Its small questions from graphs and its bigger ones from its
        compiled forward, before the learner is converted and after the
        learner's converted step and pass have been compiled and run."""
        torch.manual_seed(5)
        net = fusion().cuda()
        actor = policy_inference.precast(copy.deepcopy(net)).requires_grad_(False)
        play_graphs.graphed(actor, "cuda", amp=True, compile=True)
        asked = [inputs(rows, seed=rows, device="cuda") for rows in (40, 300, 700, 1500)]

        def answers():
            actor.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return [actor.deciding(planes, legal) for planes, legal in asked]

        before = answers()
        args = types.SimpleNamespace(compile=False, compile_learning=True)
        with compiler_settings_kept():
            as_the_trainer_converts(net)
            learn = train_combined.learning_forward(net, args)
            passing = train_combined.pass_forward(net, args)
            planes, legal = inputs(256, seed=8, device="cuda")
            net.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, value, guessed = learn(planes.to(torch.bfloat16), legal)
            (logits.float().logsumexp(1).mean() + value.float().mean() + guessed.float().mean()).backward()
            net.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                passing(planes.to(torch.bfloat16), legal)
            actor.load_state_dict(net.state_dict())
            after = answers()
        self.assertEqual(play_graphs.accounts(play_graphs.graphs_of([actor]))["compiled_answers"], 4)
        for rows, got, want in zip((40, 300, 700, 1500), after, before):
            for name, one, two in zip(("logits", "value", "phi"), got, want):
                same_bits(self, one, two, f"{rows} rows, {name}")


if __name__ == "__main__":
    unittest.main()
