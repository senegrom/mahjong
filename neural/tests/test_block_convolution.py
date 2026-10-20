"""Mortal's block convolutions work out their weight gradients as GEMMs in
the joined player's compiled learning step (`neural.block_convolution`):
the forward and the input gradient are the convolution's own to the bit,
and the weight gradient the same sums in another order. Only where a
compiled forward trains Mortal: outside a compiled forward, without a
gradient and in a generation that holds Mortal still, each convolution is
the module's own and is traced as before. The parameters, the state dict
and the optimiser see nothing new; a copy made before keeps its own
forwards; and the trainer works them out so in the learner alone, with
--compile-learning, after the playing copy is made."""

import copy
import importlib.util
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from neural import block_convolution, mortal_model, train_combined
from neural.tests.test_bfloat16_players import fusion
from neural.tests.test_channels_last import as_the_trainer_converts, inputs
from neural.tests.test_compiled_learning import compiler_settings_kept, fake_round, forget_whether_triton_runs, train
from neural.tests.test_learning_planes import encoded_like

CUDA = torch.cuda.is_available()
TRITON = importlib.util.find_spec("triton") is not None
BLOCK_CONVOLUTION = "mahjong.block_convolution.default"
BLOCK_WEIGHT_GRADIENT = "mahjong.block_weight_gradient.default"


def same_bits(test, got: torch.Tensor, want: torch.Tensor, name: str = "") -> None:
    test.assertEqual(got.dtype, want.dtype, name)
    test.assertEqual(got.shape, want.shape, name)
    width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[got.element_size()]
    test.assertTrue(torch.equal(got.contiguous().view(width), want.contiguous().view(width)), name)


def worked_out(conv: nn.Module) -> bool:
    """Whether `conv` works out its weight gradient as GEMMs when compiled:
    its forward is the one `gemm_weight_gradients` gives it."""
    forward = conv.__dict__.get("forward")
    return isinstance(forward, types.MethodType) and forward.__func__ is block_convolution._gemm_forward


def block_convolutions(module: nn.Module) -> list[nn.Conv1d]:
    return [conv for block in module.modules() if isinstance(block, mortal_model.ResBlock)
            for conv in block.res_unit if type(conv) is nn.Conv1d]


def other_convolutions(module: nn.Module) -> list[nn.Conv1d]:
    blocks = {id(conv) for conv in block_convolutions(module)}
    return [conv for conv in module.modules() if type(conv) is nn.Conv1d and id(conv) not in blocks]


def block_weights(net: nn.Module) -> set[str]:
    """The names of the block convolutions' weights among `net`'s parameters."""
    weights = {id(conv.weight) for conv in block_convolutions(net)}
    return {name for name, parameter in net.named_parameters() if id(parameter) in weights}


def exact_weight_gradient(x: torch.Tensor, grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.convolution_backward(grad, x, weight, None, [1], [1], [1], False, [0], 1,
                                               [False, True, False])[1]


def operators(run) -> dict[str, list[str]]:
    """The convolutions and the block operators, by name, in the graphs the
    compiler makes of what `run(compile)` compiles and runs, forward (or
    without a gradient) and backward. Nothing is generated; each graph runs
    as it was traced."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    seen: dict[str, list[str]] = {"forward": [], "backward": []}

    def keep(kind):
        def compiler(graph, _inputs):
            seen[kind].extend(str(node.target) for node in graph.graph.nodes if node.op == "call_function"
                              and ("conv" in str(node.target) or "mahjong" in str(node.target)))
            return make_boxed_func(graph.forward)
        return compiler

    backend = aot_autograd(fw_compiler=keep("forward"), bw_compiler=keep("backward"))
    torch._dynamo.reset()
    try:
        run(lambda function: torch.compile(function, backend=backend, dynamic=False))
    finally:
        torch._dynamo.reset()
    return seen


def boxed_backend():
    """The compiler's own tracing, forward and backward, its graphs run as
    they were traced."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    return aot_autograd(fw_compiler=lambda graph, _inputs: make_boxed_func(graph.forward),
                        bw_compiler=lambda graph, _inputs: make_boxed_func(graph.forward))


class OperatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)

    def test_its_forward_and_input_gradient_are_the_convolutions_own_to_the_bit(self):
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.randn(5, 8, 34, dtype=dtype, requires_grad=True)
                weight = (torch.randn(6, 8, 3) * 0.2).to(dtype).requires_grad_(True)
                want = nn.functional.conv1d(x, weight, None, 1, 1)
                got = block_convolution.block_convolution(x, weight)
                same_bits(self, got, want, "forward")
                grad = torch.randn_like(want)
                (want_x,) = torch.autograd.grad(want, x, grad)
                (got_x,) = torch.autograd.grad(got, x, grad)
                same_bits(self, got_x, want_x, "input gradient")

    def test_its_weight_gradient_is_the_convolutions_sums_in_another_order(self):
        """In float64 to the last few bits; in float32 within float32's own
        error, a few 1e-7 of the sum of the absolute products; in bfloat16
        the float32 sum rounded once. Channels in and out differ, so that
        an answer transposed would show, and so do rows' positions."""
        for rows, positions in ((5, 34), (3, 7)):
            with self.subTest(rows=rows, positions=positions):
                x = torch.randn(rows, 8, positions, dtype=torch.float64)
                grad = torch.randn(rows, 6, positions, dtype=torch.float64)
                weight = torch.randn(6, 8, 3, dtype=torch.float64)
                exact = exact_weight_gradient(x, grad, weight)
                scale = exact_weight_gradient(x.abs(), grad.abs(), weight)
                got = block_convolution.block_weight_gradient(x, grad)
                self.assertEqual(got.shape, (6, 8, 3))
                self.assertTrue(got.is_contiguous())
                torch.testing.assert_close(got, exact, rtol=1e-12, atol=1e-12)
                single = block_convolution.block_weight_gradient(x.float(), grad.float())
                self.assertEqual(single.dtype, torch.float32)
                self.assertLess(float(((single.double() - exact).abs() / scale).max()), 1e-6)
                half = block_convolution.block_weight_gradient(x.to(torch.bfloat16), grad.to(torch.bfloat16))
                self.assertEqual(half.dtype, torch.bfloat16)
                rounded = exact_weight_gradient(x.to(torch.bfloat16).double(), grad.to(torch.bfloat16).double(),
                                                weight).to(torch.bfloat16)
                spacing = torch.exp2(torch.floor(torch.log2(rounded.double().abs().clamp_min(1e-30))) - 7)
                gap = (half.double() - rounded.double()).abs()
                self.assertTrue(bool((gap <= spacing * 1.0001 + 1e-6 * scale).all()))

    def test_each_tap_meets_the_input_it_should_and_no_row_meets_the_next(self):
        """Tap t of the weight's gradient pairs the answer's position p with
        the input's at p + t - 1; and no row's positions meet another's,
        which the zeros between them in the layout keep apart."""
        x, grad = torch.zeros(2, 3, 5), torch.zeros(2, 2, 5)
        x[1, 2, 0] = 1.0
        grad[1, 1, 1] = 1.0
        want = torch.zeros(2, 3, 3)
        want[1, 2, 0] = 1.0
        self.assertTrue(torch.equal(block_convolution.block_weight_gradient(x, grad), want))
        x[1, 2, 0], x[1, 2, 2] = 0.0, 1.0
        want[1, 2, 0], want[1, 2, 2] = 0.0, 1.0
        self.assertTrue(torch.equal(block_convolution.block_weight_gradient(x, grad), want))
        for first, second in (((0, 4), (1, 0)), ((1, 0), (0, 4))):
            x, grad = torch.zeros(2, 3, 5), torch.zeros(2, 2, 5)
            x[first[0], :, first[1]] = 1.0
            grad[second[0], :, second[1]] = 1.0
            self.assertTrue(torch.equal(block_convolution.block_weight_gradient(x, grad), torch.zeros(2, 3, 3)))

    def test_what_the_layouts_held_before_reaches_no_sum(self):
        """The layouts are made from memory not cleared, as the card's
        allocator hands it out; whatever it held, the padding is zeros."""
        real = torch.Tensor.new_empty

        def dirty(self, *size, **options):
            return real(self, *size, **options).fill_(float("nan"))

        x = torch.randn(4, 8, 9, dtype=torch.float64)
        grad = torch.randn(4, 6, 9, dtype=torch.float64)
        want = exact_weight_gradient(x, grad, torch.zeros(6, 8, 3, dtype=torch.float64))
        with patch.object(torch.Tensor, "new_empty", dirty):
            got = block_convolution.block_weight_gradient(x, grad)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)

    def test_both_operators_pass_torch_librarys_own_checks(self):
        """Their schemas, their fake implementations against the real ones,
        the forward's gradients as registered, and the compiler's tracing
        of both."""
        x = torch.randn(3, 8, 34, requires_grad=True)
        weight = torch.randn(6, 8, 3, requires_grad=True)
        torch.library.opcheck(block_convolution.block_convolution, (x, weight))
        torch.library.opcheck(block_convolution.block_weight_gradient,
                              (torch.randn(3, 8, 34), torch.randn(3, 6, 34)))


class InstallTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        self.net = fusion()
        self.planes, self.legal = inputs(12, seed=4)

    def learn(self, net):
        def run(compile):
            logits, value, guessed = compile(net.everything)(self.planes, self.legal)
            (logits.logsumexp(1).mean() + value.mean() + guessed.mean()).backward()
        return run

    def test_compiled_each_block_convolution_that_trains_is_one_operator_each_way(self):
        """Traced as the compiler traces the learning step: each block
        convolution one operator in the forward graph and its weight
        gradient one in the backward, with its input gradient the
        convolution's own; every other convolution as it was."""
        net = self.net.train()
        before = operators(self.learn(net))
        blocks = len(block_convolutions(net.mortal))
        self.assertEqual(blocks, 4)
        self.assertEqual(block_convolution.gemm_weight_gradients(net.mortal), blocks)
        self.assertTrue(all(worked_out(conv) for conv in block_convolutions(net)))
        self.assertFalse(any(worked_out(conv) for conv in other_convolutions(net)))
        after = operators(self.learn(net))
        convolutions = before["forward"].count("aten.convolution.default")
        self.assertGreater(convolutions, 10)
        self.assertNotIn(BLOCK_CONVOLUTION, before["forward"])
        self.assertEqual(after["forward"].count(BLOCK_CONVOLUTION), blocks)
        self.assertEqual(after["forward"].count("aten.convolution.default"), convolutions - blocks)
        self.assertEqual(after["backward"].count(BLOCK_WEIGHT_GRADIENT), blocks)
        self.assertEqual(after["backward"].count("aten.convolution_backward.default"),
                         before["backward"].count("aten.convolution_backward.default"))

    def test_without_a_gradient_or_with_mortal_held_still_each_is_traced_as_before(self):
        """The baseline pass and play, in a generation that trains Mortal,
        and the learning step of one that holds Mortal still: the graphs
        the compiler made before."""
        net = self.net

        def decide(compile):
            with torch.no_grad():
                compile(net.decision)(self.planes, self.legal)
        net.set_mode("none")
        net.eval()
        deciding_before = operators(decide)
        net.set_mode("mortal")
        net.train()
        held_before = operators(self.learn(net))
        self.assertEqual(block_convolution.gemm_weight_gradients(net.mortal), 4)
        self.assertEqual(operators(self.learn(net)), held_before)
        self.assertGreater(len(held_before["backward"]), 10)
        net.set_mode("none")
        net.eval()
        self.assertEqual(operators(decide), deciding_before)
        self.assertGreater(len(deciding_before["forward"]), 10)

    def test_outside_a_compiled_forward_it_answers_and_learns_bit_for_bit_as_before(self):
        net = self.net.train()
        twin = copy.deepcopy(net)
        block_convolution.gemm_weight_gradients(twin.mortal)
        actions = torch.multinomial(self.legal.float(), 1).squeeze(1)
        for one in (net, twin):
            logits, value, guessed = one.everything(self.planes, self.legal)
            (-torch.log_softmax(logits, dim=1).gather(1, actions[:, None]).mean() + value.square().mean()
             + guessed.mean()).backward()
        for (name, want), got in zip(net.named_parameters(), twin.parameters()):
            self.assertEqual(want.grad is None, got.grad is None, name)
            if want.grad is not None:
                same_bits(self, got.grad, want.grad, name)
        with torch.no_grad():
            for got, want in zip(twin.everything(self.planes, self.legal), net.everything(self.planes, self.legal)):
                same_bits(self, got, want)

    def test_compiled_it_answers_and_learns_as_the_modules_own_convolutions(self):
        """In float32 on the processor, through the compiler's own tracing
        of the backward: the answers and every gradient but the blocks'
        weights' the same to the bit as the same graphs' with the
        convolutions' own forwards, and those the same sums in another
        order; in a generation that holds Mortal still, all of it."""
        backend = boxed_backend()
        actions = torch.multinomial(self.legal.float(), 1).squeeze(1)

        def step(net):
            torch._dynamo.reset()
            net.zero_grad(set_to_none=True)
            logits, value, guessed = torch.compile(net.everything, backend=backend, dynamic=False)(
                self.planes, self.legal)
            loss = (-torch.log_softmax(logits, dim=1).gather(1, actions[:, None]).mean() + value.square().mean()
                    + torch.log_softmax(guessed, dim=2).mean())
            loss.backward()
            return (logits, value, guessed), {name: p.grad for name, p in net.named_parameters()
                                              if p.grad is not None}

        self.addCleanup(torch._dynamo.reset)
        blocks = block_weights(self.net)
        self.assertEqual(len(blocks), 4)
        for mode in ("none", "ours+head", "mortal"):
            with self.subTest(mode=mode):
                plain = self.net
                plain.set_mode(mode)
                plain.train()
                gemm = copy.deepcopy(plain)
                block_convolution.gemm_weight_gradients(gemm.mortal)
                want, want_grads = step(plain)
                got, got_grads = step(gemm)
                for name, one, two in zip(("logits", "value", "hands"), got, want):
                    same_bits(self, one, two, name)
                self.assertEqual(sorted(got_grads), sorted(want_grads))
                self.assertEqual(blocks <= set(want_grads), mode != "mortal")
                for name in want_grads:
                    if name in blocks:
                        torch.testing.assert_close(got_grads[name], want_grads[name], rtol=1e-5, atol=1e-7,
                                                   msg=name)
                    else:
                        same_bits(self, got_grads[name], want_grads[name], name)

    def test_the_parameters_the_state_dict_and_the_optimiser_see_nothing_new(self):
        net = self.net
        parameters = list(net.named_parameters())
        state = net.state_dict()
        options = types.SimpleNamespace(lr=1.0, lr_ours=2.0, lr_mortal=3.0)
        groups = train_combined.optimiser_groups(net, options)
        saved = net.state()
        block_convolution.gemm_weight_gradients(net.mortal)
        self.assertEqual([(name, id(p)) for name, p in net.named_parameters()],
                         [(name, id(p)) for name, p in parameters])
        now = net.state_dict()
        self.assertEqual(list(now), list(state))
        for name, value in now.items():
            self.assertEqual(value.data_ptr(), state[name].data_ptr(), name)
            self.assertTrue(torch.equal(value, state[name]), name)
        again = train_combined.optimiser_groups(net, options)
        self.assertEqual([[id(p) for p in group["params"]] for group in again],
                         [[id(p) for p in group["params"]] for group in groups])
        self.assertEqual(sorted(net.state()), sorted(saved))

    def test_a_copy_made_before_keeps_its_own_forwards(self):
        """The playing copy, made before the learner's blocks are given
        their GEMMs: none of its convolutions given them, nothing of them
        traced, and its answers bit for bit as before."""
        net = self.net.train()
        actor = copy.deepcopy(net).requires_grad_(False).eval()
        with torch.no_grad():
            before = actor.decision(self.planes, self.legal)
        block_convolution.gemm_weight_gradients(net.mortal)
        self.assertFalse(any(worked_out(conv) for conv in block_convolutions(actor)))
        # Learning, as a copy that plays the learner's own rounds might.
        actor.requires_grad_(True).train()
        self.assertEqual(operators(self.learn(actor))["forward"].count(BLOCK_CONVOLUTION), 0)
        actor.requires_grad_(False).eval()
        with torch.no_grad():
            after = actor.decision(self.planes, self.legal)
        for index, (got, want) in enumerate(zip(after, before)):
            same_bits(self, got, want, str(index))

    def test_a_convolution_it_cannot_work_out_so_is_refused_and_a_subclass_left_alone(self):
        for conv in (nn.Conv1d(4, 4, 5, padding=2, bias=False), nn.Conv1d(4, 4, 3, padding=1),
                     nn.Conv1d(4, 4, 3, padding=1, bias=False, padding_mode="circular"),
                     nn.Conv1d(4, 4, 3, padding=1, stride=2, bias=False)):
            block = mortal_model.ResBlock(4)
            block.res_unit[2] = conv
            with self.subTest(conv=conv), self.assertRaises(ValueError):
                block_convolution.gemm_weight_gradients(block)

        class Own(nn.Conv1d):
            pass

        block = mortal_model.ResBlock(4)
        block.res_unit[2] = Own(4, 4, 3, padding=1, bias=False)
        self.assertEqual(block_convolution.gemm_weight_gradients(block), 1)
        self.assertEqual((worked_out(block.res_unit[2]), worked_out(block.res_unit[5])), (False, True))
        # Mortal's first and last convolutions sit outside its blocks.
        mortal = mortal_model.build(16, 2)
        self.assertEqual(block_convolution.gemm_weight_gradients(mortal), 4)
        self.assertEqual([worked_out(conv) for conv in other_convolutions(mortal)], [False, False])


class TrainerTests(unittest.TestCase):
    def test_the_trainer_gives_mortals_blocks_their_gemms_in_the_learner_alone_to_compile_its_learning(self):
        """With --compile-learning, every block convolution of the
        learner's Mortal and no other convolution, and none of the copy
        that plays the rounds, which is made before; with --compile, or
        nothing compiled, none at all."""
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
                self.assertEqual({worked_out(conv) for conv in block_convolutions(learner.mortal)}, {compiling})
                self.assertFalse(any(worked_out(conv) for conv in other_convolutions(learner)))
                for player in players:
                    self.assertEqual(player is learner, "--play-ahead" not in flags)
                    if player is not learner:
                        self.assertFalse(any(worked_out(conv) for conv in player.modules()
                                             if isinstance(conv, nn.Conv1d)))


@unittest.skipUnless(CUDA and TRITON, "cuBLAS and the compiler's kernels are the card's")
class OnTheCardTests(unittest.TestCase):
    """The operator on the card against cuDNN's weight gradient, and the
    learning step compiled as the trainer compiles it, in bfloat16, with
    the blocks' GEMMs and without."""

    def setUp(self):
        torch._dynamo.reset()
        forget_whether_triton_runs()

    def tearDown(self):
        torch._dynamo.reset()

    def test_on_the_card_its_weight_gradient_is_cudnns_but_for_the_order_of_its_sums(self):
        """At Mortal's block shape, 512 rows of 34 positions of 192
        channels, in bfloat16: cuDNN's answer to the bit but where the two
        roundings of the float32 sums fall apart, an ulp, or float32's own
        error where the sum all but cancels; as near the exact sum as
        cuDNN's; and the same answer each time."""
        generator = torch.Generator(device="cuda").manual_seed(7)
        x = nn.functional.mish(torch.randn(512, 192, 34, device="cuda", generator=generator)).to(torch.bfloat16)
        grad = (torch.randn(512, 192, 34, device="cuda", generator=generator) * 1e-3).to(torch.bfloat16)
        weight = (torch.randn(192, 192, 3, device="cuda", generator=generator) * 0.03).to(torch.bfloat16)
        cudnn = exact_weight_gradient(x, grad, weight)
        got = block_convolution.block_weight_gradient(x, grad)
        same_bits(self, block_convolution.block_weight_gradient(x, grad), got, "again")
        exact = exact_weight_gradient(x.double(), grad.double(), weight.double())
        scale = exact_weight_gradient(x.double().abs(), grad.double().abs(), weight.double())
        spacing = torch.exp2(torch.floor(torch.log2(cudnn.double().abs().clamp_min(1e-30))) - 7)
        gap = (got.double() - cudnn.double()).abs()
        self.assertGreater(float((got == cudnn).float().mean()), 0.95)
        self.assertEqual(int((gap > spacing * 1.0001 + 2e-6 * scale).sum()), 0)

        def beyond_one_rounding(answer):
            err = (answer.double() - exact).abs()
            half = 0.5 * torch.exp2(torch.floor(torch.log2(exact.abs().clamp_min(1e-30))) - 7)
            return float(((err - half).clamp_min(0) / scale).max())

        self.assertLess(beyond_one_rounding(got), max(2e-6, 4 * beyond_one_rounding(cudnn)))

    def test_compiled_on_the_card_it_learns_as_before_but_for_the_order_of_the_weights_sums(self):
        """The learning step compiled as the trainer compiles it, our
        network's and the head's convolutions channels-last, with the
        blocks' GEMMs and without: where it trains Mortal, the GEMMs work
        out the blocks' weight gradients and the step learns as it did, as
        nearly as the order of the sums allows; where it holds Mortal still,
        nothing of them runs and the step is the same to the bit. cuDNN
        picks its deterministic kernels here: some of its weight gradients,
        summed with atomics, are not the same bits twice."""
        def step(forward, net, planes, legal, actions):
            net.zero_grad(set_to_none=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, value, guessed = forward(planes, legal)
                loss = (-torch.log_softmax(logits.float(), dim=1).gather(1, actions[:, None]).mean()
                        + value.float().square().mean() + torch.log_softmax(guessed.float(), dim=2).mean())
                loss.backward()
                torch.cuda.synchronize()
            ran = {event.name for event in profile.events()}
            grads = {name: p.grad for name, p in net.named_parameters() if p.grad is not None}
            return (loss.detach(), logits.float(), value.float(), guessed.float()), grads, ran

        args = types.SimpleNamespace(compile=False, compile_learning=True)
        with compiler_settings_kept(), torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
            for mode in ("none", "mortal"):
                torch.manual_seed(3)
                plain = fusion().cuda()
                plain.set_mode(mode)
                plain.train()
                gemm = copy.deepcopy(plain)
                as_the_trainer_converts(plain)
                as_the_trainer_converts(gemm)
                self.assertEqual(block_convolution.gemm_weight_gradients(gemm.mortal), 4)
                planes = encoded_like(256, seed=4).dense("cuda", torch.bfloat16)
                legal = torch.rand(256, 46, device="cuda") < 0.4
                legal[:, 0] = True
                actions = torch.multinomial(legal.float(), 1).squeeze(1)
                want, want_grads, _ = step(train_combined.learning_forward(plain, args), plain, planes, legal,
                                           actions)
                torch._dynamo.reset()
                got, got_grads, ran = step(train_combined.learning_forward(gemm, args), gemm, planes, legal, actions)
                torch._dynamo.reset()
                blocks = block_weights(plain)
                with self.subTest(mode=mode):
                    self.assertEqual("mahjong::block_weight_gradient" in ran, mode != "mortal")
                    self.assertEqual("mahjong::block_convolution" in ran, mode != "mortal")
                    self.assertEqual(sorted(got_grads), sorted(want_grads))
                    self.assertEqual(blocks <= set(want_grads), mode != "mortal")
                    if mode == "mortal":
                        for name, one, two in zip(("loss", "logits", "value", "hands"), got, want):
                            same_bits(self, one, two, name)
                        for name in want_grads:
                            same_bits(self, got_grads[name], want_grads[name], name)
                        continue
                    for name, one, two in zip(("loss", "logits", "value", "hands"), got, want):
                        torch.testing.assert_close(one, two, rtol=1e-4, atol=1e-5, msg=name)
                    for name in want_grads:
                        one, two = got_grads[name].double(), want_grads[name].double()
                        if float(two.norm()) == 0.0:
                            continue
                        self.assertGreater(float(nn.functional.cosine_similarity(one.reshape(-1), two.reshape(-1),
                                                                                  dim=0)), 0.9999, name)
                        self.assertLess(abs(float(one.norm() / two.norm()) - 1.0), 1e-3, name)


if __name__ == "__main__":
    unittest.main()
