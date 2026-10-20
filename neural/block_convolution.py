"""Mortal's block convolutions in the joined player's compiled learning
step, with their weight gradient worked out as GEMMs (see
`gemm_weight_gradients`).

The weight gradient of one of Mortal's block convolutions sums, for each
of the kernel's three taps, the gradient at every position of every row
against the input one tap along: for each tap a (192 x R)(R x 192)
product, R the positions of a minibatch, 2,048 rows of 34. cuDNN's kernel
for that shape tiles the 192 x 576 answer of all three taps 128 x 256, six
thread blocks on the 142 SMs of an L40S, and takes 0.54 ms a convolution,
80 of them an update: 46 of the 266 ms of kernels an update in mode none,
in each of the four modes that train Mortal.

Laid out as rows of positions, (positions, channels), each row's positions
padded with zeros as the convolution pads them, the input's taps are views
of one tensor offset by a position each, and the three products are plain
GEMMs over K = 73,728 positions, which cuBLAS splits into pieces that fill
the card. Asked for its answer in float32, it sums the pieces in float32
too, and the answer is rounded to bfloat16 once, as cuDNN's is: the same
sums as cuDNN's, in another order. Answered in bfloat16 instead, as torch's
default lets it, cuBLAS kept its partial sums in bfloat16, and a gradient
stood 2.8 bfloat16 ulps from the exact one on average, against cuDNN's
0.27.

On an L40S under the cloud's torch, in Mortal's block shape, in ms a
convolution with the inputs in the card's cache and not (20 October 2026):

- cuDNN's weight gradient as the learner runs it, 0.65 and 0.63, of which
  0.54 is its kernel and 0.12 its transposes of the input and the gradient
  into the layout it works in;
- the three GEMMs, 0.12 and 0.16; the copies of the input and the gradient
  into rows, which take the place of cuDNN's transposes, 0.14; together
  0.25 and 0.26;
- the convolution's whole backward, 0.83 and 0.84 as cuDNN works it out,
  and 0.40-0.42 less with the weight gradient from the GEMMs.

Against the exact sums in float64, the answer stood within a rounding of a
float32 sum as near the exact one as cuDNN's (1.2e-7 of the sum of the
absolute products beyond the half ulp a rounding costs, against cuDNN's
4.1e-7); 98.9% of it was cuDNN's to the bit and the rest an ulp away, or,
where the sum all but cancels, within float32's own error; and in float32
throughout, with TF32 off, the GEMMs and cuDNN stood 2.6e-7 of that sum
apart. On the learning step's own tensors, from a005's gen-1240, it stood
nearer the exact sums than cuDNN's in each of the 80 convolutions (2.5e-6
of that sum beyond a rounding at most, against cuDNN's 7.6e-6), and 99.3%
of it was cuDNN's to the bit.

The convolution's forward and its input gradient are cuDNN's, the same
calls on the same tensors as the module's own, so they are its answers to
the bit. The forward and the weight gradient are custom operators, each
one node in the graphs the compiler makes, which it calls as they are
(their fake implementations tell it the shapes): the compiler neither
traces into the GEMMs nor puts its own in their place. It still compiles
what lies around them anew, so a step's other last bits may move with it:
on a005's network they did not, in any of the six modes, but on a small
test network one mode's logits moved by 3e-8.
"""

import types

import torch
from torch import nn

from .mortal_model import ResBlock


@torch.library.custom_op("mahjong::block_convolution", mutates_args=())
def block_convolution(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """One of Mortal's block convolutions, a kernel of three padded by one
    with no bias: the module's own forward, cuDNN's on the card."""
    return nn.functional.conv1d(x, weight, None, 1, 1).contiguous()


@block_convolution.register_fake
def _block_convolution_shape(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape[0], weight.shape[0], x.shape[2])


@torch.library.custom_op("mahjong::block_weight_gradient", mutates_args=())
def block_weight_gradient(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """The weight gradient of a block convolution of `x` whose answer's
    gradient is `grad`: for each tap of the kernel, the gradient's rows of
    positions against the input's, one tap along, as one GEMM (see the
    module's notes). In the inputs' precision, its sums in float32."""
    rows, _channels, positions = x.shape
    span = rows * (positions + 2)
    # Row r of the gradient's layout meets row r + tap of the input's:
    # position p of a row against the input at p + tap - 1, a zero where
    # that falls off either end, as the convolution pads; the gradient's
    # two zero positions after each row of positions keep the next row's
    # input out of its sums, and the input's two zero rows after the last
    # keep the last views inside.
    inputs = _rows(x, before=1, after=1, extra=2)
    grads = _rows(grad, before=0, after=2, extra=0).t()
    taps = [_products(grads, inputs[tap:tap + span]) for tap in range(3)]
    return torch.stack(taps, dim=2).to(x.dtype)


@block_weight_gradient.register_fake
def _block_weight_gradient_shape(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    return x.new_empty(grad.shape[1], x.shape[1], 3)


def _rows(t: torch.Tensor, *, before: int, after: int, extra: int) -> torch.Tensor:
    """`t`, (rows, channels, positions), laid out as rows of positions,
    (rows * (before + positions + after) + extra, channels): each row's
    positions after `before` zeros and before `after`, and `extra` rows of
    zeros at the end."""
    rows, channels, positions = t.shape
    width = before + positions + after
    out = t.new_empty(rows * width + extra, channels)
    laid = out[: rows * width].view(rows, width, channels)
    if before:
        laid[:, :before].zero_()
    if after:
        laid[:, before + positions:].zero_()
    if extra:
        out[rows * width:].zero_()
    laid[:, before:before + positions].copy_(t.transpose(1, 2))
    return out


def _products(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b with its sums in float32: answered in float32 for operands in
    bfloat16 or float16, so that cuBLAS keeps the pieces it splits K into
    in float32 as well; in the operands' own precision otherwise."""
    if a.dtype in (torch.bfloat16, torch.float16):
        if a.is_cuda:
            return torch.mm(a, b, out_dtype=torch.float32)
        return torch.mm(a.float(), b.float())
    return torch.mm(a, b)


def _keep(ctx, inputs, output) -> None:
    x, weight = inputs
    ctx.save_for_backward(x, weight)


def _gradients(ctx, grad: torch.Tensor):
    x, weight = ctx.saved_tensors
    grad_x = grad_weight = None
    if ctx.needs_input_grad[0]:
        grad_x = torch.ops.aten.convolution_backward(
            grad, x, weight, None, [1], [1], [1], False, [0], 1, [True, False, False])[0]
    if ctx.needs_input_grad[1]:
        grad_weight = block_weight_gradient(x, grad)
    return grad_x, grad_weight


block_convolution.register_autograd(_gradients, setup_context=_keep)


def gemm_weight_gradients(module: nn.Module) -> int:
    """Has each convolution of each of Mortal's residual blocks in `module`
    work out its weight gradient as GEMMs inside a compiled forward that
    trains it (see the module's notes): the same convolution, its forward
    and its input gradient cuDNN's to the bit, its weight gradient the
    same sums in another order. Answers how many it changed.

    Everywhere else each convolution is the module's own: outside a
    compiled forward, without a gradient (the baseline pass, play), and
    where its weight is held still (a generation that holds Mortal still),
    so those forwards are traced, compiled and answered as before. A copy
    of `module` made before this keeps its own forwards, which is how the
    trainer leaves the playing copy as it was (see `train_combined.main`).
    The parameters, the state dict, a checkpoint and an optimiser see
    nothing new. Only a block's convolution as Mortal builds it, a kernel
    of three padded by one with zeros and no bias, can be worked out so;
    any other in a block is refused, and a subclass of Conv1d, whose
    forward may be its own, is left as it is."""
    changed = 0
    for block in module.modules():
        if not isinstance(block, ResBlock):
            continue
        for conv in block.res_unit:
            if type(conv) is not nn.Conv1d:
                continue
            shape = (conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, conv.padding_mode)
            if shape != ((3,), (1,), (1,), (1,), 1, "zeros") or conv.bias is not None:
                raise ValueError(f"cannot work out the weight gradient of {conv} as a block's GEMMs")
            conv.forward = types.MethodType(_gemm_forward, conv)
            changed += 1
    return changed


def _gemm_forward(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """`conv` over `x`, by `block_convolution` inside a compiled forward
    that trains it, and as itself everywhere else (see
    `gemm_weight_gradients`). Under autocast the input and the weight are
    cast as autocast casts them for a convolution, which it does not for an
    operator it does not know."""
    if not (torch.compiler.is_compiling() and torch.is_grad_enabled() and conv.weight.requires_grad):
        return nn.Conv1d.forward(conv, x)
    weight = conv.weight
    device = x.device.type
    if torch.is_autocast_enabled(device):
        precision = torch.get_autocast_dtype(device)
        x, weight = x.to(precision), weight.to(precision)
    return block_convolution(x, weight)
