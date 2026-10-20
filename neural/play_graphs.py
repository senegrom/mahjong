"""Play's small questions answered from CUDA graphs (`--play-graphs`).

A forward of the joined player launches about two thousand kernels on the
card, and a seated Mortal's about eleven hundred, each a few microseconds
of the processor's time: tens of milliseconds a question however few rows
it asks about. Most of a round's questions are small ones. The tables of a
round finish one by one, and its last few hundred steps are played by
fewer and fewer of them; a reach's tile is asked again of a handful of
rows; and each seated player is asked about its own share of the tables
only. A CUDA graph is a forward's kernels recorded once, for one size of
batch, and launched again all together in one call, so such a question
costs the card's time and little of the processor's.

A question is padded to the smallest size in `ROWS` that holds it, with
rows of planes that are all nought and all of whose moves are open, and is
answered by the first rows of the graph's answer. A bigger one, the
thousands of rows of a round's middle steps, is answered eagerly as
before: its kernels keep the card busy for longer than they take to
launch, and padding it would add to the card's work. So is one asked
without autocast in bfloat16 (see `Graphed`).

The graph answers exactly what the forward answers eagerly at the padded
size, bit for bit, since its kernels are the ones the forward chose at that
size, on the same inputs; no row of these networks reads another's, so
what the padding holds changes nothing of the rows asked. But that is not
always what the forward answers at the size asked: on the card a batch of
another size can be worked through by other kernels, which in bfloat16 can
move an answer in its last bits, as `--skip-forced` and `--preview-reach`
do, and a near tie can then turn. Played with trained weights on this
desktop's card (a joined player of the deployed shapes, 52,000 of its
decisions and 46,000 of its seated Mortal's), seven of the joined player's
rows in ten had a logit moved by a bit or two, its policy moved by 0.0005
in total variation on average and by 0.018 at most, and about one best
move in two thousand turned, for it and for the Mortal alike. So it is a
switch, off unless a trainer is told.

With `--compile-play` a question bigger than the largest graph is answered
by the forward compiled (`torch.compile`, the batch's size left symbolic)
instead of eagerly. On an L40S the joined player's forward at 2,048 rows
took 62.6 ms compiled against 121.2 eagerly, and a seated Mortal's 23.4
against 38.2 (20 October 2026), and once the encoder was made faster the
card was what held a generation up. The compiled forward answers in other
last bits than the eager one, as a graph's padded size can. It is compiled
and warmed with the graphs, before play begins (see `Graphed.prepare`), so
that no round compiles anything on its worker beside the learning.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import time

import torch
from torch import nn

from .observe import PLANES, POSITIONS
from .zoo import MORTAL_ACTIONS

#: The sizes of batch graphs are recorded for. A question is padded to the
#: smallest of them that holds it, and one bigger than the last is asked
#: eagerly. Below sixty-four rows the card takes about as long for any
#: number of them; above, each size is at most half as big again as the
#: one before, so padding adds at most half to the card's work. On this
#: desktop's card the joined player's forward takes the card longer than
#: its launching from about five hundred rows.
ROWS = (16, 32, 64, 96, 128, 192, 256, 384, 512)


@functools.cache
def capture_stream(device: torch.device) -> torch.cuda.Stream:
    """The stream graphs are recorded on, one a device. A graph is recorded
    on a stream other than the one the card's work is queued on, and is
    launched afterwards on whichever stream asks."""
    return torch.cuda.Stream(device)


class Graphed:
    """`forward(planes, mask)` of `module`, answered from CUDA graphs for a
    batch of at most `rows[-1]` rows on the card, asked under autocast in
    bfloat16 as play asks; on the processor, in another precision, with
    gradients, in training mode or bigger, eagerly as it was.

    Only bfloat16: the card's libraries keep a workspace of their own
    choosing for each convolution, and a graph keeps the workspace its
    convolutions were recorded with for as long as it lives. Under the
    cloud's torch 2.8 on this desktop's card, a float32 forward of a small
    network at 128 rows took a workspace of four and a half gigabytes, and
    its graph held twice that; in bfloat16 the joined player's graphs of
    every size held a third of a gigabyte.

    `forward` answers a tensor or a tuple of them, one row each for each
    row asked, as the deciding forwards of every player of Mortal's moves
    do. A graph is recorded for each size the first time it is asked for,
    or all of them at once by `prepare`, with autocast's cache of cast
    weights off: a cast that cache held would be read by the graph from
    where the cache kept it, which is let go of when the region ends. The
    forward is run eagerly at that size first, so that the libraries make
    their choices and their workspaces then and not while it is recorded.

    While a graph is being recorded, the card refuses every thread a wait
    for the whole card (`torch.cuda.synchronize`, or freeing memory to it),
    and the recording thread any wait at all. Waiting for a stream of its
    own, copying and allocating are left to other threads, so a learning
    step beside a recording goes on as it was; but the trainers record
    every graph before play begins (`graphed`), and none beside the
    learning (`--play-ahead`).

    A graph reads `module`'s weights where they are stored, so it answers
    by whatever they hold when it is launched: copying new weights in
    (`load_state_dict`), as a trainer does into its playing copy before
    each round, changes them where they are, and the graph answers by the
    new ones. Weights put somewhere else, as a cast to another precision
    puts them, would leave the graph reading where they were, so each
    question first checks that every parameter and buffer is where it was
    when the graphs were recorded, and records them again if one is not.
    A new process records its own.

    Each question is copied into the graph's own inputs, and its answer
    out of the graph's before it is handed back, since the next launch
    writes over it. A player's graphs share their memory, which is safe as
    each answer is copied out before another graph is launched.

    With `compiled`, a question bigger than the largest graph is answered by
    the forward compiled with the batch's size left symbolic, made and
    warmed by `prepare`, on the thread that records the graphs: only a
    question of the kind it was warmed with, as play asks (Mortal's planes
    in float32 and their masks), and never one of a single row, which the
    compiler would specialise and compile again. Anything else, a question
    before `prepare` and one after the weights were put somewhere else
    (`forget`) among them, is answered eagerly, so nothing is compiled
    where play asks, on a worker beside the learning (`--play-ahead`). So
    is one the compiled forward refuses because the main thread is tracing
    the learning step as it compiles it: FX keeps one flag for the whole
    process that a trace is under way, and a compiled forward asked while
    it stands raises before it runs anything. A copy compiles nothing: it
    answers its bigger questions eagerly."""

    def __init__(self, forward, module: nn.Module, rows: tuple[int, ...] = ROWS,
                 compiled: bool = False) -> None:
        self.forward = forward
        self.module = module
        self.rows = tuple(sorted(rows))
        self.compiled = compiled
        # The compiled forward once `prepare` has made it, and the kind of
        # question it was warmed with.
        self.big = None
        self.big_kind: tuple | None = None
        # Recorded graphs and the answers they write, by size and by the
        # kind of question.
        self.graphs: dict[tuple, tuple[torch.cuda.CUDAGraph, object]] = {}
        # The graphs' inputs, made once at the largest size for each kind
        # of question; a graph reads the first rows of them.
        self.inputs: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.pool = None
        # The stream the last question was asked on.
        self.stream: torch.cuda.Stream | None = None
        # Every parameter and buffer, and where each was stored when the
        # graphs were recorded.
        self.tensors: list[torch.Tensor] = []
        self.places: list[int] = []
        #: How long recording took, in seconds, and how many questions were
        #: answered from a graph; how long compiling took, how many were
        #: answered by the compiled forward, and how many it refused while
        #: another thread was tracing, answered eagerly instead.
        self.seconds = 0.0
        self.replays = 0
        self.compile_seconds = 0.0
        self.compiled_answers = 0
        self.set_aside = 0

    def __deepcopy__(self, memo) -> Graphed:
        """A copy of the module answers through graphs of its own, recorded
        when it is first asked: these read this module's weights. It
        compiles nothing, as that too would be where it is first asked."""
        return Graphed(copy.deepcopy(self.forward, memo), copy.deepcopy(self.module, memo), self.rows)

    def __call__(self, planes: torch.Tensor, mask: torch.Tensor):
        n = planes.shape[0]
        if (not planes.is_cuda or n == 0 or torch.is_grad_enabled()
                or self.module.training or not torch.is_autocast_enabled("cuda")
                or torch.get_autocast_dtype("cuda") != torch.bfloat16):
            return self.forward(planes, mask)
        device = planes.device
        kind = (planes.dtype, tuple(planes.shape[1:]), mask.dtype, tuple(mask.shape[1:]), device)
        if n > self.rows[-1]:
            if self.big is not None and kind == self.big_kind:
                try:
                    answer = self.big(planes, mask)
                except RuntimeError as error:
                    # FX's flag that a trace is under way is the process's,
                    # not the thread's: while the main thread compiles the
                    # learning step, each mode the first time it is drawn,
                    # the compiled forward refuses a worker's question
                    # before it runs anything, and it is answered eagerly.
                    if "symbolically trace" not in str(error):
                        raise
                    self.set_aside += 1
                else:
                    self.compiled_answers += 1
                    return answer
            return self.forward(planes, mask)
        rows = next(size for size in self.rows if size >= n)
        stream = torch.cuda.current_stream(device)
        if self.stream is not None and self.stream != stream:
            # The last question's graph, launched on another stream, reads
            # the inputs this one is about to write.
            stream.wait_stream(self.stream)
            for tensor in self.inputs.get(kind, ()):
                tensor.record_stream(stream)
        self.stream = stream
        if self.tensors and list(map(torch.Tensor.data_ptr, self.tensors)) != self.places:
            self.forget()
        if kind not in self.inputs:
            self.inputs[kind] = (
                planes.new_zeros((self.rows[-1], *planes.shape[1:])),
                mask.new_ones((self.rows[-1], *mask.shape[1:])),
            )
        held_planes, held_mask = self.inputs[kind]
        held_planes[:n].copy_(planes)
        held_planes[n:rows].zero_()
        held_mask[:n].copy_(mask)
        held_mask[n:rows].fill_(True)
        key = (rows, *kind)
        if key not in self.graphs:
            self.graphs[key] = self.record(held_planes[:rows], held_mask[:rows])
        graph, answer = self.graphs[key]
        graph.replay()
        self.replays += 1
        if isinstance(answer, torch.Tensor):
            return answer[:n].clone()
        return tuple(None if part is None else part[:n].clone() for part in answer)

    def prepare(self, device: str | torch.device) -> None:
        """Records a graph of every size now, of a question as play asks
        one: Mortal's planes in float32 and its masks, under autocast in
        bfloat16, the module in evaluation as play has it, and back as it
        was after. With `compiled`, compiles the forward for the bigger
        questions too and asks it twice, at two sizes above the graphs', so
        that it is compiled here, once, for every size play asks of it."""
        training = self.module.training
        self.module.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for rows in self.rows:
                    self(torch.zeros((rows, PLANES, POSITIONS), device=device),
                         torch.ones((rows, MORTAL_ACTIONS), dtype=torch.bool, device=device))
                if self.compiled and self.big is None:
                    began = time.perf_counter()
                    big = torch.compile(self.forward, dynamic=True)
                    for rows in (self.rows[-1] + 1, 2 * self.rows[-1] + 1):
                        planes = torch.zeros((rows, PLANES, POSITIONS), device=device)
                        mask = torch.ones((rows, MORTAL_ACTIONS), dtype=torch.bool, device=device)
                        big(planes, mask)
                    self.big = big
                    self.big_kind = (planes.dtype, tuple(planes.shape[1:]), mask.dtype, tuple(mask.shape[1:]),
                                     planes.device)
                    self.compile_seconds += time.perf_counter() - began
        finally:
            self.module.train(training)

    def record(self, planes: torch.Tensor, mask: torch.Tensor):
        """A graph of the forward over `planes` and `mask`, and the answer
        it writes when launched."""
        began = time.perf_counter()
        if not self.tensors:
            self.tensors = [*self.module.parameters(), *self.module.buffers()]
            self.places = list(map(torch.Tensor.data_ptr, self.tensors))
        device = planes.device

        def region():
            # Inside the caller's own region, so leaving it empties nothing.
            return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)

        with region():
            self.forward(planes, mask)
        here = torch.cuda.current_stream(device)
        side = capture_stream(device)
        side.wait_stream(here)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            try:
                with region():
                    answer = self.forward(planes, mask)
            except BaseException:
                # The forward's own error is the one to see, not the
                # recording's that it cut short.
                with contextlib.suppress(RuntimeError):
                    graph.capture_end()
                raise
            graph.capture_end()
        here.wait_stream(side)
        self.pool = graph.pool()
        self.seconds += time.perf_counter() - began
        return graph, answer

    def forget(self) -> None:
        """Lets every graph go, to be recorded again when next asked, and
        the compiled forward with them: weights put elsewhere, as a cast puts
        them, would have it compiled again, perhaps on a worker, so the
        bigger questions are answered eagerly from here on."""
        self.graphs.clear()
        self.pool = None
        self.tensors = []
        self.places = []
        self.big = None
        self.big_kind = None


def graphed(player, device: str | torch.device, amp: bool, rows: tuple[int, ...] = ROWS,
            compile: bool = False):
    """`player`, the forward it decides with answered from CUDA graphs
    (see `Graphed`) when it plays on the card in bfloat16: a joined
    player's (`Combined.deciding`), or a seated Mortal's or one of ours's
    that answers in Mortal's moves (`forward`). `amp` says whether play
    asks it in bfloat16, as self-play asks a learner with --amp and as a
    seated player always asks itself on the card (see
    `policy_inference.autocast`); one asked in float32 is left as it is,
    as are one of ours over the engine's seventy-eight moves and the
    heuristic player, which has no network. For a player whose forward
    runs eagerly: a compiled one is not recorded. `rows` are the sizes
    recorded; a ruler that asks every question at one size records that
    one (see `neural.ruler`). With `compile` the bigger questions are
    answered by the forward compiled (`--compile-play`).

    Every size is recorded here, and the forward compiled, before play
    begins and while nothing else uses the card, so that nothing is
    recorded or compiled beside the learning (see `Graphed`)."""
    if torch.device(device).type != "cuda" or not amp:
        return player
    if hasattr(player, "deciding"):
        player.deciding = forward = Graphed(player.decision, player, rows, compiled=compile)
    elif isinstance(getattr(player, "net", None), nn.Module) and hasattr(player, "forward"):
        player.forward = forward = Graphed(player.forward, player.net, rows, compiled=compile)
    else:
        return player
    forward.prepare(device)
    return player


def graphs_of(players) -> list[Graphed]:
    """The forwards among `players`' that answer from graphs (see
    `graphed`)."""
    found = []
    for player in players:
        for name in ("deciding", "forward"):
            forward = getattr(player, "__dict__", {}).get(name)
            if isinstance(forward, Graphed):
                found.append(forward)
    return found


def accounts(forwards: list[Graphed]) -> dict:
    """What a trainer says of `forwards`' graphs, as it begins and in each
    generation's record: how many play holds by now, the seconds this
    process has spent recording them, and the card memory they hold, which
    the learning step no longer has. Where play's bigger questions are
    compiled (`--compile-play`), the seconds this process spent compiling
    them and how many questions the compiled forwards have answered."""
    said = {
        "play_graphs": sum(len(forward.graphs) for forward in forwards),
        "graph_seconds": round(sum(forward.seconds for forward in forwards), 1),
        "graph_gb": round(held(forwards) / (1 << 30), 2),
    }
    if any(forward.compiled for forward in forwards):
        said["play_compile_seconds"] = round(sum(forward.compile_seconds for forward in forwards), 1)
        said["compiled_answers"] = sum(forward.compiled_answers for forward in forwards)
        said["compiled_set_aside"] = sum(forward.set_aside for forward in forwards)
    return said


def held(forwards: list[Graphed]) -> int:
    """The card memory `forwards`' graphs hold, in bytes: what recording
    them left in their pools, their answers and the libraries' workspaces
    among it, and their inputs."""
    pools = {tuple(forward.pool) for forward in forwards if forward.pool is not None}
    total = sum(tensor.numel() * tensor.element_size()
                for forward in forwards for pair in forward.inputs.values() for tensor in pair)
    if pools:
        total += sum(segment["total_size"] for segment in torch.cuda.memory_snapshot()
                     if tuple(segment.get("segment_pool_id", ())) in pools)
    return total
