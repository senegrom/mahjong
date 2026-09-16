"""Export a browser-compatible network to checked ONNX, quantised by default.

The browser uses Mortal's version-4 observations and action space. Legacy
engine-plane students are refused rather than silently replacing the playable
model. The policy, value and opponent-hand heads are exported; training-only
heads stay out.

A fused checkpoint is exported whole. It carries our own half under `model`
beside the fusion under `combined`, and reading the first alone gives a
network that was never a policy: in the joined head our half is weighed
about 0.02 against Mortal's 1.00, so alone it agrees with the network that
actually plays on about a fifth of its moves. The browser shipped exactly
that from 10 to 16 September 2026. The fusion's head reads the legality
mask, so its graph takes the mask as a second input.

Usage:
  python -m neural.export network.pt web/public/model-full.onnx
  python -m neural.export fused.pt web/public/model-full.onnx --float32
  python -m neural.export network.pt measured.onnx --float32 --allow-any-operator
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .model import PolicyValueNet

# Contract of the current browser's version-4 Mortal encoder and translator.
# Integration tests compare these to the native follower/model constants.
BROWSER_PLANES = 1012
BROWSER_ACTIONS = 46
POSITIONS = 34
OUTPUTS = ("policy", "value", "hands")
RUNTIME_OPS = Path(__file__).resolve().parent.parent / "web" / "runtime" / "reduced-ops.config"


def runtime_operators(config: Path = RUNTIME_OPS) -> set[str] | None:
    """What the browser's runtime can run, or None if the list is missing."""
    if not config.exists():
        return None
    for line in reversed(config.read_text(encoding="utf-8").strip().splitlines()):
        if ";" in line:
            return {name.strip() for name in line.split(";")[-1].split(",") if name.strip()}
    return None


def without_identities(model) -> int:
    """Drop quantiser identities, except those naming a graph output."""
    outputs = {value.name for value in model.graph.output}
    removed = 0
    while True:
        spare = next(
            (node for node in model.graph.node
             if node.op_type == "Identity" and not set(node.output) & outputs),
            None,
        )
        if spare is None:
            return removed
        source, target = spare.input[0], spare.output[0]
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == target:
                    node.input[index] = source
        model.graph.node.remove(spare)
        removed += 1


def quantise(source: Path, destination: Path) -> None:
    """The weights as int8, which is what the browser's runtime expects."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(source), str(destination), weight_type=QuantType.QInt8,
        # U8S8 kernels on AVX2/non-VNNI AVX512 saturate pairwise products
        # to int16. Seven-bit weights keep 2 * 255 * 64 within that range.
        # Keep the same browser operators and the strict check_runs guard.
        reduce_range=True,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    model = onnx.load(str(destination))
    if without_identities(model):
        onnx.save(model, str(destination))


def check_operators(destination: Path, insist: bool = True) -> None:
    """Refuse missing kernels unless explicitly exporting to widen the runtime."""
    import onnx

    allowed = runtime_operators()
    needed = {node.op_type for node in onnx.load(str(destination)).graph.node}
    print(f"operators: {', '.join(sorted(needed))}")
    if allowed is None:
        message = "no runtime operator list found; the browser's build was not checked"
        if insist:
            raise SystemExit(message)
        print(message)
        return
    missing = sorted(needed - allowed)
    if missing:
        message = (
            f"the browser's runtime does not carry {', '.join(missing)}; "
            "widen web/runtime/reduced-ops.config and rebuild it before shipping this"
        )
        if insist:
            raise SystemExit(message)
        print(message)


def validate_browser_model(net, payload: dict) -> None:
    """Validate both sides of the browser contract before touching the output."""
    if ("combined" in payload) != is_fusion(net):
        raise SystemExit("A fused checkpoint must be exported as the fusion, head and all")
    if net.planes != BROWSER_PLANES or net.actions != BROWSER_ACTIONS:
        raise SystemExit(
            f"The browser requires {BROWSER_PLANES} Mortal planes and {BROWSER_ACTIONS} "
            f"actions; this checkpoint has {net.planes} planes and {net.actions} actions"
        )


def is_fusion(net) -> bool:
    """Whether this is our network and a Mortal beneath one head."""
    return all(hasattr(net, part) for part in ("ours", "mortal", "fuse"))


def load_network(checkpoint: Path):
    from .model import from_payload

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "combined" in payload:
        # `model` in a fused checkpoint is our half alone, which is not the
        # network that plays; load the fusion rather than that half.
        from .combined import load as load_fusion

        net, _state = load_fusion(checkpoint, "cpu")
        return net.eval(), payload
    return from_payload(payload, "cpu", 192, 10).eval(), payload


def validation_positions() -> tuple[torch.Tensor, torch.Tensor]:
    """Sample real decisions through the same follower used by training.

    Four deterministic tables, sampled throughout 32 legal decisions rather
    than binary noise. The tables move using the engine's legal mask; the
    numerical comparison uses the follower's legal actions in Mortal space.
    """
    import riichi_py
    from .observe import Views

    games = 4
    arena = riichi_py.Arena(games=games, seed=4, bot_places=[])
    views = Views(arena, games, {"mortal"})
    rng = np.random.default_rng(4)
    positions, masks = [], []
    for step in range(32):
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        rows = np.flatnonzero(seats != 0xFF)
        if not len(rows):
            break
        views.advance()
        people = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
        players = people[rows, seats[rows]]
        if step % 4 == 0:
            sparse, legal = views.sparse_and_masks(rows, players)
            usable = legal.any(axis=1)
            if usable.any():
                positions.append(sparse.dense("cpu")[usable])
                masks.append(torch.from_numpy(legal[usable].copy()))
        legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, -1)
        actions = np.zeros(games, dtype=np.int64)
        for row in rows:
            actions[row] = rng.choice(np.flatnonzero(legal[row]))
        arena.step(actions.tolist())
    if not positions:
        raise SystemExit("No legal validation positions were collected")
    return torch.cat(positions), torch.cat(masks)


def check_runs(
    graph: Path, wrapped: torch.nn.Module, trial: torch.Tensor, legal: torch.Tensor
) -> None:
    """Check every exported head and legal-choice agreement on real positions.

    Require mean absolute error <= 10% of each head's reference standard
    deviation (with a 0.01 scale floor), and >= 90% best-legal-move agreement
    on decisions with a choice. These are export guards, not strength claims.
    """
    import onnxruntime

    if (trial.ndim != 3 or trial.shape[0] == 0
            or tuple(trial.shape[1:]) != (BROWSER_PLANES, POSITIONS)
            or legal.dtype != torch.bool
            or tuple(legal.shape) != (len(trial), BROWSER_ACTIONS)
            or not legal.any(dim=1).all() or not torch.isfinite(trial).all()):
        raise SystemExit("Invalid browser validation positions or legal masks")
    session = onnxruntime.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    names = [entry.name for entry in session.get_inputs()]
    if names not in (["planes"], ["planes", "legal"]):
        raise SystemExit("The browser requires an input named planes, and a fusion also legal")
    if [output.name for output in session.get_outputs()] != list(OUTPUTS):
        raise SystemExit("The browser requires policy, value and hands outputs in order")
    expected_chunks, answered_chunks = [], []
    # Browser-size batches also check the dynamic axis, without a large CPU peak.
    for start in range(0, len(trial), 4):
        chunk = trial[start:start + 4]
        allowed = legal[start:start + 4].cpu()
        arguments = (chunk,) if len(names) == 1 else (chunk, allowed.float())
        feed = {"planes": chunk.cpu().numpy()}
        if len(names) == 2:
            feed["legal"] = allowed.numpy().astype(np.float32)
        with torch.no_grad():
            expected = tuple(answer.detach().cpu().float() for answer in wrapped(*arguments))
        given = session.run(list(OUTPUTS), feed)
        if len(expected) != 3 or len(given) != 3:
            raise SystemExit("Expected all three playable heads")
        shapes = ((len(chunk), BROWSER_ACTIONS), (len(chunk), 1), (len(chunk), 3, POSITIONS))
        answered = []
        for name, ours, theirs, shape in zip(OUTPUTS, expected, given, shapes):
            theirs = torch.as_tensor(theirs).float()
            if tuple(ours.shape) != shape or tuple(theirs.shape) != shape:
                raise SystemExit(f"Wrong {name} output shape; expected {shape}")
            if name == "policy":
                # Only negative infinity on an illegal move is a mask sentinel.
                # Check each side first: matching NaN/+inf is never a valid mask,
                # and a corrupt legal logit must name the side that produced it.
                for side, logits in (("reference", ours), ("exported", theirs)):
                    valid = torch.isfinite(logits) | (torch.isneginf(logits) & ~allowed)
                    if not valid.all():
                        raise SystemExit(f"Non-finite {side} policy output outside masked moves")
                if not torch.equal(torch.isfinite(ours), torch.isfinite(theirs)):
                    raise SystemExit("The exported policy answers a different set of moves")
            else:
                if not torch.isfinite(ours).all():
                    raise SystemExit(f"Non-finite reference {name} output")
                if not torch.isfinite(theirs).all():
                    raise SystemExit(f"Non-finite exported {name} output")
            answered.append(theirs)
        expected_chunks.append(expected)
        answered_chunks.append(answered)
    expected = [torch.cat([chunk[i] for chunk in expected_chunks]) for i in range(3)]
    answered = [torch.cat([chunk[i] for chunk in answered_chunks]) for i in range(3)]
    for name, ours, theirs in zip(OUTPUTS, expected, answered):
        # Double precision keeps even extreme finite outputs from overflowing
        # the error calculation and evading comparison through a NaN metric.
        if name == "policy":
            keep = torch.isfinite(ours)
            ours, theirs = ours[keep], theirs[keep]
        error = float((theirs.double() - ours.double()).abs().mean())
        scale = max(float(ours.double().std(unbiased=False)), 0.01)
        print(f"{name}: mean error {error:.6f}, reference scale {scale:.6f}")
        if error > 0.1 * scale:
            raise SystemExit(f"The exported {name} differs too far from the reference")
    choices = legal.cpu().sum(dim=1) > 1
    if choices.any():
        ours = expected[0].reshape(len(legal), -1).masked_fill(~legal.cpu(), -torch.inf).argmax(dim=1)
        theirs = answered[0].reshape(len(legal), -1).masked_fill(~legal.cpu(), -torch.inf).argmax(dim=1)
        agreement = float((ours[choices] == theirs[choices]).float().mean())
        print(f"same best legal move on {agreement:.1%} of non-forced decisions")
        if agreement < 0.9:
            raise SystemExit("The exported policy changes too many best legal moves")


class PlayableFusion(torch.nn.Module):
    """The whole fusion: Mortal, our network and the head that joins them.

    The head reads the legality mask, so the graph takes it as a second
    input rather than inventing one; the browser has the mask already. It
    arrives as floats because a boolean tensor is one more thing for a
    caller to get wrong on the way in, and the value is reshaped to the
    one-a-row the browser's contract asks for. Moves the rules forbid come
    back as negative infinity, exactly as the network leaves them.
    """

    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

    def forward(self, planes: torch.Tensor, legal: torch.Tensor) -> tuple[torch.Tensor, ...]:
        logits, value, hands = self.net.everything(planes, legal > 0.5)
        return logits, value.reshape(-1, 1), hands


def playable(net) -> torch.nn.Module:
    """The exportable wrapper for whichever kind of network this is."""
    return (PlayableFusion(net) if is_fusion(net) else Playable(net)).eval()


class Playable(torch.nn.Module):
    """Policy, value and hidden-hand heads, without training-only critics."""

    def __init__(self, net: PolicyValueNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.net.tail(self.net.tower(self.net.stem(planes)))
        pooled = features.mean(dim=2)
        tiles = self.net.policy_tiles(features)
        tiles = tiles.reshape(tiles.shape[0], -1)
        policy = torch.cat([tiles, self.net.policy_pooled(pooled)], dim=1)
        value = self.net.value(pooled)
        hands = self.net.hands_from(planes, features)
        return policy, value, hands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--float32", action="store_true", help="For measuring, not the reduced browser runtime")
    parser.add_argument("--allow-any-operator", action="store_true", help="Export while widening the runtime's operator list")
    args = parser.parse_args()
    net, payload = load_network(args.checkpoint)
    validate_browser_model(net, payload)
    wrapped = playable(net)
    fused = is_fusion(net)
    trial, legal = validation_positions()
    # A fusion reads the mask its head was trained with, so the graph takes
    # it too; everything else reads only the planes.
    example = (trial[:1],) if not fused else (trial[:1], legal[:1].float())
    names = ["planes"] if not fused else ["planes", "legal"]
    if fused and not args.float32:
        # Eight bits cost the fused value head more than a tenth of its own
        # spread, and the browser plays this network rather than measuring it.
        raise SystemExit("Export a fusion with --float32; int8 blunts its value head")
    destination = args.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Validate in a sibling temporary directory. Atomic replacement preserves
    # an existing playable model on validation failures or interrupted copies.
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        full = Path(scratch) / "float32.onnx"
        torch.onnx.export(
            wrapped, example, str(full), input_names=names,
            output_names=list(OUTPUTS),
            dynamic_axes={name: {0: "batch"} for name in (*names, *OUTPUTS)},
            opset_version=17, dynamo=False,
        )
        made = full if args.float32 else Path(scratch) / "int8.onnx"
        if not args.float32:
            quantise(full, made)
        check_operators(made, not args.allow_any_operator)
        check_runs(made, wrapped, trial, legal)
        made.replace(destination)
    print(
        f"exported {args.checkpoint.name} ({'the fusion, ' if fused else ''}"
        f"{net.channels}x{net.blocks}, generation {payload.get('generation', 0)}) -> "
        f"{destination} ({destination.stat().st_size / 1e6:.1f} MB), {net.actions} actions"
    )


if __name__ == "__main__":
    main()
