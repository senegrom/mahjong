"""Export a browser-compatible network to checked ONNX, quantised by default.

The browser uses Mortal's version-4 observations and action space. Legacy
engine-plane students and fused checkpoints need a different adapter and
are refused rather than silently replacing the playable model. The policy,
value and opponent-hand heads are exported; training-only heads stay out.

Usage:
  python -m neural.export network.pt web/public/model-full.onnx
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
    if "combined" in payload:
        raise SystemExit("A fused checkpoint needs a fusion exporter; refusing to drop its head")
    if net.planes != BROWSER_PLANES or net.actions != BROWSER_ACTIONS:
        raise SystemExit(
            f"The browser requires {BROWSER_PLANES} Mortal planes and {BROWSER_ACTIONS} "
            f"actions; this checkpoint has {net.planes} planes and {net.actions} actions"
        )


def load_network(checkpoint: Path):
    from .model import from_payload

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Reject fusion before loading its standalone subnetwork.
    if "combined" in payload:
        raise SystemExit("A fused checkpoint needs a fusion exporter; refusing to drop its head")
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
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "planes":
        raise SystemExit("The browser requires one input named planes")
    if [output.name for output in session.get_outputs()] != list(OUTPUTS):
        raise SystemExit("The browser requires policy, value and hands outputs in order")
    expected_chunks, answered_chunks = [], []
    # Browser-size batches also check the dynamic axis, without a large CPU peak.
    for start in range(0, len(trial), 4):
        chunk = trial[start:start + 4]
        with torch.no_grad():
            expected = tuple(answer.detach().cpu().float() for answer in wrapped(chunk))
        given = session.run(list(OUTPUTS), {"planes": chunk.cpu().numpy()})
        if len(expected) != 3 or len(given) != 3:
            raise SystemExit("Expected all three playable heads")
        shapes = ((len(chunk), BROWSER_ACTIONS), (len(chunk), 1), (len(chunk), 3, POSITIONS))
        answered = []
        for name, ours, theirs, shape in zip(OUTPUTS, expected, given, shapes):
            theirs = torch.as_tensor(theirs).float()
            if tuple(ours.shape) != shape or tuple(theirs.shape) != shape:
                raise SystemExit(f"Wrong {name} output shape; expected {shape}")
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
        error = float((theirs.double() - ours.double()).abs().mean())
        scale = max(float(ours.double().std(unbiased=False)), 0.01)
        print(f"{name}: mean error {error:.6f}, reference scale {scale:.6f}")
        if error > 0.1 * scale:
            raise SystemExit(f"The exported {name} differs too far from the reference")
    choices = legal.cpu().sum(dim=1) > 1
    if choices.any():
        ours = expected[0].masked_fill(~legal.cpu(), -torch.inf).argmax(dim=1)
        theirs = answered[0].masked_fill(~legal.cpu(), -torch.inf).argmax(dim=1)
        agreement = float((ours[choices] == theirs[choices]).float().mean())
        print(f"same best legal move on {agreement:.1%} of non-forced decisions")
        if agreement < 0.9:
            raise SystemExit("The exported policy changes too many best legal moves")


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
    wrapped = Playable(net).eval()
    trial, legal = validation_positions()
    example = trial[:1]
    destination = args.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Validate in a sibling temporary directory. Atomic replacement preserves
    # an existing playable model on validation failures or interrupted copies.
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        full = Path(scratch) / "float32.onnx"
        torch.onnx.export(
            wrapped, (example,), str(full), input_names=["planes"],
            output_names=list(OUTPUTS),
            dynamic_axes={name: {0: "batch"} for name in ("planes", *OUTPUTS)},
            opset_version=17, dynamo=False,
        )
        made = full if args.float32 else Path(scratch) / "int8.onnx"
        if not args.float32:
            quantise(full, made)
        check_operators(made, not args.allow_any_operator)
        check_runs(made, wrapped, trial, legal)
        made.replace(destination)
    print(
        f"exported {args.checkpoint.name} ({net.channels}x{net.blocks}, "
        f"generation {payload.get('generation', 0)}) -> {destination} "
        f"({destination.stat().st_size / 1e6:.1f} MB), {net.actions} actions"
    )


if __name__ == "__main__":
    main()
