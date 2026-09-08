"""Exports a trained network to ONNX, for the browser to play against.

The browser already carries the rules as WebAssembly; this gives it the
policy as well, so a person can play the trained opponents offline with
nothing on a server. Only the policy head is exported: choosing a move
needs the logits and the legality mask, not the value.

The weights are quantised to int8, which is both how the file stays small
enough to send to a phone and how it comes to use the operators the
browser has: `web/runtime` was compiled with only those the shipped network
needs, and quantising is what turns `Conv` and `Gemm` into `ConvInteger`
and `MatMulInteger`. What the graph asks for is checked against that list
before anything is written, so a network the page could not load is
refused here rather than failing silently in someone's browser.

Usage:
  python -m neural.export E:/tmp-claude/mahjong/run2/best.pt web/public/model.onnx
  python -m neural.export student.pt web/public/model-strong.onnx --float32
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

import riichi_py

from .model import from_payload

# The operator list the browser's runtime was built from.
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
    """Drops the identity nodes the quantiser leaves behind, rewiring the
    nodes that read them. One that feeds a graph output is left alone, since
    removing it would rename the output the page asks for."""
    outputs = {value.name for value in model.graph.output}
    removed = 0
    while True:
        spare = next(
            (
                node
                for node in model.graph.node
                if node.op_type == "Identity" and not set(node.output) & outputs
            ),
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
        str(source),
        str(destination),
        weight_type=QuantType.QInt8,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    model = onnx.load(str(destination))
    if without_identities(model):
        onnx.save(model, str(destination))


def check_operators(destination: Path) -> None:
    """Refuses a graph the browser's runtime could not load."""
    import onnx

    allowed = runtime_operators()
    needed = {node.op_type for node in onnx.load(str(destination)).graph.node}
    print(f"operators: {', '.join(sorted(needed))}")
    if allowed is None:
        print("no runtime operator list found; the browser's build was not checked")
        return
    missing = sorted(needed - allowed)
    if missing:
        raise SystemExit(
            f"the browser's runtime does not carry {', '.join(missing)}, so it "
            f"could not load this network. Either build it without what needs "
            f"them (channel attention brings ReduceMax and Sigmoid) or rebuild "
            f"the runtime in web/runtime with a wider operator list."
        )


class PolicyOnly(torch.nn.Module):
    """The network with the value head trimmed away.

    The mask is applied in the browser rather than here: an exported graph
    that fills masked entries with negative infinity is awkward to run, and
    the caller has the mask anyway.
    """

    def __init__(self, net: PolicyValueNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        features = self.net.tail(self.net.tower(self.net.stem(planes)))
        pooled = features.mean(dim=2)
        tiles = self.net.policy_tiles(features)
        tiles = tiles.reshape(tiles.shape[0], -1)
        rest = self.net.policy_pooled(pooled)
        return torch.cat([tiles, rest], dim=1)


def main() -> None:
    arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    if len(arguments) < 2:
        raise SystemExit("usage: python -m neural.export <checkpoint> <out.onnx> [--float32]")
    # Float32 is for measuring the exported graph, not for the page: the
    # runtime there has no kernels for it.
    as_float = "--float32" in sys.argv[1:]
    checkpoint = Path(arguments[0])
    destination = Path(arguments[1])
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net = from_payload(payload, "cpu", 192, 10)
    if net.kind != "engine":
        raise SystemExit(
            "the browser encodes the engine's planes, and this network sees "
            "Mortal's; it cannot be exported until Mortal's encoder runs there"
        )
    net.eval()

    wrapped = PolicyOnly(net).eval()
    example = torch.zeros(1, riichi_py.PLANES, riichi_py.POSITIONS)
    with torch.no_grad():
        reference = wrapped(example)

    # Written where it is asked for only once it has been checked, so a
    # refusal never leaves a file behind that someone could ship.
    with tempfile.TemporaryDirectory() as scratch:
        full = Path(scratch) / "float32.onnx"
        torch.onnx.export(
            wrapped,
            (example,),
            str(full),
            input_names=["planes"],
            output_names=["policy"],
            dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        made = full if as_float else Path(scratch) / "int8.onnx"
        if not as_float:
            quantise(full, made)
        check_operators(made)
        shutil.copyfile(made, destination)

    size = destination.stat().st_size
    print(
        f"exported {checkpoint.name} "
        f"({payload.get('channels', 192)}x{payload.get('blocks', 10)}, "
        f"generation {payload.get('generation', 0)}, "
        f"placement {payload.get('placement', float('nan')):.3f}) "
        f"-> {destination} ({size / 1e6:.1f} MB)"
    )
    print(f"output shape {tuple(reference.shape)}, {riichi_py.ACTIONS} actions")


if __name__ == "__main__":
    main()
