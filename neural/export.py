"""Exports a trained network to ONNX, for the browser to play against.

The browser already carries the rules as WebAssembly; this gives it the
policy as well, so a person can play the trained opponents offline with
nothing on a server. Only the policy head is exported: choosing a move
needs the logits and the legality mask, not the value.

Usage:
  python -m neural.export E:/tmp-claude/mahjong/run2/best.pt web/public/model.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import riichi_py

from .model import PolicyValueNet, load_weights


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
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m neural.export <checkpoint> <out.onnx>")
    checkpoint = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net = PolicyValueNet(payload.get("channels", 192), payload.get("blocks", 10))
    load_weights(net, payload["model"])
    net.eval()

    wrapped = PolicyOnly(net).eval()
    example = torch.zeros(1, riichi_py.PLANES, riichi_py.POSITIONS)
    with torch.no_grad():
        reference = wrapped(example)

    torch.onnx.export(
        wrapped,
        (example,),
        str(destination),
        input_names=["planes"],
        output_names=["policy"],
        dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

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
