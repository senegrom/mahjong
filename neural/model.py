"""The network: one policy and one value head over a riichi position.

A position arrives as planes over the 34 tile kinds (see the engine's
`encoding` module), so the natural shape is a one-dimensional residual tower
along those 34 positions. Neighbouring positions are neighbouring ranks
within a suit, which is exactly the locality a convolution is good at: the
shapes that matter, a sequence, a wait on either side of a pair, a run of
three, are all short spans.

Two heads:

- **policy** over the flat action space, masked to what the rules allow;
- **value**, the seat's expected result from here, in units of the final
  score divided by ten thousand.
"""

from __future__ import annotations

import torch
from torch import nn

import riichi_py

PLANES = riichi_py.PLANES
POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS


class Residual(nn.Module):
    """A pre-activation residual block along the tile axis."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(torch.relu(self.norm1(x)))
        out = self.conv2(torch.relu(self.norm2(out)))
        return x + out


class PolicyValueNet(nn.Module):
    """The policy and value network."""

    def __init__(self, channels: int = 192, blocks: int = 10) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv1d(PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[Residual(channels) for _ in range(blocks)])
        self.tail = nn.Sequential(nn.BatchNorm1d(channels), nn.ReLU())

        # The policy reads both the per-tile features, which is where the
        # thirty-four discards live, and the pooled position, which is where
        # the calls and declarations live.
        self.policy_tiles = nn.Conv1d(channels, 2, 1)
        self.policy_pooled = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Linear(256, ACTIONS - 2 * POSITIONS),
        )
        self.value = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self, planes: torch.Tensor, legal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns masked policy logits and the value, one per position."""
        features = self.tail(self.tower(self.stem(planes)))
        pooled = features.mean(dim=2)

        # Two logits per tile: discard it, or discard it declaring riichi.
        # They line up with the first sixty-eight entries of the action space.
        tiles = self.policy_tiles(features)
        tiles = tiles.reshape(tiles.shape[0], -1)
        rest = self.policy_pooled(pooled)
        logits = torch.cat([tiles, rest], dim=1)
        logits = logits.masked_fill(~legal, float("-inf"))
        return logits, self.value(pooled).squeeze(1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(channels: int = 192, blocks: int = 10, device: str = "cuda") -> PolicyValueNet:
    net = PolicyValueNet(channels, blocks).to(device)
    return net
