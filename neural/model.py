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
# The three players a network may be asked to read, in relative seat order.
OPPONENTS = riichi_py.OPPONENTS


# Group normalisation rather than batch normalisation: the network acts
# and learns in the same loop, and a layer whose behaviour depends on which
# other positions happen to share the batch makes the policy that produced
# the data differ from the one being updated.
GROUPS = 8


class Residual(nn.Module):
    """A pre-activation residual block along the tile axis."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(GROUPS, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(torch.relu(self.norm1(x)))
        out = self.conv2(torch.relu(self.norm2(out)))
        return x + out


class PolicyValueNet(nn.Module):
    """The policy and value network."""

    def __init__(self, channels: int = 320, blocks: int = 20) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv1d(PLANES, channels, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[Residual(channels) for _ in range(blocks)])
        self.tail = nn.Sequential(nn.GroupNorm(GROUPS, channels), nn.ReLU())

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

        # What the three opponents are holding, a distribution over the 34
        # kinds for each. Read from the per-tile features rather than the
        # pooled position, because the answer is per tile: this asks, of
        # each kind, how much of that opponent's hand it makes up.
        self.hands = nn.Conv1d(channels, OPPONENTS, 1)

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

    def read_hands(self, planes: torch.Tensor) -> torch.Tensor:
        """What each opponent is holding, as logits over the 34 kinds.

        Shape (batch, 3, 34), in the same relative seat order the
        observation uses: row 0 is the player to the mover's right. Softmax
        over the last axis gives the distribution the label is written in.
        """
        features = self.tail(self.tower(self.stem(planes)))
        return self.hands(features)

    def everything(
        self, planes: torch.Tensor, legal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Policy, value and opponents' hands from one pass of the tower.

        Training wants all three and the tower is the expensive part, so it
        is run once rather than three times.
        """
        features = self.tail(self.tower(self.stem(planes)))
        pooled = features.mean(dim=2)
        tiles = self.policy_tiles(features)
        tiles = tiles.reshape(tiles.shape[0], -1)
        logits = torch.cat([tiles, self.policy_pooled(pooled)], dim=1)
        logits = logits.masked_fill(~legal, float("-inf"))
        return logits, self.value(pooled).squeeze(1), self.hands(features)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(channels: int = 320, blocks: int = 20, device: str = "cuda") -> PolicyValueNet:
    net = PolicyValueNet(channels, blocks).to(device)
    return net
