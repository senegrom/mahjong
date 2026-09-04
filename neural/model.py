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
ORACLE_PLANES = riichi_py.ORACLE_PLANES
HIDDEN_HANDS_PLANES = riichi_py.HIDDEN_HANDS_PLANES


# Group normalisation rather than batch normalisation: the network acts
# and learns in the same loop, and a layer whose behaviour depends on which
# other positions happen to share the batch makes the policy that produced
# the data differ from the one being updated.
GROUPS = 8

# The oracle critic's own tower, which sees the position and the hidden
# planes together. A fraction of the main tower.
ORACLE_CHANNELS = 128
ORACLE_BLOCKS = 4

# The reader of hidden hands: the same shape, for the same reason.
READER_CHANNELS = 128
READER_BLOCKS = 4


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

        # The oracle critic. In training only, it also sees what the player
        # cannot: the opponents' concealed tiles, the draws to come and the
        # hidden indicators. Knowing them makes the return far less of a
        # surprise, so its estimate is far less noisy than the public value
        # head's, which makes it the better baseline for the policy gradient
        # and a quieter target for the public head to learn from.
        #
        # It has a small tower of its own that sees the position and the
        # hidden planes together. That is what lets it ask, per tile, the
        # one question it is for: whether the tile this seat is about to
        # throw is the tile an opponent is waiting on. A first version
        # pooled the hidden planes on their own and joined them to the
        # main tower's pooled features only at the end, which cannot say
        # that, and its loss sat on top of the public head's. The policy
        # never touches any of this and nothing at play time calls it.
        self.oracle_stem = nn.Sequential(
            nn.Conv1d(PLANES + ORACLE_PLANES, ORACLE_CHANNELS, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, ORACLE_CHANNELS),
            nn.ReLU(),
        )
        self.oracle_tower = nn.Sequential(
            *[Residual(ORACLE_CHANNELS) for _ in range(ORACLE_BLOCKS)]
        )
        self.oracle_tail = nn.Sequential(nn.GroupNorm(GROUPS, ORACLE_CHANNELS), nn.ReLU())
        self.oracle_value = nn.Sequential(
            nn.Linear(channels + ORACLE_CHANNELS, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # The reader of hidden hands: the learned distribution a search
        # weighs imagined worlds by. Shown the position and a set of three
        # hidden hands, it says how much more likely those hands are than
        # the proposal that deals from per-tile marginals would make them.
        # It learns that by telling the real hidden hands from imagined
        # ones during self-play, and what a well-trained discriminator's
        # logit converges to is exactly that likelihood ratio. What the
        # marginals miss is what it is for: shape, and the selection in
        # what an opponent kept. Nothing at play time in the browser calls
        # it; the search does.
        self.reader_stem = nn.Sequential(
            nn.Conv1d(PLANES + HIDDEN_HANDS_PLANES, READER_CHANNELS, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, READER_CHANNELS),
            nn.ReLU(),
        )
        self.reader_tower = nn.Sequential(
            *[Residual(READER_CHANNELS) for _ in range(READER_BLOCKS)]
        )
        self.reader_tail = nn.Sequential(nn.GroupNorm(GROUPS, READER_CHANNELS), nn.ReLU())
        self.reader = nn.Sequential(
            nn.Linear(READER_CHANNELS, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def read_plausibility(self, planes: torch.Tensor, hands: torch.Tensor) -> torch.Tensor:
        """How much more likely these hidden hands are, given the position,
        than the proposal made them: a logit per row, the log of the
        likelihood ratio once trained. `hands` holds HIDDEN_HANDS_PLANES
        planes, the three opponents' concealed tiles as unary counts in the
        observation's seat order, real or imagined."""
        together = torch.cat([planes, hands], dim=1)
        features = self.reader_tail(self.reader_tower(self.reader_stem(together)))
        return self.reader(features.mean(dim=2)).squeeze(1)

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

    def value_only(self, planes: torch.Tensor) -> torch.Tensor:
        """What each position is worth, in the reward's units, and nothing
        else. The search values thousands of positions a decision and
        wants none of the policy work for them."""
        features = self.tail(self.tower(self.stem(planes)))
        return self.value(features.mean(dim=2)).squeeze(1)

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

    def with_oracle(
        self, planes: torch.Tensor, legal: torch.Tensor, oracle: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Everything, and the oracle critic's value too, from one pass of
        the tower. For training; see the note on the oracle critic above."""
        features = self.tail(self.tower(self.stem(planes)))
        pooled = features.mean(dim=2)
        tiles = self.policy_tiles(features)
        tiles = tiles.reshape(tiles.shape[0], -1)
        logits = torch.cat([tiles, self.policy_pooled(pooled)], dim=1)
        logits = logits.masked_fill(~legal, float("-inf"))
        together = torch.cat([planes, oracle], dim=1)
        hidden = self.oracle_tail(self.oracle_tower(self.oracle_stem(together))).mean(dim=2)
        oracle_value = self.oracle_value(torch.cat([pooled, hidden], dim=1)).squeeze(1)
        return logits, self.value(pooled).squeeze(1), self.hands(features), oracle_value

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(channels: int = 320, blocks: int = 20, device: str = "cuda") -> PolicyValueNet:
    net = PolicyValueNet(channels, blocks).to(device)
    return net


def load_weights(net: PolicyValueNet, saved: dict[str, torch.Tensor]) -> None:
    """Loads a checkpoint into `net`, widening its first layer if the
    observation has grown planes since the checkpoint was saved.

    The new planes get zero weights, so the network plays exactly as it did
    until training teaches it what they mean. The engine only ever adds
    planes at the end of the observation, which is what makes this a pad
    rather than a shuffle.
    """
    key = "stem.0.weight"
    weight = saved[key]
    seen = weight.shape[1]
    if seen < PLANES:
        saved = dict(saved)
        pad = weight.new_zeros(weight.shape[0], PLANES - seen, weight.shape[2])
        saved[key] = torch.cat([weight, pad], dim=1)
    elif seen > PLANES:
        raise ValueError(
            f"the checkpoint saw {seen} planes and the engine now makes {PLANES}"
        )
    # A checkpoint from before the network read the opponents' hands, or
    # before it had an oracle critic, or with an oracle critic of another
    # shape, has no such head or the wrong one. Those start fresh, and
    # everything else loads as saved; any other gap is still an error.
    fresh = net.state_dict()
    saved = dict(saved)
    for key, value in fresh.items():
        if key.startswith(("hands.", "oracle_", "reader")) and (
            key not in saved or saved[key].shape != value.shape
        ):
            saved[key] = value
    for key in [key for key in saved if key not in fresh]:
        if key.startswith("oracle_"):
            del saved[key]
    net.load_state_dict(saved)
