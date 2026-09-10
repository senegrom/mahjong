"""The network: one policy and one value head over a riichi position.

A position arrives as planes over the 34 tile kinds, so the natural shape is
a one-dimensional residual tower along those 34 positions. Neighbouring
positions are neighbouring ranks within a suit, which is exactly the
locality a convolution is good at: the shapes that matter, a sequence, a
wait on either side of a pair, a run of three, are all short spans.

Which planes is the network's to say. The lineage trained from September
2026 sees Mortal's thousand and twelve (see `observe.py`): what every hand is
waiting on, every discard in order, an efficiency lookahead. Older networks
see the engine's own ninety-seven, and a checkpoint says which, so both
kinds can sit at one table and be compared.

Each block also looks at the whole line at once: a channel attention of
Mortal's kind, which pools every channel over the 34 positions, by mean and
by maximum, and gates the channels by what it finds. A convolution of width
three needs many blocks to learn that a dora turned and a discard on the
far side of the board are the same fact; the pooling says so in one.

Two heads:

- **policy** over the flat action space, masked to what the rules allow;
- **value**, the seat's expected result from here, in the reward's units.
"""

from __future__ import annotations

import torch
from torch import nn

import riichi_py

from libriichi.consts import ACTION_SPACE as MORTAL_ACTIONS

from . import observe

# The engine's own planes, which the older networks see.
ENGINE_PLANES = riichi_py.PLANES
# Mortal's, which the current lineage sees.
MORTAL_PLANES = observe.PLANES
POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS
# The three players a network may be asked to read, in relative seat order.
OPPONENTS = riichi_py.OPPONENTS
ORACLE_PLANES = riichi_py.ORACLE_PLANES
HIDDEN_HANDS_PLANES = riichi_py.HIDDEN_HANDS_PLANES

# What a network built without saying is: the current lineage.
DEFAULT_CHANNELS = 320
DEFAULT_BLOCKS = 24


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

# The critic's own tower. It also reads the policy tower's pooled features,
# so it need not be large to be good; what it must not do is train them.
CRITIC_CHANNELS = 128
CRITIC_BLOCKS = 6

# The tower behind the head that reads the table, for the same reason.
BELIEF_CHANNELS = 128
BELIEF_BLOCKS = 4

# How much narrower the attention's bottleneck is than the channels.
ATTENTION_RATIO = 16


class Attention(nn.Module):
    """Channel attention over the whole line, as Mortal's blocks have it.

    Every channel is pooled over the 34 positions twice, by mean and by
    maximum, and one small network reads each pooled vector; their sum
    through a sigmoid is a gate per channel. It costs a few thousand
    parameters a block and gives every block the whole position.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        narrow = max(channels // ATTENTION_RATIO, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, narrow),
            nn.ReLU(),
            nn.Linear(narrow, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.mlp(x.mean(dim=2)) + self.mlp(x.amax(dim=2)))
        return x * gate.unsqueeze(2)


class Residual(nn.Module):
    """A pre-activation residual block along the tile axis, with the
    channel attention on its branch when asked for."""

    def __init__(self, channels: int, attention: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(GROUPS, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.attention = Attention(channels) if attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(torch.relu(self.norm1(x)))
        out = self.conv2(torch.relu(self.norm2(out)))
        if self.attention is not None:
            out = self.attention(out)
        return x + out


class PolicyValueNet(nn.Module):
    """The policy and value network."""

    def __init__(
        self,
        channels: int = DEFAULT_CHANNELS,
        blocks: int = DEFAULT_BLOCKS,
        planes: int = MORTAL_PLANES,
        attention: bool = True,
        actions: int = ACTIONS,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.planes = planes
        self.attention = attention
        self.actions = actions
        self.stem = nn.Sequential(
            nn.Conv1d(planes, channels, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[Residual(channels, attention) for _ in range(blocks)])
        self.tail = nn.Sequential(nn.GroupNorm(GROUPS, channels), nn.ReLU())

        # The policy reads both the per-tile features, which is where the
        # discards live, and the pooled position, which is where the calls
        # and declarations live. Our own space keeps two per tile, the
        # discard and the discard that declares riichi; Mortal's keeps one,
        # because there the declaration is a move of its own.
        per_tile = 2 if actions == ACTIONS else 1
        self.policy_tiles = nn.Conv1d(channels, per_tile, 1)
        self.policy_pooled = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Linear(256, actions - per_tile * POSITIONS),
        )
        self.value = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # What the three opponents are holding, a distribution over the 34
        # kinds for each. Read per tile, because the answer is per tile:
        # this asks, of each kind, how much of that opponent's hand it
        # makes up. It has a tower of its own over the planes and reads
        # the policy tower's per-tile features without gradient. Its loss
        # is the largest the network carries, and while it trained the
        # policy's tower the policy flattened, a hundredth of entropy a
        # generation; trained alone, the policy sharpened. Nothing but the
        # policy loss trains the policy's tower now.
        self.belief_stem = nn.Sequential(
            nn.Conv1d(planes, BELIEF_CHANNELS, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, BELIEF_CHANNELS),
            nn.ReLU(),
        )
        self.belief_tower = nn.Sequential(
            *[Residual(BELIEF_CHANNELS, attention) for _ in range(BELIEF_BLOCKS)]
        )
        self.belief_tail = nn.Sequential(nn.GroupNorm(GROUPS, BELIEF_CHANNELS), nn.ReLU())
        self.hands = nn.Conv1d(channels + BELIEF_CHANNELS, OPPONENTS, 1)

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
            nn.Conv1d(planes + ORACLE_PLANES, ORACLE_CHANNELS, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, ORACLE_CHANNELS),
            nn.ReLU(),
        )
        self.oracle_tower = nn.Sequential(
            *[Residual(ORACLE_CHANNELS, attention) for _ in range(ORACLE_BLOCKS)]
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
            nn.Conv1d(planes + HIDDEN_HANDS_PLANES, READER_CHANNELS, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, READER_CHANNELS),
            nn.ReLU(),
        )
        self.reader_tower = nn.Sequential(
            *[Residual(READER_CHANNELS, attention) for _ in range(READER_BLOCKS)]
        )
        self.reader_tail = nn.Sequential(nn.GroupNorm(GROUPS, READER_CHANNELS), nn.ReLU())
        self.reader = nn.Sequential(
            nn.Linear(READER_CHANNELS, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # The critic: the value the search reads and the baseline the policy
        # gradient is measured against, on a tower of its own. The value
        # head above shares the policy's tower, and training that tower on
        # a ring of old rounds for the value's sake drifted the features
        # the policy reads from, however hard the policy's output was held:
        # entropy crept up every generation the ring was in use and stood
        # still when it was not. The critic reads the policy tower's pooled
        # features without gradient and adds a tower of its own over the
        # planes, so it can be trained on anything, as the oracle and the
        # reader are, and the policy tower is trained by the policy alone.
        self.critic_stem = nn.Sequential(
            nn.Conv1d(planes, CRITIC_CHANNELS, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, CRITIC_CHANNELS),
            nn.ReLU(),
        )
        self.critic_tower = nn.Sequential(
            *[Residual(CRITIC_CHANNELS, attention) for _ in range(CRITIC_BLOCKS)]
        )
        self.critic_tail = nn.Sequential(nn.GroupNorm(GROUPS, CRITIC_CHANNELS), nn.ReLU())
        self.critic = nn.Sequential(
            nn.Linear(channels + CRITIC_CHANNELS, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    @property
    def speaks_mortal(self) -> bool:
        """Whether its moves are Mortal's, which decides how it is asked
        for one: a riichi there names no tile until it is asked again."""
        return self.actions == MORTAL_ACTIONS

    @property
    def kind(self) -> str:
        """Which planes this network sees: `mortal` or `engine`. See
        `observe.Views`, which serves either."""
        return "mortal" if self.planes == MORTAL_PLANES else "engine"

    def hands_from(self, planes: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """What each opponent is holding, as logits over the 34 kinds, from
        the planes and the policy tower's per-tile features, which it reads
        but does not train."""
        own = self.belief_tail(self.belief_tower(self.belief_stem(planes)))
        return self.hands(torch.cat([features.detach(), own], dim=1))

    def critic_value(self, planes: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        """The critic's value of each position, from the planes and the
        policy tower's pooled features, which it reads but does not train."""
        own = self.critic_tail(self.critic_tower(self.critic_stem(planes))).mean(dim=2)
        return self.critic(torch.cat([pooled.detach(), own], dim=1)).squeeze(1)

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

    def value_only(self, planes: torch.Tensor, head: str = "critic") -> torch.Tensor:
        """What each position is worth, in the reward's units, and nothing
        else. The search values thousands of positions a decision and wants
        none of the policy work for them.

        Which head answers is the caller's to choose, because which one is
        the better judge is a question the training log asks every
        generation and does not always answer the same way. `critic` is the
        head with a tower of its own, trained on the ring of old rounds as
        well as the current one; `public` is the head that reads the policy
        tower's pooled features, which is the stronger representation and
        costs nothing extra here, the tower's pass being paid for already;
        `mean` averages them, which beats either whenever their errors are
        not the same errors."""
        features = self.tail(self.tower(self.stem(planes)))
        pooled = features.mean(dim=2)
        if head == "public":
            return self.value(pooled).squeeze(1)
        if head == "critic":
            return self.critic_value(planes, pooled)
        if head == "mean":
            return 0.5 * (self.critic_value(planes, pooled) + self.value(pooled).squeeze(1))
        raise ValueError(f"no such value head: {head}")

    def read_hands(self, planes: torch.Tensor) -> torch.Tensor:
        """What each opponent is holding, as logits over the 34 kinds.

        Shape (batch, 3, 34), in the same relative seat order the
        observation uses: row 0 is the player to the mover's right. Softmax
        over the last axis gives the distribution the label is written in.
        """
        features = self.tail(self.tower(self.stem(planes)))
        return self.hands_from(planes, features)

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
        return logits, self.value(pooled).squeeze(1), self.hands_from(planes, features)

    def with_oracle(
        self, planes: torch.Tensor, legal: torch.Tensor, oracle: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Everything, the oracle's value and the critic's, from one pass of
        the tower. For training; see the notes on the oracle and the critic
        above."""
        features = self.tail(self.tower(self.stem(planes)))
        pooled = features.mean(dim=2)
        tiles = self.policy_tiles(features)
        tiles = tiles.reshape(tiles.shape[0], -1)
        logits = torch.cat([tiles, self.policy_pooled(pooled)], dim=1)
        logits = logits.masked_fill(~legal, float("-inf"))
        together = torch.cat([planes, oracle], dim=1)
        hidden = self.oracle_tail(self.oracle_tower(self.oracle_stem(together))).mean(dim=2)
        # The oracle reads the tower's pooled features but does not train
        # them: its loss is as large as the value loss, and while it could
        # reach the tower the policy's entropy climbed from 0.36 to 0.49
        # over thirty generations. It learns on its own tower.
        oracle_value = self.oracle_value(torch.cat([pooled.detach(), hidden], dim=1)).squeeze(1)
        # The old value head reads the pooled features without gradient
        # too: it is a candidate baseline and a number to watch, not a
        # reason to move the tower.
        return (
            logits,
            self.value(pooled.detach()).squeeze(1),
            self.hands_from(planes, features),
            oracle_value,
            self.critic_value(planes, pooled),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def payload_fields(self) -> dict:
        """What a checkpoint records about the shape, so it can be rebuilt
        without being told."""
        return {
            "channels": self.channels,
            "blocks": self.blocks,
            "planes": self.planes,
            "attention": self.attention,
            "actions": self.actions,
        }


def build(
    channels: int = DEFAULT_CHANNELS,
    blocks: int = DEFAULT_BLOCKS,
    device: str = "cuda",
    planes: int = MORTAL_PLANES,
    attention: bool = True,
    actions: int = ACTIONS,
) -> PolicyValueNet:
    return PolicyValueNet(channels, blocks, planes, attention, actions).to(device)


def shape_of(payload: dict, channels: int | None = None, blocks: int | None = None) -> dict:
    """The shape a checkpoint was trained at, from what it says and, for
    checkpoints from before it said, from the weights themselves. Given
    `channels` or `blocks` they stand in where the checkpoint is silent."""
    weights = payload["model"]
    planes = payload.get("planes")
    if planes is None:
        # A network of the engine's kind sees however many planes the
        # engine makes now; a checkpoint from when it made fewer is padded
        # on loading. Only Mortal's count names itself.
        seen = int(weights["stem.0.weight"].shape[1])
        planes = MORTAL_PLANES if seen == MORTAL_PLANES else ENGINE_PLANES
    attention = payload.get("attention")
    if attention is None:
        attention = any(key.startswith("tower.0.attention.") for key in weights)
    found_channels = payload.get("channels")
    if found_channels is None:
        found_channels = channels or int(weights["stem.0.weight"].shape[0])
    found_blocks = payload.get("blocks")
    if found_blocks is None:
        found_blocks = blocks or (
            1 + max(int(key.split(".")[1]) for key in weights if key.startswith("tower."))
        )
    found_actions = payload.get("actions")
    if found_actions is None:
        # Older checkpoints predate the choice and are all of our own space;
        # the heads say so anyway, so read them rather than assume.
        per_tile = int(weights["policy_tiles.weight"].shape[0])
        rest = int(weights["policy_pooled.2.weight"].shape[0])
        found_actions = per_tile * POSITIONS + rest
    return {
        "channels": int(found_channels),
        "blocks": int(found_blocks),
        "planes": int(planes),
        "attention": bool(attention),
        "actions": int(found_actions),
    }


def from_payload(
    payload: dict, device: str = "cuda", channels: int | None = None, blocks: int | None = None
) -> PolicyValueNet:
    """A network of the shape a checkpoint was trained at, with its weights."""
    shape = shape_of(payload, channels, blocks)
    net = PolicyValueNet(**shape).to(device)
    load_weights(net, payload["model"])
    return net


def load_weights(net: PolicyValueNet, saved: dict[str, torch.Tensor]) -> None:
    """Loads a checkpoint into `net`, widening its first layer if the
    engine's observation has grown planes since the checkpoint was saved.

    The new planes get zero weights, so the network plays exactly as it did
    until training teaches it what they mean. The engine only ever adds
    planes at the end of the observation, which is what makes this a pad
    rather than a shuffle. Mortal's planes are not ours to grow, so for a
    network of that kind the count has to match.
    """
    key = "stem.0.weight"
    weight = saved[key]
    seen = weight.shape[1]
    if seen < net.planes and net.kind == "engine":
        saved = dict(saved)
        pad = weight.new_zeros(weight.shape[0], net.planes - seen, weight.shape[2])
        saved[key] = torch.cat([weight, pad], dim=1)
    elif seen != net.planes:
        raise ValueError(
            f"the checkpoint saw {seen} planes and this network sees {net.planes}"
        )
    # A checkpoint from before the network read the opponents' hands, or
    # before it had an oracle critic, or with an oracle critic of another
    # shape, has no such head or the wrong one. Those start fresh, and
    # everything else loads as saved; any other gap is still an error.
    fresh = net.state_dict()
    saved = dict(saved)
    for key, value in fresh.items():
        if key.startswith(("hands.", "oracle_", "reader", "critic", "belief_")) and (
            key not in saved or saved[key].shape != value.shape
        ):
            saved[key] = value
    for key in [key for key in saved if key not in fresh]:
        if key.startswith("oracle_"):
            del saved[key]
    net.load_state_dict(saved)
