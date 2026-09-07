"""Mortal's network, so a published Mortal can sit at our tables.

These classes are Mortal's (Equim-chan/Mortal, `mortal/model.py`, AGPL-3.0-
or-later, the licence this project shares), kept to the version-four
shapes the public third-party checkpoint uses: a residual tower with
channel attention over the thousand and twelve planes, batch-normalised
and Mish-activated, flattened to a vector of 1024, and a duelling Q head
over Mortal's forty-six actions. Nothing here is trained by this project;
it is the other player in the zoo, and a teacher.

The public weights are not part of the repository. `load` reads a
checkpoint file of the kind Mortal writes, with its config inside.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from torch import nn

from libriichi.consts import ACTION_SPACE, obs_shape


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16, actv_builder=nn.ReLU, bias: bool = True):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=bias),
            actv_builder(),
            nn.Linear(channels // ratio, channels, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(x.mean(-1))
        max_out = self.shared_mlp(x.amax(-1))
        weight = (avg_out + max_out).sigmoid()
        return weight.unsqueeze(-1) * x


class ResBlock(nn.Module):
    def __init__(self, channels: int, *, norm_builder=nn.Identity, actv_builder=nn.ReLU):
        super().__init__()
        # Pre-activation, as every version past the first has it.
        self.res_unit = nn.Sequential(
            norm_builder(),
            actv_builder(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            norm_builder(),
            actv_builder(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
        )
        self.ca = ChannelAttention(channels, actv_builder=actv_builder, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ca(self.res_unit(x)) + x


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        num_blocks: int,
        *,
        norm_builder=nn.Identity,
        actv_builder=nn.ReLU,
    ):
        super().__init__()
        blocks = [
            ResBlock(conv_channels, norm_builder=norm_builder, actv_builder=actv_builder)
            for _ in range(num_blocks)
        ]
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            *blocks,
            norm_builder(),
            actv_builder(),
            nn.Conv1d(conv_channels, 32, kernel_size=3, padding=1),
            actv_builder(),
            nn.Flatten(),
            nn.Linear(32 * 34, 1024),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Brain(nn.Module):
    """The encoder: planes in, a vector of 1024 out."""

    def __init__(self, *, conv_channels: int, num_blocks: int, version: int = 4):
        super().__init__()
        if version not in (3, 4):
            raise ValueError(f"only Mortal versions 3 and 4 are kept here, not {version}")
        self.version = version
        norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01, eps=1e-3)
        actv_builder = partial(nn.Mish, inplace=True)
        self.encoder = ResNet(
            in_channels=obs_shape(version)[0],
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            norm_builder=norm_builder,
            actv_builder=actv_builder,
        )
        self.actv = actv_builder()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actv(self.encoder(obs))


class DQN(nn.Module):
    """The duelling head: a value and an advantage per action, masked."""

    def __init__(self, *, version: int = 4):
        super().__init__()
        self.version = version
        if version == 4:
            self.net = nn.Linear(1024, 1 + ACTION_SPACE)
        elif version == 3:
            self.v_head = nn.Sequential(nn.Linear(1024, 256), nn.Mish(inplace=True), nn.Linear(256, 1))
            self.a_head = nn.Sequential(
                nn.Linear(1024, 256), nn.Mish(inplace=True), nn.Linear(256, ACTION_SPACE)
            )
        else:
            raise ValueError(f"only Mortal versions 3 and 4 are kept here, not {version}")

    def forward(self, phi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.version == 4:
            v, a = self.net(phi).split((1, ACTION_SPACE), dim=-1)
        else:
            v = self.v_head(phi)
            a = self.a_head(phi)
        a_sum = a.masked_fill(~mask, 0.0).sum(-1, keepdim=True)
        a_mean = a_sum / mask.sum(-1, keepdim=True)
        return (v + a - a_mean).masked_fill(~mask, -torch.inf)


class Mortal(nn.Module):
    """Brain and head together: planes and Mortal's action mask in, Q
    values over its forty-six actions out."""

    def __init__(self, brain: Brain, dqn: DQN) -> None:
        super().__init__()
        self.brain = brain
        self.dqn = dqn
        self.version = brain.version

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.dqn(self.brain(obs), mask)


def load(path: Path | str, device: str = "cuda") -> Mortal:
    """A Mortal from a checkpoint file of the kind Mortal writes: its
    config names the version and shape, `mortal` holds the brain and
    `current_dqn` the head."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    version = int(config["control"]["version"])
    brain = Brain(
        conv_channels=int(config["resnet"]["conv_channels"]),
        num_blocks=int(config["resnet"]["num_blocks"]),
        version=version,
    )
    dqn = DQN(version=version)
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    net = Mortal(brain, dqn).to(device).eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net
