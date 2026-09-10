"""Two players' last layers joined by a third: a fusion head.

Mortal ends in its encoder's vector of 1024 (M2) and the Q values over its
forty-six actions that a linear head reads from it (M1). Our network ends
in its tower's features, pooled over the line and per tile (A2), and the
seventy-eight masked logits its heads read from them (A1). Both see the
same planes. This puts one more layer, F, over all four: a small network
that reads M2, M1, A2 and A1 together, with M1 also mapped onto our
actions through the zoo's translation so that F sees Mortal's value of
the very move it is choosing between, and both players' legality masks.

F's output is added to A1, and its last layers start at zero, so on the
first day the joined player plays exactly as our network does and learns
from there how much of Mortal to mix in, and where. The networks beneath
it train too, by the same self-play loop: each generation draws at random
which of the two stays fixed, Mortal, ours, or neither, while F and our
value head, the baseline, train every generation. A checkpoint carries
everything, so it loads wherever a player of ours does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from libriichi.consts import ACTION_SPACE as MORTAL_ACTIONS

from . import mortal_model, zoo
from .model import ACTIONS, PolicyValueNet, from_payload, load_weights

# For each of our actions, the one of Mortal's that means it.
OURS_TO_MORTAL = np.full(ACTIONS, zoo.MORTAL_PASS, dtype=np.int64)
for _mortal_action in range(MORTAL_ACTIONS):
    for _ours in zoo.meanings(_mortal_action):
        OURS_TO_MORTAL[_ours] = _mortal_action

# Mortal's Q values are rank points, a few units; scaled to about one for F.
Q_SCALE = 0.25


class Fuse(nn.Module):
    """The layer over both players' last layers."""

    def __init__(self, channels: int, phi: int = 1024, hidden: int = 512) -> None:
        super().__init__()
        width = phi + 2 * MORTAL_ACTIONS + 2 * ACTIONS + channels + 2 * ACTIONS
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ACTIONS),
        )
        # Two per tile, discard it or discard it with riichi, from our
        # per-tile features, which is where the thirty-four discards live.
        self.tile = nn.Conv1d(channels, 2, 1)
        # One number that adds Mortal's value of each of our moves straight
        # to our logits: the shortest road to Mortal's judgement, which the
        # network above can learn but a single weight finds in a few
        # updates. Zero at first, like the rest.
        self.mix = nn.Parameter(torch.zeros(()))
        # Nothing added at first: the joined player starts as our network.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.tile.weight)
        nn.init.zeros_(self.tile.bias)

    def forward(
        self,
        phi: torch.Tensor,
        q: torch.Tensor,
        q_mask: torch.Tensor,
        pooled: torch.Tensor,
        features: torch.Tensor,
        a1: torch.Tensor,
        legal: torch.Tensor,
    ) -> torch.Tensor:
        """Our seventy-eight logits, from A1 plus what F adds."""
        q_here = q[:, torch.from_numpy(OURS_TO_MORTAL).to(q.device)]
        mask_here = q_mask[:, torch.from_numpy(OURS_TO_MORTAL).to(q.device)] & legal
        finite_q = torch.where(q_mask, q, torch.zeros_like(q)) * Q_SCALE
        finite_here = torch.where(mask_here, q_here, torch.zeros_like(q_here)) * Q_SCALE
        finite_a1 = torch.where(legal, a1, torch.zeros_like(a1))
        inputs = torch.cat(
            [
                phi,
                finite_q,
                q_mask.float(),
                finite_here,
                mask_here.float(),
                pooled,
                finite_a1,
                legal.float(),
            ],
            dim=1,
        )
        delta = self.mlp(inputs)
        tiles = self.tile(features).reshape(features.shape[0], -1)
        delta = torch.cat([delta[:, : tiles.shape[1]] + tiles, delta[:, tiles.shape[1] :]], dim=1)
        # Mortal's value of a move it does not allow here adds nothing.
        straight = torch.where(mask_here, q_here, torch.zeros_like(q_here)) * self.mix
        return (finite_a1 + delta + straight).masked_fill(~legal, float("-inf"))


class Combined(nn.Module):
    """Our network and a Mortal beneath one fusion head."""

    kind = "mortal"

    #: What a generation may hold still: either network beneath the head,
    #: the head itself, or any combination of them written with a plus.
    #: Freezing a network without its head lets the head move the policy
    #: anyway, which is not what holding something still is for.
    PARTS = ("mortal", "ours", "head")
    MODES = ("none", "mortal", "ours", "head", "mortal+head", "ours+head")

    def __init__(self, ours: PolicyValueNet, mortal: mortal_model.Mortal) -> None:
        super().__init__()
        self.ours = ours
        self.mortal = mortal
        self.fuse = Fuse(ours.channels)
        self.mortal_config: dict = {}
        self.mode = "none"
        self.set_mode("none")
        self.ours.eval()
        self.mortal.eval()
        # The forward a trainer may replace with a compiled one.
        self.backbones_forward = self.backbones

    def always_trained(self) -> list[nn.Parameter]:
        """Our value head, the baseline the advantages are measured
        against, which every generation needs whatever else is held."""
        return list(self.ours.value.parameters())

    def ours_trained(self) -> list[nn.Parameter]:
        """Our network beneath F, apart from its value head."""
        value = {id(p) for p in self.ours.value.parameters()}
        return [p for p in self.ours.parameters() if id(p) not in value]

    def mortal_trained(self) -> list[nn.Parameter]:
        return list(self.mortal.parameters())

    def set_mode(self, mode: str) -> None:
        """What stays fixed this generation, named by its parts joined with
        a plus: `mortal`, `ours`, `head`, `mortal+head`, or `none`."""
        held = set() if mode == "none" else set(mode.split("+"))
        if not held <= set(self.PARTS):
            raise ValueError(f"no such mode: {mode}")
        self.mode = mode
        for parameter in self.ours_trained():
            parameter.requires_grad_("ours" not in held)
        for parameter in self.mortal_trained():
            parameter.requires_grad_("mortal" not in held)
        for parameter in self.fuse.parameters():
            parameter.requires_grad_("head" not in held)
        for parameter in self.always_trained():
            parameter.requires_grad_(True)

    @property
    def planes(self) -> int:
        return self.ours.planes

    @property
    def channels(self) -> int:
        return self.ours.channels

    @property
    def blocks(self) -> int:
        return self.ours.blocks

    def train(self, mode: bool = True) -> Combined:  # type: ignore[override]
        """Mortal's batch-normalisation statistics stay as it learned
        them, whatever else trains; our network has no such statistics."""
        super().train(mode)
        for module in self.mortal.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
        return self

    def backbones(self, planes: torch.Tensor, legal: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Both players' last layers, from the planes: phi and Q with
        Mortal's mask made from ours, and our features, pooled features,
        logits, value and reading of the opponents' hands."""
        ours = self.ours
        features = ours.tail(ours.tower(ours.stem(planes)))
        pooled = features.mean(dim=2)
        tiles = ours.policy_tiles(features)
        tiles = tiles.reshape(tiles.shape[0], -1)
        a1 = torch.cat([tiles, ours.policy_pooled(pooled)], dim=1).masked_fill(
            ~legal, float("-inf")
        )
        value = ours.value(pooled).squeeze(1)
        guessed = ours.hands_from(planes, features)
        # Mortal's mask, made from ours: what it may do here that our
        # engine allows, which is what the zoo defines its policy over.
        means = torch.from_numpy(zoo.MEANS_BY_OURS).to(legal.device)
        q_mask = (legal.float() @ means) > 0.5
        phi = self.mortal.features(planes)
        q = self.mortal.dqn(phi, q_mask)
        return phi, q, q_mask, pooled, features, a1, value, guessed

    def everything(
        self, planes: torch.Tensor, legal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi, q, q_mask, pooled, features, a1, value, guessed = self.backbones_forward(planes, legal)
        logits = self.fuse(phi.float(), q.float(), q_mask, pooled.float(), features.float(), a1.float(), legal)
        return logits, value, guessed

    def forward(self, planes: torch.Tensor, legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value, _guessed = self.everything(planes, legal)
        return logits, value

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state(self) -> dict:
        """A checkpoint that carries everything beneath F as well as F,
        Mortal's config included, which names its shape."""
        return {
            "combined": self.fuse.state_dict(),
            "model": self.ours.state_dict(),
            "planes": self.ours.planes,
            "attention": self.ours.attention,
            "channels": self.ours.channels,
            "blocks": self.ours.blocks,
            "config": self.mortal_config,
            **self.mortal.state(),
        }


def build(ours_path: Path | str, mortal_path: Path | str, device: str) -> tuple[Combined, dict]:
    """A joined player over the two checkpoints named, F starting at zero.
    Returns it with Mortal's config, which its checkpoints carry."""
    payload = torch.load(ours_path, map_location="cpu", weights_only=True)
    ours = from_payload(payload, device)
    state = torch.load(mortal_path, map_location="cpu", weights_only=False)
    mortal = mortal_model.build(**mortal_model.shape_of(state))
    mortal.brain.load_state_dict(state["mortal"])
    mortal.dqn.load_state_dict(state["current_dqn"])
    net = Combined(ours, mortal.to(device)).to(device)
    net.mortal_config = state["config"]
    return net, state["config"]


def load(path: Path | str, device: str) -> tuple[Combined, dict]:
    """A joined player from a checkpoint this module wrote."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    ours = PolicyValueNet(
        state["channels"], state["blocks"], state["planes"], state["attention"]
    ).to(device)
    load_weights(ours, state["model"])
    mortal = mortal_model.build(**mortal_model.shape_of(state))
    mortal.brain.load_state_dict(state["mortal"])
    mortal.dqn.load_state_dict(state["current_dqn"])
    net = Combined(ours, mortal.to(device)).to(device)
    net.mortal_config = state["config"]
    # A head saved before it had the straight road keeps its other weights
    # and starts that one at zero.
    net.fuse.load_state_dict(state["combined"], strict=False)
    return net, state


def is_combined(payload: dict) -> bool:
    return isinstance(payload, dict) and "combined" in payload
