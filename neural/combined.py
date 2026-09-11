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

from libriichi.consts import ACTION_SPACE as ACTIONS

from . import mortal_learner, mortal_model, zoo
from .model import GROUPS, OPPONENTS, POSITIONS, PolicyValueNet, from_payload, load_weights


class Fuse(nn.Module):
    """The three layers over both players' last two.

    `width` is F2's, and the head that reads it mirrors ours: one value a
    tile for the discards, the remaining twelve from the pooled part.
    """

    def __init__(self, channels: int, phi: int = 1024, width: int = 256) -> None:
        super().__init__()
        self.width = width
        # F2: Mortal's vector meets our per-tile features.
        self.phi_proj = nn.Sequential(nn.Linear(phi, width), nn.ReLU())
        self.tiles = nn.Sequential(
            nn.Conv1d(channels + width, width, 1, bias=False),
            nn.GroupNorm(GROUPS, width),
            nn.ReLU(),
            nn.Conv1d(width, width, 1, bias=False),
            nn.GroupNorm(GROUPS, width),
            nn.ReLU(),
        )
        # F1: its own answer over the same moves.
        self.own_tiles = nn.Conv1d(width, 1, 1)
        self.own_pooled = nn.Sequential(
            nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, ACTIONS - POSITIONS)
        )
        # F0: what is played, weighed action by action across the three,
        # with a correction the alignment alone cannot make.
        self.weights = nn.Parameter(torch.zeros(3, ACTIONS))
        self.bias = nn.Parameter(torch.zeros(ACTIONS))
        self.correction = nn.Sequential(
            nn.Linear(4 * ACTIONS, 256), nn.ReLU(), nn.Linear(256, ACTIONS)
        )
        # What a position is worth, and what the other three are holding,
        # with Mortal's vector added to what our network alone could see.
        # Both start silent, so the fusion begins by judging and reading
        # exactly as our network did and is never worse for the change.
        self.value_fix = nn.Sequential(nn.Linear(phi, 256), nn.ReLU(), nn.Linear(256, 1))
        self.hands_fix = nn.Sequential(nn.Linear(phi, width), nn.ReLU())
        self.hands_out = nn.Conv1d(width, OPPONENTS, 1)
        # F1 starts silent, not absent: its last layers are zero, so it
        # adds nothing to the first move played, while its weight below is
        # small and not zero, so a gradient reaches F2 from the first step.
        # Born the other way round, a live layer under a zero weight, it
        # can never learn: the zero weight passes nothing back, and a
        # random layer under a rising weight is only noise, which is why
        # the weight on it sat at 0.0003 for twenty generations.
        nn.init.zeros_(self.own_tiles.weight)
        nn.init.zeros_(self.own_tiles.bias)
        nn.init.zeros_(self.own_pooled[-1].weight)
        nn.init.zeros_(self.own_pooled[-1].bias)
        with torch.no_grad():
            # Mortal alone, with our own logits only loud enough to order
            # what its values leave tied.
            self.weights[0].fill_(0.05)
            self.weights[1].fill_(1.0)
            self.weights[2].fill_(0.02)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        nn.init.zeros_(self.value_fix[-1].weight)
        nn.init.zeros_(self.value_fix[-1].bias)
        nn.init.zeros_(self.hands_out.weight)
        nn.init.zeros_(self.hands_out.bias)

    def judge(self, phi: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """What the position is worth to the fusion: our network's answer
        with what Mortal sees added to it.

        Mortal's vector is read, not trained: judging a position and reading
        a hand are worth doing well, but not at the price of reshaping the
        encoder that nearly all of the play comes from. Our own network
        holds its per-tile features back from its reader for the same
        reason.
        """
        return value + self.value_fix(phi.detach()).squeeze(1)

    def read_hands(self, phi: torch.Tensor, guessed: torch.Tensor) -> torch.Tensor:
        """What the three opponents are holding, likewise: our network's
        reading, corrected by Mortal's vector spread over the tiles, and
        likewise without training it."""
        spread = self.hands_fix(phi.detach()).unsqueeze(2).expand(-1, -1, guessed.shape[2])
        return guessed + self.hands_out(spread)

    def hidden(self, phi: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """F2, per tile and pooled, from Mortal's vector and our features."""
        spread = self.phi_proj(phi).unsqueeze(2).expand(-1, -1, features.shape[2])
        tiles = self.tiles(torch.cat([features, spread], dim=1))
        return tiles, tiles.mean(dim=2)

    def own(self, tiles: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        """F1: the fusion's own logits, read from F2."""
        per_tile = self.own_tiles(tiles)
        per_tile = per_tile.reshape(per_tile.shape[0], -1)
        return torch.cat([per_tile, self.own_pooled(pooled)], dim=1)

    def forward(
        self,
        phi: torch.Tensor,
        q: torch.Tensor,
        features: torch.Tensor,
        a1: torch.Tensor,
        legal: torch.Tensor,
    ) -> torch.Tensor:
        """The logits played, masked. `q` and `a1` are already over the same
        forty-six moves as everything else."""
        tiles, pooled = self.hidden(phi, features)
        f1 = self.own(tiles, pooled)
        # A value for a move that cannot be made says nothing; zero rather
        # than an infinity, which would poison the sum.
        mortal = torch.where(legal, q, torch.zeros_like(q))
        ours = torch.where(legal, a1, torch.zeros_like(a1))
        stacked = torch.stack([f1, mortal, ours], dim=1)
        joined = (stacked * self.weights.unsqueeze(0)).sum(dim=1) + self.bias
        joined = joined + self.correction(
            torch.cat([f1, mortal, ours, legal.float()], dim=1)
        )
        return joined.masked_fill(~legal, float("-inf"))


class Combined(nn.Module):
    """Our network and a Mortal beneath one fusion head."""

    kind = "mortal"
    actions = ACTIONS

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
        """The critic and the reading of the hands, both halves of each.

        The critic is the baseline the advantages are measured against and
        the value a search would judge a leaf by; the reading is what a
        search would deal the unseen tiles from. Neither is a policy, so
        holding the policy still is no reason to stop either learning, and
        both train every generation whatever else is held."""
        parts = [
            self.ours.value,
            self.ours.belief_stem,
            self.ours.belief_tower,
            self.ours.belief_tail,
            self.ours.hands,
            self.fuse.value_fix,
            self.fuse.hands_fix,
            self.fuse.hands_out,
        ]
        return [parameter for part in parts for parameter in part.parameters()]

    def ours_trained(self) -> list[nn.Parameter]:
        """Our network beneath F, apart from what always trains."""
        apart = {id(parameter) for parameter in self.always_trained()}
        return [p for p in self.ours.parameters() if id(p) not in apart]

    def head_trained(self) -> list[nn.Parameter]:
        """The fusion's own weights, apart from what always trains."""
        apart = {id(parameter) for parameter in self.always_trained()}
        return [p for p in self.fuse.parameters() if id(p) not in apart]

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
        for parameter in self.head_trained():
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
        """Both players' last two layers, from the planes: Mortal's vector
        and its move values, our features and our logits over the same
        moves, with our value and our reading of the opponents' hands."""
        ours = self.ours
        features = ours.tail(ours.tower(ours.stem(planes)))
        pooled = features.mean(dim=2)
        tiles = ours.policy_tiles(features)
        tiles = tiles.reshape(tiles.shape[0], -1)
        # Z1, over the same moves Mortal answers: left unmasked, since the
        # head above weighs it and masks once at the end.
        a1 = torch.cat([tiles, ours.policy_pooled(pooled)], dim=1)
        value = ours.value(pooled).squeeze(1)
        guessed = ours.hands_from(planes, features)
        phi = self.mortal.features(planes)
        q = self.mortal.dqn(phi, legal)
        return phi, q, features, a1, value, guessed

    def everything(
        self, planes: torch.Tensor, legal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi, q, features, a1, value, guessed = self.backbones_forward(planes, legal)
        phi = phi.float()
        logits = self.fuse(phi, q.float(), features.float(), a1.float(), legal)
        return (
            logits,
            self.fuse.judge(phi, value.float()),
            self.fuse.read_hands(phi, guessed.float()),
        )

    def forward(self, planes: torch.Tensor, legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value, _guessed = self.everything(planes, legal)
        return logits, value

    @torch.no_grad()
    def choose(
        self, views, rows: np.ndarray, players: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        """Its best move per row, in our engine's actions. The same two
        steps as `decide`, with nothing recorded."""
        return zoo.choose_in_mortal_space(
            lambda who, fresh: self._ask(views, who, fresh), views, rows, players, legal
        )

    @torch.no_grad()
    def _ask(self, views, who: list[tuple[int, int]], fresh: bool = False):
        rows = np.array([game for game, _player in who], dtype=np.int64)
        players = np.array([player for _game, player in who], dtype=np.int64)
        sparse, masks = views.sparse_and_masks(rows, players, fresh=fresh)
        device = str(next(self.parameters()).device)
        mask = torch.from_numpy(masks).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits, _value = self.forward(sparse.dense(device), mask)
        return logits.float().cpu().numpy(), masks

    @torch.no_grad()
    def decide(
        self,
        views,
        rows: np.ndarray,
        players: np.ndarray,
        legal: np.ndarray,
        greedy: bool = False,
        explore_share: float = 0.0,
        wanderer=None,
    ):
        """One of our engine's actions per row, and what the policy decided
        to get there. Its moves are Mortal's, so a riichi is answered in two
        steps and both are recorded."""
        return mortal_learner.decide_in_mortal_space(
            lambda planes, mask: self.forward(planes, mask)[0],
            views,
            rows,
            players,
            legal,
            greedy,
            str(next(self.parameters()).device),
            explore_share=explore_share,
            wanderer=wanderer,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state(self) -> dict:
        """A checkpoint that carries everything beneath F as well as F,
        Mortal's config included, which names its shape."""
        return {
            "combined": self.fuse.state_dict(),
            "model": self.ours.state_dict(),
            "config": self.mortal_config,
            **self.ours.payload_fields(),
            **self.mortal.state(),
        }


def build(ours_path: Path | str, mortal_path: Path | str, device: str) -> tuple[Combined, dict]:
    """A joined player over the two checkpoints named, F starting at zero.
    Returns it with Mortal's config, which its checkpoints carry."""
    payload = torch.load(ours_path, map_location="cpu", weights_only=True)
    ours = from_payload(payload, device)
    if not ours.speaks_mortal:
        raise SystemExit(
            f"{ours_path} answers over {ours.actions} moves and Mortal over "
            f"{ACTIONS}; re-head it first with neural.rehead"
        )
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
    if "mix" in state.get("combined", {}):
        raise SystemExit(
            f"{path} carries the old fusion, which mapped one action space onto "
            "the other; those lineages are not resumable here"
        )
    ours = PolicyValueNet(
        state["channels"],
        state["blocks"],
        state["planes"],
        state["attention"],
        state.get("actions", ACTIONS),
    ).to(device)
    load_weights(ours, state["model"])
    mortal = mortal_model.build(**mortal_model.shape_of(state))
    mortal.brain.load_state_dict(state["mortal"])
    mortal.dqn.load_state_dict(state["current_dqn"])
    net = Combined(ours, mortal.to(device)).to(device)
    net.mortal_config = state["config"]
    # A head saved before it judged positions and read hands keeps
    # everything it had and starts those at nothing, which is where they
    # begin anyway: the fusion then judges and reads as our network does
    # and learns from there.
    net.fuse.load_state_dict(state["combined"], strict=False)
    return net, state


def is_combined(payload: dict) -> bool:
    return isinstance(payload, dict) and "combined" in payload
