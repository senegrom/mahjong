"""Two players' last layers joined by a third: a fusion head.

Mortal ends in its encoder's vector of 1024 (M2) and the Q values over its
forty-six moves that a linear head reads from it (M1). Our network,
re-headed into the same forty-six moves (`neural.rehead`; `build` refuses
any other), ends in its tower's features, per tile and pooled over the
line (Z2), and the logits its policy heads read from them (Z1). Both see
the same planes. The head, F, joins them in three layers. F2 reads
Mortal's vector and our per-tile features together. F1 is the fusion's
own logits over the same moves, read from F2. F0, what is played, weighs
F1, M1 and Z1 move by move, adds a bias and a correction read from all
three and the legality mask, and masks what the rules forbid.

The joined player is born as Mortal. F0's weights start at 0.05 on F1, 1.0
on M1 and 0.02 on Z1, and F1's last layers, the bias and the correction
start at zero, so on the first day its logits are Mortal's values, with
ours only loud enough to order what those leave tied. From there it learns
how much of each answer to use, and where. It judges a position and reads
the opponents' hands as our network does, plus a correction from Mortal's
vector that also starts at zero (`Fuse.judge`, `Fuse.read_hands`).

The networks beneath train too, by the same self-play loop
(`train_combined`). Each generation draws what stays fixed from
`train_combined --fixed`, by default the six `Combined.MODES`: nothing,
Mortal, ours, the head, or the head with either network beneath it. Three
of the six hold the head still. Only the critic and the reading of the
hands (`Combined.always_trained`) train whatever the draw, and a
generation that teaches the critic alone (`Combined.hold_all_but_value`)
holds even the reading still. A checkpoint carries everything, so it
loads wherever a player of ours does.
"""

from __future__ import annotations

from . import policy_inference

from pathlib import Path

import numpy as np
import torch
from torch import nn

from libriichi.consts import ACTION_SPACE as ACTIONS

from . import mortal_learner, mortal_model, zoo
from .model import (
    GROUPS, OPPONENTS, POSITIONS, PolicyValueNet, from_payload, load_weights,
    require_engine_observation,
)


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
        self.hands_out = nn.Linear(width, OPPONENTS * POSITIONS)
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
        reading, corrected by a tile-dependent projection of Mortal's vector,
        likewise without training its encoder."""
        correction = self.hands_out(self.hands_fix(phi.detach()))
        return guessed + correction.reshape(-1, OPPONENTS, POSITIONS)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = prefix + "hands_out.weight", prefix + "hands_out.bias"
        if weight in state_dict and state_dict[weight].ndim == 3:
            # Old Conv1d added one constant per opponent. Replicate that same
            # constant over tile-specific rows, preserving its probabilities.
            state_dict[weight] = state_dict[weight].squeeze(-1).repeat_interleave(POSITIONS, dim=0)
            state_dict[bias] = state_dict[bias].repeat_interleave(POSITIONS, dim=0)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
    #: Whether, seated as another player, it leaves rows with a single move
    #: open unasked; see `zoo.choose_in_mortal_space`.
    skip_forced = False
    #: Whether, seated as another player, it asks a reach's tile in the
    #: first question, from the follower's preview of the reach; see
    #: `zoo.choose_in_mortal_space`. As the learner it is told by `decide`.
    preview_reach = False
    #: Whether deciding keeps Mortal's vector of every decision in its
    #: records, for a generation that learns from it rather than running
    #: Mortal again (`train_combined --reuse-phi`). A trainer turns it on
    #: for the rounds that will read it.
    keep_phi = False
    #: What deciding and a seated player's questions ask instead of
    #: `decision`, when a trainer has small questions answered from CUDA
    #: graphs (`play_graphs.graphed`); None asks `decision` itself.
    deciding = None

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
        # Where a round's deciding went, in seconds, for the play record;
        # `selfplay.play` reads and clears it. Without it the learner's own
        # share of a round, a few minutes of a generation, was recorded
        # nowhere.
        self.timing = {"encode": 0.0, "translate": 0.0, "network": 0.0, "riichi": 0.0}

    def always_trained(self) -> list[nn.Parameter]:
        """The critic and the reading of the hands, both halves of each.

        The critic is the baseline the advantages are measured against; the
        reading is what the site shows of the opponents' hands. Neither is
        a policy, so holding the policy still is no reason to stop either
        learning, and both train whatever `set_mode` holds. Only a
        generation that teaches the critic alone (`hold_all_but_value`)
        holds the reading still as well."""
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

    def value_parts(self) -> list[nn.Parameter]:
        """The critic alone, both halves: our value head and the head's
        correction from Mortal's vector. Each reads features held back from
        it (`backbones`, `Fuse.judge`), so moving these moves no policy and
        no reading of the hands."""
        return [*self.ours.value.parameters(), *self.fuse.value_fix.parameters()]

    def hold_all_but_value(self) -> None:
        """Holds everything still but the critic (see `value_parts`), as a
        generation that only teaches the critic a new return does (see
        `train_combined --value-warm-up`). The next `set_mode` frees the
        rest again as its mode says."""
        self.mode = "value"
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.value_parts():
            parameter.requires_grad_(True)

    def set_mode(self, mode: str) -> None:
        """What stays fixed this generation, named by its parts joined with
        a plus: `mortal`, `ours`, `head`, `mortal+head`, `ours+head`, or
        `none`."""
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

    def backbones(
        self,
        planes: torch.Tensor,
        legal: torch.Tensor,
        hands: bool = True,
        phi: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Both players' last two layers, from the planes: Mortal's vector
        and its move values, our features and our logits over the same
        moves, with our value and our reading of the opponents' hands.
        `hands=False` leaves the reading out, None in its place, for a
        caller that wants only the moves and the value. `phi`, Mortal's
        vector of each row when it is already known, is taken as it is
        instead of being worked out from the planes again; only its move
        values are."""
        ours = self.ours
        features = ours.tail(ours.tower(ours.stem(planes)))
        pooled = features.mean(dim=2)
        tiles = ours.policy_tiles(features)
        tiles = tiles.reshape(tiles.shape[0], -1)
        # Z1, over the same moves Mortal answers: left unmasked, since the
        # head above weighs it and masks once at the end.
        a1 = torch.cat([tiles, ours.policy_pooled(pooled)], dim=1)
        value = ours.value(pooled.detach()).squeeze(1)
        guessed = ours.hands_from(planes, features) if hands else None
        if phi is None:
            phi = self.mortal.features(planes)
        q = self.mortal.dqn(phi, legal)
        return phi, q, features, a1, value, guessed

    def everything(
        self, planes: torch.Tensor, legal: torch.Tensor, phi: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The logits, the value and the reading of the hands. `phi`, when
        given, is Mortal's vector of each row, already known: in a
        generation that holds Mortal still it is the same function of the
        same planes all generation, so the learning step can take it from
        play instead of running Mortal on every minibatch (see
        `train_combined --reuse-phi`)."""
        if phi is None:
            phi, q, features, a1, value, guessed = self.backbones_forward(planes, legal)
        else:
            phi, q, features, a1, value, guessed = self.backbones_forward(planes, legal, phi=phi)
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

    def decision(
        self, planes: torch.Tensor, legal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """What deciding asks of `everything`: the logits and the value,
        and Mortal's vector beneath both, without the reading of the
        opponents' hands, which only the learning step reads. That reading
        is a tower of its own over the planes and the head above it, a few
        per cent of the forward. The logits and the value come from the
        same kernels on the same inputs either way, so they are bit for bit
        `everything`'s.

        A forward a trainer has compiled (`backbones_forward`) is asked for
        all of it, as it always was: a graph without the hands would be
        another graph, and the compiler's answers from it can differ in
        their last bits.

        Whether it has is asked of the function the forward calls, which
        is the networks' own method's while nothing has been put in its
        place. Asked by comparing the two bound methods, as it was, the
        compiler of the cloud's torch 2.8 could not tell, broke its graph
        there and compiled what followed a frame at a time, the networks
        beneath, the head and the judge each a graph of their own: so play's
        bigger questions were answered with --compile-play."""
        if getattr(self.backbones_forward, "__func__", None) is Combined.backbones:
            phi, q, features, a1, value, _guessed = self.backbones(planes, legal, hands=False)
        else:
            phi, q, features, a1, value, _guessed = self.backbones_forward(planes, legal)
        wide = phi.float()
        logits = self.fuse(wide, q.float(), features.float(), a1.float(), legal)
        return logits, self.fuse.judge(wide, value.float()), phi

    @torch.no_grad()
    def choose(
        self, views, rows: np.ndarray, players: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        """Its best move per row, in our engine's actions. The same two
        steps as `decide`, with nothing recorded."""
        return zoo.choose_in_mortal_space(
            lambda who, fresh, allowed, ahead=(): self._ask(views, who, fresh, allowed, ahead),
            views, rows, players, legal,
            skip_forced=self.skip_forced, preview_reach=self.preview_reach,
        )

    @torch.no_grad()
    def _ask(self, views, who: list[tuple[int, int]], fresh: bool, allowed: np.ndarray, ahead=()):
        device = str(next(self.parameters()).device)
        planes, masks = zoo.asked_planes(views, who, fresh, ahead, device)
        mask = torch.from_numpy(allowed).to(device)
        with policy_inference.autocast(device, self.actions):
            logits, _value, _phi = (self.deciding or self.decision)(planes, mask)
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
        preview_reach: bool = False,
        generator: torch.Generator | None = None,
    ):
        """One of our engine's actions per row, and what the policy decided
        to get there, with the value its head gave each decision: the
        fusion's judgement, the one the baseline pass reads; and, with
        `keep_phi`, Mortal's vector of each. Its moves are Mortal's, so a
        riichi is answered in two steps and both are recorded; with
        `preview_reach`, from one forward; drawn from `generator` when one
        is given (see `mortal_learner.decide_in_mortal_space`)."""

        def score(planes: torch.Tensor, mask: torch.Tensor):
            answer = (self.deciding or self.decision)(planes, mask)
            return answer if self.keep_phi else answer[:2]

        return mortal_learner.decide_in_mortal_space(
            score,
            views,
            rows,
            players,
            legal,
            greedy,
            str(next(self.parameters()).device),
            self.timing,
            explore_share=explore_share,
            wanderer=wanderer,
            preview_reach=preview_reach,
            generator=generator,
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


def build(ours_path: Path | str, mortal_path: Path | str, device: str) -> Combined:
    """A joined player over the two checkpoints named, born as Mortal (see
    the module's notes), with Mortal's config, which its checkpoints carry,
    as `mortal_config`."""
    payload = torch.load(ours_path, map_location="cpu", weights_only=True)
    ours = from_payload(payload, device)
    if not ours.speaks_mortal:
        raise SystemExit(
            f"{ours_path} answers over {ours.actions} moves and Mortal over "
            f"{ACTIONS}; re-head it first with neural.rehead"
        )
    state = torch.load(mortal_path, map_location="cpu", weights_only=False)
    net = Combined(ours, mortal_model.from_state(state).to(device)).to(device)
    net.mortal_config = state["config"]
    return net


def load(path: Path | str, device: str) -> tuple[Combined, dict]:
    """A joined player from a checkpoint this module wrote."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "mix" in state.get("combined", {}):
        raise SystemExit(
            f"{path} carries the old fusion, which mapped one action space onto "
            "the other; those lineages are not resumable here"
        )
    require_engine_observation(state, state["planes"])
    ours = PolicyValueNet(
        state["channels"],
        state["blocks"],
        state["planes"],
        state["attention"],
        state.get("actions", ACTIONS),
    ).to(device)
    load_weights(ours, state["model"])
    net = Combined(ours, mortal_model.from_state(state).to(device)).to(device)
    net.mortal_config = state["config"]
    # A head saved before it judged positions and read hands keeps
    # everything it had and starts those at nothing, which is where they
    # begin anyway: the fusion then judges and reads as our network does
    # and learns from there.
    net.fuse.load_state_dict(state["combined"], strict=False)
    return net, state


def migrate_belief_optimizer(saved: dict, optimiser, net: Combined) -> dict:
    """Expand legacy Adam moments alongside the equivalent model conversion.

    The number/order of parameters is unchanged (Linear replaces Conv1d).
    Only these two tensors changed shape; never reset unrelated optimizer state.
    """
    result = {**saved, "state": dict(saved["state"])}
    targets = {id(net.fuse.hands_out.weight): "weight", id(net.fuse.hands_out.bias): "bias"}
    for old_group, group in zip(saved["param_groups"], optimiser.param_groups):
        for key, parameter in zip(old_group["params"], group["params"]):
            kind = targets.get(id(parameter))
            if kind is None or key not in result["state"]:
                continue
            state = dict(result["state"][key])
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(name)
                if value is None or value.shape == parameter.shape:
                    continue
                old_shape = ((OPPONENTS, net.fuse.width, 1) if kind == "weight"
                             else (OPPONENTS,))
                if tuple(value.shape) != old_shape:
                    raise ValueError("Unexpected fusion belief optimizer tensor shape")
                state[name] = (value.squeeze(-1) if kind == "weight" else value).repeat_interleave(POSITIONS, dim=0)
            result["state"][key] = state
    return result
