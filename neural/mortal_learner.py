"""Mortal as a learner in our loop: fine-tuned on our rules by self-play.

The published Mortal was trained on Tenhou's rules, by offline learning
from human games. Ours differ, red fives most of all, and its Q values are
a policy only through an argmax. This wraps it as a player our PPO loop
can train: its Q values over its own forty-six actions, through a
temperature, are the policy's logits; a fresh linear head on its encoder's
features is the value baseline, trained on our returns. The encoder and
the Q head learn by the policy gradient, the batch-normalisation
statistics stay frozen, and everything is saved in Mortal's own checkpoint
layout, so a fine-tuned Mortal loads wherever the published one does.

It decides in its own action space and the table hears ours, so a
decision is translated as the zoo does it, with our engine's legal mask
as the authority: the policy is defined over the actions of Mortal's that
our engine allows here, and that mask is what is recorded. Riichi is two
decisions, the declaration and then the tile, each recorded with the
state it was made in; both belong to the same player and earn the same
return.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from libriichi.consts import ACTION_SPACE

from . import mortal_model, zoo
from .observe import Planes, Views


@dataclass
class Records:
    """What a step's decisions leave behind for training: one row per
    decision in the learner's own action space. `slots[r]` is the row of
    the step's batch the decision belongs to, so its game and player are
    the caller's `rows[slots[r]]` and `players[slots[r]]`."""

    planes: Planes
    masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    slots: np.ndarray


def decide_in_mortal_space(
    score,
    views: Views,
    rows: np.ndarray,
    players: np.ndarray,
    legal: np.ndarray,
    greedy: bool = False,
    device: str = "cuda",
    timing: dict | None = None,
) -> tuple[np.ndarray, Records]:
    """One of our engine's actions per row, and the records of everything
    the policy decided to get there, in Mortal's action space.

    `score(planes, mask)` gives the logits over those forty-six moves. A
    riichi names no tile there, so when one is chosen the reach is told to
    the follower and the same question asked again from the state in which
    it stands; both answers are recorded, because the policy made both.
    """
    clock = time.perf_counter
    timing = timing if timing is not None else {}
    follower = views.observer.follower
    who = list(zip(np.asarray(rows).tolist(), np.asarray(players).tolist()))
    legal = np.atleast_2d(legal)
    began = clock()
    planes, _own = views.sparse_and_masks(rows, players)
    timing["encode"] = timing.get("encode", 0.0) + clock() - began
    began = clock()
    # The policy is over what our engine allows, named in Mortal's moves.
    #
    # Not what both allow. Mortal will not declare a reach with fewer than
    # four tiles left in the wall; EMA 2025 section 3.3.10 allows it down to
    # one, and that is the game being played. Keeping only what the two
    # agree on drops a legal riichi at the end of every hand, quietly, in
    # self-play and in the browser alike. Mortal's own mask is read from the
    # encoder and left unused for that reason.
    allowed = zoo.translatable(legal)
    # A row where nothing agrees is decided by our engine's first legal
    # move and not recorded; it does not happen in practice.
    decidable = allowed.any(axis=1)
    allowed[~decidable, zoo.MORTAL_PASS] = True
    timing["translate"] = timing.get("translate", 0.0) + clock() - began
    began = clock()
    mask = torch.from_numpy(allowed).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        logits = score(planes.dense(device), mask)
    logits = logits.float()
    distribution = torch.distributions.Categorical(logits=logits)
    picked = logits.argmax(dim=1) if greedy else distribution.sample()
    log_prob = distribution.log_prob(picked).cpu().numpy()
    picked = picked.cpu().numpy()
    timing["network"] = timing.get("network", 0.0) + clock() - began

    choice = zoo.first_meaning(picked, legal)
    choice = np.where(decidable & (choice >= 0), choice, legal.argmax(axis=1)).astype(np.int64)
    record_planes = [planes]
    record_masks = [allowed]
    record_actions = [picked]
    record_log_probs = [log_prob]
    record_slots = [np.arange(len(who))]
    second = np.nonzero(decidable & (picked == zoo.MORTAL_RIICHI))[0].tolist()
    if not decidable.all():
        keep = decidable
        record_planes = [planes.rows(np.nonzero(keep)[0])]
        record_masks = [allowed[keep]]
        record_actions = [picked[keep]]
        record_log_probs = [log_prob[keep]]
        record_slots = [np.nonzero(keep)[0]]

    if second:
        # The reach declared ahead of the table; then the tile, from the
        # state in which it is declared.
        for i in second:
            game, player = who[i]
            follower.tell(game, player, json.dumps({"type": "reach", "actor": player}))
        indptr, indices, values, masks = follower.encode([who[i] for i in second])
        after = Planes.from_follower(indptr, indices, values)
        allowed_after = np.asarray(masks, dtype=bool)
        for slot, i in enumerate(second):
            tiles = legal[i][zoo.RIICHI_DISCARD : zoo.TSUMO]
            allowed_after[slot, :34] &= tiles
            allowed_after[slot, 34:] = False
            if not allowed_after[slot].any():
                allowed_after[slot, :34] = tiles
        mask_after = torch.from_numpy(allowed_after).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            logits_after = score(after.dense(device), mask_after)
        logits_after = logits_after.float()
        distribution_after = torch.distributions.Categorical(logits=logits_after)
        tile = logits_after.argmax(dim=1) if greedy else distribution_after.sample()
        log_prob_after = distribution_after.log_prob(tile).cpu().numpy()
        tile = tile.cpu().numpy()
        for slot, i in enumerate(second):
            choice[i] = zoo.RIICHI_DISCARD + int(tile[slot])
        record_planes.append(after)
        record_masks.append(allowed_after)
        record_actions.append(tile)
        record_log_probs.append(log_prob_after)
        record_slots.append(np.array(second, dtype=np.int64))

    if len(record_planes) == 1:
        records = Records(
            planes=record_planes[0],
            masks=record_masks[0],
            actions=record_actions[0].astype(np.int64),
            log_probs=record_log_probs[0].astype(np.float32),
            slots=record_slots[0].astype(np.int64),
        )
    else:
        records = Records(
            planes=Planes.cat(record_planes),
            masks=np.concatenate(record_masks),
            actions=np.concatenate(record_actions).astype(np.int64),
            log_probs=np.concatenate(record_log_probs).astype(np.float32),
            slots=np.concatenate(record_slots).astype(np.int64),
        )
    return choice, records


class MortalLearner(nn.Module):
    """A Mortal that plays in our loop and learns from it."""

    kind = "mortal"
    actions = ACTION_SPACE

    def __init__(self, mortal: mortal_model.Mortal, temperature: float = 1.0) -> None:
        super().__init__()
        self.mortal = mortal
        self.temperature = temperature
        self.value_head = nn.Linear(1024, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        for parameter in self.mortal.parameters():
            parameter.requires_grad_(True)
        self.device = "cpu"
        # Where a round's deciding went, in seconds, for the play record;
        # `selfplay.play` reads and clears it.
        self.timing = {"encode": 0.0, "translate": 0.0, "network": 0.0}
        # The forward used when deciding; a trainer may set a compiled one.
        self.inference = self.policy

    def to(self, device):  # type: ignore[override]
        self.device = str(device)
        return super().to(device)

    def train(self, mode: bool = True) -> MortalLearner:  # type: ignore[override]
        """Training mode for everything but the batch normalisation, whose
        statistics were learned from a great many human games and are not
        to be moved by a few thousand of ours at a time."""
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
        return self

    def policy(self, planes: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The policy's logits over Mortal's actions, masked, and the
        value, one per row."""
        phi = self.mortal.features(planes)
        q = self.mortal.dqn(phi, mask)
        logits = (q / self.temperature).masked_fill(~mask, float("-inf"))
        return logits, self.value_head(phi).squeeze(1)

    def forward(self, planes: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(planes, mask)

    def state(self) -> dict:
        return {**self.mortal.state(), "value_head": self.value_head.state_dict()}

    @torch.no_grad()
    def decide(
        self,
        views: Views,
        rows: np.ndarray,
        players: np.ndarray,
        legal: np.ndarray,
        greedy: bool = False,
    ) -> tuple[np.ndarray, Records]:
        """One of our actions per row, and the records of the decisions
        made, in Mortal's action space, that produced them."""
        return decide_in_mortal_space(
            lambda planes, mask: self.inference(planes, mask)[0],
            views,
            rows,
            players,
            legal,
            greedy,
            self.device,
            self.timing,
        )

    def choose(
        self, views: Views, rows: np.ndarray, players: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        """Its best move per row, in our action space, as a zoo player."""
        choice, _records = self.decide(views, rows, players, legal, greedy=True)
        return choice


def from_mortal(path: Path | str, device: str, temperature: float = 1.0) -> MortalLearner:
    """A learner starting from a published Mortal."""
    return MortalLearner(mortal_model.load(path, device), temperature).to(device)


def load(path: Path | str, device: str, temperature: float = 1.0) -> tuple[MortalLearner, dict]:
    """A learner from a checkpoint this trainer wrote, or from a published
    Mortal; and the checkpoint's payload."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    net = mortal_model.build(**mortal_model.shape_of(state))
    net.brain.load_state_dict(state["mortal"])
    net.dqn.load_state_dict(state["current_dqn"])
    learner = MortalLearner(net, temperature)
    if "value_head" in state:
        learner.value_head.load_state_dict(state["value_head"])
    return learner.to(device), state
