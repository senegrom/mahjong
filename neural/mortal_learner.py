"""Mortal as a learner in our loop: fine-tuned on our rules by self-play.

The published Mortal was trained on Tenhou's rules, by offline learning
from human games. Ours differ, red fives most of all, and its Q values are
a policy only through an argmax. This wraps it as a player our PPO loop
can train: its Q values over its own forty-six actions, through a
temperature, are the policy's logits; a fresh linear head on its encoder's
features is the value baseline, trained on our returns. The encoder and
the Q head learn by the policy gradient alone: the value head reads the
encoder's features without training them, as the fusion's judge and our
own network's value heads do. Its loss reached the encoder with two and a
half to four times the policy term's gradient there, in a nearly
orthogonal direction, under the one clip on the gradient's norm, so the
encoder the policy reads from was being reshaped to predict returns. The
batch-normalisation statistics stay frozen, and everything is saved in
Mortal's own checkpoint layout, so a fine-tuned Mortal loads wherever the
published one does.

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

from . import mortal_model, policy_inference, zoo
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
    #: Which of these decisions were a legal move taken at random rather
    #: than the policy's choice. The tile a declaration names is never
    #: forced: exploring the reach and then the tile it throws would be two
    #: wanderings compounded, and the first already puts the position
    #: somewhere the policy would not have gone.
    forced: np.ndarray | None = None
    #: What the learner's value head made of each decision's position as
    #: it decided, by the forward that chose the move: the baseline a
    #: trainer can measure the returns against without a pass of its own
    #: over the round (`--baseline-from-play`). None when the score gave
    #: the logits alone.
    values: np.ndarray | None = None
    #: Which of these decisions a rule of the last hand made rather than the
    #: policy (see `selfplay_rules`), each recorded alone: its mask holds
    #: only the move played and its log-probability is nought, so the
    #: learning forward gives it probability one and the policy learns
    #: nothing from it, while the critic and the reading of the hands learn
    #: its position. None when no rules were played; a reach's tile record
    #: is never flagged.
    ruled: np.ndarray | None = None


def scored(answer) -> tuple[torch.Tensor, torch.Tensor | None]:
    """A score's logits, and the value of each row when it gave one: a
    score answers the logits alone, or the logits and the values."""
    if isinstance(answer, torch.Tensor):
        return answer, None
    logits, value = answer
    return logits, value


def joined(parts: list, cat):
    """A step's parts of one field stacked in record order by `cat`, or
    None when any part is missing."""
    if any(part is None for part in parts):
        return None
    return parts[0] if len(parts) == 1 else cat(parts)


def decide_in_mortal_space(
    score,
    views: Views,
    rows: np.ndarray,
    players: np.ndarray,
    legal: np.ndarray,
    greedy: bool = False,
    device: str = "cuda",
    timing: dict | None = None,
    explore_share: float = 0.0,
    wanderer=None,
    preview_reach: bool = False,
    generator: torch.Generator | None = None,
    rules=None,
) -> tuple[np.ndarray, Records]:
    """One of our engine's actions per row, and the records of everything
    the policy decided to get there, in Mortal's action space.

    `score(planes, mask)` gives the logits over those forty-six moves, or
    the logits and the value of each row, which are then recorded too (see
    `Records.values`). A riichi names no tile there, so when one is chosen
    the reach is told to the follower and the same question asked again
    from the state in which it stands; both answers are recorded, because
    the policy made both.

    With `preview_reach` nobody is told and nothing is asked again: every
    row that may declare riichi is asked as well, in the first forward,
    which tile it would throw as it would stand having declared, from the
    follower's preview of the reach (see `Views.sparse_and_masks`), and a
    row that chooses the reach draws its tile from that answer. The states
    and the questions are the ones telling would have made, and so are the
    records; the draws are made in the same order from batches of the same
    shapes, so the random numbers are the ones they always were. Only the
    batch the answers come from is larger, which in bfloat16 can move them
    in their last bits, so it is a switch, off unless the trainer is told.
    It saves a second forward, a few hundred kernels, wherever a reach is
    chosen, about eleven hundred times in a round of four thousand tables,
    for a few more rows in the first.

    `timing`, when given, is added to in seconds: `encode` for gathering
    the rows' planes, those previewed included, and for telling a reach,
    which first waits for any encoding a worker is still making from the
    follower (see `Views.prepare`); `translate`; `network` for the first
    answer; and `riichi` for the rest of the second.

    The moves and the reach's tiles are drawn from `generator` when one is
    given, and from torch's own generator otherwise (see `selfplay.sampled`).

    `rules`, a round's `selfplay_rules.Played`, has the moves drawn played
    by the last hand's rules it names, and keeps its account of them, the
    seconds their work takes included (`timing`'s `rules` as well): a win
    drawn that decline-last declines gives way to the pass or the best other
    move of the same answer, and then a riichi whose drawn tile the all-last
    dama rule keeps first is thrown undeclared (see `selfplay_rules`). Each
    move a rule played is recorded alone, flagged (`Records.ruled`), and a
    reach left undeclared loses its tile record. It needs `preview_reach`,
    so that nothing is told of a reach before the rules decide; checks of
    every step's records against the moves sent raise
    `selfplay_rules.RuleCheckFailed`. Without it every move and record is
    what it always was, bit for bit.
    """
    clock = time.perf_counter
    timing = timing if timing is not None else {}
    if rules is not None and not preview_reach:
        raise ValueError("the last hand's rules need the reach's tile asked with the reach (preview_reach), so "
                         "that nothing is told of a reach before they decide")
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
    # With `preview_reach`, the rows that may declare riichi, each asked
    # ahead which tile it would throw, with only those tiles allowed.
    ahead = (np.flatnonzero(allowed[:, zoo.MORTAL_RIICHI]) if preview_reach
             else np.zeros(0, dtype=np.int64))
    allowed_ahead = np.zeros((len(ahead), zoo.MORTAL_ACTIONS), dtype=bool)
    allowed_ahead[:, :34] = legal[ahead, zoo.RIICHI_DISCARD:zoo.TSUMO]
    timing["translate"] = timing.get("translate", 0.0) + clock() - began
    if len(ahead):
        began = clock()
        previews, _own_ahead = views.sparse_and_masks(
            np.asarray(rows)[ahead], np.asarray(players)[ahead], after_reach=True
        )
        timing["encode"] = timing.get("encode", 0.0) + clock() - began
    began = clock()
    if len(ahead):
        # One forward for both questions: the rows as they stand, then the
        # same players' reaches as they would stand, split apart again.
        asked = torch.from_numpy(np.concatenate([allowed, allowed_ahead])).to(device)
        logits, value = scored(score(Planes.dense_of([planes, previews], device), asked))
        logits = logits.float()
        mask = asked[: len(who)]
        logits, logits_ahead = logits[: len(who)], logits[len(who):]
        value, value_ahead = (None, None) if value is None else (value[: len(who)], value[len(who):])
    else:
        mask = torch.from_numpy(allowed).to(device)
        # Precision is owned by the caller, matching the later PPO forward.
        logits, value = scored(score(planes.dense(device), mask))
        logits = logits.float()
    if greedy:
        picked = logits.argmax(dim=1)
        # Built here alone: validating a distribution reads the card back,
        # and drawing a move builds one of its own (`selfplay.explore`).
        log_prob = torch.distributions.Categorical(logits=logits).log_prob(picked)
    else:
        # Now and then a legal move at random instead of the policy's, and
        # record the policy's likelihood and a forced-row flag. Forced
        # rows are auxiliary-only in PPO. See `selfplay.explore`: the
        # point is to show the heads that are not the policy, the value and
        # the reading of the hands, positions the policy alone would not
        # reach.
        from .selfplay import explore

        picked, log_prob, was_forced = explore(
            logits,
            mask,
            explore_share,
            wanderer if wanderer is not None else np.random.default_rng(),
            generator,
        )
    log_prob = log_prob.cpu().numpy()
    picked = picked.cpu().numpy()
    if greedy:
        was_forced = np.zeros(len(picked), dtype=bool)
    else:
        was_forced = was_forced.cpu().numpy()
    if value is not None:
        value = value.float().cpu().numpy()
    timing["network"] = timing.get("network", 0.0) + clock() - began

    # The last hand's rules, when the round is played by them: which rows a
    # rule decided (`Records.ruled`), the seconds their work takes, and
    # whether the step's moves, probabilities and forced flags are still
    # the arrays the draws made, which a rule's move must not overwrite.
    # Each rule asks the engine only about the rows it can answer, and
    # nothing at all while no game can be in its last hand (`asking`, see
    # `selfplay_rules.may_ask` and `last_hand`).
    ruled = None
    spent = 0.0
    copied = False
    asking = False
    if rules is not None:
        from . import selfplay_rules

        began = clock()
        ruled = np.zeros(len(who), dtype=bool)
        asking = selfplay_rules.may_ask(views)
        if asking and selfplay_rules.DECLINE in rules.rules:
            # Decline-last first, on every win drawn: a declined self-draw
            # can leave a riichi, which the all-last dama rule then judges.
            # The logits come back from the card only for a win declined
            # at the player's own turn, a handful a round.
            asked = np.flatnonzero(decidable & (picked == zoo.MORTAL_AGARI))
            if len(asked):
                asked = asked[selfplay_rules.last_hand(views, np.asarray(rows)[asked], np.asarray(players)[asked])]
            if len(asked):
                moves = selfplay_rules.decline(
                    views.arena, who, asked, allowed,
                    lambda at: logits[torch.from_numpy(at).to(logits.device)].cpu().numpy(), rules)
                if moves:
                    picked, log_prob, was_forced = picked.copy(), log_prob.copy(), was_forced.copy()
                    copied = True
                    at = np.fromiter(moves.keys(), dtype=np.int64, count=len(moves))
                    instead = np.fromiter(moves.values(), dtype=np.int64, count=len(moves))
                    picked[at] = instead
                    log_prob[at] = 0.0
                    was_forced[at] = False
                    ruled[at] = True
                    rules.to_riichi += int((instead == zoo.MORTAL_RIICHI).sum())
        spent += clock() - began

    choice = zoo.first_meaning(picked, legal)
    choice = np.where(decidable & (choice >= 0), choice, legal.argmax(axis=1)).astype(np.int64)
    second = np.nonzero(decidable & (picked == zoo.MORTAL_RIICHI))[0].tolist()
    # The records of the reaches' tiles, by field, or None where no reach
    # was chosen: assembled into the step's records below, once the rules
    # have said which reaches are played.
    tiles = None

    if second:
        if not preview_reach:
            # The reach declared ahead of the table, told through the views,
            # which first wait for any encoding a worker is still making
            # from the follower (see `Views.prepare`): a wait for the seated
            # others' views, counted with the encoding and not as this
            # question's time, which it is not.
            began = clock()
            for i in second:
                game, player = who[i]
                views.tell(game, player, json.dumps({"type": "reach", "actor": player}))
            timing["encode"] = timing.get("encode", 0.0) + clock() - began
        began = clock()
        if preview_reach:
            # Asked already, each from the state telling the reach would
            # have left it in: the answers of those who chose it.
            place = np.full(len(who), -1, dtype=np.int64)
            place[ahead] = np.arange(len(ahead))
            at = place[second]
            if (at < 0).any():
                raise RuntimeError("the policy declared riichi where it was not allowed to")
            after = previews.rows(at)
            allowed_after = allowed_ahead[at]
            chosen = torch.from_numpy(at).to(device)
            logits_after = logits_ahead[chosen]
            value_after = None if value_ahead is None else value_ahead[chosen]
        else:
            # Then the tile, from the state in which the reach is declared.
            indptr, indices, values, masks = follower.encode([who[i] for i in second])
            after = Planes.from_follower(indptr, indices, values)
            allowed_after = np.zeros((len(second), zoo.MORTAL_ACTIONS), dtype=bool)
            allowed_after[:, :34] = legal[second, zoo.RIICHI_DISCARD:zoo.TSUMO]
            mask_after = torch.from_numpy(allowed_after).to(device)
            logits_after, value_after = scored(score(after.dense(device), mask_after))
            logits_after = logits_after.float()
        distribution_after = torch.distributions.Categorical(logits=logits_after)
        if greedy:
            tile = logits_after.argmax(dim=1)
        else:
            from .selfplay import sampled

            tile = sampled(distribution_after, generator)
        log_prob_after = distribution_after.log_prob(tile).cpu().numpy()
        tile = tile.cpu().numpy()
        for slot, i in enumerate(second):
            choice[i] = zoo.RIICHI_DISCARD + int(tile[slot])
        tiles = {
            "planes": after,
            "masks": allowed_after,
            "actions": tile,
            "log_probs": log_prob_after,
            "slots": np.array(second, dtype=np.int64),
            # The tile a declaration throws is never a forced move; see
            # `Records.forced`.
            "forced": np.zeros(len(second), dtype=bool),
            # The value of the state the tile is chosen in, the reach
            # declared: the position the second record is.
            "values": None if value_after is None else value_after.float().cpu().numpy(),
        }
        timing["riichi"] = timing.get("riichi", 0.0) + clock() - began

    if asking and second and selfplay_rules.DAMA in rules.rules:
        # The all-last dama rule on every reach, drawn or left by a declined
        # self-draw, that the engine can answer, read at the tile it drew:
        # where that tile keeps the player first, its discard is played
        # instead and recorded alone, and the reach's tile record, of a
        # position the table never sees, is dropped. `fired` are places in
        # `second`.
        began = clock()
        slots = tiles["slots"]
        considered = np.flatnonzero(selfplay_rules.last_hand(views, np.asarray(rows)[slots],
                                                             np.asarray(players)[slots]))
        fired = []
        if len(considered):
            fired = considered[selfplay_rules.dama(views.arena, who, tiles["slots"][considered], tile[considered],
                                                   legal, rules)].tolist()
        if fired:
            if not copied:
                picked, log_prob, was_forced = picked.copy(), log_prob.copy(), was_forced.copy()
                copied = True
            at = tiles["slots"][fired]
            thrown = tile[fired].astype(np.int64)
            choice[at] = zoo.DISCARD + thrown
            picked[at] = thrown
            log_prob[at] = 0.0
            was_forced[at] = False
            rules.both += int(ruled[at].sum())
            ruled[at] = True
            kept = np.ones(len(second), dtype=bool)
            kept[fired] = False
            if not kept.any():
                tiles = None
            else:
                tiles = {
                    "planes": tiles["planes"].rows(np.flatnonzero(kept)),
                    **{name: tiles[name][kept] for name in ("masks", "actions", "log_probs", "slots", "forced")},
                    "values": None if tiles["values"] is None else tiles["values"][kept],
                }
        spent += clock() - began

    masks = allowed
    if ruled is not None and ruled.any():
        # Each move a rule played is recorded alone: the only move in its
        # mask, at the log-probability of nought set above.
        began = clock()
        flagged = np.flatnonzero(ruled)
        masks = allowed.copy()
        masks[flagged] = False
        masks[flagged, picked[flagged]] = True
        spent += clock() - began
    record_planes = [planes]
    record_masks = [masks]
    record_actions = [picked]
    record_log_probs = [log_prob]
    record_slots = [np.arange(len(who))]
    record_forced = [was_forced]
    record_values = [value]
    if not decidable.all():
        keep = decidable
        record_planes = [planes.rows(np.nonzero(keep)[0])]
        record_masks = [masks[keep]]
        record_actions = [picked[keep]]
        record_log_probs = [log_prob[keep]]
        record_slots = [np.nonzero(keep)[0]]
        record_forced = [was_forced[keep]]
        record_values = [None if value is None else value[keep]]
    if tiles is not None:
        record_planes.append(tiles["planes"])
        record_masks.append(tiles["masks"])
        record_actions.append(tiles["actions"])
        record_log_probs.append(tiles["log_probs"])
        record_slots.append(tiles["slots"])
        record_forced.append(tiles["forced"])
        record_values.append(tiles["values"])

    # Which of the records a rule made, one flag a record in the order
    # above, or None when no rules were played.
    record_ruled = None
    if rules is not None:
        # The records' flags; what is sent to the table against what is
        # recorded, each reach's tile record its row and its tile (see
        # `selfplay_rules.check_step`); and the step's account.
        began = clock()
        record_ruled = ruled if decidable.all() else ruled[decidable]
        if tiles is None:
            tile_slots = tile_actions = np.zeros(0, dtype=np.int64)
        else:
            tile_slots, tile_actions = tiles["slots"], tiles["actions"]
            # A reach's tile record is the policy's own decision, never a
            # rule's.
            record_ruled = np.concatenate([record_ruled, np.zeros(len(tile_slots), dtype=bool)])
        rules.reaches_sent += selfplay_rules.check_step(choice, picked, legal, decidable, ruled, tile_slots,
                                                        tile_actions)
        rules.records += int(np.count_nonzero(ruled))
        rules.tile_records += len(tile_slots)
        spent += clock() - began
        rules.seconds += spent
        timing["rules"] = timing.get("rules", 0.0) + spent

    # Values only when every part of the step has them.
    values = joined(record_values, np.concatenate)
    if values is not None:
        values = values.astype(np.float32)
    if len(record_planes) == 1:
        records = Records(
            planes=record_planes[0],
            masks=record_masks[0],
            actions=record_actions[0].astype(np.int64),
            log_probs=record_log_probs[0].astype(np.float32),
            slots=record_slots[0].astype(np.int64),
            forced=record_forced[0].astype(bool),
            values=values,
            ruled=record_ruled,
        )
    else:
        records = Records(
            planes=Planes.cat(record_planes),
            masks=np.concatenate(record_masks),
            actions=np.concatenate(record_actions).astype(np.int64),
            log_probs=np.concatenate(record_log_probs).astype(np.float32),
            slots=np.concatenate(record_slots).astype(np.int64),
            forced=np.concatenate(record_forced).astype(bool),
            values=values,
            ruled=record_ruled,
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
        self.timing = {"encode": 0.0, "translate": 0.0, "network": 0.0, "riichi": 0.0}
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
        value, one per row. The value reads the encoder's features without
        gradient, so only the policy's loss trains the encoder."""
        phi = self.mortal.features(planes)
        q = self.mortal.dqn(phi, mask)
        logits = (q / self.temperature).masked_fill(~mask, float("-inf"))
        return logits, self.value_head(phi.detach()).squeeze(1)

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
        explore_share: float = 0.0,
        wanderer=None,
        preview_reach: bool = False,
        generator: torch.Generator | None = None,
        rules=None,
    ) -> tuple[np.ndarray, Records]:
        """One of our actions per row, and the records of the decisions
        made, in Mortal's action space, that produced them, with the value
        head's value of each, which the same forward has worked out; a
        reach's tile from that forward too with `preview_reach`; the moves
        drawn from `generator` when one is given, and played by the last
        hand's `rules` when given (see `decide_in_mortal_space`)."""
        return decide_in_mortal_space(
            self.inference,
            views,
            rows,
            players,
            legal,
            greedy,
            self.device,
            self.timing,
            explore_share=explore_share,
            wanderer=wanderer,
            preview_reach=preview_reach,
            generator=generator,
            rules=rules,
        )

    def choose(
        self, views: Views, rows: np.ndarray, players: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        """Its best move per row, in our action space, as a zoo player: at
        the precision every player at a table chooses at (see
        `policy_inference`), as the joined player and the zoo's Mortal do.
        `decide` leaves precision to its caller, and measuring the learner
        against the heuristic players called it bare, so train_mortal chose
        its best checkpoint by a float32 policy while its self-play, and
        every table the checkpoint is later seated at, play in bfloat16 on
        the card, and a near tie could turn between the two."""
        with policy_inference.autocast(self.device, self.actions):
            choice, _records = self.decide(views, rows, players, legal, greedy=True)
        return choice


def from_mortal(path: Path | str, device: str, temperature: float = 1.0) -> MortalLearner:
    """A learner starting from a published Mortal."""
    return MortalLearner(mortal_model.load(path, device), temperature).to(device)


def load(path: Path | str, device: str, temperature: float | None = None) -> tuple[MortalLearner, dict]:
    """A learner from a checkpoint this trainer wrote, or from a published
    Mortal; and the checkpoint's payload. It plays at the temperature the
    checkpoint was trained at, 1.0 where it names none, unless given
    another: every logit is a Q value over it, so a different one is a
    different policy."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    net = mortal_model.from_state(state)
    if temperature is None:
        saved = state.get("temperature")
        temperature = 1.0 if saved is None else float(saved)
    learner = MortalLearner(net, temperature)
    if "value_head" in state:
        learner.value_head.load_state_dict(state["value_head"])
    return learner.to(device), state
