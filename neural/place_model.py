"""U: the final placement bonus a player can expect, read from the scoreboard.

The default reward credits every decision of a game with the game's
placement bonus, so an East 1 discard carries the luck of every hand dealt
after it: at East 1 the scoreboard explains 0.04 of the final placement, at
South 4 0.77 (the boundary judge, 17 September 2026), and the rest is hands
not yet played. The placement reward cuts that luck off. A decision is
credited with U at the start of the next hand, the bonus the scoreboard
then promises, and after a game's last hand with the bonus itself; the
hand's points can be added at a small weight, alpha (see
`PlacementReward`). Under EMA's rules the scoreboard is everything that
carries from one hand to the next, so nothing about the future is lost by
cutting there, and summed over a game the hand-by-hand changes in U add up
to the bonus (`hand_credits`).

U is a small network over about twenty numbers and who sits where: the
hand of the game (East 1 to South 4), the counters, the riichi sticks on
the table, where the dealer sits, the four raw scores and the gaps to the
other three, in seat order from the player's own, and which kind of player
holds each of the four seats, the learner or one of the others a run seats
(`members`). It answers the four places' probabilities, fitted to where the
players really finished, a tie shared between the places it covers; the
expected bonus follows from the engine's placement values. It is fitted on
a run's own games, held out by game (`fit`, `calibration`), and refitted as
the run's players change (`train_combined --place-refit-every`).

    python -m neural.place_model fit round-ledger.npz [...] --out place-model.pt
    python -m neural.place_model check place-model.pt round-ledger.npz [...]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

import riichi_py

from .scoreboard import DEALER, GAME, HONBA, ROUND, SCORES, STAGE_NAMES, STAGES, STICKS, KYOKU, Ledger

#: The layout of a saved model; one of another version is refused.
VERSION = 1

#: The placement bonus by place, best to worst: the engine's, which the
#: default reward and every measurement use too (`selfplay.PLACEMENT_VALUE`).
VALUES = tuple(float(value) for value in riichi_py.PLACEMENT_VALUE)

#: The kind of player who holds a seat when it is the learner, whatever
#: checkpoint the learner is that round.
LEARNER = "learner"

#: The numbers a scoreboard is read as, before who sits where: the stage,
#: one of eight; the counters and the sticks, a quarter each; where the
#: dealer sits, one of four; the four scores and the gaps to the other
#: three, in units of ten thousand points.
BASE = STAGES + 1 + 1 + 4 + 4 + 3
SCORE_UNIT = 10_000.0

#: The boundary judge's reading of the final placement from the scoreboard
#: alone, by stage, East 1 to South 4 (`the-boundary-judge-is-a-scoreboard`),
#: which U's own reading is compared with.
JUDGE = (0.04, 0.16, 0.24, 0.35, 0.44, 0.53, 0.66, 0.77)

#: The kill rule of the pre-test: U is refused when its mean answer misses
#: the mean bonus by more than this, in some tenth of its answers at some
#: stage, by more than its noise (see `calibration`).
MISS = 0.03


def width(members: Sequence[str]) -> int:
    """How many numbers U reads, given the others a run may seat."""
    return BASE + 4 * (1 + len(members))


def view(starts: np.ndarray, players: np.ndarray) -> np.ndarray:
    """The scoreboard of each hand row of `starts` (see `scoreboard.COLUMNS`)
    as `players` sees it, one player a row: the first `BASE` numbers U reads.
    Seats are counted from the player's own, in the order of play, so the
    dealer's seat and the scores mean the same from every chair."""
    starts = np.asarray(starts, dtype=np.int64).reshape(-1, starts.shape[-1])
    players = np.asarray(players, dtype=np.int64)
    rows = len(starts)
    out = np.zeros((rows, BASE), dtype=np.float32)
    stage = starts[:, ROUND] * 4 + starts[:, KYOKU] - 1
    if np.any((stage < 0) | (stage >= STAGES)):
        raise ValueError("a scoreboard outside East 1 to South 4")
    out[np.arange(rows), stage] = 1.0
    at = STAGES
    out[:, at] = starts[:, HONBA] / 4.0
    out[:, at + 1] = starts[:, STICKS] / 4.0
    at += 2
    out[np.arange(rows), at + (starts[:, DEALER] - players) % 4] = 1.0
    at += 4
    order = (players[:, None] + np.arange(4)[None, :]) % 4
    scores = np.take_along_axis(starts[:, SCORES], order, axis=1) / SCORE_UNIT
    out[:, at:at + 4] = scores
    out[:, at + 4:at + 7] = scores[:, :1] - scores[:, 1:]
    return out


def kinds_of(ledger: Ledger, members: Sequence[str]) -> np.ndarray:
    """Which kind of player holds each player's place in each game, as an
    index into (`LEARNER`, *members): (games, 4). A member of the round that
    U was not fitted with is refused, since U would be reading a seat it
    has never seen."""
    members = list(members)
    lookup = []
    for name in ledger.members:
        if name not in members:
            raise ValueError(f"the round seated {name!r}, which this place model was not fitted "
                             f"with ({members}); refit it on the run's own games")
        lookup.append(1 + members.index(name))
    lookup = np.asarray([0, *lookup], dtype=np.int64)
    # -1, the learner, lands on the first entry.
    return lookup[ledger.owners + 1]


def inputs(ledger: Ledger, members: Sequence[str], rows: np.ndarray, players: np.ndarray) -> np.ndarray:
    """What U reads for hand rows `rows` of `ledger.starts`, each from the
    view of the matching entry of `players`."""
    rows = np.asarray(rows, dtype=np.int64)
    players = np.asarray(players, dtype=np.int64)
    kinds = kinds_of(ledger, members)
    games = ledger.starts[rows, GAME]
    order = (players[:, None] + np.arange(4)[None, :]) % 4
    seated = kinds[games[:, None], order]
    count = 1 + len(members)
    onehot = np.zeros((len(rows), 4 * count), dtype=np.float32)
    onehot[np.arange(len(rows))[:, None], np.arange(4)[None, :] * count + seated] = 1.0
    return np.concatenate([view(ledger.starts[rows], players), onehot], axis=1)


def soft_places(final_scores: np.ndarray) -> np.ndarray:
    """Where each player finished, (games, 4 players, 4 places), a tie
    shared evenly between the places it covers: two level first share first
    and second, a half each. The expected bonus over such a label is the
    one `outcomes.placement_rewards` pays."""
    scores = np.asarray(final_scores)
    better = (scores[:, None, :] > scores[:, :, None]).sum(axis=2)
    tied = (scores[:, None, :] == scores[:, :, None]).sum(axis=2)
    places = np.arange(4)[None, None, :]
    inside = (places >= better[:, :, None]) & (places < (better + tied)[:, :, None])
    return (inside / tied[:, :, None]).astype(np.float32)


class PlaceModel(nn.Module):
    """U: the four places' chances for the player whose view it reads."""

    def __init__(self, members: Sequence[str], hidden: int = 64) -> None:
        super().__init__()
        if type(hidden) is not int or hidden <= 0:
            raise ValueError("hidden must be a positive integer")
        self.members = tuple(str(name) for name in members)
        if len(set(self.members)) != len(self.members) or LEARNER in self.members:
            raise ValueError("members must be distinct, and none of them the learner")
        self.hidden = hidden
        self.body = nn.Sequential(
            nn.Linear(width(self.members), hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.register_buffer("values", torch.tensor(VALUES, dtype=torch.float32), persistent=False)
        #: What the fit that made it said: the games and rows it saw and how
        #: it read on the games it did not (see `fit`).
        self.fitted: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The four places' logits."""
        return self.body(x)

    @torch.no_grad()
    def expected(self, x: np.ndarray | torch.Tensor, chunk: int = 65536) -> np.ndarray:
        """The expected bonus of each row of inputs, in float64."""
        self.eval()
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        device = next(self.parameters()).device
        out = np.empty(len(x), dtype=np.float64)
        for start in range(0, len(x), chunk):
            logits = self(x[start:start + chunk].to(device))
            out[start:start + chunk] = (torch.softmax(logits.float(), dim=1) @ self.values).double().cpu().numpy()
        return out

    def state(self) -> dict:
        """Everything that rebuilds it, readable with `weights_only=True`, so
        a trainer's checkpoint can carry it."""
        return {
            "version": VERSION,
            "members": list(self.members),
            "hidden": self.hidden,
            "weights": {name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()},
            "fitted": json.loads(json.dumps(self.fitted)),
        }

    @classmethod
    def from_state(cls, state: dict) -> PlaceModel:
        if not isinstance(state, dict) or state.get("version") != VERSION:
            raise ValueError(f"not a place model of version {VERSION}")
        model = cls(state["members"], int(state["hidden"]))
        model.load_state_dict(state["weights"])
        model.fitted = dict(state.get("fitted") or {})
        return model.eval()

    def save(self, path: Path | str) -> None:
        from .checkpoints import atomic_save

        atomic_save(self.state(), Path(path), keep_previous=False)

    @classmethod
    def load(cls, path: Path | str) -> PlaceModel:
        return cls.from_state(torch.load(Path(path), map_location="cpu", weights_only=True))


def centred(answers: np.ndarray) -> np.ndarray:
    """U's answers for the four views of each scoreboard, four rows a hand
    in player order as `Boundaries` and `expected_at_starts` lay them out,
    less their mean. The four bonuses of a game always add up to nought,
    so their expectations do too; U reads each view apart and need not
    know it, and taking out the mean only brings its answers closer."""
    answers = np.asarray(answers, dtype=np.float64)
    if answers.size % 4:
        raise ValueError("U's answers come four to a scoreboard")
    grouped = answers.reshape(-1, 4)
    return (grouped - grouped.mean(axis=1, keepdims=True)).reshape(answers.shape)


def expected_at_starts(ledger: Ledger, model: PlaceModel) -> np.ndarray:
    """U at the start of every hand of the round from each player's view,
    the four views centred (see `centred`): (hands, 4), a row of
    `ledger.starts` and a column a player."""
    hands = len(ledger.starts)
    rows = np.repeat(np.arange(hands), 4)
    players = np.tile(np.arange(4), hands)
    return centred(model.expected(inputs(ledger, model.members, rows, players)).reshape(hands, 4))


def bonuses(ledger: Ledger) -> np.ndarray:
    """The placement bonus each player of each game ended with, (games, 4):
    what the default reward pays (see `outcomes.placement_rewards`)."""
    from .outcomes import placement_rewards

    return placement_rewards(ledger.final_scores, VALUES)


def hand_returns(ledger: Ledger, at_starts: np.ndarray) -> np.ndarray:
    """What each hand was worth to each player under the placement reward
    before its points: U at the start of the next hand, and the bonus
    itself after a game's last. (hands, 4), as `expected_at_starts`."""
    at_starts = np.asarray(at_starts, dtype=np.float64)
    if at_starts.shape != (len(ledger.starts), 4):
        raise ValueError("U must be given for every hand of the round, from every player's view")
    games = ledger.starts[:, GAME]
    last = np.arange(len(ledger.starts)) == ledger.first[games] + ledger.count[games] - 1
    out = np.empty_like(at_starts)
    following = np.flatnonzero(~last)
    out[following] = at_starts[following + 1]
    out[last] = bonuses(ledger)[games[last]]
    return out


def hand_credits(ledger: Ledger, at_starts: np.ndarray) -> np.ndarray:
    """What each hand did to each player's expected bonus: U at the next
    boundary less U at its own start. A game's credits add up to its bonus
    less U at its first hand, the part no decision of the game could have
    changed; (hands, 4)."""
    return hand_returns(ledger, at_starts) - np.asarray(at_starts, dtype=np.float64)


def placement_returns(ledger: Ledger, at_starts: np.ndarray, alpha: float) -> np.ndarray:
    """Every decision's return under the placement reward: its hand's worth
    from `hand_returns`, plus alpha times the hand's points over 4,000.
    float32, one a decision in the round's order."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    worth = hand_returns(ledger, at_starts)
    rows = ledger.rows_of(ledger.games, ledger.hands)
    returns = worth[rows, ledger.players.astype(np.int64)] + float(alpha) * ledger.points
    return returns.astype(np.float32)


@dataclass(frozen=True)
class PlacementReward:
    """The placement reward with the scoreboard model `model` and the
    points' weight `alpha` (see the module's description). At alpha 0 the
    return is placement alone, what the ruler counts. At alpha 0.05 the
    points only break ties: sixty thousand points are then worth less than
    one place."""

    model: PlaceModel
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")

    def returns(self, ledger: Ledger) -> np.ndarray:
        return placement_returns(ledger, expected_at_starts(ledger, self.model), self.alpha)


@dataclass
class Boundaries:
    """Every hand start of some rounds from every player's view, with where
    that player finished: what U is fitted on. `games` names each row's
    game uniquely across the rounds, for holding games out."""

    x: np.ndarray
    labels: np.ndarray
    games: np.ndarray
    stages: np.ndarray
    bonus: np.ndarray

    @classmethod
    def of(cls, ledger: Ledger, members: Sequence[str], tag: int = 0) -> Boundaries:
        """`ledger`'s hand starts, every one four times, a player's view
        each; `tag` keeps one round's games apart from another's."""
        hands = len(ledger.starts)
        rows = np.repeat(np.arange(hands), 4)
        players = np.tile(np.arange(4), hands)
        games = ledger.starts[rows, GAME]
        labels = soft_places(ledger.final_scores)[games, players]
        return cls(
            x=inputs(ledger, members, rows, players),
            labels=labels,
            games=games.astype(np.int64) + (int(tag) << 32),
            stages=ledger.stages[rows].astype(np.int8),
            bonus=bonuses(ledger)[games, players].astype(np.float32),
        )

    @classmethod
    def cat(cls, parts: Sequence[Boundaries]) -> Boundaries:
        return cls(*(np.concatenate([getattr(part, name) for part in parts])
                     for name in ("x", "labels", "games", "stages", "bonus")))

    def __len__(self) -> int:
        return len(self.x)

    def take(self, mask: np.ndarray) -> Boundaries:
        return Boundaries(self.x[mask], self.labels[mask], self.games[mask], self.stages[mask], self.bonus[mask])


def game_folds(games: np.ndarray, folds: int, seed: int = 0) -> np.ndarray:
    """A fold, 0 to `folds` - 1, for every row, every row of a game in the
    same fold: drawn per game from `seed`, so a held-out game is held out
    whole."""
    unique, inverse = np.unique(games, return_inverse=True)
    draw = np.random.default_rng(seed).permutation(len(unique)) % folds
    return draw[inverse]


def cross_entropy(model: PlaceModel, table: Boundaries, device: str = "cpu") -> float:
    """The mean cross-entropy of `model`'s places against `table`'s."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(table), 65536):
            x = torch.from_numpy(table.x[start:start + 65536]).to(device)
            y = torch.from_numpy(table.labels[start:start + 65536]).to(device)
            total += float(-(y * torch.log_softmax(model(x), dim=1)).sum())
    return total / max(len(table), 1)


def fit(
    table: Boundaries,
    members: Sequence[str],
    *,
    hidden: int = 64,
    epochs: int = 40,
    batch: int = 4096,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    validation: float = 0.1,
    patience: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> PlaceModel:
    """U fitted to `table`'s places, tied places as shared labels.

    A tenth of the games is kept aside to stop on: the epoch whose model
    reads them best is kept, and the fit stops once `patience` epochs in a
    row have not beaten it. Games, not rows, are kept aside, since every
    row of a game shares its outcome."""
    if len(table) == 0:
        raise ValueError("nothing to fit U on")
    generator = torch.Generator().manual_seed(seed)
    # Made from a stream of its own, on the processor, which is then put
    # back: a trainer that refits U mid-run must draw its moves and its
    # minibatches as it would have without the refit. Seeding through
    # `torch.manual_seed` would reseed the card's streams too, which
    # nothing here would put back.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = PlaceModel(members, hidden)
    model = model.to(device)
    aside = game_folds(table.games, max(2, round(1 / validation)), seed) == 0
    train, kept = table.take(~aside), table.take(aside)
    if len(kept) == 0 or len(train) == 0:
        train = kept = table
    x = torch.from_numpy(train.x).to(device)
    y = torch.from_numpy(train.labels).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, waited = float("inf"), None, 0
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator).to(device)
        for start in range(0, len(order), batch):
            picks = order[start:start + batch]
            loss = -(y[picks] * torch.log_softmax(model(x[picks]), dim=1)).sum(dim=1).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        read = cross_entropy(model, kept, device)
        if read < best - 1e-6:
            best, waited = read, 0
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    model.fitted = {"games": int(len(np.unique(table.games))), "rows": int(len(table)),
                    "validation_cross_entropy": round(best, 5)}
    return model


def deciles(values: np.ndarray) -> np.ndarray:
    """Which tenth of `values`, by rank, each one falls in."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return (ranks * 10) // max(len(values), 1)


def _cell(predicted, realised, games) -> dict:
    """How far the mean answer is from the mean outcome in one cell, with
    the error of that gap counted over games, since a game's rows share its
    outcome."""
    n = len(predicted)
    residual = realised - predicted
    miss = float(-residual.mean())
    _, inverse = np.unique(games, return_inverse=True)
    sums = np.bincount(inverse, weights=residual - residual.mean())
    clusters = len(sums)
    error = float(np.sqrt((sums ** 2).sum() * clusters / max(clusters - 1, 1)) / n) if n else 0.0
    return {"rows": int(n), "games": int(clusters), "predicted": round(float(predicted.mean()), 4),
            "realised": round(float(realised.mean()), 4), "miss": round(miss, 4),
            "error": round(error, 4)}


def calibration(
    predicted: np.ndarray,
    realised: np.ndarray,
    stages: np.ndarray,
    games: np.ndarray,
    threshold: float = MISS,
    sigmas: float = 2.0,
) -> dict:
    """How well U's answers match the bonuses players really ended with, on
    games it was not fitted on: by stage, with the share of the bonus's
    spread it explains beside the boundary judge's; by tenth of its answers;
    and by tenth within each stage, where the kill rule is read.

    A cell misses when its mean answer is further than `threshold` from its
    mean outcome by more than `sigmas` of that gap's error: on a few
    thousand games the error of a tenth at East 1 is about as large as the
    threshold, and every cell would read as a miss from noise alone. The
    literal count, gaps over `threshold` whatever their error, is reported
    beside it."""
    predicted = np.asarray(predicted, dtype=np.float64)
    realised = np.asarray(realised, dtype=np.float64)
    stages = np.asarray(stages)
    games = np.asarray(games)
    by_stage, cells = [], []
    for stage in range(STAGES):
        at = stages == stage
        if not at.any():
            continue
        cell = _cell(predicted[at], realised[at], games[at])
        spread = float(((realised[at] - realised[at].mean()) ** 2).sum())
        cell["explained"] = round(1.0 - float(((realised[at] - predicted[at]) ** 2).sum()) / spread, 4) if spread else None
        cell["judge"] = JUDGE[stage]
        cell["stage"] = STAGE_NAMES[stage]
        by_stage.append(cell)
        tenth = deciles(predicted[at])
        for decile in range(10):
            inside = tenth == decile
            if inside.sum() < 2:
                continue
            row = _cell(predicted[at][inside], realised[at][inside], games[at][inside])
            row.update(stage=STAGE_NAMES[stage], decile=decile)
            cells.append(row)
    overall = deciles(predicted)
    by_decile = []
    for decile in range(10):
        inside = overall == decile
        if inside.sum() >= 2:
            by_decile.append({"decile": decile, **_cell(predicted[inside], realised[inside], games[inside])})
    for row in cells:
        row["beyond_noise"] = round(abs(row["miss"]) - sigmas * row["error"] - threshold, 4)
    failed = [row for row in cells if row["beyond_noise"] > 0]
    literal = [row for row in cells if abs(row["miss"]) > threshold]
    worst = max(cells, key=lambda row: row["beyond_noise"]) if cells else None
    return {
        "rows": int(len(predicted)),
        "games": int(len(np.unique(games))),
        "threshold": threshold,
        "sigmas": sigmas,
        "passed": not failed,
        "cells": len(cells),
        "cells_missing_beyond_noise": len(failed),
        "cells_missing_literally": len(literal),
        "largest_miss": round(max((abs(row["miss"]) for row in cells), default=0.0), 4),
        "worst": worst,
        "by_stage": by_stage,
        "by_decile": by_decile,
        "by_stage_decile": cells,
    }


def cross_fitted(table: Boundaries, members: Sequence[str], folds: int = 8, seed: int = 0,
                 device: str = "cpu", **options) -> tuple[np.ndarray, np.ndarray]:
    """U's answer for every row of `table` from a model fitted without that
    row's game, `folds` models in all, the four views of each scoreboard
    centred (see `centred`), and each row's fold. Every game is held out
    once, so the whole table can be read for calibration."""
    fold = game_folds(table.games, folds, seed)
    answers = np.empty(len(table), dtype=np.float64)
    for which in range(folds):
        held = fold == which
        if not held.any():
            continue
        model = fit(table.take(~held), members, seed=seed + which, device=device, **options)
        answers[held] = model.expected(table.x[held])
    return centred(answers), fold


class PlacementRun:
    """The placement reward as a trainer carries it from one generation to
    the next (see `train_combined --reward placement`): the reward each
    round is settled by, how many generations that teach the critic alone
    the run still owes, and U fitted again every so often on the run's own
    last rounds.

    A run that resumes takes all of it from its checkpoint (`fields`): its
    alpha, the generation the reward began at, the warm-up generations it
    has had, and the U it last settled by. Only a run starting the reward
    takes U from a file, so a relaunch given the same command line cannot
    put back the U a refit replaced."""

    def __init__(self, model: PlaceModel, alpha: float, since: int, warm_up: int, warmed: int = 0,
                 refit_every: int = 0, refit_rounds: int = 2, refits: Sequence[dict] = ()) -> None:
        from collections import deque

        self.reward = PlacementReward(model, alpha)
        self.since = int(since)
        self.warm_up = int(warm_up)
        self.warmed = int(warmed)
        self.refit_every = int(refit_every)
        self.recent: deque[Boundaries] = deque(maxlen=max(1, int(refit_rounds)))
        self.refits = list(refits)

    @classmethod
    def begin(cls, args, carried: dict, start: int, members: Sequence[str]) -> PlacementRun:
        """The run's placement reward at generation `start`: from what the
        checkpoint resumed carried (its `reward` and `place_model`), or, for
        a run starting the reward, from `--place-model`. `members` are the
        others the run seats, every one of whom U must know."""
        record = carried.get("reward")
        if record is not None and record.get("kind") == "placement":
            if float(record["alpha"]) != float(args.alpha):
                raise SystemExit(
                    f"this run is paid the placement reward at alpha {record['alpha']} and was asked "
                    f"for {args.alpha}; start a new run to change it")
            if carried.get("place_model") is None:
                raise SystemExit("the checkpoint says it trains on the placement reward but carries no U")
            model = PlaceModel.from_state(carried["place_model"])
            if args.place_model is not None:
                print(f"the run settles by the U its checkpoint carries; {args.place_model} is not read",
                      flush=True)
            run = cls(model, args.alpha, record["since"], args.value_warm_up, record.get("warmed", 0),
                      args.place_refit_every, args.place_refit_rounds, record.get("refits", ()))
        else:
            if args.place_model is None:
                raise SystemExit("--reward placement starts with a U to settle by: give --place-model "
                                 "(python -m neural.place_model fit, or the pre-test's)")
            model = PlaceModel.load(args.place_model)
            run = cls(model, args.alpha, start, args.value_warm_up, 0,
                      args.place_refit_every, args.place_refit_rounds)
            print(f"the placement reward begins at generation {start}, alpha {args.alpha}, "
                  f"U from {args.place_model}", flush=True)
        strangers = [name for name in members if name not in model.members]
        if strangers:
            raise SystemExit(f"U was fitted without {strangers}, whom this run seats; refit it on "
                             "rounds that seated them")
        return run

    @property
    def warming(self) -> bool:
        """Whether this generation teaches the critic alone."""
        return self.warmed < self.warm_up

    def remember(self, ledger: Ledger, generation: int) -> None:
        """Keeps a round's hand starts for the next refit; only the last
        `refit_rounds` are held."""
        if self.refit_every:
            self.recent.append(Boundaries.of(ledger, self.reward.model.members, tag=generation))

    def refit_due(self, generation: int) -> bool:
        """Whether U is fitted again once `generation` is learned: every
        `refit_every` generations of the reward."""
        return bool(self.refit_every) and (generation + 1 - self.since) % self.refit_every == 0

    def refit(self, generation: int, device: str = "cpu", seed: int = 0) -> dict:
        """Fits U again on the rounds remembered, an eighth of their games
        held out, and keeps the new one only where it reads those games at
        least as well as the one it would replace. Answers what the log
        says of it."""
        table = Boundaries.cat(list(self.recent))
        held = game_folds(table.games, 8, seed + generation) == 0
        model = fit(table.take(~held), self.reward.model.members, hidden=self.reward.model.hidden,
                    seed=seed + generation, device=device)
        kept_out = table.take(held)
        old = cross_entropy(self.reward.model.cpu(), kept_out)
        new = cross_entropy(model, kept_out)
        report = calibration(centred(model.expected(kept_out.x)), kept_out.bonus, kept_out.stages,
                             kept_out.games)
        adopted = new <= old
        summary = {
            "generation": generation + 1,
            "rounds": len(self.recent),
            "games": int(len(np.unique(table.games))),
            "held_out_games": report["games"],
            "old_cross_entropy": round(old, 5),
            "new_cross_entropy": round(new, 5),
            "adopted": adopted,
            "calibration_passed": report["passed"],
            "largest_miss": report["largest_miss"],
            "cells_missing_beyond_noise": report["cells_missing_beyond_noise"],
            "explained": {row["stage"]: row["explained"] for row in report["by_stage"]},
        }
        if adopted:
            model.fitted = {**model.fitted, "refit": summary}
            self.reward = PlacementReward(model, self.reward.alpha)
        self.refits.append(summary)
        return summary

    def fields(self) -> dict:
        """What a checkpoint carries of the run's placement reward."""
        return {
            "reward": {"kind": "placement", "alpha": float(self.reward.alpha), "since": self.since,
                       "warmed": self.warmed, "refits": json.loads(json.dumps(self.refits[-8:]))},
            "place_model": self.reward.model.state(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit", help="fit U on saved round ledgers and report its calibration")
    fitting.add_argument("ledgers", type=Path, nargs="+")
    fitting.add_argument("--out", type=Path, required=True)
    fitting.add_argument("--hidden", type=int, default=64)
    fitting.add_argument("--folds", type=int, default=8,
                         help="games held out a fold at a time for the calibration; the saved model sees them all")
    fitting.add_argument("--seed", type=int, default=0)
    checking = commands.add_parser("check", help="read a saved U's calibration on other rounds")
    checking.add_argument("model", type=Path)
    checking.add_argument("ledgers", type=Path, nargs="+")
    args = parser.parse_args()
    ledgers = [Ledger.load(path) for path in args.ledgers]
    if args.command == "fit":
        members = sorted({name for ledger in ledgers for name in ledger.members})
        table = Boundaries.cat([Boundaries.of(ledger, members, tag) for tag, ledger in enumerate(ledgers)])
        answers, _fold = cross_fitted(table, members, args.folds, args.seed, hidden=args.hidden)
        report = calibration(answers, table.bonus, table.stages, table.games)
        model = fit(table, members, hidden=args.hidden, seed=args.seed)
        model.fitted["calibration"] = {key: report[key] for key in (
            "passed", "rows", "games", "cells_missing_beyond_noise", "largest_miss")}
        model.save(args.out)
        report["saved"] = str(args.out)
    else:
        model = PlaceModel.load(args.model)
        table = Boundaries.cat([Boundaries.of(ledger, model.members, tag) for tag, ledger in enumerate(ledgers)])
        report = calibration(centred(model.expected(table.x)), table.bonus, table.stages, table.games)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
